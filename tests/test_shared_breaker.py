"""The circuit must be shared across workers/jobs (RQ forks a child per job, discarding in-memory state)."""
import time

from app.agents import CircuitBreaker, SharedCircuitBreaker, make_breaker


class FakeRedis:
    """Just enough of redis-py (decode_responses=True) for the breaker; one instance = one shared server."""
    def __init__(self):
        self.h, self.down = {}, False

    def hgetall(self, k):
        if self.down:
            raise ConnectionError("down")
        return dict(self.h.get(k, {}))

    def hset(self, k, mapping):
        if self.down:
            raise ConnectionError("down")
        self.h[k] = {a: str(b) for a, b in mapping.items()}

    def expire(self, k, s):
        pass


def pair(threshold=3, reset=30):
    r = FakeRedis()
    return SharedCircuitBreaker(threshold, reset, r), SharedCircuitBreaker(threshold, reset, r), r


def test_failures_in_one_worker_open_the_circuit_for_another():
    a, b, _ = pair()
    for _ in range(3):
        a.record_failure()
    assert a.state == "open"
    assert b.allow() is False          # a different worker/job now fails fast instead of retrying a dead provider


def test_in_memory_breaker_does_not_share(  ):
    """The behaviour this class exists to fix."""
    a, b = CircuitBreaker(3, 30), CircuitBreaker(3, 30)
    for _ in range(3):
        a.record_failure()
    assert b.allow() is True


def test_half_open_probe_then_close_is_shared():
    a, b, _ = pair(reset=0.01)
    for _ in range(3):
        a.record_failure()
    time.sleep(0.02)
    assert b.allow() is True and b.state == "half_open"
    b.record_success()
    assert a.allow() is True and a.state == "closed"


def test_half_open_failure_reopens_for_everyone():
    a, b, _ = pair(reset=0.01)
    for _ in range(3):
        a.record_failure()
    time.sleep(0.02)
    assert a.allow() is True
    a.record_failure()
    assert b.allow() is False or b.state == "open"


def test_redis_outage_degrades_to_local_behaviour_not_an_error():
    a, _, r = pair()
    r.down = True
    for _ in range(3):
        a.record_failure()              # must not raise
    assert a.state == "open" and a.allow() is False


def test_make_breaker_selection(monkeypatch):
    cfg = {"CB_FAILURE_THRESHOLD": 3, "CB_RESET_S": 30, "CB_BACKEND": "memory"}
    assert type(make_breaker(cfg)) is CircuitBreaker
    monkeypatch.setenv("STORE_BACKEND", "redis")
    assert isinstance(make_breaker({**cfg, "CB_BACKEND": "auto"}), SharedCircuitBreaker)
    monkeypatch.setenv("STORE_BACKEND", "memory")
    assert type(make_breaker({**cfg, "CB_BACKEND": "auto"})) is CircuitBreaker
