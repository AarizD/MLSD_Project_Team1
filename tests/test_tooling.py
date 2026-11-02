"""The measurement tooling must be correct, or every number in the report is suspect."""
import importlib.util
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).parent.parent / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ docker stats parsing (load_steps --docker)
DOCKER_STATS = """\
claims-worker-1|0.45%|90.5MiB / 3.8GiB
claims-worker-2|0.30%|91.0MiB / 3.8GiB
claims-worker-3|0.25%|89.5MiB / 3.8GiB
claims-api-1|2.10%|48MiB / 3.8GiB
claims-redis-1|0.20%|6.2MiB / 3.8GiB
claims-llm-stub-1|0.30%|62MiB / 3.8GiB
claims-prometheus-1|1.00%|120MiB / 3.8GiB
claims-grafana-1|0.50%|150MiB / 3.8GiB
"""


def test_parse_docker_stats_sums_replicas_and_ignores_untracked():
    ls = pytest.importorskip("psutil") and load("load_steps")
    s = ls.parse_docker_stats(DOCKER_STATS)
    assert set(s) == {"worker", "api", "redis", "llm_stub"}
    assert s["worker"]["containers"] == 3 and abs(s["worker"]["cpu"] - 1.0) < 1e-9
    assert abs(s["worker"]["mem_mb"] - (90.5 + 91.0 + 89.5) * 1.048576) < 0.5
    assert s["api"]["cpu"] == 2.1


def test_parse_docker_stats_handles_compose_v1_names_and_garbage():
    ls = pytest.importorskip("psutil") and load("load_steps")
    s = ls.parse_docker_stats("proj_worker_1|1.5%|10MiB / 1GiB\nnot a line\n\nproj_api_1|bad|1MiB / 1GiB".replace("bad", "0.0%"))
    assert s["worker"]["containers"] == 1 and s["api"]["cpu"] == 0.0


def test_percentile_interpolates():
    ls = pytest.importorskip("psutil") and load("load_steps")
    assert ls.pct([1, 2, 3, 4], 0.5) == 2.5
    assert ls.pct([10.0], 0.95) == 10.0 and ls.pct([], 0.5) is None


# ------------------------------------------------------------------ report.py: derived numbers from canned PromQL results
def fake_q(expr):
    table = {
        "sum by (status) (increase(claim_runs_total[10m]))": [({"status": "completed"}, 80.0), ({"status": "escalated"}, 15.0), ({"status": "degraded"}, 5.0)],
        "sum(increase(claim_run_steps_sum[10m])) / sum(increase(claim_run_steps_count[10m]))": [({}, 11.0)],
        "sum(increase(llm_calls_total[10m]))": [({}, 600.0)],
        "sum(increase(llm_retries_total[10m]))": [({}, 30.0)],
        "sum by (agent, action) (increase(llm_step_cost_usd_total[10m]))": [
            ({"agent": "coverage", "action": "decide:search_corpus"}, 0.30), ({"agent": "coverage", "action": "decide:finish"}, 0.10),
            ({"agent": "triage", "action": "classify"}, 0.05), ({"agent": "verifier", "action": "verify"}, 0.05)],
        "sum by (check) (increase(guardrail_triggers_total[10m]))": [({"check": "prompt_injection"}, 4.0)],
    }
    return table.get(expr, [])


def test_report_derives_rates_and_cost_drivers():
    rp = load("report")
    d = rp.collect(fake_q, "10m")
    assert d["runs_total"] == 100 and d["task_success_rate"] == 0.8
    assert abs(d["failed_or_degraded_rate"] - 0.05) < 1e-9          # escalated is NOT a failure
    assert d["steps_per_run"] == 11.0 and d["llm_calls_per_run"] == 6.0
    assert abs(d["llm_retry_rate"] - 0.05) < 1e-9
    assert abs(d["cost_total_usd"] - 0.50) < 1e-9 and abs(d["cost_per_100_runs_usd"] - 0.50) < 1e-9
    assert d["largest_cost_driver_agent"] == "coverage" and d["largest_cost_driver_step"] == "coverage:decide:search_corpus"
    out = rp.render(d)
    assert "COST PER 100 RUNS: $0.5000" in out and "80.0%" in out and "prompt_injection=4" in out


def test_report_with_no_traffic_does_not_divide_by_zero():
    rp = load("report")
    d = rp.collect(lambda e: [], "5m")
    assert d["task_success_rate"] is None and d["cost_per_100_runs_usd"] is None
    assert "n/a" in rp.render(d)
