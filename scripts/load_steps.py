"""Step-load driver for the SLO test (D6, A7). Protocol is fixed in docs/SLO.md section 5.

    python scripts/load_steps.py --workers 3 --users 1 3 6 7 9 12 18 --label w3 --redis-url redis://localhost:6379/0

For each concurrency step it runs scripts/locustfile.py headless, waits until >= --min-runs runs have completed
(after excluding the first ceil(N/W) as warm-up, as declared), tells the users to stop submitting, WAITS for every
in-flight run to finish (stopping a run mid-flight would hide the slowest ones), then computes the SLI.
While it runs it samples CPU per component and the RQ queue depth: those two series are the bottleneck evidence.

Writes docs/loadtest_results/<label>_u<N>.json per step and docs/loadtest_results/<label>_summary.md.

Two ways to observe the system under load (the load itself is identical):
  default   host processes (API and workers started by hand): samples CPU per process with psutil, queue depth from Redis.
  --docker  the compose stack: samples `docker stats` per container and reads the queue depth with
            `docker compose exec redis redis-cli` (the compose file does not publish Redis). This is the mode to use
            for `docker compose ... up --scale worker=N`; pass the same -f files you started the stack with:

      python scripts/load_steps.py --docker --workers 3 --label w3 \
          --compose-file docker-compose.yml --compose-file docker-compose.loadtest.yml
"""

from __future__ import annotations

import argparse
import csv
import re
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
SLO_P95_MS = 30_000
SLO_FAIL_PCT = 1.0
SERVICE_TIME_S = 10.5          # 7 calls x 1.5 s, from docs/SLO.md (design arithmetic, declared before measuring)

COMPONENTS = {   # substring of the command line -> component name
    "app.main:app": "api", "app.rq_worker": "worker", "stub_llm.app": "llm_stub",
    "redis-server": "redis", "locust": "locust",
}


class Sampler(threading.Thread):
    """Every second: summed CPU% per component (children included) and RQ queue depth."""

    def __init__(self, redis_url: str | None):
        super().__init__(daemon=True)
        self.stop_ev, self.rows, self.redis = threading.Event(), [], None
        if redis_url:
            try:
                import redis
                self.redis = redis.Redis.from_url(redis_url)
                self.redis.ping()
            except Exception:
                self.redis = None
        self.procs: dict[int, tuple[psutil.Process, str]] = {}

    def _refresh(self):
        """Match components by command line, then adopt their descendants: RQ forks a work-horse per job and the
        horse (which does all the real work) may carry a different process title than its parent."""
        for p in psutil.process_iter(["pid", "cmdline"]):
            cl = " ".join(p.info["cmdline"] or [])
            comp = next((v for k, v in COMPONENTS.items() if k in cl), None)
            if comp and "load_steps" not in cl:
                for q in [p, *p.children(recursive=True)]:
                    if q.pid not in self.procs:
                        self.procs[q.pid] = (q, comp)
                        try:
                            q.cpu_percent(None)
                        except psutil.Error:
                            pass

    def run(self):
        while not self.stop_ev.is_set():
            time.sleep(1.0)
            self._refresh()
            row = {"t": time.time(), "queue_depth": None}
            for name in set(COMPONENTS.values()):
                row[name] = 0.0
            for pid, (p, comp) in list(self.procs.items()):
                try:
                    row[comp] += p.cpu_percent(None)
                except psutil.Error:
                    self.procs.pop(pid, None)
            if self.redis:
                try:
                    row["queue_depth"] = self.redis.llen("rq:queue:claims")
                except Exception:
                    pass
            self.rows.append(row)


# ----------------------------------------------------------------------------- docker mode
_UNITS = {"b": 1, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3}
_SERVICE_RE = re.compile(r"[-_](api|worker|redis|llm-stub|llm_stub)[-_]\d+$")


def parse_mem(text: str) -> float:
    """'90.5MiB / 3.8GiB' -> MB used."""
    m = re.match(r"\s*([\d.]+)\s*([a-zA-Z]+)", text or "")
    return float(m.group(1)) * _UNITS.get(m.group(2).lower(), 1) / 1e6 if m else 0.0


def parse_docker_stats(text: str) -> dict[str, dict]:
    """Parse `docker stats --no-stream --format '{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}'` into per-service totals.

    Returns {'worker': {'cpu': <sum % over replicas>, 'mem_mb': <sum>, 'containers': n}, ...}. Compose names containers
    <project>-<service>-<index> (v2) or <project>_<service>_<index> (v1). Services we do not track are ignored."""
    out: dict[str, dict] = {}
    for line in (text or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        m = _SERVICE_RE.search(parts[0])
        if not m:
            continue
        svc = m.group(1).replace("-", "_")
        cpu = float(parts[1].strip().rstrip("%") or 0)
        d = out.setdefault(svc, {"cpu": 0.0, "mem_mb": 0.0, "containers": 0})
        d["cpu"] += cpu
        d["mem_mb"] += parse_mem(parts[2])
        d["containers"] += 1
    return out


class DockerSampler(threading.Thread):
    """Same interface as Sampler (stop_ev, rows) but reads containers instead of host processes."""

    def __init__(self, compose_files: list[str]):
        super().__init__(daemon=True)
        self.stop_ev, self.rows = threading.Event(), []
        self.compose = ["docker", "compose"] + [x for f in compose_files for x in ("-f", f)]

    def _run(self, cmd, timeout=20):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT).stdout

    def run(self):
        while not self.stop_ev.is_set():
            row = {"t": time.time(), "queue_depth": None, **{n: 0.0 for n in set(COMPONENTS.values())}, "worker_mem_mb": None,
                   "worker_containers": None}
            try:
                stats = parse_docker_stats(self._run(["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}"]))
                for comp, v in stats.items():
                    if comp in row:
                        row[comp] = v["cpu"]
                if "worker" in stats:
                    row["worker_mem_mb"], row["worker_containers"] = stats["worker"]["mem_mb"], stats["worker"]["containers"]
            except Exception:
                pass
            try:
                out = self._run(self.compose + ["exec", "-T", "redis", "redis-cli", "llen", "rq:queue:claims"], timeout=10).strip()
                row["queue_depth"] = int(out) if out.isdigit() else None
            except Exception:
                pass
            self.rows.append(row)
            self.stop_ev.wait(1.0)          # `docker stats --no-stream` itself takes ~1-2 s, so this is ~one sample / 3 s


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def n_lines(path: Path):
    return sum(1 for _ in path.open()) if path.exists() else 0


def run_step(n_users: int, workers: int, a, outdir: Path) -> dict:
    tag = f"{a.label}_u{n_users}"
    csv_path, stop_file = outdir / f"{tag}.csv", outdir / f"{tag}.stop"
    for p in (csv_path, Path(str(csv_path) + ".starts"), stop_file):
        p.unlink(missing_ok=True)
    env = {**os.environ, "LOAD_RUNS_CSV": str(csv_path), "LOAD_STOP_FILE": str(stop_file),
           "LOAD_POLL_TIMEOUT_S": str(a.poll_timeout), "LOAD_DESCRIPTION_MIX": a.mix}
    warmup = math.ceil(n_users / workers)
    need = warmup + a.min_runs
    sampler = DockerSampler(a.compose_file or ["docker-compose.yml"]) if a.docker else Sampler(a.redis_url)
    sampler.start()
    t_start = time.time()
    proc = subprocess.Popen(
        ["locust", "-f", str(ROOT / "scripts" / "locustfile.py"), "--headless", "--host", a.host,
         "-u", str(n_users), "-r", str(n_users), "--only-summary", "ClaimIntakeUser"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[u={n_users}] started; need {need} finished runs ({warmup} warm-up + {a.min_runs})", flush=True)
    try:
        while len(read_rows(csv_path)) < need:
            if proc.poll() is not None:
                raise RuntimeError("locust exited early")
            if time.time() - t_start > a.max_step_s:
                print(f"[u={n_users}] hit --max-step-s; stopping with {len(read_rows(csv_path))} runs", flush=True)
                break
            time.sleep(1)
        stop_file.write_text("stop")                                  # no new submissions; let in-flight runs finish
        drain_deadline = time.time() + a.poll_timeout + 30
        while time.time() < drain_deadline:
            if len(read_rows(csv_path)) >= n_lines(Path(str(csv_path) + ".starts")):
                break
            time.sleep(1)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        sampler.stop_ev.set()
        sampler.join(timeout=3)
    t_end = time.time()

    rows = sorted(read_rows(csv_path), key=lambda r: float(r["t_end"]))
    counted = rows[warmup:]                                           # declared warm-up exclusion
    lat = [float(r["e2e_ms"]) for r in counted]
    bad = [r for r in counted if r["timed_out"] == "1" or r["status"] in ("failed",) or str(r["status"]).startswith("poll_")]
    span = (float(counted[-1]["t_end"]) - float(counted[0]["t_end"])) if len(counted) > 1 else None
    window = [r for r in sampler.rows if t_start + 5 <= r["t"] <= t_end]
    avg = lambda k: round(sum(r[k] for r in window) / len(window), 1) if window else None
    depths = [r["queue_depth"] for r in window if r["queue_depth"] is not None]
    res = dict(
        label=a.label, users=n_users, workers=workers, runs_total=len(rows), runs_counted=len(counted), warmup_excluded=warmup,
        p50_ms=pct(lat, .5), p95_ms=pct(lat, .95), p99_ms=pct(lat, .99), max_ms=max(lat, default=None),
        mean_ms=round(sum(lat) / len(lat), 1) if lat else None,
        submit_p95_ms=pct([float(r["submit_ms"]) for r in counted], .95),
        failures=len(bad), failure_pct=round(100 * len(bad) / max(len(counted), 1), 2),
        throughput_runs_per_s=round((len(counted) - 1) / span, 4) if span else None,
        predicted_p95_ms=math.ceil(n_users / workers) * SERVICE_TIME_S * 1000,
        slo_met=bool(lat) and pct(lat, .95) <= SLO_P95_MS and 100 * len(bad) / max(len(counted), 1) < SLO_FAIL_PCT,
        cpu_pct_avg={k: avg(k) for k in ("api", "worker", "llm_stub", "redis", "locust")},
        queue_depth_avg=round(sum(depths) / len(depths), 2) if depths else None, queue_depth_max=max(depths) if depths else None,
        worker_mem_mb_avg=(round(sum(r["worker_mem_mb"] for r in window if r.get("worker_mem_mb")) / max(1, sum(1 for r in window if r.get("worker_mem_mb"))), 1)
                           if any(r.get("worker_mem_mb") for r in window) else None),
        worker_containers_seen=max((r.get("worker_containers") or 0 for r in window), default=0) or None,
        duration_s=round(t_end - t_start, 1),
    )
    (outdir / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[u={n_users}] n={len(counted)} p50={res['p50_ms'] and round(res['p50_ms'])}ms p95={res['p95_ms'] and round(res['p95_ms'])}ms "
          f"tput={res['throughput_runs_per_s']}/s fail={res['failure_pct']}% slo_met={res['slo_met']} q_avg={res['queue_depth_avg']}", flush=True)
    return res


def summary_md(label: str, results: list[dict]) -> str:
    L = [f"### {label} ({results[0]['workers']} worker(s))", "",
         "| In flight (users) | Runs counted | p50 s | p95 s | predicted p95 s | Throughput runs/s | Failures % | SLO met | worker CPU% | api CPU% | stub CPU% | redis CPU% | queue depth avg/max |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    f = lambda v: "-" if v is None else f"{v / 1000:.1f}"
    for r in results:
        c = r["cpu_pct_avg"]
        L.append(f"| {r['users']} | {r['runs_counted']} | {f(r['p50_ms'])} | {f(r['p95_ms'])} | {r['predicted_p95_ms'] / 1000:.1f} | "
                 f"{r['throughput_runs_per_s']} | {r['failure_pct']} | {'yes' if r['slo_met'] else '**NO**'} | {c['worker']} | {c['api']} | "
                 f"{c['llm_stub']} | {c['redis']} | {r['queue_depth_avg']}/{r['queue_depth_max']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, required=True, help="number of workers currently running (for warm-up and prediction)")
    ap.add_argument("--users", type=int, nargs="+", default=[1, 3, 6, 7, 9, 12, 18])
    ap.add_argument("--label", required=True)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--redis-url", help="host-process mode only: where to read the RQ queue depth")
    ap.add_argument("--docker", action="store_true", help="observe the compose stack (docker stats + compose exec redis)")
    ap.add_argument("--compose-file", action="append", help="repeat for each -f you started the stack with")
    ap.add_argument("--min-runs", type=int, default=30)
    ap.add_argument("--poll-timeout", type=float, default=120)
    ap.add_argument("--max-step-s", type=float, default=900)
    ap.add_argument("--mix", default="plain")
    ap.add_argument("--outdir", default=str(ROOT / "docs" / "loadtest_results"))
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    import urllib.request   # preflight: a dead target must fail loudly, not hang until a timeout kills us silently
    try:
        urllib.request.urlopen(a.host.rstrip("/") + "/health", timeout=5).read()
    except Exception as e:
        sys.exit(f"preflight failed: {a.host}/health is not answering ({e}). Start the stack first.")
    results = [run_step(n, a.workers, a, outdir) for n in a.users]
    seen = {r.get("worker_containers_seen") for r in results} - {None}
    if seen and seen != {a.workers}:
        print(f"WARNING: --workers {a.workers} but {sorted(seen)} worker container(s) were running; the warm-up exclusion and "
              f"the predicted p95 column assume --workers.", flush=True)
    md = summary_md(a.label, results)
    (outdir / f"{a.label}_summary.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
