# Incident log (D8)

D8 requires **at least one genuine entry recording a real failure during development**. This file is a template: write
your own entries from failures you actually hit, with the evidence you actually have. Do not write an entry for something
that did not happen, and do not copy someone else's. If a failure happened during an AI-assisted session, say so.

Existing entries from earlier in the project are in `docs/ARCHITECTURE.md` (Person A); move them here if you want one log.

## How to capture evidence while it is happening

```bash
date -u +%FT%TZ                                             # timestamp the moment you notice
docker compose logs --since 15m > incident-<n>-logs.txt     # logs around the failure (keep the file; cite it)
docker compose ps > incident-<n>-ps.txt
python scripts/report.py --window 15m > incident-<n>-report.txt    # metrics at the time
```
The run id (`run_...`) and correlation id from the failing request make the run reconstructible: `GET /runs/{id}` returns its
steps, prompt versions, per-step latency and cost.

## Entry template (copy one per incident)

### INC-YYYY-MM-DD-NN: <one line: what broke>
- **Detected:** <UTC time> by <who / what: a test, a dashboard, a demo rehearsal, a teammate>
- **Severity / impact:** <who or what was affected, for how long>
- **Symptom:** <exactly what was seen: error text, status, metric>
- **Evidence:** <log file, run id, screenshot, commit>
- **Root cause:** <the actual cause, confirmed how>
- **Resolution:** <what was done; commit hash of the fix>
- **Time to detect / time to resolve:** <...>
- **Prevention / follow-up:** <test added, alert, runbook change; link to `docs/RUNBOOK.md` if it changed>

## Entries

<!-- add yours below -->
