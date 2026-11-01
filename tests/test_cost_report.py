"""L5: the cost report's arithmetic, on synthetic records with hand-computed answers."""
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("cost_report", pathlib.Path(__file__).parent.parent / "scripts" / "cost_report.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def llm(agent, action, pt, ct, cost):
    return dict(agent=agent, action=action, prompt_version=f"{agent}-v1", prompt_tokens=pt, completion_tokens=ct, cost_usd=cost)


RUN_A = dict(run_id="a", steps=[llm("triage", "classify", 500, 100, 0.001), llm("coverage", "decide:search_corpus", 3000, 100, 0.004),
                                {"agent": "coverage", "action": "tool_call", "tool": "search_corpus"}, llm("verifier", "verify", 400, 20, 0.0005)])
RUN_B = dict(run_id="b", steps=[{"agent": "guardrail", "action": "check_input"}])        # stopped before any model call


def test_cost_per_100_runs_and_means():
    a = cr.analyse([RUN_A, RUN_B], None, None)
    assert a["runs"] == 2 and a["runs_with_no_model_call"] == 1
    assert abs(a["total_cost_usd"] - 0.0055) < 1e-12
    assert abs(a["mean_cost_usd"] - 0.00275) < 1e-12
    assert abs(a["cost_per_100_runs_usd"] - 0.275) < 1e-9          # guardrail-stopped runs dilute the mean, as they should


def test_largest_driver_identified():
    a = cr.analyse([RUN_A, RUN_B], None, None)
    assert next(iter(a["by_agent"])) == "coverage"
    assert next(iter(a["by_step"])) == "coverage:decide:search_corpus"
    assert "Largest cost driver by agent: `coverage`" in cr.render(a)


def test_repricing_uses_recorded_tokens():
    a = cr.analyse([RUN_A], price_in=0.001, price_out=0.002)       # USD / 1k tokens
    expect = (3900 * 0.001 + 220 * 0.002) / 1000
    assert abs(a["total_cost_usd"] - expect) < 1e-12 and a["repriced"]


def test_tool_steps_carry_no_cost_and_no_calls():
    a = cr.analyse([RUN_A], None, None)
    assert a["by_agent"]["coverage"]["calls"] == 1                  # the tool_call step is not an LLM call


def test_zero_prices_reported_not_hidden():
    zero = dict(steps=[llm("triage", "classify", 10, 10, 0.0)])
    assert "prices unset" in cr.render(cr.analyse([zero], None, None))
