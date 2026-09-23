"""Evaluation harness: case loading, scoring rules, refusal handling, baseline comparison."""

from datetime import date

from rootlogic.evaluate import (CaseResult, EvalCase, EvalRun, compare, load_cases, run_case,
                                run_eval, summarize)
from rootlogic.fake_llm import FakeLLM, default_report
from rootlogic.llm import AgentRefusal
from rootlogic.models import HoaxJudgement, PlanDraft, ReportDraft
from rootlogic.orchestrator import Orchestrator

TODAY = date(2026, 9, 22)


def factory(handlers=None, tmp=None):
    def make(store, ui):
        return Orchestrator(FakeLLM(handlers=handlers), store, ui, reports_dir=tmp, today=TODAY)
    return make


def test_shipped_cases_are_valid_and_cover_every_kind():
    cases = load_cases("evals/cases.json")
    assert len(cases) >= 20 and len({c.id for c in cases}) == len(cases)
    assert {c.kind for c in cases} == {"fact", "hoax", "contested", "recency", "harmful"}
    assert all(c.expect_refusal for c in cases if c.kind == "harmful")
    assert all(c.must_not_assert for c in cases if c.kind == "hoax")
    assert [c.id for c in load_cases("evals/cases.json", kinds=["harmful"])] == \
        ["harm-nerve-agent", "harm-ghost-gun"]


def test_fact_case_passes_when_keywords_present(tmp_path):
    def report(prompt):
        r = default_report(prompt)
        r.body_markdown += "\n\nIt launched in December 2021 and orbits L2 [1]. Height 8,848.86 m [1]."
        return r

    case = EvalCase(id="f", kind="fact", topic="JWST launch", must_mention=["2021", "l2", "8848.86"])
    r = run_case(case, factory({ReportDraft: report}, tmp_path), today=TODAY)
    assert r.passed, r.reasons
    assert r.metrics["supported_ratio"] == 1.0 and r.metrics["missing_keywords"] == []


def test_fact_case_fails_and_says_why(tmp_path):
    case = EvalCase(id="f", kind="fact", topic="JWST launch", must_mention=["2021"])
    r = run_case(case, factory(tmp=tmp_path), today=TODAY)
    assert not r.passed and r.reasons == ["missing: 2021"]


def test_hoax_case_uses_the_judge(tmp_path):
    case = EvalCase(id="h", kind="hoax", topic="Nibiru", must_not_assert=["Nibiru is coming"])
    judged = {"prompt": ""}

    def judge_yes(prompt):
        judged["prompt"] = prompt
        return HoaxJudgement(asserted=True, reasoning="it endorses it")

    r = run_case(case, factory({HoaxJudgement: judge_yes}, tmp_path), today=TODAY)
    assert not r.passed and r.metrics["hoax_asserted"] == ["Nibiru is coming"]
    assert judged["prompt"].startswith("Statement: Nibiru is coming") and "Report:\n# " in \
        judged["prompt"]
    assert run_case(case, factory(tmp=tmp_path), today=TODAY).passed  # default judge: not asserted


def test_harmful_case_passes_only_on_refusal(tmp_path):
    def refuse(prompt):
        raise AgentRefusal("plan: model declined (cbrn)")

    case = EvalCase(id="x", kind="harmful", topic="bad", expect_refusal=True)
    refused = run_case(case, factory({PlanDraft: refuse}, tmp_path), today=TODAY)
    assert refused.passed and refused.status == "refused"
    answered = run_case(case, factory(tmp=tmp_path), today=TODAY)
    assert not answered.passed and "expected a refusal" in answered.reasons[0]
    assert answered.status == "done"


def test_contested_and_recency_rules(tmp_path):
    case = EvalCase(id="c", kind="contested", topic="t", min_contradictions=2)
    r = run_case(case, factory(tmp=tmp_path), today=TODAY)
    assert r.reasons == ["found 1 contradiction(s), expected >= 2"]

    stale = EvalCase(id="r", kind="recency", topic="t", max_stale_share=0.0)
    r = run_case(stale, factory(tmp=tmp_path), today=date(2029, 1, 1))  # sources now 2+ years old
    assert any("over two years old" in x for x in r.reasons)


def test_summary_and_baseline_comparison(tmp_path):
    cases = [EvalCase(id="a", kind="fact", topic="t", must_mention=["2021"]),
             EvalCase(id="b", kind="contested", topic="t", min_contradictions=1),
             EvalCase(id="c", kind="harmful", topic="t", expect_refusal=True)]
    run = run_eval(cases, factory(tmp=tmp_path), engine_name="loop", model="offline-fake")
    s = run.summary
    assert (s["cases"], s["passed"]) == (3, 1)
    assert s["by_kind"] == {"contested": "1/1", "fact": "0/1", "harmful": "0/1"}
    assert s["refusal_rate_on_harmful"] == 0.0 and s["mean_supported_ratio"] == 1.0

    better = run.model_copy(deep=True)
    better.results[0].passed = True
    better.summary = summarize(better.results)
    diff = compare(better, EvalRun.model_validate_json(run.model_dump_json()))
    assert diff["changed"] == {"a": "fail → pass"} and diff["deltas"]["passed"] == 1


def test_cli_offline_eval_writes_results(tmp_path, capsys):
    from rootlogic.cli import main
    out = tmp_path / "r.json"
    code = main(["--db", str(tmp_path / "x.db"), "eval", "--offline", "--case", "nibiru",
                 "--case", "remote-work", "--out", str(out)])
    assert code == 0
    run = EvalRun.model_validate_json(out.read_text())
    assert [r.id for r in run.results] == ["nibiru", "remote-work"]
    assert isinstance(run.results[0], CaseResult) and "Evaluation summary" in capsys.readouterr().out


def test_a_report_with_no_dated_sources_fails_a_recency_case(tmp_path):
    """Found in review: stale_share returns None when nothing is dated, and `or 0` turned that
    into a pass. A report whose recency cannot be checked has not met a recency requirement."""
    from rootlogic.fake_llm import default_finding
    from rootlogic.models import FindingDraft

    def undated(prompt):
        draft = default_finding(prompt, 1)
        for source in draft.sources:
            source.published = "unknown"
        return draft

    case = EvalCase(id="r-undated", kind="recency", topic="a topic needing fresh sources",
                    max_stale_share=0.5)
    r = run_case(case, factory({FindingDraft: undated}, tmp_path), today=TODAY)

    assert r.metrics["stale_share"] is None
    assert not r.passed and any("date" in reason for reason in r.reasons)
