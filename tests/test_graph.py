"""The LangGraph engine must behave like the hand-rolled orchestrator, plus resume."""

from datetime import date

import pytest

from rootlogic.control import Command
from rootlogic.fake_llm import FakeLLM, default_finding
from rootlogic.graph import ResearchGraph
from rootlogic.llm import LLMError
from rootlogic.models import FindingDraft, PlanDraft, Reflection, SubTaskDraft
from rootlogic.orchestrator import Budget
from rootlogic.store import Store

from .test_orchestrator import ScriptedUI

TODAY = date(2026, 9, 22)
TOPIC = "impact of generative AI on newsrooms"


def make(tmp_path, ui=None, handlers=None, budget=None, store=None, checkpoint=None):
    store = store or Store(":memory:")
    holder = {}
    sink = lambda u: store.record_call(  # noqa: E731
        session_id=holder["g"].sid or None, purpose=u.purpose, model=u.model,
        input_tokens=u.input_tokens, output_tokens=u.output_tokens, cache_read_tokens=0,
        cache_write_tokens=0, web_searches=u.web_searches, cost_usd=u.cost_usd,
        stop_reason=u.stop_reason, request_id=None)
    llm = FakeLLM(sink, handlers=handlers)
    ui = ui or ScriptedUI()
    g = ResearchGraph(llm, store, ui, checkpoint_path=checkpoint or tmp_path / "cp.db",
                      budget=budget or Budget(), reports_dir=tmp_path, today=TODAY)
    holder["g"] = g
    return g, llm, store, ui


def purposes(llm):
    return [p for p, _ in llm.calls]


def test_full_run_matches_orchestrator_stages(tmp_path):
    g, llm, store, ui = make(tmp_path)
    report = g.run(TOPIC)

    assert report is not None and "## Sources" in report.to_markdown()
    p = purposes(llm)
    assert p[:2] == ["clarify", "plan"]
    assert sorted(x for x in p if x.startswith("research:")) == \
        ["research:t1", "research:t2", "research:t3"]
    assert p[-3:] == ["reflect", "analyze", "report"]
    assert store.session(g.sid)["status"] == "done"
    dropped = [s for s in store.sources(g.sid) if not s["kept"]]
    assert dropped and all("outdated" in s["reason"] for s in dropped)


def test_clarifying_questions_are_interrupts(tmp_path):
    ui = ScriptedUI(answers=["last 12 months", ""])
    g, llm, store, _ = make(tmp_path, ui=ui)
    g.run("AI")

    assert len(ui.questions) == 2
    assert "A: last 12 months" in next(pr for pu, pr in llm.calls if pu == "plan")
    # clarify ran once: the LLM call lives in its own node, not the interrupted one
    assert purposes(llm).count("clarify") == 1


def test_plan_edit_and_reject(tmp_path):
    def drop_t2(plan):
        plan.subtasks = [t for t in plan.subtasks if t.id != "t2"]

    g, llm, _, _ = make(tmp_path, ui=ScriptedUI(edit=drop_t2))
    g.run(TOPIC)
    assert "research:t2" not in purposes(llm)

    g2, llm2, store2, _ = make(tmp_path / "b", ui=ScriptedUI(approve=False))
    assert g2.run(TOPIC) is None
    assert store2.session(g2.sid)["status"] == "aborted"
    assert not any(x.startswith("research:") for x in purposes(llm2))


def test_reflection_loop_and_budget(tmp_path):
    def always_more(prompt):
        return Reflection(sufficient=False, reasoning="more",
                          new_subtasks=[SubTaskDraft(question=f"Follow-up {len(prompt)}",
                                                     rationale="r", search_queries=["q"],
                                                     depends_on=[])],
                          questions_for_user=[])

    g, llm, _, ui = make(tmp_path, handlers={Reflection: always_more}, budget=Budget(max_rounds=2))
    g.run(TOPIC)
    assert purposes(llm).count("reflect") == 2
    assert "research:t5" in purposes(llm)
    assert "loop.budget" in ui.types()


def test_dependencies_receive_earlier_findings(tmp_path):
    def dep_plan(prompt):
        return PlanDraft(objective="o", recency_days=0, subtasks=[
            SubTaskDraft(question="Base question", rationale="r", search_queries=["a"],
                         depends_on=[]),
            SubTaskDraft(question="Follow-on question", rationale="r", search_queries=["b"],
                         depends_on=[1]),
        ])

    g, llm, _, _ = make(tmp_path, handlers={PlanDraft: dep_plan})
    g.run(TOPIC)
    research = [(p, pr) for p, pr in llm.calls if p.startswith("research:")]
    assert [p for p, _ in research] == ["research:t1", "research:t2"]
    assert "Earlier finding (Base question)" in research[1][1]


def test_override_via_interrupt(tmp_path):
    ui = ScriptedUI(overrides=[[Command("skip", "t3"), Command("add", "Who funds this?"),
                                Command("note", "focus on Europe")]])
    g, llm, _, _ = make(tmp_path, ui=ui)
    g.control.request_pause()
    g.run(TOPIC)

    p = purposes(llm)
    assert "research:t3" not in p and "research:t4" in p
    assert "User guidance: focus on Europe" in next(pr for pu, pr in llm.calls
                                                    if pu == "research:t1")


def test_override_stop_and_abort(tmp_path):
    g, llm, store, _ = make(tmp_path, ui=ScriptedUI(overrides=[[Command("stop")]]))
    g.control.request_pause()
    assert g.run(TOPIC) is not None
    assert not any(x.startswith("research:") for x in purposes(llm))

    g2, llm2, store2, _ = make(tmp_path / "b", ui=ScriptedUI(overrides=[[Command("abort")]]))
    g2.control.request_pause()
    assert g2.run(TOPIC) is None
    assert store2.session(g2.sid)["status"] == "aborted"
    assert "analyze" not in purposes(llm2)


def test_failed_subagent_is_isolated(tmp_path):
    def flaky(prompt):
        if "current state" in prompt:
            raise LLMError("boom")
        return default_finding(prompt, 9)

    g, _, _, ui = make(tmp_path, handlers={FindingDraft: flaky})
    assert g.run(TOPIC) is not None
    assert "task.failed" in ui.types()


def test_resume_after_crash_skips_completed_work(tmp_path):
    """The LangGraph payoff: a run that dies at 'analyze' resumes without redoing research."""
    store = Store(tmp_path / "rl.db")
    cp = tmp_path / "cp.db"
    state = {"fail": True}

    def flaky_analysis(prompt):
        if state["fail"]:
            raise LLMError("network down")
        from rootlogic.fake_llm import default_analysis
        return default_analysis(prompt)

    from rootlogic.models import Analysis
    g, llm, _, _ = make(tmp_path, handlers={Analysis: flaky_analysis}, store=store, checkpoint=cp)
    with pytest.raises(LLMError):
        g.run(TOPIC)
    sid = g.sid
    assert store.session(sid)["status"] == "failed"

    # New process: fresh engine, same checkpoint file.
    state["fail"] = False
    g2, llm2, _, _ = make(tmp_path, handlers={Analysis: flaky_analysis}, store=store,
                          checkpoint=cp)
    report = g2.resume(sid)

    assert report is not None
    assert purposes(llm2) == ["analyze", "report"]   # no re-planning, no re-research
    assert store.session(sid)["status"] == "done"


def test_resume_while_waiting_for_plan_approval(tmp_path):
    """A session can sit at an interrupt (e.g. overnight) and be answered by a new process."""
    store = Store(tmp_path / "rl.db")
    cp = tmp_path / "cp.db"

    class QuitAtPlan(ScriptedUI):
        def review_plan(self, plan):
            raise KeyboardInterrupt  # user closed the terminal at the approval prompt

    g, _, _, _ = make(tmp_path, ui=QuitAtPlan(), store=store, checkpoint=cp)
    with pytest.raises(KeyboardInterrupt):
        g.run(TOPIC)

    g2, llm2, _, _ = make(tmp_path, store=store, checkpoint=cp)
    assert g2.resume(g.sid) is not None
    assert purposes(llm2)[0].startswith("research:")


def test_mermaid_diagram_renders(tmp_path):
    g, _, _, _ = make(tmp_path)
    diagram = g.mermaid()
    for node in ("clarify", "ask_user", "review", "dispatch", "research", "collect", "reflect"):
        assert node in diagram
