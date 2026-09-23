from datetime import date

import pytest

from rootlogic.control import Command
from rootlogic.fake_llm import FakeLLM, default_plan
from rootlogic.llm import LLMError
from rootlogic.models import (Clarification, FindingDraft, PlanDraft, Reflection, SubTaskDraft)
from rootlogic.orchestrator import Budget, Orchestrator
from rootlogic.store import Store

TODAY = date(2026, 9, 22)


class ScriptedUI:
    def __init__(self, answers=(), overrides=(), approve=True, edit=None):
        self.answers = list(answers)
        self.overrides = list(overrides)
        self.approve = approve
        self.edit = edit
        self.events = []
        self.questions = []

    def on_event(self, e):
        self.events.append(e)

    def ask(self, q):
        self.questions.append(q)
        return self.answers.pop(0) if self.answers else ""

    def review_plan(self, plan):
        if self.edit:
            self.edit(plan)
        return plan if self.approve else None

    def override(self, plan):
        return self.overrides.pop(0) if self.overrides else []

    def types(self):
        return [e.type for e in self.events]


def make(tmp_path, ui=None, handlers=None, budget=None):
    store = Store(":memory:")
    orch_holder = {}
    sink = lambda u: store.record_call(  # noqa: E731
        session_id=orch_holder["o"].sid, purpose=u.purpose, model=u.model,
        input_tokens=u.input_tokens, output_tokens=u.output_tokens, cache_read_tokens=0,
        cache_write_tokens=0, web_searches=u.web_searches, cost_usd=u.cost_usd,
        stop_reason=u.stop_reason, request_id=None)
    llm = FakeLLM(sink, handlers=handlers)
    ui = ui or ScriptedUI()
    orch = Orchestrator(llm, store, ui, budget=budget or Budget(), reports_dir=tmp_path,
                        today=TODAY)
    orch_holder["o"] = orch
    return orch, llm, store, ui


def test_full_run_produces_report_and_persists_everything(tmp_path):
    orch, llm, store, ui = make(tmp_path)
    report = orch.run("impact of generative AI on newsrooms")

    assert report is not None
    purposes = [p for p, _ in llm.calls]
    assert purposes[:2] == ["clarify", "plan"]
    assert sorted(p for p in purposes if p.startswith("research:")) == \
        ["research:t1", "research:t2", "research:t3"]
    stages = [p.split(":")[0] for p in purposes]
    assert stages[-6:] == ["reflect", "verify", "verify", "verify", "analyze", "report"]

    s = store.session(orch.sid)
    assert s["status"] == "done"
    assert (tmp_path / s["report_path"].split("/")[-1]).exists()
    assert "## Sources" in report.to_markdown()
    assert store.usage(orch.sid)["calls"] == len(llm.calls)
    assert "session.done" in ui.types()


def test_outdated_sources_are_dropped_and_logged(tmp_path):
    orch, _, store, ui = make(tmp_path)
    orch.run("impact of generative AI on newsrooms")

    dropped = [s for s in store.sources(orch.sid) if not s["kept"]]
    assert dropped and all("outdated" in s["reason"] for s in dropped)
    kept_urls = [s.url for f in orch.findings.values() for s in f.sources]
    assert not any("archive" in u for u in kept_urls)


def test_vague_topic_triggers_clarifying_questions_that_feed_the_plan(tmp_path):
    ui = ScriptedUI(answers=["last 12 months", ""])
    orch, llm, store, _ = make(tmp_path, ui=ui)
    orch.run("AI")

    assert len(ui.questions) == 2
    plan_prompt = next(p for purpose, p in llm.calls if purpose == "plan")
    assert "A: last 12 months" in plan_prompt
    kinds = [m["kind"] for m in store.messages(orch.sid)]
    assert kinds.count("clarifying_question") == 2 and kinds.count("answer") == 1
    assert "user.skipped" in ui.types()


def test_reflection_adds_follow_up_tasks_until_sufficient(tmp_path):
    calls = {"n": 0}

    def reflect(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return Reflection(sufficient=False, reasoning="Missing regulation angle",
                              new_subtasks=[SubTaskDraft(question="What regulation applies?",
                                                         rationale="gap", search_queries=["x"],
                                                         depends_on=[])],
                              questions_for_user=[])
        return Reflection(sufficient=True, reasoning="ok", new_subtasks=[], questions_for_user=[])

    orch, llm, _, ui = make(tmp_path, handlers={Reflection: reflect})
    orch.run("impact of generative AI on newsrooms")

    assert calls["n"] == 2
    assert any(p == "research:t4" for p, _ in llm.calls)
    added = [e for e in ui.events if e.type == "task.added"]
    assert added and "reflection" in added[0].message


def test_reflection_budget_caps_the_loop(tmp_path):
    def always_more(prompt):
        return Reflection(sufficient=False, reasoning="more",
                          new_subtasks=[SubTaskDraft(question=f"Follow-up {len(prompt)}",
                                                     rationale="r", search_queries=["q"],
                                                     depends_on=[])],
                          questions_for_user=[])

    orch, llm, _, ui = make(tmp_path, handlers={Reflection: always_more},
                            budget=Budget(max_rounds=2))
    orch.run("impact of generative AI on newsrooms")

    assert sum(1 for p, _ in llm.calls if p == "reflect") == 2
    assert "loop.budget" in ui.types()


def test_task_cap_is_enforced(tmp_path):
    def big_plan(prompt):
        d = default_plan(prompt)
        d.subtasks = d.subtasks * 4  # 12 tasks
        return d

    orch, llm, _, _ = make(tmp_path, handlers={PlanDraft: big_plan}, budget=Budget(max_tasks=5))
    orch.run("impact of generative AI on newsrooms")
    assert sum(1 for p, _ in llm.calls if p.startswith("research:")) == 5


def test_dependencies_run_in_order_and_receive_earlier_findings(tmp_path):
    def dep_plan(prompt):
        return PlanDraft(objective="o", recency_days=0, subtasks=[
            SubTaskDraft(question="Base question", rationale="r", search_queries=["a"],
                         depends_on=[]),
            SubTaskDraft(question="Follow-on question", rationale="r", search_queries=["b"],
                         depends_on=[1]),
        ])

    orch, llm, _, _ = make(tmp_path, handlers={PlanDraft: dep_plan})
    orch.run("impact of generative AI on newsrooms")

    research = [(p, prompt) for p, prompt in llm.calls if p.startswith("research:")]
    assert [p for p, _ in research] == ["research:t1", "research:t2"]
    assert "Earlier finding (Base question)" in research[1][1]


def test_user_can_edit_plan_before_execution(tmp_path):
    def drop_t2(plan):
        plan.subtasks = [t for t in plan.subtasks if t.id != "t2"]

    orch, llm, _, _ = make(tmp_path, ui=ScriptedUI(edit=drop_t2))
    orch.run("impact of generative AI on newsrooms")
    assert "research:t2" not in [p for p, _ in llm.calls]


def test_rejecting_plan_aborts_without_research(tmp_path):
    orch, llm, store, _ = make(tmp_path, ui=ScriptedUI(approve=False))
    assert orch.run("impact of generative AI on newsrooms") is None
    assert store.session(orch.sid)["status"] == "aborted"
    assert not any(p.startswith("research:") for p, _ in llm.calls)


def test_override_skip_add_note_then_continue(tmp_path):
    ui = ScriptedUI(overrides=[[Command("skip", "t3"), Command("add", "Who funds this?"),
                                Command("note", "focus on Europe")]])
    orch, llm, store, _ = make(tmp_path, ui=ui)
    orch.control.request_pause()  # as if the user hit Ctrl-C before the first wave
    orch.run("impact of generative AI on newsrooms")

    purposes = [p for p, _ in llm.calls]
    assert "research:t3" not in purposes and "research:t4" in purposes
    research_prompt = next(p for purpose, p in llm.calls if purpose == "research:t1")
    assert "User guidance: focus on Europe" in research_prompt
    assert {"override.skip", "override.note", "task.added"} <= set(ui.types())


def test_override_stop_skips_remaining_and_still_writes_report(tmp_path):
    ui = ScriptedUI(overrides=[[Command("stop")]])
    orch, llm, store, _ = make(tmp_path, ui=ui)
    orch.control.request_pause()
    report = orch.run("impact of generative AI on newsrooms")

    assert report is not None
    assert not any(p.startswith("research:") for p, _ in llm.calls)
    assert store.session(orch.sid)["status"] == "done"


def test_override_abort(tmp_path):
    orch, _, store, _ = make(tmp_path, ui=ScriptedUI(overrides=[[Command("abort")]]))
    orch.control.request_pause()
    assert orch.run("impact of generative AI on newsrooms") is None
    assert store.session(orch.sid)["status"] == "aborted"


def test_failed_subagent_does_not_sink_the_session(tmp_path):
    def flaky(prompt):
        if "current state" in prompt:
            raise LLMError("boom")
        from rootlogic.fake_llm import default_finding
        return default_finding(prompt, 9)

    orch, _, store, ui = make(tmp_path, handlers={FindingDraft: flaky})
    report = orch.run("impact of generative AI on newsrooms")
    assert report is not None
    assert "task.failed" in ui.types()


def test_long_term_memory_recalls_prior_session_and_suggests_topics(tmp_path):
    store = Store(":memory:")
    for topic in ("generative AI in newsrooms", "generative AI newsroom ethics"):
        orch = Orchestrator(FakeLLM(), store, ScriptedUI(), reports_dir=tmp_path, today=TODAY)
        orch.run(topic)

    events = [e for e in orch.ui.events if e.type == "memory.recalled"]
    assert events and "generative AI in newsrooms" in events[0].message
    plan_prompt = next(p for purpose, p in orch.llm.calls if purpose == "plan")
    assert "Prior research by this user" in plan_prompt
    assert store.suggestions()


def test_clarify_questions_capped(tmp_path):
    many = Clarification(needs_clarification=True, reasoning="r",
                         questions=[f"q{i}" for i in range(6)])
    ui = ScriptedUI()
    orch, _, _, _ = make(tmp_path, ui=ui, handlers={Clarification: lambda p: many})
    orch.run("x")
    assert len(ui.questions) == 3


def test_llm_error_marks_session_failed(tmp_path):
    def boom(prompt):
        raise LLMError("down")

    orch, _, store, _ = make(tmp_path, handlers={PlanDraft: boom})
    with pytest.raises(LLMError):
        orch.run("impact of generative AI on newsrooms")
    assert store.session(orch.sid)["status"] == "failed"


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_sub_agent_searches_show_up_in_the_action_log(tmp_path, kind):
    """A sub-agent used to be a silent black box between task.started and task.done: a live
    run showed nothing for minutes. Each search and fetch is now an event, tagged with the
    task it belongs to and persisted, so the log explains what the agent did."""
    from rootlogic.graph import ResearchGraph

    store, ui = Store(), ScriptedUI()
    kw = dict(reports_dir=tmp_path, today=TODAY)
    engine = (ResearchGraph(FakeLLM(), store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(FakeLLM(), store, ui, **kw))
    engine.run("impact of generative AI on newsrooms")

    searches = [e for e in ui.events if e.type == "subagent.search"]
    fetches = [e for e in ui.events if e.type == "subagent.fetch"]
    assert len(searches) == 3 and len(fetches) == 3        # one per sub-task
    assert {e.data["task"] for e in searches} == {"t1", "t2", "t3"}
    assert all(e.data["results"] for e in searches)
    assert all(e.data["url"].startswith("https://") for e in fetches)

    # every sub-agent event is between its task's start and its end, and is stored
    log = [(e["type"], (e["message"])) for e in store.events(engine.sid)]
    types = [t for t, _ in log]
    for task in ("t1", "t2", "t3"):
        window = [i for i, (t, m) in enumerate(log) if m.startswith(f"[{task}]")]
        assert types[window[0]] == "task.started" and types[window[-1]] == "task.done"
    assert types.count("subagent.search") == 3
