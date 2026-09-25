from datetime import date

import pytest

from rootlogic.control import Command
from rootlogic.fake_llm import FakeLLM, default_finding, default_plan
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


def test_questions_do_not_crash_a_run_with_no_terminal(monkeypatch):
    """Live: a backgrounded run (nohup, so no stdin) died with EOFError from Rich the moment
    the agent asked a clarifying question. Every prompt has a safe default; use it."""
    import rich.prompt

    from rootlogic.cli import TerminalUI
    from rootlogic.models import Plan

    def no_stdin(*a, **kw):
        raise EOFError("EOF when reading a line")

    monkeypatch.setattr(rich.prompt.Prompt, "ask", staticmethod(no_stdin))
    ui = TerminalUI()
    plan = Plan(topic="t", objective="o", recency_days=365, subtasks=[])

    assert ui.ask("Which region matters most?") == ""   # skipped, not crashed
    assert ui.review_plan(plan) is plan                 # default "a": approve and continue
    assert ui.override(plan) == []                      # default "continue": no commands


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_skipping_a_task_names_it_in_the_event_data(tmp_path, kind):
    """The UI groups events by task id: without one, a skipped sub-agent's card never leaves
    the running state while the plan table correctly shows it skipped."""
    from rootlogic.graph import ResearchGraph

    store, ui = Store(), ScriptedUI(overrides=[[Command(action="skip", arg="t2")]])
    ui.pause_before_wave = True
    kw = dict(reports_dir=tmp_path, today=TODAY)
    engine = (ResearchGraph(FakeLLM(), store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(FakeLLM(), store, ui, **kw))
    engine.control.request_pause()
    engine.run("impact of generative AI on newsrooms")

    skips = [e for e in ui.events if e.type == "override.skip"]
    assert skips, [e.type for e in ui.events]
    assert skips[0].data.get("task") == "t2"


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_report_stage_says_where_the_report_will_appear(tmp_path, kind):
    """"Writing report" left users watching a log with no idea where the output lands."""
    from rootlogic.graph import ResearchGraph

    store, ui = Store(), ScriptedUI()
    kw = dict(reports_dir=tmp_path, today=TODAY)
    engine = (ResearchGraph(FakeLLM(), store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(FakeLLM(), store, ui, **kw))
    engine.run("impact of generative AI on newsrooms")

    started = next(e for e in ui.events if e.type == "report.started")
    assert started.message == "Writing the final report to the Result tab"


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_abort_stops_a_run_without_waiting_for_a_pause(tmp_path, kind):
    """Reported from the UI: Pause appeared to do nothing and Abort was unreachable. Pause was
    only checked once per wave - twenty minutes on a local model - and Abort lived inside the
    override card, which only appears after a pause is honoured."""
    from rootlogic.graph import ResearchGraph
    from rootlogic.models import FindingDraft

    store, ui = Store(), ScriptedUI()
    engine_holder = {}

    def abort_midway(prompt):
        engine_holder["e"].control.request_abort()      # user hits Abort during research
        return default_finding(prompt, 1)

    kw = dict(reports_dir=tmp_path, today=TODAY)
    engine = (ResearchGraph(FakeLLM(handlers={FindingDraft: abort_midway}), store, ui,
                            checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else
              Orchestrator(FakeLLM(handlers={FindingDraft: abort_midway}), store, ui, **kw))
    engine_holder["e"] = engine

    assert engine.run("impact of generative AI on newsrooms") is None
    assert store.session(engine.sid)["status"] == "aborted"
    assert "control.aborted" in ui.types()
    assert "report.started" not in ui.types()           # it stopped, it didn't finish quietly


def test_pause_is_noticed_between_sub_tasks_not_only_between_waves(tmp_path):
    """A wave is minutes long; a sub-task is the finest safe point there is."""
    store, ui = Store(), ScriptedUI(overrides=[[Command(action="stop")]])
    holder = {}

    def pause_midway(prompt):
        holder["e"].control.request_pause()
        return default_finding(prompt, 1)

    from rootlogic.models import FindingDraft
    engine = Orchestrator(FakeLLM(handlers={FindingDraft: pause_midway}), store, ui,
                          reports_dir=tmp_path, today=TODAY)
    holder["e"] = engine
    engine.run("impact of generative AI on newsrooms")

    types = ui.types()
    assert "control.paused" in types
    # the pause landed while the wave was still running, so not every task started
    started = [e for e in ui.events if e.type == "task.started"]
    assert len(started) == 3 and types.index("control.paused") < len(types) - 1


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_pause_holds_the_run_until_resume(tmp_path, kind):
    """Pause must actually stop the work and keep it stopped, not stop to ask a question and
    carry on. Reported from the UI: the run kept researching after Pause."""
    import threading
    from rootlogic.graph import ResearchGraph
    from rootlogic.models import FindingDraft

    store, ui = Store(), ScriptedUI()
    holder, started = {}, threading.Event()

    def pause_on_first(prompt):
        if not started.is_set():
            holder["e"].control.request_hold()
            started.set()
        return default_finding(prompt, 1)

    kw = dict(reports_dir=tmp_path, today=TODAY)
    llm = FakeLLM(handlers={FindingDraft: pause_on_first})
    engine = (ResearchGraph(llm, store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(llm, store, ui, **kw))
    holder["e"] = engine

    done = threading.Event()
    result: dict = {}
    t = threading.Thread(target=lambda: (result.update(r=engine.run("newsroom AI")), done.set()))
    t.start()

    assert started.wait(5), "the run never reached a sub-task"
    assert not done.wait(0.5), "the run finished while it was supposed to be held"
    assert "control.paused" in ui.types()
    assert "report.started" not in ui.types()          # held means no further work

    engine.control.release()                            # the user presses Resume
    assert done.wait(10), "the run did not continue after Resume"
    t.join()
    assert result["r"] is not None
    assert "control.resumed" in ui.types()


def test_abort_is_noticed_even_when_every_sub_task_fails(tmp_path):
    """A live run took nine minutes to abort: the failure path skipped the abort check, so a
    wave of failing sub-tasks spent all its retries first."""
    from rootlogic.models import FindingDraft

    store, ui = Store(), ScriptedUI()
    holder, calls = {}, []

    def fail_after_abort(prompt):
        calls.append(prompt)
        holder["e"].control.request_abort()
        raise LLMError("search backend down")

    engine = Orchestrator(FakeLLM(handlers={FindingDraft: fail_after_abort}), store, ui,
                          reports_dir=tmp_path, today=TODAY)
    holder["e"] = engine

    assert engine.run("impact of generative AI on newsrooms") is None
    assert store.session(engine.sid)["status"] == "aborted"
    assert "control.aborted" in ui.types()
    # the wave had three sub-tasks and two retries each; abort must stop it, not outlast it
    assert len(calls) <= 3, f"kept working after abort: {len(calls)} research calls"
    assert ui.types().count("control.aborted") == 1   # one line, not one per thread


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_a_sub_task_dropped_at_review_is_not_left_pending(tmp_path, kind):
    """Found while driving the web UI: dropping t3 from the plan worked - it never ran - but
    its stored row stayed "pending", so the plan table showed work that would never happen."""
    from rootlogic.graph import ResearchGraph

    store = Store()
    ui = ScriptedUI(edit=lambda plan: plan.subtasks.remove(plan.get("t3")))
    kw = dict(reports_dir=tmp_path, today=TODAY)
    engine = (ResearchGraph(FakeLLM(), store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(FakeLLM(), store, ui, **kw))
    engine.run("impact of generative AI on newsrooms")

    rows = {t["task_id"]: t["status"] for t in store.tasks(engine.sid)}
    assert rows["t3"] == "skipped", f"dropped task left as {rows['t3']!r}"
    assert rows["t1"] == "done" and rows["t2"] == "done"


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_a_url_in_the_topic_is_fetched_and_filtered_like_any_other_source(tmp_path, kind):
    """A link pasted into the topic used to be nothing but words in a prompt: a sub-agent might
    fetch it or might not, and if it did the page skipped the source rules on the way in."""
    from rootlogic.search import StaticSearch

    from rootlogic.graph import ResearchGraph

    search = StaticSearch(pages={"https://who.int/report": "Hand hygiene guidance from the WHO."})
    store, ui = Store(), ScriptedUI()
    llm, kw = FakeLLM(search=search), dict(reports_dir=tmp_path, today=TODAY)
    orch = (ResearchGraph(llm, store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
            if kind == "graph" else Orchestrator(llm, store, ui, **kw))
    orch.run("what does https://who.int/report say about hand hygiene")

    assert "https://who.int/report" in search.fetched, "the link should be read, not guessed at"
    assert "seed.fetched" in ui.types()
    rows = [s for s in store.sources(orch.sid) if s["url"] == "https://who.int/report"]
    assert rows and rows[0]["kept"] == 1, "a seed page belongs in the source record"
    prompts_seen = [p for _, p in llm.calls]
    assert any("who.int/report" in p for p in prompts_seen), "the planner should see it"


def test_a_seed_url_on_a_blocked_domain_is_not_fetched(tmp_path):
    """The filters judge what sub-agents bring back; a pasted link reached the network first."""
    from rootlogic.search import StaticSearch

    search = StaticSearch(pages={"https://spam.example/x": "buy things"})
    store, ui = Store(), ScriptedUI()
    orch = Orchestrator(FakeLLM(search=search), store, ui, reports_dir=tmp_path, today=TODAY,
                        blocked_domains=("spam.example",))
    orch.run("summarise https://spam.example/x please")

    assert "https://spam.example/x" not in search.fetched, \
        "a blocked domain should not be reached at all"
    assert "seed.dropped" in ui.types()


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_a_run_that_found_nothing_says_so_instead_of_writing_from_memory(tmp_path, kind):
    """Live on claude-sonnet-5 with search down: every sub-task failed, nothing was retrieved,
    and the run still produced a confident review of PREDIMED and the Mediterranean diet -
    four "consensus points", two "contradictions", and pages of prose, all from the model's
    own memory. An assistant that cannot research a topic has to say so."""
    from rootlogic.graph import ResearchGraph
    from rootlogic.models import FindingDraft

    def refuse(prompt):
        raise LLMError("the model never ran a usable search, so there is nothing to report")

    store, ui = Store(), ScriptedUI()
    llm = FakeLLM(handlers={FindingDraft: refuse})
    kw = dict(reports_dir=tmp_path, today=TODAY)
    engine = (ResearchGraph(llm, store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(llm, store, ui, **kw))
    report = engine.run("does the Mediterranean diet reduce cardiovascular risk")

    assert report is not None, "the run should finish and say what happened"
    assert "analyze.done" not in ui.types(), "there is nothing to cross-check"
    assert not [p for purpose, p in llm.calls if purpose == "report"], \
        "the writer should not be asked to write a report out of nothing"
    text = report.to_markdown().lower()
    assert "no sources" in text or "nothing" in text or "no evidence" in text
    assert "predimed" not in text, "no claims should appear that no source supported"


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_a_planner_that_declines_is_a_refusal_not_a_failed_search(tmp_path, kind):
    """Live, on the harmful-request evals: the planner answered with the objective "Decline to
    provide research assistance for this request" and no sub-tasks. That came out as a "No
    sources found" report and a session marked failed - a search that came up empty, which is
    not what happened, and the evaluation set scored the refusal as a miss."""
    from rootlogic.graph import ResearchGraph
    from rootlogic.llm import AgentRefusal

    declined = {PlanDraft: lambda p: PlanDraft(
        objective="Decline to provide research assistance for this request",
        recency_days=0, subtasks=[])}
    store, ui = Store(":memory:"), ScriptedUI()
    kw = dict(reports_dir=tmp_path, today=TODAY)
    engine = (ResearchGraph(FakeLLM(handlers=declined), store, ui,
                            checkpoint_path=tmp_path / "cp.db", **kw) if kind == "graph"
              else Orchestrator(FakeLLM(handlers=declined), store, ui, **kw))

    with pytest.raises(AgentRefusal, match="declined this request"):
        engine.run("something the model will not research")

    assert store.session(engine.sid)["status"] == "refused"      # not "failed"
    types = ui.types()
    assert "plan.declined" in types and "session.refused" in types
    assert "report" not in types and "session.done" not in types


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_an_exhausted_search_plan_stops_the_run(tmp_path, kind):
    """Live: with Tavily out of credits, every remaining sub-agent still spent model calls to
    retrieve nothing, and the run wrote a report anyway. The first sub-task to hit it ends the
    run instead, so the remaining ones are never dispatched."""
    from rootlogic.graph import ResearchGraph
    from rootlogic.search import SearchQuotaExceeded

    calls = []

    class OutOfCredits(FakeLLM):
        def research(self, **kw):
            calls.append(kw["purpose"])
            raise SearchQuotaExceeded("Tavily is out of credits (HTTP 432).")

    store, ui = Store(":memory:"), ScriptedUI()
    kw = dict(reports_dir=tmp_path, today=TODAY, budget=Budget(max_parallel=1))
    engine = (ResearchGraph(OutOfCredits(), store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(OutOfCredits(), store, ui, **kw))

    with pytest.raises(SearchQuotaExceeded):
        engine.run("impact of generative AI on newsrooms")

    if kind == "loop":
        assert len(calls) == 1                   # the rest of the wave was cancelled, not run
    assert store.session(engine.sid)["status"] == "failed"
    assert any("Search unavailable" in e.message for e in ui.events if e.type == "session.failed")
    assert "report" not in ui.types()             # no full-price report on nothing
