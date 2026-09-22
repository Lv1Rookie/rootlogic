"""User profile (standing preferences) and follow-up research threads, on both engines."""

import sqlite3
from datetime import date

import pytest

from rootlogic.continuity import learn_profile, load_previous
from rootlogic.control import Command
from rootlogic.fake_llm import FakeLLM
from rootlogic.graph import ResearchGraph
from rootlogic.llm import LLMError
from rootlogic.models import Preference, ProfileUpdate, Reflection, SubTaskDraft
from rootlogic.orchestrator import Budget, Orchestrator
from rootlogic.store import Store

from .test_orchestrator import ScriptedUI

TODAY = date(2026, 9, 22)
TOPIC = "impact of generative AI on newsrooms"
ENGINES = ["loop", "graph"]


def engine(kind, store, tmp_path, *, ui=None, handlers=None, budget=None, use_profile=True):
    llm = FakeLLM(handlers=handlers)
    ui = ui or ScriptedUI()
    kw = dict(budget=budget or Budget(), reports_dir=tmp_path, today=TODAY,
              use_profile=use_profile)
    if kind == "graph":
        eng = ResearchGraph(llm, store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
    else:
        eng = Orchestrator(llm, store, ui, **kw)
    return eng, llm, ui


def prompt_for(llm, purpose):
    return next(p for pu, p in llm.calls if pu == purpose)


def purposes(llm):
    return [p for p, _ in llm.calls]


# =================================================================== profile


@pytest.mark.parametrize("kind", ENGINES)
def test_answers_become_standing_preferences_used_next_session(tmp_path, kind):
    store = Store(tmp_path / "rl.db")
    first, llm1, _ = engine(kind, store, tmp_path, ui=ScriptedUI(answers=["policy analysts", ""]))
    first.run("AI")  # vague topic -> 2 clarifying questions, one answered

    assert "profile" in purposes(llm1)
    prefs = store.preferences()
    assert [p["text"] for p in prefs] == ["Prefers: policy analysts"]
    assert prefs[0]["session_id"] == first.sid

    second, llm2, ui2 = engine(kind, store, tmp_path)
    second.run(TOPIC)
    for purpose in ("clarify", "plan", "research:t1", "report"):
        assert "Standing user preference (other): Prefers: policy analysts" in \
            prompt_for(llm2, purpose), purpose
    assert "profile.loaded" in ui2.types()
    assert "profile" not in purposes(llm2)  # no user input this session -> no profile call


def test_profile_update_can_remove_contradicted_preferences(tmp_path):
    store = Store()
    store.add_preference("audience", "Writes for students")
    keep = store.add_preference("region", "Focuses on the EU")
    sid = store.create_session("x")
    store.add_message(sid, "agent", "clarifying_question", "Who is the audience?")
    store.add_message(sid, "user", "answer", "Now I write for policy analysts")
    old_id = next(p["id"] for p in store.preferences() if "students" in p["text"])

    def update(prompt):
        assert f"id {old_id} (audience): Writes for students" in prompt
        assert "User answered: Now I write for policy analysts" in prompt
        return ProfileUpdate(reasoning="audience changed", remove_ids=[old_id, 999],
                             add=[Preference(category="audience",
                                             text="Writes for policy analysts")])

    added, removed = learn_profile(FakeLLM(handlers={ProfileUpdate: update}), store, sid, "x")
    assert added == ["Writes for policy analysts"] and removed == ["Writes for students"]
    assert keep and {p["text"] for p in store.preferences()} == \
        {"Focuses on the EU", "Writes for policy analysts"}


def test_duplicate_preferences_are_ignored(tmp_path):
    store = Store()
    assert store.add_preference("region", "Focuses on the EU")
    assert not store.add_preference("region", "focuses on the eu")
    assert len(store.preferences()) == 1


@pytest.mark.parametrize("kind", ENGINES)
def test_no_profile_flag_neither_reads_nor_learns(tmp_path, kind):
    store = Store(tmp_path / "rl.db")
    store.add_preference("audience", "Writes for students")
    eng, llm, _ = engine(kind, store, tmp_path, ui=ScriptedUI(answers=["x", "y"]),
                         use_profile=False)
    eng.run("AI")
    assert "Standing user preference" not in prompt_for(llm, "plan")
    assert "profile" not in purposes(llm)
    assert len(store.preferences()) == 1


def test_profile_failure_does_not_fail_the_research(tmp_path):
    def boom(prompt):
        raise LLMError("down")

    store = Store()
    eng, _, ui = engine("loop", store, tmp_path, ui=ScriptedUI(answers=["x", "y"]),
                        handlers={ProfileUpdate: boom})
    assert eng.run("AI") is not None
    assert store.session(eng.sid)["status"] == "done"
    assert "profile.skipped" in ui.types()


# =================================================================== follow-ups


@pytest.mark.parametrize("kind", ENGINES)
def test_follow_up_builds_on_earlier_findings(tmp_path, kind):
    store = Store(tmp_path / "rl.db")
    first, _, _ = engine(kind, store, tmp_path, ui=ScriptedUI(overrides=[]))
    first.control.request_pause()
    first.ui.overrides = [[Command("note", "focus on local papers")]]
    first.run(TOPIC)
    parent = first.sid

    second, llm2, ui2 = engine(kind, store, tmp_path)
    report = second.run("How are unions responding?", parent=parent)

    assert report is not None
    assert store.session(second.sid)["parent_id"] == parent
    # earlier questions are shown to the planner as already answered ...
    plan_prompt = prompt_for(llm2, "plan")
    assert f"Earlier research in this thread (session {parent})" in plan_prompt
    assert "[p1] What is the current state" in plan_prompt
    # ... are not researched again (only the new plan's t1..t3 run) ...
    assert sorted(p for p in purposes(llm2) if p.startswith("research:")) == \
        ["research:t1", "research:t2", "research:t3"]
    # ... the earlier user guidance carries over, and the report cites old + new sources
    assert "User guidance: focus on local papers" in prompt_for(llm2, "research:t1")
    assert "follows up on earlier research" in prompt_for(llm2, "report")
    urls = {s.url for s in report.sources}
    assert {f"https://example.org/report-{n}" for n in (1, 2, 3)} <= urls  # carried over
    # The fake model re-finds the same pages: they count as already seen, not re-added.
    dropped = [r for r in store.sources(second.sid) if not r["kept"]]
    assert any(r["reason"] == "duplicate" for r in dropped)
    assert len(urls) == 6
    assert "followup.loaded" in [e.type for e in ui2.events]
    tasks = {t["task_id"]: t for t in store.tasks(second.sid)}
    assert tasks["p1"]["origin"] == "previous" and tasks["p1"]["status"] == "done"


def test_follow_up_chains_and_renumbers(tmp_path):
    store = Store()
    a, _, _ = engine("loop", store, tmp_path)
    a.run(TOPIC)
    b, _, _ = engine("loop", store, tmp_path)
    b.run("follow-up one", parent=a.sid)
    prev = load_previous(store, b.sid)
    # b carried a's 3 findings (p1-p3) and added 3 of its own -> 6, renumbered p1..p6
    assert list(prev.findings) == [f"p{i}" for i in range(1, 7)]
    c, llm, _ = engine("loop", store, tmp_path)
    c.run("follow-up two", parent=b.sid)
    assert "[p6]" in prompt_for(llm, "plan")


def test_earlier_tasks_do_not_count_against_task_budget(tmp_path):
    store = Store()
    a, _, _ = engine("loop", store, tmp_path)
    a.run(TOPIC)

    def more(prompt):
        return Reflection(sufficient=False, reasoning="gap", questions_for_user=[],
                          new_subtasks=[SubTaskDraft(question="Extra angle", rationale="r",
                                                     search_queries=["q"], depends_on=[])])

    b, llm, _ = engine("loop", store, tmp_path, handlers={Reflection: more},
                       budget=Budget(max_tasks=4, max_rounds=1))
    b.run("follow-up", parent=a.sid)
    # 3 previous + 3 new; budget 4 leaves room for exactly one reflection task (t4)
    assert "research:t4" in purposes(llm)


def test_follow_up_of_unknown_session_is_rejected(tmp_path):
    for kind in ENGINES:
        eng, _, _ = engine(kind, Store(), tmp_path)
        with pytest.raises(ValueError):
            eng.run("x", parent="nope")


def test_parent_id_migration_for_old_databases(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, topic TEXT NOT NULL, status TEXT "
                 "NOT NULL, created_at TEXT NOT NULL, finished_at TEXT, plan_json TEXT, "
                 "summary TEXT, related_topics TEXT, report_path TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('old1','t','done','2026-01-01',NULL,NULL,NULL,"
                 "NULL,NULL)")
    conn.commit()
    conn.close()
    store = Store(path)
    assert store.session("old1")["parent_id"] is None
    assert store.session(store.create_session("new", parent_id="old1"))["parent_id"] == "old1"

