"""Web API: runs execute in background threads; humans answer via HTTP; events stream via SSE."""

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rootlogic.cli import create_engine  # noqa: E402
from rootlogic.store import Store  # noqa: E402
from rootlogic.web import create_app  # noqa: E402

TOPIC = "impact of generative AI on newsrooms"


@pytest.fixture
def client(tmp_path):
    store = Store(tmp_path / "rl.db")

    def factory(store, ui, **kw):
        kw["offline"] = True  # never hit the real API in tests
        return create_engine(store, ui, **kw)

    return TestClient(create_app(store, tmp_path, engine_factory=factory))


def age(store, sid, seconds):
    """Pretend a session (and its events) last showed life `seconds` ago."""
    old = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="seconds")
    store._exec("UPDATE sessions SET created_at = ? WHERE id = ?", (old, sid))
    store._exec("UPDATE events SET ts = ? WHERE session_id = ?", (old, sid))


def wait(client, run_id, until, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/api/runs/{run_id}").json()
        if until(s):
            return s
        time.sleep(0.02)
    raise AssertionError(f"timed out; last status {s}")


def pending(kind):
    return lambda s: s["pending"] and s["pending"]["kind"] == kind


def finished(s):
    return s["status"] != "running"


def sse_events(client, run_id, last_event_id=None):
    headers = {"Last-Event-ID": str(last_event_id)} if last_event_id is not None else {}
    events = []
    with client.stream("GET", f"/api/runs/{run_id}/events", headers=headers) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: {}":
                events.append(json.loads(line[6:]))
    return events


@pytest.mark.parametrize("engine", ["loop", "graph"])
def test_full_run_with_plan_approval(client, engine):
    run = client.post("/api/runs", json={"topic": TOPIC, "engine": engine}).json()
    s = wait(client, run["run_id"], pending("plan"))
    assert len(s["pending"]["plan"]["subtasks"]) == 3

    client.post(f"/api/runs/{run['run_id']}/answer",
                json={"request_id": s["pending"]["request_id"],
                      "answer": {"approved": True, "drop": ["t2"], "add": ["Who funds this?"],
                                 "recency_days": 200}})
    s = wait(client, run["run_id"], finished)
    assert s["status"] == "done"

    events = sse_events(client, run["run_id"])
    types = [e["type"] for e in events]
    assert types[0] == "session.started" and types[-1] == "run.finished"
    assert "request" in types and "request.resolved" in types and "report" in types
    started = [e["message"] for e in events if e["type"] == "task.started"]
    assert not any("[t2]" in m for m in started)
    assert any("Who funds this?" in m for m in started)
    assert [e["seq"] for e in events] == list(range(len(events)))

    detail = client.get(f"/api/sessions/{s['session_id']}").json()
    assert detail["report"].startswith("# ") and detail["usage"]["calls"] > 0
    assert client.get("/api/sessions").json()["suggestions"]


def test_clarifying_questions_round_trip(client):
    run = client.post("/api/runs", json={"topic": "AI"}).json()
    rid = run["run_id"]
    s = wait(client, rid, pending("question"))
    client.post(f"/api/runs/{rid}/answer",
                json={"request_id": s["pending"]["request_id"], "answer": "last 12 months"})
    s = wait(client, rid, lambda s: pending("question")(s) and "audience" in s["pending"]["question"])
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": ""})
    s = wait(client, rid, pending("plan"))
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": {"approved": True}})
    s = wait(client, rid, finished)
    kinds = [m["kind"] for m in client.get(f"/api/sessions/{s['session_id']}").json()["messages"]]
    assert kinds.count("clarifying_question") == 2 and kinds.count("answer") == 1


def test_reject_plan_aborts(client):
    rid = client.post("/api/runs", json={"topic": TOPIC}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": {"approved": False}})
    assert wait(client, rid, finished)["status"] == "aborted"


def test_pause_then_override_stop(client):
    rid = client.post("/api/runs", json={"topic": TOPIC}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    client.post(f"/api/runs/{rid}/checkpoint")  # lands at the first checkpoint after approval
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": {"approved": True}})
    s = wait(client, rid, pending("override"))
    client.post(f"/api/runs/{rid}/answer",
                json={"request_id": s["pending"]["request_id"],
                      "answer": {"commands": [{"action": "note", "arg": "focus on EU"},
                                              {"action": "stop"}]}})
    s = wait(client, rid, finished)
    assert s["status"] == "done"
    types = [e["type"] for e in sse_events(client, rid)]
    assert "override.note" in types and "override.stop" in types and "task.started" not in types


def test_sse_resumes_from_last_event_id(client):
    rid = client.post("/api/runs", json={"topic": TOPIC}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": {"approved": True}})
    wait(client, rid, finished)
    everything = sse_events(client, rid)
    tail = sse_events(client, rid, last_event_id=4)
    assert tail == everything[5:]


def test_stale_answer_is_rejected(client):
    rid = client.post("/api/runs", json={"topic": TOPIC}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    req = s["pending"]["request_id"]
    assert client.post(f"/api/runs/{rid}/answer",
                       json={"request_id": req, "answer": {"approved": False}}).status_code == 200
    r = client.post(f"/api/runs/{rid}/answer", json={"request_id": req, "answer": {}})
    assert r.status_code == 409


def test_graph_session_resumes_from_web(client, tmp_path):
    rid = client.post("/api/runs", json={"topic": TOPIC, "engine": "graph"}).json()["run_id"]
    wait(client, rid, pending("plan"))
    sid = client.get(f"/api/runs/{rid}").json()["session_id"]
    # Resuming a session that is still live must be refused.
    assert client.post(f"/api/sessions/{sid}/resume", json={}).status_code == 409


def test_index_and_validation(client):
    assert "rootlogic" in client.get("/").text
    assert client.post("/api/runs", json={"topic": ""}).status_code == 422
    assert client.get("/api/runs/nope").status_code == 404


def test_orphaned_sessions_marked_interrupted_on_startup(tmp_path):
    store = Store(tmp_path / "rl.db")
    sid = store.create_session("left running by a dead server")
    age(store, sid, 3600)          # quiet for an hour: nothing is working on it
    create_app(store, tmp_path, engine_factory=lambda *a, **k: None)
    assert store.session(sid)["status"] == "interrupted"


def test_profile_api_and_follow_up_run(client):
    assert client.get("/api/profile").json()["preferences"] == []
    added = client.post("/api/profile", json={"text": "Focuses on the EU", "category": "region"})
    pref = added.json()["preferences"][0]
    assert added.json()["added"] and pref["category"] == "region"
    assert not client.post("/api/profile", json={"text": "focuses on the eu"}).json()["added"]

    # a run uses the profile, then a follow-up continues it
    rid = client.post("/api/runs", json={"topic": TOPIC}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": {"approved": True}})
    parent = wait(client, rid, finished)["session_id"]
    types = [e["type"] for e in sse_events(client, rid)]
    assert "profile.loaded" in types

    assert client.post("/api/runs", json={"topic": "x y", "parent_session": "nope"}).status_code \
        == 404
    rid2 = client.post("/api/runs", json={"topic": "How are unions responding?",
                                          "parent_session": parent}).json()["run_id"]
    s = wait(client, rid2, pending("plan"))
    origins = [t["origin"] for t in s["pending"]["plan"]["subtasks"]]
    assert origins.count("previous") == 3 and origins.count("planner") == 3
    client.post(f"/api/runs/{rid2}/answer", json={"request_id": s["pending"]["request_id"],
                                                  "answer": {"approved": True}})
    child = wait(client, rid2, finished)
    assert child["status"] == "done" and child["parent_session"] == parent
    rows = {r["id"]: r for r in client.get("/api/sessions").json()["sessions"]}
    assert rows[child["session_id"]]["parent_id"] == parent

    assert client.delete(f"/api/profile/{pref['id']}").json()["preferences"] == []
    assert client.delete(f"/api/profile/{pref['id']}").status_code == 404


def test_source_rules_api(client):
    assert client.get("/api/sources").json()["rules"] == []
    rules = client.post("/api/sources", json={"domain": "https://www.WHO.int/x",
                                              "rule": "trust"}).json()["rules"]
    assert [(r["domain"], r["rule"]) for r in rules] == [("who.int", "trust")]
    assert client.post("/api/sources", json={"domain": "nodots", "rule": "block"}).status_code == 422
    assert client.post("/api/sources", json={"domain": "a.com", "rule": "nuke"}).status_code == 422
    assert client.delete("/api/sources/who.int").json()["rules"] == []
    assert client.delete("/api/sources/who.int").status_code == 404


def test_run_can_disable_verification(client):
    rid = client.post("/api/runs", json={"topic": TOPIC, "verify_claims": 0}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": {"approved": True}})
    wait(client, rid, finished)
    types = [e["type"] for e in sse_events(client, rid)]
    assert "verify.started" not in types and "report.checked" in types


def test_plan_prompt_keeps_its_bracketed_letters():
    """Rich reads [a] as a style tag: unescaped, the prompt renders as 'pprove, dit, uit'."""
    from rich.console import Console

    from rootlogic.cli import PLAN_PROMPT
    console = Console(width=80)
    with console.capture() as capture:
        console.print(PLAN_PROMPT)
    assert "[a]pprove, [e]dit, [q]uit" in capture.get()


def test_every_frontend_asset_is_served(tmp_path):
    """The UI is split into ES modules with no bundler, so a mistyped import path or a file
    the server doesn't expose is a blank page at demo time, not a build error."""
    from pathlib import Path

    client = TestClient(create_app(Store(tmp_path / "w.db"), tmp_path))
    page = client.get("/")
    assert page.status_code == 200
    assert '<script type="module" src="/static/js/main.js">' in page.text
    assert '<link rel="stylesheet" href="/static/css/app.css">' in page.text

    static = Path(__file__).resolve().parents[1] / "rootlogic" / "static"
    files = [p.relative_to(static).as_posix() for p in static.rglob("*")
             if p.is_file() and p.suffix in (".js", ".css")]
    assert len(files) >= 8, files
    for rel in files:
        r = client.get(f"/static/{rel}")
        assert r.status_code == 200, rel
        expected = "text/javascript" if rel.endswith(".js") else "text/css"
        assert expected in r.headers["content-type"], (rel, r.headers["content-type"])


def test_module_imports_resolve_to_files_that_exist():
    """Browsers resolve these paths themselves; a typo fails silently in the console."""
    import re
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "rootlogic" / "static"
    for path in (static / "js").glob("*.js"):
        for imported in re.findall(r'from\s+"(\./[^"]+)"', path.read_text()):
            assert (path.parent / imported).is_file(), f"{path.name} imports missing {imported}"


def test_prompt_endpoints_read_edit_and_reset(client):
    got = client.get("/api/prompts").json()
    assert [p["name"] for p in got["prompts"]] == ["planner", "researcher", "verifier", "writer"]
    planner = next(p for p in got["prompts"] if p["name"] == "planner")
    assert planner["default"].startswith("You are the planner")
    assert planner["text"] == planner["default"] and planner["custom"] is False

    assert client.post("/api/prompts", json={"name": "researcher",
                                             "text": "Prefer official statistics."}).status_code == 200
    got = client.get("/api/prompts").json()
    researcher = next(p for p in got["prompts"] if p["name"] == "researcher")
    assert researcher["text"] == "Prefer official statistics." and researcher["custom"] is True

    assert client.delete("/api/prompts/researcher").status_code == 200
    got = client.get("/api/prompts").json()
    assert next(p for p in got["prompts"] if p["name"] == "researcher")["custom"] is False


def test_editing_a_prompt_that_does_not_exist_is_rejected(client):
    assert client.post("/api/prompts", json={"name": "judge", "text": "x"}).status_code == 422
    assert client.delete("/api/prompts/judge").status_code == 422


def test_a_run_uses_the_stored_prompts(client, tmp_path):
    """The point of storing them: the next run picks them up without being told."""
    client.post("/api/prompts", json={"name": "writer", "text": "Write in British English."})
    run = client.post("/api/runs", json={"topic": "impact of generative AI on newsrooms",
                                         "offline": True}).json()
    # The run pauses for plan approval, which is enough: disclosure happens at session start.
    d = wait(client, run["run_id"], lambda r: r["events"] >= 3)

    events = client.get(f"/api/sessions/{d['session_id']}").json()["events"]
    assert any(e["type"] == "prompt.custom" and "writer" in e["message"] for e in events)
    assert client.get(f"/api/sessions/{d['session_id']}").json()["session"]["prompts_json"]


def test_stored_events_carry_data_as_an_object(client):
    """The live stream sends data as an object; replaying a session sent the raw JSON string,
    so the UI's ev.data.task lookups silently found nothing and the cards stayed empty."""
    run = client.post("/api/runs", json={"topic": "impact of generative AI on newsrooms",
                                         "offline": True}).json()
    d = wait(client, run["run_id"], lambda r: r["events"] >= 3)

    events = client.get(f"/api/sessions/{d['session_id']}").json()["events"]
    with_data = [e for e in events if e.get("data")]
    assert with_data, events
    for e in with_data:
        assert isinstance(e["data"], dict), e


def test_a_run_can_set_searches_and_parallelism(client):
    """A local model is one server: 8 searches per sub-agent and 4 in parallel starves it.
    The CLI has always had these knobs; the web UI could not reach them."""
    run = client.post("/api/runs", json={"topic": "impact of generative AI on newsrooms",
                                         "offline": True, "max_searches": 2,
                                         "max_parallel": 1}).json()
    d = wait(client, run["run_id"], lambda r: r["events"] >= 3)
    budget = client.get(f"/api/runs/{d['run_id']}").json()["budget"]
    assert budget["max_searches"] == 2 and budget["max_parallel"] == 1


def test_search_and_parallel_limits_are_bounded(client):
    assert client.post("/api/runs", json={"topic": "x y", "max_searches": 99}).status_code == 422
    assert client.post("/api/runs", json={"topic": "x y", "max_parallel": 0}).status_code == 422


def test_starting_a_second_server_does_not_relabel_a_live_session(tmp_path):
    """Found live: running `rootlogic web` while a run was in flight marked its session
    'interrupted'. create_app ran mark_interrupted before uvicorn bound the port, so even a
    server that failed with 'address already in use' rewrote the data first."""
    store = Store(tmp_path / "rl.db")
    fresh = store.create_session("a run that is still going")
    store.add_event(fresh, "subagent.search", "[t1] Searched “x” — 5 result(s)")
    stale = store.create_session("a run orphaned by an old server")
    age(store, stale, 3600)

    create_app(store, tmp_path)          # a second server starting up

    assert store.session(fresh)["status"] == "running", "a live session must not be touched"
    assert store.session(stale)["status"] == "interrupted"


def test_a_session_that_died_before_emitting_anything_is_still_marked(tmp_path):
    """A run killed before its first event is an orphan too, once it has gone quiet."""
    store = Store(tmp_path / "rl.db")
    sid = store.create_session("died immediately")
    age(store, sid, 3600)
    create_app(store, tmp_path)
    assert store.session(sid)["status"] == "interrupted"


def test_a_failed_run_can_be_retried_reusing_its_findings(client):
    """A run that died late has already paid for its research: one live failure cost 58 LLM
    calls and $1.26 with eight of nine sub-tasks complete, and the UI offered only Delete."""
    rid = client.post("/api/runs", json={"topic": TOPIC}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": {"approved": True}})
    s = wait(client, rid, finished)
    sid = s["session_id"]
    client.get(f"/api/sessions/{sid}")          # the run completed; pretend it failed late
    Store  # noqa: B018 - imported above

    retry = client.post(f"/api/sessions/{sid}/retry", json={})
    assert retry.status_code == 200
    body = retry.json()
    assert body["parent_session"] == sid       # a follow-up: earlier findings are carried

    s2 = wait(client, body["run_id"], pending("plan"))
    carried = [t for t in s2["pending"]["plan"]["subtasks"] if t["origin"] == "previous"]
    assert carried, "the retry must reuse the work already paid for"


def test_retry_refuses_a_live_session_and_an_unknown_one(client):
    rid = client.post("/api/runs", json={"topic": TOPIC}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    assert client.post(f"/api/sessions/{s['session_id']}/retry", json={}).status_code == 409
    assert client.post("/api/sessions/nope/retry", json={}).status_code == 404


def test_abort_stops_a_run_without_an_override_card(client):
    """Reported from the UI: Abort only existed inside the override card, so a run that
    would not pause could not be stopped."""
    rid = client.post("/api/runs", json={"topic": TOPIC}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    assert client.post(f"/api/runs/{rid}/abort").status_code == 200
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": {"approved": True}})
    s = wait(client, rid, finished)
    assert s["status"] == "aborted"
    types = [e["type"] for e in sse_events(client, rid)]
    assert "control.aborted" in types and "report.started" not in types
    assert client.post(f"/api/runs/{rid}/abort").status_code == 409   # already finished


def test_pause_holds_the_run_and_resume_releases_it(client):
    """Pause is a hold, not a question: nothing further runs until Resume is pressed."""
    rid = client.post("/api/runs", json={"topic": TOPIC}).json()["run_id"]
    s = wait(client, rid, pending("plan"))
    assert client.post(f"/api/runs/{rid}/pause").status_code == 200
    client.post(f"/api/runs/{rid}/answer", json={"request_id": s["pending"]["request_id"],
                                                 "answer": {"approved": True}})
    # The stream stays open while a run lives, so read the status instead: a held run must
    # still be running a second later, not finished.
    time.sleep(1.0)
    assert client.get(f"/api/runs/{rid}").json()["status"] == "running"

    client.post(f"/api/runs/{rid}/resume")
    assert wait(client, rid, finished)["status"] == "done"
    types = [e["type"] for e in sse_events(client, rid)]
    assert types.index("control.paused") < types.index("control.resumed")
    assert types.index("control.resumed") < types.index("report.started")


def test_report_typography_is_scoped_to_the_report_tab():
    """The action log gives every event a class from its family, so `report.started` renders
    as `<div class="ev report">`. Styling bare `.report` therefore resized log lines."""
    from pathlib import Path
    css = (Path(__file__).resolve().parents[1] / "rootlogic" / "static" / "css" / "app.css").read_text()
    for line in css.splitlines():
        selector = line.split("{")[0].strip()
        if not selector.startswith(".report"):
            continue
        assert selector.startswith(".report-actions"), \
            f"bare .report selector also hits log rows: {selector}"
