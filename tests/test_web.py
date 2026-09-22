"""Web API: runs execute in background threads; humans answer via HTTP; events stream via SSE."""

import json
import time

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
    client.post(f"/api/runs/{rid}/pause")  # lands at the first checkpoint after approval
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
