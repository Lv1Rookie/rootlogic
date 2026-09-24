"""Web UI: FastAPI + Server-Sent Events.

    rootlogic web            # http://127.0.0.1:8000

How it fits the rest of the system
  * Each research run executes an engine (loop or graph) in a background thread.
  * ``WebInteraction`` implements the same ``Interaction`` protocol as the terminal UI.
    Engine events are appended to the run's event list. When the engine needs a human
    (clarifying question, plan approval, override), ``WebInteraction`` publishes a
    ``request`` event and blocks the engine thread until the browser POSTs an answer.
  * The browser follows ``GET /api/runs/{id}/events`` (SSE). Events carry sequential ids, so
    a refresh or reconnect replays from ``Last-Event-ID`` and the page rebuilds its state
    from the stream alone.

Bound to 127.0.0.1 by default: there is no authentication, and it spends your API credits.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .backend import Backend, BackendError
from .control import Command, Event
from .filters import clean_domain
from .models import Plan, SubTaskDraft
from .orchestrator import Budget
from .prompts import DEFAULTS as prompt_defaults, EDITABLE
from .store import Store

STATIC = Path(__file__).parent / "static"


# =================================================================== run registry


class Run:
    """One engine execution plus its replayable event stream and pending human request."""

    def __init__(self, topic: str, engine: str, offline: bool, parent: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.parent = parent
        self.topic = topic
        self.engine_name = engine
        self.offline = offline
        self.engine: Any = None
        self.budget: Any = None            # the caps this run was started with
        self.status = "running"            # running | done | aborted | failed
        self.events: list[dict] = []
        self.pending: dict | None = None   # the request currently waiting for the user
        self._answer: Any = None
        self._cond = threading.Condition()

    @property
    def sid(self) -> str:
        return getattr(self.engine, "sid", "") or ""

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def push(self, event: dict) -> None:
        event.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with self._cond:
            event["seq"] = len(self.events)
            self.events.append(event)
            self._cond.notify_all()

    def since(self, cursor: int) -> list[dict]:
        with self._cond:
            return self.events[cursor:]

    # ------------------------------------------------------------- human requests
    def request(self, kind: str, payload: dict) -> Any:
        """Called from the engine thread. Blocks until ``respond`` supplies an answer."""
        rid = uuid.uuid4().hex[:8]
        with self._cond:
            self.pending = {"request_id": rid, "kind": kind, **payload}
            self._answer = None
        self.push({"type": "request", **self.pending})
        with self._cond:
            while self.pending is not None and self.pending["request_id"] == rid:
                self._cond.wait()
            answer = self._answer
        self.push({"type": "request.resolved", "request_id": rid, "kind": kind})
        return answer

    def respond(self, request_id: str, answer: Any) -> None:
        with self._cond:
            if not self.pending or self.pending["request_id"] != request_id:
                raise HTTPException(409, "No such pending request (already answered?)")
            self._answer = answer
            self.pending = None
            self._cond.notify_all()

    def summary(self) -> dict:
        return {"run_id": self.id, "session_id": self.sid, "topic": self.topic,
                "engine": self.engine_name, "offline": self.offline, "status": self.status,
                "parent_session": self.parent,
                "pending": self.pending, "events": len(self.events),
                "budget": vars(self.budget) if self.budget else None}


class WebInteraction:
    """``Interaction`` for the browser: events → stream, decisions ← HTTP."""

    def __init__(self, run: Run):
        self.run = run

    def on_event(self, e: Event) -> None:
        self.run.push({"type": e.type, "message": e.message, "ts": e.ts, "data": e.data})

    def ask(self, question: str) -> str:
        answer = self.run.request("question", {"question": question})
        return str(answer or "")

    def review_plan(self, plan: Plan) -> Plan | None:
        answer = self.run.request("plan", {"plan": plan.model_dump()}) or {}
        if not answer.get("approved"):
            return None
        return apply_plan_edits(plan, drop=answer.get("drop", []), add=answer.get("add", []),
                                recency_days=answer.get("recency_days"))

    def override(self, plan: Plan) -> list[Command]:
        answer = self.run.request("override", {"plan": plan.model_dump()}) or {}
        return [Command(action=c["action"], arg=c.get("arg", ""))
                for c in answer.get("commands", [])]


def apply_plan_edits(plan: Plan, *, drop: list[str], add: list[str],
                     recency_days: int | None) -> Plan:
    dropped = set(drop)
    plan.subtasks = [t for t in plan.subtasks if t.id not in dropped]
    for t in plan.subtasks:
        t.depends_on = [d for d in t.depends_on if d not in dropped]
    for q in (q.strip() for q in add):
        if q:
            plan.add(SubTaskDraft(question=q, rationale="Added by user", search_queries=[q],
                                  depends_on=[]), origin="user")
    if recency_days is not None and recency_days >= 0:
        plan.recency_days = recency_days
    return plan


# =================================================================== API models


class NewPrompt(BaseModel):
    name: Literal["planner", "researcher", "verifier", "writer"]
    text: str = Field(max_length=20000)


class StartRun(BaseModel):
    topic: str = Field(min_length=2, max_length=500)
    engine: Literal["loop", "graph"] = "loop"
    offline: bool = False
    max_rounds: int = Field(2, ge=0, le=5)
    max_tasks: int = Field(10, ge=1, le=20)
    parent_session: str | None = None   # follow up on this earlier session
    use_profile: bool = True
    verify_claims: int = Field(12, ge=0, le=40)  # 0 disables claim verification
    # A local model is one server: many searches and parallel sub-agents starve it, while a
    # hosted API benefits from both. The CLI has always exposed these; the UI needs them too.
    max_searches: int = Field(8, ge=1, le=20)    # web searches per sub-agent
    max_parallel: int = Field(4, ge=1, le=8)     # concurrent research sub-agents


class NewSourceRule(BaseModel):
    domain: str = Field(min_length=3, max_length=200)
    rule: Literal["block", "allow", "trust", "distrust"]


class NewPreference(BaseModel):
    text: str = Field(min_length=2, max_length=300)
    category: Literal["audience", "region", "time_window", "sources_prefer", "sources_avoid",
                      "format", "expertise", "other"] = "other"


class Answer(BaseModel):
    request_id: str
    answer: Any = None


class ResumeRun(BaseModel):
    offline: bool = False


# =================================================================== app


def create_app(store: Store, home: Path, *, engine_factory=None,
               backend: Backend | None = None) -> FastAPI:
    """``engine_factory(store, ui, engine=, offline=, budget=, home=)`` is injectable for tests."""
    if engine_factory is None:
        from .cli import create_engine as engine_factory

    app = FastAPI(title="rootlogic", docs_url="/api/docs")
    runs: dict[str, Run] = {}
    # Runs live in this process, so anything still "running" on startup was orphaned by a
    # previous server. Graph-engine sessions among them can be resumed from the UI.
    store.mark_interrupted()

    def get_run(run_id: str) -> Run:
        if run_id not in runs:
            raise HTTPException(404, "Unknown run")
        return runs[run_id]

    def launch(run: Run, target: str, arg: str, budget: Budget, *, use_profile: bool = True,
               **call_kw) -> None:
        from .search import SearchError
        try:
            run.engine = engine_factory(store, WebInteraction(run), engine=run.engine_name,
                                        offline=run.offline, budget=budget, home=home,
                                        backend=backend, use_profile=use_profile)
        except (SearchError, BackendError) as e:
            runs.pop(run.id, None)
            raise HTTPException(400, f"Model/search settings: {e}") from e

        def work():
            try:
                report = getattr(run.engine, target)(arg, **call_kw)
                if report is not None:
                    run.push({"type": "report", "markdown": report.to_markdown(),
                              "session_id": run.sid})
                    run.status = "done"
                else:
                    session = store.session(run.sid) if run.sid else None
                    run.status = (session or {}).get("status") or "aborted"
                    if run.status == "running":
                        run.status = "aborted"
            except Exception as e:  # surfaced to the browser; engine already logged it
                run.push({"type": "run.error", "message": f"{type(e).__name__}: {e}"})
                run.status = "failed"
            finally:
                run.push({"type": "run.finished", "status": run.status, "session_id": run.sid})

        threading.Thread(target=work, name=f"run-{run.id}", daemon=True).start()

    # ------------------------------------------------------------- pages
    # The UI is plain ES modules: no build step, so the browser resolves the imports itself.
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    # ------------------------------------------------------------- runs (live)
    @app.post("/api/runs")
    def start_run(body: StartRun) -> dict:
        if body.parent_session and not store.session(body.parent_session):
            raise HTTPException(404, "Unknown parent session")
        run = Run(body.topic.strip(), body.engine, body.offline, parent=body.parent_session)
        runs[run.id] = run
        run.budget = Budget(max_rounds=body.max_rounds, max_tasks=body.max_tasks,
                            verify_claims=body.verify_claims, max_searches=body.max_searches,
                            max_parallel=body.max_parallel)
        launch(run, "run", run.topic, run.budget,
               use_profile=body.use_profile, parent=body.parent_session)
        return run.summary()

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return [r.summary() for r in runs.values()]

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str) -> dict:
        return get_run(run_id).summary()

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        run = get_run(run_id)
        last = request.headers.get("last-event-id")
        cursor = int(last) + 1 if last and last.isdigit() else 0

        async def stream():
            nonlocal cursor
            idle = 0
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    return
                batch = run.since(cursor)
                for ev in batch:
                    yield f"id: {ev['seq']}\ndata: {json.dumps(ev)}\n\n"
                    cursor = ev["seq"] + 1
                if batch:
                    idle = 0
                elif run.finished:
                    yield "event: end\ndata: {}\n\n"
                    return
                else:
                    idle += 1
                    if idle % 50 == 0:  # ~10 s
                        yield ": keepalive\n\n"
                await asyncio.sleep(0.2)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/runs/{run_id}/answer")
    def answer(run_id: str, body: Answer) -> dict:
        get_run(run_id).respond(body.request_id, body.answer)
        return {"ok": True}

    @app.post("/api/runs/{run_id}/pause")
    def pause(run_id: str) -> dict:
        run = get_run(run_id)
        if run.finished:
            raise HTTPException(409, "Run already finished")
        run.engine.control.request_pause()
        run.push({"type": "control.requested", "message": "Pause requested — will stop at the "
                                                          "next checkpoint"})
        return {"ok": True}

    # ------------------------------------------------------------- sessions (history)
    @app.get("/api/sessions")
    def sessions() -> dict:
        rows = []
        for s in store.sessions(50):
            u = store.usage(s["id"])
            rows.append({"id": s["id"], "topic": s["topic"], "status": s["status"],
                         "created_at": s["created_at"], "cost_usd": u["cost_usd"],
                         "parent_id": s.get("parent_id")})
        return {"sessions": rows, "suggestions": store.suggestions()}

    @app.get("/api/sessions/{sid}")
    def session_detail(sid: str) -> dict:
        s = store.session(sid)
        if not s:
            raise HTTPException(404, "Unknown session")
        report = ""
        if s["report_path"] and Path(s["report_path"]).exists():
            report = Path(s["report_path"]).read_text()
        return {"session": s, "events": store.events(sid), "messages": store.messages(sid),
                "sources": store.sources(sid), "usage": store.usage(sid),
                "usage_by_purpose": store.usage_by_purpose(sid), "report": report}

    @app.post("/api/sessions/{sid}/resume")
    def resume(sid: str, body: ResumeRun) -> dict:
        s = store.session(sid)
        if not s:
            raise HTTPException(404, "Unknown session")
        if any(r.sid == sid and not r.finished for r in runs.values()):
            raise HTTPException(409, "Session is already running")
        run = Run(s["topic"], "graph", body.offline)
        runs[run.id] = run
        launch(run, "resume", sid, Budget())
        return run.summary()

    # ------------------------------------------------------------- system prompts
    @app.get("/api/prompts")
    def get_prompts() -> dict:
        """Every editable prompt with its default, so the UI can show and reset it."""
        edits = store.prompts()
        return {"prompts": [{"name": name, "default": prompt_defaults[name],
                             "text": edits.get(name, prompt_defaults[name]),
                             "custom": name in edits} for name in EDITABLE]}

    @app.post("/api/prompts")
    def set_prompt(body: NewPrompt) -> dict:
        text = body.text.strip()
        if not text or text == prompt_defaults[body.name]:
            store.clear_prompt(body.name)     # back to default rather than a stored copy
        else:
            store.set_prompt(body.name, text)
        return get_prompts()

    @app.delete("/api/prompts/{name}")
    def reset_prompt(name: str) -> dict:
        if name not in EDITABLE:
            raise HTTPException(422, f"Not an editable prompt: {name}")
        store.clear_prompt(name)
        return get_prompts()

    # ------------------------------------------------------------- source rules
    @app.get("/api/sources")
    def source_rules() -> dict:
        return {"rules": store.source_rules()}

    @app.post("/api/sources")
    def set_source_rule(body: NewSourceRule) -> dict:
        domain = clean_domain(body.domain)
        if "." not in domain:
            raise HTTPException(422, "Not a domain")
        store.set_source_rule(domain, body.rule)
        return {"rules": store.source_rules()}

    @app.delete("/api/sources/{domain}")
    def remove_source_rule(domain: str) -> dict:
        if not store.remove_source_rule(clean_domain(domain)):
            raise HTTPException(404, "No rule for that domain")
        return {"rules": store.source_rules()}

    # ------------------------------------------------------------- profile
    @app.get("/api/profile")
    def profile() -> dict:
        return {"preferences": store.preferences()}

    @app.post("/api/profile")
    def add_preference(body: NewPreference) -> dict:
        return {"added": store.add_preference(body.category, body.text),
                "preferences": store.preferences()}

    @app.delete("/api/profile/{pref_id}")
    def remove_preference(pref_id: int) -> dict:
        if not store.remove_preference(pref_id):
            raise HTTPException(404, "Unknown preference")
        return {"preferences": store.preferences()}

    @app.delete("/api/profile")
    def clear_profile() -> dict:
        return {"removed": store.clear_preferences(), "preferences": []}

    @app.delete("/api/sessions/{sid}")
    def forget(sid: str) -> dict:
        store.delete_session(sid)
        return {"ok": True}

    return app


def serve(db: str, home: Path, host: str = "127.0.0.1", port: int = 8000,
          backend: Backend | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(Store(db), home, backend=backend), host=host, port=port,
                log_level="warning")
