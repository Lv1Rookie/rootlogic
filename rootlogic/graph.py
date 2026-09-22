"""The same research agent, expressed as a LangGraph state machine.

Compare with ``orchestrator.py``: the stages, prompts, schemas, filters and storage are
shared; only the control flow moves from a Python ``while`` loop into a graph.

What LangGraph adds here
  * **Checkpointing** - state is saved after every step (SqliteSaver), so a run that crashes,
    hits a network error, or is closed mid-way continues with ``rootlogic resume <id>``.
  * **interrupt()** - every human decision (clarifying answers, plan approval, mid-run
    overrides) is a persisted pause, not a blocking ``input()`` call. The graph can wait
    for hours, and a web UI could answer instead of a terminal.
  * **Send** - parallel fan-out of research sub-agents, merged by a state reducer.

What it costs
  * An interrupted node re-runs *from its start* on resume, so nothing with side effects
    (LLM calls, DB writes, log events) may happen before ``interrupt()`` in that node.
    That's why "decide the question" (clarify) and "ask the question" (ask_user) are
    separate nodes.
  * State must be serializable, so it holds plain dicts; nodes rehydrate Pydantic models.

Graph:
    START → recall → clarify ─┬─────────────→ plan → review ─→ dispatch ⇄ research → collect
                              └→ ask_user ─┘                      │    ↑                │
                                    ↑                             ↓    └────────────────┘
                                    └──────────────────────── reflect → analyze → write → END
"""

from __future__ import annotations

import operator
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from . import context as ctx
from . import prompts
from .control import Command as UserCommand
from .control import Control, Event, Interaction
from .llm import LLM, AgentRefusal, LLMError
from .models import (Analysis, Clarification, Finding, FindingDraft, Plan, PlanDraft, Reflection,
                     Report, ReportDraft, SearchHit, SubTask, SubTaskDraft)
from .orchestrator import Budget
from .store import Store


def _raw_reducer(current: list[dict] | None, update: list[dict] | None) -> list[dict]:
    """Worker results accumulate in parallel; ``collect`` clears them by writing None."""
    if update is None:
        return []
    return (current or []) + update


class ResearchState(TypedDict, total=False):
    session_id: str
    topic: str
    memory: list[dict]
    context: Annotated[list[str], operator.add]      # clarifications + user guidance
    pending_questions: list[str]
    plan: dict                                       # Plan.model_dump()
    wave: list[str]                                  # task ids dispatched this step
    raw: Annotated[list[dict], _raw_reducer]         # worker outputs awaiting curation
    findings: dict[str, dict]                        # task_id -> Finding.model_dump()
    seen_urls: list[str]
    rounds: int
    stop: bool
    done_researching: bool
    status: str                                      # running | done | aborted
    report: dict


class ResearchGraph:
    def __init__(self, llm: LLM, store: Store, ui: Interaction, *, checkpoint_path: str | Path,
                 budget: Budget | None = None, reports_dir: Path | str = "reports",
                 today: date | None = None, blocked_domains: tuple[str, ...] = ()):
        self.llm = llm
        self.store = store
        self.ui = ui
        self.budget = budget or Budget()
        self.reports_dir = Path(reports_dir)
        self.today = today or date.today()
        self.blocked_domains = blocked_domains
        self.control = Control()
        self.sid = ""
        if str(checkpoint_path) != ":memory:":
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        self.app = self._build().compile(checkpointer=SqliteSaver(conn))

    # ================================================================== graph wiring
    def _build(self) -> StateGraph:
        g = StateGraph(ResearchState)
        g.add_node("recall", self.recall)
        g.add_node("clarify", self.clarify)
        g.add_node("ask_user", self.ask_user)
        g.add_node("plan", self.plan)
        g.add_node("review", self.review, destinations=("dispatch", END))
        g.add_node("dispatch", self.dispatch)
        g.add_node("research", self.research)
        g.add_node("collect", self.collect)
        g.add_node("reflect", self.reflect)
        g.add_node("analyze", self.analyze)
        g.add_node("write", self.write)

        g.add_edge(START, "recall")
        g.add_edge("recall", "clarify")
        g.add_conditional_edges("clarify", self.after_clarify, ["ask_user", "plan"])
        g.add_conditional_edges("ask_user", self.after_ask, ["plan", "dispatch", "analyze"])
        g.add_edge("plan", "review")
        g.add_conditional_edges("dispatch", self.fan_out, ["research", "reflect", "analyze", END])
        g.add_edge("research", "collect")
        g.add_edge("collect", "dispatch")
        g.add_conditional_edges("reflect", self.after_reflect, ["ask_user", "dispatch", "analyze"])
        g.add_edge("analyze", "write")
        g.add_edge("write", END)
        return g

    # ================================================================== routers
    @staticmethod
    def after_clarify(s: ResearchState) -> str:
        return "ask_user" if s.get("pending_questions") else "plan"

    @staticmethod
    def after_ask(s: ResearchState) -> str:
        if not s.get("plan"):
            return "plan"
        return "analyze" if s.get("done_researching") else "dispatch"

    @staticmethod
    def after_reflect(s: ResearchState) -> str:
        if s.get("pending_questions"):
            return "ask_user"
        return "analyze" if s.get("done_researching") else "dispatch"

    def fan_out(self, s: ResearchState) -> list[Send] | str:
        if s.get("status") == "aborted":
            return END
        plan = Plan.model_validate(s["plan"])
        if s.get("wave"):
            findings = s.get("findings", {})
            return [Send("research", {
                "task": plan.get(tid).model_dump(),
                "plan": s["plan"],
                "context": s.get("context", []),
                "deps": [findings[d] for d in plan.get(tid).depends_on if d in findings],
            }) for tid in s["wave"]]
        if s.get("stop") or s.get("rounds", 0) >= self.budget.max_rounds:
            return "analyze"
        return "reflect"

    # ================================================================== nodes
    def recall(self, s: ResearchState) -> dict:
        prior = self.store.recall(s["topic"], exclude=self.sid)
        if prior:
            self._emit("memory.recalled", f"Found {len(prior)} related past session(s): "
                       + "; ".join(p["topic"] for p in prior))
        return {"memory": prior, "rounds": 0, "findings": {}, "seen_urls": [], "status": "running"}

    def clarify(self, s: ResearchState) -> dict:
        self._emit("clarify.started", "Checking whether the topic needs clarification")
        c = self.llm.structured(purpose="clarify", system=prompts.CLARIFIER, schema=Clarification,
                                prompt=ctx.topic_block(s["topic"], self.today, s.get("memory", []),
                                                       s.get("context", [])), effort="low")
        if not c.needs_clarification or not c.questions:
            self._emit("clarify.skipped", f"Topic is clear enough: {c.reasoning}")
            return {"pending_questions": []}
        self._emit("clarify.asking", f"Asking {len(c.questions[:3])} clarifying question(s)")
        return {"pending_questions": c.questions[:3]}

    def ask_user(self, s: ResearchState) -> dict:
        questions = s["pending_questions"]
        # Nothing before interrupt(): this node re-runs from the top when resumed.
        answers: list[str] = interrupt({"kind": "questions", "questions": questions})
        added = []
        for q, a in zip(questions, answers):
            self.store.add_message(self.sid, "agent", "clarifying_question", q)
            if a.strip():
                self.store.add_message(self.sid, "user", "answer", a.strip())
                added.append(f"Q: {q}\nA: {a.strip()}")
                self._emit("user.answered", f"User answered: {q} → {a.strip()}")
            else:
                self._emit("user.skipped", f"User skipped: {q} (agent will use its judgment)")
        return {"context": added, "pending_questions": []}

    def plan(self, s: ResearchState) -> dict:
        self._emit("plan.started", "Decomposing topic into sub-tasks")
        draft = self.llm.structured(purpose="plan", system=prompts.PLANNER, schema=PlanDraft,
                                    prompt=ctx.topic_block(s["topic"], self.today,
                                                           s.get("memory", []),
                                                           s.get("context", [])))
        plan = Plan.from_draft(s["topic"], draft)
        plan.subtasks = plan.subtasks[: self.budget.max_tasks]
        for t in plan.subtasks:
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
        recency = f"sources ≤ {plan.recency_days} days old" if plan.recency_days else "any age"
        self._emit("plan.created", f"Plan: {len(plan.subtasks)} sub-tasks, {recency}")
        return {"plan": plan.model_dump()}

    def review(self, s: ResearchState) -> Command:
        # Answer is {"approved": bool, "plan": dict}. Never resume with a bare None:
        # LangGraph treats Command(resume=None) as "no resume value" and errors.
        decision = interrupt({"kind": "plan", "plan": s["plan"]})
        if not decision.get("approved"):
            self.store.update_session(self.sid, status="aborted")
            self._emit("session.aborted", "Aborted: plan rejected")
            return Command(goto=END, update={"status": "aborted"})
        plan = Plan.model_validate(decision["plan"])
        for t in plan.subtasks:
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
        self._emit("plan.approved", f"Plan approved with {len(plan.subtasks)} sub-tasks")
        self.store.update_session(self.sid, plan_json=plan.model_dump_json())
        return Command(goto="dispatch", update={"plan": decision["plan"]})

    def dispatch(self, s: ResearchState) -> dict:
        plan = Plan.model_validate(s["plan"])
        update: dict[str, Any] = {}

        if self.control.pause_pending():
            commands = interrupt({"kind": "override", "plan": s["plan"]})
            self.control.clear()
            self._emit("control.paused", "Paused by user")
            update = self._apply_overrides(plan, [UserCommand(**c) for c in commands], s)
            if update.get("status") == "aborted":
                return update
            self._emit("control.resumed", "Resumed")

        if s.get("stop") or update.get("stop"):
            update.update({"plan": plan.model_dump(), "wave": []})
            return update

        wave = plan.ready()
        for t in wave:
            t.status = "running"
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
            self._emit("task.started", f"[{t.id}] Researching: {t.question}")
        if not wave and s.get("rounds", 0) >= self.budget.max_rounds:
            self._emit("loop.budget", f"Reflection budget reached ({s.get('rounds', 0)} rounds)")
        update.update({"plan": plan.model_dump(), "wave": [t.id for t in wave]})
        return update

    def research(self, payload: dict) -> dict:
        """A research sub-agent. Receives only its own slice of state (via Send)."""
        task = SubTask.model_validate(payload["task"])
        plan = Plan.model_validate(payload["plan"])
        deps = [Finding.model_validate(d) for d in payload["deps"]]
        prompt = ctx.research_prompt(plan, task, self.today, payload["context"], deps)
        try:
            draft, hits = self.llm.research(purpose=f"research:{task.id}",
                                            system=prompts.RESEARCHER, prompt=prompt,
                                            schema=FindingDraft,
                                            max_searches=self.budget.max_searches)
        except (LLMError, AgentRefusal) as e:
            return {"raw": [{"task_id": task.id, "error": str(e)}]}
        return {"raw": [{"task_id": task.id, "draft": draft.model_dump(),
                         "hits": [h.model_dump() for h in hits]}]}

    def collect(self, s: ResearchState) -> dict:
        plan = Plan.model_validate(s["plan"])
        findings = dict(s.get("findings", {}))
        seen = set(s.get("seen_urls", []))
        for item in s.get("raw", []):
            task = plan.get(item["task_id"])
            if task is None or task.status != "running":
                continue
            if "error" in item:
                task.status = "failed"
                self.store.upsert_task(self.sid, task.id, task.question, task.status, task.origin)
                self._emit("task.failed", f"[{task.id}] Failed: {item['error']}")
                continue
            draft = FindingDraft.model_validate(item["draft"])
            hits = [SearchHit.model_validate(h) for h in item["hits"]]
            finding = ctx.curate(task, draft, hits, recency_days=plan.recency_days,
                                 today=self.today, seen_urls=seen,
                                 blocked_domains=self.blocked_domains)
            for url, reason in finding.dropped:
                self.store.add_source(self.sid, task.id, url, kept=False, reason=reason)
                self._emit("source.dropped", f"[{task.id}] Dropped {url} — {reason}")
            for src in finding.sources:
                self.store.add_source(self.sid, task.id, src.url, title=src.title,
                                      published=src.published, credibility=src.credibility.level,
                                      kept=True, data=src.model_dump_json())
            task.status = "done"
            findings[task.id] = finding.model_dump()
            self.store.upsert_task(self.sid, task.id, task.question, task.status, task.origin,
                                   finding.model_dump_json())
            self._emit("task.done", f"[{task.id}] Done: {len(finding.sources)} sources kept, "
                                    f"{len(finding.dropped)} dropped, confidence {draft.confidence}")
        self.store.update_session(self.sid, plan_json=plan.model_dump_json())
        return {"raw": None, "plan": plan.model_dump(), "findings": findings,
                "seen_urls": sorted(seen), "wave": []}

    def reflect(self, s: ResearchState) -> dict:
        plan = Plan.model_validate(s["plan"])
        round_no = s.get("rounds", 0) + 1
        self._emit("reflect.started", f"Reviewing findings (round {round_no})")
        r = self.llm.structured(purpose="reflect", system=prompts.CRITIC, schema=Reflection,
                                prompt=ctx.progress_block(plan, self._findings(s),
                                                          s.get("context", [])))
        self._emit("reflect.done", ("Sufficient. " if r.sufficient else "Gaps found. ") + r.reasoning)
        room = self.budget.max_tasks - len(plan.subtasks)
        if r.new_subtasks and room <= 0:
            self._emit("loop.budget", f"Task cap ({self.budget.max_tasks}) reached; not adding more")
        added = self._add_tasks(plan, r.new_subtasks[: max(0, room)], origin="reflection")
        questions = r.questions_for_user[:2]
        done = not added and not (questions and not r.sufficient)
        return {"plan": plan.model_dump(), "rounds": round_no, "pending_questions": questions,
                "done_researching": done}

    def analyze(self, s: ResearchState) -> dict:
        plan = Plan.model_validate(s["plan"])
        self._emit("analyze.started", "Cross-checking sources for consensus and contradictions")
        analysis = self.llm.structured(purpose="analyze", system=prompts.ANALYST, schema=Analysis,
                                       prompt=ctx.findings_block(plan, self._findings(s),
                                                                 s.get("context", [])))
        self._emit("analyze.done", f"{len(analysis.consensus)} consensus point(s), "
                                   f"{len(analysis.contradictions)} contradiction(s)")
        return {"report": {"analysis": analysis.model_dump()}}

    def write(self, s: ResearchState) -> dict:
        plan = Plan.model_validate(s["plan"])
        findings = self._findings(s)
        analysis = Analysis.model_validate(s["report"]["analysis"])
        self._emit("report.started", "Writing report")
        prompt = (ctx.findings_block(plan, findings, s.get("context", []))
                  + "\n\nAnalysis:\n" + analysis.model_dump_json(indent=1))
        draft = self.llm.structured(purpose="report", system=prompts.WRITER, prompt=prompt,
                                    schema=ReportDraft)
        report = Report(session_id=self.sid, draft=draft, sources=ctx.all_sources(findings),
                        analysis=analysis)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", plan.topic.lower()).strip("-")[:50] or "report"
        path = self.reports_dir / f"{self.sid}-{slug}.md"
        path.write_text(report.to_markdown())
        self.store.add_message(self.sid, "agent", "report", draft.executive_summary)
        self.store.update_session(self.sid, status="done", summary=draft.executive_summary,
                                  related_topics=draft.related_topics, report_path=str(path))
        self.store.remember(self.sid, plan.topic, draft.executive_summary, draft.key_takeaways)
        usage = self.store.usage(self.sid)
        self._emit("session.done", f"Report saved to {path} · {usage['calls']} LLM calls · "
                                   f"{usage['input_tokens'] + usage['output_tokens']:,} tokens · "
                                   f"${usage['cost_usd']:.2f}")
        return {"status": "done", "report": {**s["report"], "full": report.model_dump()}}

    # ================================================================== helpers
    def _findings(self, s: ResearchState) -> list[Finding]:
        return [Finding.model_validate(f) for f in s.get("findings", {}).values()]

    def _add_tasks(self, plan: Plan, drafts: list[SubTaskDraft], origin: str) -> list[SubTask]:
        existing = {t.question.strip().lower() for t in plan.subtasks}
        added = []
        for d in drafts:
            if d.question.strip().lower() in existing:
                continue
            t = plan.add(d, origin)
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
            self._emit("task.added", f"[{t.id}] Added ({origin}): {t.question}")
            added.append(t)
        return added

    def _apply_overrides(self, plan: Plan, commands: list[UserCommand], s: ResearchState) -> dict:
        update: dict[str, Any] = {}
        notes: list[str] = []
        for cmd in commands:
            if cmd.action == "abort":
                self.store.update_session(self.sid, status="aborted")
                self._emit("session.aborted", "Aborted: stopped by user")
                return {"status": "aborted", "stop": True, "wave": [], "plan": plan.model_dump()}
            if cmd.action == "stop":
                for t in plan.subtasks:
                    if t.status == "pending":
                        t.status = "skipped"
                        self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
                update["stop"] = True
                self._emit("override.stop", "User asked to stop researching and write the report")
            elif cmd.action == "skip" and (t := plan.get(cmd.arg)) and t.status == "pending":
                t.status = "skipped"
                self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
                self._emit("override.skip", f"User skipped [{t.id}] {t.question}")
            elif cmd.action == "add" and cmd.arg:
                self._add_tasks(plan, [SubTaskDraft(question=cmd.arg, rationale="Requested by user",
                                                    search_queries=[cmd.arg], depends_on=[])],
                                origin="user")
            elif cmd.action == "note" and cmd.arg:
                notes.append(f"User guidance: {cmd.arg}")
                self.store.add_message(self.sid, "user", "note", cmd.arg)
                self._emit("override.note", f"User guidance added: {cmd.arg}")
        if notes:
            update["context"] = notes
        return update

    def _emit(self, type_: str, message: str) -> None:
        if self.sid:
            self.store.add_event(self.sid, type_, message)
        self.ui.on_event(Event(type=type_, message=message))

    # ================================================================== driving the graph
    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.sid},
                "max_concurrency": self.budget.max_parallel}

    def run(self, topic: str) -> Report | None:
        self.sid = self.store.create_session(topic)
        self.store.add_message(self.sid, "user", "topic", topic)
        self._emit("session.started", f"Session {self.sid}: “{topic}” (LangGraph engine)")
        return self._drive({"session_id": self.sid, "topic": topic})

    def resume(self, session_id: str) -> Report | None:
        """Continue a session from its last checkpoint (after a crash, error, or quit)."""
        self.sid = session_id
        state = self.app.get_state(self._config())
        if not state.values:
            raise ValueError(f"No checkpoint for session {session_id}")
        if state.values.get("status") in ("done", "aborted"):
            self._emit("session.resumed", f"Session already {state.values['status']}")
            return self._report(state.values)
        self.store.update_session(self.sid, status="running")
        self._emit("session.resumed", f"Resuming at: {', '.join(state.next) or 'start'}")
        if any(i.value.get("kind") == "override" for i in state.interrupts):
            self.control.request_pause()  # restore the flag the interrupted node checks
        return self._drive(None)

    def _drive(self, graph_input: Any) -> Report | None:
        config = self._config()
        try:
            while True:
                for _ in self.app.stream(graph_input, config, stream_mode="updates"):
                    pass  # nodes emit their own events
                state = self.app.get_state(config)
                if not state.interrupts:
                    return self._report(state.values)
                graph_input = Command(resume=self._answer(state.interrupts[0].value))
        except (LLMError, AgentRefusal) as e:
            self.store.update_session(self.sid, status="failed")
            self._emit("session.failed", f"Failed: {e}. Continue later with: rootlogic resume "
                                         f"{self.sid}")
            raise

    def _answer(self, request: dict) -> Any:
        """Turn an interrupt payload into a human answer via the Interaction."""
        if request["kind"] == "questions":
            return [self.ui.ask(q) for q in request["questions"]]
        plan = Plan.model_validate(request["plan"])
        if request["kind"] == "plan":
            reviewed = self.ui.review_plan(plan)
            return {"approved": reviewed is not None,
                    "plan": reviewed.model_dump() if reviewed else None}
        if request["kind"] == "override":
            return [{"action": c.action, "arg": c.arg} for c in self.ui.override(plan)]
        raise ValueError(f"unknown interrupt kind {request['kind']}")

    @staticmethod
    def _report(values: dict) -> Report | None:
        full = (values.get("report") or {}).get("full")
        return Report.model_validate(full) if full else None

    def mermaid(self) -> str:
        return self.app.get_graph().draw_mermaid()
