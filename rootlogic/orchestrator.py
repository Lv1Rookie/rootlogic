"""Research orchestrator: clarify -> plan -> (execute waves <-> reflect) -> analyze -> write.

Control flow is plain Python so it can be read top-to-bottom and unit-tested with a fake
LLM. The LLM decides *what* to research (plan, follow-ups, questions); this module decides
*how* the run proceeds (ordering, parallelism, budgets, filtering, human checkpoints).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import context as ctx
from . import prompts
from .control import Command, Control, Event, Interaction
from .llm import LLM, AgentRefusal, LLMError
from .models import (Analysis, Clarification, Finding, FindingDraft, Plan, PlanDraft, Reflection,
                     Report, ReportDraft, SubTask, SubTaskDraft)
from .store import Store


class Aborted(Exception):
    pass


@dataclass
class Budget:
    max_rounds: int = 2        # reflection rounds after the initial plan
    max_tasks: int = 10        # hard cap on sub-tasks per session
    max_parallel: int = 4      # concurrent research sub-agents
    max_searches: int = 5      # web searches per sub-agent


class Orchestrator:
    def __init__(self, llm: LLM, store: Store, ui: Interaction, *, budget: Budget | None = None,
                 reports_dir: Path | str = "reports", today: date | None = None,
                 blocked_domains: tuple[str, ...] = ()):
        self.llm = llm
        self.store = store
        self.ui = ui
        self.budget = budget or Budget()
        self.reports_dir = Path(reports_dir)
        self.today = today or date.today()
        self.blocked_domains = blocked_domains
        self.control = Control()
        # per-session state
        self.sid = ""
        self.context: list[str] = []      # clarifications + user notes, fed to every prompt
        self.findings: dict[str, Finding] = {}
        self.seen_urls: set[str] = set()
        self._stop = False

    # ================================================================== public API
    def run(self, topic: str) -> Report | None:
        self.sid = self.store.create_session(topic)
        self.store.add_message(self.sid, "user", "topic", topic)
        self._emit("session.started", f"Session {self.sid}: “{topic}”")
        try:
            memory = self._recall(topic)
            self._clarify(topic, memory)
            plan = self._plan(topic, memory)
            reviewed = self.ui.review_plan(plan)
            if reviewed is None:
                raise Aborted("plan rejected")
            plan = reviewed
            self._emit("plan.approved", f"Plan approved with {len(plan.subtasks)} sub-tasks")
            self._save_plan(plan)
            self._research_loop(plan)
            analysis = self._analyze(plan)
            report = self._write(plan, analysis)
        except Aborted as e:
            self.store.update_session(self.sid, status="aborted")
            self._emit("session.aborted", f"Aborted: {e}")
            return None
        except (LLMError, AgentRefusal) as e:
            self.store.update_session(self.sid, status="failed")
            self._emit("session.failed", f"Failed: {e}")
            raise
        return report

    # ================================================================== stages
    def _recall(self, topic: str) -> list[dict]:
        prior = self.store.recall(topic, exclude=self.sid)
        if prior:
            self._emit("memory.recalled",
                       f"Found {len(prior)} related past session(s): "
                       + "; ".join(p["topic"] for p in prior),
                       sessions=[p["session_id"] for p in prior])
        return prior

    def _clarify(self, topic: str, memory: list[dict]) -> None:
        self._emit("clarify.started", "Checking whether the topic needs clarification")
        c = self.llm.structured(purpose="clarify", system=prompts.CLARIFIER,
                                prompt=self._topic_block(topic, memory), schema=Clarification,
                                effort="low")
        if not c.needs_clarification or not c.questions:
            self._emit("clarify.skipped", f"Topic is clear enough: {c.reasoning}")
            return
        self._emit("clarify.asking", f"Asking {len(c.questions)} clarifying question(s)",
                   reasoning=c.reasoning)
        self._ask_user(c.questions[:3])

    def _plan(self, topic: str, memory: list[dict]) -> Plan:
        self._emit("plan.started", "Decomposing topic into sub-tasks")
        draft = self.llm.structured(purpose="plan", system=prompts.PLANNER,
                                    prompt=self._topic_block(topic, memory), schema=PlanDraft)
        plan = Plan.from_draft(topic, draft)
        plan.subtasks = plan.subtasks[: self.budget.max_tasks]
        for t in plan.subtasks:
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
        recency = f"sources ≤ {plan.recency_days} days old" if plan.recency_days else "any age"
        self._emit("plan.created", f"Plan: {len(plan.subtasks)} sub-tasks, {recency}",
                   objective=plan.objective, tasks=[t.model_dump() for t in plan.subtasks])
        return plan

    def _research_loop(self, plan: Plan) -> None:
        rounds = 0
        while True:
            self._checkpoint(plan)
            if self._stop:
                break
            ready = plan.ready()
            if ready:
                self._run_wave(plan, ready)
                continue
            if rounds >= self.budget.max_rounds:
                self._emit("loop.budget", f"Reflection budget reached ({rounds} rounds)")
                break
            rounds += 1
            if not self._reflect(plan, rounds):
                break

    def _run_wave(self, plan: Plan, tasks: list[SubTask]) -> None:
        for t in tasks:
            t.status = "running"
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
            self._emit("task.started", f"[{t.id}] Researching: {t.question}", task=t.id)

        with ThreadPoolExecutor(max_workers=self.budget.max_parallel) as pool:
            futures = {pool.submit(self._research_task, plan, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    result = fut.result()
                except (LLMError, AgentRefusal) as e:
                    t.status = "failed"
                    self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
                    self._emit("task.failed", f"[{t.id}] Failed: {e}", task=t.id)
                    continue
                self._accept_finding(plan, t, result)
        self._save_plan(plan)

    def _research_task(self, plan: Plan, task: SubTask) -> tuple[FindingDraft, list]:
        """Runs in a worker thread: only calls the LLM, never the UI or shared state."""
        deps = [self.findings[d] for d in task.depends_on if d in self.findings]
        prompt = ctx.research_prompt(plan, task, self.today, self.context, deps)
        return self.llm.research(purpose=f"research:{task.id}", system=prompts.RESEARCHER,
                                 prompt=prompt, schema=FindingDraft,
                                 max_searches=self.budget.max_searches)

    def _accept_finding(self, plan: Plan, task: SubTask, result: tuple[FindingDraft, list]) -> None:
        draft, hits = result
        finding = ctx.curate(task, draft, hits, recency_days=plan.recency_days, today=self.today,
                             seen_urls=self.seen_urls, blocked_domains=self.blocked_domains)
        kept, dropped = finding.sources, finding.dropped
        for url, reason in dropped:
            self.store.add_source(self.sid, task.id, url, kept=False, reason=reason)
            self._emit("source.dropped", f"[{task.id}] Dropped {url} — {reason}", task=task.id)
        for s in kept:
            self.store.add_source(self.sid, task.id, s.url, title=s.title, published=s.published,
                                  credibility=s.credibility.level, kept=True,
                                  data=s.model_dump_json())
        self.findings[task.id] = finding
        task.status = "done"
        self.store.upsert_task(self.sid, task.id, task.question, task.status, task.origin,
                               finding.model_dump_json())
        self._emit("task.done",
                   f"[{task.id}] Done: {len(kept)} sources kept, {len(dropped)} dropped, "
                   f"confidence {draft.confidence}"
                   + (f"; gaps: {'; '.join(draft.gaps[:2])}" if draft.gaps else ""),
                   task=task.id, searched=len(hits))

    def _reflect(self, plan: Plan, round_no: int) -> bool:
        """Critic step. Returns True if the loop should continue."""
        self._emit("reflect.started", f"Reviewing findings (round {round_no})")
        r = self.llm.structured(purpose="reflect", system=prompts.CRITIC,
                                prompt=self._progress_block(plan), schema=Reflection)
        self._emit("reflect.done",
                   ("Sufficient. " if r.sufficient else "Gaps found. ") + r.reasoning,
                   new_tasks=len(r.new_subtasks), questions=len(r.questions_for_user))
        if r.questions_for_user:
            self._ask_user(r.questions_for_user[:2])

        room = self.budget.max_tasks - len(plan.subtasks)
        added = self._add_tasks(plan, r.new_subtasks[: max(0, room)], origin="reflection")
        if r.new_subtasks and room <= 0:
            self._emit("loop.budget", f"Task cap ({self.budget.max_tasks}) reached; not adding more")
        if added:
            return True
        # Nothing new to research: continue only if the user just gave us new information.
        return bool(r.questions_for_user) and not r.sufficient

    def _analyze(self, plan: Plan) -> Analysis:
        self._emit("analyze.started", "Cross-checking sources for consensus and contradictions")
        analysis = self.llm.structured(purpose="analyze", system=prompts.ANALYST,
                                       prompt=self._findings_block(plan), schema=Analysis)
        self._emit("analyze.done", f"{len(analysis.consensus)} consensus point(s), "
                                   f"{len(analysis.contradictions)} contradiction(s)")
        return analysis

    def _write(self, plan: Plan, analysis: Analysis) -> Report:
        self._emit("report.started", "Writing report")
        sources = self._all_sources()
        prompt = (self._findings_block(plan)
                  + "\n\nAnalysis:\n" + analysis.model_dump_json(indent=1))
        draft = self.llm.structured(purpose="report", system=prompts.WRITER, prompt=prompt,
                                    schema=ReportDraft)
        report = Report(session_id=self.sid, draft=draft, sources=sources, analysis=analysis)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{self.sid}-{_slug(plan.topic)}.md"
        path.write_text(report.to_markdown())
        self.store.add_message(self.sid, "agent", "report", draft.executive_summary)
        self.store.update_session(self.sid, status="done", summary=draft.executive_summary,
                                  related_topics=draft.related_topics, report_path=str(path))
        self.store.remember(self.sid, plan.topic, draft.executive_summary, draft.key_takeaways)
        usage = self.store.usage(self.sid)
        self._emit("session.done",
                   f"Report saved to {path} · {usage['calls']} LLM calls · "
                   f"{usage['input_tokens'] + usage['output_tokens']:,} tokens · "
                   f"${usage['cost_usd']:.2f}", path=str(path))
        return report

    # ================================================================== human in the loop
    def _ask_user(self, questions: list[str]) -> None:
        for q in questions:
            self.store.add_message(self.sid, "agent", "clarifying_question", q)
            answer = self.ui.ask(q).strip()
            if answer:
                self.store.add_message(self.sid, "user", "answer", answer)
                self.context.append(f"Q: {q}\nA: {answer}")
                self._emit("user.answered", f"User answered: {q} → {answer}")
            else:
                self._emit("user.skipped", f"User skipped: {q} (agent will use its judgment)")

    def _checkpoint(self, plan: Plan) -> None:
        if not self.control.consume_pause():
            return
        self._emit("control.paused", "Paused by user")
        for cmd in self.ui.override(plan):
            self._apply(plan, cmd)
        self._save_plan(plan)
        self._emit("control.resumed", "Resumed")

    def _apply(self, plan: Plan, cmd: Command) -> None:
        if cmd.action == "abort":
            raise Aborted("stopped by user")
        if cmd.action == "stop":
            self._stop = True
            for t in plan.subtasks:
                if t.status == "pending":
                    t.status = "skipped"
                    self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
            self._emit("override.stop", "User asked to stop researching and write the report")
        elif cmd.action == "skip":
            t = plan.get(cmd.arg)
            if t and t.status == "pending":
                t.status = "skipped"
                self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
                self._emit("override.skip", f"User skipped [{t.id}] {t.question}")
        elif cmd.action == "add":
            draft = SubTaskDraft(question=cmd.arg, rationale="Requested by user",
                                 search_queries=[cmd.arg], depends_on=[])
            self._add_tasks(plan, [draft], origin="user")
        elif cmd.action == "note":
            self.context.append(f"User guidance: {cmd.arg}")
            self.store.add_message(self.sid, "user", "note", cmd.arg)
            self._emit("override.note", f"User guidance added: {cmd.arg}")

    def _add_tasks(self, plan: Plan, drafts: list[SubTaskDraft], origin: str) -> list[SubTask]:
        existing = {t.question.strip().lower() for t in plan.subtasks}
        added = []
        for d in drafts:
            if d.question.strip().lower() in existing:
                continue
            t = plan.add(d, origin)
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
            self._emit("task.added", f"[{t.id}] Added ({origin}): {t.question}", task=t.id)
            added.append(t)
        return added

    # ================================================================== prompt blocks
    def _topic_block(self, topic: str, memory: list[dict]) -> str:
        return ctx.topic_block(topic, self.today, memory, self.context)

    def _progress_block(self, plan: Plan) -> str:
        return ctx.progress_block(plan, list(self.findings.values()), self.context)

    def _findings_block(self, plan: Plan) -> str:
        return ctx.findings_block(plan, list(self.findings.values()), self.context)

    def _all_sources(self):
        return ctx.all_sources(list(self.findings.values()))

    # ================================================================== bookkeeping
    def _emit(self, type_: str, message: str, **data) -> None:
        if self.sid:
            self.store.add_event(self.sid, type_, message, data or None)
        self.ui.on_event(Event(type=type_, message=message, data=data))

    def _save_plan(self, plan: Plan) -> None:
        self.store.update_session(self.sid, plan_json=plan.model_dump_json())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "report"
