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
from urllib.parse import urlsplit
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from . import context as ctx
from . import continuity, filters, verify
from .filters import SourcePolicy
from .moderation import Blocked, ModerationGate, Moderator, recovery_hint
from . import prompts
from .control import Command as UserCommand
from .control import Control, Event, Interaction, step_event
from .llm import LLM, AgentRefusal, AuthError, LLMError
from .models import (Analysis, Clarification, Credibility, Finding, FindingDraft, Plan,
                     PlanDraft, Reflection, Report, ReportDraft, SearchHit, SourceDraft, SubTask,
                     SubTaskDraft)
from .orchestrator import MAX_SEED_LINKS, SEED_EXCERPT, Budget, _nothing_found
from .search import clip
from .store import Store
from .tracing import NullTracer, Tracer


def _merge(current: dict | None, update: dict | None) -> dict:
    return {**(current or {}), **(update or {})}


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
    parent: str                                      # session this one follows up, if any
    previous_block: str                              # earlier findings shown to clarify/plan
    previous_tasks: list[dict]                       # earlier tasks, prepended to the plan
    evidence: Annotated[dict[str, str], _merge]      # url -> fetched page text (verification)
    policy_drops: Annotated[int, operator.add]       # sources removed by the user's rules


class ResearchGraph:
    def __init__(self, llm: LLM, store: Store, ui: Interaction, *, worker_llm: LLM | None = None, checkpoint_path: str | Path,
                 budget: Budget | None = None, reports_dir: Path | str = "reports",
                 today: date | None = None, blocked_domains: tuple[str, ...] = (),
                 use_profile: bool = True, source_policy: SourcePolicy | None = None,
                 moderator: Moderator | None = None,
                 prompt_set: prompts.PromptSet | None = None,
                 tracer: Tracer | None = None):
        self.llm = llm                      # planning, reflection, analysis, writing, verifying
        self.worker = worker_llm or llm     # research sub-agents: most calls, most tokens
        self.store = store
        self.ui = ui
        self.budget = budget or Budget()
        self.reports_dir = Path(reports_dir)
        self.today = today or date.today()
        self.blocked_domains = blocked_domains
        self.prompts = prompt_set or prompts.PromptSet()   # editable per run; disclosed
        self.tracer = tracer or NullTracer()   # optional Langfuse tracing
        self.use_profile = use_profile
        self.source_policy = source_policy   # None: the user's saved rules, read when needed
        self.moderation = ModerationGate(moderator, self._emit)
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
        g.add_node("verify", self.verify)
        g.add_node("analyze", self.analyze)
        g.add_node("write", self.write)

        g.add_edge(START, "recall")
        g.add_edge("recall", "clarify")
        g.add_conditional_edges("clarify", self.after_clarify, ["ask_user", "plan"])
        g.add_conditional_edges("ask_user", self.after_ask, ["plan", "dispatch", "verify"])
        g.add_edge("plan", "review")
        g.add_conditional_edges("dispatch", self.fan_out, ["research", "reflect", "verify", END])
        g.add_edge("research", "collect")
        g.add_edge("collect", "dispatch")
        g.add_conditional_edges("reflect", self.after_reflect, ["ask_user", "dispatch", "verify"])
        g.add_edge("verify", "analyze")
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
        return "verify" if s.get("done_researching") else "dispatch"

    @staticmethod
    def after_reflect(s: ResearchState) -> str:
        if s.get("pending_questions"):
            return "ask_user"
        return "verify" if s.get("done_researching") else "dispatch"

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
            return "verify"
        return "reflect"

    # ================================================================== nodes
    def recall(self, s: ResearchState) -> dict:
        """Load what this session starts from: profile, earlier thread (follow-up), memory."""
        update: dict[str, Any] = {"rounds": 0, "findings": {}, "seen_urls": [],
                                  "status": "running", "context": [], "previous_block": "",
                                  "previous_tasks": []}
        if self.use_profile and (prefs := self.store.preferences()):
            update["context"] += continuity.profile_context(prefs)
            self._emit("profile.loaded", f"Using {len(prefs)} standing preference(s): "
                       + "; ".join(p["text"] for p in prefs), count=len(prefs))
        if parent := s.get("parent"):
            prev = continuity.load_previous(self.store, parent)
            continuity.adopt_previous(self.store, self.sid, prev)
            update["findings"] = {k: f.model_dump() for k, f in prev.findings.items()}
            update["seen_urls"] = sorted(prev.seen_urls)
            update["context"] += prev.context + [prev.note()]
            update["previous_block"] = prev.block()
            update["previous_tasks"] = [t.model_dump() for t in prev.tasks]
            self._emit("followup.loaded",
                       f"Continuing “{prev.topic}” ({prev.session_id}): "
                       f"{len(prev.findings)} earlier finding(s), "
                       f"{len(prev.seen_urls)} source(s) carried over",
                       parent=prev.session_id, findings=len(prev.findings))
        prior = [p for p in self.store.recall(s["topic"], exclude=self.sid)
                 if p["session_id"] != parent]
        if prior:
            self._emit("memory.recalled", f"Found {len(prior)} related past session(s): "
                       + "; ".join(p["topic"] for p in prior),
                       sessions=[p["session_id"] for p in prior])
        update["memory"] = prior
        if line := self._policy().describe():
            update["context"] += [line]
            self._emit("policy.loaded", line)
        seeds, seen, evidence = self._seed_links(s["topic"], set(update["seen_urls"]))
        if seeds:
            update["context"] += seeds
            update["seen_urls"] = sorted(seen)
            update["evidence"] = evidence
        return update

    def _seed_links(self, topic: str, seen: set[str]) -> tuple[list[str], set[str], dict[str, str]]:
        """Read links the user put in the topic. Same rules as the loop engine: see
        ``Orchestrator._seed_links`` for why a pasted URL is filtered before it is fetched."""
        links = filters.urls_in(topic)[:MAX_SEED_LINKS]
        if not links:
            return [], seen, {}
        drafts = [SourceDraft(url=u, title=u, published="unknown", publisher=urlsplit(u).netloc,
                              summary="", key_takeaways=[],
                              credibility=Credibility(level="medium",
                                                      reason="chosen by the user, not the model"),
                              relevance="high")
                  for u in links]
        kept, dropped = filters.filter_sources(
            drafts, recency_days=0, today=self.today, seen_urls=seen,
            blocked_domains=self.blocked_domains, policy=self._policy())
        for url, reason in dropped:
            self.store.add_source(self.sid, "seed", url, kept=False, reason=reason)
            self._emit("seed.dropped", f"Not reading {url} — {reason}", url=url, reason=reason)

        fetch = verify.page_fetcher(getattr(self.worker, "search", None)
                                    or getattr(self.llm, "search", None))
        if kept and fetch is None:
            self._emit("seed.deferred",
                       f"{len(kept)} link(s) from the topic will be read by the sub-agents")
            return [], seen, {}
        lines, evidence = [], {}
        for draft in kept:
            text = fetch(draft.url)
            if not text:
                self.store.add_source(self.sid, "seed", draft.url, kept=False, reason="unreadable")
                self._emit("seed.unreadable", f"Could not read {draft.url}", url=draft.url)
                continue
            evidence[filters.normalize_url(draft.url)] = text
            self.store.add_source(self.sid, "seed", draft.url, title=draft.title,
                                  credibility=draft.credibility.level, kept=True,
                                  reason="from the topic")
            lines.append(f"The user linked {draft.url}. Its text begins:\n"
                         f"{clip(text, SEED_EXCERPT)}")
            self._emit("seed.fetched", f"Read {draft.url} (linked in the topic)", url=draft.url)
        return lines, seen, evidence

    def clarify(self, s: ResearchState) -> dict:
        self._emit("clarify.started", "Checking whether the topic needs clarification")
        c = self.llm.structured(purpose="clarify", system=prompts.CLARIFIER, schema=Clarification,
                                prompt=ctx.topic_block(s["topic"], self.today, s.get("memory", []),
                                                       s.get("context", []),
                                                       s.get("previous_block", "")), effort="low")
        if not c.needs_clarification or not c.questions:
            self._emit("clarify.skipped", f"Topic is clear enough: {c.reasoning}")
            return {"pending_questions": []}
        self._emit("clarify.asking", f"Asking {len(c.questions[:3])} clarifying question(s)",
                   reasoning=c.reasoning)
        return {"pending_questions": c.questions[:3]}

    def ask_user(self, s: ResearchState) -> dict:
        questions = s["pending_questions"]
        # Nothing before interrupt(): this node re-runs from the top when resumed.
        answers: list[str] = interrupt({"kind": "questions", "questions": questions})
        added = []
        for q, a in zip(questions, answers):
            self.store.add_message(self.sid, "agent", "clarifying_question", q)
            if a.strip() and self.moderation.input_blocked(a, "answer"):
                continue
            if a.strip():
                self.store.add_message(self.sid, "user", "answer", a.strip())
                added.append(f"Q: {q}\nA: {a.strip()}")
                self._emit("user.answered", f"User answered: {q} → {a.strip()}")
            else:
                self._emit("user.skipped", f"User skipped: {q} (agent will use its judgment)")
        return {"context": added, "pending_questions": []}

    def plan(self, s: ResearchState) -> dict:
        self._emit("plan.started", "Decomposing topic into sub-tasks")
        draft = self.llm.structured(purpose="plan", system=self.prompts.planner, schema=PlanDraft,
                                    prompt=ctx.topic_block(s["topic"], self.today,
                                                           s.get("memory", []),
                                                           s.get("context", []),
                                                           s.get("previous_block", "")))
        plan = Plan.from_draft(s["topic"], draft)
        plan.subtasks = plan.subtasks[: self.budget.max_tasks]
        for t in plan.subtasks:
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
        plan.subtasks = [SubTask.model_validate(t) for t in s.get("previous_tasks", [])] \
            + plan.subtasks
        recency = f"sources ≤ {plan.recency_days} days old" if plan.recency_days else "any age"
        self._emit("plan.created", f"Plan: {continuity.describe_tasks(plan)}, {recency}",
                   objective=plan.objective, tasks=[t.model_dump() for t in plan.subtasks])
        return {"plan": plan.model_dump()}

    def review(self, s: ResearchState) -> Command:
        # Answer is {"approved": bool, "plan": dict}. Never resume with a bare None:
        # LangGraph treats Command(resume=None) as "no resume value" and errors.
        decision = interrupt({"kind": "plan", "plan": s["plan"]})
        if not decision.get("approved"):
            self.store.update_session(self.sid, status="aborted")
            # An abort closes whatever card the run is held on, so a rejection here may be
            # an abandoned plan rather than a judged one. Say which in the log.
            if self.control.aborting and self.control.claim_abort_notice():
                self._emit("control.aborted", "Aborted by user")
            self._emit("session.aborted", "Aborted: plan rejected")
            return Command(goto=END, update={"status": "aborted"})
        plan = Plan.model_validate(decision["plan"])
        self.store.skip_tasks_absent_from(self.sid, [t.id for t in plan.subtasks])
        for t in [t for t in plan.subtasks if t.origin == "user"]:
            if self.moderation.input_blocked(t.question, "added task"):
                plan.subtasks.remove(t)
        for t in plan.subtasks:
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
        self._emit("plan.approved", f"Plan approved: {continuity.describe_tasks(plan)}")
        self.store.update_session(self.sid, plan_json=plan.model_dump_json())
        return Command(goto="dispatch", update={"plan": decision["plan"]})

    def _aborted(self, plan: Plan) -> dict:
        """End the run now, without waiting for a pause to be answered."""
        self.store.update_session(self.sid, status="aborted")
        if self.control.claim_abort_notice():
            self._emit("control.aborted", "Aborted by user")
        self._emit("session.aborted", "Aborted: stopped by user")
        return {"status": "aborted", "stop": True, "wave": [], "plan": plan.model_dump()}

    def dispatch(self, s: ResearchState) -> dict:
        plan = Plan.model_validate(s["plan"])
        update: dict[str, Any] = {}

        self.control.hold_here(self._emit)   # plain Pause: held here until Resume
        if self.control.aborting:
            return self._aborted(plan)

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
            t.attempts += 1
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
            self._emit("task.started", f"[{t.id}] Researching: {t.question}", task=t.id)
        if not wave and s.get("rounds", 0) >= self.budget.max_rounds:
            self._emit("loop.budget", f"Reflection budget reached ({s.get('rounds', 0)} rounds)")
        update.update({"plan": plan.model_dump(), "wave": [t.id for t in wave]})
        return update

    def research(self, payload: dict) -> dict:
        """A research sub-agent. Receives only its own slice of state (via Send)."""
        task = SubTask.model_validate(payload["task"])
        # A model call in flight cannot be interrupted, but the next one need not start.
        self.control.hold_here(self._emit)
        if self.control.aborting:
            return {"raw": []}
        plan = Plan.model_validate(payload["plan"])
        deps = [Finding.model_validate(d) for d in payload["deps"]]
        prompt = ctx.research_prompt(plan, task, self.today, payload["context"], deps)
        try:
            draft, hits = self.worker.research(purpose=f"research:{task.id}",
                                               system=self.prompts.researcher, prompt=prompt,
                                               schema=FindingDraft,
                                               max_searches=self.budget.max_searches,
                                               recency_days=plan.recency_days,
                                               on_step=self._step_reporter(task.id))
        except AuthError:
            raise   # credentials or billing: every other call will fail too
        except (LLMError, AgentRefusal) as e:
            return {"raw": [{"task_id": task.id, "error": str(e)}]}
        return {"raw": [{"task_id": task.id, "draft": draft.model_dump(),
                         "hits": [h.model_dump() for h in hits]}]}

    def collect(self, s: ResearchState) -> dict:
        plan = Plan.model_validate(s["plan"])
        findings = dict(s.get("findings", {}))
        seen = set(s.get("seen_urls", []))
        evidence: dict[str, str] = {}
        policy_drops = 0
        for item in s.get("raw", []):
            task = plan.get(item["task_id"])
            if task is None or task.status != "running":
                continue
            if "error" in item:
                self._task_failed(task, item["error"])
                continue
            draft = FindingDraft.model_validate(item["draft"])
            hits = [SearchHit.model_validate(h) for h in item["hits"]]
            if not draft.sources:   # nothing retrieved: a retry may do better (see orchestrator)
                self._task_failed(task, "retrieved no sources ("
                                  + ("; ".join(draft.gaps[:1]) or "nothing came back") + ")")
                continue
            finding = ctx.curate(task, draft, hits, recency_days=plan.recency_days,
                                 today=self.today, seen_urls=seen,
                                 blocked_domains=self.blocked_domains, policy=self._policy())
            evidence.update(verify.evidence_from_hits(hits))
            policy_drops += sum(1 for _, r in finding.dropped
                                if "your source rules" in r or "your allowlist" in r)
            if finding.relaxed_recency:
                self._emit("filter.relaxed",
                           f"[{task.id}] Every source was the newer-than rule's only casualty, "
                           f"so the {plan.recency_days}-day window was set aside for this "
                           f"sub-task", task=task.id, recency_days=plan.recency_days)
            for url, reason in finding.dropped:
                self.store.add_source(self.sid, task.id, url, kept=False, reason=reason)
                self._emit("source.dropped", f"[{task.id}] Dropped {url} — {reason}",
                           task=task.id)
            for src in finding.sources:
                self.store.add_source(self.sid, task.id, src.url, title=src.title,
                                      published=src.published, credibility=src.credibility.level,
                                      kept=True, data=src.model_dump_json())
            task.status = "done"
            findings[task.id] = finding.model_dump()
            self.store.upsert_task(self.sid, task.id, task.question, task.status, task.origin,
                                   finding.model_dump_json())
            self._emit("task.done", f"[{task.id}] Done: {len(finding.sources)} sources kept, "
                                    f"{len(finding.dropped)} dropped, confidence {draft.confidence}",
                       task=task.id, searched=len(hits))
        self.store.update_session(self.sid, plan_json=plan.model_dump_json())
        return {"raw": None, "plan": plan.model_dump(), "findings": findings,
                "seen_urls": sorted(seen), "wave": [], "evidence": evidence,
                "policy_drops": policy_drops}

    def verify(self, s: ResearchState) -> dict:
        """Check claims against the text of their cited pages; label corroboration."""
        findings = self._findings(s)
        if not findings:
            return {}
        if self.budget.verify_claims > 0:
            self._emit("verify.started", "Checking claims against their cited pages")
        else:  # still label corroboration (pure code) and mark every claim unchecked
            self._emit("verify.off", "Claim verification is off; claims are marked unchecked")
        search = getattr(self.worker, "search", None) or getattr(self.llm, "search", None)
        fetch = verify.page_fetcher(search)
        evidence = dict(s.get("evidence", {}))
        result = verify.verify_findings(findings, llm=self.llm, evidence=evidence, fetch=fetch,
                                        verifier=self.prompts.verifier,
                                        max_claims=self.budget.verify_claims, emit=self._emit)
        for f in findings:
            self.store.update_finding(self.sid, f.task_id, f.model_dump_json())
        c = result.counts
        self._emit("verify.done",
                   f"{c.get('supported', 0)} supported, {c.get('partially_supported', 0)} partly, "
                   f"{c.get('unsupported', 0)} unsupported, {c.get('unverifiable', 0)} unverifiable, "
                   f"{c.get('unchecked', 0)} unchecked", counts=c, fetched=result.fetched)
        return {"findings": {f.task_id: f.model_dump() for f in findings}, "evidence": evidence}

    def reflect(self, s: ResearchState) -> dict:
        plan = Plan.model_validate(s["plan"])
        round_no = s.get("rounds", 0) + 1
        self._emit("reflect.started", f"Reviewing findings (round {round_no})")
        r = self.llm.structured(purpose="reflect", system=prompts.CRITIC, schema=Reflection,
                                prompt=ctx.progress_block(plan, self._findings(s),
                                                          s.get("context", [])))
        self._emit("reflect.done", ("Sufficient. " if r.sufficient else "Gaps found. ") + r.reasoning,
                   new_tasks=len(r.new_subtasks), questions=len(r.questions_for_user))
        room = self.budget.max_tasks - continuity.new_task_count(plan)
        if r.new_subtasks and room <= 0:
            self._emit("loop.budget", f"Task cap ({self.budget.max_tasks}) reached; not adding more")
        added = self._add_tasks(plan, r.new_subtasks[: max(0, room)], origin="reflection")
        questions = r.questions_for_user[:2]
        done = not added and not (questions and not r.sufficient)
        return {"plan": plan.model_dump(), "rounds": round_no, "pending_questions": questions,
                "done_researching": done}

    def analyze(self, s: ResearchState) -> dict:
        plan = Plan.model_validate(s["plan"])
        if not self._findings(s):
            # See Orchestrator._analyze: asked to compare nothing, the model obliges from
            # memory.
            self._emit("analyze.skipped", "Nothing was retrieved, so there is nothing to compare")
            return {"report": {"analysis": Analysis(consensus=[], contradictions=[]).model_dump()}}
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
        if findings:
            self._emit("report.started", "Writing the final report to the Result tab")
            prompt = (ctx.findings_block(plan, findings, s.get("context", []))
                      + "\n\nAnalysis:\n" + analysis.model_dump_json(indent=1))
            draft = self.llm.structured(purpose="report", system=self.prompts.writer,
                                        prompt=prompt, schema=ReportDraft)
        else:
            self._emit("report.empty",
                       "Nothing was retrieved; reporting that instead of writing a report")
            draft = _nothing_found(plan)
        sources = ctx.all_sources(findings)
        checks = [c for f in findings for c in f.checks]
        quality = verify.check_report(draft, sources, checks, s.get("policy_drops", 0),
                                      self.prompts.customised)
        report = Report(session_id=self.sid, draft=draft, sources=sources, analysis=analysis,
                        checks=checks, quality=quality)
        self._emit("report.checked",
                   f"{len(quality.invalid_citations)} invalid citation(s) fixed, "
                   f"{len(quality.uncited_statements)} uncited statement(s), "
                   f"{len(quality.weak_takeaways)} weakly sourced takeaway(s)",
                   invalid=quality.invalid_citations)
        try:
            self.moderation.report(report, plan.topic)
        except Blocked as e:
            self.store.update_session(self.sid, status="blocked")
            self._emit("session.blocked", f"Stopped by content moderation: {e}"
                       + recovery_hint(e.stage, self.sid))
            return {"status": "blocked"}
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", plan.topic.lower()).strip("-")[:50] or "report"
        path = self.reports_dir / f"{self.sid}-{slug}.md"
        path.write_text(report.to_markdown())
        self.store.add_message(self.sid, "agent", "report", draft.executive_summary)
        self.store.update_session(self.sid, status="done", summary=draft.executive_summary,
                                  related_topics=draft.related_topics, report_path=str(path))
        self.store.remember(self.sid, plan.topic, draft.executive_summary, draft.key_takeaways)
        self._learn_profile(plan.topic)
        usage = self.store.usage(self.sid)
        self._emit("session.done", f"Report saved to {path} · {usage['calls']} LLM calls · "
                                   f"{usage['input_tokens'] + usage['output_tokens']:,} tokens · "
                                   f"${usage['cost_usd']:.2f}", path=str(path))
        return {"status": "done", "report": {**s["report"], "full": report.model_dump()}}

    # ================================================================== helpers
    def _task_failed(self, task: SubTask, reason: str) -> None:
        """Retry a sub-task that retrieved nothing; a retry needs no new task slot."""
        if task.attempts <= self.budget.max_retries:
            task.status = "pending"
            self.store.upsert_task(self.sid, task.id, task.question, task.status, task.origin)
            self._emit("task.retry", f"[{task.id}] {reason} — retrying "
                       f"({task.attempts}/{self.budget.max_retries + 1})", task=task.id)
            return
        task.status = "failed"
        self.store.upsert_task(self.sid, task.id, task.question, task.status, task.origin)
        self._emit("task.failed", f"[{task.id}] Failed: {reason}", task=task.id)

    def _policy(self) -> SourcePolicy:
        return self.source_policy or SourcePolicy.from_rules(self.store.source_rules())

    def _learn_profile(self, topic: str) -> None:
        if not self.use_profile:
            return
        try:
            added, removed = continuity.learn_profile(self.llm, self.store, self.sid, topic)
        except (LLMError, AgentRefusal) as e:
            self._emit("profile.skipped", f"Profile not updated: {e}")
            return
        if added or removed:
            self._emit("profile.updated", "Profile updated"
                       + (f" · learned: {'; '.join(added)}" if added else "")
                       + (f" · dropped: {'; '.join(removed)}" if removed else ""),
                       added=added, removed=removed)

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
            self._emit("task.added", f"[{t.id}] Added ({origin}): {t.question}", task=t.id)
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
                self._emit("override.skip", f"User skipped [{t.id}] {t.question}",
                           task=t.id)
            elif cmd.action in ("add", "note") and cmd.arg and \
                    self.moderation.input_blocked(cmd.arg, cmd.action):
                continue
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

    def _step_reporter(self, task_id: str):
        """Report a sub-agent's searches as they happen. Nodes fanned out with Send run in
        their own threads; ``_emit`` only touches the (locked) store and the UI."""
        def report(step) -> None:
            type_, message, data = step_event(task_id, step)
            self._emit(type_, message, **data)
        return report

    def _announce_prompts(self) -> None:
        """Say up front which prompts were rewritten: the log is the record of what produced
        this report, and a custom verifier or writer changes what its numbers mean."""
        custom = self.prompts.customised
        self.store.record_session_prompts(
            self.sid, {name: getattr(self.prompts, name) for name in custom})
        if custom:
            self._emit("prompt.custom", "Custom system prompt(s) in use: "
                       + ", ".join(custom) + " — noted in the report",
                       prompts=list(custom))

    def _emit(self, type_: str, message: str, **data) -> None:
        if self.sid:
            self.store.add_event(self.sid, type_, message, data or None)
        self.tracer.event(type_, message, data)
        if type_.startswith("session.") and type_ not in ("session.started", "session.resumed"):
            self.tracer.flush()   # the run is over: send the tail before the process moves on
        self.ui.on_event(Event(type=type_, message=message, data=data))

    # ================================================================== driving the graph
    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.sid},
                "max_concurrency": self.budget.max_parallel}

    def run(self, topic: str, parent: str | None = None) -> Report | None:
        if parent and not self.store.session(parent):
            raise ValueError(f"Unknown session {parent}")
        self.sid = self.store.create_session(topic, parent_id=parent)
        self.store.add_message(self.sid, "user", "topic", topic)
        self.tracer.session(self.sid, topic)
        self._emit("session.started", f"Session {self.sid}: “{topic}” (LangGraph engine)")
        self._announce_prompts()
        try:
            self.moderation.request(topic)
        except Blocked as e:
            self.store.update_session(self.sid, status="blocked")
            self._emit("session.blocked", f"Stopped by content moderation: {e}"
                       + recovery_hint(e.stage, self.sid))
            return None
        state: dict[str, Any] = {"session_id": self.sid, "topic": topic}
        if parent:
            state["parent"] = parent
        return self._drive(state)

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
