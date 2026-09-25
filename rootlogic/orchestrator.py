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
from urllib.parse import urlsplit

from . import context as ctx
from . import continuity, filters, verify
from .filters import SourcePolicy
from .moderation import Blocked, ModerationGate, Moderator, recovery_hint
from . import prompts
from .control import Command, Control, Event, Interaction, step_event
from .llm import LLM, AgentRefusal, AuthError, LLMError
from .models import (Analysis, Clarification, Credibility, Finding, FindingDraft, Plan,
                     PlanDraft, Reflection, Report, ReportDraft, SourceDraft, SubTask,
                     SubTaskDraft)
from .search import clip
from .store import Store
from .tracing import NullTracer, Tracer


class Aborted(Exception):
    pass




MAX_SEED_LINKS = 3      # links read from the topic; a topic is not a reading list
SEED_EXCERPT = 2000     # characters of a linked page carried into the prompts


@dataclass
class Budget:
    max_rounds: int = 2        # reflection rounds after the initial plan
    max_tasks: int = 10        # hard cap on sub-tasks per session
    max_parallel: int = 4      # concurrent research sub-agents
    max_searches: int = 8      # web searches per sub-agent
    verify_claims: int = 12    # claims checked against their cited pages (0 = no verification)
    max_retries: int = 1       # re-runs of a sub-task that found nothing (tool errors happen)


class Orchestrator:
    def __init__(self, llm: LLM, store: Store, ui: Interaction, *, worker_llm: LLM | None = None, budget: Budget | None = None,
                 reports_dir: Path | str = "reports", today: date | None = None,
                 blocked_domains: tuple[str, ...] = (), use_profile: bool = True,
                 source_policy: SourcePolicy | None = None, moderator: Moderator | None = None,
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
        self.use_profile = use_profile          # read + learn the user's standing preferences
        self.source_policy = source_policy      # None: load the user's saved source rules
        self.moderation = ModerationGate(moderator, self._emit)  # request, user input, report
        self.policy = SourcePolicy()
        self.policy_drops = 0
        self.evidence: dict[str, str] = {}      # normalized url -> fetched page text
        self.control = Control()
        # per-session state
        self.sid = ""
        self.context: list[str] = []      # clarifications + user notes, fed to every prompt
        self.findings: dict[str, Finding] = {}
        self.seen_urls: set[str] = set()
        self.previous: continuity.Previous | None = None   # set when following up a session
        self._stop = False

    # ================================================================== public API
    def run(self, topic: str, parent: str | None = None) -> Report | None:
        """Research ``topic``. With ``parent``, continue that earlier session (a follow-up)."""
        previous = continuity.load_previous(self.store, parent) if parent else None
        self.sid = self.store.create_session(topic, parent_id=parent)
        self.store.add_message(self.sid, "user", "topic", topic)
        self.tracer.session(self.sid, topic)
        self._emit("session.started", f"Session {self.sid}: “{topic}”")
        self._announce_prompts()
        try:
            self.moderation.request(topic)
            self._load_profile()
            self._load_policy()
            self._seed_links(topic)
            self._continue_from(previous)
            memory = self._recall(topic)
            self._clarify(topic, memory)
            plan = self._plan(topic, memory)
            reviewed = self.ui.review_plan(plan)
            # An abort closes whatever card the run is held on, so a plan that comes back
            # rejected may have been abandoned rather than judged. Say which in the log.
            self._abort_if_requested()
            if reviewed is None:
                raise Aborted("plan rejected")
            plan = reviewed
            self.store.skip_tasks_absent_from(self.sid, [t.id for t in plan.subtasks])
            self._screen_user_tasks(plan)  # tasks the user typed into the plan
            self._emit("plan.approved", f"Plan approved: {continuity.describe_tasks(plan)}")
            self._save_plan(plan)
            self._research_loop(plan)
            self._verify()
            analysis = self._analyze(plan)
            report = self._write(plan, analysis)
        except Aborted as e:
            self.store.update_session(self.sid, status="aborted")
            self._emit("session.aborted", f"Aborted: {e}")
            return None
        except Blocked as e:
            self.store.update_session(self.sid, status="blocked")
            self._emit("session.blocked", f"Stopped by content moderation: {e}"
                       + recovery_hint(e.stage, self.sid))
            return None
        except (LLMError, AgentRefusal) as e:
            self.store.update_session(self.sid, status="failed")
            self._emit("session.failed", f"Failed: {e}")
            raise
        return report

    # ================================================================== stages
    def _load_profile(self) -> None:
        if not self.use_profile:
            return
        prefs = self.store.preferences()
        if prefs:
            self.context.extend(continuity.profile_context(prefs))
            self._emit("profile.loaded", f"Using {len(prefs)} standing preference(s): "
                       + "; ".join(p["text"] for p in prefs), count=len(prefs))

    def _load_policy(self) -> None:
        self.policy = self.source_policy or SourcePolicy.from_rules(self.store.source_rules())
        if line := self.policy.describe():
            self.context.append(line)
            self._emit("policy.loaded", line)

    def _continue_from(self, previous: continuity.Previous | None) -> None:
        if previous is None:
            return
        self.previous = previous
        continuity.adopt_previous(self.store, self.sid, previous)
        self.findings.update(previous.findings)
        self.seen_urls |= previous.seen_urls
        self.context.extend(previous.context)
        self.context.append(previous.note())
        self._emit("followup.loaded",
                   f"Continuing “{previous.topic}” ({previous.session_id}): "
                   f"{len(previous.findings)} earlier finding(s), "
                   f"{len(previous.seen_urls)} source(s) carried over",
                   parent=previous.session_id, findings=len(previous.findings))

    def _search_provider(self):
        """Whatever the adapters can fetch pages with, or None on Claude's hosted tools."""
        return getattr(self.worker, "search", None) or getattr(self.llm, "search", None)

    def _seed_links(self, topic: str) -> None:
        """Read links the user put in the topic, and treat them as sources like any other.

        A pasted URL used to be nothing but words in a prompt: a sub-agent might fetch it or
        might not, and if it did the page arrived without passing the source rules. Here it is
        fetched once, up front, through the same filters as anything a sub-agent brings back,
        and recorded in `sources` so the report can cite it and the log can show it.

        The page is untrusted like any other web content - it goes into the same evidence store
        the verifier reads, and the researcher prompt already forbids obeying instructions found
        in fetched text.
        """
        links = filters.urls_in(topic)[:MAX_SEED_LINKS]
        if not links:
            return
        drafts = [SourceDraft(url=u, title=u, published="unknown",
                              publisher=urlsplit(u).netloc, summary="",
                              key_takeaways=[],
                              credibility=Credibility(level="medium",
                                                      reason="chosen by the user, not the model"),
                              relevance="high")
                  for u in links]
        # recency is not applied: the reader asked for this page, whatever its date
        kept, dropped = filters.filter_sources(
            drafts, recency_days=0, today=self.today, seen_urls=self.seen_urls,
            blocked_domains=self.blocked_domains, policy=self.policy)
        for url, reason in dropped:
            self.store.add_source(self.sid, "seed", url, kept=False, reason=reason)
            self._emit("seed.dropped", f"Not reading {url} — {reason}", url=url, reason=reason)

        fetch = verify.page_fetcher(self._search_provider())
        if kept and fetch is None:
            # Claude's hosted tools fetch inside the model's own turn, so the link is left for
            # the sub-agents, who can reach it; saying so beats silently doing nothing.
            self._emit("seed.deferred",
                       f"{len(kept)} link(s) from the topic will be read by the sub-agents")
            return
        for draft in kept:
            text = fetch(draft.url)
            if not text:
                self.store.add_source(self.sid, "seed", draft.url, kept=False, reason="unreadable")
                self._emit("seed.unreadable", f"Could not read {draft.url}", url=draft.url)
                continue
            self.evidence[filters.normalize_url(draft.url)] = text
            self.store.add_source(self.sid, "seed", draft.url, title=draft.title,
                                  credibility=draft.credibility.level, kept=True, reason="from the topic")
            self.context.append(f"The user linked {draft.url}. Its text begins:\n"
                                f"{clip(text, SEED_EXCERPT)}")
            self._emit("seed.fetched", f"Read {draft.url} (linked in the topic)", url=draft.url)

    def _recall(self, topic: str) -> list[dict]:
        prior = [p for p in self.store.recall(topic, exclude=self.sid)
                 if not self.previous or p["session_id"] != self.previous.session_id]
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
        draft = self.llm.structured(purpose="plan", system=self.prompts.planner,
                                    prompt=self._topic_block(topic, memory), schema=PlanDraft)
        plan = Plan.from_draft(topic, draft)
        plan.subtasks = plan.subtasks[: self.budget.max_tasks]
        for t in plan.subtasks:
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
        continuity.merge_previous(plan, self.previous)
        recency = f"sources ≤ {plan.recency_days} days old" if plan.recency_days else "any age"
        self._emit("plan.created", f"Plan: {continuity.describe_tasks(plan)}, {recency}",
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
            t.attempts += 1
            self.store.upsert_task(self.sid, t.id, t.question, t.status, t.origin)
            self._emit("task.started", f"[{t.id}] Researching: {t.question}", task=t.id)

        with ThreadPoolExecutor(max_workers=self.budget.max_parallel) as pool:
            futures = {pool.submit(self._research_task, plan, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    result = fut.result()
                except AuthError:
                    raise   # credentials or billing: every other call will fail too
                except (LLMError, AgentRefusal) as e:
                    self._task_failed(t, str(e))
                else:
                    self._accept_finding(plan, t, result)
                # A sub-task boundary is the finest safe point there is: a model call in
                # flight cannot be interrupted, but the next one need not start. A failed
                # sub-task is a boundary too: skipping the check here meant a wave of
                # failures ran its retries out before an abort was noticed.
                self._hold_if_requested()
                self._abort_if_requested()
        self._save_plan(plan)
        self._checkpoint(plan)   # a pause pressed mid-wave is answered when the wave ends

    def _task_failed(self, task: SubTask, reason: str) -> None:
        """Retry a sub-task that found nothing: tool outages are usually transient, and a
        retry shouldn't need a new task slot from the budget."""
        if task.attempts <= self.budget.max_retries:
            task.status = "pending"
            self.store.upsert_task(self.sid, task.id, task.question, task.status, task.origin)
            self._emit("task.retry", f"[{task.id}] {reason} — retrying "
                       f"({task.attempts}/{self.budget.max_retries + 1})", task=task.id)
            return
        task.status = "failed"
        self.store.upsert_task(self.sid, task.id, task.question, task.status, task.origin)
        self._emit("task.failed", f"[{task.id}] Failed: {reason}", task=task.id)

    def _research_task(self, plan: Plan, task: SubTask) -> tuple[FindingDraft, list]:
        """Runs in a worker thread: calls the LLM and reports its searches, nothing else."""
        # A pause or abort pressed before this task started is honoured now, rather than
        # spending a model call whose result will be thrown away.
        self._hold_if_requested()
        self._abort_if_requested()
        deps = [self.findings[d] for d in task.depends_on if d in self.findings]
        prompt = ctx.research_prompt(plan, task, self.today, self.context, deps)
        return self.worker.research(purpose=f"research:{task.id}", system=self.prompts.researcher,
                                    prompt=prompt, schema=FindingDraft,
                                    max_searches=self.budget.max_searches,
                                    recency_days=plan.recency_days,
                                    on_step=self._step_reporter(task.id))

    def _step_reporter(self, task_id: str):
        """Turn a sub-agent's searches into events as they happen (from its worker thread:
        the store is locked and every UI queues or prints, so this is safe)."""
        def report(step) -> None:
            type_, message, data = step_event(task_id, step)
            self._emit(type_, message, **data)
        return report

    def _accept_finding(self, plan: Plan, task: SubTask, result: tuple[FindingDraft, list]) -> None:
        draft, hits = result
        finding = ctx.curate(task, draft, hits, recency_days=plan.recency_days, today=self.today,
                             seen_urls=self.seen_urls, blocked_domains=self.blocked_domains,
                             policy=self.policy)
        kept, dropped = finding.sources, finding.dropped
        self.evidence.update(verify.evidence_from_hits(hits))
        self.policy_drops += sum(1 for _, r in dropped if "your source rules" in r
                                 or "your allowlist" in r)
        if finding.relaxed_recency:
            self._emit("filter.relaxed",
                       f"[{task.id}] Every source was newer-than rule's only casualty, so the "
                       f"{plan.recency_days}-day window was set aside for this sub-task",
                       task=task.id, recency_days=plan.recency_days)
        for url, reason in dropped:
            self.store.add_source(self.sid, task.id, url, kept=False, reason=reason)
            self._emit("source.dropped", f"[{task.id}] Dropped {url} — {reason}", task=task.id)
        for s in kept:
            self.store.add_source(self.sid, task.id, s.url, title=s.title, published=s.published,
                                  credibility=s.credibility.level, kept=True,
                                  data=s.model_dump_json())
        # Nothing retrieved at all (tool outage, dead searches) is worth retrying. Sources
        # that were retrieved and then filtered out (duplicates in a follow-up, outdated,
        # blocked) are a real result, not a failure.
        if not draft.sources:
            self._task_failed(task, "retrieved no sources ("
                              + ("; ".join(draft.gaps[:1]) or "nothing came back") + ")")
            return
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

        room = self.budget.max_tasks - continuity.new_task_count(plan)
        added = self._add_tasks(plan, r.new_subtasks[: max(0, room)], origin="reflection")
        if r.new_subtasks and room <= 0:
            self._emit("loop.budget", f"Task cap ({self.budget.max_tasks}) reached; not adding more")
        if added:
            return True
        # Nothing new to research: continue only if the user just gave us new information.
        return bool(r.questions_for_user) and not r.sufficient

    def _verify(self) -> None:
        """Check claims against the text of their cited pages; label corroboration."""
        if not self.findings:
            return
        if self.budget.verify_claims > 0:
            self._emit("verify.started", "Checking claims against their cited pages")
        else:  # still label corroboration (pure code) and mark every claim unchecked
            self._emit("verify.off", "Claim verification is off; claims are marked unchecked")
        search = getattr(self.worker, "search", None) or getattr(self.llm, "search", None)
        fetch = verify.page_fetcher(search)
        result = verify.verify_findings(list(self.findings.values()), llm=self.llm,
                                        verifier=self.prompts.verifier,
                                        evidence=self.evidence, fetch=fetch,
                                        max_claims=self.budget.verify_claims, emit=self._emit)
        for f in self.findings.values():
            self.store.update_finding(self.sid, f.task_id, f.model_dump_json())
        c = result.counts
        self._emit("verify.done",
                   f"{c.get('supported', 0)} supported, {c.get('partially_supported', 0)} partly, "
                   f"{c.get('unsupported', 0)} unsupported, {c.get('unverifiable', 0)} unverifiable, "
                   f"{c.get('unchecked', 0)} unchecked", counts=c, fetched=result.fetched)

    def _analyze(self, plan: Plan) -> Analysis:
        if not self.findings:
            # Asked to cross-check nothing, a capable model obliges from memory: live on
            # claude-sonnet-5 with search down this returned four consensus points and two
            # contradictions about a topic it had read not one page on.
            self._emit("analyze.skipped", "Nothing was retrieved, so there is nothing to compare")
            return Analysis(consensus=[], contradictions=[])
        self._emit("analyze.started", "Cross-checking sources for consensus and contradictions")
        analysis = self.llm.structured(purpose="analyze", system=prompts.ANALYST,
                                       prompt=self._findings_block(plan), schema=Analysis)
        self._emit("analyze.done", f"{len(analysis.consensus)} consensus point(s), "
                                   f"{len(analysis.contradictions)} contradiction(s)")
        return analysis

    def _write(self, plan: Plan, analysis: Analysis) -> Report:
        if self.findings:
            self._emit("report.started", "Writing the final report to the Result tab")
            sources = self._all_sources()
            prompt = (self._findings_block(plan)
                      + "\n\nAnalysis:\n" + analysis.model_dump_json(indent=1))
            draft = self.llm.structured(purpose="report", system=self.prompts.writer,
                                        prompt=prompt, schema=ReportDraft)
        else:
            # Asked for a research report with nothing to report from, a capable model writes
            # one anyway: the run that prompted this returned a fluent review of trials it had
            # never retrieved. Saying so is not the model's to write, so code writes it.
            self._emit("report.empty",
                       "Nothing was retrieved; reporting that instead of writing a report")
            sources, draft = [], _nothing_found(plan)
        checks = [c for f in self.findings.values() for c in f.checks]
        quality = verify.check_report(draft, sources, checks, self.policy_drops,
                                      self.prompts.customised)
        report = Report(session_id=self.sid, draft=draft, sources=sources, analysis=analysis,
                        checks=checks, quality=quality)
        self._emit("report.checked",
                   f"{len(quality.invalid_citations)} invalid citation(s) fixed, "
                   f"{len(quality.uncited_statements)} uncited statement(s), "
                   f"{len(quality.weak_takeaways)} weakly sourced takeaway(s)",
                   invalid=quality.invalid_citations)
        self.moderation.report(report, plan.topic)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{self.sid}-{_slug(plan.topic)}.md"
        path.write_text(report.to_markdown())
        self.store.add_message(self.sid, "agent", "report", draft.executive_summary)
        self.store.update_session(self.sid, status="done", summary=draft.executive_summary,
                                  related_topics=draft.related_topics, report_path=str(path))
        self.store.remember(self.sid, plan.topic, draft.executive_summary, draft.key_takeaways)
        self._learn_profile(plan.topic)
        usage = self.store.usage(self.sid)
        self._emit("session.done",
                   f"Report saved to {path} · {usage['calls']} LLM calls · "
                   f"{usage['input_tokens'] + usage['output_tokens']:,} tokens · "
                   f"${usage['cost_usd']:.2f}", path=str(path))
        return report

    def _screen_user_tasks(self, plan: Plan) -> None:
        for t in [t for t in plan.subtasks if t.origin == "user"]:
            if self.moderation.input_blocked(t.question, "added task"):
                plan.subtasks.remove(t)

    def _learn_profile(self, topic: str) -> None:
        """Best effort: a failed profile update never fails the finished research."""
        if not self.use_profile:
            return
        try:
            added, removed = continuity.learn_profile(self.llm, self.store, self.sid, topic)
        except (LLMError, AgentRefusal) as e:
            self._emit("profile.skipped", f"Profile not updated: {e}")
            return
        if added or removed:
            self._emit("profile.updated",
                       "Profile updated"
                       + (f" · learned: {'; '.join(added)}" if added else "")
                       + (f" · dropped: {'; '.join(removed)}" if removed else ""),
                       added=added, removed=removed)

    # ================================================================== human in the loop
    def _ask_user(self, questions: list[str]) -> None:
        for q in questions:
            self.store.add_message(self.sid, "agent", "clarifying_question", q)
            answer = self.ui.ask(q).strip()
            if answer and self.moderation.input_blocked(answer, "answer"):
                continue
            if answer:
                self.store.add_message(self.sid, "user", "answer", answer)
                self.context.append(f"Q: {q}\nA: {answer}")
                self._emit("user.answered", f"User answered: {q} → {answer}")
            else:
                self._emit("user.skipped", f"User skipped: {q} (agent will use its judgment)")

    def _hold_if_requested(self) -> None:
        """Plain Pause: stop here and stay stopped until the user presses Resume."""
        self.control.hold_here(self._emit)

    def _abort_if_requested(self) -> None:
        if self.control.aborting:
            if self.control.claim_abort_notice():
                self._emit("control.aborted", "Aborted by user")
            raise Aborted("stopped by user")

    def _checkpoint(self, plan: Plan) -> None:
        self._hold_if_requested()
        self._abort_if_requested()
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
                self._emit("override.skip", f"User skipped [{t.id}] {t.question}",
                           task=t.id)
        elif cmd.action in ("add", "note") and self.moderation.input_blocked(cmd.arg, cmd.action):
            return
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
        return ctx.topic_block(topic, self.today, memory, self.context,
                               previous=self.previous.block() if self.previous else "")

    def _progress_block(self, plan: Plan) -> str:
        return ctx.progress_block(plan, list(self.findings.values()), self.context)

    def _findings_block(self, plan: Plan) -> str:
        return ctx.findings_block(plan, list(self.findings.values()), self.context)

    def _all_sources(self):
        return ctx.all_sources(list(self.findings.values()))

    # ================================================================== bookkeeping
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
        if type_.startswith("session.") and type_ != "session.started":
            self.tracer.flush()   # the run is over: send the tail before the process moves on
        self.ui.on_event(Event(type=type_, message=message, data=data))

    def _save_plan(self, plan: Plan) -> None:
        self.store.update_session(self.sid, plan_json=plan.model_dump_json())


def _nothing_found(plan: Plan) -> ReportDraft:
    """The report for a run that retrieved nothing: what was attempted, and no claims."""
    failed = [t for t in plan.subtasks if t.status == "failed"]
    why = "; ".join(dict.fromkeys(t.error for t in failed if getattr(t, "error", ""))) \
        or "no sub-task returned usable results"
    return ReportDraft(
        title=f"No sources found: {plan.objective}",
        executive_summary=(
            "This run retrieved no sources, so it has nothing to report about the topic. "
            f"Every sub-task ended without usable results ({why}). Nothing here is a finding: "
            "it is a record of what was attempted. Check the search backend, then retry."),
        key_takeaways=[],
        body_markdown="## What was attempted\n\n" + "\n".join(
            f"- **{t.id}** — {t.question} · *{t.status}*" for t in plan.subtasks),
        open_questions=[t.question for t in plan.subtasks][:5],
        related_topics=[])


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "report"
