"""Context engineering shared by both engines (hand-rolled orchestrator and LangGraph).

Pure functions: given run state, build the text each LLM role sees, and curate a
worker's raw findings into a filtered ``Finding``. No I/O, no LLM calls.
"""

from __future__ import annotations

from datetime import date

from .filters import SourcePolicy, fill_dates_from_hits, filter_sources, normalize_url
from .verify import claim_label
from .models import Finding, FindingDraft, Plan, SearchHit, SourceDraft, SubTask


def context_block(context: list[str]) -> str:
    return ("User clarifications and guidance:\n" + "\n".join(context) + "\n") if context else ""


def topic_block(topic: str, today: date, memory: list[dict], context: list[str],
                previous: str = "") -> str:
    parts = [f"Topic: {topic}", f"Today's date: {today.isoformat()}"]
    if previous:
        parts.append(previous)
    if memory:
        parts.append("Prior research by this user:\n" + "\n".join(
            f"- {m['topic']} ({m['created_at'][:10]}): {m['summary']}" for m in memory))
    if context:
        parts.append(context_block(context))
    return "\n\n".join(parts)


def recency_text(plan: Plan) -> str:
    return f"prefer sources from the last {plan.recency_days} days" if plan.recency_days \
        else "age does not matter"


def research_prompt(plan: Plan, task: SubTask, today: date, context: list[str],
                    dependencies: list[Finding]) -> str:
    deps = "".join(f"\nEarlier finding ({f.question}): {f.answer}\n" for f in dependencies)
    return (
        f"Overall objective: {plan.objective}\n"
        f"Today's date: {today.isoformat()}\n"
        f"Recency requirement: {recency_text(plan)}\n"
        f"{context_block(context)}"
        f"Sub-task question: {task.question}\n"
        f"Why it matters: {task.rationale}\n"
        f"Starting queries: {'; '.join(task.search_queries)}\n"
        f"{deps}"
    )


def progress_block(plan: Plan, findings: list[Finding], context: list[str]) -> str:
    lines = [f"Objective: {plan.objective}", context_block(context), "Plan status:"]
    lines += [f"- [{t.id}] ({t.status}) {t.question}" for t in plan.subtasks]
    lines.append("\nFindings so far:")
    for f in findings:
        lines.append(f"\n[{f.task_id}] {f.question}\nConfidence: {f.confidence}; "
                     f"{len(f.sources)} sources\n{f.answer}\nGaps: {'; '.join(f.gaps) or 'none'}")
    return "\n".join(lines)


def all_sources(findings: list[Finding]) -> list[SourceDraft]:
    out, seen = [], set()
    for f in findings:
        for s in f.sources:
            if (k := normalize_url(s.url)) not in seen:
                seen.add(k)
                out.append(s)
    return out


def findings_block(plan: Plan, findings: list[Finding], context: list[str]) -> str:
    sources = all_sources(findings)
    index = {normalize_url(s.url): i for i, s in enumerate(sources, start=1)}
    lines = [f"Topic: {plan.topic}", f"Objective: {plan.objective}", context_block(context),
             "Sources:"]
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] {s.title} — {s.publisher}, {s.published}, credibility "
                     f"{s.credibility.level}: {s.summary}")
    lines.append("\nFindings:")
    failed: list[str] = []
    for f in findings:
        lines.append(f"\n## {f.question} (confidence {f.confidence})\n{f.answer}")
        checks = f.checks or [None] * len(f.claims)
        for c, check in zip(f.claims, checks):
            refs = sorted({index[k] for u in c.source_urls if (k := normalize_url(u)) in index})
            cite = "".join(f"[{r}]" for r in refs)
            if check is not None and check.verdict == "unsupported":
                failed.append(f"- {c.text} {cite} ({check.note})")
                continue
            if refs:
                label = claim_label(check) if check is not None else ""
                lines.append(f"- {c.text} {cite}" + (f" ({label})" if label else ""))
        if f.gaps:
            lines.append("Gaps: " + "; ".join(f.gaps))
    if failed:
        lines.append("\nClaims that FAILED verification (their cited pages don't support them; "
                     "do not present as fact):")
        lines += failed
    return "\n".join(lines)


def curate(task: SubTask, draft: FindingDraft, hits: list[SearchHit], *, recency_days: int,
           today: date, seen_urls: set[str], blocked_domains: tuple[str, ...] = (),
           policy: SourcePolicy | None = None) -> Finding:
    """Apply source filters to a worker's raw output. Mutates ``seen_urls``."""
    fill_dates_from_hits(draft.sources, hits)
    kept, dropped = filter_sources(draft.sources, recency_days=recency_days, today=today,
                                   seen_urls=seen_urls, blocked_domains=blocked_domains,
                                   policy=policy)

    # The window is a guess the planner makes before any source has been seen, and a wrong
    # guess is otherwise unrecoverable: a live run on walking and type 2 diabetes set 730 days
    # and threw away peer-reviewed 2024 papers from PubMed and the BJSM, leaving one sub-task
    # with nothing at all - while a 2023 PDF survived because its date would not parse, so
    # declaring a date was the thing being punished. When age is the only thing that removed
    # every source, the window was wrong for this question rather than the sources being bad.
    relaxed = False
    if recency_days and not kept and dropped and all(r.startswith("outdated") for _, r in dropped):
        relaxed = True
        kept, dropped = filter_sources(draft.sources, recency_days=0, today=today,
                                       seen_urls=seen_urls, blocked_domains=blocked_domains,
                                       policy=policy)

    return Finding(task_id=task.id, question=task.question, answer=draft.answer, sources=kept,
                   claims=draft.claims, gaps=draft.gaps, confidence=draft.confidence,
                   dropped=dropped, relaxed_recency=relaxed)
