"""Deterministic stand-in for the LLM: powers unit tests and the offline demo (`--offline`).

Responses are produced per schema type. Tests can override any of them by passing
``handlers={SchemaType: callable(prompt) -> instance}``.
"""

from __future__ import annotations

import re
import threading
from typing import Callable, TypeVar

from pydantic import BaseModel

from .llm import Usage, UsageSink
from .models import (Analysis, ClaimDraft, ClaimVerdictDraft, Clarification, Contradiction,
                     Credibility, FindingDraft, HoaxJudgement, PlanDraft, Preference,
                     ProfileUpdate, Reflection, ReportDraft, SearchHit, SourceDraft, Step,
                     SubTaskDraft, VerificationDraft)

T = TypeVar("T", bound=BaseModel)


def _topic(prompt: str) -> str:
    m = re.search(r"Topic: (.+)", prompt)
    return m.group(1).strip() if m else "the topic"


def default_clarify(prompt: str) -> Clarification:
    vague = len(_topic(prompt).split()) < 3 and "Q:" not in prompt
    return Clarification(
        needs_clarification=vague,
        reasoning="Topic is very short; scope and audience are unclear." if vague else "Clear.",
        questions=["What time window matters most?", "Who is the audience?"] if vague else [],
    )


def default_plan(prompt: str) -> PlanDraft:
    t = _topic(prompt)
    angles = ["current state and recent developments", "key evidence and data",
              "criticism and opposing views"]
    return PlanDraft(
        objective=f"Give a balanced, current overview of {t}.",
        recency_days=365,
        subtasks=[SubTaskDraft(question=f"What is the {a} of {t}?", rationale=f"Covers {a}.",
                               search_queries=[f"{t} {a}"], depends_on=[]) for a in angles],
    )


def default_finding(prompt: str, n: int) -> FindingDraft:
    q = re.search(r"Sub-task question: (.+)", prompt)
    question = q.group(1) if q else "question"
    sources = [
        SourceDraft(url=f"https://example.org/report-{n}", title=f"Primary report {n}",
                    published="2026-06-01", publisher="Example Institute",
                    summary=f"Primary data relevant to: {question}",
                    key_takeaways=["Figures rose year over year."],
                    credibility=Credibility(level="high", reason="Primary data"),
                    relevance="high"),
        SourceDraft(url=f"https://news.example.com/story-{n}", title=f"News analysis {n}",
                    published="2026-08-15", publisher="Example News",
                    summary="Reports a smaller increase than the primary data.",
                    key_takeaways=["Growth is slowing."],
                    credibility=Credibility(level="medium", reason="Secondary reporting"),
                    relevance="high"),
        SourceDraft(url=f"https://old.example.net/archive-{n}", title="Archived post",
                    published="2019-01-01", publisher="Old Blog",
                    summary="Outdated commentary.", key_takeaways=["n/a"],
                    credibility=Credibility(level="low", reason="Unsourced blog"),
                    relevance="medium"),
    ]
    return FindingDraft(
        answer=f"Synthesized answer to: {question}",
        sources=sources,
        claims=[ClaimDraft(text="Figures rose year over year.", source_urls=[sources[0].url]),
                ClaimDraft(text="Growth is slowing.", source_urls=[sources[1].url])],
        gaps=["No independent replication found."] if n == 1 else [],
        confidence="medium",
    )


def default_reflect(prompt: str) -> Reflection:
    return Reflection(sufficient=True, reasoning="All sub-tasks answered with recent sources.",
                      new_subtasks=[], questions_for_user=[])


def default_analysis(prompt: str) -> Analysis:
    return Analysis(
        consensus=["Figures rose year over year."],
        contradictions=[Contradiction(topic="Growth rate", claim_a="Strong growth",
                                      source_a="Primary report 1", claim_b="Growth is slowing",
                                      source_b="News analysis 1",
                                      assessment="Primary data is better supported.")],
    )


def default_profile(prompt: str) -> ProfileUpdate:
    """Every answer and note becomes an 'other' preference (the real model is choosier)."""
    said = re.findall(r"(?:User answered|User note): (.+)", prompt)
    return ProfileUpdate(reasoning="offline demo", remove_ids=[],
                         add=[Preference(category="other", text=f"Prefers: {s.strip()}")
                              for s in said])


PRIMARY_PAGE = (
    "Annual data release. Figures rose year over year in every region we track, driven by "
    "adoption in mid-sized organisations. The survey ran from January to March and collected "
    "responses from 1,200 organisations across 14 countries, weighted by sector and size. "
    "Growth was strongest in the first half of the year before flattening in the autumn. "
    "Methodology, sampling frame, weighting and caveats follow in the appendix.")


def default_verification(prompt: str) -> VerificationDraft:
    """Supports each claim with the first sentence of its evidence (a real verbatim quote),
    attributed to the source the evidence block was labelled with."""
    checks = []
    for m in re.finditer(r"Claim (\d+): .*?\nEvidence \(source: (\S+?)\):\n<<<\n(.*?)\n>>>",
                         prompt, re.S):
        quote = m.group(3).split(". ")[0][:120]
        checks.append(ClaimVerdictDraft(claim_number=int(m.group(1)), verdict="supported",
                                        quote=quote, quote_source_url=m.group(2),
                                        note="Evidence states it."))
    return VerificationDraft(checks=checks)


def default_hoax_judgement(prompt: str) -> HoaxJudgement:
    return HoaxJudgement(asserted=False, reasoning="offline demo")


def default_report(prompt: str) -> ReportDraft:
    t = _topic(prompt)
    return ReportDraft(
        title=f"Research brief: {t}",
        executive_summary=f"An offline demo brief on {t}. Primary data shows growth; "
                          "secondary coverage suggests it is slowing.",
        key_takeaways=["Figures rose year over year [1].", "Growth may be slowing [2]."],
        body_markdown="## Findings\n\nPrimary data [1] and news analysis [2] disagree on pace.",
        open_questions=["Is the slowdown seasonal?"],
        related_topics=[f"{t} regulation", f"{t} market outlook", f"History of {t}"],
    )


class FakeLLM:
    def __init__(self, usage_sink: UsageSink | None = None,
                 handlers: dict[type, Callable[[str], BaseModel]] | None = None,
                 search=None):
        self.usage_sink = usage_sink or (lambda u: None)
        self.search = search      # the real adapters carry one; code reads it off either
        self.handlers = {Clarification: default_clarify, PlanDraft: default_plan,
                         Reflection: default_reflect, Analysis: default_analysis,
                         ReportDraft: default_report, ProfileUpdate: default_profile,
                         VerificationDraft: default_verification,
                         HoaxJudgement: default_hoax_judgement, **(handlers or {})}
        self.calls: list[tuple[str, str]] = []  # (purpose, prompt)
        self._n = 0
        self._lock = threading.Lock()

    def _record(self, purpose: str, prompt: str, searches: int = 0) -> None:
        with self._lock:
            self.calls.append((purpose, prompt))
        self.usage_sink(Usage(purpose=purpose, model="offline-fake",
                              input_tokens=len(prompt) // 4, output_tokens=200,
                              web_searches=searches, stop_reason="end_turn"))

    def structured(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                   effort: str = "high") -> T:
        self._record(purpose, prompt)
        return self.handlers[schema](prompt)  # type: ignore[return-value]

    def research(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                 max_searches: int = 8, recency_days: int = 0,
                 on_step=None, seen=()) -> tuple[T, list[SearchHit]]:
        self._record(purpose, prompt, searches=2)
        with self._lock:
            self._n += 1
            n = self._n
        handler = self.handlers.get(schema)
        finding = handler(prompt) if handler else default_finding(prompt, n)
        hits = [SearchHit(url=s.url, title=s.title, page_age=s.published) for s in finding.sources]
        if on_step:  # the offline demo shows the same live progress a real sub-agent reports
            on_step(Step(kind="search", detail=f"{purpose} query", results=len(hits)))
            for hit in hits[:1]:
                on_step(Step(kind="fetch", detail=hit.url, results=1))
        if finding.sources:  # the sub-agent "fetched" its primary source: evidence to verify
            hits.append(SearchHit(url=finding.sources[0].url, title=finding.sources[0].title,
                                  text=PRIMARY_PAGE))
        return finding, hits  # type: ignore[return-value]
