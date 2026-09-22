"""Domain models.

Two families live here:

* **LLM-facing schemas** (``*Draft``, ``Clarification``, ``Reflection`` ...) are sent to
  the model as JSON Schema for structured output. Every field is required and extra
  keys are forbidden so the schema is valid for strict structured outputs.
* **Runtime state** (``SubTask``, ``Plan``, ``Finding`` ...) adds bookkeeping fields the
  orchestrator owns (status, ids) that the model never writes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- LLM-facing


class Clarification(Strict):
    needs_clarification: bool
    reasoning: str = Field(description="Why the topic is or isn't clear enough to plan.")
    questions: list[str] = Field(description="At most 3 short questions. Empty if clear.")


class SubTaskDraft(Strict):
    question: str = Field(description="One focused research question.")
    rationale: str = Field(description="Why this question matters for the objective.")
    search_queries: list[str] = Field(description="2-4 concrete web search queries.")
    depends_on: list[int] = Field(
        description="1-based indexes of earlier sub-tasks that must finish first. Usually empty."
    )


class PlanDraft(Strict):
    objective: str = Field(description="One sentence: what a good final report answers.")
    recency_days: int = Field(
        description="Max source age in days. 0 means age does not matter (e.g. history)."
    )
    subtasks: list[SubTaskDraft] = Field(description="3-6 sub-tasks, most important first.")


class Credibility(Strict):
    level: Literal["high", "medium", "low"]
    reason: str


class SourceDraft(Strict):
    url: str
    title: str
    published: str = Field(description="ISO date (YYYY-MM-DD) if known, else 'unknown'.")
    publisher: str
    summary: str = Field(description="2-3 sentence neutral summary of what this source says.")
    key_takeaways: list[str] = Field(description="1-3 takeaways relevant to the question.")
    credibility: Credibility
    relevance: Literal["high", "medium", "low"]


class ClaimDraft(Strict):
    text: str = Field(description="A single factual claim.")
    source_urls: list[str] = Field(description="URLs (from sources) that support the claim.")


class FindingDraft(Strict):
    answer: str = Field(description="Direct answer to the sub-task question, 1-2 paragraphs.")
    sources: list[SourceDraft]
    claims: list[ClaimDraft]
    gaps: list[str] = Field(description="What could not be found or stayed ambiguous.")
    confidence: Literal["high", "medium", "low"]


class Reflection(Strict):
    sufficient: bool = Field(description="True if findings answer the objective well enough.")
    reasoning: str
    new_subtasks: list[SubTaskDraft] = Field(description="Follow-ups to fill gaps. Max 3.")
    questions_for_user: list[str] = Field(
        description="Only when a gap can't be resolved by searching (user intent, scope). Max 2."
    )


class Contradiction(Strict):
    topic: str
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    assessment: str = Field(description="Which side is better supported and why, or 'unresolved'.")


class Analysis(Strict):
    consensus: list[str] = Field(description="Claims multiple independent sources agree on.")
    contradictions: list[Contradiction]


class ReportDraft(Strict):
    title: str
    executive_summary: str = Field(description="3-5 sentences.")
    key_takeaways: list[str] = Field(description="3-7 bullets, each ending with [n] citations.")
    body_markdown: str = Field(
        description="Report body in Markdown with numbered [n] citations matching the source list."
    )
    open_questions: list[str]
    related_topics: list[str] = Field(description="3-5 follow-up research topics for this user.")


# --------------------------------------------------------------------------- runtime

TaskStatus = Literal["pending", "running", "done", "skipped", "failed"]


class SubTask(BaseModel):
    id: str
    question: str
    rationale: str
    search_queries: list[str]
    depends_on: list[str] = []
    status: TaskStatus = "pending"
    origin: Literal["planner", "reflection", "user"] = "planner"


class Plan(BaseModel):
    topic: str
    objective: str
    recency_days: int
    subtasks: list[SubTask]

    def get(self, task_id: str) -> SubTask | None:
        return next((t for t in self.subtasks if t.id == task_id), None)

    def next_id(self) -> str:
        # Max existing id, not len(): ids must stay unique after the user drops a task.
        nums = [int(t.id[1:]) for t in self.subtasks if t.id[1:].isdigit()]
        return f"t{max(nums, default=0) + 1}"

    def add(self, draft: SubTaskDraft, origin: str) -> SubTask:
        task = SubTask(
            id=self.next_id(),
            question=draft.question,
            rationale=draft.rationale,
            search_queries=draft.search_queries,
            origin=origin,  # type: ignore[arg-type]
        )
        self.subtasks.append(task)
        return task

    def ready(self) -> list[SubTask]:
        """Pending tasks whose dependencies are all resolved (done/skipped/failed)."""
        resolved = {t.id for t in self.subtasks if t.status in ("done", "skipped", "failed")}
        return [
            t for t in self.subtasks
            if t.status == "pending" and all(d in resolved for d in t.depends_on)
        ]

    @classmethod
    def from_draft(cls, topic: str, draft: PlanDraft) -> Plan:
        tasks: list[SubTask] = []
        for i, d in enumerate(draft.subtasks, start=1):
            deps = [f"t{j}" for j in d.depends_on if 0 < j < i]
            tasks.append(SubTask(
                id=f"t{i}", question=d.question, rationale=d.rationale,
                search_queries=d.search_queries, depends_on=deps,
            ))
        return cls(topic=topic, objective=draft.objective,
                   recency_days=max(0, draft.recency_days), subtasks=tasks)


class SearchHit(BaseModel):
    """A raw web_search result as returned by the search tool, before LLM curation."""
    url: str
    title: str
    page_age: str | None = None


class Finding(BaseModel):
    task_id: str
    question: str
    answer: str
    sources: list[SourceDraft]
    claims: list[ClaimDraft]
    gaps: list[str]
    confidence: str
    dropped: list[tuple[str, str]] = []  # (url, reason) removed by source filters


class Report(BaseModel):
    session_id: str
    draft: ReportDraft
    sources: list[SourceDraft]
    analysis: Analysis

    def to_markdown(self) -> str:
        d = self.draft
        lines = [f"# {d.title}", "", "## Executive summary", "", d.executive_summary, "",
                 "## Key takeaways", ""]
        lines += [f"- {t}" for t in d.key_takeaways]
        lines += ["", d.body_markdown, ""]
        if self.analysis.contradictions:
            lines += ["## Contradictions between sources", ""]
            for c in self.analysis.contradictions:
                lines.append(f"- **{c.topic}** — “{c.claim_a}” ({c.source_a}) vs "
                             f"“{c.claim_b}” ({c.source_b}). _{c.assessment}_")
            lines.append("")
        if d.open_questions:
            lines += ["## Open questions", ""] + [f"- {q}" for q in d.open_questions] + [""]
        if d.related_topics:
            lines += ["## Suggested next research", ""] + [f"- {t}" for t in d.related_topics] + [""]
        lines += ["## Sources", ""]
        for i, s in enumerate(self.sources, start=1):
            lines.append(f"{i}. [{s.title}]({s.url}) — {s.publisher}, {s.published} "
                         f"(credibility: {s.credibility.level})")
        return "\n".join(lines) + "\n"
