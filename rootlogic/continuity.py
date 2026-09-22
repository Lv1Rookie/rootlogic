"""Continuity across sessions: the user's standing profile and follow-up research threads.

Shared by both engines. Everything here is either a pure function or a small, explicit
store/LLM call, so each behaviour is testable without running a whole session.

Profile
  After a session in which the user answered questions or left notes, ``learn_profile``
  asks the model to distil *durable* preferences (audience, region, source likes/dislikes,
  time window ...) and applies them to the ``preferences`` table. ``profile_context`` turns
  the stored profile into context lines that every prompt of the next session receives.

Follow-ups
  ``load_previous`` rebuilds a finished session's findings, tasks, sources and user context
  so a new session can continue it: earlier findings count as already done (``previous``
  tasks), their sources are deduplicated, and the planner is told to plan only new work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts
from .llm import LLM
from .models import Finding, Plan, ProfileUpdate, SubTask
from .filters import normalize_url
from .store import Store

PROFILE_PREFIX = "Standing user preference"


# =================================================================== profile


def profile_context(prefs: list[dict]) -> list[str]:
    return [f"{PROFILE_PREFIX} ({p['category']}): {p['text']}" for p in prefs]


def session_conversation(store: Store, sid: str) -> list[str]:
    """This session's own user input, as lines: questions with answers, and notes."""
    lines, question = [], None
    for m in store.messages(sid):
        if m["kind"] == "clarifying_question":
            question = m["content"]
        elif m["kind"] == "answer":
            lines.append(f"- Agent asked: {question or '?'}\n  User answered: {m['content']}")
            question = None
        elif m["kind"] == "note":
            lines.append(f"- User note: {m['content']}")
    return lines


def learn_profile(llm: LLM, store: Store, sid: str, topic: str) -> tuple[list[str], list[str]]:
    """Update the profile from this session. Returns (added texts, removed texts).

    Skips the LLM call entirely when the user gave no input this session.
    """
    conversation = session_conversation(store, sid)
    if not conversation:
        return [], []
    existing = store.preferences()
    prompt = (f"Research topic this session: {topic}\n\n"
              "Current profile:\n"
              + ("\n".join(f"- id {p['id']} ({p['category']}): {p['text']}" for p in existing)
                 or "(empty)")
              + "\n\nSession conversation:\n" + "\n".join(conversation))
    update = llm.structured(purpose="profile", system=prompts.PROFILER, prompt=prompt,
                            schema=ProfileUpdate, effort="low")
    by_id = {p["id"]: p for p in existing}
    removed = [by_id[i]["text"] for i in update.remove_ids
               if i in by_id and store.remove_preference(i)]
    added = [p.text for p in update.add if p.text.strip()
             and store.add_preference(p.category, p.text, sid)]
    return added, removed


# =================================================================== follow-ups


@dataclass
class Previous:
    session_id: str
    topic: str
    objective: str
    summary: str
    tasks: list[SubTask] = field(default_factory=list)       # renamed p1.., status done
    findings: dict[str, Finding] = field(default_factory=dict)
    context: list[str] = field(default_factory=list)          # user answers/notes to carry on
    seen_urls: set[str] = field(default_factory=set)

    def block(self) -> str:
        """What the clarifier and planner see about the earlier research."""
        lines = [f"Earlier research in this thread (session {self.session_id}): {self.topic}",
                 f"Its objective: {self.objective}"]
        if self.summary:
            lines.append(f"Its summary: {self.summary}")
        lines.append("Questions already answered (do not research these again):")
        for f in self.findings.values():
            answer = f.answer if len(f.answer) <= 400 else f.answer[:400] + "…"
            lines.append(f"- [{f.task_id}] {f.question} → {answer}")
        return "\n".join(lines)

    def note(self) -> str:
        return (f"This session follows up on earlier research “{self.topic}”. Build on its "
                "findings and lead with what is new.")


def load_previous(store: Store, sid: str) -> Previous:
    """Rebuild a session so a follow-up can continue it. Raises ValueError if unknown."""
    s = store.session(sid)
    if not s:
        raise ValueError(f"Unknown session {sid}")
    objective = ""
    if s["plan_json"]:
        objective = Plan.model_validate_json(s["plan_json"]).objective
    prev = Previous(session_id=sid, topic=s["topic"], objective=objective or s["topic"],
                    summary=s["summary"] or "")

    # Findings (including ones this session itself inherited) get fresh ids p1, p2 ...
    for row in store.tasks(sid):
        if row["status"] != "done" or not row["finding_json"]:
            continue
        new_id = f"p{len(prev.tasks) + 1}"
        finding = Finding.model_validate_json(row["finding_json"])
        finding.task_id = new_id
        prev.findings[new_id] = finding
        prev.tasks.append(SubTask(id=new_id, question=row["question"], rationale="Earlier research",
                                  search_queries=[], status="done", origin="previous"))
        prev.seen_urls.update(normalize_url(src.url) for src in finding.sources)

    # User context carries forward: inherited items first, then this session's answers/notes.
    question = None
    for m in store.messages(sid):
        if m["kind"] == "inherited":
            prev.context.append(m["content"])
        elif m["kind"] == "clarifying_question":
            question = m["content"]
        elif m["kind"] == "answer":
            prev.context.append(f"Q: {question or '?'}\nA: {m['content']}")
        elif m["kind"] == "note":
            prev.context.append(f"User guidance: {m['content']}")
    return prev


def adopt_previous(store: Store, sid: str, prev: Previous) -> None:
    """Persist what the new session inherits, so it can itself be followed up later."""
    for t in prev.tasks:
        store.upsert_task(sid, t.id, t.question, t.status, t.origin,
                          prev.findings[t.id].model_dump_json())
    for item in prev.context:
        store.add_message(sid, "user", "inherited", item)


def merge_previous(plan: Plan, prev: Previous | None) -> Plan:
    """Put the earlier (already done) tasks in front of the new plan."""
    if prev is not None:
        plan.subtasks = [t.model_copy() for t in prev.tasks] + plan.subtasks
    return plan


def new_task_count(plan: Plan) -> int:
    """Tasks that count against this session's budget (earlier research is free)."""
    return sum(1 for t in plan.subtasks if t.origin != "previous")
