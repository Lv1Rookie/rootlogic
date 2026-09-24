"""Human-in-the-loop surface: events out, commands in.

The orchestrator never talks to a terminal or browser directly. It emits ``Event``s and
calls an ``Interaction`` for the few decisions a human owns (answer a question, approve
the plan, override mid-run). CLI, web UI and tests each implement ``Interaction``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from .models import Plan, Step

Action = Literal["continue", "skip", "add", "note", "stop", "abort"]


@dataclass
class Event:
    type: str          # e.g. plan.created, task.started, source.dropped, reflect.done
    message: str       # human-readable one-liner for the action log
    data: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


@dataclass
class Command:
    """A user override. ``arg`` is a task id (skip), a question (add), or free text (note)."""
    action: Action
    arg: str = ""


class Interaction(Protocol):
    def on_event(self, event: Event) -> None: ...
    def ask(self, question: str) -> str: ...
    def review_plan(self, plan: Plan) -> Plan | None: ...
    def override(self, plan: Plan) -> list[Command]: ...


class Control:
    """Thread-safe pause and abort flags, set by a UI (Ctrl-C, a button) and read by the
    engines at safe points.

    Pause hands control to ``Interaction.override``; abort ends the run. Neither can
    interrupt a model call that is already in flight, so the finest granularity either can
    have is one sub-task: a wave of sub-agents runs for minutes, and checking only between
    waves made Pause look broken and left Abort unreachable behind it.
    """

    def __init__(self) -> None:
        self._pause = threading.Event()
        self._abort = threading.Event()
        self._go = threading.Event()   # set means "not held"; a hold clears it
        self._go.set()
        self._lock = threading.Lock()
        self._announced = False        # so a wave of threads logs one pause, not one each

    def request_pause(self) -> None:
        self._pause.set()

    def consume_pause(self) -> bool:
        if self._pause.is_set():
            self._pause.clear()
            return True
        return False

    # LangGraph re-runs an interrupted node from its start on resume, so the graph engine
    # peeks at the flag and clears it only after the override interrupt has been answered.
    def pause_pending(self) -> bool:
        return self._pause.is_set()

    # ---------------------------------------------------------------- hold / resume
    # A hold is the plain Pause a user expects: the run stops at its next safe point and
    # stays stopped until Resume, with nothing to fill in. It is separate from the pause
    # flag above, which asks for an override card and continues once that is answered.
    def request_hold(self) -> None:
        self._go.clear()

    def release(self) -> None:
        self._go.set()

    @property
    def held(self) -> bool:
        return not self._go.is_set()

    def wait_while_held(self) -> bool:
        """Block until Resume (or Abort). True if the caller actually waited.

        Abort wakes the gate so a held run can still be stopped - the caller checks for an
        abort straight after.
        """
        if not self.held:
            return False
        while not self._go.wait(timeout=0.05):
            if self._abort.is_set():
                return True
        return True

    def hold_here(self, emit) -> bool:
        """Safe point: block while held, announcing the pause and the resume once.

        Several sub-agent threads can arrive here at the same time, so the first one to
        arrive logs the pause and the last one to leave logs the resume.
        """
        if not self.held:
            return False
        with self._lock:
            first, self._announced = not self._announced, True
        if first:
            emit("control.paused", "Paused — press Resume to continue")
        self.wait_while_held()
        with self._lock:
            last, self._announced = self._announced, False
        if last and not self.aborting:
            emit("control.resumed", "Resumed")
        return True

    def request_abort(self) -> None:
        """End the run at the next safe point, without waiting for a pause to be answered."""
        self._abort.set()

    @property
    def aborting(self) -> bool:
        return self._abort.is_set()

    def clear(self) -> None:
        self._pause.clear()
        self._abort.clear()
        self._go.set()


def step_event(task_id: str, step: Step) -> tuple[str, str, dict]:
    """A sub-agent's search or fetch as (event type, message, data) for the action log.

    Both engines report sub-agent progress the same way, so the wording lives here once.
    """
    why = f": {step.error}" if step.error else ""
    if step.kind == "search":
        outcome = f"{step.results} result(s)" if step.ok else f"search failed{why}"
        return ("subagent.search",
                f"[{task_id}] Searched \u201c{step.detail}\u201d \u2014 {outcome}",
                {"task": task_id, "query": step.detail, "results": step.results,
                 "error": step.error})
    verb = "Read" if step.ok else "Could not read"
    return ("subagent.fetch", f"[{task_id}] {verb} {step.detail}{why}",
            {"task": task_id, "url": step.detail, "error": step.error})
