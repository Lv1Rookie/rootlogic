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

from .models import Plan

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
    """Thread-safe pause flag. A UI sets it (Ctrl-C, a Pause button); the orchestrator
    checks it at safe points between waves and hands control to ``Interaction.override``."""

    def __init__(self) -> None:
        self._pause = threading.Event()

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

    def clear(self) -> None:
        self._pause.clear()
