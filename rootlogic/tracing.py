"""Optional Langfuse tracing.

The project already records every model request in ``llm_calls`` and every decision in
``events``, so tracing adds a viewer and eval tooling rather than data we lack. It is off
unless ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set, the dependency is an
extra (``pip install -e '.[langfuse]'``), and every call is wrapped: a tracing backend that
is down, misconfigured or slow must never cost a run its research.

One trace per session, one generation per LLM request, one event per action-log entry - which
is the same shape the action log has, so a reader of either sees the same run.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from .llm import Usage


class Tracer(Protocol):
    def session(self, session_id: str, topic: str) -> None: ...
    def generation(self, usage: Usage) -> None: ...
    def event(self, type_: str, message: str, data: dict | None = None) -> None: ...
    def flush(self) -> None: ...


class NullTracer:
    """What every run uses unless tracing is configured."""

    enabled = False

    def session(self, session_id: str, topic: str) -> None:
        pass

    def generation(self, usage: Usage) -> None:
        pass

    def event(self, type_: str, message: str, data: dict | None = None) -> None:
        pass

    def flush(self) -> None:
        pass


class LangfuseTracer:
    """Sends a run to Langfuse. ``client`` is injectable so tests need no network.

    Written against the v4 SDK, whose API is OpenTelemetry-shaped: observations are created
    against a trace context rather than on a trace object. One trace id is derived from the
    session id, so every generation and event of a run lands on the same trace and a
    resumed run continues the trace it belongs to.
    """

    enabled = True

    def __init__(self, client: Any):
        self.client = client
        self.context: Any = None

    def _safe(self, owner: Any, method: str, **kwargs) -> Any:
        """Call ``owner.method(**kwargs)``, tolerating both failure and absence.

        The SDK's surface differs between majors (v2 had trace()/generation(), v4 has
        start_observation()), so a missing attribute is treated like a failed call: tracing
        degrades, the run continues.
        """
        try:
            call = getattr(owner, method, None)
            return call(**kwargs) if callable(call) else None
        except Exception:  # noqa: BLE001 - tracing must not raise into a run
            return None

    def session(self, session_id: str, topic: str) -> None:
        trace_id = self._safe(self.client, "create_trace_id", seed=session_id)
        self.context = {"trace_id": trace_id} if trace_id else None
        span = self._safe(self.client, "start_observation", trace_context=self.context,
                          name="research", as_type="span", input=topic,
                          metadata={"session_id": session_id})
        if span is not None:
            self._safe(span, "update", name="research", input=topic,
                       metadata={"session_id": session_id})
            self._safe(span, "end")

    def generation(self, usage: Usage) -> None:
        gen = self._safe(
            self.client, "start_observation", trace_context=self.context, name=usage.purpose,
            as_type="generation", model=usage.model,
            usage_details={"input": usage.input_tokens, "output": usage.output_tokens,
                           "cache_read_input_tokens": usage.cache_read_tokens},
            cost_details={"total": usage.cost_usd},
            metadata={"stop_reason": usage.stop_reason, "request_id": usage.request_id,
                      "web_searches": usage.web_searches})
        if gen is not None:
            self._safe(gen, "end")

    def event(self, type_: str, message: str, data: dict | None = None) -> None:
        self._safe(self.client, "create_event", trace_context=self.context, name=type_,
                   input=message, metadata=data or {})

    def flush(self) -> None:
        self._safe(self.client, "flush")


def traced_sink(sink, tracer: Tracer):
    """Wrap a usage sink so every model request is also a generation on the trace.

    Accounting comes first: the store row is written even if tracing fails.
    """
    if not getattr(tracer, "enabled", False):
        return sink

    def record(usage: Usage) -> None:
        sink(usage)
        tracer.generation(usage)

    return record


def tracer_from_env() -> Tracer:
    """A real tracer when Langfuse is configured and installed, otherwise a no-op one."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return NullTracer()
    try:
        from langfuse import Langfuse
    except ImportError:
        return NullTracer()      # configured but not installed: run, don't crash
    try:
        return LangfuseTracer(Langfuse())
    except Exception:  # noqa: BLE001 - bad host, bad keys, unreachable server
        return NullTracer()
