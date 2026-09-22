"""LLM boundary.

The orchestrator depends only on the small ``LLM`` protocol below, so tests (and the
offline demo) swap in ``FakeLLM`` without touching orchestration code.

``AnthropicLLM`` implements it with the Claude Messages API:
  * ``structured``  -> JSON-schema structured output (plan, reflection, analysis, report)
  * ``research``    -> a small tool loop: server-side ``web_search`` + ``web_fetch`` tools,
                       ending when the model calls the client tool ``submit_findings``
Every request's token usage is reported to a ``UsageSink`` for cost accounting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from .models import SearchHit

T = TypeVar("T", bound=BaseModel)

MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens (Anthropic first-party list price). Cache write = 1.25x input,
# cache read = 0.1x input. Web search is billed per search.
PRICES = {"claude-opus-5": (5.00, 25.00), "claude-opus-4-8": (5.00, 25.00),
          "claude-sonnet-5": (2.00, 10.00), "claude-haiku-4-5": (1.00, 5.00)}
WEB_SEARCH_USD = 10.00 / 1000


@dataclass
class Usage:
    purpose: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    stop_reason: str | None = None
    request_id: str | None = None

    @property
    def cost_usd(self) -> float:
        if self.model == "offline-fake":
            return 0.0
        inp, out = PRICES.get(self.model, PRICES[MODEL])
        return (self.input_tokens * inp + self.cache_write_tokens * inp * 1.25
                + self.cache_read_tokens * inp * 0.1 + self.output_tokens * out) / 1e6 \
            + self.web_searches * WEB_SEARCH_USD


UsageSink = Callable[[Usage], None]


class AgentRefusal(RuntimeError):
    """The model declined the request (stop_reason == "refusal") even after fallback."""


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def structured(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                   effort: str = "high") -> T: ...

    def research(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                 max_searches: int = 5) -> tuple[T, list[SearchHit]]: ...


def json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema -> JSON schema acceptable to strict structured outputs."""
    return model.model_json_schema()


class AnthropicLLM:
    def __init__(self, usage_sink: UsageSink, *, model: str = MODEL,
                 client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.usage_sink = usage_sink

    # ------------------------------------------------------------------ plumbing
    def _create(self, purpose: str, **kwargs):
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                betas=[FALLBACK_BETA],
                fallbacks="default",  # re-run a safety-declined request on a fallback model
                **kwargs,
            )
        except anthropic.APIConnectionError as e:
            raise LLMError(f"network error during {purpose}: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate limited during {purpose}; try again shortly") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error {e.status_code} during {purpose}: {e.message}") from e

        u = response.usage
        stu = getattr(u, "server_tool_use", None)
        self.usage_sink(Usage(
            purpose=purpose,
            model=response.model or self.model,
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            web_searches=(getattr(stu, "web_search_requests", 0) or 0) if stu else 0,
            stop_reason=response.stop_reason,
            request_id=getattr(response, "_request_id", None),
        ))
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise AgentRefusal(f"{purpose}: model declined ({getattr(details, 'category', None)})")
        return response

    # ------------------------------------------------------------------ structured calls
    def structured(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                   effort: str = "high") -> T:
        response = self._create(
            purpose,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort,
                           "format": {"type": "json_schema", "schema": json_schema(schema)}},
        )
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{purpose}: output truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError(f"{purpose}: no text block in response")
        return schema.model_validate_json(text)

    # ------------------------------------------------------------------ research subagent
    def research(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                 max_searches: int = 5) -> tuple[T, list[SearchHit]]:
        tools = [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches},
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 3},
            {
                "name": "submit_findings",
                "description": "Submit your final, source-backed findings for this sub-task. "
                               "Call exactly once, when research is complete.",
                "strict": True,
                "input_schema": json_schema(schema),
            },
        ]
        messages: list[dict] = [{"role": "user", "content": prompt}]
        hits: list[SearchHit] = []

        for _ in range(8):
            response = self._create(purpose, max_tokens=16000, system=system, tools=tools,
                                    messages=messages, output_config={"effort": "medium"})
            hits.extend(_search_hits(response.content))

            submit = next((b for b in response.content
                           if b.type == "tool_use" and b.name == "submit_findings"), None)
            if submit is not None:
                data = submit.input if isinstance(submit.input, dict) else json.loads(submit.input)
                return schema.model_validate(data), hits

            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue  # server tool loop hit its iteration cap; resend to resume
            if response.stop_reason == "max_tokens":
                raise LLMError(f"{purpose}: output truncated at max_tokens")
            messages.append({"role": "user", "content":
                             "Stop searching now and call submit_findings with what you have."})
        raise LLMError(f"{purpose}: research did not converge")


def _search_hits(content) -> list[SearchHit]:
    hits = []
    for block in content:
        if block.type != "web_search_tool_result":
            continue
        results = block.content
        if not isinstance(results, list):  # error object, e.g. max_uses_exceeded
            continue
        for r in results:
            url = getattr(r, "url", None)
            if url:
                hits.append(SearchHit(url=url, title=getattr(r, "title", "") or "",
                                      page_age=getattr(r, "page_age", None)))
    return hits
