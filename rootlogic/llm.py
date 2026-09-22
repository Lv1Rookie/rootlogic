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
from .search import SearchProvider, clip
from .tools import MAX_FETCHES, NUDGE, SUBMIT_DESCRIPTION, WEB_TOOL_SPECS, WebToolbox

T = TypeVar("T", bound=BaseModel)

MODEL = "claude-opus-5"
DIRECT_BELOW = 5   # below this many searches, skip dynamic filtering (see _research_server_tools)
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
    prices: tuple[float, float] | None = None  # (input, output) $/MTok for non-Claude models

    @property
    def cost_usd(self) -> float:
        if self.model == "offline-fake":
            return 0.0
        if self.prices is not None:
            inp, out = self.prices
            return ((self.input_tokens + self.cache_read_tokens) * inp
                    + self.output_tokens * out) / 1e6
        if self.model not in PRICES:
            return 0.0  # unknown/local model: unpriced rather than guessed
        inp, out = PRICES[self.model]
        return (self.input_tokens * inp + self.cache_write_tokens * inp * 1.25
                + self.cache_read_tokens * inp * 0.1 + self.output_tokens * out) / 1e6 \
            + self.web_searches * WEB_SEARCH_USD


UsageSink = Callable[[Usage], None]


class AgentRefusal(RuntimeError):
    """The model declined the request (stop_reason == "refusal") even after fallback."""


class LLMError(RuntimeError):
    pass


class AuthError(LLMError):
    """Credentials or billing: a setup problem, not a research failure."""


class LLM(Protocol):
    def structured(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                   effort: str = "high") -> T: ...

    def research(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                 max_searches: int = 8, recency_days: int = 0) -> tuple[T, list[SearchHit]]: ...


def json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema -> JSON schema acceptable to strict structured outputs."""
    return model.model_json_schema()


class AnthropicLLM:
    def __init__(self, usage_sink: UsageSink, *, model: str = MODEL,
                 client: anthropic.Anthropic | None = None, zdr: bool = False,
                 search: SearchProvider | None = None):
        """``search=None`` uses Claude's server-side ``web_search``/``web_fetch`` tools.
        Passing a ``SearchProvider`` swaps in our own client-side tools backed by it, the same
        loop any tool-calling model can run.

        ``zdr=True`` makes the server web tools Zero-Data-Retention eligible by setting
        ``allowed_callers: ["direct"]``. That turns off dynamic filtering (Claude pre-filtering
        search results in code), which usually costs more context tokens, so it is opt-in."""
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.usage_sink = usage_sink
        self.zdr = zdr
        self.search = search

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
        except anthropic.AuthenticationError as e:
            raise AuthError(
                "Anthropic rejected the API key. Check ANTHROPIC_API_KEY holds a real key from "
                "console.anthropic.com (it should be ~100 characters, not the 'sk-ant-...' "
                "placeholder).") from e
        except anthropic.PermissionDeniedError as e:
            raise AuthError(f"This API key isn't allowed to do that: {e.message}") from e
        except anthropic.APIStatusError as e:
            if "credit balance" in (e.message or "").lower():
                raise AuthError(
                    "The Anthropic API account is out of credits. Add credits at "
                    "console.anthropic.com under Plans & Billing. (API usage is billed "
                    "separately from a Claude.ai Pro or Max subscription.)") from e
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
                 max_searches: int = 8, recency_days: int = 0) -> tuple[T, list[SearchHit]]:
        submit_tool = {"name": "submit_findings", "description": SUBMIT_DESCRIPTION,
                       "strict": True, "input_schema": json_schema(schema)}
        if self.search is None:
            return self._research_server_tools(purpose, system, prompt, schema, submit_tool,
                                               max_searches)
        return self._research_client_tools(purpose, system, prompt, schema, submit_tool,
                                           max_searches, recency_days)

    @staticmethod
    def _submitted(response, schema: type[T]) -> T | None:
        submit = next((b for b in response.content
                       if b.type == "tool_use" and b.name == "submit_findings"), None)
        if submit is None:
            return None
        data = submit.input if isinstance(submit.input, dict) else json.loads(submit.input)
        return schema.model_validate(data)

    def _research_server_tools(self, purpose, system, prompt, schema, submit_tool,
                               max_searches) -> tuple:
        """Claude's hosted web tools: Anthropic runs searches inside the request."""
        web_tools = [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches},
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": MAX_FETCHES},
        ]
        # Dynamic filtering runs searches from inside code execution, which can fire a batch
        # at once and exhaust a small budget before any result comes back. Under DIRECT_BELOW
        # searches, call the tools directly so each search costs exactly one use.
        if self.zdr or max_searches < DIRECT_BELOW:
            for t in web_tools:
                t["allowed_callers"] = ["direct"]
        tools = [*web_tools, submit_tool]
        messages: list[dict] = [{"role": "user", "content": prompt}]
        hits: list[SearchHit] = []

        for _ in range(8):
            response = self._create(purpose, max_tokens=16000, system=system, tools=tools,
                                    messages=messages, output_config={"effort": "medium"})
            hits.extend(_search_hits(response.content))
            if (found := self._submitted(response, schema)) is not None:
                return found, hits

            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue  # server tool loop hit its iteration cap; resend to resume
            if response.stop_reason == "max_tokens":
                raise LLMError(f"{purpose}: output truncated at max_tokens")
            messages.append({"role": "user", "content": NUDGE})
        raise LLMError(f"{purpose}: research did not converge")

    def _research_client_tools(self, purpose, system, prompt, schema, submit_tool,
                               max_searches, recency_days) -> tuple:
        """Our own web tools backed by ``self.search``: portable to any tool-calling model."""
        assert self.search is not None
        box = WebToolbox(self.search, max_searches=max_searches, recency_days=recency_days)
        tools = [{"name": n, "description": d, "strict": True, "input_schema": p}
                 for n, d, p in WEB_TOOL_SPECS] + [submit_tool]
        messages: list[dict] = [{"role": "user", "content": prompt}]

        for _ in range(box.max_turns):
            response = self._create(purpose, max_tokens=16000, system=system, tools=tools,
                                    messages=messages, output_config={"effort": "medium"})
            if (found := self._submitted(response, schema)) is not None:
                return found, box.hits
            if response.stop_reason == "max_tokens":
                raise LLMError(f"{purpose}: output truncated at max_tokens")

            messages.append({"role": "assistant", "content": response.content})
            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                messages.append({"role": "user", "content": NUDGE})
                continue
            # All results go back in ONE user message (keeps parallel tool calling working).
            results = []
            for call in calls:
                content, is_error = box.run(call.name, call.input)
                results.append({"type": "tool_result", "tool_use_id": call.id,
                                "content": content, "is_error": is_error})
            messages.append({"role": "user", "content": results})
        raise LLMError(f"{purpose}: research did not converge")


def _search_hits(content) -> list[SearchHit]:
    """Search results, plus fetched pages with their text (evidence for verification)."""
    hits = []
    for block in content:
        if block.type == "web_fetch_tool_result":
            if (page := _fetched_page(block.content)) is not None:
                hits.append(page)
            continue
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


def _fetched_page(result) -> SearchHit | None:
    """web_fetch_result -> document -> text source. Errors and non-text (PDF bytes) -> None."""
    if getattr(result, "type", None) != "web_fetch_result":
        return None
    document = getattr(result, "content", None)
    source = getattr(document, "source", None)
    if getattr(source, "type", None) != "text" or not getattr(source, "data", None):
        return None
    return SearchHit(url=result.url, title=getattr(document, "title", None) or "",
                     text=clip(source.data))
