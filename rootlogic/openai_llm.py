"""OpenAI-compatible LLM adapter: GPT models, local models (Ollama, LM Studio, vLLM), OpenRouter.

Implements the same two-method ``LLM`` protocol as ``AnthropicLLM`` over the Chat Completions
wire format (``POST /v1/chat/completions``), which many providers accept. Point ``base_url`` at
a compatible server to switch providers:

    OpenAI      base_url=None (default)               api key from OPENAI_API_KEY
    Ollama      base_url="http://localhost:11434/v1"   no key needed
    OpenRouter  base_url="https://openrouter.ai/api/v1"

Research needs a ``SearchProvider``: hosted search tools are vendor-specific, so sub-agents use
our own ``web_search``/``web_fetch`` tools (``tools.WebToolbox``), exactly like
``AnthropicLLM`` with ``--search tavily``.

Servers differ in what they support. ``strict=True`` (default) uses strict JSON-schema
structured outputs and strict function tools. For servers that reject them, ``strict=False``
asks for JSON in the prompt instead, validates it, and gives the model one chance to fix it.
"""

from __future__ import annotations

import json
import os
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

from .llm import AgentRefusal, LLMError, Usage, UsageSink, json_schema
from .models import SearchHit
from .search import SearchProvider
from .tools import NUDGE, SUBMIT_DESCRIPTION, WEB_TOOL_SPECS, WebToolbox

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleLLM:
    def __init__(self, usage_sink: UsageSink, *, model: str, search: SearchProvider,
                 base_url: str | None = None, api_key: str | None = None, strict: bool = True,
                 prices: tuple[float, float] | None = None, client: openai.OpenAI | None = None):
        """``prices`` = (input, output) USD per million tokens, for cost accounting; None
        records $0 (right for local models; set it for paid APIs)."""
        if search is None:
            raise LLMError("OpenAI-compatible models need a SearchProvider (e.g. --search "
                           "tavily): Claude's web tools only work with Claude.")
        if client is None:
            key = api_key or os.environ.get("OPENAI_API_KEY") or ("local" if base_url else None)
            client = openai.OpenAI(base_url=base_url, api_key=key)
        self.client = client
        self.model = model
        self.search = search
        self.strict = strict
        self.prices = prices
        self.usage_sink = usage_sink

    # ------------------------------------------------------------------ plumbing
    def _create(self, purpose: str, **kwargs):
        try:
            response = self.client.chat.completions.create(model=self.model, **kwargs)
        except openai.APIConnectionError as e:
            raise LLMError(f"network error during {purpose}: {e}") from e
        except openai.RateLimitError as e:
            raise LLMError(f"rate limited during {purpose}; try again shortly") from e
        except openai.APIStatusError as e:
            raise LLMError(f"API error {e.status_code} during {purpose}: {e.message}") from e

        u = response.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        usage = Usage(purpose=purpose, model=response.model or self.model,
                      input_tokens=((u.prompt_tokens or 0) - cached) if u else 0,
                      output_tokens=(u.completion_tokens or 0) if u else 0,
                      cache_read_tokens=cached, stop_reason=response.choices[0].finish_reason,
                      request_id=getattr(response, "_request_id", None), prices=self.prices)
        self.usage_sink(usage)

        choice = response.choices[0]
        if choice.finish_reason == "content_filter" or getattr(choice.message, "refusal", None):
            raise AgentRefusal(f"{purpose}: model declined "
                               f"({choice.message.refusal or choice.finish_reason})")
        if choice.finish_reason == "length":
            raise LLMError(f"{purpose}: output truncated at the model's token limit")
        return choice.message

    # ------------------------------------------------------------------ structured calls
    def structured(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                   effort: str = "high") -> T:
        """``effort`` is Claude-specific and ignored here."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        if self.strict:
            fmt = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": json_schema(schema), "strict": True}}
            message = self._create(purpose, messages=messages, response_format=fmt)
            return _parse(purpose, schema, message.content)

        messages[1]["content"] += _json_instruction(schema)
        message = self._create(purpose, messages=messages)
        try:
            return _parse(purpose, schema, message.content)
        except LLMError as first_error:  # one repair attempt, showing the model its mistake
            messages += [{"role": "assistant", "content": message.content or ""},
                         {"role": "user", "content": f"That was not valid: {first_error}. "
                                                     "Reply with only the corrected JSON."}]
            return _parse(purpose, schema, self._create(purpose, messages=messages).content)

    # ------------------------------------------------------------------ research subagent
    def research(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                 max_searches: int = 5, recency_days: int = 0) -> tuple[T, list[SearchHit]]:
        box = WebToolbox(self.search, max_searches=max_searches, recency_days=recency_days)
        specs = WEB_TOOL_SPECS + [("submit_findings", SUBMIT_DESCRIPTION, json_schema(schema))]
        tools = [{"type": "function", "function": {
            "name": n, "description": d, "parameters": p, **({"strict": True} if self.strict else {})}}
            for n, d, p in specs]
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}]

        for _ in range(box.max_turns):
            message = self._create(purpose, messages=messages, tools=tools)
            calls = message.tool_calls or []
            messages.append(_assistant_turn(message))
            if not calls:
                messages.append({"role": "user", "content": NUDGE})
                continue

            finding: T | None = None
            for call in calls:  # every tool call must get a tool message back
                name, args = call.function.name, call.function.arguments
                if name == "submit_findings":
                    try:
                        finding = schema.model_validate_json(args or "{}")
                        content = "Received."
                    except ValidationError as e:
                        content = f"Invalid findings, fix and resubmit: {_brief(e)}"
                else:
                    content, _ = box.run(name, args)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            if finding is not None:
                return finding, box.hits
        raise LLMError(f"{purpose}: research did not converge")


# =================================================================== helpers


def _assistant_turn(message) -> dict:
    turn: dict = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        turn["tool_calls"] = [{"id": c.id, "type": "function",
                               "function": {"name": c.function.name,
                                            "arguments": c.function.arguments}}
                              for c in message.tool_calls]
    return turn


def _json_instruction(schema: type[BaseModel]) -> str:
    return ("\n\nReply with ONLY a JSON object (no prose, no code fences) that matches this "
            f"JSON Schema:\n{json.dumps(json_schema(schema))}")


def _parse(purpose: str, schema: type[T], text: str | None) -> T:
    if not text:
        raise LLMError(f"{purpose}: empty response")
    cleaned = text.strip()
    if cleaned.startswith("```"):  # tolerate a fenced reply in non-strict mode
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        return schema.model_validate_json(cleaned)
    except ValidationError as e:
        raise LLMError(f"{purpose}: response did not match {schema.__name__}: {_brief(e)}") from e


def _brief(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:5])
