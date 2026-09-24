"""OpenAI-compatible LLM adapter: GPT models, local models (Ollama, LM Studio, vLLM), OpenRouter.

Implements the same two-method ``LLM`` protocol as ``AnthropicLLM`` over the Chat Completions
wire format (``POST /v1/chat/completions``), which many providers accept. Point ``base_url`` at
a compatible server to switch providers:

    OpenAI      base_url=None (default)               api key from OPENAI_API_KEY
    Ollama      base_url="http://localhost:20128/v1"   no key needed
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
from dataclasses import dataclass, field
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

from .llm import (AgentRefusal, AuthError, LLMError, StepSink, Usage, UsageSink,
                  json_schema)
from .models import SearchHit
from .search import SearchProvider
from .tools import NUDGE, SUBMIT_DESCRIPTION, WEB_TOOL_SPECS, WRAP_UP, WebToolbox

T = TypeVar("T", bound=BaseModel)

MUTE_LIMIT = 2   # prose turns in a row before a sub-agent is declared stuck (see research())


class OpenAICompatibleLLM:
    def __init__(self, usage_sink: UsageSink, *, model: str, search: SearchProvider,
                 base_url: str | None = None, api_key: str | None = None, strict: bool = True,
                 prices: tuple[float, float] | None = None, reasoning_effort: str | None = None,
                 stream: bool = False, client: openai.OpenAI | None = None):
        """``prices`` = (input, output) USD per million tokens, for cost accounting; None
        records $0 (right for local models; set it for paid APIs).

        ``reasoning_effort`` is sent with every request when set. Thinking models reason
        before every call, which on a local server is most of the wall clock: qwen3:8b spent
        72s on a clarification that takes 1.3s with ``"none"``. Servers that don't know the
        parameter reject it, so it is opt-in rather than a default."""
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
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.usage_sink = usage_sink

    # ------------------------------------------------------------------ plumbing
    def _create(self, purpose: str, **kwargs):
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.stream:
            # Gateways time out waiting for the first byte of a slow model's answer
            # (OmniRoute: 30s). A streamed reply starts within a second or two, so the
            # whole class of "upstream timeout" failures disappears.
            kwargs["stream"] = True
            kwargs.setdefault("stream_options", {"include_usage": True})
        try:
            response = self.client.chat.completions.create(model=self.model, **kwargs)
            if self.stream:
                response = _collect(response)
        except openai.APIConnectionError as e:
            raise LLMError(f"network error during {purpose}: {e}") from e
        except openai.RateLimitError as e:
            raise LLMError(f"rate limited during {purpose}; try again shortly") from e
        except openai.AuthenticationError as e:
            raise AuthError("The API key was rejected. Check OPENAI_API_KEY (or --base-url for "
                            "a local server, which usually needs no key).") from e
        except openai.APIStatusError as e:
            if "quota" in (e.message or "").lower() or "billing" in (e.message or "").lower():
                raise AuthError(f"The model provider rejected the request for billing reasons: "
                                f"{e.message}") from e
            raise LLMError(f"API error {e.status_code} during {purpose}: {e.message}") from e
        except openai.APIError as e:
            # The base class, and not a subclass of any of the above: a streamed request that
            # fails mid-stream raises it, which escaped as a traceback from a live run.
            raise LLMError(f"API error during {purpose}: {e}") from e

        # Not every OpenAI-compatible server returns a well-formed completion: routers and
        # local servers can answer 200 with an error object and no choices at all.
        choice = (response.choices or [None])[0]
        u = response.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        usage = Usage(purpose=purpose, model=response.model or self.model,
                      input_tokens=((u.prompt_tokens or 0) - cached) if u else 0,
                      output_tokens=(u.completion_tokens or 0) if u else 0, cache_read_tokens=cached,
                      stop_reason=choice.finish_reason if choice else "no_choices",
                      request_id=getattr(response, "_request_id", None), prices=self.prices)
        self.usage_sink(usage)

        if choice is None:
            raise LLMError(f"{purpose}: the provider returned no completion. {_why(response)}")
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
            try:
                return _parse(purpose, schema, message.content)
            except LLMError as e:
                # Asking for a schema is not the same as getting one: a gateway may drop
                # response_format on its way upstream, and the reply comes back as prose or
                # half-fenced JSON. Seen live, where a small schema parsed and a larger one
                # broke mid-object. Repair the same way non-strict mode always has.
                return self._repair(purpose, messages, schema, message, e)

        messages[1]["content"] += _json_instruction(schema)
        message = self._create(purpose, messages=messages)
        try:
            return _parse(purpose, schema, message.content)
        except LLMError as e:
            return self._repair(purpose, messages, schema, message, e)

    def _repair(self, purpose: str, messages: list[dict], schema: type[T], message,
                error: LLMError) -> T:
        """One more attempt, showing the model its own reply and what was wrong with it."""
        messages = [*messages,
                    {"role": "assistant", "content": message.content or ""},
                    {"role": "user", "content": f"That was not valid: {error}. Reply with only "
                                                f"the corrected JSON.{_json_instruction(schema)}"}]
        return _parse(purpose, schema, self._create(purpose, messages=messages).content)

    # ------------------------------------------------------------------ research subagent
    def research(self, *, purpose: str, system: str, prompt: str, schema: type[T],
                 max_searches: int = 8, recency_days: int = 0,
                 on_step: StepSink | None = None) -> tuple[T, list[SearchHit]]:
        box = WebToolbox(self.search, max_searches=max_searches, recency_days=recency_days,
                         on_step=on_step)
        specs = WEB_TOOL_SPECS + [("submit_findings", SUBMIT_DESCRIPTION, json_schema(schema))]
        tools = [{"type": "function", "function": {
            "name": n, "description": d, "parameters": p, **({"strict": True} if self.strict else {})}}
            for n, d, p in specs]
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}]

        mute_turns = 0   # turns in a row that answered in prose instead of calling a tool
        for _ in range(box.max_turns):
            message = self._create(purpose, messages=messages, tools=tools)
            calls = message.tool_calls or []
            messages.append(_assistant_turn(message))
            if not calls:
                # Small models drift out of tool calling as the conversation fills with web
                # text: seen live with qwen3:8b, which wrote the answer as prose from ~6k
                # tokens on. Take the answer if it is really the findings in disguise.
                if (found := _findings_in_text(schema, message.content)) is not None:
                    return found, box.hits
                mute_turns += 1
                if mute_turns >= MUTE_LIMIT:
                    return self._wrap_up(purpose, messages, schema, box)
                messages.append({"role": "user", "content": NUDGE})
                continue
            mute_turns = 0

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
            if box.exhausted:
                # Searches and fetches are spent: another tool-armed turn can only be the
                # model calling a dead tool. Ask for the findings directly.
                return self._wrap_up(purpose, messages, schema, box)
        return self._wrap_up(purpose, messages, schema, box)

    def _wrap_up(self, purpose: str, messages: list[dict], schema: type[T],
                 box: WebToolbox) -> tuple[T, list[SearchHit]]:
        """Ask for the findings as constrained structured output, with no tools offered.

        A sub-agent that has searched but won't call ``submit_findings`` has done the work and
        is failing at the call format. Live, an 8B model through Ollama searched correctly and
        then wrote prose every turn: tool-call arguments are free-form text to it, while a
        JSON-schema ``response_format`` is grammar-constrained and it handles that reliably.
        """
        if not box.hits:
            raise LLMError(f"{purpose}: the model never ran a usable search, so there is "
                           "nothing to report")
        convo = [*messages, {"role": "user", "content": WRAP_UP}]
        if self.strict:
            fmt = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": json_schema(schema), "strict": True}}
            message = self._create(purpose, messages=convo, response_format=fmt)
        else:
            convo[-1]["content"] += _json_instruction(schema)
            message = self._create(purpose, messages=convo)
        found = _findings_in_text(schema, message.content)
        if found is None:
            raise LLMError(f"{purpose}: the model stopped calling tools and could not produce "
                           "its findings as JSON either. Try a model that is stronger at "
                           "function calling, or fewer --searches so the conversation stays "
                           "short.")
        return found, box.hits


# =================================================================== helpers


@dataclass
class _Fn:
    name: str = ""
    arguments: str = ""


@dataclass
class _ToolCall:
    id: str = ""
    type: str = "function"
    function: _Fn = field(default_factory=_Fn)


@dataclass
class _Message:
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[_ToolCall] | None = None


@dataclass
class _Choice:
    message: _Message
    finish_reason: str | None = None


@dataclass
class _Response:
    """What ``_create`` needs from a completion, rebuilt from streamed chunks."""
    choices: list[_Choice]
    usage: object | None = None
    model: str = ""


def _collect(chunks) -> _Response:
    """Reassemble a streamed completion into the object the rest of the adapter expects.

    Deltas arrive in pieces: text by fragment, tool calls by index with the arguments JSON
    split across chunks. Usage comes in a final chunk (``stream_options.include_usage``),
    which some servers omit entirely.
    """
    message = _Message()
    calls: dict[int, _ToolCall] = {}
    finish_reason = None
    usage = None
    model = ""

    for chunk in chunks:
        model = getattr(chunk, "model", "") or model
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        choice = (getattr(chunk, "choices", None) or [None])[0]
        if choice is None:
            continue
        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        if getattr(delta, "content", None):
            message.content = (message.content or "") + delta.content
        if getattr(delta, "refusal", None):
            message.refusal = (message.refusal or "") + delta.refusal
        for part in getattr(delta, "tool_calls", None) or []:
            call = calls.setdefault(part.index, _ToolCall())
            if getattr(part, "id", None):
                call.id = part.id
            fn = getattr(part, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    call.function.name = fn.name
                if getattr(fn, "arguments", None):
                    call.function.arguments += fn.arguments

    message.tool_calls = [calls[i] for i in sorted(calls)] or None
    return _Response(choices=[_Choice(message=message, finish_reason=finish_reason)],
                     usage=usage, model=model)


def _why(response) -> str:
    """Whatever the server said instead of a completion (routers put an error object here)."""
    error = getattr(response, "error", None)
    if error is None and isinstance(getattr(response, "model_extra", None), dict):
        error = response.model_extra.get("error")
    if isinstance(error, dict):
        detail = error.get("message") or error
        code = error.get("code")
        return f"Provider said: {detail}" + (f" (code {code})" if code else "")
    if error:
        return f"Provider said: {error}"
    return ("No error message was included. The model may be rate limited or unavailable: try "
            "another --model, or --no-strict if it rejects strict JSON schemas.")


def _findings_in_text(schema: type[T], text: str | None) -> T | None:
    """A model that answers in prose sometimes still emits the findings JSON, just not as a
    tool call. Accept that rather than nudging a model that has stopped calling tools."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return schema.model_validate_json(cleaned[start:end + 1])
    except ValidationError:
        return None


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
