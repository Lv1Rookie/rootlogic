"""Which model and which web search a run uses, validated in one place.

The CLI and the web server both build a ``Backend`` from their flags and hand it to
``cli.create_engine``. Invalid combinations fail fast with a clear message instead of a
stack trace halfway through a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .llm import MODEL, UsageSink
from .moderation import Moderator


class BackendError(ValueError):
    pass


@dataclass(frozen=True)
class Backend:
    provider: str = "anthropic"          # anthropic | openai (any OpenAI-compatible server)
    model: str | None = None             # default: claude-opus-5 for anthropic; required for openai
    base_url: str | None = None          # OpenAI-compatible server URL, e.g. Ollama
    search: str = "anthropic"            # anthropic (Claude's hosted tools) | tavily
    zdr: bool = False                    # Claude hosted web tools in Zero-Data-Retention mode
    strict: bool = True                  # strict JSON-schema outputs/tools (openai provider)
    prices: tuple[float, float] | None = None  # $/MTok (input, output) for non-Claude models
    worker_model: str | None = None      # cheaper model for research sub-agents (same provider)
    worker_prices: tuple[float, float] | None = None
    moderation: str = "auto"             # auto | none | openai | llama-guard
    moderation_model: str | None = None  # llama-guard: model id (default llama-guard3)
    moderation_base_url: str | None = None
    moderation_strict: bool = False      # block on every flag, not just harm-enabling ones

    def validate(self) -> Backend:
        if self.provider not in ("anthropic", "openai"):
            raise BackendError(f"unknown provider {self.provider!r}")
        if self.moderation not in ("auto", "none", "openai", "llama-guard"):
            raise BackendError(f"unknown moderation {self.moderation!r}")
        if self.provider == "openai":
            if not self.model:
                raise BackendError("--provider openai needs --model (e.g. gpt-5-mini, llama3.3)")
            if self.search == "anthropic":
                raise BackendError("--provider openai needs --search tavily: Claude's built-in "
                                   "web tools only work with Claude models")
            if self.zdr:
                raise BackendError("--zdr applies to Claude's web tools only")
        elif self.base_url:
            raise BackendError("--base-url is for --provider openai")
        self.resolved_moderation()   # last: the safety default, after the obvious mistakes
        return self

    def resolved_moderation(self) -> str:
        """Claude screens content itself; other models need a moderator or an explicit opt-out."""
        if self.moderation != "auto":
            return self.moderation
        if self.provider == "anthropic":
            return "none"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        raise BackendError(
            "non-Claude models have no built-in safety screening. Set OPENAI_API_KEY to use "
            "OpenAI's free moderation API, or --moderation llama-guard with a local Llama Guard "
            "(ollama pull llama-guard3), or --moderation none to run unscreened.")

    @property
    def label(self) -> str:
        model = self.model or (MODEL if self.provider == "anthropic" else "?")
        where = f" @ {self.base_url}" if self.base_url else ""
        worker = f" · workers: {self.worker_model}" if self.worker_model else ""
        return (f"{self.provider}:{model}{where}{worker} · search: {self.search} · moderation: "
                f"{self.resolved_moderation()}")

    def make_moderator(self) -> Moderator | None:
        choice = self.resolved_moderation()
        if choice == "none":
            return None
        if choice == "openai":
            from .moderation import OpenAIModerator
            return OpenAIModerator(strict=self.moderation_strict)
        from .moderation import LlamaGuardModerator
        return LlamaGuardModerator(
            base_url=self.moderation_base_url or self.base_url or "http://localhost:11434/v1",
            model=self.moderation_model or "llama-guard3", strict=self.moderation_strict)

    def make_llm(self, usage_sink: UsageSink, *, worker: bool = False):
        """The lead model, or (``worker=True``) the cheaper one the research sub-agents use."""
        from .search import get_search_provider
        self.validate()
        search = get_search_provider(self.search)
        model = (self.worker_model or self.model) if worker else self.model
        prices = (self.worker_prices or self.prices) if worker else self.prices
        if self.provider == "openai":
            from .openai_llm import OpenAICompatibleLLM
            return OpenAICompatibleLLM(usage_sink, model=model, search=search,
                                       base_url=self.base_url, strict=self.strict, prices=prices)
        from .llm import AnthropicLLM
        return AnthropicLLM(usage_sink, model=model or MODEL, zdr=self.zdr, search=search)

    def make_worker_llm(self, usage_sink: UsageSink):
        """None when no separate worker model is configured: engines then reuse the lead."""
        return self.make_llm(usage_sink, worker=True) if self.worker_model else None


def parse_prices(text: str | None) -> tuple[float, float] | None:
    """'0.25,2.0' -> (0.25, 2.0)."""
    if not text:
        return None
    try:
        inp, out = (float(x) for x in text.split(","))
    except ValueError as e:
        raise BackendError("--prices expects IN,OUT in USD per million tokens, e.g. 0.25,2") from e
    return inp, out
