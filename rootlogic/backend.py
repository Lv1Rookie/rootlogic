"""Which model and which web search a run uses, validated in one place.

The CLI and the web server both build a ``Backend`` from their flags and hand it to
``cli.create_engine``. Invalid combinations fail fast with a clear message instead of a
stack trace halfway through a run.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import MODEL, UsageSink


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

    def validate(self) -> Backend:
        if self.provider not in ("anthropic", "openai"):
            raise BackendError(f"unknown provider {self.provider!r}")
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
        return self

    @property
    def label(self) -> str:
        model = self.model or (MODEL if self.provider == "anthropic" else "?")
        where = f" @ {self.base_url}" if self.base_url else ""
        return f"{self.provider}:{model}{where} · search: {self.search}"

    def make_llm(self, usage_sink: UsageSink):
        from .search import get_search_provider
        self.validate()
        search = get_search_provider(self.search)
        if self.provider == "openai":
            from .openai_llm import OpenAICompatibleLLM
            return OpenAICompatibleLLM(usage_sink, model=self.model, search=search,
                                       base_url=self.base_url, strict=self.strict,
                                       prices=self.prices)
        from .llm import AnthropicLLM
        return AnthropicLLM(usage_sink, model=self.model or MODEL, zdr=self.zdr, search=search)


def parse_prices(text: str | None) -> tuple[float, float] | None:
    """'0.25,2.0' -> (0.25, 2.0)."""
    if not text:
        return None
    try:
        inp, out = (float(x) for x in text.split(","))
    except ValueError as e:
        raise BackendError("--prices expects IN,OUT in USD per million tokens, e.g. 0.25,2") from e
    return inp, out
