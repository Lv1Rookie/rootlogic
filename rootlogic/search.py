"""Search providers: the web access that research sub-agents use, behind one small protocol.

Why this exists
  Provider-hosted search tools (Anthropic ``web_search``/``web_fetch``, OpenAI ``web_search``,
  Gemini grounding) only work inside their own vendor's API. Putting search behind
  ``SearchProvider`` and exposing it to the model as ordinary client tools means any LLM
  that supports tool calling can drive the same research loop, so models become swappable.

Implementations
  * ``TavilySearch``  - Tavily Search + Extract REST API (needs ``TAVILY_API_KEY``).
  * ``StaticSearch``  - canned results for tests and offline demos.

Everything a provider returns is untrusted web content; the researcher prompt tells the
model to treat it as data, never as instructions.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from pydantic import BaseModel

from .filters import clean_domain

DEFAULT_MAX_CHARS = 20_000
MAX_DOMAINS = 3          # sites one search may be restricted to


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    published: str | None = None   # ISO date or provider string, if known


class FetchedPage(BaseModel):
    url: str
    text: str = ""
    error: str | None = None


class SearchError(RuntimeError):
    pass


class SearchQuotaExceeded(SearchError):
    """The search plan is used up. Unlike a failed query, waiting will not help: every
    remaining sub-agent would spend model calls on a provider that has already said no."""


# Tavily answers an exhausted plan with 432, and an unpaid one with 402. Both mean "not this
# month", as opposed to 429, which means "not this second" and is worth retrying.
QUOTA_STATUS = frozenset({402, 432})


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, *, max_results: int = 5, recency_days: int = 0,
               domains: list[str] | None = None) -> list[SearchResult]:
        """Web search. ``recency_days`` > 0 asks the provider to prefer/limit to recent pages.

        ``domains`` restricts results to those sites, which is how a sub-agent reaches a
        primary source it cannot otherwise surface - the official consultation on gov.uk
        rather than the trade-press write-up of it.
        """
        ...

    def fetch(self, url: str) -> FetchedPage:
        """Readable text of one page. Failures come back as ``FetchedPage.error``, not raises."""
        ...


def clip(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Bound page text for the model's context, saying so explicitly when content is cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[truncated: {len(text) - max_chars:,} more characters]"


# =================================================================== Tavily


def _time_range(recency_days: int) -> str | None:
    """Map a day window onto Tavily's coarse ``time_range`` buckets (widest that still fits)."""
    if recency_days <= 0:
        return None
    for limit, bucket in ((1, "day"), (7, "week"), (31, "month"), (366, "year")):
        if recency_days <= limit:
            return bucket
    return None  # older than a year: don't filter; our own recency filter still applies


class TavilySearch:
    """https://docs.tavily.com - ``POST /search`` (1 credit, basic depth) and ``POST /extract``."""

    name = "tavily"
    BASE = "https://api.tavily.com"

    def __init__(self, api_key: str | None = None, *, timeout: float = 30.0,
                 max_chars: int = DEFAULT_MAX_CHARS):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            raise SearchError("TAVILY_API_KEY is not set")
        self.timeout = timeout
        self.max_chars = max_chars

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.BASE + path, data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in QUOTA_STATUS:
                raise SearchQuotaExceeded(
                    f"Tavily is out of credits (HTTP {e.code}). Research needs search: top up "
                    f"the plan, wait for the quota to reset, or use --search anthropic with "
                    f"Claude's own web tools.") from e
            raise SearchError(f"Tavily {path} failed: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise SearchError(f"Tavily {path} failed: {e}") from e

    def search(self, query: str, *, max_results: int = 5, recency_days: int = 0,
               domains: list[str] | None = None) -> list[SearchResult]:
        body: dict = {"query": query, "max_results": max(1, min(max_results, 20)),
                      "search_depth": "basic", "include_published_date": True}
        if tr := _time_range(recency_days):
            body["time_range"] = tr
        if domains:
            # A site-restricted search is a narrow haystack, and "basic" depth on a narrow
            # haystack often returns the section page rather than the document. The whole
            # point of naming the site is to reach the document, so pay for the deeper pass.
            body["include_domains"] = [clean_domain(d) for d in domains[:MAX_DOMAINS] if d]
            body["search_depth"] = "advanced"
        data = self._post("/search", body)
        return [SearchResult(url=r["url"], title=r.get("title") or "",
                             snippet=r.get("content") or "",
                             published=r.get("published_date"))
                for r in data.get("results", []) if r.get("url")]

    def fetch(self, url: str) -> FetchedPage:
        try:
            data = self._post("/extract", {"urls": [url], "format": "text"})
        except SearchError as e:
            return FetchedPage(url=url, error=str(e))
        for r in data.get("results", []):
            return FetchedPage(url=r.get("url", url), text=clip(r.get("raw_content") or "",
                                                                self.max_chars))
        failed = data.get("failed_results") or [{}]
        return FetchedPage(url=url, error=failed[0].get("error") or "extraction failed")


# =================================================================== Static (tests, offline)


class StaticSearch:
    """Deterministic provider: every query returns ``results``; ``pages`` maps url -> text."""

    name = "static"

    def __init__(self, results: list[SearchResult] | None = None,
                 pages: dict[str, str] | None = None):
        self.results = results or []
        self.pages = pages or {}
        self.queries: list[tuple[str, int]] = []   # (query, recency_days)
        self.domain_queries: list[tuple[str, list[str]]] = []
        self.fetched: list[str] = []

    def search(self, query: str, *, max_results: int = 5, recency_days: int = 0,
               domains: list[str] | None = None) -> list[SearchResult]:
        self.queries.append((query, recency_days))
        self.domain_queries.append((query, list(domains or [])))
        if domains:
            allowed = {clean_domain(d) for d in domains}
            hits = [r for r in self.results
                    if any(clean_domain(r.url).endswith(d) for d in allowed)]
            return hits[:max_results]
        return self.results[:max_results]

    def fetch(self, url: str) -> FetchedPage:
        self.fetched.append(url)
        if url in self.pages:
            return FetchedPage(url=url, text=clip(self.pages[url]))
        return FetchedPage(url=url, error="not found")


def get_search_provider(name: str) -> SearchProvider | None:
    """``anthropic`` -> None (use Claude's server-side web tools); otherwise a client provider."""
    if name == "anthropic":
        return None
    if name == "tavily":
        return TavilySearch()
    raise ValueError(f"unknown search provider {name!r} (choose anthropic or tavily)")
