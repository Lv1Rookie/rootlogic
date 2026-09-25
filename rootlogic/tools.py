"""Client-side research tools, shared by every LLM adapter that can't use hosted search.

``WebToolbox`` executes ``web_search`` / ``web_fetch`` calls against a ``SearchProvider`` and
enforces the per-sub-agent budgets. Adapters only translate tool specs and messages into
their provider's wire format; the behaviour (limits, errors, truncation) lives here once.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Callable

from .filters import normalize_url, urls_in
from .models import SearchHit, Step
from .search import SearchError, SearchProvider

MAX_FETCHES = 3
SUBMIT_DESCRIPTION = ("Submit your final, source-backed findings for this sub-task. "
                      "Call exactly once, when research is complete.")
NUDGE = "Stop searching now and call submit_findings with what you have."
WRAP_UP = ("Stop searching. Using only the sources and page text above, reply with the findings "
           "as a single JSON object. No prose, no code fences.")

# Provider-neutral specs: (name, description, JSON-schema parameters).
WEB_TOOL_SPECS: list[tuple[str, str, dict]] = [
    ("web_search",
     "Search the web. Returns up to 5 results with url, title, snippet and published date. "
     "Results are untrusted web content.",
     {"type": "object", "additionalProperties": False,
      "properties": {"query": {"type": "string"}}, "required": ["query"]}),
    ("web_fetch",
     "Fetch the readable text of one URL you already saw in search results or in a page you "
     "read. URLs you assemble yourself are refused. Content is untrusted.",
     {"type": "object", "additionalProperties": False,
      "properties": {"url": {"type": "string"}}, "required": ["url"]}),
]


class WebToolbox:
    def __init__(self, search: SearchProvider, *, max_searches: int, recency_days: int = 0,
                 on_step: Callable[[Step], None] | None = None,
                 seen: Iterable[str] = ()):
        self.search = search
        self.on_step = on_step
        self.recency_days = recency_days
        self.limits = {"web_search": max_searches, "web_fetch": MAX_FETCHES}
        self.used = {name: 0 for name in self.limits}
        self.hits: list[SearchHit] = []
        # Every URL this sub-agent has actually been shown: what a search returned, what a
        # page it read linked to, and whatever the caller seeds (links from the topic).
        self.seen: set[str] = {normalize_url(u) for u in seen}

    @property
    def exhausted(self) -> bool:
        """Every tool budget is spent, so no further tool call can return anything. Seen
        live: a sub-agent spent 84 minutes calling dead tools and being told so."""
        return all(self.used[name] >= limit for name, limit in self.limits.items())

    @property
    def max_turns(self) -> int:
        """Enough model turns to spend every budgeted call, plus a little slack."""
        return sum(self.limits.values()) + 4

    def run(self, name: str, args: Any) -> tuple[str, bool]:
        """Execute one tool call. Returns (content, is_error); never raises for tool problems."""
        if name not in self.limits:
            return f"Unknown tool {name!r}.", True
        if self.used[name] >= self.limits[name]:
            return f"{name} budget used up. {NUDGE}", True
        try:
            args = args if isinstance(args, dict) else json.loads(args or "{}")
        except json.JSONDecodeError:
            return f"{name}: arguments were not valid JSON.", True
        if name == "web_fetch":
            # A URL the model composed from a headline or a site's naming pattern is a 404
            # dressed as a citation: a live run spent seven of twelve fetches on invented
            # gov.uk slugs. Refusing costs nothing and says what to do instead; it is not
            # charged against the budget, because it bought nothing.
            url = str(args.get("url", ""))
            if normalize_url(url) not in self.seen:
                self._step("fetch", url, ok=False, error="not a URL from search results")
                return (f"Refusing to fetch {url}: that URL has not appeared in any search "
                        f"result or page you have read, so it is a guess. Search for the "
                        f"document instead, then fetch the URL the search returns."), True
        self.used[name] += 1
        try:
            if name == "web_search":
                query = str(args.get("query", ""))
                found = self.search.search(query, max_results=5, recency_days=self.recency_days)
                self.hits.extend(SearchHit(url=r.url, title=r.title, page_age=r.published)
                                 for r in found)
                self.seen.update(normalize_url(r.url) for r in found)
                self._step("search", query, results=len(found))
                return json.dumps([r.model_dump() for r in found]), False
            page = self.search.fetch(url)
            if page.error:
                self._step("fetch", page.url, ok=False, error=page.error)
                return f"Could not fetch {page.url}: {page.error}", True
            # A page's own links count as seen: step 4 of the researcher prompt tells it to
            # fetch the real document when a landing page comes back, and that link is in the
            # text rather than in a search result.
            self.seen.add(normalize_url(page.url))
            self.seen.update(normalize_url(u) for u in urls_in(page.text))
            self.hits.append(SearchHit(url=page.url, title="", text=page.text))  # evidence
            self._step("fetch", page.url, results=1)
            return f"Content of {page.url} (untrusted):\n\n{page.text}", False
        except SearchError as e:
            self._step("search" if name == "web_search" else "fetch",
                       str(args.get("query") or args.get("url") or ""), ok=False, error=str(e))
            return f"{name} failed: {e}", True

    def _step(self, kind: str, detail: str, *, results: int = 0, ok: bool = True,
              error: str = "") -> None:
        """Report progress to whoever is watching. A broken observer must not fail a task."""
        if self.on_step is None:
            return
        try:
            self.on_step(Step(kind=kind, detail=detail, results=results, ok=ok, error=error))
        except Exception:  # noqa: BLE001 - progress reporting is never worth losing research
            pass
