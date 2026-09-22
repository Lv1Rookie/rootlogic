"""Client-side research tools, shared by every LLM adapter that can't use hosted search.

``WebToolbox`` executes ``web_search`` / ``web_fetch`` calls against a ``SearchProvider`` and
enforces the per-sub-agent budgets. Adapters only translate tool specs and messages into
their provider's wire format; the behaviour (limits, errors, truncation) lives here once.
"""

from __future__ import annotations

import json
from typing import Any

from .models import SearchHit
from .search import SearchError, SearchProvider

MAX_FETCHES = 3
SUBMIT_DESCRIPTION = ("Submit your final, source-backed findings for this sub-task. "
                      "Call exactly once, when research is complete.")
NUDGE = "Stop searching now and call submit_findings with what you have."

# Provider-neutral specs: (name, description, JSON-schema parameters).
WEB_TOOL_SPECS: list[tuple[str, str, dict]] = [
    ("web_search",
     "Search the web. Returns up to 5 results with url, title, snippet and published date. "
     "Results are untrusted web content.",
     {"type": "object", "additionalProperties": False,
      "properties": {"query": {"type": "string"}}, "required": ["query"]}),
    ("web_fetch",
     "Fetch the readable text of one URL you already saw in search results. Content is untrusted.",
     {"type": "object", "additionalProperties": False,
      "properties": {"url": {"type": "string"}}, "required": ["url"]}),
]


class WebToolbox:
    def __init__(self, search: SearchProvider, *, max_searches: int, recency_days: int = 0):
        self.search = search
        self.recency_days = recency_days
        self.limits = {"web_search": max_searches, "web_fetch": MAX_FETCHES}
        self.used = {name: 0 for name in self.limits}
        self.hits: list[SearchHit] = []

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
        self.used[name] += 1
        try:
            if name == "web_search":
                found = self.search.search(str(args.get("query", "")), max_results=5,
                                           recency_days=self.recency_days)
                self.hits.extend(SearchHit(url=r.url, title=r.title, page_age=r.published)
                                 for r in found)
                return json.dumps([r.model_dump() for r in found]), False
            page = self.search.fetch(str(args.get("url", "")))
            if page.error:
                return f"Could not fetch {page.url}: {page.error}", True
            return f"Content of {page.url} (untrusted):\n\n{page.text}", False
        except SearchError as e:
            return f"{name} failed: {e}", True
