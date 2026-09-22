"""Deterministic source filtering: recency, relevance, credibility, de-duplication.

Kept free of LLM calls so the rules are unit-testable and explainable in a demo:
the model *proposes* sources, code *decides* which ones survive, and every drop is
logged with a reason.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

from .models import SearchHit, SourceDraft

_RELATIVE = re.compile(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", re.I)
_UNIT_DAYS = {"minute": 0, "hour": 0, "day": 1, "week": 7, "month": 30, "year": 365}
_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%B %Y")


def parse_date(value: str | None, today: date) -> date | None:
    """Best-effort parse of ISO dates, 'March 3, 2025', or '3 days ago'. None if unknown."""
    if not value or value.strip().lower() in ("unknown", "n/a", "none"):
        return None
    text = value.strip()
    if m := _RELATIVE.search(text):
        return today - timedelta(days=int(m.group(1)) * _UNIT_DAYS[m.group(2).lower()])
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in _FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalize_url(url: str) -> str:
    """Lowercase host, drop fragment, tracking params, and trailing slash."""
    parts = urlsplit(url.strip())
    query = "&".join(
        p for p in parts.query.split("&") if p and not p.lower().startswith(("utm_", "ref=", "fbclid"))
    )
    host = parts.netloc.lower().removeprefix("www.")
    return urlunsplit((parts.scheme.lower() or "https", host, parts.path.rstrip("/"), query, ""))


def fill_dates_from_hits(sources: list[SourceDraft], hits: list[SearchHit]) -> None:
    """Use the search tool's page_age when the model couldn't determine a date."""
    ages = {normalize_url(h.url): h.page_age for h in hits if h.page_age}
    for s in sources:
        if s.published in ("", "unknown") and (age := ages.get(normalize_url(s.url))):
            s.published = age


def filter_sources(
    sources: list[SourceDraft],
    *,
    recency_days: int,
    today: date,
    seen_urls: set[str] | None = None,
    blocked_domains: tuple[str, ...] = (),
) -> tuple[list[SourceDraft], list[tuple[str, str]]]:
    """Return (kept, dropped) where dropped is a list of (url, reason).

    Rules, in order:
      1. blocked domain
      2. duplicate (already seen this session or earlier in this list)
      3. low relevance
      4. older than ``recency_days`` (0 disables). Unknown dates are kept.

    Low-credibility sources are kept on purpose: the contradiction analysis weighs them
    rather than silently hiding a dissenting view.
    """
    seen = seen_urls if seen_urls is not None else set()
    kept: list[SourceDraft] = []
    dropped: list[tuple[str, str]] = []
    cutoff = today - timedelta(days=recency_days) if recency_days > 0 else None

    for s in sources:
        key = normalize_url(s.url)
        host = urlsplit(key).netloc
        if any(host == d or host.endswith("." + d) for d in blocked_domains):
            dropped.append((s.url, f"blocked domain {host}"))
            continue
        if key in seen:
            dropped.append((s.url, "duplicate"))
            continue
        if s.relevance == "low":
            dropped.append((s.url, "low relevance"))
            continue
        if cutoff is not None:
            published = parse_date(s.published, today)
            if published is not None and published < cutoff:
                dropped.append((s.url, f"outdated ({published.isoformat()} < {cutoff.isoformat()})"))
                continue
        seen.add(key)
        kept.append(s)
    return kept, dropped
