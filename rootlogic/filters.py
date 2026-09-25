"""Deterministic source filtering: recency, relevance, credibility, de-duplication.

Kept free of LLM calls so the rules are unit-testable and explainable in a demo:
the model *proposes* sources, code *decides* which ones survive, and every drop is
logged with a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

from .models import Credibility, SearchHit, SourceDraft

RULES = ("block", "allow", "trust", "distrust")


def host_of(url: str) -> str:
    return urlsplit(normalize_url(url)).netloc


def domain_matches(host: str, domains) -> bool:
    """True if host is one of ``domains`` or a subdomain of one."""
    return any(host == d or host.endswith("." + d) for d in domains)


def clean_domain(text: str) -> str:
    """'https://www.Example.com/path' -> 'example.com'."""
    text = text.strip().lower()
    if "//" not in text:
        text = "https://" + text
    return urlsplit(text).netloc.removeprefix("www.")


@dataclass(frozen=True)
class SourcePolicy:
    """The user's source rules. ``allowed`` non-empty means allowlist mode."""
    blocked: frozenset[str] = field(default_factory=frozenset)
    allowed: frozenset[str] = field(default_factory=frozenset)
    trusted: frozenset[str] = field(default_factory=frozenset)
    distrusted: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_rules(cls, rules: list[dict], *, block: tuple[str, ...] = (),
                   only: tuple[str, ...] = ()) -> SourcePolicy:
        by = {r: {x["domain"] for x in rules if x["rule"] == r} for r in RULES}
        return cls(blocked=frozenset(by["block"] | {clean_domain(d) for d in block}),
                   allowed=frozenset(by["allow"] | {clean_domain(d) for d in only}),
                   trusted=frozenset(by["trust"]), distrusted=frozenset(by["distrust"]))

    @property
    def empty(self) -> bool:
        return not (self.blocked or self.allowed or self.trusted or self.distrusted)

    def describe(self) -> str:
        """One context line for prompts, so sub-agents search accordingly."""
        parts = []
        if self.allowed:
            parts.append("use ONLY these sites: " + ", ".join(sorted(self.allowed)))
        if self.blocked:
            parts.append("never use: " + ", ".join(sorted(self.blocked)))
        if self.trusted:
            parts.append("trusted: " + ", ".join(sorted(self.trusted)))
        if self.distrusted:
            parts.append("treat as unreliable: " + ", ".join(sorted(self.distrusted)))
        return "User source rules: " + "; ".join(parts) if parts else ""

    def verdict(self, url: str) -> str | None:
        """Drop reason under this policy, or None to keep."""
        host = host_of(url)
        if domain_matches(host, self.blocked):
            return f"blocked by your source rules ({host})"
        if self.allowed and not domain_matches(host, self.allowed):
            return f"not on your allowlist ({host})"
        return None

    def adjust(self, source: SourceDraft) -> None:
        """Trusted/distrusted sites override the model's credibility rating."""
        host = host_of(source.url)
        if domain_matches(host, self.distrusted):
            source.credibility = Credibility(level="low", reason="You marked this site unreliable")
        elif domain_matches(host, self.trusted):
            source.credibility = Credibility(level="high", reason="You marked this site trusted")


_RELATIVE = re.compile(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", re.I)
_UNIT_DAYS = {"minute": 0, "hour": 0, "day": 1, "week": 7, "month": 30, "year": 365}
_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%B %Y")


def parse_date(value: str | None, today: date) -> date | None:
    """Best-effort parse of ISO, RFC 1123, 'March 3, 2025' or '3 days ago'. None if unknown."""
    if not value or value.strip().lower() in ("unknown", "n/a", "none"):
        return None
    text = value.strip()
    if m := _RELATIVE.search(text):
        return today - timedelta(days=int(m.group(1)) * _UNIT_DAYS[m.group(2).lower()])
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    # RFC 1123, e.g. "Tue, 24 Mar 2026 04:00:00 GMT": what Tavily returns for many pages.
    # Missed live, and a date that won't parse means the outdated filter never fires.
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError):
        pass
    for fmt in _FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# Bare enough to catch a pasted link, strict enough not to swallow the sentence around it.
_URL = re.compile(r"https?://[^\s<>\"'\]]+", re.I)


def urls_in(text: str) -> list[str]:
    """Links the user typed, in order, without repeats.

    Trailing punctuation is part of the sentence rather than the address: "see https://x.org."
    ends in a full stop. Brackets are counted rather than banned, because Wikipedia is full of
    addresses like ``/wiki/Mercury_(planet)``: a closing bracket is only dropped when nothing
    in the URL opened it, which is the markdown case ``(https://x.org/a)``.
    """
    found: list[str] = []
    for raw in _URL.findall(text or ""):
        url = raw.rstrip(".,;:!?'\"")
        while url.endswith(")") and url.count("(") < url.count(")"):
            url = url[:-1]
        if url not in found:
            found.append(url)
    return found


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
    policy: SourcePolicy | None = None,
) -> tuple[list[SourceDraft], list[tuple[str, str]]]:
    """Return (kept, dropped) where dropped is a list of (url, reason).

    Rules, in order:
      1. blocked domain, or not on the allowlist (``policy``)
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
        if domain_matches(host, blocked_domains):
            dropped.append((s.url, f"blocked domain {host}"))
            continue
        if policy is not None and (reason := policy.verdict(s.url)):
            dropped.append((s.url, reason))
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
        if policy is not None:
            policy.adjust(s)
        seen.add(key)
        kept.append(s)
    return kept, dropped
