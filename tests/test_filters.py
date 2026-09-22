from datetime import date

from rootlogic.filters import fill_dates_from_hits, filter_sources, normalize_url, parse_date
from rootlogic.models import Credibility, SearchHit, SourceDraft

TODAY = date(2026, 9, 22)


def src(url, published="2026-01-01", relevance="high", cred="high"):
    return SourceDraft(url=url, title="t", published=published, publisher="p", summary="s",
                       key_takeaways=[], credibility=Credibility(level=cred, reason="r"),
                       relevance=relevance)


def test_parse_date_formats():
    assert parse_date("2025-03-04", TODAY) == date(2025, 3, 4)
    assert parse_date("2025-03-04T10:00:00Z", TODAY) == date(2025, 3, 4)
    assert parse_date("March 4, 2025", TODAY) == date(2025, 3, 4)
    assert parse_date("3 days ago", TODAY) == date(2026, 9, 19)
    assert parse_date("2 weeks ago", TODAY) == date(2026, 9, 8)
    assert parse_date("unknown", TODAY) is None
    assert parse_date("sometime", TODAY) is None


def test_normalize_url_strips_noise():
    assert normalize_url("https://WWW.Example.com/a/?utm_source=x&id=2#frag") == \
        "https://example.com/a?id=2"


def test_filter_rules():
    sources = [
        src("https://a.com/1"),
        src("https://a.com/1/"),                              # duplicate
        src("https://b.com/old", published="2020-01-01"),     # outdated
        src("https://c.com/x", relevance="low"),              # irrelevant
        src("https://spam.io/x"),                             # blocked
        src("https://d.com/nodate", published="unknown"),     # kept: unknown date
        src("https://e.com/weak", cred="low"),                # kept: low credibility is flagged, not hidden
    ]
    kept, dropped = filter_sources(sources, recency_days=365, today=TODAY,
                                   blocked_domains=("spam.io",))
    assert [s.url for s in kept] == ["https://a.com/1", "https://d.com/nodate", "https://e.com/weak"]
    reasons = dict(dropped)
    assert reasons["https://a.com/1/"] == "duplicate"
    assert reasons["https://b.com/old"].startswith("outdated")
    assert reasons["https://c.com/x"] == "low relevance"
    assert reasons["https://spam.io/x"].startswith("blocked")


def test_recency_zero_disables_age_filter():
    kept, _ = filter_sources([src("https://b.com/old", published="1990-01-01")],
                             recency_days=0, today=TODAY)
    assert len(kept) == 1


def test_seen_urls_dedupe_across_calls():
    seen: set[str] = set()
    filter_sources([src("https://a.com/1")], recency_days=0, today=TODAY, seen_urls=seen)
    kept, dropped = filter_sources([src("https://a.com/1")], recency_days=0, today=TODAY,
                                   seen_urls=seen)
    assert not kept and dropped[0][1] == "duplicate"


def test_fill_dates_from_search_hits():
    s = src("https://a.com/1", published="unknown")
    fill_dates_from_hits([s], [SearchHit(url="https://www.a.com/1", title="", page_age="2 days ago")])
    assert s.published == "2 days ago"
