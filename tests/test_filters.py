from datetime import date

import pytest

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


def test_plan_ids_stay_unique_after_drop():
    from rootlogic.models import Plan, SubTask, SubTaskDraft
    plan = Plan(topic="x", objective="o", recency_days=0, subtasks=[
        SubTask(id=f"t{i}", question=f"q{i}", rationale="r", search_queries=[]) for i in (1, 2, 3)])
    plan.subtasks = [t for t in plan.subtasks if t.id != "t2"]
    new = plan.add(SubTaskDraft(question="new", rationale="r", search_queries=[], depends_on=[]),
                   "user")
    assert new.id == "t4"
    assert len({t.id for t in plan.subtasks}) == len(plan.subtasks)


@pytest.mark.parametrize("value,expected", [
    ("Tue, 24 Mar 2026 04:00:00 GMT", date(2026, 3, 24)),     # Tavily's published_date
    ("Thu, 19 Mar 2026 00:00:00 GMT", date(2026, 3, 19)),
    ("Mon, 01 Jan 2019 00:00:00 -0500", date(2019, 1, 1)),    # with a numeric offset
    ("24 Mar 2026 04:00:00 GMT", date(2026, 3, 24)),          # no weekday
])
def test_rfc_1123_dates_are_parsed(value, expected):
    """Live run: Tavily returned RFC-1123 dates, parse_date gave None for every one of them,
    and the outdated filter silently kept everything."""
    assert parse_date(value, date(2026, 9, 22)) == expected


def test_outdated_sources_are_dropped_when_dated_rfc_1123():
    """The filter must act on those dates, not just parse them."""
    old = src("https://old.example/a", published="Mon, 01 Jan 2019 00:00:00 GMT")
    kept, dropped = filter_sources([old], recency_days=365, today=TODAY)
    assert kept == [] and "outdated (2019-01-01" in dropped[0][1]


def test_urls_in_reads_links_out_of_a_sentence():
    """A topic is prose with links in it, not a list of addresses."""
    from rootlogic.filters import urls_in

    assert urls_in("no links here") == []
    assert urls_in("see https://x.org/a.") == ["https://x.org/a"]        # the stop is the sentence's
    assert urls_in("[x](https://y.org/p) end") == ["https://y.org/p"]    # markdown's bracket
    # Wikipedia is full of these, so brackets are counted rather than banned
    assert urls_in("see https://en.wikipedia.org/wiki/Mercury_(planet) now") == \
        ["https://en.wikipedia.org/wiki/Mercury_(planet)"]
    assert urls_in("a https://e.org b https://e.org") == ["https://e.org"]   # once each
    assert urls_in("http://a.org, https://b.org.") == ["http://a.org", "https://b.org"]


def test_the_age_window_gives_way_when_it_would_take_every_source():
    """Live on walking and type 2 diabetes: the planner picked 730 days, and one sub-task lost
    all three of its sources for being a few months too old - peer-reviewed 2024 papers from
    PubMed and the BJSM - while a 2023 PDF survived because its date would not parse. The
    window is a guess made before any source is seen; when it is the only thing that removed
    everything, it was the guess that was wrong."""
    from datetime import date

    from rootlogic.context import curate
    from rootlogic.models import Credibility, FindingDraft, SubTask

    def source(url, published):
        return SourceDraft(url=url, title="t", published=published, publisher="p", summary="s",
                           key_takeaways=[], relevance="high",
                           credibility=Credibility(level="high", reason="peer reviewed"))

    task = SubTask(id="t1", question="q", rationale="r", search_queries=[])
    draft = FindingDraft(answer="a", claims=[], gaps=[], confidence="medium", sources=[
        source("https://pubmed.ncbi.nlm.nih.gov/38050034", "2024-01-01"),
        source("https://bjsm.bmj.com/content/58/6/334", "2024-01-01")])

    finding = curate(task, draft, [], recency_days=730, today=date(2026, 9, 25), seen_urls=set())

    assert [s.url for s in finding.sources] == [
        "https://pubmed.ncbi.nlm.nih.gov/38050034", "https://bjsm.bmj.com/content/58/6/334"]
    assert finding.relaxed_recency, "the run should say it set its own rule aside"
    assert not finding.dropped


def test_the_age_window_stands_when_something_else_survived():
    """Relaxing is for a rule that took everything, not for a rule that did its job."""
    from datetime import date

    from rootlogic.context import curate
    from rootlogic.models import Credibility, FindingDraft, SubTask

    def source(url, published):
        return SourceDraft(url=url, title="t", published=published, publisher="p", summary="s",
                           key_takeaways=[], relevance="high",
                           credibility=Credibility(level="high", reason="r"))

    task = SubTask(id="t1", question="q", rationale="r", search_queries=[])
    draft = FindingDraft(answer="a", claims=[], gaps=[], confidence="medium", sources=[
        source("https://old.example/a", "2019-01-01"),
        source("https://new.example/b", "2026-09-01")])

    finding = curate(task, draft, [], recency_days=365, today=date(2026, 9, 25), seen_urls=set())

    assert [s.url for s in finding.sources] == ["https://new.example/b"]
    assert not finding.relaxed_recency
    assert finding.dropped and "outdated" in finding.dropped[0][1]
