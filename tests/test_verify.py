"""Guardrails: claim verification, corroboration, source policy, report checks, evidence capture."""

from datetime import date
from types import SimpleNamespace

import pytest

from rootlogic.context import findings_block
from rootlogic.fake_llm import FakeLLM
from rootlogic.filters import SourcePolicy, clean_domain, filter_sources
from rootlogic.llm import LLMError, _search_hits
from rootlogic.models import (CheckedClaim, ClaimDraft, ClaimVerdictDraft, Credibility, Finding,
                              FindingDraft,
                              Plan, ReportDraft, SearchHit, SourceDraft, VerificationDraft)
from rootlogic.store import Store
from rootlogic.verify import (FETCH_FLOOR, check_report, corroborate, evidence_from_hits,
                              quote_in, registrable_domain, relevant_excerpt, verify_findings)

TODAY = date(2026, 9, 22)


def src(url, cred="high", relevance="high"):
    return SourceDraft(url=url, title=url, published="2026-01-01", publisher="p", summary="s",
                       key_takeaways=[], credibility=Credibility(level=cred, reason="r"),
                       relevance=relevance)


def finding(task_id, claims, sources):
    return Finding(task_id=task_id, question=f"q {task_id}", answer="a", sources=sources,
                   claims=[ClaimDraft(text=t, source_urls=u) for t, u in claims], gaps=[],
                   confidence="high")


PAGE = ("The survey was published in March by the national statistics agency. Unemployment fell "
        "to 3.9 percent in 2025, the agency said, the lowest rate recorded since the series "
        "began. Regional breakdowns, the sampling frame and revisions to earlier quarters are "
        "set out in the accompanying tables and methodology notes.")


def verdicts(*items):
    """Handler returning fixed verdicts: (claim_number, verdict, quote[, quote_source_url])."""
    return {VerificationDraft: lambda p: VerificationDraft(checks=[
        ClaimVerdictDraft(claim_number=i[0], verdict=i[1], quote=i[2],
                          quote_source_url=i[3] if len(i) > 3 else "", note="n")
        for i in items])}


# ------------------------------------------------------------------ verification


def test_supported_claim_needs_a_real_verbatim_quote():
    f = finding("t1", [("Unemployment fell to 3.9% in 2025", ["https://a.gov/r"]),
                       ("Wages doubled", ["https://a.gov/r"])], [src("https://a.gov/r")])
    llm = FakeLLM(handlers=verdicts(
        (1, "supported", "Unemployment fell to 3.9 percent in 2025"),       # really in the page
        (2, "supported", "wages doubled across the whole economy")))       # invented quote
    result = verify_findings([f], llm=llm, evidence={"https://a.gov/r": PAGE})

    c1, c2 = f.checks
    assert c1.verdict == "supported" and c1.quote
    assert c2.verdict == "partially_supported" and c2.quote == ""
    assert c2.note.startswith("Downgraded")
    assert result.counts == {"supported": 1, "partially_supported": 1}


def test_claims_without_page_text_are_unverifiable_not_trusted():
    f = finding("t1", [("x happened", ["https://nowhere.org/a"])], [src("https://nowhere.org/a")])
    llm = FakeLLM()
    verify_findings([f], llm=llm, evidence={})
    assert f.checks[0].verdict == "unverifiable"
    assert not any(p.startswith("verify") for p, _ in llm.calls)  # no pointless LLM call


def test_missing_evidence_is_fetched_within_budget():
    fetched = []
    f = finding("t1", [(f"claim {i}", [f"https://s{i}.org/p"]) for i in range(4)],
                [src(f"https://s{i}.org/p") for i in range(4)])

    def fetch(url):
        fetched.append(url)
        return ("claim text here and more words for the quote, padded out to the length of a "
                "real page so it counts as usable evidence rather than navigation boilerplate. "
                "Methodology and further detail follow below in the appendix of this report.")

    verify_findings([f], llm=FakeLLM(), evidence={}, fetch=fetch, max_fetches=2)
    assert len(fetched) == 2
    assert [c.verdict for c in f.checks].count("unverifiable") == 2


def test_claim_budget_is_spread_across_findings_and_rest_is_unchecked():
    fs = [finding(f"t{i}", [(f"c{i}{j}", [f"https://a{i}.org"]) for j in range(3)],
                  [src(f"https://a{i}.org")]) for i in range(3)]
    page = ("c00 c01 c02 c10 c11 c12 c20 c21 c22 evidence page with enough surrounding text to "
            "count as real content rather than a navigation stub, including methodology notes "
            "and a description of how the figures were collected and weighted.")
    evidence = {f"https://a{i}.org": page for i in range(3)}
    verify_findings(fs, llm=FakeLLM(), evidence=evidence, max_claims=4)
    checked_per_finding = [sum(c.verdict != "unchecked" for c in f.checks) for f in fs]
    assert sorted(checked_per_finding) == [1, 1, 2]


def test_already_checked_findings_are_skipped():
    f = finding("p1", [("old", ["https://a.org"])], [src("https://a.org")])
    f.checks = [CheckedClaim(text="old", source_urls=["https://a.org"], verdict="supported")]
    llm = FakeLLM()
    verify_findings([f], llm=llm, evidence={"https://a.org": "old page"})
    assert f.checks[0].verdict == "supported" and not llm.calls


def test_verifier_failure_leaves_claims_unchecked():
    def boom(prompt):
        raise LLMError("down")

    f = finding("t1", [("c", ["https://a.org"])], [src("https://a.org")])
    events = []
    verify_findings([f], llm=FakeLLM(handlers={VerificationDraft: boom}),
                    evidence={"https://a.org": "c page " * 40},
                    emit=lambda t, m, **k: events.append(t))
    assert f.checks[0].verdict == "unchecked" and events == ["verify.skipped"]


def test_quote_matching_ignores_whitespace_quotes_and_case():
    assert quote_in('"Unemployment  fell to 3.9 PERCENT"', [PAGE])
    assert not quote_in("fell", [PAGE])                     # too short to prove anything
    assert not quote_in("Unemployment fell to 4.9 percent", [PAGE])


def test_relevant_excerpt_picks_matching_chunks():
    page = ("filler " * 400) + "The dam holds 40 billion cubic metres of water. " + ("filler " * 400)
    excerpt = relevant_excerpt(page, "The dam holds 40 billion cubic metres")
    assert "40 billion cubic metres" in excerpt and len(excerpt) < len(page)


# ------------------------------------------------------------------ corroboration


@pytest.mark.parametrize("urls,creds,label", [
    (["https://a.com/1", "https://b.org/2"], ["high", "medium"], "corroborated"),
    (["https://news.bbc.co.uk/1", "https://www.bbc.co.uk/2"], ["high", "high"], "single_source"),
    (["https://a.com/1", "https://blog.io/2"], ["high", "low"], "single_source"),
    (["https://blog.io/1", "https://rumor.net/2"], ["low", "low"], "weak"),
    (["https://not-in-sources.com"], [], "none"),
])
def test_corroboration_labels(urls, creds, label):
    sources = {u: src(u, c) for u, c in zip(urls, creds)}
    claim = CheckedClaim(text="c", source_urls=urls)
    corroborate(claim, sources)
    assert claim.corroboration == label


def test_registrable_domain():
    assert registrable_domain("news.bbc.co.uk") == "bbc.co.uk"
    assert registrable_domain("www.nytimes.com") == "nytimes.com"
    assert registrable_domain("a.b.example.org") == "example.org"


# ------------------------------------------------------------------ prompts see the verdicts


def test_writer_sees_labels_and_failed_claims_are_quarantined():
    f = finding("t1", [("good claim", ["https://a.com"]), ("bad claim", ["https://a.com"])],
                [src("https://a.com")])
    f.checks = [CheckedClaim(text="good claim", source_urls=["https://a.com"], verdict="supported",
                             corroboration="single_source", domains=["a.com"]),
                CheckedClaim(text="bad claim", source_urls=["https://a.com"],
                             verdict="unsupported", note="page says otherwise")]
    plan = Plan(topic="t", objective="o", recency_days=0, subtasks=[])
    block = findings_block(plan, [f], [])
    main, failed = block.split("FAILED verification")
    assert "- good claim [1] (verified; single source)" in main and "bad claim" not in main
    assert "- bad claim [1] (page says otherwise)" in failed


# ------------------------------------------------------------------ report checks


def test_check_report_fixes_bad_citations_and_flags_uncited_and_weak():
    sources = [src("https://a.com"), src("https://blog.io", "low")]
    draft = ReportDraft(title="T", executive_summary="s",
                        key_takeaways=["Sales rose 20% [1].", "It is rumoured [2].", "Bad ref [7]."],
                        body_markdown="## Findings\n\nRevenue grew 12 percent in 2025. "
                                      "This is cited [1]. What next?\n\n| a | b |",
                        open_questions=[], related_topics=[])
    checks = [CheckedClaim(text="x", source_urls=[], verdict="supported",
                           corroboration="corroborated"),
              CheckedClaim(text="y", source_urls=[], verdict="unsupported",
                           corroboration="single_source")]
    q = check_report(draft, sources, checks, policy_drops=2)

    assert q.invalid_citations == ["[7]"] and draft.key_takeaways[2] == "Bad ref [?]."
    assert q.uncited_statements == ["Revenue grew 12 percent in 2025."]
    assert q.weak_takeaways == ["It is rumoured [2]."]
    assert q.claims == {"supported": 1, "unsupported": 1} and q.supported_ratio == 0.5
    assert q.policy_drops == 2


def test_report_markdown_includes_confidence_and_claim_table(tmp_path):
    from rootlogic.fake_llm import FakeLLM
    from rootlogic.orchestrator import Orchestrator

    from .test_orchestrator import ScriptedUI
    report = Orchestrator(FakeLLM(), Store(), ScriptedUI(), reports_dir=tmp_path,
                          today=TODAY).run("impact of generative AI on newsrooms")
    md = report.to_markdown()
    assert "## Confidence and limitations" in md and "## Claim check" in md
    row = next(ln for ln in md.splitlines() if ln.startswith("| Figures rose year over year."))
    assert "supported" in row and "single source" in row and "[1]*" in row
    assert report.quality.claims == {"supported": 3, "unverifiable": 3}


def test_no_verify_budget_skips_the_stage(tmp_path):
    from rootlogic.orchestrator import Budget, Orchestrator

    from .test_orchestrator import ScriptedUI
    llm = FakeLLM()
    report = Orchestrator(llm, Store(), ScriptedUI(), reports_dir=tmp_path, today=TODAY,
                          budget=Budget(verify_claims=0)).run("impact of generative AI on newsrooms")
    assert not any(p.startswith("verify") for p, _ in llm.calls)
    assert report.quality.claims == {"unchecked": 6}


# ------------------------------------------------------------------ source policy


def test_policy_block_allow_trust_distrust():
    policy = SourcePolicy.from_rules(
        [{"domain": "spam.io", "rule": "block"}, {"domain": "who.int", "rule": "trust"},
         {"domain": "tabloid.co.uk", "rule": "distrust"}],
        block=("https://www.junk.net/x",))
    sources = [src("https://spam.io/a"), src("https://junk.net/b"), src("https://who.int/c", "low"),
               src("https://www.tabloid.co.uk/d", "high"), src("https://neutral.org/e")]
    kept, dropped = filter_sources(sources, recency_days=0, today=TODAY, policy=policy)

    assert [s.url for s in kept] == ["https://who.int/c", "https://www.tabloid.co.uk/d",
                                     "https://neutral.org/e"]
    assert all("your source rules" in r for _, r in dropped)
    assert kept[0].credibility.level == "high" and kept[1].credibility.level == "low"

    only = SourcePolicy.from_rules([], only=("who.int",))
    kept, dropped = filter_sources([src("https://who.int/a"), src("https://x.org/b")],
                                   recency_days=0, today=TODAY, policy=only)
    assert [s.url for s in kept] == ["https://who.int/a"]
    assert "not on your allowlist" in dropped[0][1]
    assert "use ONLY these sites: who.int" in only.describe()


def test_store_source_rules_one_rule_per_domain():
    store = Store()
    store.set_source_rule("who.int", "trust")
    store.set_source_rule("who.int", "block")
    assert store.source_rules() == [{"domain": "who.int", "rule": "block",
                                     "created_at": store.source_rules()[0]["created_at"]}]
    assert store.remove_source_rule("who.int") and not store.remove_source_rule("who.int")
    assert clean_domain("HTTPS://www.Who.int/path") == "who.int"


def test_engine_applies_saved_rules_and_tells_the_agents(tmp_path):
    from rootlogic.orchestrator import Orchestrator

    from .test_orchestrator import ScriptedUI
    store = Store()
    store.set_source_rule("news.example.com", "block")
    llm = FakeLLM()
    ui = ScriptedUI()
    report = Orchestrator(llm, store, ui, reports_dir=tmp_path, today=TODAY).run(
        "impact of generative AI on newsrooms")
    assert not any("news.example.com" in s.url for s in report.sources)
    assert "never use: news.example.com" in next(p for pu, p in llm.calls if pu == "research:t1")
    assert report.quality.policy_drops == 3 and "policy.loaded" in ui.types()


# ------------------------------------------------------------------ evidence capture


def test_server_web_fetch_results_become_evidence():
    doc = SimpleNamespace(source=SimpleNamespace(type="text", data="page body"), title="T")
    blocks = [SimpleNamespace(type="web_fetch_tool_result", content=SimpleNamespace(
                  type="web_fetch_result", url="https://a.com/x", content=doc)),
              SimpleNamespace(type="web_fetch_tool_result",
                              content=SimpleNamespace(type="web_fetch_tool_error")),
              SimpleNamespace(type="web_fetch_tool_result", content=SimpleNamespace(
                  type="web_fetch_result", url="https://a.com/pdf", content=SimpleNamespace(
                      source=SimpleNamespace(type="base64", data="JVBER"))))]
    hits = _search_hits(blocks)
    assert [(h.url, h.text) for h in hits] == [("https://a.com/x", "page body")]
    assert evidence_from_hits(hits + [SearchHit(url="https://b.com", title="")]) == \
        {"https://a.com/x": "page body"}


def test_unreadable_pages_are_unverifiable_never_unsupported():
    """Live run bug: a landing page of navigation text marked 7 true claims 'unsupported'.

    Failing to read a page is not evidence against a claim.
    """
    f = finding("t1", [("64% cited back-end automation", ["https://inst.org/report"])],
                [src("https://inst.org/report")])
    llm = FakeLLM(handlers=verdicts((1, "no_usable_evidence", "", )))
    verify_findings([f], llm=llm, evidence={"https://inst.org/report": "Home About Subscribe " * 20})
    assert f.checks[0].verdict == "unverifiable"


def test_navigation_sized_evidence_is_not_even_sent_to_the_verifier():
    f = finding("t1", [("a claim", ["https://a.org/x"])], [src("https://a.org/x")])
    llm = FakeLLM()
    verify_findings([f], llm=llm, evidence={"https://a.org/x": "Skip to content. Menu."})
    assert f.checks[0].verdict == "unverifiable" and not llm.calls


# ------------------------------------------------------------------ retrying dead sub-tasks


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_subtask_that_retrieves_nothing_is_retried_then_failed(tmp_path, kind):
    """Live run: two sub-tasks came back with zero sources after a search outage, were marked
    'done', and the task cap then blocked the critic from re-running them."""
    from rootlogic.fake_llm import FakeLLM, default_finding
    from rootlogic.graph import ResearchGraph
    from rootlogic.orchestrator import Budget, Orchestrator

    from .test_orchestrator import ScriptedUI

    attempts = {"n": 0}

    def flaky(prompt):
        if "current state" not in prompt:
            return default_finding(prompt, 2)
        attempts["n"] += 1
        empty = default_finding(prompt, 1)
        empty.sources, empty.claims = [], []
        empty.gaps = ["web search returned nothing"]
        return empty

    store = Store()
    ui = ScriptedUI()
    kw = dict(budget=Budget(max_retries=1), reports_dir=tmp_path, today=TODAY)
    llm = FakeLLM(handlers={FindingDraft: flaky})
    engine = (ResearchGraph(llm, store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(llm, store, ui, **kw))
    report = engine.run("impact of generative AI on newsrooms")

    assert attempts["n"] == 2                       # one retry, then given up on
    types = ui.types()
    assert "task.retry" in types and "task.failed" in types
    assert {t["task_id"]: t["status"] for t in store.tasks(engine.sid)}["t1"] == "failed"
    assert report is not None                       # the other sub-tasks still produced a report


# ------------------------------------------------------------------ fetching cited pages


def test_page_fetcher_returns_text_and_swallows_failures():
    """Live crash: the fetcher was a lambda whose walrus ran after the condition that read it
    (UnboundLocalError on the first cited page missing from evidence)."""
    from rootlogic.search import SearchError, StaticSearch
    from rootlogic.verify import page_fetcher

    assert page_fetcher(None) is None

    fetch = page_fetcher(StaticSearch(pages={"https://a.com": "page text"}))
    assert fetch("https://a.com") == "page text"
    assert fetch("https://missing.example") is None       # provider reported an error

    class Broken(StaticSearch):
        def fetch(self, url):
            raise SearchError("provider down")

    assert page_fetcher(Broken())("https://a.com") is None   # a dead provider is not fatal


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_verification_fetches_a_cited_page_the_sub_agents_did_not_keep(tmp_path, kind):
    """End to end with a search provider attached, which is when the fetcher is built at all:
    the Claude path has search=None, so every mocked test skipped this code."""
    from rootlogic.fake_llm import FakeLLM
    from rootlogic.graph import ResearchGraph
    from rootlogic.orchestrator import Orchestrator
    from rootlogic.search import StaticSearch
    from rootlogic.store import Store
    from tests.test_orchestrator import TODAY, ScriptedUI

    # A provider holding the pages the fake sub-agents cite but never fetched themselves.
    pages = {f"https://example.org/report-{n}": PAGE for n in range(1, 5)}

    class Searching(FakeLLM):
        search = StaticSearch(pages=pages)

    store, ui = Store(), ScriptedUI()
    kw = dict(reports_dir=tmp_path, today=TODAY)
    engine = (ResearchGraph(Searching(), store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(Searching(), store, ui, **kw))

    report = engine.run("impact of generative AI on newsrooms")

    assert report is not None
    assert store.session(engine.sid)["status"] == "done"
    assert "verify.done" in ui.types()


def test_report_sources_show_one_date_format():
    """Providers mix ISO and RFC 1123 in the same run; the sources list shouldn't."""
    from rootlogic.models import display_date

    assert display_date("Thu, 19 Mar 2026 00:00:00 GMT") == "2026-03-19"
    assert display_date("2026-03-17") == "2026-03-17"
    assert display_date("sometime in spring") == "sometime in spring"   # shown as it came


# ------------------------------------------------------------------ evidence quality


def test_a_numeric_claim_needs_its_number_inside_the_quote():
    """AttrScore documents numbers as a blind spot of attribution checkers: a quote from the
    right page, about the right topic, with the wrong figure still reads as support."""
    f = finding("t1", [("Unemployment fell to 7.2% in 2025", ["https://a.gov/r"])],
                [src("https://a.gov/r")])
    llm = FakeLLM(handlers=verdicts(
        (1, "supported", "Unemployment fell to 3.9 percent in 2025")))   # real quote, wrong number
    verify_findings([f], llm=llm, evidence={"https://a.gov/r": PAGE})

    check = f.checks[0]
    assert check.verdict == "partially_supported"
    assert "7.2" in check.note


def test_a_numeric_claim_passes_when_the_number_is_in_the_quote():
    f = finding("t1", [("Unemployment fell to 3.9% in 2025", ["https://a.gov/r"])],
                [src("https://a.gov/r")])
    llm = FakeLLM(handlers=verdicts((1, "supported", "Unemployment fell to 3.9 percent in 2025")))
    verify_findings([f], llm=llm, evidence={"https://a.gov/r": PAGE})
    assert f.checks[0].verdict == "supported"


def test_a_quote_is_attributed_to_the_single_page_it_came_from():
    """ALCE measures citation precision per citation. With several pages concatenated, a claim
    citing three sources looks supported when only one of them says anything."""
    other = ("An unrelated page about transport funding and timetables, long enough to count as "
             "real page content rather than navigation boilerplate. " * 3)
    f = finding("t1", [("Unemployment fell to 3.9% in 2025",
                        ["https://b.org/x", "https://a.gov/r"])],
                [src("https://b.org/x"), src("https://a.gov/r")])
    llm = FakeLLM(handlers=verdicts((1, "supported", "Unemployment fell to 3.9 percent in 2025")))

    verify_findings([f], llm=llm,
                    evidence={"https://b.org/x": other, "https://a.gov/r": PAGE})

    check = f.checks[0]
    assert check.verdict == "supported"
    assert check.quote_url == "https://a.gov/r"     # the page the quote is actually in


def test_a_claim_that_cannot_be_checked_is_separated_from_an_unreadable_page():
    """VeriScore distinguishes 'no evidence found' from 'not a checkable claim'. Reporting an
    opinion as 'unverifiable (page text unavailable)' blames the fetcher for a category error."""
    f = finding("t1", [("This is the most exciting development in economics",
                        ["https://a.gov/r"])], [src("https://a.gov/r")])
    llm = FakeLLM(handlers=verdicts((1, "not_a_factual_claim", "")))
    verify_findings([f], llm=llm, evidence={"https://a.gov/r": PAGE})

    check = f.checks[0]
    assert check.verdict == "unverifiable"
    assert check.unverifiable_reason == "not_a_factual_claim"


def test_an_unreadable_page_is_labelled_as_such():
    f = finding("t1", [("Unemployment fell to 3.9% in 2025", ["https://a.gov/r"])],
                [src("https://a.gov/r")])
    llm = FakeLLM(handlers=verdicts((1, "no_usable_evidence", "")))
    verify_findings([f], llm=llm, evidence={"https://a.gov/r": PAGE})

    check = f.checks[0]
    assert check.verdict == "unverifiable" and check.unverifiable_reason == "page_unreadable"


def test_limitations_line_says_why_claims_were_unverifiable():
    from rootlogic.models import CheckedClaim, ReportDraft
    from rootlogic.verify import check_report

    checks = [CheckedClaim(text="a", source_urls=[], verdict="unverifiable",
                           unverifiable_reason="page_unreadable"),
              CheckedClaim(text="b", source_urls=[], verdict="unverifiable",
                           unverifiable_reason="not_a_factual_claim"),
              CheckedClaim(text="c", source_urls=[], verdict="supported")]
    draft = ReportDraft(title="t", executive_summary="s", key_takeaways=[], body_markdown="b",
                        open_questions=[], related_topics=[])

    quality = check_report(draft, [src("https://a.gov/r")], checks)

    assert quality.unverifiable_reasons == {"page_unreadable": 1, "not_a_factual_claim": 1}
    assert quality.unverifiable_detail == "1 page text unavailable, 1 not a checkable claim"


def test_claim_table_marks_the_source_the_quote_came_from(tmp_path):
    """"Which source actually backs this?" should be answerable at a glance: the cited source
    carrying the verified quote is marked, not just listed among the claim's citations."""
    from rootlogic.models import (Analysis, CheckedClaim, Report, ReportDraft,
                                  ReportQuality)

    checks = [CheckedClaim(text="Unemployment fell to 3.9% in 2025",
                           source_urls=["https://b.org/x", "https://a.gov/r"],
                           verdict="supported", quote="Unemployment fell to 3.9 percent",
                           quote_url="https://a.gov/r", corroboration="single_source")]
    draft = ReportDraft(title="t", executive_summary="s", key_takeaways=[], body_markdown="b",
                        open_questions=[], related_topics=[])
    report = Report(session_id="s", draft=draft,
                    analysis=Analysis(consensus=[], contradictions=[]),
                    sources=[src("https://b.org/x"), src("https://a.gov/r")], checks=checks,
                    quality=ReportQuality(claims={"supported": 1}))

    table = [ln for ln in report.to_markdown().splitlines() if "Unemployment fell" in ln]
    assert table and "[2]*" in table[0]      # source 2 carries the quote; source 1 does not
    assert "[1]" in table[0]


def test_a_claim_with_no_page_text_records_why_it_was_unverifiable():
    """The 'no usable page text' path set the verdict but no reason, so the limitations line
    said '3 unverifiable' with no explanation."""
    f = finding("t1", [("Wages doubled", ["https://a.gov/r"])], [src("https://a.gov/r")])
    verify_findings([f], llm=FakeLLM(), evidence={})       # nothing fetched, nothing cached
    assert f.checks[0].verdict == "unverifiable"
    assert f.checks[0].unverifiable_reason == "page_unreadable"


# ------------------------------------------------------------------ numeric matching


@pytest.mark.parametrize("claim,quote,missing", [
    # the genuine catch: the claim's figure appears nowhere in the quote
    ("Unemployment fell to 7.2% in 2025", "Unemployment fell to 3.9 percent in 2025", {"7.2"}),
    # one substantive figure is enough; an incidental date need not be repeated
    ("In July 2026 the extent was 15.39 million km2", "the extent was 15.39 million km2", set()),
    # bare years are context, not the claim's evidence
    ("The WHO has not released coverage data for 2026", "no data has been published", set()),
    # same value, different spelling
    ("Growth reached 3.90 million", "growth reached 3.9 million", set()),
    ("Growth reached 3.9 million", "growth reached 3.90 million", set()),
    # no figures at all: nothing to check
    ("Adoption is widespread", "adoption is widespread across newsrooms", set()),
])
def test_numeric_matching_catches_wrong_figures_without_false_alarms(claim, quote, missing):
    """Live runs produced three false downgrades: an incidental date, a negative claim about a
    year, and 3.90 vs 3.9. Requiring every figure was too strict to be useful."""
    from rootlogic.verify import _missing_figures

    assert _missing_figures(claim, quote) == missing


def test_a_claim_whose_only_figure_is_a_year_still_checks_that_year():
    """A year can be the substance: then it is the only figure there is to check."""
    from rootlogic.verify import _missing_figures

    assert _missing_figures("The telescope launched in 2021", "it launched in 2019") == {"2021"}
    assert _missing_figures("The telescope launched in 2021", "it launched in 2021") == set()


def test_fetch_budget_follows_the_claim_budget():
    """A claim whose page was never fetched scores unverifiable, so a bigger claim budget
    against a fixed six fetches would just make a bigger pile of unverifiable claims."""
    claims = [(f"Claim {i}", [f"https://a.gov/{i}"]) for i in range(20)]
    f = finding("t1", claims, [src(f"https://a.gov/{i}") for i in range(20)])
    fetched = []

    def fetch(url):
        fetched.append(url)
        return PAGE

    llm = FakeLLM(handlers=verdicts(*[(i + 1, "supported", "Unemployment fell to 3.9 percent "
                                       "in 2025") for i in range(20)]))
    verify_findings([f], llm=llm, evidence={}, fetch=fetch, max_claims=20)
    assert len(fetched) == 10                       # half the claim budget, not six

    fetched.clear()
    verify_findings([finding("t2", claims, [src(f"https://a.gov/{i}") for i in range(20)])],
                    llm=llm, evidence={}, fetch=fetch, max_claims=8)
    assert len(fetched) == FETCH_FLOOR              # small budgets keep the floor, not half
