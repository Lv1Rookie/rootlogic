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
from rootlogic.verify import (check_report, corroborate, evidence_from_hits, quote_in,
                              registrable_domain, relevant_excerpt, verify_findings)

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
    """Handler returning fixed verdicts: items are (claim_number, verdict, quote)."""
    return {VerificationDraft: lambda p: VerificationDraft(checks=[
        ClaimVerdictDraft(claim_number=n, verdict=v, quote=q, note="n") for n, v, q in items])}


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
    assert "| Figures rose year over year. | supported | single source | [1] |" in md
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
