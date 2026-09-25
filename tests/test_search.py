"""SearchProvider: Tavily request/response mapping and the client-side research tool loop."""

import json
from types import SimpleNamespace

import pytest

from rootlogic.llm import AnthropicLLM
from rootlogic.models import FindingDraft
from rootlogic.search import (SearchError, SearchResult, StaticSearch, TavilySearch, _time_range,
                              clip, get_search_provider)

FINDING = {"answer": "a", "sources": [], "claims": [], "gaps": [], "confidence": "medium"}


# ------------------------------------------------------------------ Tavily (no network)

class FakeTavily(TavilySearch):
    def __init__(self, responses):
        super().__init__(api_key="tvly-test")
        self.responses = responses
        self.sent = []

    def _post(self, path, body):
        self.sent.append((path, body))
        return self.responses[path]


def test_tavily_search_request_and_parsing():
    t = FakeTavily({"/search": {"results": [
        {"url": "https://a.com/x", "title": "A", "content": "snippet", "score": 0.9,
         "published_date": "2026-08-01"},
        {"title": "no url"}]}})
    results = t.search("solar recycling", max_results=50, recency_days=30)

    path, body = t.sent[0]
    assert path == "/search"
    assert body == {"query": "solar recycling", "max_results": 20, "search_depth": "basic",
                    "include_published_date": True, "time_range": "month"}
    assert results == [SearchResult(url="https://a.com/x", title="A", snippet="snippet",
                                    published="2026-08-01")]


@pytest.mark.parametrize("days,bucket", [(0, None), (1, "day"), (7, "week"), (8, "month"),
                                         (365, "year"), (730, None)])
def test_time_range_buckets(days, bucket):
    assert _time_range(days) == bucket


def test_tavily_fetch_success_failure_and_clipping():
    long = "x" * 25_000
    t = FakeTavily({"/extract": {"results": [{"url": "https://a.com", "raw_content": long}]}})
    page = t.fetch("https://a.com")
    assert t.sent[0] == ("/extract", {"urls": ["https://a.com"], "format": "text"})
    assert page.error is None and "[truncated: 5,000 more characters]" in page.text

    t = FakeTavily({"/extract": {"results": [],
                                 "failed_results": [{"url": "https://b.com", "error": "403"}]}})
    assert t.fetch("https://b.com").error == "403"


def test_tavily_requires_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SearchError):
        TavilySearch()
    with pytest.raises(SearchError):
        get_search_provider("tavily")
    assert get_search_provider("anthropic") is None
    with pytest.raises(ValueError):
        get_search_provider("bing")


def test_clip_leaves_short_text_alone():
    assert clip("short") == "short"


# ------------------------------------------------------------------ client-side research loop

def usage():
    return SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                           cache_creation_input_tokens=0, server_tool_use=None)


def tool_use(name, inp, id_):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=id_)


class ScriptedMessages:
    """Replays assistant turns; records every request."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})  # snapshot: list keeps growing
        content = self.turns.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop, usage=usage(),
                               model="claude-opus-5")


def make_llm(turns, search):
    messages = ScriptedMessages(turns)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return AnthropicLLM(lambda u: None, client=client, search=search), messages  # type: ignore


def test_client_tools_loop_searches_fetches_and_submits():
    search = StaticSearch(
        results=[SearchResult(url="https://a.com/r", title="Report", snippet="s",
                              published="2026-07-01")],
        pages={"https://a.com/r": "full text"})
    llm, messages = make_llm([
        [tool_use("web_search", {"query": "q1"}, "u1")],
        [tool_use("web_fetch", {"url": "https://a.com/r"}, "u2")],
        [tool_use("submit_findings", FINDING, "u3")],
    ], search)

    finding, hits = llm.research(purpose="research:t1", system="s", prompt="p",
                                 schema=FindingDraft, recency_days=90)

    assert finding.confidence == "medium"
    assert search.queries == [("q1", 90)] and search.fetched == ["https://a.com/r"]
    assert [(h.url, h.page_age) for h in hits if h.text is None] == [("https://a.com/r",
                                                                     "2026-07-01")]
    assert [(h.url, h.text) for h in hits if h.text] == [("https://a.com/r", "full text")]
    # no Anthropic server tools in the portable path
    tools = messages.calls[0]["tools"]
    assert [t["name"] for t in tools] == ["web_search", "web_fetch", "submit_findings"]
    assert all("type" not in t for t in tools)
    # tool results are sent back as a user message
    search_result = messages.calls[1]["messages"][-1]["content"][0]
    assert search_result["tool_use_id"] == "u1" and not search_result["is_error"]
    assert json.loads(search_result["content"])[0]["url"] == "https://a.com/r"
    fetch_result = messages.calls[2]["messages"][-1]["content"][0]
    assert "full text" in fetch_result["content"]


def test_client_tools_enforce_budgets_and_report_errors():
    search = StaticSearch(results=[])
    llm, messages = make_llm([
        # two searches in parallel with a budget of 1, plus a fetch of an unknown page
        [tool_use("web_search", {"query": "a"}, "u1"), tool_use("web_search", {"query": "b"}, "u2"),
         tool_use("web_fetch", {"url": "https://nope"}, "u3"), tool_use("hack", {}, "u4")],
        [tool_use("submit_findings", FINDING, "u5")],
    ], search)
    llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft,
                 max_searches=1)

    results = messages.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["u1", "u2", "u3", "u4"]  # one message, all ids
    assert [r["is_error"] for r in results] == [False, True, True, True]
    assert "budget used up" in results[1]["content"]
    assert "Unknown tool" in results[3]["content"]
    assert search.queries == [("a", 0)]


def test_nudges_when_model_answers_in_text():
    llm, messages = make_llm([
        [SimpleNamespace(type="text", text="I think...")],
        [tool_use("submit_findings", FINDING, "u1")],
    ], StaticSearch())
    llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft)
    assert "call submit_findings" in messages.calls[1]["messages"][-1]["content"]


def test_engines_pass_plan_recency_to_research(tmp_path):
    from rootlogic.fake_llm import FakeLLM
    from rootlogic.orchestrator import Orchestrator
    from rootlogic.store import Store
    from tests.test_orchestrator import ScriptedUI, TODAY

    seen = []

    class Recording(FakeLLM):
        def research(self, **kw):
            seen.append(kw["recency_days"])
            return super().research(**kw)

    Orchestrator(Recording(), Store(), ScriptedUI(), reports_dir=tmp_path, today=TODAY).run(
        "impact of generative AI on newsrooms")
    assert seen and set(seen) == {365}


# ------------------------------------------------------------------ live progress


def test_client_tool_loop_reports_each_search_and_fetch():
    """A sub-agent is otherwise silent between task.started and task.done. Every search and
    fetch it runs must be reported as it happens, including the ones that fail."""
    search = StaticSearch(
        results=[SearchResult(url="https://a.com/r", title="Report", snippet="s")],
        pages={"https://a.com/r": "full text"})
    llm, _ = make_llm([
        [tool_use("web_search", {"query": "newsroom AI 2026"}, "u1")],
        [tool_use("web_fetch", {"url": "https://a.com/r"}, "u2")],
        [tool_use("web_fetch", {"url": "https://gone.example/404"}, "u3")],
        [tool_use("submit_findings", FINDING, "u4")],
    ], search)

    steps = []
    llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft,
                 on_step=steps.append)

    assert [(s.kind, s.detail, s.results, s.ok) for s in steps] == [
        ("search", "newsroom AI 2026", 1, True),
        ("fetch", "https://a.com/r", 1, True),
        ("fetch", "https://gone.example/404", 0, False),
    ]


def test_a_failing_progress_observer_does_not_fail_the_sub_task():
    """Progress reporting is a UI nicety; research it is watching must still finish."""
    search = StaticSearch(results=[SearchResult(url="https://a.com/r", title="R", snippet="s")])
    llm, _ = make_llm([
        [tool_use("web_search", {"query": "q"}, "u1")],
        [tool_use("submit_findings", FINDING, "u2")],
    ], search)

    def boom(step):
        raise RuntimeError("the UI went away")

    finding, _ = llm.research(purpose="research:t1", system="s", prompt="p",
                              schema=FindingDraft, on_step=boom)
    assert finding.confidence == "medium"


def test_a_failed_search_reports_why_it_failed():
    """Live run: the log said '[t1] Searched "..." — search failed' three times and nothing
    more, so the cause (a rejected Tavily key) was invisible from the action log."""
    from rootlogic.control import step_event
    from rootlogic.search import SearchError

    class Rejecting(StaticSearch):
        def search(self, query, *, max_results=5, recency_days=0):
            raise SearchError("Tavily /search failed: HTTP 401")

    llm, _ = make_llm([
        [tool_use("web_search", {"query": "q"}, "u1")],
        [tool_use("submit_findings", FINDING, "u2")],
    ], Rejecting())

    steps = []
    llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft,
                 on_step=steps.append)

    assert steps[0].ok is False and steps[0].error == "Tavily /search failed: HTTP 401"
    _, message, data = step_event("t1", steps[0])
    assert "search failed: Tavily /search failed: HTTP 401" in message
    assert data["error"] == "Tavily /search failed: HTTP 401"


def test_a_failed_fetch_reports_why_too():
    """A URL the search did return, whose page will not come back."""
    search = StaticSearch(results=[SearchResult(url="https://gone.example", title="A",
                                               snippet="s")])
    llm, _ = make_llm([
        [tool_use("web_search", {"query": "one"}, "u0")],
        [tool_use("web_fetch", {"url": "https://gone.example"}, "u1")],
        [tool_use("submit_findings", FINDING, "u2")],
    ], search)

    steps = []
    llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft,
                 on_step=steps.append)
    fetches = [s for s in steps if s.kind == "fetch"]
    assert fetches[0].error == "not found"


def test_anthropic_client_loop_also_stops_when_budgets_are_spent():
    """The same waste applies to Claude with a SearchProvider: once nothing can be retrieved,
    more tool-armed turns only re-send a growing conversation."""
    search = StaticSearch(results=[SearchResult(url=f"https://a.com/{i}", title="A", snippet="s")
                                   for i in range(4)],
                          pages={f"https://a.com/{i}": "page text " * 80 for i in range(4)})
    llm, messages = make_llm([
        [tool_use("web_search", {"query": "one"}, "u1")],
        [tool_use("web_fetch", {"url": "https://a.com/0"}, "u2")],
        [tool_use("web_fetch", {"url": "https://a.com/1"}, "u3")],
        [tool_use("web_fetch", {"url": "https://a.com/2"}, "u4")],
        [tool_use("submit_findings", FINDING, "u5")],
        [tool_use("web_search", {"query": "again"}, "u6")],
    ], search)

    llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft,
                 max_searches=1)

    assert len(messages.calls) == 5
    last = messages.calls[-1]
    assert [t["name"] for t in last["tools"]] == ["submit_findings"]   # nothing left to search


# ------------------------------------------------------------------ invented URLs

def box(search, **kw):
    from rootlogic.tools import WebToolbox
    return WebToolbox(search, max_searches=3, **kw)


def test_a_url_no_search_returned_is_refused_without_spending_a_fetch():
    """Live: a run spent 7 of 12 fetches on gov.uk slugs the model built from headlines, all
    404. The prompt asks it not to; this makes the ask enforceable."""
    tb = box(StaticSearch(results=[SearchResult(url="https://a.gov/real", title="A", snippet="s")],
                          pages={"https://a.gov/real": "text " * 40}))
    tb.run("web_search", {"query": "q"})

    content, is_error = tb.run("web_fetch", {"url": "https://a.gov/guidance/invented-slug"})

    assert is_error and "has not appeared in any search result" in content
    assert "Search for the document instead" in content
    assert tb.used["web_fetch"] == 0          # a refusal bought nothing, so it costs nothing


def test_a_url_the_search_returned_is_fetched():
    search = StaticSearch(results=[SearchResult(url="https://a.gov/real", title="A", snippet="s")],
                          pages={"https://a.gov/real": "real page text"})
    tb = box(search)
    tb.run("web_search", {"query": "q"})

    content, is_error = tb.run("web_fetch", {"url": "https://a.gov/real"})

    assert not is_error and "real page text" in content
    assert search.fetched == ["https://a.gov/real"]


def test_a_link_inside_a_fetched_page_counts_as_seen():
    """Step 4 of the prompt tells it to fetch the real document when a landing page comes
    back, and that link is in the page text rather than in a search result."""
    search = StaticSearch(
        results=[SearchResult(url="https://a.gov/index", title="A", snippet="s")],
        pages={"https://a.gov/index": "See the full report at https://a.gov/report.pdf for more.",
               "https://a.gov/report.pdf": "the actual report"})
    tb = box(search)
    tb.run("web_search", {"query": "q"})
    tb.run("web_fetch", {"url": "https://a.gov/index"})

    content, is_error = tb.run("web_fetch", {"url": "https://a.gov/report.pdf"})

    assert not is_error and "the actual report" in content


def test_a_link_from_the_topic_counts_as_seen():
    search = StaticSearch(pages={"https://en.wikipedia.org/wiki/Heat_pump": "seeded page"})
    tb = box(search, seen=["https://en.wikipedia.org/wiki/Heat_pump"])

    content, is_error = tb.run("web_fetch", {"url": "https://en.wikipedia.org/wiki/Heat_pump"})

    assert not is_error and "seeded page" in content


def test_seen_urls_are_matched_after_normalisation():
    """A result and a fetch that differ only in www, a trailing slash or a tracking param are
    the same page; refusing the second would be pedantry, not a guard."""
    search = StaticSearch(results=[SearchResult(url="https://www.a.gov/doc/", title="A",
                                                snippet="s")],
                          pages={"https://a.gov/doc?utm_source=x": "same page"})
    tb = box(search)
    tb.run("web_search", {"query": "q"})

    content, is_error = tb.run("web_fetch", {"url": "https://a.gov/doc?utm_source=x"})

    assert not is_error and "same page" in content
