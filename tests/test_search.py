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
