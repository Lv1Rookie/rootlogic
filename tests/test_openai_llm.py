"""OpenAI-compatible adapter: wire format, structured outputs, tool loop, refusals, usage.

Responses are real ``openai`` SDK objects built with ``model_validate``, replayed by a fake
client, so the adapter is exercised against the SDK's actual types without any network.
"""

import json

import pytest
from openai.types.chat import ChatCompletion

from rootlogic.backend import Backend, BackendError, parse_prices
from rootlogic.llm import AgentRefusal, LLMError
from rootlogic.models import Clarification, FindingDraft
from rootlogic.openai_llm import OpenAICompatibleLLM
from rootlogic.search import SearchResult, StaticSearch

FINDING = {"answer": "a", "sources": [], "claims": [], "gaps": [], "confidence": "medium"}
CLARIFY = {"needs_clarification": False, "reasoning": "clear", "questions": []}


def completion(content=None, tool_calls=None, finish="stop", refusal=None, usage=None):
    msg = {"role": "assistant", "content": content, "refusal": refusal}
    if tool_calls:
        msg["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)
                                           if not isinstance(a, str) else a}}
                             for i, (n, a) in enumerate(tool_calls)]
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "gpt-test",
        "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                           "prompt_tokens_details": {"cached_tokens": 40}}})


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kw):
        self.calls.append({**kw, "messages": [dict(m) for m in kw["messages"]]})
        return self.replies.pop(0)


def make(replies, *, strict=True, search=None, prices=None):
    completions = FakeCompletions(replies)
    client = type("C", (), {"chat": type("Ch", (), {"completions": completions})()})()
    usages = []
    llm = OpenAICompatibleLLM(usages.append, model="gpt-test", search=search or StaticSearch(),
                              strict=strict, prices=prices, client=client)
    return llm, completions, usages


# ------------------------------------------------------------------ structured


def test_structured_strict_uses_json_schema_response_format():
    llm, api, usages = make([completion(json.dumps(CLARIFY))], prices=(1.0, 2.0))
    out = llm.structured(purpose="clarify", system="sys", prompt="topic", schema=Clarification)

    assert out.reasoning == "clear"
    req = api.calls[0]
    assert req["model"] == "gpt-test"
    assert req["messages"] == [{"role": "system", "content": "sys"},
                               {"role": "user", "content": "topic"}]
    fmt = req["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "Clarification"
    # usage: cached tokens split out of prompt tokens; priced with the supplied rates
    u = usages[0]
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens) == (60, 40, 20)
    assert u.cost_usd == pytest.approx((100 * 1.0 + 20 * 2.0) / 1e6)


def test_unpriced_models_record_zero_cost():
    llm, _, usages = make([completion(json.dumps(CLARIFY))])
    llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)
    assert usages[0].cost_usd == 0.0


def test_non_strict_mode_prompts_for_json_and_repairs_once():
    llm, api, _ = make([completion("Sure! here you go"),
                        completion("```json\n" + json.dumps(CLARIFY) + "\n```")], strict=False)
    out = llm.structured(purpose="clarify", system="s", prompt="topic", schema=Clarification)

    assert out.needs_clarification is False
    assert "response_format" not in api.calls[0]
    assert "JSON Schema" in api.calls[0]["messages"][1]["content"]
    repair = api.calls[1]["messages"][-1]["content"]
    assert repair.startswith("That was not valid") and "corrected JSON" in repair


def test_non_strict_gives_up_after_one_repair():
    llm, _, _ = make([completion("nope"), completion("still nope")], strict=False)
    with pytest.raises(LLMError, match="did not match Clarification"):
        llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)


@pytest.mark.parametrize("reply", [completion(refusal="I can't help with that"),
                                   completion("x", finish="content_filter")])
def test_refusals_raise_agent_refusal(reply):
    llm, _, _ = make([reply])
    with pytest.raises(AgentRefusal):
        llm.structured(purpose="plan", system="s", prompt="p", schema=Clarification)


def test_truncation_raises():
    llm, _, _ = make([completion('{"needs', finish="length")])
    with pytest.raises(LLMError, match="truncated"):
        llm.structured(purpose="plan", system="s", prompt="p", schema=Clarification)


# ------------------------------------------------------------------ research loop


def test_research_tool_loop_in_chat_completions_format():
    search = StaticSearch(results=[SearchResult(url="https://a.com", title="A", snippet="s",
                                                published="2026-08-01")],
                          pages={"https://a.com": "page text"})
    llm, api, _ = make([
        completion(tool_calls=[("web_search", {"query": "q"})], finish="tool_calls"),
        completion(tool_calls=[("web_fetch", {"url": "https://a.com"})], finish="tool_calls"),
        completion(tool_calls=[("submit_findings", FINDING)], finish="tool_calls"),
    ], search=search)

    finding, hits = llm.research(purpose="research:t1", system="sys", prompt="p",
                                 schema=FindingDraft, recency_days=30)

    assert finding.confidence == "medium"
    assert search.queries == [("q", 30)] and search.fetched == ["https://a.com"]
    assert [h.page_age for h in hits if h.text is None] == ["2026-08-01"]
    assert [h.text for h in hits if h.text] == ["page text"]  # fetched page kept as evidence
    tools = api.calls[0]["tools"]
    assert [t["function"]["name"] for t in tools] == ["web_search", "web_fetch", "submit_findings"]
    assert all(t["type"] == "function" and t["function"]["strict"] for t in tools)
    # second request carries the assistant tool call and the matching tool message
    assistant, tool = api.calls[1]["messages"][-2:]
    assert assistant["tool_calls"][0]["function"]["name"] == "web_search"
    assert tool == {"role": "tool", "tool_call_id": "call_0",
                    "content": json.dumps([{"url": "https://a.com", "title": "A", "snippet": "s",
                                            "published": "2026-08-01"}])}
    assert "page text" in api.calls[2]["messages"][-1]["content"]


def test_invalid_submission_is_sent_back_for_repair():
    llm, api, _ = make([
        completion(tool_calls=[("submit_findings", {"answer": "missing fields"})],
                   finish="tool_calls"),
        completion(tool_calls=[("submit_findings", FINDING)], finish="tool_calls"),
    ])
    finding, _ = llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft)
    assert finding.answer == "a"
    assert api.calls[1]["messages"][-1]["content"].startswith("Invalid findings, fix and resubmit")


def test_every_parallel_call_gets_a_tool_message_and_budgets_hold():
    llm, api, _ = make([
        completion(tool_calls=[("web_search", {"query": "a"}), ("web_search", {"query": "b"}),
                               ("web_fetch", "{not json")], finish="tool_calls"),
        completion("I'm done thinking"),
        completion(tool_calls=[("submit_findings", FINDING)], finish="tool_calls"),
    ])
    llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft,
                 max_searches=1)
    tool_msgs = [m for m in api.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_0", "call_1", "call_2"]
    assert "budget used up" in tool_msgs[1]["content"]
    assert "not valid JSON" in tool_msgs[2]["content"]
    assert "call submit_findings" in api.calls[2]["messages"][-1]["content"]  # nudge


def test_non_strict_tools_omit_strict_flag():
    llm, api, _ = make([completion(tool_calls=[("submit_findings", FINDING)],
                                   finish="tool_calls")], strict=False)
    llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft)
    assert all("strict" not in t["function"] for t in api.calls[0]["tools"])


# ------------------------------------------------------------------ backend validation


@pytest.mark.parametrize("kw,message", [
    (dict(provider="openai", search="tavily", moderation="none"), "needs --model"),
    (dict(provider="openai", model="m", moderation="none"), "needs --search tavily"),
    (dict(provider="openai", model="m", search="tavily", zdr=True, moderation="none"),
     "Claude's web tools only"),
    (dict(base_url="http://x"), "--base-url is for --provider openai"),
    (dict(provider="gemini"), "unknown provider"),
])
def test_backend_rejects_invalid_combinations(kw, message):
    with pytest.raises(BackendError, match=message):
        Backend(**kw).validate()


def test_backend_builds_openai_adapter(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    llm = Backend(provider="openai", model="llama3.3", base_url="http://localhost:11434/v1",
                  search="tavily", prices=(0.0, 0.0),
                  moderation="none").make_llm(lambda u: None)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert str(llm.client.base_url).startswith("http://localhost:11434/v1")
    assert Backend().label == ("anthropic:claude-opus-5 · search: anthropic · moderation: none")


def test_parse_prices():
    assert parse_prices("0.25,2") == (0.25, 2.0) and parse_prices(None) is None
    with pytest.raises(BackendError):
        parse_prices("cheap")


def test_cli_reports_invalid_backend_cleanly(tmp_path, capsys):
    from rootlogic.cli import main
    code = main(["--db", str(tmp_path / "x.db"), "research", "--provider", "openai",
                 "--model", "m", "--moderation", "none", "-y", "topic"])
    assert code == 2 and "needs --search tavily" in capsys.readouterr().out


def empty_completion(error=None):
    """A 200 response carrying no choices: what OpenRouter returned on a free model."""
    # The SDK builds responses leniently, so a malformed body reaches us as-is: choices=None
    # rather than a validation error. model_construct reproduces that.
    body = {"id": "x", "object": "chat.completion", "created": 0, "model": "gpt-test",
            "choices": None, "usage": None}
    if error:
        body["error"] = error
    return ChatCompletion.model_construct(**body)


def test_missing_choices_explains_itself_instead_of_crashing():
    llm, _, usages = make([empty_completion(
        {"message": "Rate limit exceeded: free-models-per-day", "code": 429})])
    with pytest.raises(LLMError, match="Rate limit exceeded: free-models-per-day"):
        llm.structured(purpose="plan", system="s", prompt="p", schema=Clarification)
    assert usages[0].stop_reason == "no_choices"   # still recorded, cost stays visible


def test_missing_choices_without_an_error_message_suggests_what_to_try():
    llm, _, _ = make([empty_completion()])
    with pytest.raises(LLMError, match="try another --model, or --no-strict"):
        llm.structured(purpose="plan", system="s", prompt="p", schema=Clarification)
