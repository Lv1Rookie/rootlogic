"""OpenAI-compatible adapter: wire format, structured outputs, tool loop, refusals, usage.

Responses are real ``openai`` SDK objects built with ``model_validate``, replayed by a fake
client, so the adapter is exercised against the SDK's actual types without any network.
"""

import json
from types import SimpleNamespace

import openai
import pytest
from openai.types.chat import ChatCompletion

from rootlogic.backend import Backend, BackendError, parse_prices
from rootlogic.llm import AgentRefusal, LLMError
from rootlogic.models import Clarification, FindingDraft, PlanDraft
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
    llm = Backend(provider="openai", model="llama3.3", base_url="http://localhost:20128/v1",
                  search="tavily", prices=(0.0, 0.0),
                  moderation="none").make_llm(lambda u: None)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert str(llm.client.base_url).startswith("http://localhost:20128/v1")
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


# ------------------------------------------------------------------ worker model split


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_research_uses_the_worker_model_and_everything_else_the_lead(tmp_path, kind):
    """Sub-agents make most of the calls and burn most of the tokens (330k of 636k in one live
    run), so they can run on a cheaper model than planning, analysis and writing."""
    from rootlogic.fake_llm import FakeLLM
    from rootlogic.graph import ResearchGraph
    from rootlogic.orchestrator import Orchestrator
    from rootlogic.store import Store

    from .test_orchestrator import TODAY, ScriptedUI

    lead, worker = FakeLLM(), FakeLLM()
    ui = ScriptedUI()
    kw = dict(reports_dir=tmp_path, today=TODAY, worker_llm=worker)
    engine = (ResearchGraph(lead, Store(), ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(lead, Store(), ui, **kw))
    assert engine.run("impact of generative AI on newsrooms") is not None

    assert all(p.startswith("research:") for p, _ in worker.calls)
    assert {p.split(":")[0] for p, _ in lead.calls} == {"clarify", "plan", "reflect", "verify",
                                                        "analyze", "report"}


def test_without_a_worker_model_everything_runs_on_one_llm(tmp_path):
    from rootlogic.fake_llm import FakeLLM
    from rootlogic.orchestrator import Orchestrator
    from rootlogic.store import Store

    from .test_orchestrator import TODAY, ScriptedUI

    llm = FakeLLM()
    Orchestrator(llm, Store(), ScriptedUI(), reports_dir=tmp_path, today=TODAY).run(
        "impact of generative AI on newsrooms")
    assert any(p.startswith("research:") for p, _ in llm.calls)


def test_backend_builds_a_cheaper_worker_client(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    plain = Backend()
    assert plain.make_worker_llm(lambda u: None) is None        # no split configured
    assert "workers" not in plain.label

    split = Backend(model="claude-opus-5", worker_model="claude-haiku-4-5")
    assert split.make_llm(lambda u: None).model == "claude-opus-5"
    assert split.make_worker_llm(lambda u: None).model == "claude-haiku-4-5"
    assert "workers: claude-haiku-4-5" in split.label

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")   # the client needs some credential
    priced = Backend(provider="openai", model="big", worker_model="small", search="tavily",
                     moderation="none", prices=(5.0, 25.0), worker_prices=(0.25, 2.0))
    assert priced.make_worker_llm(lambda u: None).prices == (0.25, 2.0)
    assert priced.make_llm(lambda u: None).prices == (5.0, 25.0)


def test_cli_wires_the_worker_model_through(tmp_path, monkeypatch):
    from rootlogic.cli import backend_from_args, main
    import argparse
    args = argparse.Namespace(worker_model="claude-haiku-4-5", worker_prices="0.25,2",
                              model="claude-opus-5")
    b = backend_from_args(args)
    assert b.worker_model == "claude-haiku-4-5" and b.worker_prices == (0.25, 2.0)
    # offline runs ignore the split (one fake model), but the flag must parse and not crash
    assert main(["--db", str(tmp_path / "x.db"), "research", "--offline", "-y",
                 "--worker-model", "claude-haiku-4-5",
                 "impact of generative AI on newsrooms"]) == 0


def test_reasoning_effort_is_sent_on_every_request_when_set():
    """Thinking models reason before every call: qwen3:8b took 72s on a clarification that
    takes 1.3s with reasoning_effort 'none'. Opt-in, since servers that don't know the
    parameter reject the request."""
    completions = FakeCompletions([completion(json.dumps(CLARIFY))])
    client = type("C", (), {"chat": type("Ch", (), {"completions": completions})()})()
    llm = OpenAICompatibleLLM(lambda u: None, model="qwen3:8b", search=StaticSearch(),
                              base_url="http://localhost:20128/v1", reasoning_effort="none",
                              client=client)
    llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)
    assert completions.calls[0]["reasoning_effort"] == "none"


def test_reasoning_effort_is_omitted_by_default():
    llm, api, _ = make([completion(json.dumps(CLARIFY))])
    llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)
    assert "reasoning_effort" not in api.calls[0]


def test_reasoning_effort_is_rejected_for_claude():
    with pytest.raises(BackendError, match="reasoning-effort"):
        Backend(reasoning_effort="none").validate()


# ------------------------------------------------------------------ weak local models


def test_findings_sent_as_text_are_accepted_instead_of_nudged_forever():
    """Live with qwen3:8b: from ~6k tokens of web text on, it stopped calling tools and wrote
    the answer out instead. If that answer is really the findings, take it."""
    finding = dict(FINDING, answer="prose but valid")
    llm, api, _ = make([completion(tool_calls=[("web_search", {"query": "q"})],
                                   finish="tool_calls"),
                        completion("Here are my findings:\n```json\n"
                                   + json.dumps(finding) + "\n```")],
                       search=StaticSearch(results=[SearchResult(
                           url="https://a.com", title="A", snippet="s")]))
    out, hits = llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft)
    assert out.answer == "prose but valid"
    assert [h.url for h in hits] == ["https://a.com"]       # the search it did still counts
    assert len(api.calls) == 2                              # no nudge round trip


def test_a_model_that_never_searched_fails_fast_with_advice():
    """Ten prose turns at two minutes each is twenty minutes for nothing. With no sources
    there is nothing to wrap up, so stop after two and say what to change."""
    llm, api, _ = make([completion("I think the answer is...") for _ in range(4)])
    with pytest.raises(LLMError, match="never ran a usable search"):
        llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft)
    assert len(api.calls) == 2                              # not box.max_turns


def test_a_sub_agent_that_searched_submits_through_structured_output():
    """Live with qwen3:8b and llama3.1:8b: both searched correctly, then wrote prose instead of
    calling submit_findings. Tool-call arguments are free text to them; a JSON-schema
    response_format is grammar-constrained, so ask for the findings that way instead."""
    finding = dict(FINDING, answer="wrapped up")
    llm, api, _ = make([completion(tool_calls=[("web_search", {"query": "q"})],
                                   finish="tool_calls"),
                        completion("Let me explain what I found..."),
                        completion("Still explaining, no tool call..."),
                        completion(json.dumps(finding))],
                       search=StaticSearch(results=[SearchResult(
                           url="https://a.com", title="A", snippet="s")]))

    out, hits = llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft)

    assert out.answer == "wrapped up"
    assert [h.url for h in hits] == ["https://a.com"]       # the real search still counts
    wrap_up = api.calls[-1]
    assert "tools" not in wrap_up                           # no tools offered on the last call
    assert wrap_up["response_format"]["json_schema"]["strict"] is True
    assert "single JSON object" in wrap_up["messages"][-1]["content"]


# ------------------------------------------------------------------ streaming


def chunk(content=None, tool_calls=None, finish=None, usage=None, refusal=None):
    """One streamed chunk, shaped like the SDK's ChatCompletionChunk."""
    delta = SimpleNamespace(content=content, refusal=refusal, tool_calls=tool_calls)
    choices = [SimpleNamespace(index=0, delta=delta, finish_reason=finish)]
    return SimpleNamespace(model="qwen3:8b", choices=choices, usage=usage)


def tc(index, *, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=arguments))


def streaming(chunks, **kw):
    llm, api, usages = make([chunks], **kw)
    llm.stream = True
    return llm, api, usages


def test_streamed_text_is_reassembled_and_usage_recorded():
    """A gateway times out waiting for a slow model's first byte (OmniRoute: 30s). Streaming
    starts immediately, so the adapter must rebuild the reply from deltas."""
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=40))
    body = json.dumps(CLARIFY)
    llm, api, usages = streaming([chunk(content=body[:10]), chunk(content=body[10:]),
                                  chunk(finish="stop"), chunk(usage=usage)])

    out = llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)

    assert out.needs_clarification is False
    assert api.calls[0]["stream"] is True
    assert api.calls[0]["stream_options"] == {"include_usage": True}
    assert usages[0].input_tokens == 60 and usages[0].cache_read_tokens == 40
    assert usages[0].output_tokens == 20 and usages[0].stop_reason == "stop"
    assert usages[0].model == "qwen3:8b"


def test_streamed_tool_call_arguments_are_joined_across_chunks():
    """Tool-call JSON arrives split at arbitrary points, keyed by index."""
    args = json.dumps({"query": "newsroom AI"})
    llm, _, _ = streaming([
        chunk(tool_calls=[tc(0, id="call_0", name="web_search", arguments=args[:6])]),
        chunk(tool_calls=[tc(0, arguments=args[6:])]),
        chunk(finish="tool_calls"),
    ], search=StaticSearch(results=[SearchResult(url="https://a.com", title="A", snippet="s")]))

    message = llm._create("research:t1", messages=[])
    call = message.tool_calls[0]
    assert call.id == "call_0" and call.function.name == "web_search"
    assert json.loads(call.function.arguments) == {"query": "newsroom AI"}


def test_streamed_refusal_and_truncation_are_still_detected():
    llm, _, _ = streaming([chunk(refusal="I can't help"), chunk(finish="stop")])
    with pytest.raises(AgentRefusal):
        llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)

    llm, _, _ = streaming([chunk(content="{"), chunk(finish="length")])
    with pytest.raises(LLMError, match="truncated"):
        llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)


def test_a_server_that_sends_no_usage_chunk_still_works():
    """Not every OpenAI-compatible server honours stream_options.include_usage."""
    llm, _, usages = streaming([chunk(content=json.dumps(CLARIFY)), chunk(finish="stop")])
    llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)
    assert usages[0].input_tokens == 0 and usages[0].stop_reason == "stop"


def test_streaming_is_off_by_default():
    llm, api, _ = make([completion(json.dumps(CLARIFY))])
    llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)
    assert "stream" not in api.calls[0]


def test_stream_flag_is_rejected_for_claude():
    with pytest.raises(BackendError, match="--stream"):
        Backend(stream=True).validate()


def test_a_bare_api_error_is_reported_like_any_other(monkeypatch):
    """Live: a streamed request that failed mid-stream raised openai.APIError, which is the
    base class and not a subclass of APIStatusError, so it escaped as a traceback."""
    class Failing:
        def create(self, **kw):
            raise openai.APIError("[504]: Direct response did not start within 30000ms",
                                  request=None, body=None)

    client = type("C", (), {"chat": type("Ch", (), {"completions": Failing()})()})()
    llm = OpenAICompatibleLLM(lambda u: None, model="m", search=StaticSearch(),
                              base_url="http://localhost:20128/v1", client=client)
    with pytest.raises(LLMError, match="API error during clarify"):
        llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)


def test_the_loop_stops_offering_tools_once_their_budgets_are_spent():
    """Live: a sub-agent made 12 calls and emitted nothing for 84 minutes. Its searches and
    fetches were used up, so every turn was the model calling a dead tool and being told
    'budget used up' - each costing minutes on a local model."""
    search = StaticSearch(results=[SearchResult(url="https://a.com", title="A", snippet="s")],
                          pages={f"https://a.com/{i}": "page text " * 80 for i in range(4)})
    finding = dict(FINDING, answer="wrapped up after budgets spent")
    llm, api, _ = make([
        completion(tool_calls=[("web_search", {"query": "one"})], finish="tool_calls"),
        completion(tool_calls=[("web_fetch", {"url": "https://a.com/0"})], finish="tool_calls"),
        completion(tool_calls=[("web_fetch", {"url": "https://a.com/1"})], finish="tool_calls"),
        completion(tool_calls=[("web_fetch", {"url": "https://a.com/2"})], finish="tool_calls"),
        completion(json.dumps(finding)),      # the wrap-up call: no tools offered
        completion(tool_calls=[("web_search", {"query": "again"})], finish="tool_calls"),
    ], search=search)

    out, hits = llm.research(purpose="research:t1", system="s", prompt="p",
                             schema=FindingDraft, max_searches=1)

    assert out.answer == "wrapped up after budgets spent"
    assert len(api.calls) == 5                      # one search, three fetches, one wrap-up
    assert "tools" not in api.calls[-1]             # the last call asked for findings only
    assert api.calls[-1]["response_format"]["json_schema"]["strict"] is True
    assert len(hits) == 4                           # the search hit plus three fetched pages


def test_strict_mode_recovers_when_a_server_ignores_the_schema():
    """Live through a gateway: clarify parsed but plan came back as unparseable text -
    'Invalid JSON: expected value at line 1 column 186'. Strict mode assumed the server had
    honoured response_format and gave up; non-strict mode had always repaired."""
    plan = {"objective": "o", "recency_days": 365, "subtasks": [
        {"question": "q", "rationale": "r", "search_queries": ["s"], "depends_on": []}]}
    llm, api, _ = make([completion("Here is the plan:\n```json\n{\"objective\": \"o\","),
                        completion(json.dumps(plan))])

    out = llm.structured(purpose="plan", system="s", prompt="p", schema=PlanDraft)

    assert out.objective == "o"
    assert len(api.calls) == 2
    assert "response_format" in api.calls[0]           # asked properly the first time
    retry = api.calls[1]["messages"]
    assert retry[-1]["role"] == "user" and "corrected JSON" in retry[-1]["content"]
    assert "JSON Schema" in retry[-1]["content"]       # and spells the schema out


def test_strict_mode_gives_up_after_one_repair():
    llm, api, _ = make([completion("not json"), completion("still not json")])
    with pytest.raises(LLMError, match="did not match"):
        llm.structured(purpose="plan", system="s", prompt="p", schema=PlanDraft)
    assert len(api.calls) == 2


def test_a_model_that_echoes_the_schema_is_told_so_by_name():
    """Live failure on llama3.1: analyse returned the schema itself - $defs, properties,
    required, title - because the prompt shows it a schema and says "match this". The repair
    turn repeated the same schema, so the model echoed it again and the run died. The nudge
    now names the fields wanted instead of showing the schema a second time."""
    from rootlogic.openai_llm import json_schema

    echo = json.dumps(json_schema(Clarification))          # the model parrots its instructions
    llm, fake, _ = make([completion(echo), completion(json.dumps(CLARIFY))], strict=False)

    got = llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)
    assert got.needs_clarification is False                # recovered on the repair turn

    nudge = fake.calls[-1]["messages"][-1]["content"]
    assert "schema" in nudge.lower()
    for field in Clarification.model_fields:
        assert field in nudge, f"the repair turn should name {field}"
    # the error text may quote offending keys like $defs; what must not come back is the
    # schema itself, which is what the model copied the first time
    assert "matches this JSON Schema" not in nudge, "re-showing the schema caused the echo"
    assert '"additionalProperties"' not in nudge


def test_a_complete_object_followed_by_chatter_is_salvaged():
    """Live failure on llama3.1: the planner emitted a valid PlanDraft and then carried on
    writing, and the whole reply was rejected for "trailing characters" - throwing away an
    answer that was already complete."""
    reply = json.dumps(CLARIFY) + "\n\nLet me know if you would like me to expand on this!"
    llm, fake, _ = make([completion(reply)], strict=False)

    got = llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)
    assert got.needs_clarification is False
    assert len(fake.calls) == 1, "salvaging costs nothing; a repair turn costs a call"
