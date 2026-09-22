import pytest
from types import SimpleNamespace

from rootlogic.llm import AnthropicLLM, Usage, json_schema
from rootlogic.models import Clarification, FindingDraft, PlanDraft, ReportDraft
from rootlogic.store import Store


def test_usage_cost_math():
    u = Usage(purpose="p", model="claude-opus-5", input_tokens=1_000_000,
              output_tokens=100_000, cache_read_tokens=1_000_000, web_searches=10)
    # 5.00 input + 2.50 output + 0.50 cache read + 0.10 search
    assert round(u.cost_usd, 2) == 8.10


def test_store_usage_and_memory():
    store = Store(":memory:")
    sid = store.create_session("solar panel recycling")
    store.record_call(session_id=sid, purpose="plan", model="claude-opus-5", input_tokens=100,
                      output_tokens=50, cache_read_tokens=0, cache_write_tokens=0,
                      web_searches=0, cost_usd=0.01, stop_reason="end_turn", request_id=None)
    assert store.usage(sid)["input_tokens"] == 100
    store.update_session(sid, status="done", related_topics=["battery recycling"])
    store.remember(sid, "solar panel recycling", "Summary about recycling.", ["k1"])
    assert store.recall("recycling of panels")[0]["session_id"] == sid
    assert store.suggestions() == ["battery recycling"]
    store.delete_session(sid)
    assert store.session(sid) is None and store.recall("recycling") == []


def test_llm_schemas_are_strict():
    """Every LLM-facing schema must forbid extra keys and require every property."""
    def walk(schema):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
            assert set(schema.get("required", [])) == set(schema.get("properties", {}))
        for v in schema.get("properties", {}).values():
            walk(v)
            if "items" in v:
                walk(v["items"])
        for d in schema.get("$defs", {}).values():
            walk(d)

    for model in (Clarification, PlanDraft, FindingDraft, ReportDraft):
        walk(json_schema(model))


class FakeMessages:
    """Mimics client.beta.messages: first turn searches and pauses, second submits."""

    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0,
                                server_tool_use=SimpleNamespace(web_search_requests=1))
        if len(self.calls) == 1:
            content = [SimpleNamespace(type="web_search_tool_result", content=[
                SimpleNamespace(url="https://a.com", title="A", page_age="1 day ago")])]
            return SimpleNamespace(content=content, stop_reason="pause_turn", usage=usage,
                                   model="claude-opus-5")
        finding = {"answer": "a", "sources": [], "claims": [], "gaps": [], "confidence": "low"}
        content = [SimpleNamespace(type="tool_use", name="submit_findings", input=finding, id="x")]
        return SimpleNamespace(content=content, stop_reason="tool_use", usage=usage,
                               model="claude-opus-5")


def test_research_loop_handles_pause_turn_and_collects_hits():
    usages = []
    messages = FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    llm = AnthropicLLM(usages.append, client=client)  # type: ignore[arg-type]

    finding, hits = llm.research(purpose="research:t1", system="s", prompt="p", schema=FindingDraft)

    assert finding.confidence == "low"
    assert [h.url for h in hits] == ["https://a.com"]
    assert len(messages.calls) == 2 and len(usages) == 2
    assert sum(u.web_searches for u in usages) == 2
    tool_names = [t["name"] for t in messages.calls[0]["tools"]]
    assert tool_names == ["web_search", "web_fetch", "submit_findings"]


def _tools_for(zdr):
    messages = FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    AnthropicLLM(lambda u: None, client=client, zdr=zdr).research(  # type: ignore[arg-type]
        purpose="research:t1", system="s", prompt="p", schema=FindingDraft)
    return {t["name"]: t for t in messages.calls[0]["tools"]}


def test_zdr_mode_disables_dynamic_filtering_on_web_tools():
    tools = _tools_for(zdr=True)
    assert tools["web_search"]["allowed_callers"] == ["direct"]
    assert tools["web_fetch"]["allowed_callers"] == ["direct"]
    assert "allowed_callers" not in tools["submit_findings"]


def test_default_mode_keeps_dynamic_filtering():
    tools = _tools_for(zdr=False)
    assert "allowed_callers" not in tools["web_search"]
    assert "allowed_callers" not in tools["web_fetch"]


def test_bad_api_key_gives_a_clear_message_not_a_traceback(tmp_path, capsys):
    """A wrong key is a setup problem: the CLI should explain it and exit non-zero."""
    import anthropic

    from rootlogic.cli import main
    from rootlogic.llm import AuthError

    class Rejecting:
        def create(self, **kw):
            raise anthropic.AuthenticationError(
                "invalid", response=SimpleNamespace(status_code=401, headers={},
                                                    request=None), body=None)

    client = SimpleNamespace(beta=SimpleNamespace(messages=Rejecting()))
    llm = AnthropicLLM(lambda u: None, client=client)  # type: ignore[arg-type]
    with pytest.raises(AuthError, match="console.anthropic.com"):
        llm.structured(purpose="clarify", system="s", prompt="p", schema=Clarification)

    # sanity: the offline path still works (a clear topic, so nothing waits on stdin)
    code = main(["--db", str(tmp_path / "x.db"), "--", "research", "--offline", "-y",
                 "impact of generative AI on newsrooms"])
    assert code == 0
