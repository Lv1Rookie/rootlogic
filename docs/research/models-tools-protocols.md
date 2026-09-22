# Models, Tools, Storage and Agent Protocols in rootlogic: Beginner Q&A

*Compiled 2026-09-22. Claims about **our code** cite repo files as `path:line`. Claims about **external systems** cite primary sources (vendor docs, specs, source repos). Anything I could not confirm from a primary source is marked **unverified**. This note complements [agentic-research-assistant.md](agentic-research-assistant.md) and links to it rather than repeating it.*

---

## TL;DR

1. **One provider today: Anthropic.** Every real LLM call goes through `client.beta.messages.create` in the `anthropic` Python SDK with model `claude-opus-5`, the server-side fallback beta, and an `effort` level (`rootlogic/llm.py:26-27`, `rootlogic/llm.py:92-96`). Clarify, plan, reflect, analyze and report are **single structured-output calls** (`output_config.format` = `json_schema`). Research is a **small tool loop** that ends when the model calls our `submit_findings` tool. `FakeLLM` replaces all of this offline (`rootlogic/fake_llm.py:109`).
2. **We already have the seam for swapping models.** The `LLM` Protocol has exactly two methods, `structured` and `research` (`rootlogic/llm.py:69-74`). The hard part of going model-agnostic is **search, not chat**. Provider-hosted search tools (Anthropic `web_search`/`web_fetch`, OpenAI `web_search`, Gemini Google Search grounding) each work only inside their own vendor's API. A portable design moves search into our own client-side tool behind a `SearchProvider` protocol.
3. **Research subagents use three tools:** Anthropic's server-side `web_search_20260209` (capped at `max_searches`, default 5), server-side `web_fetch_20260209` (capped at 3), and our client tool `submit_findings`, which has a strict JSON schema (`rootlogic/llm.py:144-154`). Search costs $10 per 1,000 searches plus tokens. Fetch costs tokens only ([web search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool), [web fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)).
4. **Yes, conversations are already saved in a database.** A SQLite file (`.rootlogic/rootlogic.db`) holds sessions, the human↔agent messages, an append-only event log, tasks with their findings, kept and dropped sources, per-call token usage, and an FTS5 memory index (`rootlogic/store.py:19-88`). The LangGraph engine also checkpoints full graph state to `.rootlogic/checkpoints.db`. **Not saved:** raw LLM requests and responses, the subagents' internal message histories, search queries, and fetched page text.
5. **There is no network protocol between our agents.** The workers never talk to each other. All communication is in-process and hub-and-spoke: the orchestrator builds a prompt string, a worker returns a typed `FindingDraft`, and the orchestrator curates it and feeds it into later prompts. In the LangGraph engine the same idea is expressed as typed shared state, `Send()` messages and a reducer. Industry protocols do exist: MCP (agent↔tools), A2A (agent↔agent across services, v1.0, now under the Linux Foundation's Agentic AI Foundation) and AG-UI (agent↔UI). We would need A2A only if workers became separate services or came from other vendors.
6. **The "data abstraction" analogy is mostly right.** Each subagent sits behind an interface (a prompt goes in, a typed result comes out) and hides its search loop and scratch context, like an abstract data type. But the work is **not hidden from the user**: rootlogic deliberately shows the plan, the status of each task, dropped sources with reasons, and cost. It also doesn't resolve "constantly in the background". It runs in bounded waves with budgets and pause points.

---

## 1. Which LLM APIs does the project use today?

### 1.1 The single real client: `AnthropicLLM`

| Aspect | What the code does | Verified against |
|---|---|---|
| SDK / endpoint | `anthropic.Anthropic().beta.messages.create(...)` (`rootlogic/llm.py:85`, `rootlogic/llm.py:92`). The **beta** namespace is needed only because of the fallback beta header | [Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback) |
| Model | `MODEL = "claude-opus-5"` (`rootlogic/llm.py:26`) | Opus 5 is listed as supporting structured outputs and all five effort levels ([structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [effort](https://platform.claude.com/docs/en/build-with-claude/effort)) |
| Server-side fallback | `betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"` on **every** call (`rootlogic/llm.py:27`, `rootlogic/llm.py:94-95`). If the model still refuses, `stop_reason == "refusal"` raises `AgentRefusal` (`rootlogic/llm.py:118-120`) | With `"default"`, a **safety-classifier decline** is re-run on an Anthropic-recommended model inside the same API call, and the response names the model that answered. Rate limits and overloads are *not* retried. The `2026-07-01` header is required for the `"default"` form. The feature is not available on Batches, Bedrock, Vertex or Foundry ([docs](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)) |
| Cost accounting | Every response's `usage` is turned into a `Usage` row. It records `response.model`, so a fallback model is priced correctly, and it counts `server_tool_use.web_search_requests` (`rootlogic/llm.py:105-117`). Prices are hard-coded in `PRICES` / `WEB_SEARCH_USD` (`rootlogic/llm.py:31-33`) | Field names match the SDK usage type (see [existing brief §4.2](agentic-research-assistant.md#42-token-usage-and-cost-fields-returned-by-apis)) |
| Errors | Connection, rate-limit and status errors are wrapped in `LLMError` (`rootlogic/llm.py:98-103`) | — |

### 1.2 Which calls go where

| Stage (purpose string) | Method | Schema (Pydantic → JSON Schema) | Effort | Call site |
|---|---|---|---|---|
| `clarify` | `structured` | `Clarification` | `low` | `rootlogic/orchestrator.py:96-98` |
| `plan` | `structured` | `PlanDraft` | `high` (default) | `rootlogic/orchestrator.py:108-109` |
| `research:tN` | **`research`** (tool loop) | `FindingDraft` (as the `submit_findings` input schema) | `medium` | `rootlogic/orchestrator.py:156-162`, `rootlogic/llm.py:160` |
| `reflect` | `structured` | `Reflection` | `high` | `rootlogic/orchestrator.py:189-190` |
| `analyze` | `structured` | `Analysis` | `high` | `rootlogic/orchestrator.py:208-209` |
| `report` | `structured` | `ReportDraft` | `high` | `rootlogic/orchestrator.py:219-220` |

The LangGraph engine makes the same six kinds of call from its nodes (`rootlogic/graph.py:166`, `:192`, `:253`, `:301`, `:317`, `:331`).

**`structured` in one sentence:** it sends one user message and asks for `output_config={"effort": ..., "format": {"type": "json_schema", "schema": ...}}`, then parses the first text block with `schema.model_validate_json` (`rootlogic/llm.py:124-139`).
- Anthropic's structured outputs are GA with no beta header. `output_config.format` replaces the deprecated `output_format`.
- The schema rules require `additionalProperties: false` on every object. They don't support numeric or string-length constraints or recursive schemas ([docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)).
- Our schemas meet these rules because every LLM-facing model inherits `Strict` (`extra="forbid"`) and uses only descriptions, not `min`/`max` constraints (`rootlogic/models.py:19-20`, `rootlogic/models.py:5-9`).
- Structured outputs are **incompatible with Anthropic's Citations feature** (same doc). That is one reason our report cites `[n]` numbers that we compute ourselves (`rootlogic/context.py:69-86`) rather than API citation spans.

**Effort** is `output_config.effort`. The levels are `low | medium | high | xhigh | max`, and `high` is identical to omitting the parameter. Effort affects *all* output tokens, including tool calls: "Lower effort also means fewer and terser tool calls" ([effort docs](https://platform.claude.com/docs/en/build-with-claude/effort)). So `medium` for research workers is also a quiet brake on how much searching they do.

### 1.3 Offline: `FakeLLM`

`FakeLLM` implements the same two methods. It picks a canned handler **by schema type** (`Clarification → default_clarify`, and so on), records every `(purpose, prompt)` in memory, and reports fake usage with model `offline-fake`, which costs $0 (`rootlogic/fake_llm.py:109-141`, `rootlogic/llm.py:50-51`). The CLI's `--offline` flag and the web UI's `offline` switch select it (`rootlogic/cli.py:146-151`).

---

## 2. Going model-agnostic: options and trade-offs

### 2.1 The seam we already have

```python
class LLM(Protocol):
    def structured(self, *, purpose, system, prompt, schema, effort="high") -> T: ...
    def research(self, *, purpose, system, prompt, schema, max_searches=5) -> tuple[T, list[SearchHit]]: ...
```
(`rootlogic/llm.py:69-74`)

Both engines depend only on this Protocol (`rootlogic/orchestrator.py:19`, `rootlogic/graph.py:45`). That is how the fake model plugs in, and any new provider would plug in the same way. `structured` is easy to port because every major provider now has JSON-schema output. `research` is the hard method because it bundles three things:
1. a tool loop,
2. **the vendor's own search engine**, and
3. vendor-specific result shapes (`web_search_tool_result`, `page_age`) that `_search_hits` parses (`rootlogic/llm.py:179-192`).

### 2.2 The key trade-off: server-side tools don't travel

| Provider-hosted search | How you turn it on | What comes back | Portable? |
|---|---|---|---|
| Anthropic `web_search_2026xxxx` | tool entry in `tools` | `server_tool_use` + `web_search_tool_result` blocks (`url`, `title`, `page_age`, `encrypted_content`) ([docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)) | No |
| OpenAI `web_search` | Responses API tool (Chat Completions only via special search models) | `web_search_call` items + `url_citation` annotations + optional `sources` ([docs](https://developers.openai.com/api/docs/guides/tools-web-search)) | No |
| Gemini Grounding with Google Search | `google_search` tool | search-call/result items and `url_citation` annotations. Gemini 3 bills per executed query ([docs](https://ai.google.dev/gemini-api/docs/google-search)) | No |
| Local (Ollama / vLLM) | none built in | — | n/a |

Wrappers don't erase this difference either:
- **LiteLLM** passes `web_search_options` through to "each provider['s] own search backend", and its docs don't describe any citation normalization ([LiteLLM web search](https://docs.litellm.ai/docs/completion/web_search)).
- **Pydantic AI's** `WebSearchTool` is "executed by the model provider's infrastructure" and supported only on OpenAI Responses, Anthropic, Google, xAI, Groq and OpenRouter. On any other model it raises `UserError` ([built-in tools](https://pydantic.dev/docs/ai/tools-toolsets/builtin-tools/)).
- **LangChain** likewise runs provider built-in tools server-side ([models](https://docs.langchain.com/oss/python/langchain/models)).

**Consequence:** a truly model-agnostic `research` runs search **on our side**, as a *client tool* the model calls and our code executes (Tavily, Exa, Brave, SearXNG… see [existing brief §7.1](agentic-research-assistant.md#71-web-search-apis)). What changes inside `research`:

| Today (`rootlogic/llm.py:142-176`) | Client-side search |
|---|---|
| The API runs searches in its own server-side loop. We handle only `pause_turn` | We run the loop ourselves: on `stop_reason == "tool_use"`, call `SearchProvider.search(q)` and send back `tool_result` blocks |
| `max_uses` enforces the search cap | Our loop counts calls and refuses (returns an error result) after `max_searches` |
| `page_age` comes from Anthropic's index | We map the provider's published date into `SearchHit.page_age` (`rootlogic/models.py:173-177`) |
| Cost = `web_search_requests × $0.01` | Cost = provider price × calls. `Usage` needs a `search_provider` field |
| `web_fetch` is server-side. The URL must already appear in context (exfiltration guard) | We need our own fetcher (Jina Reader, Firecrawl, trafilatura) and must enforce our own URL/domain policy |
| Dynamic filtering (code execution trims results) comes for free | Lost. We should trim results ourselves (top-k snippets, `max_chars`) |

The `submit_findings` pattern (a tool whose schema *is* the result) **ports well**. OpenAI, Ollama and vLLM all support strict or JSON-schema function parameters. It is also the standard workaround for providers that can't combine tools and structured output in one call.

One Anthropic-specific bonus: when a *client* tool returns results, it can return them as `search_result` content blocks, and Claude will then cite them like native web results. These blocks are GA and need no beta header ([search results](https://platform.claude.com/docs/en/build-with-claude/search-results)). So moving search client-side does not cost citations on Claude.

### 2.3 Structured-output support across providers

| Provider | How to request JSON-schema output | Notable limits |
|---|---|---|
| Anthropic | `output_config.format: {type: "json_schema", schema}`; `strict: true` on tools | `additionalProperties:false` required; no numeric/string constraints; incompatible with citations ([docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)) |
| OpenAI | Responses: `text.format: {type: "json_schema", name, schema, strict: true}`. Chat Completions: `response_format` ([docs](https://developers.openai.com/api/docs/guides/structured-outputs)) | `additionalProperties:false`, **all** properties listed in `required`. Combining with the hosted `web_search` tool in one call: **unverified** |
| Google Gemini | `response_format` with `mime_type: application/json` and a JSON Schema subset. Gemini 3 can combine structured output with Google Search, URL context and function calling (**preview**) ([docs](https://ai.google.dev/gemini-api/docs/structured-output)) | Schema subset (types, `enum`, `format`, `minItems`/`maxItems`…) |
| Ollama (local) | Native `format: <schema>`, or `response_format` on `/v1/chat/completions` ([docs](https://docs.ollama.com/capabilities/structured-outputs)) | Ollama **Cloud** does not support structured outputs. The OpenAI-compat layer lacks `tool_choice` and stateful Responses ([compat](https://docs.ollama.com/api/openai-compatibility)) |
| vLLM (local) | `response_format` `json_schema`, or extra `structured_outputs` (`json`, `regex`, `choice`, `grammar`). Backends xgrammar / guidance / outlines ([docs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)) | Tool-call parsing depends on model/parser config (**unverified** detail) |

Caveat for local models: grammar-constrained decoding guarantees the output *shape*, not *quality*. Small models following a strict `FindingDraft` schema can still produce weak or invented citations. Keep `curate()` (`rootlogic/context.py:89-97`) as a check that doesn't depend on which model produced the output.

### 2.4 The options compared

| Option | What it is | Normalizes | Does **not** normalize | Fit for rootlogic |
|---|---|---|---|---|
| **(a) Write more adapters ourselves** (`OpenAILLM`, `GeminiLLM`, `OllamaLLM`) | ~150 lines each implementing the 2-method Protocol | Whatever we choose | — | **Best teaching value.** Keeps the loop visible and testable, matching the project's "no hidden orchestration" stance ([existing brief §3](agentic-research-assistant.md#3-framework-trade-offs)) |
| **(b) LiteLLM** ([repo](https://github.com/BerriAI/litellm)) | "Call 100+ LLM APIs in OpenAI (or native) format". MIT except `enterprise/` ([LICENSE](https://github.com/BerriAI/litellm/blob/main/LICENSE)); 59.4k★ (GitHub API, 2026-09-22) | Request/response shape; `response_format` across providers, with `supports_response_schema()` and optional client-side `enable_json_schema_validation` ([JSON mode](https://docs.litellm.ai/docs/completion/json_mode)); cost tracking | Provider server tools and their result blocks; citations; Anthropic-only features (fallback beta, `pause_turn`, `encrypted_content` replay) | Good if we want one `LiteLLMAdapter` that reaches many providers, **paired with client-side search** |
| **(c) Pydantic AI** ([models](https://pydantic.dev/docs/ai/models/overview/)) | `Model` + `Provider` + profile; `"openai:gpt-5.2"` strings; `FallbackModel`; OpenAI-compatible providers include Ollama, vLLM, OpenRouter, LiteLLM. MIT, 20.1k★ | Typed outputs from Pydantic models (fits our `*Draft` classes); tool calling; fallbacks | Built-in tools only on some providers (above) | Strong fit for *typed* outputs. Its `Agent` would largely replace `research()`'s loop |
| **(d) LangChain chat models** | `init_chat_model("provider:model")`; `.with_structured_output(schema, method="json_schema"/"function_calling"/"json_mode")`; `.bind_tools()` ([docs](https://docs.langchain.com/oss/python/langchain/models)) | Message types, tool calls, structured output | Provider built-in tools execute server-side and differ by provider | We already depend on LangGraph (`pyproject.toml`), but **not** on `langchain`. We call our own `LLM`, not LangChain models (`rootlogic/graph.py:45`). Adding it means another dependency and a second abstraction |
| **(e) Any OpenAI-compatible endpoint** | One `OpenAICompatLLM(base_url=...)` covers OpenAI, Ollama `/v1`, vLLM, OpenRouter, many hosts | Chat Completions shape | Per-server gaps (e.g. Ollama: no `tool_choice`) | **Highest leverage per line of code** |
| **(f) OpenRouter** ([web search](https://openrouter.ai/docs/features/web-search)) | Hosted router exposing many models over one OpenAI-style API | Search too: `:online` suffix, `web` plugin or `openrouter:web_search` server tool. Uses the vendor's native search for Anthropic/Google/OpenAI/Perplexity models and **Exa** for others. Results come back as standardized `url_citation` annotations (Exa: $0.007/request) | It's another hosted dependency and data processor. Native-search pricing varies by model | Handy for quickly trying many models. It is itself a way to get *normalized hosted search* |

### 2.5 Recommended design (proposal only, no code changes)

```
LLM (Protocol, unchanged: structured, research)
├── AnthropicLLM        # today; keep "native search" mode (server tools) as the default
├── OpenAICompatLLM     # base_url + api_key: OpenAI, Ollama /v1, vLLM, OpenRouter
│     structured -> response_format json_schema (strict)
│     research   -> our own tool loop with client tools
└── FakeLLM             # unchanged

SearchProvider (new Protocol)
    search(query, *, recency_days, max_results) -> list[SearchHit]   # SearchHit gains snippet
    fetch(url, *, max_chars) -> FetchedPage | None
├── AnthropicNativeSearch   # marker: "let the API do it" (current behaviour)
├── TavilySearch / BraveSearch / ExaSearch
└── FakeSearch              # for tests
```

- `research()` in the new adapters would expose tools `search`, `fetch` and `submit_findings`. Search and fetch would dispatch to the injected `SearchProvider` and count calls against `max_searches`.
- `AnthropicLLM` could accept either `AnthropicNativeSearch` (current path) or a client provider. With a client provider it would return results as `search_result` blocks.
- Add `provider` and `search_provider` to `Usage`/`llm_calls`, and move `PRICES` into config.
- Leave `curate()` and the recency and domain filters exactly where they are. They already ignore which model produced the output.

---

## 3. What tools do the research subagents use?

### 3.1 The three tools (from `rootlogic/llm.py:144-154`)

| Tool | Kind | Our settings | What it does / costs |
|---|---|---|---|
| `web_search` (`web_search_20260209`) | **Server tool.** Anthropic runs it | `max_uses = max_searches` (Budget default 5, `rootlogic/orchestrator.py:34`) | Returns `url`, `title`, `page_age` ("when the site was last updated") and `encrypted_content`. Citations are always on. **$10 per 1,000 searches** plus tokens; failed searches aren't billed. Exceeding `max_uses` produces a `max_uses_exceeded` error *inside* a 200 response ([docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)) |
| `web_fetch` (`web_fetch_20260209`) | **Server tool** | `max_uses = 3` | Fetches full page or PDF text. **No extra charge beyond tokens** (a 10 kB page ≈ 2,500 tokens). No JavaScript rendering. It can only fetch URLs that already appeared in the conversation, as an exfiltration guard. Citations are optional and off by default. We don't enable them ([docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)) |
| `submit_findings` | **Client tool** (ours) | `strict: True`, `input_schema = FindingDraft` JSON Schema | Never "executed": its *input* is the result. The model must fill `answer`, `sources[]` (url, title, published, publisher, summary, key_takeaways, credibility, relevance), `claims[]` (text + source_urls), `gaps`, `confidence` (`rootlogic/models.py:54-75`) |

**Dynamic filtering is on by default.** Because we use the `_20260209` versions, both web tools default to `allowed_callers: ["code_execution_20260120"]`. Claude can write code that filters results *before* they enter its context, and the code execution this uses costs nothing extra ([web search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)). Side effect: the `_20260209`+ versions are **not ZDR-eligible** unless you set `allowed_callers: ["direct"]` (rootlogic now does this when run with `--zdr`) ([server tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools#zdr-and-allowed-callers)). Newer versions exist (`web_search_20260318`, `web_fetch_20260318`) that add `response_inclusion` to shrink responses.

### 3.2 The loop, step by step (`rootlogic/llm.py:155-176`)

1. Start with `messages = [user: research_prompt]` and loop **at most 8 times**.
2. Call the API with the tools, `effort: medium` and `max_tokens=16000`. Harvest `SearchHit`s from every `web_search_tool_result` block. Error objects are skipped (`rootlogic/llm.py:179-192`).
3. If the response contains a `tool_use` block named `submit_findings`, validate its input as `FindingDraft` and **return immediately**. No `tool_result` is ever sent back.
4. Otherwise append the assistant content to `messages` (verbatim, which keeps `encrypted_content` intact, as the API requires).
5. If `stop_reason == "pause_turn"`, the API paused its server-side search loop. We just resend, which is what the docs prescribe: "Pass the paused response back as-is" and keep the same tools ([server tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools#the-server-side-loop-and-pause-turn)).
6. If `max_tokens` was hit, raise `LLMError`.
7. Otherwise (the model ended its turn with prose instead of calling the tool) append the **nudge** user message: *"Stop searching now and call submit_findings with what you have."*
8. After 8 rounds without a submission, raise `LLMError("research did not converge")`. The orchestrator then marks the task `failed` and carries on (`rootlogic/orchestrator.py:148-152`).

**What comes back to the orchestrator:** `(FindingDraft, list[SearchHit])`. The hits (`url`, `title`, `page_age`) are used only by `curate()`:
- `fill_dates_from_hits` backfills `published` dates the model left as `unknown`.
- Then the recency, domain and duplicate filters run (`rootlogic/context.py:89-97`).

The hits themselves are not stored (see §4.3).

**Edge case worth knowing:** if Claude calls `web_search` and `submit_findings` in the same parallel group, the API returns `stop_reason: "tool_use"` and leaves the search un-run ([server tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools#mixing-server-tools-and-client-tools-in-one-turn)). Our loop takes the submission and returns, so that search simply never happens. That is harmless.

### 3.3 Tools we could add

The existing brief already compares pricing, licenses and MCP servers for these in [§7](agentic-research-assistant.md#7-tools-for-research-subagents). Here is how each would slot into rootlogic:

| Tool | Role in rootlogic | Where it plugs in |
|---|---|---|
| Tavily, Exa, Brave Search | Client-side web search (the portable replacement for `web_search`) | `SearchProvider.search` |
| Jina Reader, Firecrawl | Client-side fetch/clean to Markdown; Firecrawl handles JS pages | `SearchProvider.fetch` |
| arXiv, Semantic Scholar, OpenAlex | Scholarly search with real publication dates. Better recency signals than `page_age` | extra client tool `academic_search` |
| Wikipedia / MediaWiki | Background and disambiguation, useful for the **clarify** step | extra client tool, or a clarify pre-step |

Keep a worker to about 3–5 tools. Anthropic's tool guidance argues against large overlapping toolsets (see [existing brief §1.4](agentic-research-assistant.md#14-writing-tools-for-agents-anthropic-2025-09-11)).

---

## 4. Can we save research conversations in a database?

**Yes. rootlogic already does.** `Store` opens one SQLite file, by default `.rootlogic/rootlogic.db` (`rootlogic/cli.py:28`, `rootlogic/cli.py:296`). It enables `foreign_keys` and `journal_mode = WAL` (`rootlogic/store.py:104-105`). Worker threads share one connection guarded by a lock (`rootlogic/store.py:99-102`).

### 4.1 The tables (`rootlogic/store.py:19-88`)

| Table | One row per… | Key columns | Written by |
|---|---|---|---|
| `sessions` | research session | `topic`, `status` (running/done/aborted/failed/interrupted), `plan_json`, `summary`, `related_topics` (JSON), `report_path` | `create_session`, `update_session` (`rootlogic/store.py:119-131`) |
| `messages` | human↔agent utterance | `role` (user/agent), `kind` (topic, clarifying_question, answer, note, report), `content`. For `report`, only the **executive summary** is stored (`rootlogic/orchestrator.py:226`) | `add_message` |
| `events` | action-log entry (append-only) | `type` (e.g. `task.started`, `source.dropped`), `message`, `data` JSON. Every `_emit` writes one (`rootlogic/orchestrator.py:309-312`). The graph engine's `_emit` now stores the same `data` fields (fixed after this note was written) | `add_event` |
| `tasks` | sub-task | `(session_id, task_id)` PK, `question`, `status`, `origin` (planner/reflection/user), `finding_json` (the curated `Finding`, including dropped URLs) | `upsert_task` (`rootlogic/store.py:166-175`) |
| `sources` | URL per session | `(session_id, url)` PK, so the first task to claim a URL wins (`INSERT OR IGNORE`); `kept` 1/0, `reason` for drops, `data` = full `SourceDraft` JSON for kept ones | `add_source` (`rootlogic/store.py:177-184`) |
| `llm_calls` | API request | `purpose`, `model`, input/output/cache tokens, `web_searches`, `cost_usd`, `stop_reason`, `request_id`. No foreign key, so `delete_session` removes these rows explicitly (`rootlogic/store.py:145-148`) | the `usage_sink` wired in `create_engine` (`rootlogic/cli.py:139-144`) |
| `memory_fts` | finished session | FTS5 table: `session_id UNINDEXED`, `topic`, `summary`, `takeaways`. `recall()` ORs the query's words and sorts by `bm25()` (`rootlogic/store.py:228-240`) | `remember` at report time |

SQLite facts behind these choices:
- **FTS5** is a virtual-table full-text index. `bm25()` returns *lower* values for better matches, so `ORDER BY bm25(...)` ascending is correct. `UNINDEXED` columns are stored but not searchable ([FTS5](https://www.sqlite.org/fts5.html)).
- **WAL** lets readers and the single writer proceed concurrently, and the setting persists in the file. But only one writer runs at a time, and all processes must be on the same host, with no network filesystems ([WAL](https://www.sqlite.org/wal.html)). That is fine for a local app, and a real limit for multi-user hosting.

### 4.2 The other two persistence layers

- **LangGraph checkpoints** (`--engine graph` only): `SqliteSaver` on `.rootlogic/checkpoints.db` (`rootlogic/cli.py:155`, `rootlogic/graph.py:92-93`), keyed by `thread_id = session_id` (`rootlogic/graph.py:402-404`).
  - LangGraph saves a state snapshot (values, next nodes, config, metadata, pending tasks/interrupts) **at every super-step**, serialized with `JsonPlusSerializer` (optionally encrypted) ([persistence](https://docs.langchain.com/oss/python/langgraph/persistence)).
  - Our state is the `ResearchState` dict (`rootlogic/graph.py:59-74`), so each checkpoint includes the plan, findings, `seen_urls` and pending questions.
  - This is what makes `rootlogic resume <id>` work (`rootlogic/graph.py:412-425`). The docs position `SqliteSaver` for local use and `PostgresSaver` for production.
- **Report files:** Markdown written to `.rootlogic/reports/<sid>-<slug>.md` (`rootlogic/orchestrator.py:223-225`). The DB stores only the path. The web UI reads the file back (`rootlogic/web.py:292-294`). If the file is deleted, the full report is gone, though every `Finding` survives in `tasks.finding_json`.

The web server's live `Run` objects (event lists, pending requests) are **in memory only** (`rootlogic/web.py:176`). Engine events are also written to `events`, but the `request`/`request.resolved` and `report` stream events are not.

### 4.3 What is *not* stored today

| Missing | Where it lives now | Why you might want it |
|---|---|---|
| Raw request/response per LLM call (system prompt, full prompt, raw content blocks) | Nowhere. Only usage is recorded. `FakeLLM.calls` keeps prompts in RAM for tests (`rootlogic/fake_llm.py:116`) | Debugging, evals ("read the transcripts"), exact replay |
| Each subagent's message history (search queries, `server_tool_use` inputs, result blocks, the nudge) | Local variable `messages` inside `research()`, discarded on return (`rootlogic/llm.py:155`) | Audit trail of *what was searched*; resuming a half-finished worker |
| Raw `SearchHit`s | Passed to `curate()`, then dropped | Measuring search quality ("seen but not cited") |
| Fetched page text | Only inside the API's context window | Source cache, quote verification, reuse across sessions |
| Titles/metadata of **dropped** sources | `add_source` for drops stores only url + reason (`rootlogic/orchestrator.py:169-171`) | Nicer "why dropped" UI |

### 4.4 Options to add it

1. **`transcripts` table** (smallest change): `(id, session_id, llm_call_id, purpose, direction, ts, blocks_json)`. Wrap the SDK call in `_create` so request kwargs and `response.to_json()` are written through a `TranscriptSink` next to `usage_sink`. Store blocks **verbatim**. If you ever resend web-search results, `encrypted_content` must be unchanged or the API returns 400 ([web search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)). Compress with zlib, or keep blobs in files, if size matters.
2. **`page_cache` table** for fetched text, keyed by normalized URL and content hash. This only becomes possible once fetch runs client-side (§2.5). The server-side `web_fetch` result is also in the response blocks, so option 1 captures it indirectly.
3. **Postgres (+ pgvector) for multi-user.** Swap `Store` for a Postgres implementation, use `PostgresSaver` for checkpoints, add a `user_id` to every table, and use pgvector if semantic recall is wanted. Options are compared in [existing brief §4.4](agentic-research-assistant.md#44-long-term-memory-options).
4. **Retention and privacy.** Transcripts contain user questions and third-party page text:
   - Add a retention window (e.g. purge transcripts after N days, keep summaries).
   - Make `forget` cover the new tables (it already cascades through FKs, `rootlogic/store.py:145-148`).
   - Keep the default local.
   - Note that our current web tool versions are not ZDR-eligible by default (§3.1).
   - See [existing brief §4.6](agentic-research-assistant.md#46-privacy-and-retention).

---

## 5. Do our agents use a protocol to talk to each other?

### 5.1 In rootlogic: no network protocol, just hub-and-spoke function calls

The research workers **never talk to each other**, and none of them is a separate process or service. The orchestrator is the only hub.

```
             ┌──────────── Orchestrator (one Python process) ────────────┐
 user ⇄ UI ⇄ │ clarify → plan → [wave: t1 t2 t3 in threads] → reflect → … │
             └───────▲──────────────┬───────────────┬────────────────────┘
                     │ FindingDraft │ prompt str    │ prompt str
                  worker t1      worker t2       worker t3   (no links between workers)
```

In the hand-rolled engine, one round of "communication" looks like this:
1. **Down (orchestrator → worker):** a plain **prompt string** built by `context.research_prompt`. It contains the objective, today's date, the recency rule, user clarifications, the sub-task question, starting queries, and the answers of any `depends_on` tasks (`rootlogic/context.py:34-46`, `rootlogic/orchestrator.py:156-162`).
2. **Worker:** `llm.research(...)` runs its own tool loop (§3.2).
3. **Up (worker → orchestrator):** a **typed `FindingDraft`**. The `submit_findings` tool's strict JSON Schema acts as the *contract*: the model cannot return anything else and still count as done.
4. **Orchestrator:** `curate()` filters the result into a `Finding` (`rootlogic/orchestrator.py:164-184`).
5. **Onward:** condensed findings flow into later prompts through `progress_block` (reflect) and `findings_block` (analyze and report) (`rootlogic/context.py:49-86`), and into dependent tasks' prompts.

So the "protocol" is really a **typed function interface plus shared prompt conventions**, all in one address space.

In the **LangGraph engine** the same design looks different:
- `fan_out` returns one `Send("research", payload)` per ready task. Each worker node receives **only its own slice** (task, plan, context, deps), not the full state (`rootlogic/graph.py:140-151`, `rootlogic/graph.py:246-260`).
- Workers write to `raw`, a state key with a custom reducer that *appends* parallel results (`rootlogic/graph.py:52-56`, `rootlogic/graph.py:67`). `collect` then curates and clears it (`rootlogic/graph.py:262-295`).
- LangGraph describes its runtime as Pregel-inspired message passing in "super-steps", and `Send` as the map-reduce primitive ([graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)). Even so, workers still only message the graph, never each other.

**Human ↔ agent** does have a defined protocol, in two layers:
- **In-process:** `Interaction`, a Python `typing.Protocol` ([PEP 544](https://peps.python.org/pep-0544/)) with `on_event`, `ask`, `review_plan`, `override` (`rootlogic/control.py:35-39`). Commands are `continue | skip | add | note | stop | abort` (`rootlogic/control.py:17`). A thread-safe pause flag is checked only between waves (`rootlogic/control.py:42-64`).
- **Over the network (web UI):**
  - Engine events go out as **Server-Sent Events** on `GET /api/runs/{id}/events`, with sequential `id:`s so a reconnect resumes from `Last-Event-ID` (`rootlogic/web.py:231-260`; [SSE spec](https://html.spec.whatwg.org/multipage/server-sent-events.html)).
  - Decisions come back as `POST /api/runs/{id}/answer` and `/pause` (`rootlogic/web.py:262-275`).
  - `WebInteraction.ask` blocks the engine thread on a condition variable until the POST arrives (`rootlogic/web.py:78-98`).

### 5.2 Real protocols in industry

| Protocol | Connects | Wire format / transport | Key objects | Status (2026-09) |
|---|---|---|---|---|
| **MCP** ([spec 2026-07-28](https://modelcontextprotocol.io/specification/latest)) | an AI app (host/client) ↔ **tools and data** (servers) | JSON-RPC 2.0. Transports: **stdio** (newline-delimited) and **Streamable HTTP** (each message is a POST; the reply is JSON or a request-scoped SSE stream) ([transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)) | Server features: resources, prompts, tools. Client feature: elicitation. Opt-in extensions: Tasks, MCP Apps | In this revision every request carries its own protocol version and capabilities (earlier revisions used an `initialize` handshake), and servers no longer initiate requests. Donated to the Linux Foundation's **Agentic AI Foundation (AAIF)** in Dec 2025 ([Anthropic](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)) |
| **A2A** ([spec](https://a2a-protocol.org/latest/specification/)) | **agent ↔ agent** across services and vendors ("opaque agentic applications", [repo](https://github.com/a2aproject/A2A)) | Three bindings: JSON-RPC 2.0, gRPC, HTTP+JSON/REST. Updates by polling, **SSE streaming**, or push-notification webhooks | **Agent Card** at `/.well-known/agent-card.json` (skills, capabilities, url, security). **Task** with states (submitted, working, input-required, completed, failed, canceled, rejected, auth-required). **Message** (role user/agent, `parts`: text/file/data). **Artifact** (the outputs, made of parts). Operations include `SendMessage`, `SendStreamingMessage`, `GetTask`, `CancelTask`, `SubscribeToTask` | Spec **v1.0.0**. Linux Foundation project since June 2025 ([LF](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)). Joined **AAIF** as a Growth Stage project on 2026-08-27 ([A2A blog](https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/)). Apache-2.0, 25.9k★ |
| **AG-UI** ([events](https://docs.ag-ui.com/concepts/events)) | agent backend ↔ **user interface** | Event stream (SSE or WebSockets) | `RUN_STARTED/FINISHED/ERROR`, `STEP_*`, `TEXT_MESSAGE_*`, `TOOL_CALL_*` (incl. `TOOL_CALL_RESULT`), `STATE_SNAPSHOT/DELTA` (JSON Patch), `SUBAGENT_STARTED/FINISHED/ERROR`, `CUSTOM`. HITL via `RUN_FINISHED` with `outcome: interrupt` | MIT, 16.0k★ |
| **ACP** (IBM/BeeAI "Agent Communication Protocol") | agent ↔ agent | REST | — | **Merged into A2A.** The repo is archived and its README says "ACP is now part of A2A under the Linux Foundation" ([repo](https://github.com/i-am-bee/acp)). Don't adopt. (The unrelated AGNTCY "Agent Connect Protocol" spec repo is also archived: [repo](https://github.com/agntcy/acp-spec)) |

Rule of thumb from the A2A project: A2A is the **horizontal** layer between agents, and MCP is the **vertical** layer from an agent to its tools ([A2A blog](https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/)). Our `/api/runs` + SSE design is already close to AG-UI in spirit: `task.started` would map to `STEP_STARTED`, a human `request` to an interrupt outcome, and so on.

### 5.3 When would rootlogic need A2A, and how would it map?

You would need it when **the workers stop being in-process functions**. Examples:
- running workers as separate services that scale independently;
- calling another vendor's research agent (say, a legal-database agent) as one of the subtasks;
- letting *other* orchestrators call rootlogic.

For a single local app, A2A only adds overhead.

| rootlogic today | A2A equivalent |
|---|---|
| `llm.research(prompt, schema=FindingDraft)` | Orchestrator acts as an **A2A client** and calls `SendMessage` (or `SendStreamingMessage`) on a worker's endpoint |
| Research worker | An **A2A server** publishing an Agent Card with a `research_subtask` skill, its input modes, and `capabilities.streaming` |
| `research_prompt` string | A `Message` with a text part (the prompt) plus a data part (task id, recency, deps) |
| `FindingDraft` via `submit_findings` | An **Artifact** with a data part holding the `FindingDraft` JSON. The Task reaches `completed` |
| `LLMError` / did-not-converge | Task state `failed`. Pause/abort becomes `CancelTask` |
| Clarifying question inside a worker (not possible today) | Task state `input-required`. The orchestrator relays the question to the user through `Interaction.ask` |
| Events (`task.started`, …) | Task status updates streamed over SSE |

Our code already has the right pieces for this: typed schemas at the boundary, a task id, and a status lifecycle (`rootlogic/models.py:114`). Mapping to A2A would add transport and auth. It would not require redesigning the orchestrator.

**MCP** is the cheaper step, and it points the other way: expose `search`/`fetch` (or the whole `research` capability) as an MCP server so other hosts can use them, or *consume* existing search MCP servers ([existing brief §7](agentic-research-assistant.md#7-tools-for-research-subagents)).

---

## 6. "Is it like data abstraction, with tasks resolved while hidden from the user?"

### 6.1 Where the analogy holds

- **Encapsulation and information hiding.** `llm.research(...)` works like a function or an abstract data type. The caller knows only the interface (prompt in, `FindingDraft` + hits out, or an exception). How many searches ran, which queries, which pages were fetched, how often it paused or was nudged — all of that is internal and invisible to the orchestrator (`rootlogic/llm.py:142-176`). Because of this, `FakeLLM` can replace the entire search machinery with one line per schema (`rootlogic/fake_llm.py:132-141`). Replaceable implementations are exactly what abstraction buys you.
- **Separation of concerns.** The module docstring says it: "The LLM decides *what* to research … this module decides *how* the run proceeds" (`rootlogic/orchestrator.py:4-5`). Prompt building is pure functions in `context.py`. Persistence is `store.py`. Human I/O sits behind `Interaction`.
- **Context isolation, the part that matters for LLMs.** Each worker has **its own context window**. Its raw search results never enter the orchestrator's prompts, only the condensed finding does. Anthropic describes the same mechanism: "Subagents facilitate compression by operating in parallel with their own context windows", which also gives "separation of concerns" ([Anthropic, multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).
- **Futures and async task queues.** `_run_wave` submits every ready task to a `ThreadPoolExecutor` and consumes results with `as_completed` (`rootlogic/orchestrator.py:142-153`). A `Future` is a handle to a result that isn't ready yet ([concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)). The orchestrator doesn't care how the result is produced, only that it eventually resolves to a value or an exception.
- **Message passing.** Workers share no mutable state. `_research_task` "only calls the LLM, never the UI or shared state" (`rootlogic/orchestrator.py:157`). They communicate only by returning values, and the LangGraph version literally sends `Send` messages to a reducer.

### 6.2 Where it breaks down

- **Not hidden from the user.** Transparency is a design requirement, so rootlogic deliberately *surfaces* what abstraction would hide:
  - the plan for approval and editing (`rootlogic/orchestrator.py:65-68`);
  - every stage transition as an event;
  - each task's status;
  - **every dropped source with its reason** (`rootlogic/orchestrator.py:169-171`);
  - token and cost totals per purpose (`rootlogic/store.py:215-220`).

  The abstraction boundary sits between *orchestrator and worker*, not between *system and user*.
- **Not "constantly" resolving in the background.** Work happens in **bounded waves**:
  - at most `max_parallel=4` workers, `max_tasks=10`, `max_rounds=2` reflection rounds, `max_searches=5` per worker (`rootlogic/orchestrator.py:29-34`);
  - at most 8 API rounds per worker (`rootlogic/llm.py:158`);
  - a **checkpoint** between waves where the user can pause, skip, add, note, stop or abort (`rootlogic/orchestrator.py:249-256`).

  Nothing runs after the report is written, and there is no daemon or queue that outlives the session.
- **It is a leaky abstraction on purpose.** The `search_queries` and `rationale` fields of a sub-task are visible plans, and a worker's `gaps` and `confidence` are part of its public result. The worker's output was designed for the next consumer (the critic and the writer). It is not an opaque token.

### 6.3 What is actually hidden today, and how to expose it

- Hidden today: individual **search queries**, fetched URLs, `pause_turn` resumes, nudges and per-round token counts inside a worker. The only visible trace is `task.done … searched=N` (`rootlogic/orchestrator.py:180-184`) plus one `llm_calls` row per API round.
- One way to expose them: give `LLM.research` an optional `on_step` callback and have the orchestrator emit a `subagent.tool_call` event (`{task, tool, query|url, round}`) for every `server_tool_use` block.
- That would put worker internals into the action log and SSE stream (and map neatly to AG-UI `TOOL_CALL_*` events) without changing the abstraction's return type.

---

## What we could change next

- **`SearchProvider` protocol + client-side search** (Tavily/Brave/Exa) so `research()` doesn't depend on a vendor's hosted search (§2.2, §2.5).
- **`OpenAICompatLLM` adapter** (one class: OpenAI, Ollama `/v1`, vLLM, OpenRouter) implementing `structured` via `response_format` and `research` via our own tool loop (§2.4e).
- **Price table in config plus `provider` column** in `llm_calls`, so costs stay right across providers (§2.5).
- **`transcripts` table** (verbatim request/response blocks per call) behind a `TranscriptSink`, with a retention setting (§4.4).
- ~~`SearchProvider` protocol~~: done (`rootlogic/search.py`, `--search tavily`).
- ~~OpenAI-compatible `LLM` adapter~~: done (`rootlogic/openai_llm.py`, `--provider openai`).
- **Store raw `SearchHit`s and dropped-source titles** to measure search quality (§4.3).
- **`subagent.tool_call` events** via an `on_step` callback, so the action log shows each query and fetch (§6.3).
- **Pin tool versions deliberately:** consider `web_search_20260318` with `response_inclusion`, or `allowed_callers: ["direct"]` where ZDR matters (§3.1).
- **Optional MCP server** exposing `search`/`fetch`/`research`. **A2A only** if workers become separate services or third-party agents join (§5.3).

---

## Source list

**Anthropic (primary)**
- Structured outputs — https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Effort — https://platform.claude.com/docs/en/build-with-claude/effort
- Refusals and fallback (server-side fallback beta) — https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
- Web search tool — https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
- Web fetch tool — https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
- Server tools (pause_turn, mixed turns, ZDR/allowed_callers) — https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools
- Search result content blocks — https://platform.claude.com/docs/en/build-with-claude/search-results
- How we built our multi-agent research system — https://www.anthropic.com/engineering/multi-agent-research-system
- Donating MCP / AAIF — https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation

**Other model providers and local runtimes**
- OpenAI structured outputs — https://developers.openai.com/api/docs/guides/structured-outputs
- OpenAI web search — https://developers.openai.com/api/docs/guides/tools-web-search
- Gemini structured output — https://ai.google.dev/gemini-api/docs/structured-output
- Gemini Grounding with Google Search — https://ai.google.dev/gemini-api/docs/google-search
- Ollama structured outputs — https://docs.ollama.com/capabilities/structured-outputs
- Ollama OpenAI compatibility — https://docs.ollama.com/api/openai-compatibility
- vLLM structured outputs — https://docs.vllm.ai/en/latest/features/structured_outputs.html
- OpenRouter web search — https://openrouter.ai/docs/features/web-search

**Abstraction libraries**
- LiteLLM repo / LICENSE — https://github.com/BerriAI/litellm , https://github.com/BerriAI/litellm/blob/main/LICENSE
- LiteLLM JSON mode — https://docs.litellm.ai/docs/completion/json_mode
- LiteLLM web search — https://docs.litellm.ai/docs/completion/web_search
- Pydantic AI models — https://pydantic.dev/docs/ai/models/overview/
- Pydantic AI built-in tools — https://pydantic.dev/docs/ai/tools-toolsets/builtin-tools/
- LangChain models (`init_chat_model`, `with_structured_output`) — https://docs.langchain.com/oss/python/langchain/models
- LangGraph persistence — https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Graph API (Send, reducers, super-steps) — https://docs.langchain.com/oss/python/langgraph/graph-api

**Storage**
- SQLite FTS5 — https://www.sqlite.org/fts5.html
- SQLite WAL — https://www.sqlite.org/wal.html

**Protocols**
- MCP specification 2026-07-28 — https://modelcontextprotocol.io/specification/latest
- MCP transports — https://modelcontextprotocol.io/specification/2026-07-28/basic/transports
- A2A specification — https://a2a-protocol.org/latest/specification/
- A2A repo — https://github.com/a2aproject/A2A
- Linux Foundation launches A2A project — https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents
- A2A joins AAIF (2026-08-27) — https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/
- AG-UI events — https://docs.ag-ui.com/concepts/events
- ACP (archived, merged into A2A) — https://github.com/i-am-bee/acp ; AGNTCY ACP spec (archived) — https://github.com/agntcy/acp-spec
- Server-Sent Events (WHATWG) — https://html.spec.whatwg.org/multipage/server-sent-events.html

**Python**
- PEP 544 (Protocols) — https://peps.python.org/pep-0544/
- concurrent.futures — https://docs.python.org/3/library/concurrent.futures.html

*GitHub stars and licenses: GitHub REST API, 2026-09-22.*
