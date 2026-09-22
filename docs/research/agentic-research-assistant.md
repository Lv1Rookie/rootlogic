# Agentic AI Personal Research Assistant — Research Brief

*Compiled 2026-09-22. Primary sources only (vendor docs, engineering blogs, repos, specs, arXiv). GitHub star counts were pulled from the GitHub REST API on 2026-09-22 and will drift. Anything I could not confirm from a primary source is marked **unverified**.*

---

## TL;DR

1. **Use orchestrator-workers, not a swarm.** Anthropic's production Research system runs one lead agent that plans and spawns parallel subagents, plus a separate CitationAgent. It beat a single agent by 90.2% on Anthropic's internal eval, but it costs about 15× the tokens of a chat, compared with about 4× for a single agent ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)).
2. **Scale effort explicitly in the prompt.** Simple fact-finding: 1 agent, 3–10 tool calls. Comparisons: 2–4 subagents with 10–15 calls each. Complex topics: 10+ subagents. Run subagents and tool calls in parallel; parallelism cut research time by up to 90% ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)).
3. **Have subagents return condensed findings** (about 1,000–2,000 tokens) and keep bulky content outside the context window, fetched just-in-time ([Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)). **Write the final report in one pass from the gathered notes.** LangChain found that section-by-section writing by parallel subagents produced disjointed reports ([L. Martin / LangChain](https://rlancemartin.github.io/2025/07/30/bitter_lesson/)).
4. **Ask clarifying questions before researching, with a cheap model.** ChatGPT Deep Research does this with an intermediate model. The Deep Research API skips it and expects the developer to add it ([OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/deep_research_api/introduction_to_deep_research_api.ipynb)). Gemini Deep Research shows a multi-step plan that the user can "revise or approve" ([Google](https://blog.google/products/gemini/google-gemini-deep-research/)). Implement both. Together they cover the "adaptive" and "transparent" grading criteria.
5. **For this assignment, start with the raw API loop.** Anthropic recommends plain API calls over frameworks because frameworks "obscure debugging" ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents)). The graders explicitly want orchestration "in clear, testable code", so a small hand-written orchestrator is the better showcase.
6. **Store an append-only event log in SQLite.** Every plan, tool call, LLM call (with `usage`), user override and source becomes a row. That one table gives you the action log UI, cost accounting, resume after crash (replay), and a demo-able audit trail. Add FTS5 (built into SQLite) for long-term memory. Add `sqlite-vec` only if you need semantic recall ([SQLite FTS5](https://www.sqlite.org/fts5.html), [sqlite-vec](https://github.com/asg017/sqlite-vec)).
7. **Account for tokens precisely.** Anthropic's `usage` has `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, a `cache_creation` breakdown by TTL, and `server_tool_use.{web_search_requests, web_fetch_requests}`. Total input is the sum of the three input fields ([Prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [SDK types](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/usage.py)).
8. **Search tools.** Anthropic's server-side `web_search` costs $10 per 1,000 searches plus tokens, with `allowed_domains`/`blocked_domains`, `max_uses`, `page_age` and always-on citations. `web_fetch` has no extra charge ([web search](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool), [web fetch](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-fetch-tool)). Tavily is the easiest client-side option: 1,000 free credits per month, no card ([Tavily](https://docs.tavily.com/documentation/api-credits)). The Bing Search API was retired on 2025-08-11 ([Microsoft](https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement)). Google Custom Search JSON API no longer accepts new customers ([Google](https://developers.google.com/custom-search/v1/overview)).
9. **Treat web content as untrusted.** OWASP LLM01 recommends segregating external content, least privilege, and human approval for high-risk actions ([OWASP](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)). Anthropic's `web_fetch` refuses to fetch URLs that appear only in the model's own output, as an exfiltration defense ([web fetch docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-fetch-tool)).
10. **Use protocols as optional differentiators.** The candidates are MCP (spec `2026-07-28`, JSON-RPC 2.0, stdio + Streamable HTTP) for tools ([MCP spec](https://modelcontextprotocol.io/specification/latest)), OpenTelemetry GenAI semantic conventions for traces (still in development) ([OTel](https://opentelemetry.io/docs/specs/semconv/gen-ai/)), and AG-UI for streaming agent events to a UI ([AG-UI](https://docs.ag-ui.com/introduction)). None is needed to pass. Exposing your search tool as an MCP server is cheap originality.
11. **Evaluate on end states.** Use an LLM-judge rubric (factual accuracy, citation accuracy, completeness, source quality, tool efficiency) on about 20 realistic queries ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)). Grade outcomes, not paths, and read transcripts ([Anthropic evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
12. **Recommendation: Architecture A.** Python, Anthropic Messages API, a hand-rolled orchestrator-workers loop, asyncio parallel workers, SQLite event log + FTS5 memory, FastAPI + SSE web UI (or a Textual CLI). Keep an `LLMClient` protocol so tests run against a fake model. LangGraph (Architecture B) is the runner-up if you want interrupts and checkpointing out of the box.

---

## 1. Agentic development best practices

### 1.1 Workflows vs agents (Anthropic, "Building effective agents", 2024-12-19)
- **Definitions.** Workflows are "LLMs and tools orchestrated through predefined code paths." Agents are systems "where LLMs dynamically direct their own processes and tool usage" ([source](https://www.anthropic.com/engineering/building-effective-agents)).
- **Five patterns** ([source](https://www.anthropic.com/engineering/building-effective-agents)):

| Pattern | Use when | Research-assistant mapping |
|---|---|---|
| Prompt chaining | Fixed sequence of subtasks | clarify → plan → execute → synthesize → verify |
| Routing | Inputs fall into distinct classes | classify query type: factual / comparative / literature review / news |
| Parallelization (sectioning, voting) | Independent subtasks, or confidence via votes | parallel subagents per subtopic; voting for contradiction adjudication |
| **Orchestrator-workers** | Subtasks can't be predicted up front | lead planner spawns search/read workers dynamically |
| **Evaluator-optimizer** | Clear evaluation criteria exist, and iteration helps | critic checks draft report for uncited claims and gaps, then loops |

- **Three principles.** Keep it simple, make the planning steps transparent, and document and test tools carefully. Use "poka-yoke" tool arguments that make mistakes hard ([source](https://www.anthropic.com/engineering/building-effective-agents)).
- **Frameworks.** Start with direct API calls, since many patterns take only a few lines. Frameworks add abstraction that obscures prompts and responses ([source](https://www.anthropic.com/engineering/building-effective-agents)).

### 1.2 Anthropic's multi-agent Research system (engineering post)
Source for all bullets: [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).
- **Architecture.** A lead agent (Claude Opus 4 at the time) plans and spawns subagents (Sonnet 4) that search in parallel with interleaved thinking. A **CitationAgent** attributes claims to sources at the end.
- **Cost.** Agents use about 4× the tokens of chat; multi-agent systems about 15×. Token usage alone explained 80% of performance variance on BrowseComp. This means multi-agent only pays off for high-value, breadth-first, parallelizable tasks.
- **Result.** Multi-agent (Opus lead + Sonnet subagents) beat single-agent Opus 4 by 90.2% on the internal research eval.
- **Effort-scaling rules embedded in the prompt:**
  - simple: 1 agent, 3–10 calls
  - comparison: 2–4 subagents × 10–15 calls
  - complex: 10+ subagents
- **Parallelism.** The lead spins up 3–5 subagents in parallel, and each subagent makes 3+ tool calls in parallel. This cut time by up to 90% on complex queries.
- **Subagent prompt design.** Each delegation needs an objective, output format, tool/source guidance and clear task boundaries. Vague delegations such as "research the semiconductor shortage" caused duplicated work.
- **Search strategy.** Go broad then narrow: start with short, broad queries, then drill down. Use extended thinking to plan and interleaved thinking to evaluate results after each tool call.
- **Tool descriptions.** Bad descriptions send agents down wrong paths. Claude itself was used to rewrite tool descriptions.
- **Evaluation.** Start with about 20 queries from real usage. An LLM judge scores 0.0–1.0 on factual accuracy, citation accuracy, completeness, source quality and tool efficiency. Human testers caught a bias toward SEO content farms over authoritative sources.
- **End-state evaluation.** For agents that mutate state, judge the final state rather than the path, because agents find alternative valid routes.
- **Long-horizon memory.** Summarize completed phases and store the plan in external memory before hitting the context limit (200k at the time). Spawn fresh subagents with clean contexts and hand off via memory.
- **Reliability.** Checkpoint so runs resume rather than restart. Let the model know when a tool fails so it can adapt. Use full production tracing, plus "rainbow deployments" for long-running agents.
- **Known limitation.** The lead waits synchronously for each batch of subagents, so it can't steer mid-flight. Async execution is harder to coordinate.

### 1.3 Context engineering (Anthropic, 2025-09-29)
Source: [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
- **Context rot.** Recall degrades as context grows. Treat tokens as a finite "attention budget" and aim for "the smallest set of high-signal tokens."
- **System prompt altitude.** Avoid both brittle hard-coded if/else prompts and vague platitudes. Give heuristics plus examples.
- **Just-in-time retrieval.** Keep references (URLs, source IDs) in context, not full documents, and let the agent fetch on demand.
- **Long-horizon techniques:**
  - *compaction*: summarize and restart the context
  - *structured note-taking*: persistent notes outside the context
  - *sub-agent architectures*: subagents return condensed summaries of about 1,000–2,000 tokens

### 1.4 Writing tools for agents (Anthropic, 2025-09-11)
Source: [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents).
- **Build fewer, workflow-shaped tools** instead of wrapping every API endpoint. For example, `search_and_summarize` rather than separate list/get/filter tools.
- **Namespace tools**, e.g. `web_search`, `web_fetch`, `arxiv_search`, `memory_search`.
- **Return meaningful context.** Use human-readable identifiers, not UUIDs. Add a `response_format: "concise" | "detailed"` parameter.
- **Be token-efficient.** Paginate, filter and truncate with sensible defaults. Claude Code caps tool responses at 25,000 tokens by default. Error messages should steer the agent toward a better strategy.
- **Iterate against evals**, and let an agent analyze the transcripts and refactor the tools.

### 1.5 OpenAI, "A practical guide to building agents" (PDF)
Source: [OpenAI PDF](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) (the PDF body did not parse in my fetcher; bullets are confirmed via the [landing page](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/) search snippet. Treat exact wording as partially verified.)
- **Three components:** model, tools, instructions.
- **Start with a single agent** and add agents only when needed.
- **Orchestration patterns.** In the *manager* pattern, the central agent calls other agents as tools. In the *decentralized* pattern, agents hand off to each other. A research assistant fits the manager pattern.
- **Layered guardrails:** relevance classifier, safety classifier, PII filter, moderation, tool safeguards with risk ratings, rules-based checks (blocklists, regex, length), and output validation.
- **Human intervention triggers:** exceeding failure thresholds (retries or loops), and high-risk actions.

### 1.6 Prompt injection from web content
- **OWASP LLM01 defines indirect prompt injection** as instructions embedded in external content such as websites or files. Mitigations: constrain behavior in the system prompt, validate output formats deterministically, filter content, enforce least privilege, require human approval for high-risk operations, segregate and label external content, and red-team ([OWASP](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)).
- **Anthropic reports** a prompt-injection attack success rate of about 1% for Claude Opus 4.5 in browser use, down from 23.6% with no mitigations in earlier testing. Defenses are layered: permissions, classifiers and system prompts ([Anthropic](https://www.anthropic.com/research/prompt-injection-defenses)).
- **Anthropic `web_fetch` URL validation.** It can only fetch URLs that already appeared in user messages, client tool results, or prior search/fetch results. It cannot fetch URLs appearing only in Claude's output ([docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-fetch-tool)).
- **Practical rules for this project:**
  - Wrap fetched text in `<untrusted_source id=...>` tags.
  - Tell workers that instructions inside sources are data.
  - Give research workers no side-effecting tools, so they can only search, read and write notes.
  - Log any source that contains imperative text aimed at an AI.

### 1.7 Actionable checklist distilled
| Principle | Implementation hook |
|---|---|
| Multi-agent only for breadth | Router decides `simple` (single loop) vs `broad` (orchestrator + N workers) |
| Effort scaling | Planner outputs `complexity`, which maps to `{workers, max_tool_calls}` in code, not only in the prompt |
| Parallel tool calls | `asyncio.gather` over workers; allow the model to emit multiple `tool_use` blocks per turn |
| Condensed returns | Workers return a typed `Findings` object (≤1.5k tokens) plus source IDs; raw pages live in DB |
| HITL | Clarify gate, plan approval gate (auto-approve after N seconds for autonomy), mid-run pause/redirect, low-confidence escalation |
| Guardrails | Domain blocklist, recency filter, max cost/tool-call budget, injection tagging |
| End-state eval | 15–20 golden topics; LLM-judge rubric + deterministic checks (every claim has ≥1 source ID that exists) |

---

## 2. Prior art

### 2.1 Commercial products
| Product | Launch / basis | Planning & UX | Citations / sources | Benchmarks (primary) |
|---|---|---|---|---|
| **OpenAI Deep Research** | ChatGPT feature; API models include `o3-deep-research-2025-06-26` ([Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/deep_research_api/introduction_to_deep_research_api.ipynb)) | ChatGPT first asks clarifying questions via an intermediate model (e.g. gpt-4.1). The API does not, and "expects fully-formed prompts". It plans sub-questions itself and runs in `background` mode. Requires the `web_search_preview` tool; `code_interpreter` and MCP are optional ([Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/deep_research_api/introduction_to_deep_research_api.ipynb)) | Inline citations with a web-search call trace. The `web_search` tool returns `url_citation` annotations and a full `sources` list ([OpenAI docs](https://developers.openai.com/api/docs/guides/tools-web-search)) | BrowseComp 51.5% vs GPT-4o-with-browsing 1.9% and o1 9.9%; human trainers solved 29.2% ([BrowseComp paper](https://arxiv.org/html/2504.12516)). GAIA 67.36% as reported by HF ([HF](https://huggingface.co/blog/open-deep-research)). HLE ~26.6% is **unverified** (openai.com blocked the fetch) |
| **Gemini Deep Research** | 2024-12-11, first on Gemini 1.5 Pro ([Google](https://blog.google/products/gemini/google-gemini-deep-research/)); upgraded to 2.0 Flash Thinking in Mar 2025 ([Google](https://blog.google/products/gemini/new-gemini-app-features-march-2025/)) | Shows a **multi-step research plan to "revise or approve"** before it starts. Streams its thoughts while browsing ([Google](https://blog.google/products/gemini/new-gemini-app-features-march-2025/)) | Report with linked sources; exports to Google Docs | — |
| **Claude Research** | 2025-04-15 ([Claude blog](https://claude.com/blog/research)) | Agentic multi-search that "build[s] on each other"; orchestrator-workers design ([engineering](https://www.anthropic.com/engineering/multi-agent-research-system)) | Dedicated CitationAgent pass | Internal eval only (+90.2% vs single agent) |
| **Perplexity Deep Research / `sonar-deep-research`** | API model page exists ([docs](https://docs.perplexity.ai/getting-started/models/models/sonar-deep-research)) | "Exhaustive searches and generating comprehensive reports" | Citations | Launch-blog numbers (HLE, SimpleQA) **unverified** (403 on fetch) |

### 2.2 Open-source systems (stars as of 2026-09-22, via GitHub API)
| Project | Lang / License / Stars | Architecture & planning | Citations & source filtering | Maturity notes |
|---|---|---|---|---|
| **GPT Researcher** ([repo](https://github.com/assafelovic/gpt-researcher)) | Python / Apache-2.0 / 29.6k | Planner agent generates research questions. Execution/crawler agents gather per question. A publisher aggregates. "Deep research" mode is recursive with configurable depth and breadth (~5 min, ~$0.40 per task per README). A multi-agent variant uses LangGraph and AG2 | Aggregates 20+ sources and tracks a source for every summary. Retrievers include Tavily and MCP. Exports PDF/DOCX/MD | Very active, widely forked (4.0k forks) |
| **STORM / Co-STORM** (Stanford OVAL) ([repo](https://github.com/stanford-oval/storm)) | Python / MIT / 31.5k | STORM has a pre-writing stage: *perspective-guided question asking* plus a *simulated conversation* between a writer and a topic expert, which produces an outline, then the article. Co-STORM adds a moderator agent, lets a human join the discourse, and keeps a dynamic **mind map** | Citation-grounded Wikipedia-style articles. Many retrievers: You.com, Bing, Tavily, DuckDuckGo, Brave, SearXNG, Serper, Google, Azure AI Search, VectorRM | Papers: STORM NAACL 2024 ([arXiv:2402.14207](https://arxiv.org/abs/2402.14207)), Co-STORM EMNLP 2024 ([arXiv:2408.15232](https://arxiv.org/abs/2408.15232)) |
| **LangChain open_deep_research** ([repo](https://github.com/langchain-ai/open_deep_research)) | Python / MIT / 12.7k. **Repo is archived** (GitHub API `archived: true`, checked 2026-09-22) | LangGraph with a clarify step, then a supervisor, then parallel researcher subagents, then compression, then a one-shot report. Separate configurable models for summarization, research, compression and report. Legacy `src/legacy/` holds a plan-and-execute workflow with human-in-the-loop | Tavily default, native Anthropic/OpenAI web search, MCP | #6 on DeepResearch Bench, RACE 0.4344 (2025-08-02, README) |
| **HF smolagents Open Deep Research** ([blog](https://huggingface.co/blog/open-deep-research)) | Python / Apache-2.0 (smolagents 29.4k) | A `CodeAgent` writes Python actions. Tools: a text browser and a text inspector adapted from Magentic-One | — | GAIA validation 55.15% (vs 46% previous OSS SOTA by Magentic-One). Switching code actions to JSON dropped the score to 33%. Code actions use about 30% fewer steps. Built in about 24 h (2025-02-04) |
| **dzhng/deep-research** ([repo](https://github.com/dzhng/deep-research)) | TypeScript / MIT / 19.7k | Recursive loop driven by `breadth` and `depth`. It generates SERP queries, extracts "learnings" and follow-up directions, then recurses. Goal: under 500 LoC | Firecrawl for search and extraction. Markdown report | Great minimal reference |
| **Jina node-DeepResearch** ([repo](https://github.com/jina-ai/node-DeepResearch)) | TypeScript / Apache-2.0 / 5.2k | Actions are search, visit, reflect (sub-questions) and answer. It loops "until an answer is found (or the token budget is exceeded)" | Jina Reader and Search. Aims at concise correct answers, not long reports | Explicit token-budget stopping |
| **Microsoft Magentic-One** ([MSR](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)) | Python (AutoGen) | Orchestrator keeps a **Task Ledger** (facts, guesses, plan) and a **Progress Ledger** (per-step self-reflection), with outer and inner loops and re-planning on stall. Agents: WebSurfer, FileSurfer, Coder, ComputerTerminal | — | Evaluated on GAIA, AssistantBench and WebArena (2024-11-04). AutoGen is now in maintenance mode ([repo](https://github.com/microsoft/autogen)) |

**Pattern to borrow for originality:** combine
- Magentic-One's *ledger* (explicit facts / open questions / plan, re-planned on stall),
- STORM's *perspective-guided questions* (planner generates 3–5 stakeholder perspectives, then sub-questions per perspective),
- Jina's *token-budget stop*,
- Anthropic's *citation pass*.

### 2.3 Benchmarks (primary)
| Benchmark | What it measures | Size | Source |
|---|---|---|---|
| **BrowseComp** (OpenAI, 2025-04) | Persistent browsing for hard-to-find, entangled facts | 1,266 Qs | [arXiv:2504.12516](https://arxiv.org/abs/2504.12516) |
| **GAIA** | Real-world assistant tasks (reasoning, multimodality, browsing, tools). Humans 92% vs GPT-4 with plugins 15% at release | 466 Qs (300 held out for leaderboard) | [arXiv:2311.12983](https://arxiv.org/abs/2311.12983) |
| **Humanity's Last Exam** | Closed-ended frontier academic questions not answerable by quick retrieval; CC BY 4.0 | 2,500 Qs | [arXiv:2501.14249](https://arxiv.org/abs/2501.14249) |
| **DeepResearch Bench** | PhD-level research *reports*. RACE scores report quality; FACT scores citation count and accuracy | 100 tasks, 22 fields | [arXiv:2506.11763](https://arxiv.org/abs/2506.11763) |

For a take-home, borrow the **FACT idea**: count effective citations and check that each cited URL actually supports its claim. It is cheap to demo as a "citation accuracy" metric.

---

## 3. Framework trade-offs

Stars and licenses are from the GitHub API on 2026-09-22 unless another link is given.

| Option | Lang | License | Abstraction / control | HITL / interrupts | Persistence / checkpointing | Observability | Multi-provider | Learning curve | Fit for this assignment |
|---|---|---|---|---|---|---|---|---|---|
| **Anthropic Messages API, raw tool loop** ([docs](https://platform.claude.com/docs/en/api/messages)) | Py/TS/Go/Java/etc. | SDK MIT (**unverified** for all SDKs) | Lowest; you own the loop. SDK also has a beta "tool runner" ([Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)) | DIY (easy: loop stops at `stop_reason`, waits for user) | DIY (your event log) | DIY; `usage` on every response | Claude only (wrap behind an interface) | Low | **Best for "clear, testable orchestration" and originality** |
| **LangGraph** ([repo](https://github.com/langchain-ai/langgraph)) | Python (+JS) | MIT, 42.1k★ | Low-level graph/state machine | First-class: `interrupt()` + `Command(resume=…)`, requires checkpointer + `thread_id`. The node **re-runs from its start** on resume ([docs](https://docs.langchain.com/oss/python/langgraph/interrupts)) | `InMemorySaver`, `SqliteSaver`, `PostgresSaver`; time-travel; `Store` for cross-thread memory ([docs](https://docs.langchain.com/oss/python/langgraph/persistence)) | LangSmith | Yes (LangChain chat models) | Medium | Strong runner-up; HITL and checkpointing for free |
| **OpenAI Agents SDK** ([repo](https://github.com/openai/openai-agents-python)) | Python (+JS) | MIT, 29.6k★ | Light: Agents, Handoffs, Guardrails, Sessions, Tracing | Built-in human-in-the-loop mechanisms | Sessions (SQLAlchemy/SQLite, optional Redis) | Built-in tracing | "100+ LLMs" via LiteLLM / any-llm | Low | Good, especially if using GPT |
| **Claude Agent SDK** ([docs](https://code.claude.com/docs/en/agent-sdk/overview)) | Python, TS | Repo MIT (8.1k★ py). Use governed by Anthropic Commercial ToS | High: "Claude Code as a library". Runs the Claude Code binary with built-in tools, subagents, hooks, permissions, MCP | Permissions + hooks | Sessions (resume/fork) | Hooks | Claude only | Low–Med | Powerful, but the orchestration is hidden in Claude Code, so it scores less on "your orchestration logic" |
| **Google ADK** ([repo](https://github.com/google/adk-python)) | Py, Java, Go, TS, Kotlin | Apache-2.0, 21.6k★ | Medium; workflow agents plus graph orchestration | Tool confirmation (HITL) | Session/memory services | Built-in eval tooling | "Model-agnostic", Gemini-optimized | Medium | Good if Gemini |
| **CrewAI** ([repo](https://github.com/crewAIInc/crewAI)) | Python | MIT, 58.9k★ | High (Crews = role-play autonomy; Flows = event-driven control); no LangChain dependency | Human input supported | Flow state | AMP suite (commercial) tracing | Yes | Low | Fast demos, but role abstraction hides the planning logic |
| **AutoGen** ([repo](https://github.com/microsoft/autogen)) | Python/.NET | CC-BY-4.0 + MIT, 61.1k★ | Core / AgentChat / Extensions | Yes | Yes | Yes | Yes | Medium | **Maintenance mode**; Microsoft says new users should use **Microsoft Agent Framework** (MIT, 13.7k★, GA, graph workflows, checkpointing, HITL, OTel, A2A, MCP) ([repo](https://github.com/microsoft/agent-framework)) |
| **AG2** ([repo](https://github.com/ag2ai/ag2)) | Python | Apache-2.0, 4.9k★ | Community fork of AutoGen. v1.0 is protocol-driven; classic moved to `ag2-classic` | `context.input()` + `hitl_hook` | — | — | Yes | Medium | Niche |
| **smolagents** ([repo](https://github.com/huggingface/smolagents)) | Python | Apache-2.0, 29.4k★ | Minimal (<1,000 LoC core). `CodeAgent` vs `ToolCallingAgent` | Manual | Manual | OTel via integrations (**unverified**) | LiteLLM, HF, Ollama, OpenAI, Anthropic, Bedrock | Low | Good for local models. The local Python executor "can be bypassed", so sandbox it (E2B/Docker/Modal) |
| **Pydantic AI** ([repo](https://github.com/pydantic/pydantic-ai)) | Python | MIT, 20.1k★ | Light, type-safe; Pydantic Graph for control flow | Tool-approval HITL | Durable execution via Temporal, DBOS, Prefect, Restate… | OTel-native + Logfire | Very broad | Low–Med | Excellent for typed plans and findings; pairs well with a hand-rolled orchestrator |
| **LlamaIndex** ([repo](https://github.com/run-llama/llama_index)) | Python/TS | MIT, 52.3k★ | Data/RAG-centric; event-driven Workflows | Via Workflows | Yes | Integrations | Yes | Medium | Best if the assignment leans on documents and RAG |
| **Mastra** ([repo](https://github.com/mastra-ai/mastra)) | TypeScript | Apache-2.0 core + Enterprise License for `ee/` | Agents + graph workflows (`.then/.branch/.parallel`) | Suspend/resume | Yes (memory) | Built-in evals + observability | 40+ providers | Medium | Best TS option |
| **Vercel AI SDK** ([repo](https://github.com/vercel/ai)) | TypeScript | Apache-2.0 ([LICENSE](https://github.com/vercel/ai/blob/main/LICENSE)), 26.9k★ | Low-level core (`generateText`, `ToolLoopAgent`) + UI hooks (`useChat`) | Tool invocation states in UI (approval flow **unverified**) | DIY | Via integrations | Many providers | Low | Best for a Next.js streaming front end |

### 3.1 LLM choice trade-offs
- **Hosted Claude** (prices from [models overview](https://platform.claude.com/docs/en/about-claude/models/overview), 2026-09-22):

| Model | Price per MTok (in / out) | Notes |
|---|---|---|
| `claude-opus-5` | $5 / $25 | Anthropic's recommended default |
| `claude-sonnet-5` | $2 / $10 | — |
| `claude-haiku-4-5` | $1 / $5 | 200K context |
| `claude-fable-5-1` | $10 / $50 | Demanding long-horizon work |

  Batch is 50% off. Cache reads cost 0.1× input; 5-minute cache writes 1.25×, 1-hour writes 2× ([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). A natural split: Opus 5 or Sonnet 5 as lead/planner, Haiku 4.5 for per-source summarization.
- **Hosted OpenAI:** gpt-5.5 $5 / $30, gpt-5.4-mini $0.75 / $4.50, gpt-5-mini $0.25 / $2.00 per MTok. Web search $10 per 1k calls plus content tokens ([OpenAI pricing](https://developers.openai.com/api/docs/pricing)).
- **Hosted Gemini:** Flash-tier models listed at $0.75 in / $3.75 out per MTok through 2026-12-31. Grounding with Google Search on Gemini 3.x: 5,000 free requests per month, then $14 per 1k ([Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)).
- **Local (Ollama/vLLM):** Ollama supports tool calling, including parallel calls and streaming with tools ([Ollama docs](https://docs.ollama.com/capabilities/tool-calling)). Local models are free and private, but multi-step planning and citation faithfulness are weaker, and the demo depends on the machine. Offer it as a pluggable provider, not the demo default.
- **Recommendation:** use a provider interface (`LLMClient.complete(messages, tools, schema) -> Response`) with an Anthropic implementation plus a `FakeLLM` for tests. Optionally add LiteLLM or an OpenAI-compatible adapter so Ollama works.

---

## 4. Storage mechanisms

### 4.1 What to persist
| Entity | Why | Notes |
|---|---|---|
| **sessions** | One row per research session (topic, status, settings, budget) | Resume key |
| **messages** | Raw user/assistant/tool content blocks | Anthropic requires web-search results (`encrypted_content`) to be sent back unchanged in multi-turn requests; otherwise you get a 400 ([web search docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool)). Store blocks verbatim as JSON |
| **events** (append-only) | Action log, UI timeline, audit, replay/resume | Event-sourcing: state = fold(events). Types: `plan_proposed`, `plan_approved`, `plan_edited`, `task_started`, `tool_called`, `tool_result`, `source_accepted`, `source_rejected(reason)`, `clarification_asked`, `user_override`, `contradiction_found`, `report_drafted`, `verification_failed`, `session_completed` |
| **llm_calls** | Cost and latency accounting | One row per API call, with model, role (planner/worker/summarizer/judge), usage fields, stop_reason, ms |
| **tasks** | Plan DAG (id, parent, kind, status, depends_on, assigned_worker) | Lets the user edit or skip tasks |
| **sources** | Document cache (url, canonical_url, title, publisher, published_at, retrieved_at, content_hash, text, credibility score, recency score, status) | Dedupe by canonical URL + content hash; reuse across sessions |
| **source_summaries** | Per-source summary + key takeaways (bonus requirement) | Linked to source + session |
| **claims** / **claim_sources** | Claim-to-evidence links (quote span, support label) | Drives citations and contradiction detection |
| **memories** | Long-term: topics researched, user preferences, key conclusions | FTS5, optional vectors |

### 4.2 Token usage and cost fields returned by APIs
- **Anthropic `usage`** ([SDK type](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/usage.py), [caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)):
  - `input_tokens` counts only tokens after the last cache breakpoint.
  - Other fields: `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}`, `server_tool_use.{web_search_requests, web_fetch_requests}`, `service_tier` (`standard|priority|batch`), `output_tokens_details`, `inference_geo`.
  - **Total input = `cache_read_input_tokens + cache_creation_input_tokens + input_tokens`.**
  - `stop_reason` values: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn`, `refusal`, `model_context_window_exceeded` ([SDK](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/stop_reason.py)). Handle `pause_turn` for long server-tool turns by re-sending the paused assistant message ([web search docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool)).
- **OpenAI Responses `usage`:** `input_tokens`, `input_tokens_details.{cached_tokens, cache_write_tokens}`, `output_tokens`, `output_tokens_details.reasoning_tokens`, `total_tokens`. Chat Completions uses `prompt_tokens` / `completion_tokens` and `*_details` ([openai-python types](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_usage.py)).
- **Cost formula to implement** (Anthropic):
  `cost = in*P_in + cache_write_5m*1.25*P_in + cache_write_1h*2*P_in + cache_read*0.1*P_in + out*P_out + web_searches*$0.01`
  Sources: [caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [web search $10/1k](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool). Keep the price table in config, not code.
- **Prompt caching tip:** mark the static system prompt and tool definitions as cacheable. The minimum cacheable prefix is 512 tokens on Opus 5 and 1,024 on Sonnet 5; below it, the prefix silently isn't cached ([docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)).

### 4.3 Checkpointing for resume and override
- **LangGraph:** checkpoints are saved per super-step per `thread_id`. `interrupt()` pauses and `Command(resume=…)` continues. Operations before `interrupt()` must be idempotent because the node re-executes on resume ([interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [persistence](https://docs.langchain.com/oss/python/langgraph/persistence)).
- **DIY equivalent:** persist events before acting (write-ahead). On restart, rebuild orchestrator state by folding events and re-dispatch tasks whose status is `running` without a `task_completed` event. Make workers idempotent by caching tool results keyed on `(tool, args_hash)`. The "time travel" demo is to fork a session from event N with an edited plan.
- **Anthropic's production system** also relies on checkpoint-and-resume rather than restart ([source](https://www.anthropic.com/engineering/multi-agent-research-system)).

### 4.4 Long-term memory options
| Option | License / ★ | Deployment | Search | Fit |
|---|---|---|---|---|
| **SQLite + FTS5** ([docs](https://www.sqlite.org/fts5.html)) | Public domain (SQLite) | Embedded, zero-ops | BM25 via `bm25()`, `snippet()`/`highlight()`, porter/trigram tokenizers, external-content tables | **Default.** Keyword recall of past topics and sources |
| **sqlite-vec** ([repo](https://github.com/asg017/sqlite-vec)) | MIT/Apache-2.0, 8.1k★ | Embedded, pure C | `vec0` virtual table, KNN, metadata columns | Add semantic recall in the same DB file. **Pre-v1** ("expect breaking changes") |
| **Postgres + pgvector** ([repo](https://github.com/pgvector/pgvector)) | PostgreSQL License ([LICENSE](https://github.com/pgvector/pgvector/blob/master/LICENSE)), 23.1k★ | Server | HNSW / IVFFlat; L2, inner product, cosine, L1, Hamming, Jaccard; up to 16k dims | Production; overkill for a demo |
| **Chroma** ([repo](https://github.com/chroma-core/chroma)) | Apache-2.0, 29.4k★ | Embedded or client-server | Vectors + metadata `where` + `where_document` | Easy Python vector store |
| **LanceDB** ([repo](https://github.com/lancedb/lancedb)) | Apache-2.0, 11.5k★ | Embedded | Vector + full-text + SQL (hybrid) | Good embedded hybrid option |
| **Qdrant** ([repo](https://github.com/qdrant/qdrant)) | Apache-2.0, 34.7k★ | Server / Edge (in-process) / Cloud | Dense + sparse + RRF hybrid, rich payload filters | Heavier |
| **mem0** ([repo](https://github.com/mem0ai/mem0)) | Apache-2.0, 65.8k★ | Library/service | Extracts memories automatically (user/session/agent levels); semantic + BM25 + entity fusion; paper [arXiv:2504.19413](https://arxiv.org/abs/2504.19413) | Drop-in "remembers preferences". README claims LoCoMo 92.5 and LongMemEval 94.4 (vendor-reported) |
| **Letta (ex-MemGPT)** ([repo](https://github.com/letta-ai/letta)) | Apache-2.0, 24.8k★ | Server | Stateful agents with self-edited memory; MemGPT paper [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) | A whole agent runtime, too much for this assignment |
| **Zep / Graphiti** ([repo](https://github.com/getzep/graphiti)) | Apache-2.0, 31.1k★ | Needs Neo4j / FalkorDB / Neptune | Bi-temporal knowledge graph; paper [arXiv:2501.13956](https://arxiv.org/abs/2501.13956) | Good for "facts that changed over time"; heavy infrastructure |

**Recommendation:** SQLite + FTS5 for memory and source cache. Optionally add `sqlite-vec` with a small embedding model for "related topics". Build the memory write step yourself: at session end, an LLM extracts `{topic, key_findings[], open_questions[], user_prefs[]}`. That is more explainable in a demo than mem0.

### 4.5 Recommended SQLite schema (sketch)
```sql
CREATE TABLE sessions(id TEXT PRIMARY KEY, topic TEXT, status TEXT, created_at TEXT,
  settings_json TEXT, budget_usd REAL, parent_session_id TEXT, fork_from_event INTEGER);
CREATE TABLE events(id INTEGER PRIMARY KEY, session_id TEXT, ts TEXT, actor TEXT, -- planner|worker:3|user|system
  type TEXT, task_id TEXT, payload_json TEXT);                                  -- append-only
CREATE TABLE llm_calls(id INTEGER PRIMARY KEY, session_id TEXT, event_id INTEGER, role TEXT, model TEXT,
  input_tokens INT, output_tokens INT, cache_read INT, cache_write_5m INT, cache_write_1h INT,
  web_search_requests INT, web_fetch_requests INT, stop_reason TEXT, latency_ms INT, cost_usd REAL);
CREATE TABLE tasks(id TEXT PRIMARY KEY, session_id TEXT, parent_id TEXT, kind TEXT, -- search|read|compare|synthesize|verify
  goal TEXT, depends_on_json TEXT, status TEXT, result_json TEXT);
CREATE TABLE sources(id TEXT PRIMARY KEY, canonical_url TEXT UNIQUE, title TEXT, publisher TEXT,
  published_at TEXT, retrieved_at TEXT, content_hash TEXT, text TEXT,
  credibility REAL, recency REAL, relevance REAL, status TEXT, reject_reason TEXT);
CREATE TABLE source_summaries(source_id TEXT, session_id TEXT, summary TEXT, takeaways_json TEXT,
  PRIMARY KEY(source_id, session_id));
CREATE TABLE claims(id TEXT PRIMARY KEY, session_id TEXT, text TEXT, confidence REAL);
CREATE TABLE claim_sources(claim_id TEXT, source_id TEXT, quote TEXT, stance TEXT); -- supports|contradicts|mentions
CREATE TABLE memories(id INTEGER PRIMARY KEY, session_id TEXT, kind TEXT, content TEXT, created_at TEXT);
CREATE VIRTUAL TABLE memories_fts USING fts5(content, content='memories', content_rowid='id', tokenize='porter unicode61');
CREATE VIRTUAL TABLE sources_fts  USING fts5(title, text, content='sources', tokenize='porter unicode61');
```
Note: FTS5 external-content tables need triggers to stay in sync ([FTS5 docs](https://www.sqlite.org/fts5.html)).

### 4.6 Privacy and retention
- Store everything locally by default (a single `.db` file) and document it in the README. Provide `/forget <session>` and a `--no-memory` flag.
- Hard-delete per session, including FTS rows. Note this in the demo as a privacy feature.
- Redact API keys from logs and never log raw request headers.
- Anthropic web search and web fetch have Zero Data Retention (ZDR) notes and `allowed_callers` caveats ([web search](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool)). Brave offers ZDR on its Enterprise plan ([Brave](https://brave.com/search/api/)).
- The MCP spec requires explicit user consent before exposing user data or invoking tools ([MCP](https://modelcontextprotocol.io/specification/latest)). Mirror that principle in your UI.

---

## 5. Protocols and research methodology

### 5.1 Technical protocols
| Protocol | What / status | Relevance |
|---|---|---|
| **MCP** ([spec](https://modelcontextprotocol.io/specification/latest)) | Current spec version `2026-07-28`; JSON-RPC 2.0 between hosts, clients and servers. Server features: **resources, prompts, tools**. Client feature: **elicitation** (server asks the user for input). Standard transports: **stdio** and **Streamable HTTP** (POST to one endpoint; reply as JSON or a request-scoped SSE stream) ([transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)). Optional extensions: Tasks (async long-running), MCP Apps (inline UI) | Expose `research_search` / `source_fetch` as an MCP server, or consume existing search MCP servers (§7). Reference servers include Fetch and Memory ([servers repo](https://github.com/modelcontextprotocol/servers)); the registry is at `registry.modelcontextprotocol.io` |
| **A2A** ([repo](https://github.com/a2aproject/A2A)) | Linux Foundation, Apache-2.0, 25.9k★. Agent Cards for discovery, JSON-RPC 2.0 over HTTP(S), SSE streaming, push notifications, task lifecycle. "Complements MCP" | Overkill here. Mention it as a future path ("expose the researcher as an A2A agent") |
| **OpenTelemetry GenAI semconv** ([OTel](https://opentelemetry.io/docs/specs/semconv/gen-ai/), [repo](https://github.com/open-telemetry/semantic-conventions-genai)) | Still development status; moved to a dedicated repo. Attributes include `gen_ai.operation.name` (`invoke_agent`, `execute_tool`, `create_agent`), `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens` / `output_tokens`; opt-in via `OTEL_SEMCONV_STABILITY_OPT_IN` | Name your event and log fields after these attributes, so the action log is OTel-mappable. Pydantic AI and Microsoft Agent Framework emit OTel natively |
| **AG-UI** ([docs](https://docs.ag-ui.com/introduction)) | Created by CopilotKit; event-based agent-to-UI protocol (lifecycle, text deltas, tool calls, state snapshots and deltas) over SSE or WebSockets. Supported by LangGraph, CrewAI, ADK, Pydantic AI, MS Agent Framework and others. Repo MIT, 16.0k★ | Model your SSE event stream on AG-UI event names for free credibility |
| **Anthropic Citations API** ([docs](https://platform.claude.com/docs/en/build-with-claude/citations)) | Pass sources as `document` blocks (text/PDF/custom content) or `search_result` blocks with `citations.enabled`. The response returns `char_location` / `page_location` / `content_block_location` citations with `cited_text`. `cited_text` doesn't count toward output tokens and pointers are guaranteed valid. **Incompatible with structured outputs** (400 error) | Best way to get grounded synthesis: feed accepted sources as documents and let the API produce verifiable spans |

### 5.2 Methodological protocols (implement as code plus prompts)
1. **Query decomposition**
   - *Self-ask*: the model asks and answers follow-up sub-questions, and a search engine can plug in to answer them ([arXiv:2210.03350](https://arxiv.org/abs/2210.03350)).
   - *STORM perspectives*: discover perspectives first, then questions per perspective ([arXiv:2402.14207](https://arxiv.org/abs/2402.14207)).
   - *Anthropic*: go broad, then narrow ([source](https://www.anthropic.com/engineering/multi-agent-research-system)).
   - Output: a typed plan of `{id, question, kind, depends_on, success_criteria}`.
2. **Source credibility scoring**
   - *CRAAP test* (Currency, Relevance, Authority, Accuracy, Purpose), created by Sarah Blakeslee at CSU Chico's Meriam Library ([CSU Chico](https://library.csuchico.edu/sites/default/files/craap-test.pdf), [Chico State Today](https://today.csuchico.edu/how-to-craap-test/)).
   - *SIFT* (Stop; Investigate the source; Find better coverage; Trace claims to original context), from Mike Caulfield ([hapgood.us](https://hapgood.us/2019/06/19/sift-the-four-moves/)).
   - *Lateral reading*: professional fact-checkers leave a site to check it elsewhere and reach more warranted conclusions faster than PhD historians ([Wineburg & McGrew 2019, TCR](https://journals.sagepub.com/doi/10.1177/016146811912101102)).
   - Implementation: a rubric score 0–1 per CRAAP dimension, computed from domain type (`.gov`, `.edu`, peer-reviewed DOI via Crossref, known news outlets), author and date presence, and an LLM "purpose" label. The *lateral* step is one extra search, "who is <publisher>?", for unknown domains. Log the reason for every rejected source. Anthropic's human testers found agents over-selecting SEO farms, so this rubric matters ([source](https://www.anthropic.com/engineering/multi-agent-research-system)).
3. **Recency filtering**
   - The planner sets `recency_window` from the topic: news-like topics 90 days, technology 18 months, fundamentals unbounded. If ambiguous, ask the user.
   - Date signals: Anthropic's `page_age` per search result ([docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool)); trafilatura extracts page dates ([repo](https://github.com/adbar/trafilatura)); the arXiv API supports `sortBy=submittedDate|lastUpdatedDate` ([arXiv](https://info.arxiv.org/help/api/user-manual.html)); GDELT has a rolling 3-month window ([GDELT](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)).
   - Crossref exposes retractions via Retraction Watch data ([Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)). Use it to reject retracted papers.
4. **Citation grounding and claim-to-source verification**
   - *Chain-of-Verification*: draft, plan verification questions, answer them independently, then revise ([arXiv:2309.11495](https://arxiv.org/abs/2309.11495)).
   - *SAFE*: split a response into atomic facts and check each against search. It agreed with humans 72% of the time and was more than 20× cheaper ([arXiv:2403.18802](https://arxiv.org/abs/2403.18802)).
   - Implementation: after synthesis, extract claims and require at least one `claim_sources` row with a verbatim quote (Citations API spans). Drop or flag unsupported claims. Show "N/M claims verified" in the UI.
5. **Contradiction detection**
   - Cluster claims by subject, then compare pairs with an NLI-style prompt that labels each pair `agree | contradict | unrelated` and gives reasons.
   - Adjudicate with credibility and recency. If it can't be resolved, present both sides with sources.
   - Magentic-One's ledger of "verified facts vs educated guesses" is a good mental model ([MSR](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)).
6. **Stopping criteria** (combine these):
   - (a) every plan question has at least k accepted sources and a confidence of at least τ
   - (b) the marginal-novelty rate falls below a threshold over the last N sources
   - (c) the token or USD budget is exhausted: Jina's "until answer found or token budget exceeded" ([repo](https://github.com/jina-ai/node-DeepResearch)), dzhng's depth/breadth caps ([repo](https://github.com/dzhng/deep-research))
   - (d) a hard cap on tool calls, using `max_uses` on Anthropic server tools ([docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool))
   - (e) the user presses "wrap up"

---

## 6. UX for input and output

### 6.1 What deep-research products do
| Behavior | Product (primary source) | Copy it? |
|---|---|---|
| Clarifying questions before starting, via a lighter model | ChatGPT Deep Research ([OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/deep_research_api/introduction_to_deep_research_api.ipynb)) | **Yes**: 0–3 questions, only when ambiguity is detected, each with a default assumption so the user can skip |
| Editable plan: "revise or approve" | Gemini ([Google](https://blog.google/products/gemini/google-gemini-deep-research/)) | **Yes**: show the plan as a checklist with add/remove/reorder, then "Start". Auto-start after a timeout so it stays autonomous |
| Streaming "thoughts while browsing" | Gemini ([Google](https://blog.google/products/gemini/new-gemini-app-features-march-2025/)) | **Yes**: timeline of events |
| Report with inline citations plus a full source list | OpenAI (`url_citation` + `sources`) ([docs](https://developers.openai.com/api/docs/guides/tools-web-search)); Anthropic requires showing citations when displaying web search output ([docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool)) | **Yes** |
| Export | Gemini → Google Docs; GPT Researcher → PDF/DOCX/MD ([repo](https://github.com/assafelovic/gpt-researcher)) | Markdown required; PDF optional |
| Human joins the discourse and sees a mind map | Co-STORM ([repo](https://github.com/stanford-oval/storm)) | A small originality win: a live "knowledge map" of subtopics to sources |

### 6.2 UI stack options
| Stack | Pros | Cons | Primary refs |
|---|---|---|---|
| **CLI: Rich / Textual** (Python) | Fastest to build. Rich `Live` suits progress trees; Textual gives full TUI panes (plan / log / report). Both MIT (Rich 57.4k★, Textual 37.3k★) | Harder to show citations as clickable; less "wow" | [Rich](https://github.com/Textualize/rich), [Textual](https://github.com/Textualize/textual) |
| **Streamlit** | `st.status` container with running/complete/error states and `.update()`, ideal for per-task progress ([docs](https://docs.streamlit.io/develop/api-reference/status/st.status)). Apache-2.0, 45.8k★ | Rerun model complicates long-running async agents and mid-run interrupts | [repo](https://github.com/streamlit/streamlit) |
| **Gradio** | `ChatMessage.metadata` renders collapsible "thought / tool" accordions with `status: pending/done`, `duration`, and nesting via `parent_id` ([guide](https://www.gradio.app/guides/agents-and-tool-usage)). Apache-2.0, 43.6k★ | Less layout control | [repo](https://github.com/gradio-app/gradio) |
| **Chainlit** | Chat UI with step visualization | **Community-maintained since 2025-05-01**; original team stepped back ([README](https://github.com/Chainlit/chainlit/blob/main/backend/README.md)) | — |
| **FastAPI + SSE + small HTML/JS or React** | Full control. FastAPI has native `EventSourceResponse` / `ServerSentEvent` with `yield` ([docs](https://fastapi.tiangolo.com/tutorial/server-sent-events/)). A POST endpoint handles approve/edit/override. Clean separation keeps the orchestrator testable | More frontend work | — |
| **Next.js + Vercel AI SDK** | `useChat` streaming, tool-invocation states ([repo](https://github.com/vercel/ai)) | TypeScript backend or a bridge needed | — |

### 6.3 Recommended interaction model
1. **Input:** topic plus optional constraints (recency, depth, preferred source types, output format). Show "Related from your past research" from FTS5 memory.
2. **Clarify:** zero to three questions with defaults, plus a "Just go" button.
3. **Plan card:** tasks, estimated calls and cost, recency window. The user can edit, approve or auto-start (e.g. 10 s countdown).
4. **Live run:**
   - Left pane: task tree with status.
   - Center: event timeline (search queries, accepted/rejected sources with reasons, contradictions, clarifications).
   - Right: running cost and tokens.
   - Controls: **Pause**, **Redirect** (free-text steering injected at the next orchestrator step), **Skip task**, **Block domain**, **Add source URL**.
5. **Mid-run clarification:** when a worker hits ambiguity (e.g. two meanings of an acronym), the orchestrator emits `clarification_asked`. The UI shows it inline and the run continues on other tasks while waiting. This is the literal "ask user and adapt in real time" requirement.
6. **Output:**
   - Report with numbered inline citations; hovering shows the quote.
   - Per-source cards: summary, 3 takeaways, credibility and recency badges.
   - "Contradictions" section.
   - "Verified N/M claims" badge.
   - "Suggested next topics".
   - Export as Markdown (and PDF optionally).
7. **Event names:** follow AG-UI for the SSE stream (`RUN_STARTED`, `TEXT_MESSAGE_CONTENT`, `TOOL_CALL_START`, `STATE_DELTA`…). The event categories come from the [AG-UI docs](https://docs.ag-ui.com/introduction); the exact event names are **unverified** here.

---

## 7. Tools for research subagents

### 7.1 Web search APIs
| Tool | Pricing / free tier (primary) | Notes | MCP server |
|---|---|---|---|
| **Anthropic `web_search`** (server tool) | $10 per 1,000 searches plus tokens; errors not billed ([docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool)) | Versions `web_search_20250305`, `_20260209` (dynamic filtering via code execution), `_20260318` (`response_inclusion`). Options: `max_uses`, `allowed_domains` or `blocked_domains` (not both), `user_location`. Results carry `page_age`; citations always on (`cited_text` ≤150 chars, not billed) | n/a (native) |
| **Anthropic `web_fetch`** (server tool) | No extra charge beyond tokens ([docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-fetch-tool)) | Versions up to `web_fetch_20260318`. Options: `max_content_tokens`, optional `citations`, PDF support, `use_cache`. **No JS rendering.** URL must already appear in context. Rough size: 10 kB page ≈ 2.5k tokens | n/a |
| **OpenAI `web_search`** (Responses) | $10 per 1k calls plus content tokens ([pricing](https://developers.openai.com/api/docs/pricing)) | Up to 100 allowed/blocked domains; `url_citation` annotations plus `sources` ([docs](https://developers.openai.com/api/docs/guides/tools-web-search)) | n/a |
| **Gemini Grounding with Google Search** | Gemini 3.x: 5,000 free requests per month, then $14 per 1k ([pricing](https://ai.google.dev/gemini-api/docs/pricing)) | — | n/a |
| **Tavily** | 1,000 free credits per month, no card. Basic search 1 credit, advanced 2; extract 1 credit per 5 URLs; PAYG $0.008 per credit ([docs](https://docs.tavily.com/documentation/api-credits)) | LLM-oriented; default in open_deep_research and GPT Researcher | [tavily-ai/tavily-mcp](https://github.com/tavily-ai/tavily-mcp) (MIT, 2.4k★) |
| **Exa** | Search $7 per 1k requests; contents $1 per 1k pages; $20 signup credit plus $10 per month free ([pricing](https://exa.ai/pricing)) | Neural/semantic search; also deep-search endpoints | [exa-labs/exa-mcp-server](https://github.com/exa-labs/exa-mcp-server) (MIT, 5.0k★) |
| **Brave Search API** | $5 per 1k requests, with $5 per month free credit; 50 QPS ([Brave](https://brave.com/search/api/)) | Independent index of 30B+ pages; LLM Context endpoint; ZDR on Enterprise | [brave/brave-search-mcp-server](https://github.com/brave/brave-search-mcp-server) (MIT, 1.5k★) |
| **Serper** (Google SERP) | 2,500 free queries, no card ([serper.dev](https://serper.dev/)); paid tiers **unverified** | Fast Google results | community only (**unverified**) |
| **SerpApi** | 250 free searches per month; Starter $25/mo ([pricing](https://serpapi.com/pricing)) | Many engines (Scholar, News…) | — |
| **Bing Search API** | **Retired 2025-08-11**; replaced by "Grounding with Bing Search" in Azure AI Agents ([Microsoft](https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement)) | Don't use | — |
| **Google Custom Search JSON API** | 100 queries/day free, $5 per 1k; **closed to new customers**; existing customers must migrate by 2027-01-01 ([Google](https://developers.google.com/custom-search/v1/overview)) | Don't use | — |
| **DuckDuckGo** | No official web search API documented (**unverified**; community libraries scrape) | Fragile; OK as a zero-key fallback in a demo | — |
| **SearXNG** (self-host) | Free; AGPL-3.0, 37.5k★ ([repo](https://github.com/searxng/searxng)) | Metasearch across up to ~260 services, JSON output, no tracking ([docs](https://docs.searxng.org/)) | community |

### 7.2 Scraping and extraction
| Tool | License / pricing | Notes | MCP |
|---|---|---|---|
| **Jina Reader** `r.jina.ai/<url>`, `s.jina.ai/<q>` ([repo](https://github.com/jina-ai/reader)) | Apache-2.0; free tier (limits **unverified**); API key raises quotas | Page, PDF or Office file to Markdown; headers `x-target-selector`, `x-max-tokens`, generated image alt text | [jina-ai/MCP](https://github.com/jina-ai/MCP) (Apache-2.0) |
| **Firecrawl** ([pricing](https://www.firecrawl.dev/pricing)) | 1,000 free credits/month; Hobby $16/5k; scrape 1 credit per page, search 2 credits per 10 results | Scrape/crawl/map/search to Markdown/JSON; handles JS | [firecrawl-mcp-server](https://github.com/firecrawl/firecrawl-mcp-server) (MIT, 7.5k★) |
| **trafilatura** ([repo](https://github.com/adbar/trafilatura)) | Apache-2.0 (≥v1.8.0), 6.8k★, free/local | Main-text plus metadata extraction (**date**, author, site name) | — |
| **Crawl4AI** ([repo](https://github.com/unclecode/crawl4ai)) | Apache-2.0, 84.0k★, v0.9.3 | Playwright-based; BM25 content filter; LLM/CSS/schema extraction | — |
| **Playwright** | Apache-2.0 (**unverified** here) | Needed only for JS-heavy pages; Anthropic also offers a client-side browser-use tool ([web fetch docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-fetch-tool)) | — |
| **MCP reference "Fetch" server** ([repo](https://github.com/modelcontextprotocol/servers)) | Apache-2.0 / MIT | "Web content fetching and conversion for efficient LLM usage" | itself |

### 7.3 Academic, news and reference
| Source | Access (primary) | Notes | MCP |
|---|---|---|---|
| **arXiv API** ([manual](https://info.arxiv.org/help/api/user-manual.html)) | Free; Atom XML; keep ~3 s between calls; max 2,000 per slice | `sortBy=submittedDate` for recency | [blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server) (Apache-2.0, 3.2k★) |
| **Semantic Scholar** ([API](https://www.semanticscholar.org/product/api)) | Free. Unauthenticated: 1,000 RPS shared across all users. With key: 1 RPS introductory | Papers, citations, recommendations | covered by [openags/paper-search-mcp](https://github.com/openags/paper-search-mcp) (MIT, 2.7k★) |
| **OpenAlex** | Free, CC0 data (**unverified**: docs moved to help.openalex.org and I couldn't reach the rate-limit page) | Open scholarly graph | — |
| **Crossref REST** ([docs](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)) | Free, no sign-up; `mailto=` for the polite pool | DOIs, licenses, **retractions (Retraction Watch)** | — |
| **PubMed E-utilities** | Rate limits (3/s without key, 10/s with) **unverified** (NCBI page returned a CAPTCHA) | Biomedical | — |
| **GDELT DOC 2.0** ([blog](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)) | Free; rolling 3-month window; 65 machine-translated languages; JSON/CSV/RSS | Global news; tone filters | — |
| **NewsAPI** ([pricing](https://newsapi.org/pricing)) | Developer plan: 100 req/day, 24 h delay, **dev-only** (not production); Business $449/mo | Avoid for a live demo | — |
| **Wikipedia / MediaWiki API** | Free; rate-limit page moved (**unverified**) | Background and disambiguation for clarifying questions | — |

**Suggested default toolset for workers** (namespaced, 4–5 tools, per Anthropic's tool guidance):
- `web_search(query, recency_days?, domains?)` — Anthropic server tool or Tavily
- `fetch_source(url) -> {source_id, title, published_at, excerpt}` — stores the full text in the DB and returns a trimmed view
- `academic_search(query, since_year?)` — arXiv + Semantic Scholar
- `memory_search(query)` — FTS5 over past sessions
- `record_finding(claim, source_id, quote, stance)`
- The orchestrator additionally has `spawn_workers(tasks[])`, `ask_user(question, default)` and `finish(reason)`.

---

## 8. Recommendation: two candidate architectures

### Architecture A: Python, raw Anthropic API orchestrator-workers, SQLite event log, FastAPI + SSE (or Textual CLI)

```
User ──► UI (FastAPI+SSE web page  |  Textual TUI)
            │  POST /sessions, /approve, /override, /answer      ▲ SSE events (AG-UI-style)
            ▼                                                   │
     Orchestrator (plain Python state machine, asyncio) ────────┘
      phases: CLARIFY → PLAN → [await approval | auto] → EXECUTE ⇄ REFLECT/REPLAN → SYNTHESIZE → VERIFY → MEMORIZE
            │ spawn N (effort-scaled)                 │ ask_user() when ambiguous (non-blocking for other tasks)
            ▼                                          ▼
     Worker agents (tool loop, Haiku/Sonnet)      EventStore (SQLite: events, llm_calls, tasks,
      tools: web_search, fetch_source,             sources, claims, memories + FTS5)
      academic_search, record_finding              ▲ every step appended before/after acting
            │                                          │
     SourceFilter (recency, CRAAP-ish score, dedupe, injection tagging) ──┘
     Synthesizer (Citations API over accepted sources) → Verifier (CoVe-style claim check) → Report.md
```
- **Modules:**
  - `llm/` (LLMClient protocol, AnthropicClient, FakeLLM)
  - `agents/` (orchestrator.py, worker.py, prompts/*.md)
  - `tools/` (search.py, fetch.py, academic.py)
  - `research/` (planner schema, filter.py, contradictions.py, verify.py)
  - `store/` (events.py, repo.py, schema.sql)
  - `ui/` (api.py, static/)
  - `tests/`
- **Pros:**
  - Maximum "clear, testable orchestration": the state machine is unit-testable with FakeLLM and recorded tool fixtures, with no LLM needed.
  - Follows Anthropic's "start with the API" advice ([source](https://www.anthropic.com/engineering/building-effective-agents)).
  - The event log gives the action log, cost meter, resume and fork for free, and it is highly original to demo.
  - Server-side `web_search` / `web_fetch` need zero extra keys (one Anthropic key for everything). The Citations API gives verifiable quotes.
- **Cons:**
  - You implement pause/resume and checkpointing yourself (about 150 LoC with event sourcing).
  - Claude-only unless you add adapters.
  - Server-tool nuances: `pause_turn`, and `encrypted_content` must round-trip ([docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool)).
  - Mixing a client-side `ask_user` with server tools in the same turn triggers `stop_reason: "tool_use"` ordering rules ([docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool)).
  - Mitigation: use client-side Tavily for workers to keep everything in your own loop, and reserve server tools for a "native mode" flag.

### Architecture B: LangGraph with interrupts, SqliteSaver checkpointer and Tavily; Streamlit or FastAPI UI
- **Graph nodes:** `clarify` (interrupt) → `plan` → `approve_plan` (interrupt) → `supervisor` → fan-out `researcher` subgraphs (Send API) → `compress` → `reflect` (conditional edge: replan / ask user (interrupt) / continue) → `write_report` → `verify`.
- **State:** a typed dict of plan, findings, sources and budget. `SqliteSaver` provides threads, history, time-travel and resume; `Store` provides cross-session memory ([persistence](https://docs.langchain.com/oss/python/langgraph/persistence)). LangSmith handles tracing.
- **Pros:**
  - HITL (`interrupt()` / `Command(resume=…)`), checkpointing and time-travel are first-class ([interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)).
  - Multi-provider.
  - Close to LangChain's reference open_deep_research (repo archived, but readable), so there is proven structure to borrow.
- **Cons:**
  - Framework concepts (reducers, Send, super-steps, node re-execution on resume) need explaining in the live demo.
  - Less "your own orchestration" for originality.
  - The re-execution-on-resume semantics can surprise you (non-idempotent tool calls before `interrupt()`).
  - LangSmith is a hosted SaaS for the nice traces.

### Scoring against the rubric
| Criterion | A (raw loop) | B (LangGraph) |
|---|---|---|
| Agentic behavior: autonomy, adaptability, proactivity, transparency | High (you design replan, `ask_user` and the event timeline explicitly) | High (interrupts make HITL easy) |
| Code quality / modularity / testability | **Highest** (pure-Python state machine, FakeLLM tests) | Good (nodes testable; graph wiring framework-bound) |
| Depth of LLM integration | High: effort-scaling prompts, Citations API, prompt caching, model tiering | High: same, but through the LangChain wrappers |
| UX clarity | Equal (depends on UI) | Equal |
| Originality | **Higher** (event-sourced log with fork/replay, credibility rubric, contradiction map) | Lower (common stack) |
| Build risk in a take-home timeframe | Medium | Medium-low |

**Pick A.** Build it so the orchestrator is a pure class (`Orchestrator(llm, tools, store, clock)`), and put all side effects behind interfaces. This lets you run a deterministic demo (recorded fixtures) if the network fails. That is a real live-demo safety net.

**Demo script (5–7 min):**
1. Enter an ambiguous topic, e.g. "Is SMR nuclear viable?"; the agent asks 1–2 clarifying questions (region? timeframe?).
2. The plan card appears; edit one task and approve.
3. Watch parallel workers stream; one source is rejected ("published 2019 > 18-month window"), one is flagged as low credibility.
4. A mid-run clarification pops up; answer it and the plan adapts (visible `plan_revised` event).
5. Hit Redirect: "focus on cost data".
6. The report shows citations, a contradictions section and "Verified 23/25 claims".
7. Show the SQLite event log, the cost breakdown including cache reads, and "suggested next topics".
8. Start a new session on a related topic; memory surfaces the prior session.

**Build order (if time is tight):** (1) CLI plus core loop and event log → (2) source filter and per-source summaries → (3) clarify and plan approval → (4) verification and contradictions → (5) web UI with SSE → (6) memory and related topics → (7) MCP server wrapper (optional).

---

## Source list

**Anthropic**
- Building effective agents: https://www.anthropic.com/engineering/building-effective-agents
- How we built our multi-agent research system: https://www.anthropic.com/engineering/multi-agent-research-system
- Effective context engineering for AI agents: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Writing effective tools for agents: https://www.anthropic.com/engineering/writing-tools-for-agents
- Demystifying evals for AI agents: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Prompt injection defenses: https://www.anthropic.com/research/prompt-injection-defenses
- Claude Research launch: https://claude.com/blog/research
- Models overview: https://platform.claude.com/docs/en/about-claude/models/overview
- Prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Citations: https://platform.claude.com/docs/en/build-with-claude/citations
- Web search tool: https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-search-tool
- Web fetch tool: https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/web-fetch-tool
- Messages API: https://platform.claude.com/docs/en/api/messages
- SDK usage type: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/usage.py
- SDK stop reasons: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/stop_reason.py
- Agent SDK overview: https://code.claude.com/docs/en/agent-sdk/overview
- Claude Agent SDK (Python): https://github.com/anthropics/claude-agent-sdk-python

**OpenAI**
- A practical guide to building agents (PDF): https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- Guide landing page: https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
- Deep Research API cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/deep_research_api/introduction_to_deep_research_api.ipynb
- Web search tool: https://developers.openai.com/api/docs/guides/tools-web-search
- Pricing: https://developers.openai.com/api/docs/pricing
- Responses usage type: https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_usage.py
- Agents SDK: https://github.com/openai/openai-agents-python

**Google**
- Gemini Deep Research launch: https://blog.google/products/gemini/google-gemini-deep-research/
- Gemini app update, March 2025: https://blog.google/products/gemini/new-gemini-app-features-march-2025/
- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- Custom Search JSON API: https://developers.google.com/custom-search/v1/overview
- ADK (Python): https://github.com/google/adk-python

**Microsoft**
- Magentic-One: https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/
- AutoGen: https://github.com/microsoft/autogen
- Agent Framework: https://github.com/microsoft/agent-framework
- Bing Search API retirement: https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement

**Perplexity**
- sonar-deep-research: https://docs.perplexity.ai/getting-started/models/models/sonar-deep-research

**Open-source research agents**
- GPT Researcher: https://github.com/assafelovic/gpt-researcher
- STORM / Co-STORM: https://github.com/stanford-oval/storm
- LangChain open_deep_research: https://github.com/langchain-ai/open_deep_research
- Open deep research evolution (L. Martin): https://rlancemartin.github.io/2025/07/30/bitter_lesson/
- HF open Deep Research: https://huggingface.co/blog/open-deep-research
- dzhng/deep-research: https://github.com/dzhng/deep-research
- Jina node-DeepResearch: https://github.com/jina-ai/node-DeepResearch

**Frameworks**
- LangGraph: https://github.com/langchain-ai/langgraph
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- CrewAI: https://github.com/crewAIInc/crewAI
- AG2: https://github.com/ag2ai/ag2
- smolagents: https://github.com/huggingface/smolagents
- Pydantic AI: https://github.com/pydantic/pydantic-ai
- LlamaIndex: https://github.com/run-llama/llama_index
- Mastra: https://github.com/mastra-ai/mastra
- Vercel AI SDK: https://github.com/vercel/ai
- Ollama tool calling: https://docs.ollama.com/capabilities/tool-calling

**Papers and benchmarks**
- BrowseComp: https://arxiv.org/abs/2504.12516 (results table: https://arxiv.org/html/2504.12516)
- GAIA: https://arxiv.org/abs/2311.12983
- Humanity's Last Exam: https://arxiv.org/abs/2501.14249
- DeepResearch Bench: https://arxiv.org/abs/2506.11763
- STORM: https://arxiv.org/abs/2402.14207
- Co-STORM: https://arxiv.org/abs/2408.15232
- Self-ask: https://arxiv.org/abs/2210.03350
- Chain-of-Verification: https://arxiv.org/abs/2309.11495
- SAFE: https://arxiv.org/abs/2403.18802
- mem0: https://arxiv.org/abs/2504.19413
- MemGPT: https://arxiv.org/abs/2310.08560
- Zep: https://arxiv.org/abs/2501.13956

**Protocols**
- MCP spec: https://modelcontextprotocol.io/specification/latest
- MCP transports: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports
- MCP reference servers: https://github.com/modelcontextprotocol/servers
- A2A: https://github.com/a2aproject/A2A
- OTel GenAI semconv: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OTel GenAI semconv repo: https://github.com/open-telemetry/semantic-conventions-genai
- AG-UI: https://docs.ag-ui.com/introduction
- OWASP LLM01: https://genai.owasp.org/llmrisk/llm01-prompt-injection/

**Source evaluation methods**
- CRAAP test (PDF): https://library.csuchico.edu/sites/default/files/craap-test.pdf
- CRAAP background: https://today.csuchico.edu/how-to-craap-test/
- SIFT: https://hapgood.us/2019/06/19/sift-the-four-moves/
- Lateral reading: https://journals.sagepub.com/doi/10.1177/016146811912101102

**Storage**
- SQLite FTS5: https://www.sqlite.org/fts5.html
- sqlite-vec: https://github.com/asg017/sqlite-vec
- pgvector: https://github.com/pgvector/pgvector
- Chroma: https://github.com/chroma-core/chroma
- LanceDB: https://github.com/lancedb/lancedb
- Qdrant: https://github.com/qdrant/qdrant
- mem0: https://github.com/mem0ai/mem0
- Letta: https://github.com/letta-ai/letta
- Graphiti: https://github.com/getzep/graphiti

**Search, scraping and data sources**
- Tavily credits: https://docs.tavily.com/documentation/api-credits
- Exa pricing: https://exa.ai/pricing
- Brave Search API: https://brave.com/search/api/
- Serper: https://serper.dev/
- SerpApi pricing: https://serpapi.com/pricing
- SearXNG docs: https://docs.searxng.org/
- SearXNG repo: https://github.com/searxng/searxng
- Firecrawl pricing: https://www.firecrawl.dev/pricing
- Jina Reader: https://github.com/jina-ai/reader
- trafilatura: https://github.com/adbar/trafilatura
- Crawl4AI: https://github.com/unclecode/crawl4ai
- arXiv API: https://info.arxiv.org/help/api/user-manual.html
- Semantic Scholar API: https://www.semanticscholar.org/product/api
- Crossref REST API: https://www.crossref.org/documentation/retrieve-metadata/rest-api/
- GDELT DOC 2.0: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- NewsAPI pricing: https://newsapi.org/pricing

**MCP servers for search tools**
- Tavily: https://github.com/tavily-ai/tavily-mcp
- Exa: https://github.com/exa-labs/exa-mcp-server
- Firecrawl: https://github.com/firecrawl/firecrawl-mcp-server
- Jina: https://github.com/jina-ai/MCP
- Brave: https://github.com/brave/brave-search-mcp-server
- arXiv: https://github.com/blazickjp/arxiv-mcp-server
- Paper search: https://github.com/openags/paper-search-mcp

**UI**
- Rich: https://github.com/Textualize/rich
- Textual: https://github.com/Textualize/textual
- Streamlit `st.status`: https://docs.streamlit.io/develop/api-reference/status/st.status
- Streamlit repo: https://github.com/streamlit/streamlit
- Gradio agents guide: https://www.gradio.app/guides/agents-and-tool-usage
- Gradio repo: https://github.com/gradio-app/gradio
- Chainlit README: https://github.com/Chainlit/chainlit/blob/main/backend/README.md
- FastAPI SSE: https://fastapi.tiangolo.com/tutorial/server-sent-events/
