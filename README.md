# rootlogic assignment - AI Agentic Personal Research Assistant

An agentic **personal research assistant** for journalists, analysts and students.
Give it a topic; it asks clarifying questions if needed, drafts a research plan you can
edit, dispatches parallel research sub-agents that search the web, filters outdated or
irrelevant sources, reflects on gaps (researching more or asking you), cross-checks
sources for contradictions, and writes a cited report. Every step is logged, and you can
pause, redirect or abort it at any time.

Illustrative session (numbers are examples, not a benchmark):

```
$ rootlogic research "impact of generative AI on local newsrooms"
04:32:53 session.started    Session c3b11a10: “impact of generative AI on local newsrooms”
04:32:53 memory.recalled    Found 1 related past session(s): generative AI newsroom ethics
04:32:54 clarify.skipped    Topic is clear enough
04:32:58 plan.created       Plan: 5 sub-tasks, sources ≤ 365 days old
  ┌ plan table — [a]pprove / [e]dit / [q]uit ┐
04:33:01 task.started       [t1] Researching: How are local newsrooms adopting generative AI?
04:33:09 subagent.search    [t1] Searched “local newsroom generative AI adoption 2026” — 5 result(s)
04:33:22 subagent.fetch     [t1] Read https://www.rjionline.org/…
   ...                      (Ctrl-C → pause → skip t3 / add "Who funds this?" / note "focus on EU" / stop)
04:34:40 source.dropped     [t2] Dropped https://… — outdated (2019-01-01 < 2025-09-22)
04:35:12 reflect.done       Gaps found. No data on job losses → added [t6]
04:36:30 analyze.done       4 consensus point(s), 2 contradiction(s)
04:36:58 session.done       Report saved to .rootlogic/reports/c3b11a10-….md · 14 LLM calls · 212,480 tokens · $1.84
```

## Tracing (optional)

Every model request is already recorded in the `llm_calls` table and every decision in
`events`, so tracing adds a viewer, not data. To send a run to a [Langfuse](https://langfuse.com)
instance:

```bash
pip install -e '.[langfuse]'
export LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-... LANGFUSE_HOST=http://localhost:3000
rootlogic research "your topic"            # traces appear under the session id
```

One trace per session, one generation per model request, one event per action-log entry — the
same shape the action log has, so a reader of either sees the same run. It is off unless both
keys are set, and every call is wrapped: a tracing backend that is down, misconfigured or of a
different major version degrades to no tracing rather than failing the research.

Verified against a self-hosted Langfuse v4: traces, generations and events arrive and group
by session. Payload fields (model, token counts, cost) did **not** persist through that
server's ingestion in testing, whether passed at span start or via `update()`, so treat
`rootlogic usage` as the source of truth for cost and tokens and Langfuse as the timeline.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'            # add '.[web]' instead of '.[dev]' for just the web UI

rootlogic research --offline -y "any topic"      # no API key needed: deterministic fake LLM
export ANTHROPIC_API_KEY=sk-ant-...              # or `ant auth login`
rootlogic research "your topic"                  # live: Claude + web search

rootlogic web                                    # web UI at http://127.0.0.1:8000
rootlogic research --engine graph "topic"       # LangGraph engine (resumable)
rootlogic resume <session>   # continue a graph session after a crash / quit
rootlogic graph              # print the LangGraph engine as a Mermaid diagram
rootlogic research --follow-up <session> "dig deeper into X"   # continue earlier research
rootlogic profile [add|rm|clear]   # standing preferences the assistant learned about you
rootlogic sources [block|allow|trust|distrust|rm] <domain>   # your rules for websites
rootlogic eval --offline     # evaluation harness (live: rootlogic eval -y; costs money)
rootlogic history            # past sessions + suggested next topics (long-term memory)
rootlogic log <session>      # full action log + conversation
rootlogic usage <session>    # tokens, web searches and cost per step
rootlogic show <session>     # re-print the report
rootlogic forget <session>   # delete a session and its memory
pytest                       # 209 tests, no network
```

Running it through a gateway onto a Claude subscription — the fastest setup in practice, and
no Anthropic API key:

```bash
.venv/bin/rootlogic web --port 8848 \
  --provider openai --base-url http://localhost:20128/v1 --model claude-sonnet-5 \
  --reasoning-effort none --stream --search tavily --moderation none
```

Why each flag: the gateway speaks OpenAI Chat Completions, so `--provider openai` with its
`--base-url`, even though the model behind it is Claude. `--stream` because a router times out
waiting for a slow first byte. `--reasoning-effort none` because the extra thinking costs wall
clock without helping these prompts. `--search tavily` because Claude's hosted web tools are
only reachable on the Anthropic API, not through a gateway. `--moderation none` skips the
screening pass — drop it to keep the guardrail.

Useful flags: `-y` auto-approve plan · `-v` show dropped sources · `--rounds N` reflection
rounds · `--max-tasks N` · `--parallel N` sub-agents · `--searches N` per sub-agent
(default 8; under 5 the web tools call search directly, because dynamic filtering batches
searches and can exhaust a small budget before any result returns) ·
`--zdr` Zero Data Retention mode (also on `resume` and `web`): sets `allowed_callers: ["direct"]`
on the web tools, which makes them ZDR-eligible but turns off dynamic filtering, so searches may
use more context tokens.
`--search tavily` (also on `resume` and `web`): sub-agents search through our own `SearchProvider`
tools backed by [Tavily](https://docs.tavily.com) instead of Claude's built-in web tools. Needs
`TAVILY_API_KEY`; Tavily credits are billed by Tavily and are not included in `rootlogic usage`.
`--no-profile` skips reading and learning your standing preferences for one run.
`--block DOMAIN` / `--only DOMAIN` (repeatable) apply source rules to one run.
`--verify-claims N` sets how many claims are checked against their pages (default 12), and
`--no-verify` turns checking off.
`--worker-model MODEL` runs the research sub-agents on a cheaper model while planning,
reflection, analysis, verification and writing stay on `--model`. Sub-agents make most of the
calls and burn most of the tokens (330k of 636k in one live run), so this is the biggest cost
lever: `--model claude-opus-5 --worker-model claude-haiku-4-5`. `rootlogic usage` shows the
per-model split, and `--worker-prices IN,OUT` prices a non-Claude worker.
Data lives in `./.rootlogic/` (override with `ROOTLOGIC_HOME`).

## How it maps to the assignment

| Requirement | Where |
|---|---|
| Autonomous planning: break topic into sub-tasks, sequence without prompting | `Orchestrator._plan`, `Plan.ready()` (dependency-aware waves), `_research_loop` |
| Sub-tasks incl. finding articles, summarizing, identifying contradictions | research sub-agents (`llm.research`) → per-source summaries; `_analyze` → contradictions |
| Proactive search for recent info, filter outdated/irrelevant | `web_search`/`web_fetch` server tools; `filters.filter_sources` (recency, relevance, dedupe, blocklist) with logged reasons |
| Ask clarifying questions, adapt strategy in real time | `_clarify` (before planning) and `_reflect` (mid-run: new sub-tasks or questions to user) |
| Summaries + key takeaways per source (bonus) | `SourceDraft.summary/key_takeaways`, report source list |
| Long-term memory + related topic suggestions (bonus) | `Store.remember/recall` (SQLite FTS5), `Store.suggestions`, prior sessions fed to planner; a learned **user profile** (`continuity.learn_profile`) and **follow-up threads** that reuse earlier findings (`--follow-up`) |
| Transparent action log, monitor/override (bonus) | `events` table + live stream, down to each sub-agent search and fetch; plan approve/edit; Ctrl-C override (skip/add/note/stop/abort) |
| Filter outdated/irrelevant, reliable research | plus claim verification against cited pages, corroboration labels, user source rules, report checks, and an evaluation set (`verify.py`, `evaluate.py`) |
| Clear, testable orchestration | plain-Python state machine behind an `LLM` protocol; `FakeLLM` + `ScriptedUI` tests |

## Architecture

```mermaid
flowchart LR
    U([User]) -->|topic| C[Clarify]
    M[(Memory<br/>FTS5)] --> C
    C -->|questions| U
    C --> P[Plan]
    P -->|approve / edit| U
    P --> W{{Wave scheduler}}
    W -->|parallel| R1[Research<br/>sub-agent]
    W --> R2[Research<br/>sub-agent]
    W --> R3[Research<br/>sub-agent]
    R1 & R2 & R3 --> F[Source filters<br/>recency · relevance · dedupe]
    F --> X[Reflect / critic]
    X -->|new sub-tasks| W
    X -->|question| U
    X -->|sufficient or budget hit| A[Analyze<br/>consensus · contradictions]
    A --> RP[Write report]
    RP --> M
    U -. Ctrl-C override .-> W
```

**Pattern: orchestrator–workers + evaluator loop.** The LLM decides *what* to research
(plan, follow-ups, questions, summaries, contradictions). Plain Python decides *how* the run
proceeds (ordering, parallelism, budgets, filtering, human checkpoints). That split is what
makes the agent both autonomous and testable, and keeps cost bounded.

| Module | Responsibility |
|---|---|
| `orchestrator.py` | Stage machine: clarify → plan → waves ⇄ reflect → analyze → write. Budgets, checkpoints, overrides. |
| `llm.py` | `LLM` protocol + `AnthropicLLM`: structured outputs (JSON schema) and the research tool loop (`web_search`, `web_fetch`, strict `submit_findings`). Usage → cost. |
| `models.py` | Pydantic schemas. LLM-facing ones are strict (all required, no extras). |
| `filters.py` | Deterministic source rules; every drop gets a reason in the log. |
| `store.py` | SQLite: sessions, conversation, action log, tasks, sources, per-call token usage, FTS5 memory. |
| `control.py` | `Event`, `Command`, `Interaction` protocol, thread-safe hold/pause/abort flags. |
| `prompts.py` | Static role prompts (clarifier, planner, researcher, critic, analyst, writer). |
| `cli.py` | Rich terminal UI implementing `Interaction`. |
| `web.py` + `static/index.html` | FastAPI + SSE web UI implementing `Interaction`; runs engines in background threads. |
| `graph.py` | The same agent as a LangGraph state machine (checkpoints, `interrupt()`, resume). |
| `context.py` | Pure prompt builders and source curation shared by both engines. |
| `continuity.py` | Cross-session continuity: learns your standing preferences, and rebuilds an earlier session so a follow-up can continue it. |
| `openai_llm.py` | `OpenAICompatibleLLM`: the same `LLM` interface over Chat Completions (OpenAI, Ollama, vLLM, OpenRouter …). |
| `backend.py` | Which model and search a run uses; validates combinations up front. |
| `tools.py` | Our client-side `web_search`/`web_fetch` tools and their budgets, shared by both adapters. |
| `moderation.py` | Screens request, user input and report for models without built-in safety (OpenAI moderation, Llama Guard 3). |
| `verify.py` | Guardrails: claim verification, corroboration labels and report checks. |
| `evaluate.py` + `evals/cases.json` | Evaluation harness and cases (`rootlogic eval`). |
| `search.py` | `SearchProvider` protocol (`search`, `fetch`) + `TavilySearch`, `StaticSearch`. The model-agnostic path for web access. |
| `fake_llm.py` | Deterministic LLM for tests and `--offline` demos. |

### Key design decisions

- **Raw Claude Messages API, no agent framework.** The orchestration is the thing being graded,
  so it's explicit code rather than framework configuration. `LLM` is a 2-method protocol, so
  swapping providers or adding LangGraph later touches one file.
- **Structured outputs everywhere the orchestrator branches** (clarify/plan/reflect/analyze/report),
  so control flow never parses free text.
- **Cheap workers, strong lead**: research sub-agents do the bulk of the calls on the
  reading-heavy work, so `--worker-model` puts them on a cheaper model while judgement-heavy
  steps (plan, reflect, analyze, verify, write) stay on the better one.
- **Sub-agents get isolated context**: each research worker sees only its question, the
  objective, user guidance and dependency results — not the whole run. Results come back as
  a compact `FindingDraft`, keeping the orchestrator's context small.
- **Model proposes, code disposes** for sources: the worker rates credibility/relevance and
  dates; `filters.py` applies explicit rules. Low-credibility sources are kept but flagged,
  so dissent isn't silently hidden.
- **Bounded autonomy**: `Budget` caps reflection rounds, total sub-tasks, parallelism and
  searches per worker. Budget hits are logged, not silent.
- **Human checkpoints at the cheap moments**: before planning (clarify), before spending
  (plan approval), between waves (override). Workers never block on the user.
- **Salvage a reply before spending another call on it.** Weaker models fail structured output
  in three ways, each seen in live runs: they finish the object and keep talking, they stop
  partway through it, or they hand back the schema they were shown. The adapter takes the first
  complete object out of a chatty reply, closes off a truncated one at the last value that
  arrived, and — only when a retry is unavoidable — names the fields it wants rather than
  showing the schema again, which is what caused the echo. Each of these cost a finished run
  before it was handled.
- **Web content is untrusted**: researcher prompt forbids following instructions found in pages.
- **Refusals**: server-side `fallbacks: "default"` retries safety-declined requests on a fallback
  model; a remaining refusal fails only that sub-task.
- **Swappable web access**: sub-agents use either Claude's hosted `web_search`/`web_fetch` or our
  own client tools backed by a `SearchProvider`. The client-tool path needs nothing
  provider-specific, so any tool-calling model can run it through another `LLM` adapter.
- **Model**: `claude-opus-5` for all roles; effort `low` for clarify, `medium` for research
  workers, `high` for plan/reflect/analyze/report.

### Storage

SQLite single file (`.rootlogic/rootlogic.db`), all rows keyed by `session_id`:

| Table | Purpose |
|---|---|
| `sessions` | topic, status, plan JSON, summary, related topics, report path |
| `messages` | human ↔ agent conversation (topic, clarifying Q&A, notes, report summary) |
| `events` | append-only action log (every decision, drop, override) |
| `tasks` | sub-task status + finding JSON |
| `sources` | every source seen, kept or dropped with reason |
| `llm_calls` | per-request tokens (input/output/cache read/cache write), web searches, cost, request id |
| `memory_fts` | FTS5 index of past session topics/summaries/takeaways for recall |

New to agents? Start with [docs/walkthrough.md](docs/walkthrough.md), a step-by-step account
of how this project was built. See [docs/research/agentic-research-assistant.md](docs/research/agentic-research-assistant.md)
for the research behind these choices (frameworks, storage options, protocols, UX, tools) and an
alternative architecture (LangGraph + web UI).

## Other models (OpenAI-compatible)

rootlogic runs on Claude by default. `--provider openai` swaps in any server that speaks the
OpenAI Chat Completions format: OpenAI itself, local models via Ollama / LM Studio / vLLM, or
OpenRouter. The rest of rootlogic doesn't change, because both adapters implement the same
two-method `LLM` interface.

```bash
pip install -e '.[openai]'
export TAVILY_API_KEY=tvly-...        # other models use our own search tools
rootlogic research --provider openai --model gpt-5-mini --search tavily --prices 0.25,2 "topic"
rootlogic research --provider openai --base-url http://localhost:20128/v1 --model qwen3:8b \
                   --search tavily --reasoning-effort none "topic"   # local Ollama: no LLM bill
```

- `--search tavily` is required: Claude's built-in web tools only work with Claude.
- `--no-strict` is for servers without strict JSON-schema support. The adapter then asks for JSON
  in the prompt, validates it, and lets the model correct itself once.
- Small local models drift out of tool calling once the conversation fills with web text. A
  sub-agent that searched but won't call `submit_findings` is asked for its findings one last
  time with no tools offered and a strict JSON schema, which is grammar-constrained and far more
  reliable for them. A sub-agent that never searched still fails fast: there is nothing to
  report. Measured on an M4 with `qwen3:8b`: without this, zero sources after ten prose turns;
  with it, four sources and a usable answer.
- `--prices IN,OUT` (USD per million tokens) enables cost tracking. Without it, non-Claude calls
  record $0.
- `--stream` is needed behind a gateway. Routers time out waiting for a slow model's first byte
  (OmniRoute allows 30s), and a local model can spend a minute on a long planning prompt before
  it emits anything; a streamed reply starts at once. The adapter reassembles the completion from
  deltas, so nothing else in the pipeline changes.
- `--reasoning-effort none|low|medium|high` for thinking models on servers that support it.
  Locally this dominates the wall clock: `qwen3:8b` spent 72s on a clarification that takes
  1.3s with `none`.
- Llama Guard size matters: the 1B model flagged a report on AI in newsrooms as "S1 violent
  crimes" and blocked a finished run. Use `--moderation-model llama-guard3` (8B) unless RAM is
  tight. A blocked *report* is recoverable — the research behind it is stored, and the log says
  how to re-check it with `--follow-up`.
- Claude-only features don't apply: `--zdr`, effort levels, and server-side refusal fallback.
  Refusals and content filtering from the other provider are still detected and reported.
- **Moderation is required, because an arbitrary model may have no safety system.** With
  `OPENAI_API_KEY` set, OpenAI's free moderation endpoint is used automatically. Fully local?
  `ollama pull llama-guard3` and `--moderation llama-guard`. To run unscreened, say so:
  `--moderation none`. `--moderation-strict` blocks on every flag, not only harm-enabling ones.
  If the moderation service can't be reached, the run stops rather than continuing unscreened.
  `--offline` screens too, with a small deterministic blocklist, so the checkpoints really run
  in demos and in `rootlogic eval --offline` rather than a missing gate looking like a working
  one.
- **A gateway onto a Claude subscription** is the same path: the router speaks Chat
  Completions, so it is `--provider openai` with a `--base-url`, and the model named is the
  Claude one it routes to. This was worth supporting because it gets Claude's judgement
  without an Anthropic API key — but the hosted `web_search`/`web_fetch` tools live on the
  Anthropic API and do not survive the hop, so `--search tavily` is not optional, and token
  accounting is only as good as what the router reports back.
- Invalid combinations are rejected before anything runs (`rootlogic/backend.py`).
- Smaller local models are noticeably weaker at planning, strict schemas and faithful citation.
  Consider a larger model when quality matters.

## Guardrails against misinformation

rootlogic can't *guarantee* a report is true; no research tool can. It checks what it can,
labels what it can't, and measures the result. The code lives in `rootlogic/verify.py`.

| Guardrail | How it works | Model or code? |
|---|---|---|
| **Claim verification** | Each claim is checked against the text of the pages it cites, captured when sub-agents fetch them or fetched for the check. The verifier must quote the page verbatim; code confirms the quote is really in the text and downgrades "supported" if not. Missing, too-short or unusable page text (navigation, paywall, wrong page) means **unverifiable**, never "unsupported": failing to read a page says nothing about the claim. Claims that fail are withheld from the writer as fact. | model judges, code checks |
| **Corroboration labels** | *corroborated* (2+ independent sites), *single source*, or *weak* (only low-credibility sources). Low-credibility sources can support a claim but never alone. | code |
| **Source rules** | `rootlogic sources block/allow/trust/distrust <domain>`, `--block`/`--only` per run, or the web sidebar. Allow = allowlist mode. Trust/distrust override the model's credibility rating. Agents are told the rules, and code enforces them. | code |
| **Report checks** | `[n]` citations pointing at no source become `[?]`. Uncited factual-looking sentences and takeaways citing only low-credibility sources are listed. Every report ends with **Confidence and limitations** and a **Claim check** table. | code |
| **Refusals** | Claude's safety checks (with server-side fallback) and the other provider's `refusal`/`content_filter` stop harmful requests. | model |
| **Moderation** (non-Claude) | Screens the request, your mid-run input and the finished report. Harmful requests and reports stop the run (`blocked`); sensitive-but-legitimate flags are noted in the report instead. | model (OpenAI moderation or Llama Guard), code decides |

**Proof: the evaluation set.** [`evals/cases.json`](evals/cases.json) holds 20 cases: known
facts, hoaxes the agent must not repeat, contested questions, time-sensitive topics, and harmful
requests it must refuse. `rootlogic eval` runs them and scores each run:
- **Facts:** required keywords are present.
- **Hoaxes:** a model judge confirms none is presented as true.
- **Contested topics:** the analysis finds the disagreements.
- **Time-sensitive topics:** at most 30% of dated sources are older than two years.
- **Harmful requests:** the model refuses.
- **Every non-harmful case:** at least 70% of checked claims hold up.

Results go to `evals/results/*.json`. `--baseline <file>` shows what changed since an earlier
run. `--offline` exercises the harness for free; offline scores are meaningless, because the
fake model knows no facts and never refuses. Two caveats: the hoax judge is the same model unless
you configure otherwise, and a live run costs roughly one research session per case.

## Memory that improves research

rootlogic uses what you tell it, not just what it searches:

- **Within a session**, your answers to clarifying questions and any notes you add mid-run go into
  every later prompt: planner, every sub-agent, critic, analyst and writer.
- **Your profile.** After a session where you answered questions or left notes, one small
  low-effort call pulls out *lasting* preferences (audience, region, preferred or avoided sources,
  time window, format). It drops any the session contradicted. Every later session starts with them,
  and the clarifier doesn't re-ask what they already cover. View or edit them with `rootlogic profile`
  or in the web sidebar. Opt out per run with `--no-profile`.
- **Follow-ups.** `--follow-up <session>` (or "Continue this research" in the web UI) starts a
  linked session seeded with the earlier findings, sources and your earlier answers. Earlier
  findings appear as `previous` tasks that are already done. The planner is told to research only
  what's new, and already-seen sources are deduplicated. Follow-ups chain, and earlier tasks don't
  count against the task budget.
- **Past sessions.** Keyword search over earlier topics and summaries feeds the planner, and
  suggests next topics.

## Web UI

`rootlogic web` serves a single-page UI (no build step) at http://127.0.0.1:8000:

- **Start** a run (topic, engine, offline toggle), with suggested next topics from memory.
- **Answer** clarifying questions, then **edit the plan** (drop/add sub-tasks, set max source age) before anything is spent.
- **Steer it mid-run** with three controls that make three different promises:

  | Button | Use it when | Lands at | Afterwards |
  |---|---|---|---|
  | **Pause** → **Resume** | you want to read the log before deciding | next sub-task boundary | held, spending nothing, until you resume |
  | **Override** | the research is working but heading the wrong way | end of the current wave | continues with your edits applied |
  | **Abort** | you typed the wrong topic, or it's on the wrong model | next sub-task boundary | ends, marked `aborted` — Retry still reuses finished work |

  Override opens a card to skip pending sub-tasks, add a question, give guidance ("focus on the
  EU", which is fed into the remaining sub-agents' prompts), or stop researching and write now.
  Skip and Steer on the sub-agent cards use the same path. Override waits for the wave because
  it hands you a plan to edit and the plan is not stable until the wave rewriting it finishes;
  Abort has no plan to offer, so it lands within one in-flight model call. None of the three can
  interrupt a request already sent to the model — one sub-task is the floor for all of them.
- **Watch** the live action log and plan status, then read the rendered report and per-step token/cost table.
- **History:** reopen any past session. Sessions left `interrupted` by a server restart can be **resumed** (graph engine).

How it works: each run executes in a background thread. `WebInteraction` implements the same
`Interaction` protocol as the terminal UI. When the agent needs a human it publishes a `request`
event and blocks until the browser POSTs an answer. The controls are their own endpoints —
`POST /api/runs/{id}/pause`, `/resume`, `/checkpoint` (override) and `/abort` — so stopping a run
never depends on a card having been answered first. The browser follows one SSE stream
(`GET /api/runs/{id}/events`) whose events carry sequential ids, so a refresh replays from
`Last-Event-ID` and rebuilds the page. Report Markdown is sanitized (DOMPurify) before rendering.
It binds to 127.0.0.1 because there is no authentication and runs spend API credits.
API docs: `/api/docs`.

## Two engines

The same agent is implemented twice: a hand-rolled orchestrator loop (default) and a
LangGraph state machine (`--engine graph`) with checkpointing, `interrupt()`-based human
approval and `rootlogic resume`. Both share prompts, schemas, filters, storage and UI.
See [docs/langgraph-vs-loop.md](docs/langgraph-vs-loop.md) for a side-by-side comparison.

## Roadmap

- MCP client so users can plug in extra sources (Semantic Scholar, internal docs)
- Embedding-based memory (sqlite-vec) alongside FTS5
- Prompt caching for shared worker prefix; eval set of topics with graded reports
