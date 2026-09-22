# How rootlogic was built: a walkthrough

This document explains, step by step, how this project was designed and built. It's written
for someone who has never built an AI agent. If you rebuild it yourself, follow the same order.

---

## Step 0: What an "AI agent" actually is

A chatbot takes one input and gives one answer. An **agent** is an LLM running in a **loop**:
it decides what to do next, uses **tools** (for example web search), looks at the results,
and decides again until the job is done.

There are two ways to build one:

- **Workflow:** your code decides the steps, and the LLM fills them in. This is predictable and easy to test.
- **Agent:** the LLM decides the steps. This is flexible but harder to control.

rootlogic is a **hybrid**. The LLM decides *what* to research: the plan, follow-up
questions, summaries and contradictions. Plain Python decides *how* the run proceeds:
order, parallel work, budgets, and when to ask the user. This split is the most important
idea to explain in a demo. It's what makes the agent autonomous and testable at the same time.

## Step 1: Turn the assignment into a checklist

Every line of the assignment PDF was mapped to a feature before any code was written:

| Assignment says | Feature built |
|---|---|
| Break topic into sub-tasks, sequence them | Planner → list of sub-tasks with dependencies |
| Search online, filter outdated/irrelevant | Research workers with web search + source filter rules; later, claims checked against their cited pages, corroboration labels and user source rules (Step 18) |
| Ask clarifying questions, adapt | Clarify step before planning; reflect step during the run |
| Summaries per source (bonus) | Each source gets a summary and key takeaways |
| Long-term memory (bonus) | Past sessions feed new plans and suggest topics; a learned user profile; follow-up threads that build on earlier findings |
| Action log + override (bonus) | Every step is logged; plan approve/edit; Ctrl-C pause menu |
| Testable orchestration code | LLM hidden behind a small interface, so tests swap in a fake |

The same mapping appears in the README, which is where graders look for it.

## Step 2: Choose an architecture pattern

The pattern is **orchestrator–workers with an evaluator loop**, as described in Anthropic's
"Building effective agents" guide:

```
You → Clarify → Plan → [you approve] → Workers research in parallel
                                          ↓
                          Filter sources → Reflect: enough?
                             ↑  no: add tasks / ask you  ↓ yes
                             └──────────────   Verify claims → Analyze contradictions
                                                  → Write report → Check report → Save to memory
```

The two checking steps (verify claims, check report) were added last, in Step 18; the rest is
the original design.

- The **orchestrator** is the manager. It holds the plan and decides what runs next.
- The **workers** (subagents) each research one question with their own clean context. They
  don't see the whole run, which keeps them focused and cheaper.
- The **evaluator** (the reflect step) reviews the results and decides whether more research is needed.

## Step 3: Pick the tech stack

| Choice | Why |
|---|---|
| **Python** | Best-supported language for LLM work; readable for graders |
| **Claude API** (`claude-opus-5`) | Strong reasoning, plus built-in `web_search` and `web_fetch` tools, so no separate search service is needed |
| **No agent framework** for the main engine | The orchestration logic is what's being graded, so it's written as plain code rather than hidden in a framework |
| **SQLite** | Single-file storage: no server, and it ships with Python |
| **Rich** | Nicely formatted terminal output: tables, colors, prompts |
| **Pydantic** | Defines data shapes and validates what the LLM returns |

Added later, each for a specific reason:

| Addition | Why | Step |
|---|---|---|
| **LangGraph** | A second engine with checkpointing and resume | 13 |
| **FastAPI** | A web UI that streams progress to the browser | 14 |
| **Tavily** (optional) | Web search that doesn't depend on one LLM vendor | 15 |
| **OpenAI SDK** (optional) | Run on GPT, local models (Ollama) or OpenRouter | 17 |

Nothing new was needed for Step 18 (guardrails and evaluation): it's plain Python on top of the
same interfaces.

## Step 4: Define the data shapes first → [`models.py`](../rootlogic/models.py)

Before any logic, the project defines what each step produces:

- `Clarification`: does the topic need questions, and which ones?
- `PlanDraft` / `SubTask`: the research plan.
- `FindingDraft`: a worker's answer, its sources, claims and gaps.
- `SourceDraft`: URL, date, summary, takeaways, credibility and relevance ratings.
- `Reflection`, `Analysis`, `ReportDraft`: the later stages.

**Why this comes first:** these shapes are sent to Claude as **structured output** schemas,
which force it to reply in exact JSON. The code never has to guess what the AI meant from
loose text. It just checks `reflection.sufficient == True` and branches. This is the single
biggest reliability technique in agent building.

## Step 5: Wrap the LLM behind a small interface → [`llm.py`](../rootlogic/llm.py)

The rest of the program only knows about two methods:

- `structured(...)`: ask Claude something and get back one of the shapes above.
- `research(...)`: a mini-loop in which Claude calls `web_search` and `web_fetch` as many
  times as it needs. It finishes by calling a tool defined here, `submit_findings`, whose
  input is a `FindingDraft`.

This file also:

- Records **token counts and cost** for every API call.
- Handles errors: network failures, rate limits, and refusals, which are retried
  server-side on a fallback model.
- Handles `pause_turn`, a signal that a long search turn paused partway and needs to be resumed.

**Why wrap it:** this boundary is what makes a `FakeLLM` for tests possible. It also means
switching to another provider later only touches one file. Step 15 finishes that job for web
search, which was the one part still tied to Claude, and Step 17 adds the second adapter.

## Step 6: Write the orchestrator → [`orchestrator.py`](../rootlogic/orchestrator.py)

This is the heart of the project. `run()` reads top to bottom like a recipe:

1. **Recall:** search memory for related past sessions.
2. **Clarify:** ask Claude whether the topic is vague, and if so ask the user up to 3 questions.
3. **Plan:** Claude produces 3–6 sub-tasks and chooses how recent sources must be (e.g. 365
   days for tech news, no limit for history).
4. **User reviews the plan:** approve, edit or quit.
5. **Research loop:**
   - Run every sub-task that's ready, in parallel threads. A task is ready once the tasks
     it depends on have finished.
   - Filter each worker's sources.
   - When nothing is left to run, **reflect**: is this enough? If not, add follow-up tasks
     or ask the user something.
   - Stop when sufficient or when the budget runs out.
6. **Verify** (added in Step 18): check claims against the text of their cited pages and
   label how well each is corroborated.
7. **Analyze:** find what sources agree on and where they contradict each other.
8. **Write:** produce a Markdown report with numbered citations, run code checks on it (bad
   citations, uncited statements), and save it to memory.

A `Budget` caps reflection rounds, total tasks, parallel workers, searches per worker and, since
Step 18, how many claims get verified.
Without caps, an agent can loop forever and run up a large bill.

The text each role sees (topic, findings, sources, progress) is built by pure functions in
[`context.py`](../rootlogic/context.py), so both engines share one copy.

## Step 7: Apply source rules in code, not in the AI → [`filters.py`](../rootlogic/filters.py)

The worker *rates* each source, and code *decides* whether to keep it. A source is dropped if
it is from a blocked domain, a duplicate, low relevance, or older than the recency limit.
Every drop is logged with a reason, for example "outdated (2019-01-01 < 2025-09-22)".

**Why:** rules written in code are predictable, testable and easy to explain. Low-credibility
sources are *kept but flagged*, so a dissenting view isn't silently hidden.

Step 18 extends this file with the user's own **source rules** (block, allow-only, trust,
distrust), applied the same way: code decides, and every drop has a reason.

## Step 8: Human in the loop → [`control.py`](../rootlogic/control.py)

The orchestrator never talks to the terminal directly. It emits **events** (log lines) and
calls an `Interaction` object for four things: `ask`, `review_plan`, `override` and `on_event`.

- The terminal UI implements `Interaction`, and so do the tests.
- The web UI (Step 14) is one more implementation, with no change to the core.
- Ctrl-C only sets a "pause" flag. The orchestrator checks it between waves, so it never
  stops halfway through a step.

## Step 9: Storage → [`store.py`](../rootlogic/store.py)

One SQLite file holds nine tables. All except `preferences` and `source_rules` are keyed by session:

| Table | What it holds |
|---|---|
| `sessions` | topic, status, plan, summary, related topics, report path, and `parent_id` for follow-ups |
| `messages` | the conversation with the user (questions, answers, notes, context inherited by a follow-up) |
| `events` | the full action log |
| `tasks` | each sub-task's status and finding |
| `sources` | every source seen, kept or dropped (with the reason) |
| `llm_calls` | tokens in/out, cache usage, web searches and dollar cost per call |
| `memory_fts` | full-text search index over past sessions, which powers recall and topic suggestions |
| `preferences` | the user's standing preferences (the learned profile, Step 16) |
| `source_rules` | the user's block / allow / trust / distrust rules for websites (Step 18) |

The LangGraph engine also saves its checkpoints in a second file, `checkpoints.db`. New columns
are added to older databases automatically on startup, so an upgrade never loses history.

## Step 10: Prompts → [`prompts.py`](../rootlogic/prompts.py)

Each role has its own short instructions: clarifier, planner, researcher, critic, analyst,
writer, a profiler that extracts lasting user preferences (Step 16), a verifier and an
evaluation judge (Step 18).
Two details are worth knowing:

- **Prompt-injection defense:** the researcher is told that web pages are *untrusted data*
  and that it must never follow instructions written inside them.
- **Autonomy by default:** the critic is told to ask the user only when no search could
  resolve the gap. Otherwise it keeps working on its own.

## Step 11: Test without spending money → [`fake_llm.py`](../rootlogic/fake_llm.py) + [`tests/`](../tests/)

`FakeLLM` returns canned but realistic answers, and `ScriptedUI` plays the user. The tests
check that:

- Stages run in order.
- Old sources are dropped.
- Vague topics trigger questions.
- Reflection adds tasks, and budgets stop the loop.
- Dependencies run in order.
- Overrides (skip/add/note/stop/abort) work.
- A failing worker doesn't crash the session.
- Memory recalls past topics.
- The LangGraph engine resumes after a crash, and saves the same log details as the loop engine.
- The web API round-trips questions, plan edits and overrides, and replays SSE.
- The search tools enforce their limits, and Tavily requests and responses map correctly.
- The OpenAI-compatible adapter sends the right wire format and handles refusals, truncation,
  invalid JSON and the repair retry, tested against the `openai` SDK's own response types.
- The profile is learned, used, edited and can be switched off.
- Follow-ups reuse earlier findings, chain, and don't count earlier tasks against the budget.
- Guardrails behave as designed:
  - Invented quotes are downgraded.
  - Claims without page text are "unverifiable".
  - Corroboration labels are correct.
  - Source rules block, allow and override credibility.
  - Bad citations are fixed and uncited statements flagged.
- The evaluation harness scores facts, hoaxes, contradictions, stale sources and refusals correctly.

Most scenarios run on **both** engines. When a bug is fixed, a test that failed before the fix
is added first. All 157 tests run in a few seconds with no internet.

## Step 12: Terminal UI → [`cli.py`](../rootlogic/cli.py)

It shows:

- A colored live log.
- The plan table, with approve/edit.
- The override menu when you press Ctrl-C.

It also adds the commands `history`, `log`, `usage`, `show`, `forget`, `resume`, `graph`, `web`
and `profile`. Useful flags on `research`:

| Flag | What it does |
|---|---|
| `--offline` | Runs everything on the fake LLM, so the demo is safe from network problems |
| `--engine graph` | Uses the LangGraph engine (Step 13) |
| `--follow-up <id>` | Continues an earlier session (Step 16) |
| `--search tavily` | Uses our own search tools instead of Claude's (Step 15) |
| `--zdr` | Zero Data Retention mode for Claude's web tools (Step 15) |
| `--no-profile` | Doesn't use or update the learned profile for this run (Step 16) |
| `--provider openai --model … [--base-url …]` | Runs on another model (Step 17) |
| `--block` / `--only DOMAIN` | Source rules for one run (Step 18) |
| `--verify-claims N` / `--no-verify` | How many claims to verify against their pages (Step 18) |
| `--moderation {auto,none,openai,llama-guard}` | Content screening for non-Claude models (Step 19) |

Two more commands came with Step 18: `rootlogic sources` (manage website rules) and
`rootlogic eval` (run the evaluation set).

## Step 13: A second engine with LangGraph → [`graph.py`](../rootlogic/graph.py)

After the plain-code engine worked, the same agent was ported to **LangGraph**, a framework
that models agents as a graph of steps. Run it with `--engine graph`. It adds:

- **Checkpointing:** progress is saved after every step, and `rootlogic resume <id>`
  continues after a crash or quit without redoing finished work.
- **`interrupt()`:** clarifying questions, plan approval and overrides become saved pauses.
  A session can wait overnight.

The trade-offs are more code, state that has to be kept as plain data, and the rule that
nothing with side effects may run before an `interrupt()` in the same step (because the
step re-runs on resume). The full comparison is in
[langgraph-vs-loop.md](langgraph-vs-loop.md).

Every later feature was built for **both** engines. In the graph, that meant new state fields
and, for Step 18, a new `verify` node between research and analysis. `rootlogic graph` prints
the current diagram.

## Step 14: A web UI → [`web.py`](../rootlogic/web.py) + [`static/index.html`](../rootlogic/static/index.html)

`rootlogic web` starts a FastAPI server. The design reuses Step 8's idea: the web UI is just
another implementation of `Interaction`.

- Each research run executes in a **background thread**, so the server stays responsive.
- `WebInteraction.on_event` appends each event to the run's list.
- `ask`, `review_plan` and `override` publish a `request` event, then **block the engine
  thread** until the browser POSTs `/api/runs/{id}/answer`.
- The browser follows **Server-Sent Events (SSE)**, a one-way stream from server to browser
  over plain HTTP. It's simpler than WebSockets and enough here, because the browser's own
  input goes over normal POSTs.
- Events are numbered, so after a refresh the browser reconnects with `Last-Event-ID`,
  replays the stream, and rebuilds the page.
- The page is one HTML file with plain JavaScript (no build step). Report Markdown is
  sanitized before display, because it originates from web content.
- Later additions:
  - A **Your profile** panel in the sidebar and a **Continue this research** box on finished
    sessions (Step 16).
  - A **Source rules** panel and a **Verify claims** toggle (Step 18).
  - Reports now show the **Confidence and limitations** section and the **Claim check** table.

Testing it in a real browser caught two bugs the unit tests missed:
- A new task could reuse an existing id after the user dropped one. `Plan.next_id()` now
  uses the highest id, not the count.
- Sessions orphaned by a server restart stayed "running" forever. They're now marked
  `interrupted` on startup, and graph-engine ones can be resumed.

## Step 15: Swappable web search → [`search.py`](../rootlogic/search.py)

The `LLM` interface (Step 5) made the model swappable in principle, but web search still used
Claude's built-in tools, which only work with Claude. So search got its own interface:

- **`SearchProvider`** has two methods: `search(query, max_results, recency_days)` and
  `fetch(url)`. `TavilySearch` implements it with the Tavily API, and `StaticSearch` returns
  canned results for tests.
- With `--search tavily`, subagents get **our own** `web_search` and `web_fetch` tools, which
  call the provider. This path uses only plain tool calling, which almost every modern model
  supports. That makes it the path another model's adapter would use.
- Because our code now runs the tools, our code enforces the limits: the search cap and 3 page
  fetches per subagent. All results go back in one message, and long pages are cut off with a
  visible `[truncated]` note, never silently.

Two related changes, both found by a research pass over the code:

- **`--zdr`** makes Claude's built-in web tools eligible for Anthropic's Zero Data Retention
  (ZDR). It sets `allowed_callers: ["direct"]`, which also turns off "dynamic filtering"
  (Claude trimming search results before they fill its context). It's opt-in for that reason.
- The LangGraph engine wasn't saving log details (which task an entry belongs to). Now it
  matches the loop engine.

## Step 16: Memory that makes research better → [`continuity.py`](../rootlogic/continuity.py)

Two features let rootlogic use the user's own input across sessions:

- **A learned profile.** At the end of a session in which the user answered questions or
  left notes, one small structured call (`ProfileUpdate`) pulls out *lasting* preferences into a
  `preferences` table: audience, region, preferred or avoided sources. It also removes any the
  user contradicted. The next session puts them into every prompt, and the clarifier doesn't
  re-ask what they already cover. The user can view and delete entries, which matters for trust.
- **Follow-up threads.** `load_previous` rebuilds a finished session's findings, sources and
  user answers. A new session is linked to it by `parent_id` and starts with those findings as
  already-done `previous` tasks. The planner sees which questions are already answered and plans
  only new ones, and already-seen URLs count as duplicates. Earlier tasks don't count against
  the task budget, and log lines say "3 new sub-tasks (+3 from earlier research)".

Where the user's own words go:

| Input | Used in |
|---|---|
| Answers and notes, this session | Every later prompt in the session |
| The learned profile | Every prompt of every later session |
| An earlier session you follow up on | Its findings, sources and your answers, carried into the new session |
| Past sessions in general | Their topics and summaries, shown to the clarifier and planner |

Design choices worth explaining:
- Learning is **best-effort**: if the profile call fails, the finished research still succeeds.
- The call is **skipped entirely** when the user said nothing, so it costs nothing on
  auto-approved runs.
- Remembered text is treated as **data, not instructions**, like web content.

## Step 17: A second model adapter → [`openai_llm.py`](../rootlogic/openai_llm.py)

"OpenAI-compatible" means the Chat Completions request format (`POST /v1/chat/completions`),
which many services accept: OpenAI, local servers like Ollama, and routers like OpenRouter.
`OpenAICompatibleLLM` implements the same two methods as `AnthropicLLM`, so nothing else in
rootlogic changes when you pass `--provider openai`.

- **`structured()`** sends our Pydantic schema as a strict `json_schema` response format. For
  servers without that (`--no-strict`), it asks for JSON in the prompt, validates it, and shows
  the model its mistake once before giving up.
- **`research()`** runs the same client-side tool loop as `--search tavily`. The tool code and
  its limits live in [`tools.py`](../rootlogic/tools.py), so both adapters behave identically.
  Invalid findings are sent back for correction, and every tool call gets a reply, even failed ones.
- **Refusals** (`refusal` field or `content_filter`) and **truncation** (`length`) raise the same
  errors as with Claude, so the engines handle them the same way.
- **Cost:** token usage is recorded for every call. Prices come from `--prices`, because an
  unknown model is recorded at $0 rather than guessed. Building this also exposed an old bug:
  unknown models used to be priced as Claude Opus.
- **[`backend.py`](../rootlogic/backend.py)** gathers the model and search settings in one place
  and rejects bad combinations up front, e.g. `--provider openai` without `--search tavily`.

The adapter was written against the installed `openai` SDK's actual types, not from memory. Its
tests replay real SDK response objects, so no network is needed.

## Step 18: Guardrails and proof → [`verify.py`](../rootlogic/verify.py) + [`evaluate.py`](../rootlogic/evaluate.py)

The honest starting point: nothing earlier *ensured* reports were accurate, and nothing
*measured* it. The tests proved the orchestration works, not that the answers are right. This
step adds layers that check what can be checked, label what can't, and measure the result.

A new **verify** stage runs between research and analysis in both engines:

1. **Claim verification.**
   - Pages the subagents fetched are kept as *evidence* (`SearchHit.text`). With our own search
     tools, missing pages can also be fetched for the check.
   - For each claim, the model is shown the most relevant parts of its cited pages. It says
     supported, partially supported, or unsupported, and must quote the page **word for word**.
   - **Code** then checks the quote is really in the page. If it isn't, "supported" becomes
     "partially supported". This stops the verifier from inventing evidence.
   - No page text means **unverifiable**, never assumed true. Failed claims are withheld from the
     writer as fact.
2. **Corroboration labels** (pure code): *corroborated* by 2+ independent sites, *single source*,
   or *weak* (only low-credibility sources).
3. **Source rules:** block, allowlist, trust or distrust sites. They're stored in the database,
   applied by code, and also told to the agents so they search accordingly.
4. **Report checks** (pure code): citations pointing at no source become `[?]`, and uncited
   factual sentences and weakly sourced takeaways are listed. Every report ends with
   **Confidence and limitations** and a **Claim check** table.

**Proving it: the evaluation set.** Guardrails are only as good as their measured effect.
[`evals/cases.json`](../evals/cases.json) has 20 cases, each targeting one failure:

| Case kind | Pass condition |
|---|---|
| Known facts | Required keywords are present |
| Hoaxes | A judge model confirms none is presented as true |
| Contested questions | The analysis finds the disagreements |
| Time-sensitive topics | Sources are mostly recent |
| Harmful requests | The model refuses |

`rootlogic eval` scores every run the same way and saves a JSON file. `--baseline` compares two
runs, so "the guardrail helped" becomes a number.

Two design decisions worth explaining:
- **Checks that fail never turn into passes.** If the verifier call errors, its claims stay
  "unchecked"; they're never marked supported by default.
- **Turning verification off is visible.** With `--no-verify`, corroboration labels (free,
  pure code) are still computed and every claim is marked "unchecked", so the report says
  verification didn't happen instead of staying silent. A test caught the silent version.

Honest limits to explain in a demo:
- Verification proves a claim matches its source, not that the source is right.
- Unverifiable claims are common when subagents don't fetch pages.
- The hoax judge is a model too.
- A live evaluation costs money.

## Step 19: Screening content for models without safety systems → [`moderation.py`](../rootlogic/moderation.py)

Claude runs Anthropic's safety classifiers on every request. An arbitrary OpenAI-compatible or
local model may have none, so `--provider openai` needed its own screening. Three checkpoints,
used by both engines through one shared `ModerationGate`:

1. **The request**, before any research runs. A blocked topic ends the session as `blocked`, and
   nothing is ever sent to the model.
2. **Your input during a run**: clarifying answers, notes, tasks you add by hand. Flagged input
   is ignored (it never reaches a prompt) but doesn't end the run.
3. **The report**, before it is saved or shown. A blocked report is not written to disk.

Two backends: OpenAI's moderation endpoint (free with an API key) and **Llama Guard 3** through
any OpenAI-compatible server, which keeps a local setup fully offline.

The judgement call worth explaining in a demo: a research tool for journalists must be able to
*discuss* violence, crime, health and elections. So each backend's categories split in two:

| | Examples | What happens |
|---|---|---|
| **Block** | weapons instructions, sexual content involving minors, self-harm instructions | the run stops |
| **Warn** | violence, specialised (medical/legal) advice, defamation | noted in the report's "Confidence and limitations" |

`--moderation-strict` blocks on every flag instead. Two more decisions:
- **Fail closed:** if the moderation service is unreachable, the run stops rather than
  continuing unscreened.
- **No silent default:** a non-Claude model with no moderation available refuses to start and
  explains the three options, rather than quietly running unscreened.

The evaluation harness counts a moderation block as a refusal, so the harmful cases pass either
way: the model refuses, or moderation stops it.

## Step 20: Check it, then publish

Every change went through the same routine:

1. Wrote or updated tests, then ran them.
2. Ran the offline demo, and for UI work, used it in a real browser. The first demo caught
   Rich hiding `[t1]` as a formatting tag. The browser caught the duplicate task ids and the
   orphaned "running" sessions.
3. Checked external facts (API shapes, pricing, protocol versions) against official docs. For
   the OpenAI adapter, that meant reading the installed SDK's own types, not relying on memory.
4. For the guardrails, ran the evaluation harness offline to confirm it **fails** cases it
   should fail (the fake model knows no facts), then checked in the browser that a distrust
   rule visibly changes a report's corroboration labels.
5. Committed with a message explaining *why*, and pushed to GitHub.

---

## What to do next

1. **Get an API key** at console.anthropic.com. Run `export ANTHROPIC_API_KEY=...`, then
   `rootlogic research "a topic you know well"`. The live Claude and Tavily paths haven't been
   run yet, so the first real run may surface small fixes. The same goes for the
   OpenAI-compatible adapter. Trying it free with Ollama is a good first test.
2. **Read the code in this order:** `models.py` → `orchestrator.py` (start with `run()`) →
   `llm.py` → `filters.py` → `context.py` → `continuity.py` → `search.py` → `tools.py` →
   `openai_llm.py` → `verify.py` → `moderation.py` → `evaluate.py` → the tests → `graph.py` →
   `web.py`.
3. **Rehearse the demo:**
   - A vague topic (shows clarifying questions).
   - Editing the plan.
   - Ctrl-C with an `add` or `note` override.
   - `rootlogic usage <id>` for cost.
   - A second related topic (shows memory).
   - Answer a clarifying question, then run another topic: the profile is used and not re-asked.
   - `--follow-up <id>` (or "Continue this research"): only the new questions get researched.
   - Show a report's **Confidence and limitations** and **Claim check** sections, then
     `rootlogic sources distrust <site>` and run again: the labels change.
   - `rootlogic eval --kind hoax --kind harmful` (live) to show the guardrails measured.
   - `--engine graph`: kill it mid-run, then `rootlogic resume <id>`.
   - `rootlogic web`: the same flow in the browser, including refresh mid-run.
4. **Be ready to explain:**
   - The LLM-decides-what / code-decides-how split.
   - Why structured outputs matter.
   - Why subagents get isolated context.
   - Why budgets exist.
   - Why the source rules live in code.
   - What LangGraph added and what it cost.
   - Why search sits behind its own interface, and what that means for swapping models.
   - How the profile and follow-ups make research improve over time, and how the user
     controls them.
   - Which guardrails are model judgments and which are code, and why the verbatim-quote
     check matters.
   - Why non-Claude models need their own moderation step, and why sensitive topics are
     warned about rather than blocked.
   - What the evaluation proves and what it can't.

For wider context, read the two research notes:

- [research/agentic-research-assistant.md](research/agentic-research-assistant.md): prior art,
  framework trade-offs, storage options, research protocols, UX patterns and search tools.
- [research/models-tools-protocols.md](research/models-tools-protocols.md): which LLM APIs we
  use, how to swap models, what subagent tools do, what's stored, and how agents communicate
  (MCP, A2A).
