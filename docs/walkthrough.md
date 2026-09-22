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
| Search online, filter outdated/irrelevant | Research workers with web search + source filter rules |
| Ask clarifying questions, adapt | Clarify step before planning; reflect step during the run |
| Summaries per source (bonus) | Each source gets a summary and key takeaways |
| Long-term memory (bonus) | Past sessions are searchable, feed new plans, and suggest topics |
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
                             └──────────────   Analyze contradictions → Write report → Save to memory
```

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

LangGraph was added later as a second engine. See Step 13.

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
switching to another provider later only touches one file.

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
6. **Analyze:** find what sources agree on and where they contradict each other.
7. **Write:** produce a Markdown report with numbered citations and save it to memory.

A `Budget` caps reflection rounds, total tasks, parallel workers and searches per worker.
Without caps, an agent can loop forever and run up a large bill.

The text each role sees (topic, findings, sources, progress) is built by pure functions in
[`context.py`](../rootlogic/context.py), so both engines share one copy.

## Step 7: Apply source rules in code, not in the AI → [`filters.py`](../rootlogic/filters.py)

The worker *rates* each source, and code *decides* whether to keep it. A source is dropped if
it is from a blocked domain, a duplicate, low relevance, or older than the recency limit.
Every drop is logged with a reason, for example "outdated (2019-01-01 < 2025-09-22)".

**Why:** rules written in code are predictable, testable and easy to explain. Low-credibility
sources are *kept but flagged*, so a dissenting view isn't silently hidden.

## Step 8: Human in the loop → [`control.py`](../rootlogic/control.py)

The orchestrator never talks to the terminal directly. It emits **events** (log lines) and
calls an `Interaction` object for four things: `ask`, `review_plan`, `override` and `on_event`.

- The terminal UI implements `Interaction`, and so do the tests.
- The web UI (Step 14) is one more implementation, with no change to the core.
- Ctrl-C only sets a "pause" flag. The orchestrator checks it between waves, so it never
  stops halfway through a step.

## Step 9: Storage → [`store.py`](../rootlogic/store.py)

One SQLite file holds seven tables, all keyed by session:

| Table | What it holds |
|---|---|
| `sessions` | topic, status, plan, summary, related topics, report path |
| `messages` | the conversation with the user (questions, answers, notes) |
| `events` | the full action log |
| `tasks` | each sub-task's status and finding |
| `sources` | every source seen, kept or dropped (with the reason) |
| `llm_calls` | tokens in/out, cache usage, web searches and dollar cost per call |
| `memory_fts` | full-text search index over past sessions, which powers recall and topic suggestions |

## Step 10: Prompts → [`prompts.py`](../rootlogic/prompts.py)

Each role has its own short instructions: clarifier, planner, researcher, critic, analyst and writer.
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
- The LangGraph engine resumes after a crash.
- The web API round-trips questions, plan edits and overrides, and replays SSE.

All 79 tests run in under a second with no internet.

## Step 12: Terminal UI → [`cli.py`](../rootlogic/cli.py)

It shows:

- A colored live log.
- The plan table, with approve/edit.
- The override menu when you press Ctrl-C.

It also adds the commands `history`, `log`, `usage`, `show`, `forget`, `resume`, `graph` and `web`.
`--offline` runs everything on the fake LLM, which makes the demo safe from network problems.

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

Testing it in a real browser caught two bugs the unit tests missed:
- A new task could reuse an existing id after the user dropped one. `Plan.next_id()` now
  uses the highest id, not the count.
- Sessions orphaned by a server restart stayed "running" forever. They're now marked
  `interrupted` on startup, and graph-engine ones can be resumed.

## Step 15: Memory that makes research better → [`continuity.py`](../rootlogic/continuity.py)

Two features let rootlogic use the user's own input across sessions:

- **A learned profile.** At the end of a session in which the user answered questions or
  left notes, one small structured call (`ProfileUpdate`) pulls out *lasting* preferences into a
  `preferences` table: audience, region, preferred or avoided sources. It also removes any the
  user contradicted. The next session puts them into every prompt, and the clarifier doesn't
  re-ask what they already cover. The user can view and delete entries, which matters for trust.
- **Follow-up threads.** `load_previous` rebuilds a finished session's findings, sources and
  user answers. A new session is linked to it by `parent_id` and starts with those findings as
  already-done `previous` tasks. The planner sees which questions are already answered and plans
  only new ones, and already-seen URLs count as duplicates.

Design choices worth explaining:
- Learning is **best-effort**: if the profile call fails, the finished research still succeeds.
- The call is **skipped entirely** when the user said nothing, so it costs nothing on
  auto-approved runs.
- Remembered text is treated as **data, not instructions**, like web content.

## Step 16: Check it, then publish

The build was checked in this order:

1. Ran the tests.
2. Ran the offline demo. It caught one bug: Rich treated `[t1]` as a formatting tag and hid it.
3. Initialized git and committed.
4. Created the GitHub repo and pushed.

---

## What to do next

1. **Get an API key** at console.anthropic.com. Run `export ANTHROPIC_API_KEY=...`, then
   `rootlogic research "a topic you know well"`. The live Claude path has not been exercised
   yet, so the first run may surface small API-shape fixes.
2. **Read the code in this order:** `models.py` → `orchestrator.py` (start with `run()`) →
   `llm.py` → `filters.py` → `context.py` → the tests → `graph.py`.
3. **Rehearse the demo:**
   - A vague topic (shows clarifying questions).
   - Editing the plan.
   - Ctrl-C with an `add` or `note` override.
   - `rootlogic usage <id>` for cost.
   - A second related topic (shows memory).
   - `--engine graph`: kill it mid-run, then `rootlogic resume <id>`.
   - `rootlogic web`: the same flow in the browser, including refresh mid-run.
4. **Be ready to explain:**
   - The LLM-decides-what / code-decides-how split.
   - Why structured outputs matter.
   - Why subagents get isolated context.
   - Why budgets exist.
   - Why the source rules live in code.
   - What LangGraph added and what it cost.

For the wider context (prior art, framework trade-offs, storage options, research protocols,
UX patterns, search tools), read the research brief:
[research/agentic-research-assistant.md](research/agentic-research-assistant.md).
