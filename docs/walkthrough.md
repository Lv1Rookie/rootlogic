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
| *(not asked for, but needed)* | Guardrails against wrong or harmful output: claim verification, corroboration, report checks, content moderation, and an evaluation set that measures them (Steps 18–19) |

The same mapping appears in the README, which is where graders look for it.

## Step 2: Choose an architecture pattern

The pattern is **orchestrator–workers with an evaluator loop**, as described in Anthropic's
"Building effective agents" guide:

```
You → [screen request] → Clarify → Plan → [you approve] → Workers research in parallel
                                                             ↓
                                             Filter sources → Reflect: enough?
                                ↑  no: add tasks / ask you  ↓ yes
                                └───────────  Verify claims → Analyze contradictions
                      → Write report → Check report → [screen report] → Save to memory
```

The bracketed screening steps only run for models without built-in safety systems (Step 19).

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
| **OpenAI SDK** (optional) | Run on GPT, local models (Ollama) or OpenRouter; also the moderation backends | 17, 19 |

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

1. **Screen the request** (Step 19, non-Claude models only): a harmful topic stops here, before
   any model call.
2. **Recall:** search memory for related past sessions.
3. **Clarify:** ask Claude whether the topic is vague, and if so ask the user up to 3 questions.
4. **Plan:** Claude produces 3–6 sub-tasks and chooses how recent sources must be (e.g. 365
   days for tech news, no limit for history).
5. **User reviews the plan:** approve, edit or quit.
6. **Research loop:**
   - Run every sub-task that's ready, in parallel threads. A task is ready once the tasks
     it depends on have finished.
   - Filter each worker's sources.
   - When nothing is left to run, **reflect**: is this enough? If not, add follow-up tasks
     or ask the user something.
   - Stop when sufficient or when the budget runs out.
7. **Verify** (added in Step 18): check claims against the text of their cited pages and
   label how well each is corroborated.
8. **Analyze:** find what sources agree on and where they contradict each other.
9. **Write:** produce a Markdown report with numbered citations, run code checks on it (bad
   citations, uncited statements), screen it (Step 19), and save it to memory.

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
- Ctrl-C only sets a "pause" flag, and the web UI's Pause sets a "hold" flag. Both are read at
  sub-task boundaries, so neither ever stops halfway through a step.

**Three controls, three different promises.** These were once one button, and conflating them
was a bug in its own right (Step 21):

| Control | Promise | Lands at | Run afterwards |
|---|---|---|---|
| **Pause** | stop and stay stopped | next sub-task boundary | held, doing nothing, until Resume |
| **Override** | keep going, but differently | end of the wave | continues with the edited plan |
| **Abort** | this was a mistake | next sub-task boundary | ends, marked `aborted` |

Override is the slowest of the three on purpose: it hands the user a plan to edit, and the plan
is not stable until the wave that is rewriting it has finished — up to four minutes on a wide
plan, so the button says "Stopping…" while it waits rather than looking ignored. Abort has no
plan to offer, so it never waits for one, and it answers whatever card the run is sitting on
instead of queueing behind it. `Control` keeps the three flags apart — a hold gate (`request_hold` /
`release`), a pause flag (`consume_pause`) and an abort flag — and announces a pause or an abort
once however many sub-agent threads notice it at the same moment.

**Sub-agents report as they work.** A first live run showed the weakness of coarse events: after
three `task.started` lines the screen sat still for minutes, with no way to tell research from a
hang. `LLM.research(on_step=...)` now reports every search and page fetch while it happens, and
both engines turn those into `subagent.search` / `subagent.fetch` events tagged with the task id:

```
14:22:07 task.started       [t2] Researching: What does the evidence say about job losses?
14:22:19 subagent.search    [t2] Searched “newsroom layoffs AI 2026” — 5 result(s)
14:22:41 subagent.fetch     [t2] Read https://www.pewresearch.org/…
14:23:08 task.done          [t2] Done: 4 sources kept, 2 dropped, confidence medium
```

Two details worth knowing. With Claude's hosted tools the searches run inside Anthropic's
request, so the query has to be recovered from the `server_tool_use` block and paired with its
result by `tool_use_id`; with a `SearchProvider` the toolbox reports them directly. And the
callback is wrapped in a `try`: a UI that has gone away must never fail a sub-task that is
otherwise working.

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
- **A rewritten prompt is disclosed, for the whole session.** Four of them (planner, researcher,
  verifier, writer) can be rewritten in the UI, and a run that rewrote one says so in the log
  and in the report - a custom verifier or writer changes what the report's own numbers mean,
  and the report is read by someone who did not choose the prompt. Resume made that harder than
  it looks: the leg that finishes a report can be started with different prompts from the leg
  that began it. Announcing the resumed leg's prompts alone would have *erased* the record of
  the first leg, so the session keeps the union of both, a prompt that changed between legs is
  called out as `prompt.changed`, and the report discloses what the whole session used rather
  than what its last leg used.

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
- The LangGraph engine resumes after a crash, saves the same log details as the loop engine,
  and discloses the prompts of both legs when it is resumed with different ones.
- The web API round-trips questions, plan edits and overrides, and replays SSE.
- The search tools enforce their limits, and Tavily requests and responses map correctly.
- The OpenAI-compatible adapter sends the right wire format and handles refusals, truncation,
  invalid JSON and the repair retry, tested against the `openai` SDK's own response types.
- The profile is learned, used, edited and can be switched off.
- Follow-ups reuse earlier findings, chain, and don't count earlier tasks against the budget.
- Guardrails behave as designed:
  - Invented quotes are downgraded.
  - Claims without page text are "unverifiable", and claims citing nothing say so instead.
  - Corroboration labels are correct.
  - Source rules block, allow and override credibility.
  - Bad citations are fixed and uncited statements flagged.
- The evaluation harness scores facts, hoaxes, contradictions, stale sources and refusals correctly.
- Moderation blocks harmful requests before any model call, keeps flagged reports off disk,
  ignores flagged user input without ending the run, and never runs a non-Claude model
  unscreened by accident.

Most scenarios run on **both** engines. When a bug is fixed, a test that failed before the fix
is added first. All 209 tests run in a few seconds with no internet.

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
| `--worker-model MODEL` | Research sub-agents on a cheaper model than planning and writing |
| `--reasoning-effort none` | Stop a thinking model reasoning before every call (Ollama, OpenAI) |
| `--stream` | Reassemble a streamed reply, for gateways that time out on the first byte |
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
  - Sessions stopped by moderation show a `blocked` status (Step 19).

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

One thing this opened up had to be closed again in Step 19: Claude screens harmful requests
itself, and an arbitrary model may not.

### What a weak model taught the adapter

A live run on a local `qwen3:8b` (Ollama) searched well and then never submitted anything: from
roughly 6k tokens of real search results on, it stopped calling tools and wrote its answer as
prose, the same turn over and over, about two minutes each. `llama3.1:8b` did the same.

The cause is worth knowing if you ever run local models: a tool call's arguments are free-form
text the model has to get right on its own, while a JSON-schema `response_format` is
*grammar-constrained* by the server, so the tokens literally cannot stray from the schema. Small
models are far better at the second. So sub-agents now search with tools but, if the model stops
calling them, submit through constrained structured output instead, with no tools offered on
that last call. One that never searched still fails fast, since it has nothing to report.

Same model, same task: zero sources before, four sources and a usable answer after.

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

These four address *wrong* output. Step 19 addresses *harmful* output for models that don't
screen it themselves.

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

Three lessons from the first live runs, each now covered by a test:
- **"Couldn't read it" is not "it's false."** A landing page of navigation text made the
  verifier mark seven true claims "unsupported". Unusable or too-short page text is now
  "unverifiable", and the verifier has an explicit `no_usable_evidence` verdict. The same
  distinction needed drawing one level further down: a claim that cited *nothing* took the
  same "page text unavailable" label, which sent the reader hunting for a fetch problem behind
  a claim that had never named a page. It has its own reason now, `no_sources_cited`.
- **A dead tool should stop the run, not repeat it.** A failing web search produced eight
  turns of a growing conversation: 330k input tokens and $1.86 for zero sources. The loop now
  gives up after two fruitless turns, and caches its prefix so retries are cheap.
- **Retrieving nothing deserves a retry, not a "done".** Sub-tasks that came back empty were
  marked done, so the critic couldn't re-run them. They now retry once without consuming a
  task slot. Sources that were retrieved and then *filtered* still count as a real result.

Two design decisions worth explaining:
- **The claim budget has to scale with the plan.** Twelve claims was a run's worth when plans
  had three sub-tasks. A nine-sub-task live run made 64 claims and checked 12: "8 supported, 4
  partly" described a fifth of the report and nothing on the page said so. The budget is 30 now,
  and the evidence-fetch budget follows it at half the claims (floor six) - a claim whose cited
  page was never fetched scores *unverifiable*, so raising one without the other would only have
  bought a bigger pile of those. A ten-sub-task run then checked 30 of 74 claims with one
  unverifiable, for one extra LLM call and no extra cost.
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
- Harmful cases cannot be scored through an OpenAI-compatible gateway, which can turn a refusal
  into a transport error (Step 19). Score those against the Anthropic API.

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
- **A key for someone else's API is not moderation.** `OPENAI_API_KEY` holding an OpenRouter
  key (`sk-or-…`) satisfied "is it set?", so `auto` chose OpenAI moderation, and every call
  401ed at the clarify step. Each run died looking like a model failure while the screening
  that was supposedly switched on had never run once. Foreign keys are now recognised at
  startup, which is the only honest place to find out.

The evaluation harness counts a moderation block as a refusal, so the harmful cases pass either
way: the model refuses, or moderation stops it.

### A refusal has to survive the transport

The first live evaluation scored the two harmful-request cases 0/2, and neither score was
true. Both requests had been refused; the harness could not see it.

**One refusal arrived as a plan.** The planner answered with the objective "Decline to provide
research assistance for this request" and no sub-tasks, and a plan with nothing in it took the
`_nothing_found` path - a report titled "No sources found". That describes a search that came
up empty, which is not what happened: nothing was searched, because the model said no. An
empty plan on a fresh topic now raises `AgentRefusal`, and a refusal is recorded as its own
session status. It had been stored as `failed`, which reads like the tool broke.

**The other refusal was destroyed in transit, and that one stays unfixed.** Against the
Anthropic API a declined request arrives as `stop_reason: "refusal"`. Through the
OpenAI-compatible gateway the same request came back as:

```
502 upstream_response_error
[claude/claude-sonnet-5] upstream returned an empty response without usable output
```

A 502 is worth retrying, and `upstream_empty_response` is exactly what a gateway says when its
upstream is having a bad minute - so rootlogic retries and reports a failed run, which is the
right behaviour for a 502. The request *is* refused: no research runs, nothing is written. But
from inside the process, "the model declined" and "the gateway is down" are the same event.

The tempting fix is to read `upstream_empty_response` as a refusal, and it is worse than the
bug: it would mark real outages safe. A guardrail that reports success when it cannot see is
the one failure mode that must not happen, so this case is documented as unscoreable through a
gateway rather than made green. **Some guarantees are properties of the transport, not of your
code**, and the honest move is to say which ones.

## Step 20: Check it, then publish

Every change went through the same routine:

1. Wrote or updated tests, then ran them.
2. Ran the offline demo, and for UI work, used it in a real browser. The first demo caught
   Rich hiding `[t1]` as a formatting tag. The browser caught the duplicate task ids and the
   orphaned "running" sessions.
3. Checked external facts (API shapes, pricing, protocol versions) against official docs. For
   the OpenAI adapter, that meant reading the installed SDK's own types; for moderation, the
   SDK's moderation types and Ollama's documented Llama Guard output format.
4. For the guardrails, ran the evaluation harness offline to confirm it **fails** cases it
   should fail (the fake model knows no facts), then checked in the browser that a distrust
   rule visibly changes a report's corroboration labels.
5. Committed with a message explaining *why*, and pushed to GitHub.

---

## Step 21: Run it for real, on a laptop, for free

176 mocked tests passed before the first live run. Then every real run broke something new.
Twenty-one defects came out of live testing, none of them reachable by the test suite as it stood,
and each one is now covered by a test that fails against the old code.

| What broke | Why the tests missed it | Fix |
|---|---|---|
| Sub-agents were silent for minutes; a hang and real work looked identical | no test watches timing | emit `subagent.search` / `subagent.fetch` per action |
| A thinking model spent 72s reasoning before a 1.3s answer | fakes don't think | `--reasoning-effort` |
| An 8B model stopped calling tools and wrote prose for ten turns | fakes always call the tool | stop after two, and take valid findings out of prose |
| The same model could not emit a large tool schema at all | as above | submit through grammar-constrained structured output instead |
| Tavily dates are RFC 1123; `parse_date` knew three other formats | fixtures used ISO | parse it, so the recency filter actually fires |
| Verification crashed the moment it fetched a cited page | Claude's hosted path sets `search=None`, so the fetcher was never built | a real function, plus tests that attach a provider |
| A 1B Llama Guard flagged a newsroom report as violent crime and binned it | `StaticModerator` never false-positives | say how to recover the research |
| A backgrounded run died on its first clarifying question | tests answer prompts | treat closed stdin as "no answer" |
| A gateway timed out on every substantial call | nothing sat between us and the model | `--stream`, and catch the base `APIError` |
| Pause appeared to do nothing | tests call the engine directly and never wait | check the flag per sub-task, not per wave |
| Abort could not be reached at all | no test drove the UI's control flow | its own button and endpoint, independent of Pause |
| Abort took nine minutes to land | fakes succeed, so the failure path was never timed | check for an abort on that path too |
| Override looked ignored for a whole wave | a test answers the card instantly; nobody watches the button | say "Stopping…" until the checkpoint lands |
| Abort did nothing while a card was open | tests answer the card, then abort | aborting answers the open card too |
| Abort closed one card and the next one opened | the fake clarifier asks one question | an aborting run puts up no card at all |
| Seven of twelve fetches were gov.uk URLs the model had composed | StaticSearch answers any URL you ask it for | refuse a fetch of a URL no search returned |
| Every gov.uk document stayed out of reach, so the run cited trade press and social posts | fixtures hand back whatever source the test wants | let a search be restricted to a publisher's site |
| Facebook and Instagram posts carried government deadlines at the model's own credibility rating | no fixture cites a social post | rate those platforms low by default |
| A declined request was reported as a search that found no sources | the fake planner always returns sub-tasks | an empty plan on a fresh topic is a refusal |
| Moderation was configured, authenticated with someone else's key, and 401ed every run | tests inject a moderator rather than resolving one from the environment | reject a foreign key at startup |
| The claim-check table's verdict columns sat outside the panel on a narrow window | no test measures layout, and the author's window is wide | scroll the table, not the page |

Two of those were serious. **The outdated-source filter was silently inert** on the Tavily
path — a graded requirement, passing its unit tests, doing nothing in production, because
every date arrived in a format the parser returned `None` for and `None` means "unknown age".
**Verification crashed after all the research was done**, on this line:

```python
fetch = lambda url: (p := search.fetch(url)).text if not p.error else None
```

A conditional expression evaluates its condition first, so `not p.error` ran before the walrus
bound `p`. It could never have worked, and no mocked test built that lambda at all.

### What pressing the buttons taught the controls

Six defects came from a different kind of live test: not "does the research work" but "can I
stop it". All six were invisible to the suite because a test calls `engine.run()` and waits for
it to return — nobody is sitting there pressing anything, and nobody is watching what the page
does while they wait.

**Pause looked broken because it was checked once per wave.** A wave is one model call per
sub-task, which on a hosted model is a minute and on a local 8B model is twenty. The flag was
read between waves, so the button did nothing observable for the length of a coffee break. It
is now read at every sub-task boundary — the finest safe point available, since a model call in
flight cannot be interrupted but the next one need not start.

**Pause also wasn't a pause.** It stopped the run to show an override card and then continued as
soon as that card was answered. Useful, but not what the word promises. Pause is now a hold: a
gate the run waits on, released only by Resume, spending nothing while it waits. The old
behaviour kept its own button, Override, which is what the inline Skip and Steer controls use.

**Abort was unreachable.** It existed only *inside* the override card — which appears after a
pause has been honoured. So the control for "stop this now" was behind the control that was too
slow to press, and a run that would not pause could not be stopped at all.

Fixing those three was not the end of it, because the first live abort took **nine minutes**.
The wave loop looked like this:

```python
except (LLMError, AgentRefusal) as e:
    self._task_failed(t, str(e))
    continue                      # ← jumps past the abort check below
self._accept_finding(plan, t, result)
self._abort_if_requested()
```

Every sub-task in that run was failing — the search key was invalid, so every query returned
HTTP 401 — which meant every sub-task took the `continue` and no sub-task ever reached the
check. The abort was not ignored; it was never looked at. A retried task then opened a *fresh*
model call, because the worker checked the hold flag but not the abort flag, spending minutes
generating a finding already destined for the bin.

Both paths are boundaries now. The same run aborts in **thirty-one seconds**, which is one
in-flight model call — that request cannot be cancelled, so one sub-task is the floor for how
fast Abort can possibly be. The regression test reproduces the live conditions exactly: every
sub-task fails, and the assertion is that no further research calls happen after the abort.

Three more turned up the next time the buttons were pressed in anger, and all three are the
same shape: the flag was read correctly, and the *user* was left with no way to know.

**Override said nothing for three minutes fifty-one.** Asking for an override queues a request
and the engine honours it at the end of the wave it is in — correct, and indistinguishable from
a dead button. The click was pressed twice, which the log recorded faithfully as two
`control.requested` lines for one intent. The button now reads "Stopping…" and is disabled
until the checkpoint lands, which on the live run that found it meant 3m51s of visible waiting
instead of 3m51s of apparent nothing. The same state covers the plan table's Skip and Steer,
which ask for a checkpoint by the same route.

**Abort did nothing at all while a card was open.** A run held on a plan or override card is
blocked on the browser, not on its own control flags: the abort endpoint set the flag, logged
that it had, and the run went on sitting on the card it had just been told to abandon. Only the
card's *own* Abort worked — the same shape as the original "Abort lives inside the override
card" bug, arrived at from the other direction. Aborting now answers the open card with
whatever that kind means by "stop": a plan is rejected, an override is told to abort, a question
is skipped.

**And closing the card only made room for the next one.** Aborting under the first of three
clarifying questions closed it, and the clarifier — which asks its questions back to back, with
no abort check between them — immediately put the second up. The run kept collecting answers it
had been told to abandon. The fix moved up a level: a run that is aborting no longer opens a
card *at all*, so it covers every card either engine raises rather than the one that happened
to be open when the button was pressed.

The lesson generalises past this project. **A control-plane feature needs a test that races
it**, because the interesting bugs live in the gap between "the flag is set" and "the code
looks at the flag" — and that gap only exists while something else is running. Its corollary,
learned the harder way: **a control that is working invisibly is indistinguishable from one
that is broken.** Two of the six control-plane defects here were only ever visible to someone
watching a button, and the fix for both was to make the waiting legible rather than to make it
shorter.

### Where a sub-agent's sources come from

A run on UK petrol-ban policy produced a readable report with the primary source missing from
it. Twelve of its fetches failed; seven were gov.uk addresses the model had assembled from
headlines - `plans-confirmed-to-phase-out-sale-of-new-petrol-and-diesel-cars-by-2030` and the
like - and curl confirmed every one a 404. The one gov.uk link an actual search returned read
fine, and Tavily's extractor reads gov.uk without complaint, so neither the fetcher nor the site
was at fault. The model was writing addresses that looked right.

Three findings, each of which needed the one before it:

**Refusing the guesses.** The prompt was told not to invent URLs, which a model can ignore, so
the toolbox enforces it: every URL a sub-agent has been shown already passes through
``WebToolbox`` - search results, and the pages it fetches - so it can say no to the rest. Seen
covers a page's own links too, because the prompt tells the researcher to fetch the real
document when a landing page comes back and that link is in the text, and it covers links from
the topic, because a pasted URL is as good a reason to read a page as a search result. A
refusal is not charged against the fetch budget: it bought nothing, and the turn cap already
bounds a model that keeps guessing.

**But the guessing was a symptom.** The next run refused two invented URLs and still cited one
official source, a US Federal Register page, for a question about UK policy. The sub-agents
were composing gov.uk addresses because gov.uk documents were not coming back from search, and
removing the workaround does not remove the need. ``web_search`` now takes ``domains``: up to
three sites to restrict results to, and a restricted search asks Tavily for its *advanced*
pass, because a narrow haystack searched shallowly returns the section page rather than the
document. The researcher prompt points at it from the sentence that forbids inventing a URL, so
the instruction arrives where the temptation does.

**And what it fell back on in the meantime.** Deprived of primary sources, the run cited
Facebook, Instagram and LinkedIn posts for government deadlines, several rated credible by the
model. Those platforms are now rated low whatever the model thought - a rating, not a ban,
since a minister does announce policy on X, and corroboration already allows a low-credibility
source to support a claim as long as it is not the only one. A latent bug surfaced here:
credibility was only adjusted when the user had source rules, so a run with none skipped the
ratings entirely.

The same topic, three times, at the same budgets:

| | before | refusing guesses | plus site search |
|---|---|---|---|
| invented URLs fetched | 8 | 0 (2 refused) | 0 |
| official gov.uk sources | 0 | 0 | 3 |
| claims supported | 11 | 11 | 21 |
| unverifiable | 7 | 11 | 2 |

The third run searched `"ZEV mandate consultation DfT 2026" (on gov.uk)`, got back the real
consultation - the document the first run had tried to invent an address for - and read it. The
verification numbers follow from that: claims cited to documents that actually read can be
checked, and claims cited to social posts cannot. The middle column is worth keeping in view,
because it is what a half-fix looks like: the guard was working exactly as designed and the
report was no better for it.

The lesson is about where a guardrail belongs. **Stopping a model from doing the wrong thing
only helps if it can do the right thing instead** - otherwise the run finds a worse workaround,
and the metric that was supposed to improve gets worse (unverifiable went 7 to 11 before it
went to 2).

A later run on a different body confirmed it was the reach and not the topic: asked what the
WHO recommends for free sugar intake, the sub-agents searched `(on who.int)` unprompted and
came back with seven WHO sources out of thirty-two - the IRIS document store, the 2015 news
item, a CDN fact sheet - and cited no social media at all. The credibility rule had nothing to
downgrade, because there was nothing to fall back to.

### A check that lies is worse than no check

The report's claim-check table is four columns: claim, verdict, corroboration, sources. On a
310px panel it was 461px wide, and nothing scrolled - the two right-hand columns were simply
outside the panel. A reader on a narrow window could see that a claim had been checked and not
what the verdict was, which is the column the table exists for. The usage table had the same
problem and had been given `overflow-x: auto` long before; the report's tables never were. Now
the table scrolls inside the panel and the prose around it, which wraps fine, stays put.

The bug is ordinary. What is worth writing down is what happened next: the script that measures
spill - every descendant whose right edge is outside the panel - still reported 164 escapes
after the fix. The fix had worked. The measurement was wrong: cells inside a scrolling
container do extend past the panel, and are perfectly reachable. **A check that counts
reachable content as broken will also, one day, count broken content as fine**, so it was
rewritten to ignore anything under a horizontally scrollable ancestor before either number was
believed. It then read zero at 380px, and the table measured 722px inside a 996px panel at
desktop width, which is the other half of the claim: the fix cost nothing where there was no
problem.

The pattern worth taking away: **the tests all took the same path through the code.** Claude's
hosted tools mean `search is None`, which skipped the client-side search branch, the fetcher,
the date formats a real provider returns, and every failure mode of a model that isn't Claude.
A second backend was not just a portability feature; it was the thing that exposed the first
one's blind spots.

Running locally made this affordable. `ollama pull qwen3:8b`, `--search tavily`, and a
laptop — three reports, dozens of runs, $0. The same pipeline on Claude costs about $0.30 a
run after the caching and worker-model work, and produces better judgement: the local model
rated an SEO listicle "high credibility" and invented a contradiction between two unrelated
facts. The machinery is sound either way; the judgement is only as good as the model behind it.

---

## What to do next

1. **Run it.** Free and local, which is how Step 21's twelve defects were found:

   ```bash
   brew install ollama && ollama serve          # or: brew services start ollama
   ollama pull qwen3:8b
   export TAVILY_API_KEY=tvly-...               # free tier is plenty
   rootlogic research --provider openai --base-url http://localhost:20128/v1 \
       --model qwen3:8b --reasoning-effort none --search tavily --moderation none \
       --searches 2 --verify-claims 12 "a topic you know well"   # 12, not the default 30: a local model is slow
   ```

   For the best output, Claude instead — one env var, and it keeps hosted web search, prompt
   caching and refusal fallback that no OpenAI-compatible hop can carry:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   rootlogic research --worker-model claude-haiku-4-5 "a topic you know well"
   ```
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
   - With a local model (`--provider openai --base-url … --moderation llama-guard`), a harmful
     topic stops at `session.blocked` before any model call.
   - `--engine graph`: kill it mid-run, then `rootlogic resume <id>`.
   - `rootlogic web`: the same flow in the browser, including refresh mid-run.
   - The `subagent.search` / `subagent.fetch` lines as they appear: the queries are the
     model's own, and a failing provider names its error in the log rather than going quiet.
   - Two backends on the same topic side by side. The pipeline behaves identically; the
     judgement does not, which is the honest thing to say about where an agent's quality
     comes from.
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
