# Shipped production AI research assistants: how they make information verifiable

Scope note on method: findings below are marked **[fetched]** where I read the first-party page in full, and **[search-summary]** where the only access I had was a search engine's summary of the first-party page (403s and redirects blocked several direct fetches: `openai.com/index/introducing-deep-research`, `help.consensus.app`, `perplexity.ai/hub/blog`, and the Vertex REST reference). Treat **[search-summary]** items as first-party in origin but second-hand in transcription — the report writer should re-verify exact wording before quoting.

Currency note: all first-party pages were read on 2026-09-23. Where a feature is older than 2026 I say so inline.

---

## Q1. How do these products attach sources to claims — per-sentence, per-paragraph, or per-report?

### Takeaway
There are three distinct architectures in production. (a) **Character-span annotation over the generated text** — OpenAI (`url_citation` with `start_index`/`end_index`) and the current Gemini API. (b) **Structured citation objects attached to individual output text blocks, with the quoted source text carried back** — Anthropic (`citations` array on each text block, containing `cited_text`). (c) **Index-span grounding metadata computed over the answer after generation** — Vertex/Gemini `groundingSupports`, which maps response character spans to retrieved chunks. Only Anthropic returns the verbatim supporting passage as a first-class API field; the others return a pointer (URL + span) that the client must resolve.

### Cited Findings

**Anthropic — Citations API (GA; first shipped Jan 2025, still GA in 2026)**
- Citations are attached per **output text block**, and each citation object carries the exact supporting passage. Fields for a text document: `"type": "char_location"`, `cited_text`, `document_index`, `document_title`, `start_char_index` (0-indexed), `end_char_index` (exclusive) — [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) [fetched]
- Granularity is set by **chunking of the source document**, not by the model: "Document contents are 'chunked' to define the minimum granularity of possible citations. For example, sentence chunking lets Claude cite a single sentence or chain together multiple consecutive sentences to cite a paragraph or longer passage." Plain text and PDFs are "chunked into sentences"; custom content documents are used as-is with no further chunking — [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) [fetched]
- PDFs use `"type": "page_location"` with `start_page_number`/`end_page_number` instead of char indices; custom content uses block indices (0-indexed) — [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) [fetched]
- The key engineering claim: "**Because the API parses citations into the response formats described in the following sections and extracts `cited_text` directly, citations are guaranteed to contain valid pointers to the provided documents.**" i.e. the citation *pointer* cannot be fabricated even if the surrounding prose is wrong — [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) [fetched]
- Cost mechanics that matter for design: "Internally, the model outputs citations in a standardized format that are then parsed into cited text and document location indices. The `cited_text` field is provided for convenience and does not count toward output tokens," and is not counted as input tokens when passed back — [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) [fetched]

**Anthropic — `search_result` content blocks (RAG citation, no beta header)**
- A `search_result` block has required `source` (URL or identifier), `title`, `content` (array of text blocks), and optional `citations: {enabled: true}` — [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results) [fetched]
- Citations come back as `"type": "search_result_location"` with fields: `source`, `title`, `cited_text` ("the full text of the cited block(s), concatenated"), `search_result_index` (0-based index across all search_result blocks in the request), `start_block_index`, `end_block_index` (exclusive) — [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results) [fetched]
- Explicit statement of the minimal citable unit: "**The text block is the minimal citable unit: Claude cites whole blocks, not substrings within a block. To get finer-grained citations, split your search result content into smaller blocks.**" This is the copyable lever — citation granularity is a retrieval-pipeline decision — [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results) [fetched]
- "Claude cites the search results automatically when citations are enabled. No special prompting is needed" — [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results) [fetched]

**Anthropic — web search tool**
- "**Citations are always enabled for web search**" (not optional), and each `web_search_result_location` includes `url`, `title`, `encrypted_index` (must be passed back for multi-turn), and `cited_text`: "**Up to 150 characters of the cited content**" — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]
- The citation fields `cited_text`, `title`, `url` "do not count toward input or output token usage" — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]
- Search result blocks separately carry `url`, `title`, `page_age` ("When the site was last updated"), and `encrypted_content` — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]

**OpenAI — Deep Research API and web search**
- Citations are **character-span annotations on the final message**: each `url_citation` annotation has `url`, `title`, `start_index`, `end_index` — "the character span in the text the citation refers to. This structure will allow you to build a citation list or bibliography, add clickable hyperlinks in downstream apps, and highlight & trace data-backed claims in your report" — [Deep research guide](https://developers.openai.com/api/docs/guides/deep-research) [fetched]; [Web search guide](https://developers.openai.com/api/docs/guides/tools-web-search) [fetched]
- A separate `sources` field returns "the complete list of URLs consulted, which often exceeds the number of inline citations shown," including real-time feeds labelled `oai-sports`, `oai-weather`, `oai-finance`. **This is a distinct consulted-vs-cited distinction that only OpenAI exposes explicitly** — [Web search guide](https://developers.openai.com/api/docs/guides/tools-web-search) [fetched]
- Intermediate steps are inspectable in the `output` array: `web_search_call` (queries and page-open actions), `code_interpreter_call`, `file_search_call`, `mcp_tool_call` — so the full research trajectory is auditable, not just the bibliography — [Deep research guide](https://developers.openai.com/api/docs/guides/deep-research) [fetched]
- Display obligation on the integrator: "When displaying web results or information contained in web results to end users, inline citations should be made clearly visible and clickable in your user interface" — [Web search guide](https://developers.openai.com/api/docs/guides/tools-web-search) [fetched]

**Google — Gemini API (ai.google.dev), current surface**
- The current Gemini API grounding page documents a **step/annotation** model rather than the older groundingMetadata: `google_search_call` ("Contains the search `queries` the model executed"), `google_search_result` ("Contains `search_suggestions`, an HTML snippet for rendering search suggestions in your UI"), `annotations` of type `url_citation` with `url`, `title`, `start_index`, `end_index`, and a `steps` array containing `thought`, `google_search_call`, `google_search_result`, `model_output` — [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search) [fetched]
- Notably, **confidence scores and dynamic retrieval are not documented on this page** — they belong to the older Vertex `groundingMetadata` surface (see Q2) — [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search) [fetched]
- Grounding with Google Search "can also be used in combination with the URL context tool to ground responses in both public web data and the specific URLs you provide" — [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search) [fetched]

**Perplexity (Sonar / Agent API)**
- Response carries `citations` ("URLs of sources used to generate the response") and `search_results` ("Search results used for context in the response") with title, URL, publication date, last update date, snippet, and source type — [Chat Completions API reference](https://docs.perplexity.ai/api-reference/chat-completions-post) [fetched]
- Attachment is **per-report with numbered inline markers**, not per-span: the API exposes a flat source list, and the model emits bracketed indices into that list. No character-span field is documented — [Chat Completions API reference](https://docs.perplexity.ai/api-reference/chat-completions-post) [fetched]
- Deprecation relevant to 2026 currency: "Sonar Chat Completions is now Agent API," with Sonar support ending **September 27, 2026** — [Models](https://docs.perplexity.ai/getting-started/models) [fetched]

**Consensus**
- Per-claim citation with the verbatim quote surfaced in the UI: the platform shows "the exact quote behind every citation," with hover to preview and click to "jump straight to that passage in the paper"; enabled by "full-text partnerships." Described as "a verification layer" — [Citation Grounding: Evidence in Every Answer](https://consensus.app/home/workshops/citation-grounding/) [fetched]
- Corpus: "220M+ peer reviewed research papers" — [Citation Grounding](https://consensus.app/home/workshops/citation-grounding/) [fetched]. A separate help page states 200M+ papers, so the figure has moved; use the 220M figure with its date caveat — [How Consensus Works](https://help.consensus.app/en/articles/9922673-how-consensus-works) [search-summary]

**Elicit**
- Grounding is by **extraction, not generation**: "We ensure that the info we use is extracted directly from papers or generated based on research papers. We then highlight the source of the content within the relevant papers." Attachment is per-extracted-cell (column value per paper) rather than per-sentence of prose — [Elicit's reliability](https://support.elicit.com/en/articles/552897) [fetched]

**Gemini Deep Research (consumer)**
- Reports are "neatly organized with links to the original sources"; during the run the UI exposes "Show thinking" and "Sites browsed," the latter listing every site consulted — [Use Deep Research in Gemini Apps](https://support.google.com/gemini/answer/15719111) [search-summary]; [Try Deep Research](https://blog.google/products/gemini/google-gemini-deep-research/) [search-summary] (Dec 2024 launch post — older feature)

**Anthropic Research (claude.ai)**
- Claude "conduct[s] multiple searches that build on each other while determining exactly what to investigate next... This approach delivers thorough answers, complete with easy-to-check citations" — [Claude takes research to new places](https://www.anthropic.com/news/research) [search-summary] (April 2025 launch — older feature); availability at launch was "early beta for Max, Team, and Enterprise plans in the United States, Japan, and Brazil" — [Use research on Claude](https://support.claude.com/en/articles/11088861-use-research-on-claude) [search-summary]

### Inferences
- Two engineering patterns are copyable today. **Pointer-based** (OpenAI, Gemini annotations): cheap, but a citation can point at a real URL that does not contain the claim — verification requires re-fetching. **Extract-and-carry** (Anthropic `cited_text`, Consensus exact-quote): the supporting text is returned with the answer, so a client can verify support without a second network round trip, and can diff the quote against the claim.
- Anthropic's design deliberately makes the *citation pointer* unfalsifiable while leaving the *prose* fallible. That is a narrower guarantee than "no hallucination" and should be described that way.
- Citation granularity in Anthropic's stack is a property of how the caller chunks retrieval output. An engineer copying this gets per-sentence citations for free on plain text/PDF, and must split blocks manually for custom content.

### Gaps
- No first-party documentation found for how Perplexity's **sonar-deep-research** attaches citations differently from plain Sonar (e.g. per-section). The models page describes capability, not citation mechanics.
- No first-party spec found for citation granularity inside Gemini Deep Research consumer reports (per-sentence vs per-paragraph) — the help page only confirms links to sources exist.

---

## Q2. What happens to an unsupported claim? Grounding scores and confidence output

### Takeaway
Only Google ever shipped a numeric per-span support score (`confidenceScores` in `groundingSupports`, 0–1), and **it has been retired for Gemini 2.5 and later — the list is now empty and must be ignored**. No other vendor exposes a confidence or grounding score in its API. Abstention is a product-level behaviour (Consensus claims it; OpenAI's own system card documents the opposite tendency) rather than an API contract.

### Cited Findings

**Google Vertex AI — `GroundingSupport` (the "grounding score" people cite)**
- `groundingChunkIndices`: "A list of indices into 'grounding_chunk' specifying the citations associated with the claim. For instance [1,3,4] means that grounding_chunk[1], grounding_chunk[3], grounding_chunk[4] are the retrieved content attributed to the claim" — [GroundingMetadata REST reference](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/GroundingMetadata) [search-summary]
- `confidenceScores`: "**Confidence score of the support references. Ranges from 0 to 1. 1 is the most confident. This list is parallel to the groundingChunkIndices list.**" — [GroundingMetadata REST reference](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/GroundingMetadata) [search-summary]
- **Version cutoff (critical for 2026 currency):** "For Gemini 2.0 and before, the confidenceScores list must have the same size as the groundingChunkIndices. **For Gemini 2.5 and after, this list will be empty and should be ignored.**" — [GroundingMetadata REST reference](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/GroundingMetadata) [search-summary]; same text mirrored in the client library reference — [GroundingSupport (Node.js client)](https://docs.cloud.google.com/nodejs/docs/reference/generativelanguage/latest/generativelanguage/protos.google.ai.generativelanguage.v1beta.groundingsupport) [search-summary]
- A `groundingSupport` object contains a `segment` with `startIndex`/`endIndex`/`text`, plus `groundingChunkIndices` and (legacy) `confidenceScores`; a worked example shows `groundingChunkIndices: [1,2]` with `confidenceScores: [0.6626542, 0.82018316]` — [GroundingMetadata REST reference](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/GroundingMetadata) [search-summary]
- Container fields: `groundingMetadata` holds `groundingChunks` (retrieved passages/sources), `groundingSupports`, `webSearchQueries` (queries executed), `searchEntryPoint`, and `retrievalMetadata`; `dynamicRetrievalConfig` configures "dynamic retrieval threshold settings, controlling when additional information is fetched" — [Grounding overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview) [fetched]
- **Caveat the report writer must carry:** I could not verify `dynamicThreshold`'s default value or the `googleSearchDynamicRetrievalScore` field by direct fetch (the Vertex pages 403'd or returned navigation chrome only). Dynamic retrieval is also a Gemini-1.5-era feature; the current ai.google.dev page does not document it at all — [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search) [fetched]

**OpenAI Deep Research — documented *reduced* willingness to abstain**
- The system card states deep research, "similar to o1-preview, is **significantly less likely to select that it doesn't know an answer to a question**," reported alongside 63% accuracy on the BBQ ambiguous-questions split — [Deep Research System Card](https://deploymentsafety.openai.com/deep-research) [fetched] (Feb 2025 — older, but it is the only first-party statement on abstention behaviour I found)
- No automated guarantee against hallucination is offered; the API guide's mitigation is procedural — "log and review tool calls and model messages," and screen links before sharing with users — [Deep research guide](https://developers.openai.com/api/docs/guides/deep-research) [fetched]

**Consensus — documented abstention**
- "If Consensus can't find sufficient relevant evidence, it will tell you rather than filling in the gaps with outside information" — [How Consensus Works](https://help.consensus.app/en/articles/9922673-how-consensus-works) [search-summary]. This is the clearest vendor statement of an abstention policy I found; it is vendor-reported and unquantified.

**Anthropic — no confidence field**
- Neither the Citations doc nor the web search doc exposes any score. The guarantee is structural (valid pointers, verbatim `cited_text`) rather than probabilistic — [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) [fetched]; [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]

**Perplexity — no confidence field**
- No grounding score, support score, or abstention flag appears in the Chat Completions response schema — [Chat Completions API reference](https://docs.perplexity.ai/api-reference/chat-completions-post) [fetched]

### Inferences
- **An engineer wanting a numeric per-claim support score cannot get one from any current-generation model API.** The Vertex `confidenceScores` field is the only shipped example and it is dead on Gemini 2.5+. Anyone copying this mechanism in 2026 must compute support themselves (e.g. NLI or a judge model over `cited_text` vs the claim) — which is exactly why Anthropic's `cited_text` and Consensus's exact-quote surfacing are the more useful primitives.
- Google's retirement of `confidenceScores` without replacement, combined with the move to `url_citation` annotations, suggests a deliberate shift from "score the grounding" to "expose the span and let the client verify."

### Gaps
- Why Google retired `confidenceScores` for 2.5+ is not explained in any first-party page I found.
- No vendor documents what happens to a *drafted* claim that fails grounding — whether it is dropped, rewritten, or emitted uncited. This appears to be undocumented internal behaviour across all six products.
- I found no first-party "grounding score" analogue at OpenAI, Anthropic, Perplexity, Elicit, or Consensus.

---

## Q3. Source quality and recency: allowlists, reputation, date filters, peer-reviewed modes

### Takeaway
Domain allow/deny lists are near-universal and are the main copyable lever; the limits differ sharply (OpenAI 100 domains, Perplexity 20, Anthropic unbounded but with a `request_too_large` failure mode). Only Perplexity ships true publication-date filters. Academic-corpus modes exist at Perplexity (`search_mode: "academic"`), Consensus (peer-reviewed corpus by construction) and Elicit.

### Cited Findings

**Perplexity — richest filter surface, all first-party documented**
- `search_domain_filter`: "Limit search results to specific domains"; allowlist = bare domain, denylist = `-` prefix (e.g. `"-reddit.com"`); **maximum 20 domains or URLs per request**; path filtering supported (`"nature.com/articles"`); "You can use either allowlist or denylist mode, but not both simultaneously in the same request" — [Search filters guide](https://docs.perplexity.ai/guides/academic-filter-guide) [fetched]; [Chat Completions API reference](https://docs.perplexity.ai/api-reference/chat-completions-post) [fetched]
- Date filters: `search_after_date_filter`, `search_before_date_filter`, `last_updated_before_filter`, `last_updated_after_filter`, all `MM/DD/YYYY`; `search_recency_filter` with `hour`, `day`, `week`, `month`, `year`. "Recency cannot combine with other date parameters" — [Search filters guide](https://docs.perplexity.ai/guides/academic-filter-guide) [fetched]
- The **publication-date vs last-updated distinction is unique to Perplexity** among these products and is directly copyable for freshness handling — [Search filters guide](https://docs.perplexity.ai/guides/academic-filter-guide) [fetched]
- `search_mode` accepts `web`, **`academic`**, and **`sec`** — a corpus switch, not a domain filter — [Chat Completions API reference](https://docs.perplexity.ai/api-reference/chat-completions-post) [fetched]. Academic focus "prioritizes peer-reviewed journals and scholarly articles" — [Academic and Scholarly Search cookbook](https://docs.perplexity.ai/docs/cookbook/articles/academic-search/README) [search-summary]
- `search_language_filter`: ISO 639-1 codes, **max 10** — [Search filters guide](https://docs.perplexity.ai/guides/academic-filter-guide) [fetched]
- `web_search_options`: `search_context_size` (low/medium/high), `search_type` (fast/pro/auto), `user_location` (`country` ISO-3166, `region`, `city`, `latitude`, `longitude`) — [Chat Completions API reference](https://docs.perplexity.ai/api-reference/chat-completions-post) [fetched]; [Search filters guide](https://docs.perplexity.ai/guides/academic-filter-guide) [fetched]

**OpenAI**
- `filters.allowed_domains`: "up to 100 domains"; `blocked_domains`: "Exclude up to 100 domains." Domains are written without scheme (`openai.com`), and "**subdomains are automatically included**" — [Web search guide](https://developers.openai.com/api/docs/guides/tools-web-search) [fetched]
- `user_location` with `country` (two-letter ISO), `city`, `region`, `timezone` (IANA) — but "**User location is unsupported for deep research models**" — [Web search guide](https://developers.openai.com/api/docs/guides/tools-web-search) [fetched]
- `search_context_size`: `low` (simple lookups), `medium` (balanced default), `high` (detailed answers) — a retrieval-depth knob, not a quality knob — [Web search guide](https://developers.openai.com/api/docs/guides/tools-web-search) [fetched]
- No publication-date filter is documented for OpenAI web search or deep research.

**Anthropic**
- `allowed_domains` / `blocked_domains` on the tool definition: "Use `allowed_domains` or `blocked_domains`, not both" — supplying both returns a 400. "Entries are bare domains with an optional path, for example `example.com` or `example.com/blog`, without a scheme" — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]
- No documented count limit, but an over-long list has a documented failure mode: error code `request_too_large` — "The search request is too large, typically because of a long domain filter list" — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]
- **Three enforcement layers, which is the interesting architectural detail:** (1) per-request tool fields; (2) org-level restriction in the Claude Console where an administrator "can also restrict which domains it searches," applying to Messages API requests only; (3) on Claude Managed Agents, sessions "use only the per-tool `allowed_domains` and `blocked_domains` lists on the agent toolset," ignoring the console settings — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]
- Freshness signal without a filter: each result carries `page_age`, "When the site was last updated" — the model sees recency but the caller cannot filter on it — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]
- **Dynamic filtering (new, `web_search_20260209` and later; Claude 4.6+):** "Claude can write and run code that filters the search results before they reach the context window... so only relevant content reaches the context window." It runs inside code execution — `allowed_callers` defaults to `["code_execution_20260120"]` on these versions; set `allowed_callers: ["direct"]` to bypass. `web_search_20260318` adds `response_inclusion: "excluded"` to drop consumed result blocks from the response — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]. This is a **programmatic, model-authored source filter** — a genuinely different mechanism from static allowlists and the most novel 2026 item in this research.

**Consensus — reputation-weighted ranking (the only documented reputation model)**
- "The high-precision model factors in **recency, citation count, and journal impact**, so the final list reflects both relevance and rigor" — [How Consensus Works](https://help.consensus.app/en/articles/9922673-how-consensus-works) [search-summary]
- Corpus is peer-reviewed by construction: 220M+ peer-reviewed papers — [Citation Grounding](https://consensus.app/home/workshops/citation-grounding/) [fetched]

**Google**
- Gemini Deep Research source control is by **source selection**, not domain filtering: "By default, Gemini includes Google Search as a source... You can change or add other sources, like your personal Gmail or Drive... You can also upload files and add NotebookLM notebooks" — [Use Deep Research in Gemini Apps](https://support.google.com/gemini/answer/15719111) [search-summary]
- Deep Research Max (blog dated **April 21, 2026**) can "search the web, arbitrary remote MCPs, file uploads and connected file stores — **or any subset of them**," producing "fully cited analyses" that draw "from authoritative sources like SEC filings and open-access peer-reviewed journals," and is described as consulting "a diverse array of sources and carefully weighing conflicting evidence against each other" — [Deep Research Max](https://blog.google/innovation-and-ai/models-and-research/gemini-models/next-generation-gemini-deep-research/) [fetched]
- The `dynamicRetrievalConfig` threshold (Gemini 1.5 era) is the one documented *automatic* decide-whether-to-ground mechanism — [Grounding overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview) [fetched]

### Inferences
- Domain allowlisting is the only quality mechanism available at the API layer for the general-purpose products; "source quality" in web-search products is delegated entirely to the underlying search index's own ranking, which no vendor documents.
- Consensus is the only product in this set that documents reputation signals (citation count, journal impact) as part of ranking. An engineer wanting reputation-weighted ranking has to build it, using those three signals as the template.
- Anthropic's dynamic filtering converts source selection from a static config into generated code — meaning the filter can be arbitrarily expressive (date parsing, keyword exclusion, deduplication) but is itself model-authored and therefore not auditable as a fixed policy.

### Gaps
- No vendor publishes a domain reputation list, allowlist seed set, or ranking weights.
- No first-party documentation found for a "peer-reviewed only" toggle in Elicit's interface, though its corpus is scholarly by construction.
- Whether Gemini Deep Research applies publication-date filtering internally is undocumented.

---

## Q4. Published hallucination / citation-accuracy rates, and how measured

### Takeaway
Published numbers are sparse, vendor-reported, and mostly measured on **factual-QA benchmarks rather than on citation support** — nobody in this set publishes a "fraction of claims actually supported by the cited source" figure. Elicit is the outlier: it reports task-level extraction accuracy and a hallucination rate benchmarked against human reviewers.

### Cited Findings
- **OpenAI (vendor-reported):** hallucinations evaluated on **PersonQA**, "which contains 18 categories of facts about people." OpenAI states the deep research model "is significantly more accurate and hallucinates less than prior models," and that "the heavy reliance on online search is designed to reduce such errors" — [Deep Research System Card](https://deploymentsafety.openai.com/deep-research) [fetched]
- **OpenAI's own caveat on its number (important):** "the hallucination rate noted above actually **overstates** how often deep research hallucinates, because in some instances its outputs were accurate and the information in the test set was out of date — for example, when queried about a well-known person's children, the model may accurately return more children than in the test set." This is a first-party admission that static fact benchmarks mismeasure live-retrieval agents — [Deep Research System Card](https://deploymentsafety.openai.com/deep-research) [fetched]
- **Elicit (vendor-reported):** reported 96% accuracy on Methods, Participants, and Interventions extraction, and "in a randomized noninferiority trial, Elicit's **1.0% hallucination rate matched experienced human reviewers across 5,100 data points**" — [Trust at scale: Auto-evaluation for high-stakes LLM accuracy](https://elicit.com/blog/auto-evaluation/) [search-summary]. **Flag:** these figures reached me via search summary, not direct fetch; the writer should verify wording and the trial's design before quoting the noninferiority claim.
- **Elicit's stated methods (vendor-reported):** "process supervision, prompt engineering, ensembling multiple models, double-checking our results with custom models and internal evaluations," plus "lots of internal evaluations to test how common hallucinations are" — [Elicit's reliability](https://support.elicit.com/en/articles/552897) [fetched]
- **Elicit's published research method:** Factored Verification — decomposing a summary into claims and verifying each against the source using AI supervision — [Factored Verification: Detecting and Reducing Hallucinations in Frontier Models Using AI Supervision](https://elicit.com/blog/factored-verification-detecting-and-reducing-hallucinations-in-frontier-models-using-ai-supervision/) [search-summary]. This is the most directly copyable evaluation mechanism found in this research: split output into atomic claims, verify each against its cited source, report the fraction unsupported.
- **Elicit publishes a public benchmark page** for literature-review performance — [Benchmarks for Scientific Literature Review Performance](https://elicit.com/review/e5f458be-e904-40f5-ad7b-03651429b788) [search-summary]
- **Google (vendor-reported, April 2026):** Deep Research Max shows "a leap in performance across industry-standard benchmarks tracking retrieval and reasoning capabilities," illustrated with "Win-rates of Deep Research 4/26 vs. Deep Research 12/25 on an internal Deep Research expert eval." **No numeric figures and no named external benchmark are disclosed in the post** — [Deep Research Max](https://blog.google/innovation-and-ai/models-and-research/gemini-models/next-generation-gemini-deep-research/) [fetched]
- **Anthropic:** no hallucination or citation-accuracy rate is published on the Citations, Search results, or Web search pages. The only accuracy-adjacent claim is structural — "citations are guaranteed to contain valid pointers to the provided documents" — and a qualitative "Better citation reliability" heading — [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) [fetched]
- **Perplexity:** no factuality benchmark appears on the models page; the page covers capability positioning only — [Models](https://docs.perplexity.ai/getting-started/models) [fetched]. Perplexity's Deep Research launch post (which reportedly carries SimpleQA/HLE numbers) returned 403 and could not be verified.
- **Consensus:** no measured accuracy metrics are published on the citation-grounding page — it documents features, not numbers — [Citation Grounding](https://consensus.app/home/workshops/citation-grounding/) [fetched]

### Inferences
- The industry measures the wrong thing for this use case. PersonQA and SimpleQA measure *answer* correctness; the question "does the cited source support this sentence" is measured publicly by nobody in this set except Elicit's Factored Verification line of work.
- OpenAI's staleness caveat generalises: any fixed-answer benchmark will penalise a live-retrieval agent for being *more* current than the test set. An engineer building an eval for a research agent should timestamp gold answers or use support-checking rather than answer-matching.

### Gaps
- No verified numeric hallucination rate for OpenAI deep research — the system card's actual percentage is behind the 403'd PDF and the HTML hub page did not surface it. **Do not report a figure for OpenAI without re-verification.**
- Perplexity's self-reported SimpleQA/Humanity's Last Exam figures could not be retrieved from a first-party page (403).
- No vendor publishes an independent third-party audit of citation support. All numbers above are vendor-reported.

---

## Q5. User mechanisms to constrain or override which sources are used

### Takeaway
Three tiers exist: **API-level** (domain allow/deny, corpus mode, date filters, max search count), **admin/org-level** (Anthropic Console domain restriction; Managed Agents toolset lists), and **product-level source selection** (Gemini Deep Research source pickers; Perplexity focus modes). Anthropic and Google are the only vendors documenting an *administrator* tier distinct from the developer tier.

### Cited Findings
- **Anthropic, per-request:** `max_uses` "limits the number of searches performed. If Claude attempts more searches than allowed, the `web_search_tool_result` is an error with the `max_uses_exceeded` error code." Guidance: "Simple factual queries typically use 1–3 searches; comparative or multientity research can use 10 or more." Also noted: search triggering "is steerable through your system prompt... **For a hard constraint, use `max_uses`**" — an explicit statement that prompt steering is soft and the parameter is the hard lever — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]
- **Anthropic, org-level:** "Web search is enabled for your organization unless an administrator has disabled it in the Claude Console, where they can also restrict which domains it searches." Disabled → request fails with a 400 `invalid_request_error`, "rather than an error code inside a search result" — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]
- **Anthropic, Managed Agents:** domain lists are set "on the `web_search` entry of the agent toolset"; these sessions "use only the per-tool `allowed_domains` and `blocked_domains` lists," not the Console settings — [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) [fetched]
- **Anthropic, RAG override:** the `search_result` content block lets a caller substitute its own corpus entirely while keeping the same citation machinery — "Claude cite[s] your own content the same way it cites web search results" — [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results) [fetched]
- **OpenAI:** `filters.allowed_domains` / `blocked_domains` (100 each) constrain web search; for deep research, sources can be redirected to private corpora via MCP servers and vector stores — "supports connecting up to two vector stores simultaneously and allows **disabling web search** when accessing sensitive MCP data" — [Web search guide](https://developers.openai.com/api/docs/guides/tools-web-search) [fetched]; [Deep research guide](https://developers.openai.com/api/docs/guides/deep-research) [fetched]
- **OpenAI, trust guidance for custom sources:** "Only connect **trusted MCP servers** (servers you operate or have audited)"; run public research first then isolated calls with private data; "Apply **schema or regex validation** to tool arguments so the model cannot smuggle arbitrary payloads" — [Deep research guide](https://developers.openai.com/api/docs/guides/deep-research) [fetched]
- **Perplexity:** `search_domain_filter` (20 max, allow or deny), `search_mode` (`web`/`academic`/`sec`), the four date filters plus `search_recency_filter`, and `search_language_filter` (10 max) — all caller-controlled per request — [Search filters guide](https://docs.perplexity.ai/guides/academic-filter-guide) [fetched]; [Chat Completions API reference](https://docs.perplexity.ai/api-reference/chat-completions-post) [fetched]
- **Google, consumer:** the user can "change or add other sources, like your personal Gmail or Drive," upload files, and add NotebookLM notebooks; Deep Research Max allows "any subset of" web, remote MCPs, file uploads and connected file stores — [Use Deep Research in Gemini Apps](https://support.google.com/gemini/answer/15719111) [search-summary]; [Deep Research Max](https://blog.google/innovation-and-ai/models-and-research/gemini-models/next-generation-gemini-deep-research/) [fetched]
- **Google, plan approval as a source-control mechanism:** Deep Research "creates a multi-step research plan for you to either revise or approve" before executing — the user constrains sources indirectly by editing the plan — [Try Deep Research](https://blog.google/products/gemini/google-gemini-deep-research/) [search-summary] (Dec 2024 — older feature, still current in the product)
- **Anthropic Research (claude.ai):** enabled via the "+" button then "Research"; "Web search must be turned on for research to function" — i.e. the user's web-search toggle is the source gate — [Use research on Claude](https://support.claude.com/en/articles/11088861-use-research-on-claude) [search-summary]

### Inferences
- The plan-approval step (Google) and the `max_uses` hard cap (Anthropic) are two different answers to "let the user bound the search" — one semantic, one numeric. Both are cheap to copy.
- For an engineer, the strongest available override pattern is Anthropic's `search_result` blocks + OpenAI's MCP/vector-store routing: replace the corpus entirely while retaining the vendor's citation formatting, so the trust surface is the caller's own retrieval, not the vendor's index.

### Gaps
- No documented mechanism in any product for the *end user of a report* (as opposed to the developer) to retroactively exclude a source and regenerate.
- Elicit and Consensus source-restriction controls (filters by study type, sample size, year) are visible in their products but I could not confirm them from a fetched first-party page — the Consensus help page 403'd.
