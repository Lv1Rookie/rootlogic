# Attribution & Factuality Verification Methods for Generated Claims

Scope note: this file covers measurable techniques an engineer could implement to check (a) whether a generated claim is actually supported by the source it cites, and (b) how factual a long-form generated report is overall. Dates are flagged throughout; anything published before 2024 is marked **[pre-2024]** as potentially superseded.

---

## Q1. Attribution evaluation: AIS, ALCE, AttrScore, and citation precision/recall

### Takeaway
The field converged on one definition — a claim is "attributable" if a reader would accept "According to [source], [claim]" — and on one operationalization: run an NLI/entailment model over (cited passage(s), claim) pairs and aggregate into citation recall (are sentences supported?) and citation precision (is each citation doing work?). The foundational papers (AIS 2021/2023, ALCE 2023, AttrScore 2023) are all pre-2024, but their metric definitions are still the de facto standard; what has been superseded is the *entailment model* used inside them, not the metric.

### Cited Findings

**AIS (Attributable to Identified Sources)** — the conceptual foundation
- AIS is "an evaluation framework for assessing whether the output of natural language models only contains information about the external world that is verifiable in source documents" — [Google Research AIS repo](https://github.com/google-research-datasets/AIS)
- The core criterion is the intuitive test: a response is attributable to a source if "According to that source, the statement" holds — [search summary of Rashkin et al.](https://arxiv.org/abs/2112.12870)
- AIS specifies a **two-stage annotation pipeline**: stage 1 checks interpretability/self-containedness of the statement (is it even a decidable proposition, once decontextualized?); stage 2 checks attribution against the source. This two-stage structure is why modern pipelines decontextualize claims before verifying them. — [Rashkin et al., arXiv:2112.12870](https://arxiv.org/abs/2112.12870)
- AIS ships human annotations over four source datasets: CNN/DM, QReCC, Wizard of Wikipedia, ToTTo, instantiated for conversational QA, summarization, and table-to-text. — [AIS repo](https://github.com/google-research-datasets/AIS)
- Publication: Hannah Rashkin et al., arXiv:2112.12870 (Dec 2021), published in *Computational Linguistics* 49(4), 2023. **[pre-2024]** — [MIT Press](https://direct.mit.edu/coli/article/49/4/777/116438/Measuring-Attribution-in-Natural-Language); [ACL Anthology](https://aclanthology.org/2023.cl-4.2/)

**ALCE** — the benchmark that made citation precision/recall computable
- ALCE is "the first benchmark for Automatic LLMs' Citation Evaluation," providing questions, retrieval corpora, and automatic metrics across three dimensions: **fluency, correctness, and citation quality**. — [Gao et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.398/)
- **Citation recall** = the fraction of statements (sentences) that are entailed by their cited passages. Computed per-statement, binary: a statement scores 1 iff it carries ≥1 citation **and** the *concatenation* of all its cited passages entails it. — [ALCE metric summary](https://www.emergentmind.com/papers/2305.14627)
- **Citation precision** = the fraction of individual citations that actually help support the sentence. Computed per-citation, binary: a citation is scored irrelevant (0) if it alone cannot support the statement **and** removing it does not break entailment from the remaining cited passages. This second clause is what stops models from spraying citations. — [ALCE metric summary](https://www.emergentmind.com/papers/2305.14627)
- The entailment judgment is made by an **NLI model — TRUE (T5-11B fine-tuned on NLI data)** — not by an LLM judge. — [ALCE metric summary](https://www.emergentmind.com/papers/2305.14627); [ALCE PDF](https://arxiv.org/pdf/2305.14627)
- Datasets: three QA settings — ASQA (ambiguous factoid QA), QAMPARI (multi-answer list QA), and ELI5 (long-form QA). (Note: one fetch of the PDF returned "QAMPARI / LFQA / BIOGRAPHY," which conflicts with the canonical ASQA/QAMPARI/ELI5 triple reported elsewhere; treat the extraction as unreliable and verify against the paper.) — [ALCE PDF](https://arxiv.org/pdf/2305.14627)
- Headline result: "On the ELI5 dataset, even the best models lack complete citation support 50% of the time." — [Gao et al., EMNLP 2023 abstract](https://aclanthology.org/2023.emnlp-main.398/)
- The authors state the automatic metrics "demonstrate strong correlation with human judgements," but the specific correlation coefficient was not recoverable from the abstract/landing page. — [Gao et al.](https://aclanthology.org/2023.emnlp-main.398/)
- Publication: arXiv:2305.14627 (May 2023), EMNLP 2023. **[pre-2024]**

**AttrScore** — attribution evaluation as a 3-way classification task
- AttrScore frames attribution checking not as binary but as three labels: **attributable / extrapolatory / contradictory**. The fine-grained categorization "aids humans in better understanding the type of an attribution error." — [Yue et al., arXiv:2305.06311](https://arxiv.org/html/2305.06311v2)
- Two implementation routes are compared: (1) prompting LLMs zero/few-shot, and (2) fine-tuning smaller LMs on **simulated and repurposed data from QA, fact-checking, NLI, and summarization**. — [Yue et al.](https://arxiv.org/html/2305.06311v2)
- Finding relevant to model choice: "Fine-tuning on NLI-related data is beneficial to attribution evaluation," with **T5-XXL-TRUE** and **AttrScore-Flan-T5 (3B)** performing strongly on both in-distribution and out-of-distribution sets. A 3B fine-tuned model is competitive with prompted frontier LLMs on this task. — [Yue et al.](https://arxiv.org/html/2305.06311v2)
- Documented **failure modes** of automatic attribution evaluators: (a) insensitivity to fine-grained information comparison — overlooking contextual cues in the reference; (b) disregard for **numerical values**; (c) failure at **symbolic operations** (e.g. arithmetic or date reasoning implied by the claim). — [Yue et al.](https://arxiv.org/html/2305.06311v2)
- Dataset is public on HuggingFace as `osunlp/AttrScore`. — [HF dataset](https://huggingface.co/datasets/osunlp/AttrScore)
- Publication: arXiv:2305.06311 (May 2023), EMNLP 2023 Findings. **[pre-2024]**

**AttributionBench** — the "how hard is this actually" follow-up
- AttributionBench unifies multiple attribution-evaluation datasets into a single benchmark and asks how hard automatic attribution evaluation is; it is the more recent (2024) reference point for expected accuracy ceilings. — [OSU NLP Group](https://osu-nlp-group.github.io/AttributionBench/); [ACL Findings 2024](https://aclanthology.org/2024.findings-acl.886.pdf)
- Publication: arXiv:2402.15089 (Feb 2024), ACL 2024 Findings.

### Inferences
- For an implementer, the ALCE formulation is directly portable and cheap: it needs only a sentence splitter, a citation parser, and an entailment model. The per-citation precision rule (drop the citation, re-test entailment) is a leave-one-out ablation and is the part most implementations get wrong by omitting it.
- Because citation recall concatenates *all* cited passages before testing entailment, a system can score recall=1 while individual citations are junk. Precision and recall must be reported together; either alone is gameable.
- AttrScore's three-way labeling is more useful than binary supported/unsupported for a research agent, because "extrapolatory" (source is topically right but doesn't say this) and "contradictory" (source says the opposite) demand different remediation.

### Gaps
- Exact numeric human-agreement / correlation figures for ALCE's automatic citation metrics vs. human annotators could not be extracted from the abstract or landing page; they are in the paper body. Not reported here.
- Specific AttributionBench accuracy numbers (fine-tuned vs. GPT-4) were not retrieved in this session.
- ALCE's dataset list has a conflict between two extractions (see above) — unresolved.

---

## Q2. Long-form factuality: FActScore, SAFE/LongFact, VeriScore

### Takeaway
All three share the same three-stage skeleton — **decompose into atomic claims → retrieve evidence per claim → adjudicate each claim** — and differ mainly in (a) where evidence comes from (fixed Wikipedia corpus vs. live Google Search) and (b) whether every claim is assumed verifiable. Reported human agreement ranges from ~72% (SAFE, crowdworkers, 16k facts) to sub-2% error vs. human FActScore estimates, but these numbers are not comparable — they measure different things.

### Cited Findings

**FActScore** (Min et al., arXiv:2305.14251, EMNLP 2023) **[pre-2024]**
- Method: break a generation into **atomic facts** — short statements each conveying one piece of information — then compute the **percentage of atomic facts supported by a reliable knowledge source** (Wikipedia). — [Min et al., arXiv:2305.14251](https://arxiv.org/abs/2305.14251)
- Human annotation protocol: each atomic fact gets one of three labels — **"Irrelevant"** (not related to the prompt), **"Supported"**, **"Not-supported"**. The Irrelevant bucket matters: it is the escape hatch that later metrics (VeriScore) expand on. — [search summary of FActScore](https://aclanthology.org/2023.emnlp-main.741.pdf)
- **Inter-annotator agreement: 96% (InstructGPT), 90% (ChatGPT), 88% (PerplexityAI)** on the biography task. — [FActScore](https://aclanthology.org/2023.emnlp-main.741.pdf)
- Annotators were **recruited freelance fact-checkers** who passed a 2-hour qualification test. This is the cost that automation is trying to remove. — [FActScore](https://aclanthology.org/2023.emnlp-main.741.pdf)
- **The automated estimator (retrieval + a strong LM) achieves <2% error rate** relative to the human-computed FActScore. Note this is error on the *aggregate score*, not per-claim accuracy — aggregate errors can cancel. — [Min et al.](https://arxiv.org/abs/2305.14251)
- Known limitation: FActScore is **precision-only**. A model that says almost nothing scores perfectly. — inferred from the metric definition; this is exactly the gap SAFE's F1@K closes.
- Published by Meta AI + UW. — [AI at Meta publication page](https://ai.meta.com/research/publications/factscore-fine-grained-atomic-evaluation-of-factual-precision-in-long-form-text-generation/)

**SAFE / LongFact** (Wei et al., Google DeepMind, arXiv:2403.18802, **NeurIPS 2024**)
- **LongFact**: a prompt set of thousands of questions spanning **38 topics**, generated with GPT-4, for benchmarking open-domain long-form factuality. — [arXiv:2403.18802](https://arxiv.org/abs/2403.18802)
- **SAFE** pipeline: an LLM (a) splits the response into individual facts, (b) **revises each fact to be self-contained** (decontextualization), (c) for each fact runs a **multi-step reasoning loop issuing Google Search queries**, and (d) decides supported / not-supported / irrelevant from the search results. The iterative search loop is the key difference from FActScore's single fixed-corpus retrieval. — [arXiv:2403.18802](https://arxiv.org/abs/2403.18802); [DeepMind repo](https://github.com/google-deepmind/long-form-factuality)
- **Agreement with crowdsourced human annotators: 72%**, measured on ~**16,000 individual facts**. — [arXiv:2403.18802](https://arxiv.org/abs/2403.18802)
- On a random subset of **100 disagreement cases**, SAFE was judged correct **76%** of the time vs. the human annotator's **19%**. This is the basis for the "superhuman" framing. — [arXiv:2403.18802](https://arxiv.org/abs/2403.18802); [HuggingFace paper page](https://huggingface.co/papers/2403.18802)
- **Cost: >20× cheaper than human annotators.** — [arXiv:2403.18802](https://arxiv.org/abs/2403.18802)
- **F1@K**: extends F1 by treating **precision** = % of supported facts in the response, and **recall** = number of provided facts relative to **K**, a hyperparameter standing for the user's preferred response length. This explicitly makes long-form factuality length-aware and closes FActScore's precision-only gap. The exact formula was not recoverable from the abstract. — [arXiv:2403.18802](https://arxiv.org/abs/2403.18802)
- **Credibility caveat on "superhuman":** Gary Marcus publicly argued "superhuman" here may only mean "better than an underpaid crowd worker," not better than an expert fact-checker. The comparison baseline was crowdsourced annotators, not trained fact-checkers (contrast FActScore, which used qualified freelance fact-checkers). — [VentureBeat](https://venturebeat.com/ai/google-deepmind-unveils-superhuman-ai-system-that-excels-in-fact-checking-saving-costs-and-improving-accuracy)
- Code is open source at [google-deepmind/long-form-factuality](https://github.com/google-deepmind/long-form-factuality), with SAFE under `eval/safe`.

**VeriScore** (Song, Kim, Iyyer — UMass Amherst; arXiv:2406.19276, June 2024, **Findings of EMNLP 2024**)
- Core critique of FActScore and SAFE: both "assume that every claim is verifiable (i.e., can plausibly be proven true or false)." Real long-form generation is a mix of verifiable and unverifiable content (opinions, hedges, instructions, subjective statements). — [arXiv:2406.19276](https://arxiv.org/abs/2406.19276)
- Pipeline: proceeds **sentence-by-sentence**; for each sentence a **claim extraction model extracts only the verifiable claims (with surrounding context supplied)**; each claim becomes a **Google Search query**; a **claim verification model** adjudicates the claim against retrieved evidence. — [VeriScore summary](https://arxiv.org/abs/2406.19276)
- Works with **either closed or fine-tuned open-weight models** for the extractor and verifier — i.e., the pipeline is deliberately implementable without frontier-model API spend. — [arXiv:2406.19276](https://arxiv.org/abs/2406.19276)
- **Human evaluation confirms VeriScore's extracted claims are "more sensible" than those from competing methods across eight different long-form tasks.** Note: this validates the *decomposition* step, not the adjudication step. — [arXiv:2406.19276](https://arxiv.org/abs/2406.19276)
- Evaluated **16 different models** including GPT-4o and Mixtral-8x22. — [arXiv:2406.19276](https://arxiv.org/abs/2406.19276)

**VeriFastScore** (arXiv:2505.16973, May 2025) — the latency fix
- Motivation stated explicitly: VeriScore "can take upwards of **100 seconds to evaluate a single response**." VeriFastScore distills extraction+verification into a single fine-tuned model pass. — [arXiv:2505.16973](https://arxiv.org/abs/2505.16973)
- This is the most practically relevant recent work for anyone putting atomic-claim verification in a production loop.

### Inferences
- The three-stage skeleton (decompose → retrieve → adjudicate) is the implementable core. An engineer should treat decomposition, retrieval, and adjudication as three separately-swappable and separately-measurable components, because the published agreement numbers attach to different stages (FActScore validated adjudication; VeriScore validated decomposition; SAFE validated end-to-end).
- The 72% SAFE / ~90%+ FActScore IAA gap is mostly explained by annotator quality and by open-domain vs. Wikipedia-bounded verification. Do not expect 90%+ agreement on open-web verification of arbitrary report claims.
- SAFE's disagreement-adjudication result (76% vs 19%) is a genuinely useful methodology to copy: when your automated verifier disagrees with a human, re-adjudicate the disagreements rather than assuming the human is right.
- VeriScore's verifiable/unverifiable split is the single most important design decision for scoring a *research report*, which is full of hedged synthesis sentences that are not propositions. A FActScore-style metric applied to a report will produce noise on those sentences.

### Gaps
- The exact F1@K formula was not extracted; it is in the SAFE paper body.
- VeriScore's specific human-agreement percentage and the reported fraction of claims that are unverifiable were not recoverable from the abstract. Not reported here.
- VeriFastScore's reported correlation with VeriScore and its speedup factor were not retrieved.

---

## Q3. NLI-based entailment checking as a cheaper alternative to an LLM judge

### Takeaway
This is the strongest cost/quality result in the whole area: a **~770M-parameter Flan-T5 model (MiniCheck) reaches 74.7% balanced accuracy on LLM-AggreFact, statistically level with Claude-3 Opus (74.1%) and near GPT-4 (75.3%), at ~400× lower cost** ($0.24 vs. $107 for the same 13K-example test set). For claim-vs-passage verification specifically, an LLM judge is not buying much.

### Cited Findings
- **MiniCheck-Flan-T5: 74.7% balanced accuracy on LLM-AggreFact**, vs. **GPT-4: 75.3%**, **Claude-3 Opus: 74.1%**, and the previous best specialized model **AlignScore: 70.4%** — a 4.3-point improvement over AlignScore. — [Tang et al., arXiv:2404.10774](https://arxiv.org/abs/2404.10774); [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.499/)
- **Cost: $0.24 vs. $107** for inference on the 13K-example test set — **400× cheaper** than GPT-4 at GPT-4-level accuracy. — [MiniCheck](https://arxiv.org/abs/2404.10774)
- Training method: **synthetic data generated by GPT-4** via a structured procedure that produces "realistic yet challenging instances of factual errors," teaching the model to check each fact in the claim and to **recognize synthesis of information across sentences** (multi-sentence/multi-hop grounding, which sentence-level NLI models fail at). — [MiniCheck](https://arxiv.org/abs/2404.10774)
- **LLM-AggreFact** is the benchmark to use for model selection: it unifies **10 existing datasets** across news, dialogue, and Wikipedia, covering summarization, RAG, and closed-book QA. — [MiniCheck](https://arxiv.org/abs/2404.10774)
- Publication: arXiv:2404.10774 (April 2024), **EMNLP 2024**. Code: [Liyan06/MiniCheck](https://github.com/Liyan06/MiniCheck).
- The predecessor models still in wide use: **AlignScore** (70.4% on LLM-AggreFact) and **TRUE / T5-11B-TRUE**, the latter being the model ALCE uses for citation precision/recall and one of the strongest in the AttrScore evaluation. — [MiniCheck](https://arxiv.org/abs/2404.10774); [AttrScore](https://arxiv.org/html/2305.06311v2)
- Known failure mode (from AttrScore, generalizes to NLI verifiers): insensitivity to numerical values and inability to perform symbolic operations implied by the claim. An NLI model will often mark "revenue grew 40%" as entailed by a passage stating different numbers. — [Yue et al.](https://arxiv.org/html/2305.06311v2)
- **FactCG** (arXiv:2501.17144, Jan 2025) is a more recent graph-based multi-hop entrant in the same family, indicating the specialized-checker line is still active. — [arXiv:2501.17144](https://arxiv.org/pdf/2501.17144)

### Inferences
- Recommended default stack for an implementer: use MiniCheck (or its successors) as the per-claim entailment verifier, and reserve an LLM judge for the *decomposition* step and for adjudicating low-confidence or contradictory cases. This inverts the common instinct of using the LLM for everything.
- Because balanced accuracy tops out around 75% even for GPT-4, per-claim verification is roughly a 1-in-4-wrong process. Aggregate scores over many claims are far more trustworthy than any single claim verdict — a verification layer should be used to flag and rank, not to auto-reject.
- Numerical and multi-hop claims are the known weak spot; a practical system should route claims containing numbers, dates, or comparisons to a stricter check (e.g. a required verbatim quote containing the figure).

### Gaps
- No source found giving accuracy specifically for verifying a claim against a *live web retrieval* (as opposed to a provided grounding document). LLM-AggreFact assumes the grounding document is given, so these numbers are an upper bound for a retrieval-augmented agent where retrieval itself can fail.

---

## Q4. Verbatim-quote requirements: does demanding an exact supporting quote reduce unsupported claims?

### Takeaway
The best-measured evidence is **GopherCite (DeepMind, 2022)**, which required models to emit verbatim quotes and reported **80% / 67% human-rated high-quality-and-supported** on NaturalQuestions/ELI5, rising to **90% / 80% when allowed to abstain**. This is the clearest measured effect, but it is pre-2024 and conflates the quote requirement with RLHP training and reranking. Recent claims of specific percentage reductions from citation enforcement come from **vendor blogs without published methodology and should not be cited as evidence.**

### Cited Findings
- **GopherCite** (Menick, Trebacz, Mikulik, Aslanides, Song, Chadwick, Glaese, Young, Campbell-Gillingham, Irving, McAleese; arXiv:2203.11147, **21 March 2022** **[pre-2024]**): the model generates answers **with verbatim quoted evidence** drawn from documents found via a search engine or from a single user-provided document, trained with **reinforcement learning from human preferences (RLHP)** plus reranking. — [arXiv:2203.11147](https://arxiv.org/abs/2203.11147)
- **Measured effect: human raters judged responses high-quality and supported 80% of the time on NaturalQuestionsFiltered and 67% on ELI5.** — [arXiv:2203.11147](https://arxiv.org/abs/2203.11147)
- **With abstention** (declining to answer when uncertain), those rise to **90% and 80%** respectively. The abstention mechanism is the larger marginal gain in the reported numbers. — [arXiv:2203.11147](https://arxiv.org/abs/2203.11147)
- **Explicitly stated limitation, and the key caveat for anyone building on this:** "not all claims supported by evidence are true" — citation/quoting verifies *attribution*, not *truth*. A model quoting a wrong source verbatim is perfectly attributable and perfectly false. — [arXiv:2203.11147](https://arxiv.org/abs/2203.11147)
- **Cite Before You Speak** (arXiv:2503.04830, March 2025) studies enforced citation in e-commerce conversational LLM agents; a citation-generation paradigm is reported to improve grounding performance by **13.83%**. Flagging that this figure came via a search-result summary and the baseline/metric definition was not verified from the paper itself. — [arXiv:2503.04830](https://arxiv.org/abs/2503.04830)
- **Low-confidence / unusable claims flagged:** a vendor blog asserts "citation-required prompting reduces unsupported claims by 50-80% on the same retrieval stack in customer-support chatbots" ([zeroentropy.dev](https://zeroentropy.dev/concepts/grounded-generation/)), and another asserts "citation enforcement reduces fabricated details to under 3%" ([hidekazu-konishi.com](https://hidekazu-konishi.com/entry/llm_output_verification_patterns.html)). **Neither publishes a methodology, dataset, or baseline. Do not treat these as measured effect sizes.**

### Inferences
- The verbatim-quote requirement has a structural, not just statistical, benefit that is independent of measured effect size: an exact-string quote is **mechanically checkable** with `substring in retrieved_document` — no model, no cost, no error rate. This converts a fuzzy verification problem into a deterministic one for the quote itself, leaving only "does this quote support this claim?" to a model. That is a strictly smaller and easier problem than "is this claim supported by this document?"
- GopherCite's abstention result suggests the biggest wins come from letting the system decline rather than from the quoting itself. For a research agent, an explicit "no supporting quote found → mark claim unsupported and surface it" path is likely higher-value than tuning the quote extraction.
- The GopherCite caveat is the single most important framing point for the whole survey: attribution ≠ truth. Every metric in Q1 and Q3 measures attribution. Only FActScore/SAFE/VeriScore-style pipelines against a trusted corpus or live search approach truth, and only to the extent the corpus is right.

### Gaps
- **No post-2024 peer-reviewed study was found that isolates the effect of requiring a verbatim quote vs. requiring only a citation ID, holding retrieval and model constant.** This is a real gap in the literature and the cleanest ablation an engineer could run themselves.
- GopherCite's numbers do not decompose the contribution of the quote requirement from RLHP and reranking.

---

## Q5. LLM-as-judge reliability: biases and the different-model recommendation

### Takeaway
GPT-4-class judges reach **>80% agreement with human preferences — the same level as human-human agreement** (MT-Bench) — but the named biases (position, verbosity, self-preference) are reproducible and documented. The mechanism behind self-preference appears to be **perplexity**: judges over-reward low-perplexity (familiar-looking) text regardless of who wrote it, which is a cleaner explanation than "models like themselves" and predicts why using a different judge model helps.

### Cited Findings
- **MT-Bench / Chatbot Arena** (Zheng et al., arXiv:2306.05685, **NeurIPS 2023 Datasets & Benchmarks**) **[pre-2024]**: "strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, **achieving over 80% agreement, the same level of agreement between humans**." Validated on **3K expert votes** and **30K conversations**. — [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- MT-Bench names and studies the canonical bias set: **position bias, verbosity bias, self-enhancement bias**, plus **limited reasoning ability** as a judge constraint, and proposes mitigations. — [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- **Self-Preference Bias in LLM-as-a-Judge** (arXiv:2410.21819, Oct 2024; rev. June 2025; **NeurIPS 2024 Safe Generative AI Workshop**): introduces a quantitative metric for self-preference and finds **GPT-4 exhibits a significant degree of self-preference bias**. — [arXiv:2410.21819](https://arxiv.org/abs/2410.21819)
- **Mechanism finding (important):** "LLMs assign significantly higher evaluations to outputs with **lower perplexity** than human evaluators, **regardless of whether the outputs were self-generated**." Self-preference is a side effect of familiarity/fluency preference, not identity recognition. — [arXiv:2410.21819](https://arxiv.org/abs/2410.21819)
- **LLMs-as-Judges: A Comprehensive Survey** (arXiv:2412.05579, Dec 2024) is the current survey reference for evaluation methods and bias taxonomy. — [arXiv:2412.05579](https://arxiv.org/pdf/2412.05579)
- More recent primary work on bias mitigation: **Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines** (arXiv:2604.23178) and **Who Judges Matters: Measuring Family-Conditioned Preference in LLM-as-Judge Panels** (arXiv:2609.17857) — the latter is directly on point for the "use a different model family to judge" recommendation. — [arXiv:2604.23178](https://arxiv.org/pdf/2604.23178); [arXiv:2609.17857](https://arxiv.org/pdf/2609.17857)
- **Low-confidence / vendor-blog figures, flagged as such:** a 2026 vendor blog states position bias is "10 to 15 points of winrate swing depending on slot order," verbosity bias "15 to 30 points of inflated preference for longer outputs across GPT-4, Claude, and PaLM-2 judges," and self-preference "confirmed at 10 to 25 percent." ([futureagi.com](https://futureagi.com/blog/evaluating-llm-judge-bias-mitigation-2026/)). These are plausible-magnitude but **not traceable to a specific paper from the source given**; treat as indicative only.

### Inferences
- The perplexity mechanism gives a concrete, testable reason for the different-judge-than-generator rule: if a judge over-rewards text that is low-perplexity *under its own distribution*, then judging your own generator's output systematically inflates scores. Using a different model family decorrelates the generator's and judge's perplexity landscapes.
- For *attribution checking specifically* (Q1/Q3), the position/verbosity biases matter much less than in pairwise preference judging, because attribution checking is a **single-item classification** (does passage P entail claim C?), not a pairwise comparison. Position bias needs two candidates to exist. This is another argument for using a specialized NLI checker: it sidesteps the entire bias taxonomy, which is largely a pairwise-preference phenomenon.
- Where an LLM judge *is* needed in this pipeline (claim decomposition, adjudicating hard cases), the residual risk is verbosity/fluency preference marking well-written unsupported claims as supported — exactly the failure mode a research agent produces.

### Gaps
- **The explicit, citable recommendation "use a different model for judging than for generating" was not located in a primary peer-reviewed source during this session.** It is widely repeated, and the perplexity-mechanism paper (2410.21819) supports it by implication, but a direct quotable recommendation with a measured effect size was not found. arXiv:2609.17857 (family-conditioned preference in judge panels) is the most likely place to find it and should be checked.
- MT-Bench's specific numeric magnitudes for position/verbosity/self-enhancement bias are in the paper body and were not extracted. Not reported here.

---

## Q6. Corroboration across independent sources, and the source-copying failure mode

### Takeaway
There is a substantial and directly relevant pre-LLM database/data-integration literature on this — **truth discovery** — and it identified the copying failure mode explicitly, along with the key detection heuristic: **sources that share *errors* are likely dependent, because agreeing on a truth is uninformative but agreeing on a falsehood is not**. The literature also names the case where this heuristic breaks. I found **no LLM-era paper** applying this to research-agent citation corroboration.

### Cited Findings
- **Truth discovery** is the established name for the problem: "detecting true values from conflicting data provided by multiple sources on the same data items." — [A Survey on Truth Discovery, arXiv:1505.02463](https://arxiv.org/pdf/1505.02463) **[pre-2024, but this is a mature, stable literature]**
- Naive corroboration assumes **source independence**; the literature explicitly "relaxed source independence assumptions by attempting to detect copying relationships among sources and adjust source weights accordingly." — [Truth Discovery survey](https://arxiv.org/pdf/1505.02463)
- **The core copy-detection principle: "if sources make many common mistakes, they are likely not independent of each other."** Shared *errors* are the signal, not shared agreement. — [Truth Discovery survey](https://arxiv.org/pdf/1505.02463)
- **The documented failure mode of copy detection itself: the principle "becomes ineffective when sources copy information from a good source."** If ten outlets all syndicate an accurate wire story, they make no common mistakes and are indistinguishable from ten independent confirmations. This is exactly the press-release/syndication case. — [Truth Discovery survey](https://arxiv.org/pdf/1505.02463)
- **Dong et al., "Truth Discovery and Copying Detection in a Dynamic World"** (VLDB 2009) uses **Hidden Markov Models over update histories** to detect evolving copying relationships — i.e., uses *timing* of updates as a dependence signal, which works even when the copied content is correct. This is the mechanism that survives the failure mode above. — [VLDB 2009 PDF](http://www.vldb.org/pvldb/vol2/vldb09-335.pdf) **[pre-2024]**
- **Bayesian formulations** infer copying relationships among sources with detection "tightly combined with truth discovery so that detected relationships and discovered truths are iteratively updated." — [A Bayesian Approach to Discovering Truth from Conflicting Sources, arXiv:1203.0058](https://arxiv.org/pdf/1203.0058) **[pre-2024]**
- A doctoral thesis dedicated to the problem: **"Corroborating Information from Multiple Sources," Minji Wu (Rutgers)**. — [Rutgers](https://rucore.libraries.rutgers.edu/rutgers-lib/51498/PDF/1/play/)
- **LLM-era adjacent work:** "Grading the Narrators: An Isnad-Rijal Framework for Claim-Level Provenance in Multi-Agent Knowledge Systems" (arXiv:2607.24117) applies chain-of-transmission provenance grading to multi-agent knowledge systems — the closest recent work to applying source-dependence reasoning to LLM agents, though I did not verify its results. — [arXiv:2607.24117](https://arxiv.org/pdf/2607.24117)

### Inferences
- The actionable engineering heuristic from this literature: **corroboration count is only a truth signal after deduplicating by origin.** Practical proxies an implementer can use — near-duplicate text detection (shingling/MinHash) across retrieved passages, shared publication timestamps clustered within hours, presence of wire-service or press-release boilerplate, and identical unusual phrasings or identical numbers with identical rounding. Sources passing these filters should be weighted as one source, not N.
- Timing-based dependence detection (Dong et al.'s HMM insight) is the only approach that survives the "everyone copied a correct source" case, and it is cheap to approximate: if N sources all published within a short window of one earlier source, treat them as one.
- Because this literature predates LLMs entirely, the AI-generated-content variant is unaddressed: many retrieved web pages may now be LLM summaries of each other, which produces correlated *errors* as well as correlated content — making the classic shared-error heuristic potentially more useful again, but also making the source pool systematically contaminated.

### Gaps
- **I found no paper measuring the effect of source-independence weighting on LLM research-agent factuality.** The truth-discovery literature and the LLM-attribution literature appear not to have been connected. This is a genuine open area, and any claim that multi-source corroboration improves a research agent's accuracy by a specific amount would currently be unsupported.
- No measured numbers found for how often web retrieval results are syndicated duplicates of one another in practice.
- The Isnad-Rijal paper (arXiv:2607.24117) was surfaced by search but not read; its claims are unverified here.

---

## Cross-cutting: framework metric definitions (RAGAS, DeepEval)

### Takeaway
The two most-used open-source frameworks implement essentially the same metric under the name "faithfulness" — claims-supported / total-claims — which is FActScore restricted to the retrieved context. Neither implements ALCE-style per-citation precision, which is the gap an engineer would need to fill themselves.

### Cited Findings
- **RAGAS Faithfulness** = (number of claims in the response supported by the retrieved context) / (total number of claims in the response), range 0–1. Computed by (1) identifying all claims in the response, (2) checking each against the retrieved context. — [RAGAS docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/); [source repo](https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/faithfulness.md)
- RAGAS definition of faithful: "A response is considered faithful if all its claims can be supported by the retrieved context." Note this is **groundedness in retrieval, not truth** — same caveat as GopherCite. — [RAGAS docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- **DeepEval Faithfulness** = Number of Truthful Claims / Total Number of Claims, measured against `retrieval_context`. It is a "self-explaining LLM-Eval" — it emits a reason alongside the score. — [DeepEval docs](https://deepeval.com/docs/metrics-faithfulness)
- **DeepEval Hallucination** is a *different* metric: the fraction of provided context documents that **contradict** the `actual_output` (0 is perfect), and it compares against a **ground-truth context** rather than retrieved context. — [DeepEval docs](https://deepeval.com/docs/metrics-hallucination)
- The practical distinction: **faithfulness** tests whether the generator respects whatever the retriever supplied (even if retrieval was wrong); **hallucination** tests against a reference truth. A research agent needs both, and they fail differently. — [DeepEval docs](https://deepeval.com/docs/metrics-hallucination)

### Inferences
- Both frameworks' "faithfulness" is claim-level recall-of-support with an LLM judge as the adjudicator. Swapping that adjudicator for MiniCheck would keep the metric definition and cut cost ~400× (Q3), at roughly equal accuracy.
- Neither framework, as documented, implements the ALCE leave-one-out **citation precision** check. For a report where every sentence carries citations, faithfulness alone will not catch citation spraying.

### Gaps
- LangSmith's and OpenAI Evals' specific groundedness metric definitions were not retrieved in this session — not reported.
- Anthropic's first-party evaluation documentation was not consulted in this session.

---

## Summary table for the report writer

| Method | Year / Venue | What it scores | Adjudicator | Reported reliability | Main failure mode |
|---|---|---|---|---|---|
| AIS | 2021 arXiv / 2023 *Comp. Ling.* **[pre-2024]** | Human attribution judgment, 2-stage | Humans | n/a (it's the gold standard def.) | Requires decontextualization; expensive |
| ALCE citation prec./recall | EMNLP 2023 **[pre-2024]** | Citation quality per sentence & per citation | NLI (TRUE / T5-11B) | "strong correlation with human"; exact number not reported | Recall gameable alone; concatenation hides bad citations |
| AttrScore | EMNLP 2023 Findings **[pre-2024]** | 3-way: attributable/extrapolatory/contradictory | Fine-tuned LM or prompted LLM | Fine-tuned 3B ≈ prompted LLMs | Numbers, symbolic ops, fine-grained context |
| FActScore | EMNLP 2023 **[pre-2024]** | % atomic facts supported by Wikipedia | Retrieval + LM | Human IAA 88–96%; automated estimator <2% error on aggregate score | Precision-only; short answers win |
| SAFE / LongFact | NeurIPS 2024 | Atomic facts vs. live Google Search; F1@K | LLM + search loop | 72% agreement over 16k facts; wins 76% of 100 disagreements; >20× cheaper | Crowdworker baseline; "superhuman" contested |
| VeriScore | EMNLP 2024 Findings | Verifiable claims only, vs. Google Search | Extractor + verifier models | Claims judged "more sensible" than competitors across 8 tasks; exact % not reported | ~100s per response |
| VeriFastScore | arXiv 2505.16973 (2025) | Same, single-pass distilled | Fine-tuned model | not retrieved | not retrieved |
| MiniCheck | EMNLP 2024 | Claim vs. grounding doc entailment | Flan-T5 (small) | **74.7% bal. acc. on LLM-AggreFact vs. GPT-4 75.3%, 400× cheaper** | Assumes grounding doc given; ~1-in-4 error |
| AlignScore | ACL 2023 **[pre-2024]** | Same | Specialized model | 70.4% on LLM-AggreFact | Superseded by MiniCheck |
| GopherCite | arXiv 2022 **[pre-2024]** | Verbatim-quote-supported answers | Human raters + RLHP | 80%/67% supported; 90%/80% with abstention | Attribution ≠ truth (stated by authors) |
| RAGAS / DeepEval faithfulness | ongoing docs | supported claims / total claims vs. retrieval | LLM judge | not reported | Groundedness in retrieval, not truth; no citation precision |
| Truth discovery / copy detection | VLDB 2009, surveys **[pre-2024]** | Source corroboration with dependence weighting | Bayesian / HMM | not applicable to LLM setting | Breaks when sources copy a *correct* source |
