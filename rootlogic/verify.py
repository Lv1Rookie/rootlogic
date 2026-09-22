"""Guardrails on research quality: claim verification, corroboration, and report checks.

Three layers, each explainable on its own:

1. ``verify_findings`` - every claim a sub-agent made is checked against the *text of the pages
   it cites* (captured when sub-agents fetched them, or fetched here through the search
   provider). The model judges support, but must quote the evidence verbatim, and **code**
   confirms the quote really appears in the page. A "supported" verdict whose quote can't be
   found is downgraded. Claims with no page text are "unverifiable", never silently trusted.
2. ``corroborate`` - pure code: how many independent sites back a claim, and whether all of
   them are low-credibility. Low-credibility sources can support a claim but never alone.
3. ``check_report`` - pure code on the written report: citations pointing at no source are
   replaced with [?], uncited factual-looking sentences and weakly sourced takeaways are
   listed, and the counts feed the report's "Confidence and limitations" section.

Verification can't prove a claim true, only that its cited source says it. The evaluation
harness (``rootlogic eval``) measures how well these layers work across many topics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import prompts
from .filters import host_of, normalize_url
from .llm import LLM, AgentRefusal, LLMError
from .models import (CheckedClaim, Finding, ReportDraft, ReportQuality, SearchHit, SourceDraft,
                     VerificationDraft)

CHUNK = 1500                # characters per evidence chunk
CHUNKS_PER_SOURCE = 3       # most relevant chunks shown per cited page
SOURCES_PER_CLAIM = 3
MIN_QUOTE = 12              # shorter "quotes" prove nothing
_MULTI_PART_SUFFIXES = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "org.au", "gov.au",
                        "co.jp", "co.nz", "com.br", "co.in", "gov.in", "co.za", "com.cn"}

Fetch = Callable[[str], str | None]
Emit = Callable[..., None]


# =================================================================== evidence


def evidence_from_hits(hits: list[SearchHit]) -> dict[str, str]:
    return {normalize_url(h.url): h.text for h in hits if h.text}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9%]+", text.lower()) if len(w) > 3 or w.isdigit()}


def relevant_excerpt(text: str, claim: str) -> str:
    """The chunks of a page that share the most words with the claim, in page order."""
    if len(text) <= CHUNK * CHUNKS_PER_SOURCE:
        return text
    chunks = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]
    target = _words(claim)
    ranked = sorted(range(len(chunks)), key=lambda i: -len(target & _words(chunks[i])))
    return "\n…\n".join(chunks[i] for i in sorted(ranked[:CHUNKS_PER_SOURCE]))


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[“”\"'‘’]", "", text)).strip().lower()


def quote_in(quote: str, texts: list[str]) -> bool:
    """Code-level check that the verifier's 'verbatim' quote really is in the evidence."""
    q = _squash(quote).rstrip(".…")
    return len(q) >= MIN_QUOTE and any(q in _squash(t) for t in texts)


# =================================================================== corroboration


def registrable_domain(host: str) -> str:
    """news.bbc.co.uk -> bbc.co.uk; www.nytimes.com -> nytimes.com (heuristic, no PSL)."""
    parts = host.removeprefix("www.").split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _MULTI_PART_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def corroborate(claim: CheckedClaim, sources: dict[str, SourceDraft]) -> None:
    """Label a claim by the independent, non-low-credibility sites that back it."""
    cited = [sources[k] for u in claim.source_urls if (k := normalize_url(u)) in sources]
    if not cited:
        claim.corroboration, claim.domains = "none", []
        return
    credible = [s for s in cited if s.credibility.level != "low"]
    claim.domains = sorted({registrable_domain(host_of(s.url)) for s in cited})
    if not credible:
        claim.corroboration = "weak"
        return
    independent = {registrable_domain(host_of(s.url)) for s in credible}
    claim.corroboration = "corroborated" if len(independent) >= 2 else "single_source"


# =================================================================== verification


@dataclass
class VerifyResult:
    counts: dict[str, int] = field(default_factory=dict)
    fetched: int = 0
    calls: int = 0


def verify_findings(findings: list[Finding], *, llm: LLM, evidence: dict[str, str],
                    fetch: Fetch | None = None, max_claims: int = 12, max_fetches: int = 6,
                    emit: Emit | None = None) -> VerifyResult:
    """Fill ``finding.checks`` for every finding that hasn't been checked yet (earlier
    sessions' findings arrive already checked). Mutates ``evidence`` with pages it fetches."""
    emit = emit or (lambda *a, **k: None)
    result = VerifyResult()
    sources = {normalize_url(s.url): s for f in findings for s in f.sources}
    pending = [f for f in findings if not f.checks]

    # Budget: round-robin across findings so every sub-task gets some of its claims checked.
    selected: set[tuple[str, int]] = set()
    depth = 0
    while len(selected) < max_claims and any(depth < len(f.claims) for f in pending):
        for f in pending:
            if depth < len(f.claims) and len(selected) < max_claims:
                selected.add((f.task_id, depth))
        depth += 1

    for f in pending:
        checks = [CheckedClaim(text=c.text, source_urls=c.source_urls) for c in f.claims]
        to_judge: list[tuple[int, CheckedClaim, dict[str, str]]] = []
        for i, check in enumerate(checks):
            if (f.task_id, i) not in selected:
                continue  # stays "unchecked"
            pages: dict[str, str] = {}
            for url in check.source_urls[:SOURCES_PER_CLAIM]:
                key = normalize_url(url)
                if key not in evidence and fetch and result.fetched < max_fetches:
                    result.fetched += 1
                    if text := fetch(url):
                        evidence[key] = text
                if key in evidence:
                    pages[url] = evidence[key]
            if pages:
                to_judge.append((i + 1, check, pages))
            else:
                check.verdict = "unverifiable"
                check.note = "No page text available for the cited sources."
        if to_judge:
            _judge(f, to_judge, llm, result, emit)
        for check in checks:
            corroborate(check, sources)
        f.checks = checks

    for f in findings:
        for check in f.checks:
            result.counts[check.verdict] = result.counts.get(check.verdict, 0) + 1
    return result


def _judge(f: Finding, items: list[tuple[int, CheckedClaim, dict[str, str]]], llm: LLM,
           result: VerifyResult, emit: Emit) -> None:
    blocks = [f"Sub-task question: {f.question}", ""]
    for n, check, pages in items:
        blocks.append(f"Claim {n}: {check.text}")
        for url, text in pages.items():
            blocks += [f"Evidence (source: {url}):", "<<<", relevant_excerpt(text, check.text), ">>>"]
        blocks.append("")
    result.calls += 1
    try:
        draft = llm.structured(purpose=f"verify:{f.task_id}", system=prompts.VERIFIER,
                               prompt="\n".join(blocks), schema=VerificationDraft, effort="medium")
    except (LLMError, AgentRefusal) as e:
        emit("verify.skipped", f"[{f.task_id}] Verification unavailable: {e}", task=f.task_id)
        return  # claims stay "unchecked": never marked supported without a check
    verdicts = {v.claim_number: v for v in draft.checks}
    for n, check, pages in items:
        v = verdicts.get(n)
        if v is None:
            continue
        found = quote_in(v.quote, list(pages.values()))
        check.note = v.note
        check.quote = v.quote if found else ""
        if v.verdict == "supported" and not found:
            check.verdict = "partially_supported"
            check.note = ("Downgraded: the supporting quote was not found in the source text. "
                          + v.note)
        else:
            check.verdict = v.verdict


# =================================================================== prompt labels


def claim_label(check: CheckedClaim) -> str:
    verdict = {"supported": "verified", "partially_supported": "partly verified",
               "unverifiable": "unverified", "unchecked": "unverified"}.get(check.verdict, "")
    corr = {"corroborated": f"{len(check.domains)} independent sites",
            "single_source": "single source",
            "weak": "only low-credibility sources"}.get(check.corroboration, "")
    return "; ".join(x for x in (verdict, corr) if x)


# =================================================================== report checks

_CITE = re.compile(r"\[(\d+)\]")
_FACTUAL = re.compile(r"\d|%|\b(percent|million|billion|increase|decrease|majority)\b", re.I)


def _sentences(markdown: str) -> list[str]:
    out = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "|", ">")):
            continue
        line = re.sub(r"^[-*]\s+|^\d+\.\s+", "", line)
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if s.strip()]
    return out


def check_report(draft: ReportDraft, sources: list[SourceDraft], checks: list[CheckedClaim],
                 policy_drops: int = 0) -> ReportQuality:
    """Deterministic checks. Mutates ``draft`` only to replace citations that point nowhere."""
    n = len(sources)
    invalid: list[str] = []

    def fix(text: str) -> str:
        def sub(m: re.Match) -> str:
            if 1 <= int(m.group(1)) <= n:
                return m.group(0)
            invalid.append(m.group(0))
            return "[?]"
        return _CITE.sub(sub, text)

    draft.key_takeaways = [fix(t) for t in draft.key_takeaways]
    draft.body_markdown = fix(draft.body_markdown)

    uncited = [s for s in _sentences(draft.body_markdown) + draft.key_takeaways
               if "[" not in s and _FACTUAL.search(s) and not s.endswith("?")]
    weak = []
    for t in draft.key_takeaways:
        refs = [int(x) for x in _CITE.findall(t) if 1 <= int(x) <= n]
        if refs and all(sources[r - 1].credibility.level == "low" for r in refs):
            weak.append(t)

    quality = ReportQuality(invalid_citations=sorted(set(invalid), key=invalid.index),
                            uncited_statements=uncited[:10], weak_takeaways=weak,
                            policy_drops=policy_drops)
    for c in checks:
        quality.claims[c.verdict] = quality.claims.get(c.verdict, 0) + 1
        quality.corroboration[c.corroboration] = quality.corroboration.get(c.corroboration, 0) + 1
    return quality
