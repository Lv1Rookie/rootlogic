"""Content moderation for models without built-in safety systems.

Claude runs Anthropic's safety classifiers server-side (with fallbacks), so a separate step
is optional there. An arbitrary OpenAI-compatible or local model may have none, so rootlogic
screens three things before they can do harm:

  * the **request**: the topic, before any research runs;
  * **user input** during a run: clarifying answers, notes, tasks added by hand;
  * the **report**: before it is saved or shown.

Two backends implement ``Moderator``:
  * ``OpenAIModerator``    OpenAI's moderation endpoint (``omni-moderation-latest``).
  * ``LlamaGuardModerator`` Llama Guard 3 on any OpenAI-compatible server, e.g. Ollama
                            (``ollama pull llama-guard3``), fully local.

Block vs warn: research legitimately *discusses* violence, crime, health or elections. So each
backend's categories split into ones that indicate content *enabling* harm (block) and ones that
indicate sensitive subject matter (warn: logged and noted in the report's limitations).
``strict=True`` blocks on every flag. The split below is this project's judgement call, not a
provider default; adjust ``BLOCK`` to your needs.

Failures are closed: if the moderation service can't be reached, the run stops rather than
continuing unscreened.
"""

from __future__ import annotations

import os
from typing import Literal, Protocol

import openai
from pydantic import BaseModel

from .llm import LLMError

Stage = Literal["request", "input", "report"]
CHUNK = 6000   # characters per moderation input (reports are split, any flagged chunk counts)


class ModerationError(LLMError):
    """The moderation service failed; the run stops instead of continuing unscreened."""


class Blocked(Exception):
    """Content moderation stopped the run (a harmful request or report).

    ``stage`` is "request" (nothing was researched) or "report" (the research is done and
    stored, so the run can be recovered through a follow-up)."""

    def __init__(self, message: str, stage: str = "request"):
        super().__init__(message)
        self.stage = stage


class ModerationResult(BaseModel):
    stage: str
    provider: str
    blocked: bool = False
    blocked_categories: list[str] = []
    warn_categories: list[str] = []

    @property
    def flagged(self) -> bool:
        return bool(self.blocked_categories or self.warn_categories)


class Moderator(Protocol):
    name: str

    def check(self, text: str, *, stage: Stage, context: str = "") -> ModerationResult:
        """``context`` is the user's request when ``text`` is model output (the report)."""
        ...


def chunks(text: str, size: int = CHUNK) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _result(stage: str, provider: str, flagged: set[str], block: set[str],
            strict: bool) -> ModerationResult:
    blocked = sorted(flagged if strict else flagged & block)
    return ModerationResult(stage=stage, provider=provider, blocked=bool(blocked),
                            blocked_categories=blocked, warn_categories=sorted(flagged - set(blocked)))


# =================================================================== engine-facing gate


class ModerationGate:
    """The three checkpoints both engines use. A no-op when ``moderator`` is None."""

    def __init__(self, moderator: Moderator | None, emit):
        self.moderator = moderator
        self.emit = emit

    def _check(self, text: str, stage: Stage, context: str = "") -> ModerationResult | None:
        if self.moderator is None:
            return None
        result = self.moderator.check(text, stage=stage, context=context)
        if result.flagged:
            self.emit("moderation.flagged",
                      f"{result.provider} flagged the {stage}: "
                      + ", ".join(result.blocked_categories + result.warn_categories)
                      + (" (blocking)" if result.blocked else " (not blocking)"),
                      stage=stage, blocked=result.blocked_categories,
                      warned=result.warn_categories)
        return result

    def request(self, topic: str) -> None:
        """Raises ``Blocked`` before any research runs."""
        result = self._check(topic, "request")
        if result is not None and result.blocked:
            raise Blocked("the request was flagged ("
                          + ", ".join(result.blocked_categories) + ")", stage="request")

    def input_blocked(self, text: str, what: str) -> bool:
        """Flagged user input mid-run is ignored (never reaches a prompt), not fatal."""
        result = self._check(text, "input")
        if result is not None and result.blocked:
            self.emit("moderation.ignored", f"Ignored a flagged {what}; it was not used")
            return True
        return False

    def report(self, report, topic: str) -> None:
        """Raises ``Blocked`` for a harmful report; records non-blocking flags in its quality."""
        result = self._check(report.to_markdown(), "report", context=topic)
        if result is None:
            return
        if result.blocked:
            raise Blocked("the report was flagged (" + ", ".join(result.blocked_categories)
                          + "); it was not saved", stage="report")
        if result.warn_categories and report.quality is not None:
            report.quality.moderation_warnings = result.warn_categories
            report.quality.moderation_provider = result.provider


# =================================================================== OpenAI moderation


class OpenAIModerator:
    """https://platform.openai.com/docs/guides/moderation - free for OpenAI API users."""

    name = "openai-moderation"
    MODEL = "omni-moderation-latest"
    # Categories that indicate content enabling harm. Others (violence, hate, harassment,
    # sexual, self-harm, violence/graphic) are common in legitimate reporting: warn only.
    BLOCK = {"sexual/minors", "self-harm/instructions", "self-harm/intent", "illicit",
             "illicit/violent", "hate/threatening", "harassment/threatening"}

    def __init__(self, *, api_key: str | None = None, strict: bool = False,
                 client: openai.OpenAI | None = None):
        self.client = client or openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.strict = strict

    def check(self, text: str, *, stage: Stage, context: str = "") -> ModerationResult:
        try:
            response = self.client.moderations.create(model=self.MODEL, input=chunks(text))
        except openai.OpenAIError as e:
            raise ModerationError(f"moderation ({stage}) unavailable: {e}") from e
        flagged: set[str] = set()
        for r in response.results:
            if r.flagged:
                cats = r.categories.model_dump(by_alias=True)
                flagged |= {name for name, hit in cats.items() if hit}
        return _result(stage, self.name, flagged, self.BLOCK, self.strict)


# =================================================================== Llama Guard 3


LLAMA_GUARD_CATEGORIES = {
    "S1": "violent crimes", "S2": "non-violent crimes", "S3": "sex-related crimes",
    "S4": "child sexual exploitation", "S5": "defamation", "S6": "specialized advice",
    "S7": "privacy", "S8": "intellectual property", "S9": "indiscriminate weapons",
    "S10": "hate", "S11": "suicide & self-harm", "S12": "sexual content", "S13": "elections",
}


class LlamaGuardModerator:
    """Llama Guard 3 over an OpenAI-compatible endpoint (Ollama: ``llama-guard3``)."""

    name = "llama-guard"
    # Defamation, specialized (medical/legal/financial) advice and IP are routine in research
    # summaries: warn only. Everything else blocks.
    BLOCK = {"S1", "S2", "S3", "S4", "S7", "S9", "S10", "S11", "S12", "S13"}

    def __init__(self, *, base_url: str = "http://localhost:11434/v1",
                 model: str = "llama-guard3", strict: bool = False,
                 client: openai.OpenAI | None = None):
        self.client = client or openai.OpenAI(base_url=base_url, api_key="local")
        self.model = model
        self.strict = strict

    def check(self, text: str, *, stage: Stage, context: str = "") -> ModerationResult:
        flagged: set[str] = set()
        for part in chunks(text):
            # Model output is classified as the assistant turn of a conversation.
            messages = ([{"role": "user", "content": context or "Research request"},
                         {"role": "assistant", "content": part}] if stage == "report"
                        else [{"role": "user", "content": part}])
            try:
                reply = self.client.chat.completions.create(model=self.model, messages=messages)
            except openai.OpenAIError as e:
                raise ModerationError(f"moderation ({stage}) unavailable: {e}") from e
            flagged |= parse_llama_guard(reply.choices[0].message.content or "")
        named = {f"{c} {LLAMA_GUARD_CATEGORIES.get(c, '')}".strip() for c in flagged}
        block = {f"{c} {LLAMA_GUARD_CATEGORIES.get(c, '')}".strip() for c in self.BLOCK}
        return _result(stage, self.name, named, block, self.strict)


def parse_llama_guard(output: str) -> set[str]:
    """'safe' -> set(); 'unsafe\\nS1,S10' -> {'S1', 'S10'}. Anything else is an error."""
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        raise ModerationError("moderation: empty Llama Guard reply")
    verdict = lines[0].lower()
    if verdict == "safe":
        return set()
    if verdict == "unsafe":
        codes = {c.strip().upper() for line in lines[1:] for c in line.split(",") if c.strip()}
        return codes or {"unspecified"}
    raise ModerationError(f"moderation: unexpected Llama Guard reply {lines[0][:40]!r}")


# =================================================================== testing


class StaticModerator:
    """Blocks or warns on configured phrases; records what it saw. For tests and demos."""

    name = "static"

    def __init__(self, block: tuple[str, ...] = (), warn: tuple[str, ...] = (),
                 fail: bool = False):
        self.block, self.warn, self.fail = block, warn, fail
        self.seen: list[tuple[str, str]] = []

    def check(self, text: str, *, stage: Stage, context: str = "") -> ModerationResult:
        self.seen.append((stage, text))
        if self.fail:
            raise ModerationError(f"moderation ({stage}) unavailable: test failure")
        low = text.lower()
        blocked = sorted(p for p in self.block if p in low)
        warned = sorted(p for p in self.warn if p in low)
        return ModerationResult(stage=stage, provider=self.name, blocked=bool(blocked),
                                blocked_categories=blocked, warn_categories=warned)


# Phrases the offline screen blocks on. Deliberately narrow and literal: the offline path
# has no model to judge intent, so it catches the obvious harm-enabling requests (and the
# harmful evaluation cases) without pretending to be a classifier.
OFFLINE_BLOCK = ("nerve agent", "sarin", "untraceable firearm", "ghost gun", "pipe bomb",
                 "improvised explosive", "bioweapon", "child sexual")


def offline_moderator() -> "StaticModerator":
    """The screen an --offline run uses. Without one, offline runs never reach a moderation
    checkpoint, so the harmful evaluation cases score on the fake model's refusal alone and a
    broken gate would look exactly like a working one."""
    return StaticModerator(block=OFFLINE_BLOCK)


def recovery_hint(stage: str, session_id: str) -> str:
    """What to do about a block. A flagged report is worth recovering: the research behind it
    is already stored, and small guard models do misfire (a 1B Llama Guard called a report on
    AI in newsrooms "S1 violent crimes"). A flagged request researched nothing to recover."""
    if stage != "report" or not session_id:
        return ""
    return (f" The research is kept: re-check it with a stronger moderator, or none, using "
            f"--follow-up {session_id} (e.g. --moderation-model llama-guard3, or "
            f"--moderation none).")
