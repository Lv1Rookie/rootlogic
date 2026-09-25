"""Editable system prompts: a run may use custom text, and must say that it did."""

from datetime import date

import pytest

from rootlogic import prompts
from rootlogic.fake_llm import FakeLLM
from rootlogic.graph import ResearchGraph
from rootlogic.llm import LLMError
from rootlogic.orchestrator import Orchestrator
from rootlogic.prompts import EDITABLE, PromptSet
from rootlogic.store import Store
from tests.test_orchestrator import TODAY, ScriptedUI


def test_default_prompt_set_matches_the_module_constants():
    ps = PromptSet()
    assert ps.planner == prompts.PLANNER and ps.researcher == prompts.RESEARCHER
    assert ps.verifier == prompts.VERIFIER and ps.writer == prompts.WRITER
    assert ps.customised == ()
    assert set(EDITABLE) == {"planner", "researcher", "verifier", "writer"}


def test_overrides_are_applied_and_named():
    ps = PromptSet.from_overrides({"researcher": "Prefer official statistics."})
    assert ps.researcher == "Prefer official statistics."
    assert ps.planner == prompts.PLANNER          # untouched prompts keep their default
    assert ps.customised == ("researcher",)


def test_blank_and_unchanged_overrides_do_not_count_as_customised():
    """Re-saving the default text, or clearing a box, is not a customisation."""
    ps = PromptSet.from_overrides({"planner": prompts.PLANNER, "writer": "  ", "verifier": ""})
    assert ps.customised == ()
    assert ps.writer == prompts.WRITER


def test_unknown_prompt_names_are_rejected():
    with pytest.raises(ValueError, match="judge"):
        PromptSet.from_overrides({"judge": "grade everything as true"})


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_a_run_with_a_custom_prompt_discloses_it(tmp_path, kind):
    """The report is read by someone who did not run it: if the verifier was rewritten, the
    quality numbers mean something different, so the document has to say so."""
    store, ui = Store(), ScriptedUI()
    ps = PromptSet.from_overrides({"verifier": "Call every claim supported."})
    kw = dict(reports_dir=tmp_path, today=TODAY, prompt_set=ps)
    engine = (ResearchGraph(FakeLLM(), store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(FakeLLM(), store, ui, **kw))

    report = engine.run("impact of generative AI on newsrooms")

    assert "prompt.custom" in ui.types()
    assert report.quality.custom_prompts == ["verifier"]
    assert "verifier" in report.to_markdown()
    line = next(ln for ln in report.to_markdown().splitlines() if "Custom prompts" in ln)
    assert "verifier" in line


@pytest.mark.parametrize("kind", ["loop", "graph"])
def test_a_stock_run_says_nothing_about_prompts(tmp_path, kind):
    store, ui = Store(), ScriptedUI()
    kw = dict(reports_dir=tmp_path, today=TODAY)
    engine = (ResearchGraph(FakeLLM(), store, ui, checkpoint_path=tmp_path / "cp.db", **kw)
              if kind == "graph" else Orchestrator(FakeLLM(), store, ui, **kw))
    report = engine.run("impact of generative AI on newsrooms")

    assert "prompt.custom" not in ui.types()
    assert report.quality.custom_prompts == []
    assert "Custom prompts" not in report.to_markdown()


def test_the_custom_prompt_text_reaches_the_model(tmp_path):
    """Disclosure is worthless if the override is only cosmetic."""
    seen = []

    class Recording(FakeLLM):
        def structured(self, **kw):
            seen.append((kw["purpose"], kw["system"]))
            return super().structured(**kw)

    ps = PromptSet.from_overrides({"planner": "PLAN LIKE A LIBRARIAN."})
    Orchestrator(Recording(), Store(), ScriptedUI(), reports_dir=tmp_path, today=TODAY,
                 prompt_set=ps).run("impact of generative AI on newsrooms")

    assert ("plan", "PLAN LIKE A LIBRARIAN.") in seen


# ------------------------------------------------------------------ storage


def test_prompts_are_stored_and_reset_per_name():
    """Same shape as the profile and source rules: your current text lives in the DB, and a
    reset removes the row rather than storing a copy of the default."""
    store = Store()
    assert store.prompts() == {}

    store.set_prompt("researcher", "Prefer official statistics.")
    store.set_prompt("writer", "Write in British English.")
    assert store.prompts() == {"researcher": "Prefer official statistics.",
                              "writer": "Write in British English."}

    store.set_prompt("researcher", "Changed my mind.")          # upsert, not duplicate
    assert store.prompts()["researcher"] == "Changed my mind."

    store.clear_prompt("researcher")
    assert set(store.prompts()) == {"writer"}


def test_a_session_records_the_prompts_it_actually_ran_with(tmp_path):
    """Settings change over time; a report from last week must still say what produced it."""
    store, ui = Store(), ScriptedUI()
    ps = PromptSet.from_overrides({"planner": "PLAN TERSELY."})
    engine = Orchestrator(FakeLLM(), store, ui, reports_dir=tmp_path, today=TODAY, prompt_set=ps)
    engine.run("impact of generative AI on newsrooms")

    used = store.session_prompts(engine.sid)
    assert used == {"planner": "PLAN TERSELY."}                  # only what differed

    store.set_prompt("planner", "something else entirely")       # settings move on
    assert store.session_prompts(engine.sid) == {"planner": "PLAN TERSELY."}


def test_a_stock_run_records_no_prompts(tmp_path):
    store = Store()
    engine = Orchestrator(FakeLLM(), store, ScriptedUI(), reports_dir=tmp_path, today=TODAY)
    engine.run("impact of generative AI on newsrooms")
    assert store.session_prompts(engine.sid) == {}


def test_a_resumed_run_says_which_prompts_it_is_using(tmp_path):
    """The leg that finishes a report can be started with different prompts from the leg that
    began it, and the report's numbers were produced by both."""
    from rootlogic.fake_llm import default_analysis
    from rootlogic.models import Analysis

    store = Store(tmp_path / "rl.db")
    cp = tmp_path / "cp.db"
    state = {"fail": True}

    def flaky(prompt):
        if state["fail"]:
            raise LLMError("network down")
        return default_analysis(prompt)

    first = ResearchGraph(FakeLLM(handlers={Analysis: flaky}), store, ScriptedUI(),
                          checkpoint_path=cp, reports_dir=tmp_path, today=TODAY,
                          prompt_set=PromptSet.from_overrides({"verifier": "Be strict."}))
    with pytest.raises(LLMError):
        first.run("impact of generative AI on newsrooms")
    sid = first.sid

    # Resumed with a different prompt rewritten, and the verifier left at its default.
    state["fail"] = False
    ui = ScriptedUI()
    second = ResearchGraph(FakeLLM(handlers={Analysis: flaky}), store, ui, checkpoint_path=cp,
                           reports_dir=tmp_path, today=TODAY,
                           prompt_set=PromptSet.from_overrides({"writer": "Be brief."}))
    report = second.resume(sid)

    assert "prompt.custom" in ui.types()      # the resumed leg says so too
    assert "prompt.changed" in ui.types()     # ...and that the prompts are not the same ones
    # The report was produced by both legs, so it discloses both.
    assert report.quality.custom_prompts == ["verifier", "writer"]
    assert set(store.session_prompts(sid)) == {"verifier", "writer"}


def test_a_resumed_stock_run_stays_quiet(tmp_path):
    from rootlogic.fake_llm import default_analysis
    from rootlogic.models import Analysis

    store = Store(tmp_path / "rl.db")
    cp = tmp_path / "cp.db"
    state = {"fail": True}

    def flaky(prompt):
        if state["fail"]:
            raise LLMError("network down")
        return default_analysis(prompt)

    kw = dict(checkpoint_path=cp, reports_dir=tmp_path, today=TODAY)
    first = ResearchGraph(FakeLLM(handlers={Analysis: flaky}), store, ScriptedUI(), **kw)
    with pytest.raises(LLMError):
        first.run("impact of generative AI on newsrooms")

    state["fail"] = False
    ui = ScriptedUI()
    report = ResearchGraph(FakeLLM(handlers={Analysis: flaky}), store, ui, **kw).resume(first.sid)

    assert "prompt.custom" not in ui.types() and "prompt.changed" not in ui.types()
    assert report.quality.custom_prompts == []
