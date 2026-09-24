"""Optional Langfuse tracing: off unless configured, and never able to break a run."""

from datetime import date

import pytest

from rootlogic.fake_llm import FakeLLM
from rootlogic.llm import Usage
from rootlogic.orchestrator import Orchestrator
from rootlogic.store import Store
from rootlogic.tracing import NullTracer, Tracer, traced_sink, tracer_from_env
from tests.test_orchestrator import TODAY, ScriptedUI


class Observation:
    def __init__(self, sink, kind, kw):
        self.sink, self.kind, self.kw = sink, kind, kw

    def update(self, **kw):
        self.sink.trace_updates.append(kw)

    def end(self):
        self.sink.ended.append(self.kw.get("name"))


class Recorder:
    """Stands in for the v4 Langfuse client: records what would have been sent."""

    def __init__(self, explode=False):
        self.observations, self.events, self.trace_updates, self.ended = [], [], [], []
        self.flushed = 0
        self.explode = explode

    def _check(self):
        if self.explode:
            raise RuntimeError("langfuse is down")

    def create_trace_id(self, seed=None):
        self._check()
        return "trace-" + (seed or "x")

    def start_observation(self, **kw):
        self._check()
        self.observations.append(kw)
        return Observation(self, kw.get("as_type"), kw)

    def create_event(self, **kw):
        self._check()
        self.events.append(kw)

    def flush(self):
        self.flushed += 1

    # convenience views for the assertions below
    @property
    def generations(self):
        return [o for o in self.observations if o.get("as_type") == "generation"]


def test_tracing_is_off_without_credentials(monkeypatch):
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(tracer_from_env(), NullTracer)


def test_a_null_tracer_accepts_everything_and_does_nothing():
    t: Tracer = NullTracer()
    t.session("s1", "a topic")
    t.generation(Usage(purpose="plan", model="m", input_tokens=1, output_tokens=2))
    t.event("task.done", "[t1] Done", {"task": "t1"})
    t.flush()


def test_a_run_reports_every_llm_call_and_event(tmp_path):
    from rootlogic.tracing import LangfuseTracer

    client = Recorder()
    tracer = LangfuseTracer(client)
    store, ui = Store(), ScriptedUI()
    llm = FakeLLM(traced_sink(lambda u: None, tracer))
    engine = Orchestrator(llm, store, ui, reports_dir=tmp_path, today=TODAY, tracer=tracer)
    engine.run("impact of generative AI on newsrooms")

    spans = [o for o in client.observations if o.get("as_type") == "span"]
    assert len(spans) == 1 and spans[0]["name"] == "research"
    assert spans[0]["input"] == "impact of generative AI on newsrooms"
    assert client.trace_updates[0]["metadata"]["session_id"] == engine.sid  # one per session

    purposes = [g["name"] for g in client.generations]
    assert "plan" in purposes and any(p.startswith("research:") for p in purposes)
    plan = next(g for g in client.generations if g["name"] == "plan")
    assert plan["model"] == "offline-fake"
    assert plan["usage_details"]["input"] > 0 and "output" in plan["usage_details"]

    kinds = [e["name"] for e in client.events]
    assert "session.started" in kinds and "report.checked" in kinds
    assert client.flushed >= 1


def test_a_broken_tracer_never_fails_the_run(tmp_path):
    """Observability is not worth losing research over."""
    from rootlogic.tracing import LangfuseTracer

    tracer = LangfuseTracer(Recorder(explode=True))
    store = Store()
    llm = FakeLLM(traced_sink(lambda u: None, tracer))
    report = Orchestrator(llm, store, ScriptedUI(), reports_dir=tmp_path, today=TODAY,
                          tracer=tracer).run("impact of generative AI on newsrooms")

    assert report is not None
    assert store.session(report.session_id)["status"] == "done"
