// A live run: starting one, following its event stream, and reacting to what arrives.

import { $, md } from "./dom.js";
import { api } from "./api.js";
import { state, resetView, setStatus, setPaused, setOverriding, showTab } from "./state.js";
import { trackTask } from "./plan.js";
import { logEvent } from "./log.js";
import { showRequest } from "./requests.js";
import { renderUsage, maybeResume } from "./usage.js";
import { loadHistory } from "./history.js";
import { loadProfile } from "./panels.js";
import { trackStage } from "./stages.js";
import { trackAgent } from "./agents.js";
import { setReport } from "./export.js";
import { refreshOutline } from "./outline.js";

export async function startRun(topic, parent = null) {
  const run = await api("/api/runs", {
    method: "POST",
    body: {
      topic, engine: $("#engine").value, offline: $("#offline").checked,
      use_profile: $("#use-profile").checked, parent_session: parent,
      verify_claims: $("#verify").checked ? 30 : 0,
      max_searches: parseInt($("#searches").value || "4", 10),
      max_parallel: parseInt($("#parallel").value || "2", 10),
    },
  });
  resetView(topic);
  state.run = run.run_id;
  setStatus("running");
  if (parent) $("#v-meta").textContent = "follow-up of " + parent;
  follow(run.run_id);
}

/** Subscribe to a run's server-sent events. Reconnection and replay are the browser's job:
 *  the server sends an id with every event, so EventSource resumes where it left off. */
export function follow(runId) {
  const es = new EventSource(`/api/runs/${runId}/events`);
  state.source = es;
  es.onmessage = e => handle(JSON.parse(e.data));
  es.addEventListener("end", () => es.close());
}

export function handle(ev) {
  switch (ev.type) {
    case "request": return showRequest(ev);
    case "request.resolved": {
      const c = $(`#req-${ev.request_id}`);
      if (c) c.remove();
      refreshOutline();      // ...and out of it again when the card is answered
      return;
    }
    case "report":
      $("#report-body").innerHTML = md(ev.markdown);
      setReport(ev.markdown, { topic: $("#v-topic").textContent, session: state.sid });
      showTab("report");
      refreshOutline();          // the report brings its own headings to jump to
      return;
    case "run.error": return logEvent({ ...ev, type: "run.error" });
    case "run.finished":
      state.sid = ev.session_id || state.sid;
      trackStage(ev);
      setOverriding(false);     // nothing left to stop at
      setStatus(ev.status);
      $("#cards").innerHTML = "";
      afterFinish();
      return;
  }
  if (ev.type === "session.started") {
    const m = /Session (\w+)/.exec(ev.message);
    if (m) { state.sid = m[1]; $("#v-meta").textContent = "session " + m[1]; }
  }
  // The engine is the authority on whether it is actually held: the button flipped
  // optimistically on click, these events correct it.
  if (ev.type === "control.paused") { setOverriding(false); setPaused(true); }
  if (ev.type === "control.resumed" || ev.type === "control.aborted") setPaused(false);
  if (ev.type === "override.stop") {
    for (const id of state.order) if (state.tasks[id].status === "pending") state.tasks[id].status = "skipped";
  }
  trackStage(ev);
  trackAgent(ev);
  trackTask(ev);
  logEvent(ev);
}

async function afterFinish() {
  if (state.source) { state.source.close(); state.source = null; }
  loadHistory();
  loadProfile();
  $("#continue").classList.toggle("hidden", $("#v-status").textContent !== "done");
  refreshOutline();
  if (!state.sid) return;
  $("#forget").classList.remove("hidden");
  try {
    const d = await api(`/api/sessions/${state.sid}`);
    renderUsage(d);
    maybeResume(d);
  } catch {}
}
