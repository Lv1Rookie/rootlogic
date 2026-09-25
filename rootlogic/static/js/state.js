// The run view's shared state, and the three functions that move it between views.
//
// `state` is a single mutable object imported by the other modules: one live run at a time,
// so a store would be ceremony. Replacing its contents (rather than the binding) keeps every
// importer pointing at the same object.

import { $ } from "./dom.js";
import { resetStages } from "./stages.js";
import { resetAgents } from "./agents.js";
import { resetSteer } from "./steer.js";
import { clearReport } from "./export.js";
import { refreshOutline } from "./outline.js";

export const state = { run: null, sid: null, source: null, tasks: {}, order: [], objective: "",
                       paused: false };

export function resetState() {
  if (state.source) state.source.close();
  Object.assign(state, { run: null, sid: null, source: null, tasks: {}, order: [], objective: "",
                         paused: false });
}

// The waiting state is authored in index.html, so it is captured once rather than duplicated
// here in a template string that would drift from it.
const waitingMarkup = $("#report-body")?.innerHTML ?? "";

export function setStatus(s) {
  const p = $("#v-status");
  p.className = "pill " + s;
  p.textContent = s;
  const live = s === "running" && !!state.run;
  // Pause is a hold: it swaps for Resume and the run waits there, doing nothing, until
  // Resume is pressed. Override is the other thing the old "Pause & override" did.
  $("#pause").classList.toggle("hidden", !live || state.paused);
  $("#unpause").classList.toggle("hidden", !live || !state.paused);
  $("#override").classList.toggle("hidden", !live || state.paused);
  // Abort used to live only inside the override card, which appears after a pause is
  // honoured - so a run that would not pause could not be stopped at all.
  $("#abort").classList.toggle("hidden", !live);
}

/** Reflect a control.paused / control.resumed event in the buttons. */
export function setPaused(paused) {
  state.paused = paused;
  setStatus($("#v-status").textContent);
}

export function showTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  ["log", "report", "usage"].forEach(n => $("#tab-" + n).classList.toggle("hidden", n !== name));
}

/** Clear the run view and show it for `topic`. */
export function resetView(topic) {
  resetState();
  $("#welcome").classList.add("hidden");
  $("#view").classList.remove("hidden");
  $("#v-topic").textContent = topic;
  $("#v-meta").textContent = "";
  $("#cards").innerHTML = "";
  resetStages();
  resetAgents();
  resetSteer();
  $("#tab-log").innerHTML = "";
  $("#report-body").innerHTML = waitingMarkup;
  clearReport();
  $("#tab-usage").innerHTML = `<div class="empty">Usage appears when the session ends.</div>`;
  $("#plan-rows").innerHTML = "";
  $("#plan-objective").textContent = "";
  $("#plan-panel").classList.add("hidden");
  state.paused = false;
  ["#resume", "#forget", "#continue"].forEach(s => $(s).classList.add("hidden"));
  setStatus("running");
  showTab("log");
  refreshOutline();
}
