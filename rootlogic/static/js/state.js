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

export const state = { run: null, sid: null, source: null, tasks: {}, order: [], objective: "" };

export function resetState() {
  if (state.source) state.source.close();
  Object.assign(state, { run: null, sid: null, source: null, tasks: {}, order: [], objective: "" });
}

export function setStatus(s) {
  const p = $("#v-status");
  p.className = "pill " + s;
  p.textContent = s;
  const live = s === "running" && !!state.run;
  $("#pause").classList.toggle("hidden", !live);
  // Abort used to live only inside the override card, which appears after a pause is
  // honoured - so a run that would not pause could not be stopped at all.
  $("#abort").classList.toggle("hidden", !live);
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
  $("#report-body").innerHTML =
    `<div class="empty">The finished report appears here. Research in progress…</div>`;
  clearReport();
  $("#tab-usage").innerHTML = `<div class="empty">Usage appears when the session ends.</div>`;
  $("#plan-rows").innerHTML = "";
  $("#plan-objective").textContent = "";
  $("#plan-panel").classList.add("hidden");
  ["#resume", "#forget", "#continue"].forEach(s => $(s).classList.add("hidden"));
  setStatus("running");
  showTab("log");
}
