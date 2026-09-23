// The pipeline strip: which stage the run is in, and which are done.
//
// Stages are derived from the event types the engines already emit rather than from new
// engine code, so both engines light the same strip and the orchestration stays untouched.

import { $, esc } from "./dom.js";

const STAGES = [
  { key: "clarify",  label: "Clarify",  starts: ["clarify.started"],
    ends: ["clarify.skipped", "clarify.asking", "user.answered", "user.skipped"] },
  { key: "plan",     label: "Plan",     starts: ["plan.started"],
    ends: ["plan.created", "plan.approved"] },
  { key: "research", label: "Research", starts: ["task.started"],
    ends: ["reflect.started", "verify.started"] },
  { key: "reflect",  label: "Reflect",  starts: ["reflect.started"], ends: ["reflect.done"] },
  { key: "verify",   label: "Verify",   starts: ["verify.started", "verify.off"],
    ends: ["verify.done", "verify.off"] },
  { key: "analyze",  label: "Analyse",  starts: ["analyze.started"], ends: ["analyze.done"] },
  { key: "report",   label: "Report",   starts: ["report.started"],
    ends: ["report.checked", "session.done"] },
];

let reached = new Set();
let current = null;

export function resetStages() {
  reached = new Set();
  current = null;
  render();
}

/** Advance the strip from one event. Unknown event types are ignored. */
export function trackStage(ev) {
  let changed = false;
  for (const s of STAGES) {
    if (s.starts.includes(ev.type) && current !== s.key) { current = s.key; changed = true; }
    if (s.ends.includes(ev.type) && !reached.has(s.key)) { reached.add(s.key); changed = true; }
  }
  // A finished run has no current stage: everything it did is behind it.
  if (["session.done", "session.failed", "session.blocked", "session.aborted",
       "run.finished"].includes(ev.type)) {
    current = null;
    changed = true;
  }
  if (changed) render();
}

function render() {
  const box = $("#stages");
  if (!box) return;
  box.innerHTML = STAGES.map(s => {
    const state = s.key === current ? "active" : reached.has(s.key) ? "done" : "todo";
    return `<span class="stage ${state}" title="${esc(s.label)}">${esc(s.label)}</span>`;
  }).join('<span class="stage-sep" aria-hidden="true">→</span>');
}
