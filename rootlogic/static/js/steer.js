// Steering a run in flight, without reading the manual.
//
// The engines only accept overrides at a checkpoint between waves, so a click here does two
// things: queue the command, and ask the run to pause. When the engine reaches its checkpoint
// and asks what to do, the queued commands are sent automatically and the run continues.

import { $ } from "./dom.js";
import { api } from "./api.js";
import { state } from "./state.js";

let queued = [];

export const hasQueued = () => queued.length > 0;

/** Forget anything queued. A command is meant for the run it was clicked on: a run that ends
 *  before its next checkpoint must not hand its skips to whatever runs next. */
export function resetSteer() {
  queued = [];
  note("");
}

export function drain() {
  const cmds = queued;
  queued = [];
  note("");
  return cmds;
}

/** Queue a command and pause the run so it can be applied. */
export async function steer(...cmds) {
  if (!state.run) return;
  queued.push(...cmds);
  note(describe(queued) + " — applying at the next checkpoint…");
  try {
    await api(`/api/runs/${state.run}/checkpoint`, { method: "POST" });
  } catch (e) {
    queued = [];
    note("");
    alert(e.message);
  }
}

function describe(cmds) {
  const skips = cmds.filter(c => c.action === "skip").map(c => c.arg);
  const parts = [];
  if (skips.length) parts.push(`skip ${skips.join(", ")}`);
  if (cmds.some(c => c.action === "add")) parts.push("add a sub-task");
  if (cmds.some(c => c.action === "note")) parts.push("note guidance");
  return parts.join(" · ") || "no change";
}

function note(text) {
  const el = $("#steer-note-out");
  if (el) el.textContent = text;
}
