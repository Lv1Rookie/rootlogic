// One live card per research sub-agent.
//
// A flat log interleaves three sub-agents working in parallel, which is exactly when a reader
// most wants to know who is doing what. Each card shows its own question, the searches and
// fetches as they arrive, and how its sources were judged when it finishes.

import { $, esc } from "./dom.js";

const agents = new Map();   // task id -> { question, status, searches, fetches, note, error }

export function resetAgents() {
  agents.clear();
  render();
}

function get(id) {
  if (!agents.has(id)) {
    agents.set(id, { question: "", status: "pending", searches: [], fetches: [], note: "", error: "" });
  }
  return agents.get(id);
}

/** Fold one event into the cards. Returns true if anything changed. */
export function trackAgent(ev) {
  const id = ev.data?.task;
  if (!id) return false;
  const a = get(id);

  switch (ev.type) {
    case "task.started":
      a.status = "running";
      a.question = (ev.message || "").replace(/^\[\w+\]\s*Researching:\s*/, "");
      break;
    case "subagent.search":
      a.searches.push({ text: ev.data.query || "", results: ev.data.results ?? 0,
                        error: ev.data.error || "" });
      break;
    case "subagent.fetch":
      a.fetches.push({ text: ev.data.url || "", error: ev.data.error || "" });
      break;
    case "task.done":
      a.status = "done";
      a.note = (ev.message || "").replace(/^\[\w+\]\s*Done:\s*/, "");
      break;
    case "task.failed":
      a.status = "failed";
      a.error = (ev.message || "").replace(/^\[\w+\]\s*Failed:\s*/, "");
      break;
    case "task.retry":
      a.status = "running";
      a.error = (ev.message || "").replace(/^\[\w+\]\s*/, "");
      break;
    case "override.skip":
      a.status = "skipped";
      break;
    default:
      return false;
  }
  render();
  return true;
}

function render() {
  const box = $("#agents");
  if (!box) return;
  const ids = [...agents.keys()].sort();
  box.classList.toggle("hidden", !ids.length);
  box.innerHTML = ids.map(id => cardHTML(id, agents.get(id))).join("");
}

function step(s, kind) {
  const detail = s.error
    ? `<span class="bad">${esc(kind === "search" ? "search failed" : "unreadable")}: ${esc(s.error)}</span>`
    : kind === "search" ? `${s.results} result(s)` : "read";
  return `<li><span class="what">${esc(s.text)}</span> <span class="meta">${detail}</span></li>`;
}

function cardHTML(id, a) {
  const steps = [...a.searches.map(s => step(s, "search")), ...a.fetches.map(s => step(s, "fetch"))];
  return `<div class="agent ${esc(a.status)}">
    <div class="agent-head">
      <span class="id">${esc(id)}</span>
      <span class="q">${esc(a.question || "…")}</span>
      <span class="pill ${esc(a.status)}">${esc(a.status)}</span>
      ${a.status === "running" ? `<button type="button" class="btn small skip-task" data-task="${esc(id)}">Skip</button>` : ""}
    </div>
    ${steps.length ? `<ul class="steps">${steps.join("")}</ul>` : ""}
    ${a.note ? `<div class="meta">${esc(a.note)}</div>` : ""}
    ${a.error ? `<div class="meta bad">${esc(a.error)}</div>` : ""}
  </div>`;
}
