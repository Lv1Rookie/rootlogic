// The plan table, and keeping each sub-task's status in step with the event stream.

import { $, esc } from "./dom.js";
import { state } from "./state.js";

export function setPlan(plan) {
  state.objective = plan.objective;
  state.tasks = {};
  state.order = [];
  for (const t of plan.subtasks) {
    state.tasks[t.id] = { question: t.question, status: t.status, origin: t.origin };
    state.order.push(t.id);
  }
  renderPlan();
}

export function renderPlan() {
  $("#plan-panel").classList.toggle("hidden", !state.order.length);
  $("#plan-objective").textContent = state.objective;
  $("#plan-rows").innerHTML = state.order.map(id => {
    const t = state.tasks[id];
    return `<tr><td class="id">${esc(id)}</td><td>${esc(t.question)}${
      t.origin === "previous" ? '<span class="tag">earlier</span>' : ""
    }</td><td><span class="pill ${esc(t.status)}">${esc(t.status)}</span></td></tr>`;
  }).join("");
}

/** Forget sub-tasks the user dropped at plan review, so the table shows what will run. */
export function dropTasks(ids) {
  for (const id of ids) delete state.tasks[id];
  state.order = state.order.filter(id => !ids.includes(id));
  renderPlan();
}

/** Read a task id out of an event message and move that task to its new status. */
export function trackTask(ev) {
  const m = /^\[(t\d+)\]\s*(?:Added \([^)]*\): |Researching: )?(.*)$/.exec(ev.message || "");
  const skip = /skipped \[(t\d+)\]/.exec(ev.message || "");
  const id = m?.[1] || skip?.[1];
  if (!id) return;
  const status = { "task.started": "running", "task.done": "done", "task.failed": "failed",
                   "task.added": "pending", "override.skip": "skipped" }[ev.type];
  if (!status) return;
  if (!state.tasks[id]) {
    state.tasks[id] = { question: m?.[2] || "", status };
    state.order.push(id);
  }
  state.tasks[id].status = status;
  renderPlan();
}
