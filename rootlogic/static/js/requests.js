// Human-in-the-loop: the three things the agent stops to ask for.
//
// A clarifying question, plan review, and a mid-run override. Each arrives as a "request"
// event carrying a request_id, and each is answered by POSTing that id back.

import { $, esc } from "./dom.js";
import { api } from "./api.js";
import { state } from "./state.js";
import { setPlan } from "./plan.js";
import { hasQueued, drain } from "./steer.js";

function card(ev, title, bodyHTML) {
  const el = document.createElement("div");
  el.className = "panel card";
  el.id = `req-${ev.request_id}`;
  el.innerHTML = `<h3>${esc(title)}</h3>${bodyHTML}`;
  $("#cards").appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return el;
}

const send = (ev, answer) => api(`/api/runs/${state.run}/answer`, {
  method: "POST", body: { request_id: ev.request_id, answer },
}).catch(e => alert(e.message));

export function showRequest(ev) {
  if (ev.kind === "question") return askQuestion(ev);
  if (ev.kind === "plan") return reviewPlan(ev);
  if (ev.kind === "override") return override(ev);
}

function askQuestion(ev) {
  const el = card(ev, "The agent has a question", `<p>${esc(ev.question)}</p>
    <form class="row"><input type="text" style="flex:1" placeholder="Your answer (or skip)">
    <button class="btn primary">Answer</button><button type="button" class="btn skip">Skip</button></form>`);
  const input = $("input", el);
  input.focus();
  $("form", el).onsubmit = e => { e.preventDefault(); send(ev, input.value); };
  $(".skip", el).onclick = () => send(ev, "");
}

function reviewPlan(ev) {
  setPlan(ev.plan);
  const rows = ev.plan.subtasks.map(t => `<tr data-id="${esc(t.id)}"><td class="id">${esc(t.id)}</td>
    <td>${esc(t.question)}<div class="meta">${esc(t.rationale)}${
      t.depends_on.length ? " · after " + esc(t.depends_on.join(", ")) : ""}</div></td>
    <td>${t.origin === "previous" ? '<span class="tag">earlier · done</span>'
      : '<label class="inline"><input type="checkbox" checked> keep</label>'}</td></tr>`).join("");
  const el = card(ev, "Review the research plan", `<p class="meta">${esc(ev.plan.objective)}</p>
    <table><tbody>${rows}</tbody></table>
    <div class="stack" style="margin-top:12px">
      <div class="row"><input type="text" class="add" style="flex:1" placeholder="Add a sub-task question">
        <button type="button" class="btn addbtn">Add</button></div>
      <div class="added meta"></div>
      <div class="row"><label class="inline">Max source age (days, 0 = any)
        <input type="number" class="recency" min="0" style="width:90px" value="${ev.plan.recency_days}"></label></div>
      <div class="row"><button class="btn primary approve">Approve &amp; start</button><button class="btn danger reject">Reject</button></div>
    </div>`);

  const added = [];
  el.querySelectorAll("tr input").forEach(cb =>
    cb.onchange = () => cb.closest("tr").classList.toggle("dropped", !cb.checked));
  const addInput = $(".add", el);
  const addTask = () => {
    const q = addInput.value.trim();
    if (!q) return;
    added.push(q);
    addInput.value = "";
    $(".added", el).textContent = "Will add: " + added.join(" · ");
  };
  addInput.onkeydown = e => { if (e.key === "Enter") { e.preventDefault(); addTask(); } };
  $(".addbtn", el).onclick = addTask;
  $(".approve", el).onclick = () => (addTask(), send(ev, {
    approved: true, add: added, recency_days: parseInt($(".recency", el).value || "0", 10),
    drop: [...el.querySelectorAll("tr")].filter(r => $("input", r) && !$("input", r).checked)
                                        .map(r => r.dataset.id),
  }));
  $(".reject", el).onclick = () => send(ev, { approved: false });
}

function override(ev) {
  setPlan(ev.plan);
  // The pause came from a Skip or Steer click: apply what was asked and keep going, rather
  // than making the user confirm a panel they didn't open.
  if (hasQueued()) return send(ev, { commands: drain() });
  const pending = ev.plan.subtasks.filter(t => t.status === "pending");
  const el = card(ev, "Paused — override the agent", `
    ${pending.length ? `<p class="meta">Skip pending sub-tasks:</p><div class="chips skips">${pending.map(t =>
      `<label class="chip"><input type="checkbox" value="${esc(t.id)}"> ${esc(t.id)}: ${esc(t.question.slice(0, 60))}</label>`).join("")}</div>` : ""}
    <div class="stack" style="margin-top:10px">
      <input type="text" class="add" placeholder="Add a sub-task question">
      <input type="text" class="note" placeholder="Guidance for the agent, e.g. focus on the EU">
      <div class="row"><button class="btn primary go">Apply &amp; continue</button>
        <button class="btn stop">Stop &amp; write report</button><button class="btn danger abort">Abort</button></div>
    </div>`);

  const collect = () => {
    const cmds = [...el.querySelectorAll(".skips input:checked")].map(i => ({ action: "skip", arg: i.value }));
    const a = $(".add", el).value.trim(), n = $(".note", el).value.trim();
    if (a) cmds.push({ action: "add", arg: a });
    if (n) cmds.push({ action: "note", arg: n });
    return cmds;
  };
  $(".go", el).onclick = () => send(ev, { commands: collect() });
  $(".stop", el).onclick = () => send(ev, { commands: [...collect(), { action: "stop" }] });
  $(".abort", el).onclick = () =>
    confirm("Abort this research session?") && send(ev, { commands: [{ action: "abort" }] });
}
