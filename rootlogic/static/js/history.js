// Past sessions: the sidebar list, suggested follow-up topics, and reopening a session.

import { $, esc, md } from "./dom.js";
import { api } from "./api.js";
import { state, resetView, setStatus, showTab } from "./state.js";
import { setPlan, trackTask } from "./plan.js";
import { logEvent } from "./log.js";
import { renderUsage, maybeResume } from "./usage.js";
import { trackStage } from "./stages.js";
import { trackAgent } from "./agents.js";

export async function loadHistory() {
  const { sessions, suggestions } = await api("/api/sessions");
  const h = $("#history");
  h.innerHTML = sessions.length ? "" : `<div class="meta">No sessions yet.</div>`;
  for (const s of sessions) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "item" + (s.id === state.sid ? " active" : "");
    el.innerHTML = `<span class="t">${esc(s.topic)}</span><span class="pill ${esc(s.status)}">${esc(s.status)}</span>
      <span class="m">${s.parent_id ? "↳ follows " + esc(s.parent_id) + " · " : ""}${
        esc(s.created_at.slice(0, 16).replace("T", " "))} · $${s.cost_usd.toFixed(2)} · ${esc(s.id)}</span>`;
    el.onclick = () => openSession(s.id);
    h.appendChild(el);
  }
  $("#suggest-wrap").classList.toggle("hidden", !suggestions.length);
  $("#suggestions").innerHTML = "";
  for (const t of suggestions) {
    const c = document.createElement("button");
    c.type = "button";
    c.className = "chip";
    c.textContent = t;
    c.onclick = () => { $("#topic").value = t; $("#topic").focus(); };
    $("#suggestions").appendChild(c);
  }
}

export async function openSession(sid) {
  const d = await api(`/api/sessions/${sid}`);
  resetView(d.session.topic);
  state.sid = sid;
  setStatus(d.session.status);
  if (d.session.plan_json) setPlan(JSON.parse(d.session.plan_json));
  // Same three views as a live run: the stored events are the only difference.
  for (const ev of d.events) { trackStage(ev); trackAgent(ev); trackTask(ev); logEvent(ev); }
  if (d.report) { $("#tab-report").innerHTML = md(d.report); showTab("report"); }
  renderUsage(d);
  maybeResume(d);
  $("#forget").classList.remove("hidden");
  $("#continue").classList.toggle("hidden", d.session.status !== "done");
  loadHistory();
}
