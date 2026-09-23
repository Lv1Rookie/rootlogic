// Entry point: wire the DOM to the modules, then attach to anything already running.

import { $ } from "./dom.js";
import { api } from "./api.js";
import { state, resetView, setStatus, showTab } from "./state.js";
import { loadHistory } from "./history.js";
import { loadProfile, loadRules } from "./panels.js";
import { loadPrompts } from "./prompts.js";
import { startRun, follow } from "./run.js";

// --------------------------------------------------------------- tabs
document.querySelectorAll(".tab").forEach(t => t.onclick = () => showTab(t.dataset.tab));

// --------------------------------------------------------------- starting research
$("#start").onsubmit = async e => {
  e.preventDefault();
  const topic = $("#topic").value.trim();
  if (!topic) return;
  try { await startRun(topic); $("#topic").value = ""; } catch (err) { alert(err.message); }
};

$("#continue").onsubmit = async e => {
  e.preventDefault();
  const q = $("#continue-text").value.trim();
  if (!q || !state.sid) return;
  try { await startRun(q, state.sid); $("#continue-text").value = ""; } catch (err) { alert(err.message); }
};

$("#topic").onkeydown = e => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#start").requestSubmit();
};

// --------------------------------------------------------------- settings forms
$("#rule-add").onsubmit = async e => {
  e.preventDefault();
  const domain = $("#rule-domain").value.trim();
  if (!domain) return;
  try {
    await api("/api/sources", { method: "POST", body: { domain, rule: $("#rule-kind").value } });
    $("#rule-domain").value = "";
    loadRules();
  } catch (err) { alert(err.message); }
};

$("#pref-add").onsubmit = async e => {
  e.preventDefault();
  const text = $("#pref-text").value.trim();
  if (text.length < 2) return;
  try {
    await api("/api/profile", { method: "POST", body: { text } });
    $("#pref-text").value = "";
    loadProfile();
  } catch (err) { alert(err.message); }
};

// --------------------------------------------------------------- run controls
$("#pause").onclick = () =>
  api(`/api/runs/${state.run}/pause`, { method: "POST" }).catch(e => alert(e.message));

$("#resume").onclick = async () => {
  const sid = state.sid, topic = $("#v-topic").textContent;
  try {
    const run = await api(`/api/sessions/${sid}/resume`, {
      method: "POST", body: { offline: $("#offline").checked },
    });
    resetView(topic);
    state.run = run.run_id;
    state.sid = sid;
    setStatus("running");
    follow(run.run_id);
  } catch (err) { alert(err.message); }
};

$("#forget").onclick = async () => {
  if (!state.sid || !confirm("Delete this session, its log and its memory?")) return;
  await api(`/api/sessions/${state.sid}`, { method: "DELETE" });
  $("#view").classList.add("hidden");
  $("#welcome").classList.remove("hidden");
  state.sid = null;
  loadHistory();
};

// Remember engine/offline choices (per-browser convenience only).
for (const [id, prop] of [["engine", "value"], ["offline", "checked"],
                          ["use-profile", "checked"], ["verify", "checked"]]) {
  try {
    const v = localStorage.getItem("rootlogic." + id);
    if (v !== null) $("#" + id)[prop] = JSON.parse(v);
  } catch {}
  $("#" + id).addEventListener("change", e => {
    try { localStorage.setItem("rootlogic." + id, JSON.stringify(e.target[prop])); } catch {}
  });
}

// Re-attach to a live run after a page refresh.
(async () => {
  await Promise.all([loadHistory(), loadProfile(), loadRules(), loadPrompts()]);
  const live = (await api("/api/runs")).filter(r => r.status === "running").pop();
  if (live) {
    resetView(live.topic);
    state.run = live.run_id;
    setStatus("running");
    follow(live.run_id);
  }
})();
