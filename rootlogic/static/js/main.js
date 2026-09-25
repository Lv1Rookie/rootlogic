// Entry point: wire the DOM to the modules, then attach to anything already running.

import { $ } from "./dom.js";
import { api } from "./api.js";
import { state, resetView, setStatus, setPaused, setOverriding, showTab } from "./state.js";
import { loadHistory } from "./history.js";
import { loadProfile, loadRules } from "./panels.js";
import { loadPrompts } from "./prompts.js";
import { startRun, follow } from "./run.js";
import { steer } from "./steer.js";
import { savePdf, saveMarkdown } from "./export.js";
import { initTheme } from "./theme.js";
import { initScrollTop } from "./scrolltop.js";
import { initCheckboxes } from "./checkbox.js";
import { initSidebar } from "./sidebar.js";
import { initOutline } from "./outline.js";

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

// Enter starts the research; Shift+Enter (and Cmd/Ctrl+Enter, which some editors send) is a
// newline. An IME composing a character owns the Enter key while it commits that character,
// so submitting on it would swallow the word being typed.
$("#topic").onkeydown = e => {
  if (e.key !== "Enter" || e.shiftKey || e.isComposing || e.keyCode === 229) return;
  e.preventDefault();
  $("#start").requestSubmit();
};

// --------------------------------------------------------------- steering a live run
// The cards are re-rendered on every event, so the click is delegated to their container.
$("#agents").addEventListener("click", e => {
  const btn = e.target.closest(".skip-task");
  if (btn) steer({ action: "skip", arg: btn.dataset.task });
});

$("#steer").onsubmit = e => {
  e.preventDefault();
  const add = $("#steer-add").value.trim(), guidance = $("#steer-note").value.trim();
  const cmds = [];
  if (add) cmds.push({ action: "add", arg: add });
  if (guidance) cmds.push({ action: "note", arg: guidance });
  if (!cmds.length) return;
  $("#steer-add").value = "";
  $("#steer-note").value = "";
  steer(...cmds);
};

// --------------------------------------------------------------- exporting the result
// One button, one format picker beside it: the choice is the dropdown's, the action the
// button's, rather than two buttons that look alike and do different things.
$("#download").onclick = () =>
  ($("#dl-format").value === "md" ? saveMarkdown : savePdf)();

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
$("#pause").onclick = () => {
  setPaused(true);   // optimistic: the button flips now, the engine stops at its next boundary
  api(`/api/runs/${state.run}/pause`, { method: "POST" })
    .catch(e => { setPaused(false); alert(e.message); });
};

$("#unpause").onclick = () => {
  setPaused(false);
  api(`/api/runs/${state.run}/resume`, { method: "POST" }).catch(e => alert(e.message));
};

$("#override").onclick = () => {
  setOverriding(true);
  api(`/api/runs/${state.run}/checkpoint`, { method: "POST" })
    .catch(e => { setOverriding(false); alert(e.message); });
};

// Fold the status bar away. It is sticky, so while a long report is being read it sits on
// top of the text; the chevron gives that back without giving up the run controls for good.
$("#bar-toggle").onclick = () => {
  const bar = $("#statusbar");
  const collapsed = bar.classList.toggle("collapsed");
  $("#bar-toggle").setAttribute("aria-expanded", String(!collapsed));
  $("#bar-toggle").title = collapsed ? "Show the status bar" : "Hide the status bar";
};

$("#abort").onclick = () => {
  if (!confirm("Stop this run? Work already finished is kept, but no further research runs.")) return;
  api(`/api/runs/${state.run}/abort`, { method: "POST" }).catch(e => alert(e.message));
};

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

$("#retry").onclick = async () => {
  if (!state.sid) return;
  try {
    const run = await api(`/api/sessions/${state.sid}/retry`, {
      method: "POST", body: { offline: $("#offline").checked, engine: $("#engine").value },
    });
    const topic = $("#v-topic").textContent;
    resetView(topic);
    state.run = run.run_id;
    state.sid = run.session_id || null;
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
                          ["use-profile", "checked"], ["verify", "checked"],
                          ["searches", "value"], ["parallel", "value"]]) {
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
  initTheme();
  initScrollTop();
  initCheckboxes();
  initSidebar();
  initOutline();
  await Promise.all([loadHistory(), loadProfile(), loadRules(), loadPrompts()]);
  const live = (await api("/api/runs")).filter(r => r.status === "running").pop();
  if (live) {
    resetView(live.topic);
    state.run = live.run_id;
    setStatus("running");
    follow(live.run_id);
  }
})();
