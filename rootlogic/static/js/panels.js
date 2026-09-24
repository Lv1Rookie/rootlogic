// The two standing-settings panels: source rules and the long-term profile.

import { $, esc } from "./dom.js";
import { api } from "./api.js";

const RULE_TEXT = { block: "blocked", allow: "only these", trust: "trusted", distrust: "unreliable" };

export async function loadRules() {
  const { rules } = await api("/api/sources");
  const box = $("#rules");
  box.innerHTML = rules.length ? "" : `<div class="meta">No rules yet.</div>`;
  for (const r of rules) {
    const row = document.createElement("div");
    row.className = "pref";
    row.innerHTML = `<div><div class="cat">${esc(RULE_TEXT[r.rule])}</div>${esc(r.domain)}</div>
      <button class="x" type="button" title="Remove rule" aria-label="Remove rule for ${esc(r.domain)}">×</button>`;
    $(".x", row).onclick = async () => {
      await api(`/api/sources/${encodeURIComponent(r.domain)}`, { method: "DELETE" });
      loadRules();
    };
    box.appendChild(row);
  }
}

export async function loadProfile() {
  const { preferences } = await api("/api/profile");
  const box = $("#profile");
  box.innerHTML = preferences.length ? ""
    : `<div class="meta">Empty. It learns from your answers and notes, or add your own.</div>`;
  for (const p of preferences) {
    const row = document.createElement("div");
    row.className = "pref";
    row.innerHTML = `<div><div class="cat">${esc(p.category.replace("_", " "))}</div>${esc(p.text)}</div>
      <button class="x" type="button" title="Forget this preference" aria-label="Forget ${esc(p.text)}">×</button>`;
    $(".x", row).onclick = async () => {
      await api(`/api/profile/${p.id}`, { method: "DELETE" });
      loadProfile();
    };
    box.appendChild(row);
  }
}
