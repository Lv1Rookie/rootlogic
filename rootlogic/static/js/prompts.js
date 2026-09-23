// Editing the four system prompts.
//
// An edit is a stored setting, so the next run picks it up without being told. A run that
// used a customised prompt says so in its action log and in the report's limitations, because
// a rewritten verifier or writer changes what the report's own numbers mean.

import { $, esc } from "./dom.js";
import { api } from "./api.js";

const BLURB = {
  planner: "How the topic is broken into sub-tasks.",
  researcher: "How each sub-agent searches and judges sources.",
  verifier: "How claims are checked against the pages they cite.",
  writer: "How the final report is written.",
};

export async function loadPrompts() {
  const { prompts } = await api("/api/prompts");
  const box = $("#prompts");
  box.innerHTML = "";
  for (const p of prompts) box.appendChild(row(p));
  $("#prompts-custom").textContent = prompts.some(p => p.custom)
    ? prompts.filter(p => p.custom).map(p => p.name).join(", ") + " customised"
    : "All four are the defaults.";
}

function row(p) {
  const el = document.createElement("details");
  el.className = "prompt" + (p.custom ? " custom" : "");
  el.innerHTML = `
    <summary><span class="pname">${esc(p.name)}</span>
      <span class="meta">${esc(BLURB[p.name] || "")}</span>
      ${p.custom ? '<span class="tag">custom</span>' : ""}</summary>
    <textarea rows="10" spellcheck="false">${esc(p.text)}</textarea>
    <div class="row">
      <button type="button" class="btn primary save">Save</button>
      <button type="button" class="btn reset"${p.custom ? "" : " disabled"}>Reset to default</button>
      <span class="meta saved hidden">Saved — used by the next run.</span>
    </div>`;

  const text = $("textarea", el);
  $(".save", el).onclick = async () => {
    try {
      await api("/api/prompts", { method: "POST", body: { name: p.name, text: text.value } });
      $(".saved", el).classList.remove("hidden");
      await loadPrompts();
    } catch (e) { alert(e.message); }
  };
  $(".reset", el).onclick = async () => {
    try {
      await api(`/api/prompts/${p.name}`, { method: "DELETE" });
      await loadPrompts();
    } catch (e) { alert(e.message); }
  };
  return el;
}
