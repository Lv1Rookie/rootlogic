// Token and cost accounting for a finished session, plus the resume affordance.

import { $, esc } from "./dom.js";
import { state } from "./state.js";

export function renderUsage(d) {
  const rows = d.usage_by_purpose.map(r => `<tr><td>${esc(r.purpose)}</td><td class="num">${r.calls}</td>
    <td class="num">${(r.input_tokens || 0).toLocaleString()}</td><td class="num">${(r.output_tokens || 0).toLocaleString()}</td>
    <td class="num">${r.web_searches || 0}</td><td class="num">$${(r.cost_usd || 0).toFixed(3)}</td></tr>`).join("");
  const u = d.usage;
  $("#tab-usage").innerHTML = rows ? `<table><thead><tr><th>Step</th><th class="num">Calls</th><th class="num">Input</th>
    <th class="num">Output</th><th class="num">Searches</th><th class="num">Cost</th></tr></thead><tbody>${rows}</tbody>
    <tfoot><tr><td><b>Total</b></td><td class="num">${u.calls}</td><td class="num">${u.input_tokens.toLocaleString()}</td>
    <td class="num">${u.output_tokens.toLocaleString()}</td><td class="num">${u.web_searches}</td>
    <td class="num"><b>$${u.cost_usd.toFixed(2)}</b></td></tr></tfoot></table>
    <p class="meta">Cache reads: ${u.cache_read_tokens.toLocaleString()} tokens. Offline runs cost $0.</p>`
    : `<div class="empty">No LLM calls recorded.</div>`;
  $("#v-meta").textContent = `session ${d.session.id} · ${u.calls} LLM calls · $${u.cost_usd.toFixed(2)}`;
}

export function maybeResume(d) {
  const resumable = ["failed", "interrupted"].includes(d.session.status) && !state.run;
  $("#resume").classList.toggle("hidden", !resumable);
}
