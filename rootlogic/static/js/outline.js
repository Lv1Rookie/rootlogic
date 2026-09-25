// "Jump to…": the panels of the page, and the report's own headings.
//
// A finished report is long, and the panels above and below it - the plan, the progress
// cards, the follow-up box - end up separated by screenfuls of prose. This lists what is on
// the page and scrolls to the one picked, rather than leaving the reader to hunt for it.

import { $ } from "./dom.js";

// Panels worth jumping to, in the order they appear. A panel that is hidden is left out.
const PANELS = [
  ["#start", "New research"],
  ["#statusbar", "This session"],
  ["#cards", "Waiting for you"],
  ["#continue", "Continue this research"],
  ["#progress-panel", "Progress"],
  ["#plan-panel", "Plan"],
  ["#result-panel", "Result"],
];

const visible = el => !!el && !el.classList.contains("hidden") && el.getBoundingClientRect().height > 0;

/** Rebuild the menu from what is actually on the page. */
export function refreshOutline() {
  const menu = $("#jump");
  if (!menu) return;

  const options = [];
  for (const [sel, label] of PANELS) {
    if (visible($(sel))) options.push([sel, label]);
  }

  // The report's own sections, indented under it, so a long paper is navigable by heading.
  $("#report-body")?.querySelectorAll("h1, h2").forEach((h, i) => {
    if (!h.id) h.id = `report-h-${i}`;
    const text = h.textContent.trim();
    if (text) options.push([`#${h.id}`, (h.tagName === "H1" ? "  " : "    ") + text]);
  });

  menu.innerHTML = `<option value="">Jump to…</option>` +
    options.map(([v, t]) => `<option value="${v}">${t.replace(/</g, "&lt;")}</option>`).join("");
  menu.classList.toggle("hidden", options.length === 0);
}

export function initOutline() {
  const menu = $("#jump");
  if (!menu) return;
  menu.onchange = () => {
    const target = menu.value && document.querySelector(menu.value);
    menu.value = "";                       // the menu says where to go, it does not hold a state
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  refreshOutline();
}
