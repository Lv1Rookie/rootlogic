// Showing and hiding the sidebar.
//
// Wide enough, it is a column beside the page and hiding it gives the report the width. Below
// the layout's breakpoint it is a drawer over the page: a single-column grid would otherwise
// append it under everything, so the settings would sit a whole report's worth of scrolling
// below the thing being read.
//
// It starts hidden on a narrow window and shown on a wide one, and remembers what the reader
// last chose at that width.

import { $ } from "./dom.js";

const KEY = "rootlogic-rail";
const WIDE = "(min-width: 901px)";

const wide = () => window.matchMedia(WIDE).matches;

function apply(hidden) {
  document.body.classList.toggle("rail-hidden", hidden);
  $("#rail-close")?.setAttribute("aria-expanded", String(!hidden));
  $("#rail-open")?.setAttribute("aria-expanded", String(!hidden));
  // The scrim only exists for the drawer: over a column there is nothing to dismiss.
  const scrim = $("#rail-scrim");
  if (scrim) scrim.hidden = hidden || wide();
}

function set(hidden) {
  apply(hidden);
  try { localStorage.setItem(KEY, hidden ? "hidden" : "shown"); } catch { /* private window */ }
}

export function initSidebar() {
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch { /* fall back to the width */ }
  apply(saved ? saved === "hidden" : !wide());

  $("#rail-close").onclick = () => set(true);
  $("#rail-open").onclick = () => set(false);
  $("#rail-scrim").onclick = () => set(true);      // tapping the page behind closes the drawer

  // Crossing the breakpoint changes what the sidebar is, so it changes what open means. A
  // column that was showing becomes a drawer over the page, which is not what the reader
  // asked for by shrinking the window, so narrowing closes it. Widening restores the choice
  // they made when it was last a column.
  window.matchMedia(WIDE).addEventListener("change", e => {
    if (!e.matches) return apply(true);
    let saved = null;
    try { saved = localStorage.getItem(KEY); } catch { /* fall back to showing it */ }
    apply(saved === "hidden");
  });

  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !wide() && !document.body.classList.contains("rail-hidden")) set(true);
  });
}
