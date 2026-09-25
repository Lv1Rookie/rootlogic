// Showing and hiding the sidebar.
//
// It is a column at every width, never a sheet over the page and never a slab appended under
// it: both stay readable, and the reader folds away whichever one they are not using. A
// narrow window only makes the column narrower, and starts it folded so the page has room.

import { $ } from "./dom.js";

const KEY = "rootlogic-rail";
const WIDE = "(min-width: 901px)";

const wide = () => window.matchMedia(WIDE).matches;

function apply(hidden) {
  document.body.classList.toggle("rail-hidden", hidden);
  $("#rail-close")?.setAttribute("aria-expanded", String(!hidden));
  $("#rail-open")?.setAttribute("aria-expanded", String(!hidden));
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

  // Narrowing the window takes room away from both columns, so the sidebar gets out of the
  // way; widening gives back whatever was last chosen. Neither writes to the saved choice -
  // resizing a window is not the reader changing their mind.
  window.matchMedia(WIDE).addEventListener("change", e => {
    if (!e.matches) return apply(true);
    let saved = null;
    try { saved = localStorage.getItem(KEY); } catch { /* fall back to showing it */ }
    apply(saved === "hidden");
  });
}
