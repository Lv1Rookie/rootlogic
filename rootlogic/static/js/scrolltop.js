// The back-to-top button: hidden until there is something to go back up to.
//
// The action log and a finished report both run long, and the run controls live at the top of
// the page, so a reader deep in a report has a way back that is not the scrollbar.

import { $ } from "./dom.js";

const SHOW_AFTER = 320;    // px; roughly one screenful on a laptop

export function initScrollTop() {
  const btn = $("#to-top");
  if (!btn) return;

  const sync = () => btn.classList.toggle("show", window.scrollY > SHOW_AFTER);
  // passive: this listener never calls preventDefault, and saying so keeps scrolling smooth
  window.addEventListener("scroll", sync, { passive: true });
  sync();

  btn.onclick = () => window.scrollTo({ top: 0, behavior: "smooth" });
}
