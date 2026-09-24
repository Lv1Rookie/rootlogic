// Light/dark toggle.
//
// The stylesheet already follows the system setting; this only exists for the reader whose
// system setting is wrong for the room they are in. It sets `data-theme` on <html>, which the
// stylesheet honours over the media query, and remembers the choice per browser.
//
// Nothing else depends on it: a browser with storage disabled just gets the system theme.

import { $ } from "./dom.js";

const KEY = "rootlogic-theme";

function systemTheme() {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function apply(theme) {
  document.documentElement.dataset.theme = theme;
  // The switch shows a daytime scene unchecked and a night one checked, so "checked" is dark.
  const box = $("#theme");
  if (box) {
    box.checked = theme === "dark";
    box.closest(".uiv-theme")?.setAttribute("aria-label",
      theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
  }
}

export function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch { /* private window: use the system theme */ }
  apply(saved || systemTheme());

  const box = $("#theme");
  if (!box) return;
  box.onchange = () => {
    const next = box.checked ? "dark" : "light";
    apply(next);
    try { localStorage.setItem(KEY, next); } catch { /* not worth failing a click over */ }
  };
}
