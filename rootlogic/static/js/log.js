// The action log: one line per event, colour-coded by family.
//
// It follows the newest line the way a tail does, and it does that inside its own box: the
// page never moves, so a reader partway down a report is not dragged back every time a
// sub-agent runs a search. Scrolling up inside the log stops it following - they are reading
// history - and scrolling back to the bottom starts it again.
//
// Following is tracked as a flag rather than measured at each new line. Measuring looked
// equivalent and was not: while the log's tab is hidden the box has no height, every
// comparison says "not at the bottom", and the log quietly stopped following for the rest of
// the session. That is how a live run ended up parked 9,000 pixels above its newest line.

import { $, esc, localTime } from "./dom.js";

const NEAR_BOTTOM = 30;   // px of slack, so a resting scroll position still counts as "at the end"

let following = true;
let watching = null;      // the box we attached the scroll listener to

function watch(box) {
  if (watching === box) return;
  watching = box;
  box.addEventListener("scroll", () => {
    following = box.scrollTop + box.clientHeight >= box.scrollHeight - NEAR_BOTTOM;
  }, { passive: true });
}

/** Start following again. Called when the view is reset for a new or reopened session. */
export function resetLog() {
  following = true;
}

export function logEvent(ev) {
  const box = $("#tab-log");
  if (!box) return;
  watch(box);

  const el = document.createElement("div");
  const group = ev.type.split(".")[0];
  el.className = `ev ${group}` + (/failed|error|aborted/.test(ev.type) ? " bad" : "");
  el.innerHTML = `<span class="ts">${esc(localTime(ev.ts))}</span>` +
                 `<span class="ty">${esc(ev.type)}</span><span class="msg"></span>`;
  $(".msg", el).textContent = ev.message ?? "";
  box.appendChild(el);

  // Only this box scrolls. scrollIntoView would take the page with it.
  if (following) box.scrollTop = box.scrollHeight;
}
