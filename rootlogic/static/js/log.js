// The action log: one line per event, colour-coded by family, auto-scrolling only when the
// reader is already at the bottom (scrolling up to read history shouldn't be yanked back).

import { $, esc, localTime } from "./dom.js";

export function logEvent(ev) {
  const el = document.createElement("div");
  const group = ev.type.split(".")[0];
  el.className = `ev ${group}` + (/failed|error|aborted/.test(ev.type) ? " bad" : "");
  el.innerHTML = `<span class="ts">${esc(localTime(ev.ts))}</span>` +
                 `<span class="ty">${esc(ev.type)}</span><span class="msg"></span>`;
  $(".msg", el).textContent = ev.message ?? "";
  const box = $("#tab-log");
  const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
  box.appendChild(el);
  if (stick) box.scrollTop = box.scrollHeight;
}
