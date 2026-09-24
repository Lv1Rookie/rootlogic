// Give every checkbox the same drawn box, including the ones rendered mid-run.
//
// The plan review table and the override card's skip chips are built from event data, so a
// one-off pass at startup would miss them. An observer upgrades whatever appears later.
//
// The input itself is never replaced or moved out of its label: other modules find it with
// `$("input", row)` and read `.checked`, and those keep working because the element is the
// same one, only wrapped.

const MARK = `<div class="checkbox-wrapper"><div class="checkbox-bg"></div>
  <svg fill="none" viewBox="0 0 24 24" class="checkbox-icon" aria-hidden="true">
    <path stroke-linejoin="round" stroke-linecap="round" stroke-width="3" stroke="currentColor"
          d="M4 12L10 18L20 6" class="check-path"></path></svg></div>`;

// A component that draws its own control owns that input. The day/night switch is one: its
// scene flips on `input:checked ~ .switch-bg`, so moving the input inside a wrapper breaks
// the selector and the page would be left with a checkbox drawn on top of a switch.
const NOT_OURS = ".uiv-theme";

function upgrade(input) {
  if (input.dataset.uiv || input.closest(NOT_OURS)) return;
  input.dataset.uiv = "1";
  const shell = document.createElement("span");
  shell.className = "uiv-check";
  input.replaceWith(shell);
  shell.appendChild(input);          // same node, so existing handlers survive
  shell.insertAdjacentHTML("beforeend", MARK);
}

function upgradeAll(root = document) {
  root.querySelectorAll?.('input[type="checkbox"]:not([data-uiv])').forEach(upgrade);
}

export function initCheckboxes() {
  upgradeAll();
  new MutationObserver(records => {
    for (const r of records) for (const node of r.addedNodes) upgradeAll(node);
  }).observe(document.body, { childList: true, subtree: true });
}
