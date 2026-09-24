// Give every .btn the glass treatment, including the ones rendered mid-run.
//
// The component needs three elements per button - a wrapper, the button with its label in a
// span, and a shadow layer - and the request cards, the override chips and the history rows
// are all built from event data, so markup written into index.html would only cover half of
// them. The same observer pattern as the checkboxes does it once, wherever they appear.
//
// The button element is wrapped, never replaced: `$("#pause").onclick`, `classList.toggle`
// and `.textContent` reads elsewhere all keep pointing at the same node.

const DONE = "uivBtn";

function wrap(btn) {
  if (btn.dataset[DONE]) return;
  btn.dataset[DONE] = "1";

  // The label moves into a span, which the sheen and the press animation are drawn against.
  const span = document.createElement("span");
  while (btn.firstChild) span.appendChild(btn.firstChild);
  btn.appendChild(span);

  const shell = document.createElement("span");
  shell.className = "uiv-btn-wrap";
  btn.replaceWith(shell);
  shell.appendChild(btn);
  shell.insertAdjacentHTML("beforeend", '<span class="uiv-btn-shadow"></span>');
}

function wrapAll(root = document) {
  root.querySelectorAll?.(`.btn:not([data-${"uiv-btn"}])`).forEach(b => {
    if (!b.dataset[DONE]) wrap(b);
  });
}

export function initButtons() {
  wrapAll();
  new MutationObserver(records => {
    for (const r of records) for (const node of r.addedNodes) wrapAll(node);
  }).observe(document.body, { childList: true, subtree: true });
}
