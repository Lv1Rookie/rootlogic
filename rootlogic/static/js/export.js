// Taking the report out of the browser: a PDF, or the Markdown source.
//
// The PDF is produced by the browser's own print engine against print.css, which keeps the
// text selectable and searchable and the file small. A canvas-based PDF library would
// rasterise every page, and a server-side renderer would mean system libraries in a project
// that otherwise installs with pip alone.

import { $, esc } from "./dom.js";
import { state } from "./state.js";

let markdown = "";      // the report exactly as the agent wrote it
let meta = {};          // topic, session id, generated-at, model

/** Remember the report so it can be exported, and reveal the buttons. */
export function setReport(md, info = {}) {
  markdown = md || "";
  meta = info;
  const actions = $("#report-actions");
  if (actions) actions.classList.toggle("hidden", !markdown);
}

export function clearReport() {
  markdown = "";
  meta = {};
  const actions = $("#report-actions");
  if (actions) actions.classList.add("hidden");
  titleBlock().remove();
}

function titleBlock() {
  return $("#print-title") || Object.assign(document.createElement("div"), { id: "print-title" });
}

/** A title page the on-screen report doesn't need but a paper does. */
function addTitleBlock() {
  const el = titleBlock();
  el.className = "print-title";
  const when = new Date().toLocaleDateString(undefined,
    { year: "numeric", month: "long", day: "numeric" });
  el.innerHTML = `<h1>${esc(meta.topic || state.topic || "Research report")}</h1>
    <div class="byline">Prepared by rootlogic, an agentic research assistant</div>
    <div class="meta">${esc(when)}${meta.session ? " · session " + esc(meta.session) : ""}${
      meta.model ? " · " + esc(meta.model) : ""}</div>`;
  const body = $("#report-body");
  body.parentNode.insertBefore(el, body);
}

/** The report's own title, as it is printed: the heading the writer gave it, or the topic. */
function reportTitle() {
  const heading = $("#report-body h1")?.textContent.trim();
  return (heading || meta.topic || state.topic || "Research report").replace(/\s+/g, " ");
}

/** Ask the server for the PDF, so the browser offers to save a file like it does for .md. */
export async function savePdf() {
  if (!markdown || !meta.session) return;
  try {
    const support = await (await fetch("/api/pdf-support")).json();
    if (support.available) {
      // A plain link, so the browser's own Save panel opens - the same one .md gets.
      const a = Object.assign(document.createElement("a"), {
        href: `/api/sessions/${encodeURIComponent(meta.session)}/report.pdf`, download: "",
      });
      document.body.appendChild(a);
      a.click();
      a.remove();
      return;
    }
  } catch { /* server unreachable: fall through to the print engine */ }
  printPdf();   // the 'pdf' extra is not installed, so the browser renders it instead
}

function printPdf() {
  if (!markdown) return;
  addTitleBlock();
  // The browser takes the PDF's title, and the name it offers to save under, from the
  // document's own - which is "rootlogic", the name of the app rather than of the paper. It
  // is lent the report's title for the duration of the print and given its own back after.
  const pageTitle = document.title;
  const restore = () => { document.title = pageTitle; };
  document.title = reportTitle();
  window.addEventListener("afterprint", restore, { once: true });
  setTimeout(restore, 60_000);   // afterprint is not fired by every browser

  // The dialog is the browser's: "Save as PDF" is its own destination on every platform.
  window.print();
}

export function saveMarkdown() {
  if (!markdown) return;
  const name = (meta.topic || "report").toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 60);
  const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: `${name}.md` });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
