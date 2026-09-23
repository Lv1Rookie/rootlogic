// Tiny DOM helpers shared by every module. No framework, no build step.

export const $ = (s, el = document) => el.querySelector(s);

export const esc = s => String(s ?? "").replace(/[&<>"']/g, c => (
  {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

/** Markdown -> sanitised HTML. Falls back to escaped text if the CDN libs didn't load. */
export const md = text => {
  if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(text));
  return `<pre style="white-space:pre-wrap">${esc(text)}</pre>`;
};
