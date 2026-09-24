// Tiny DOM helpers shared by every module. No framework, no build step.

export const $ = (s, el = document) => el.querySelector(s);

export const esc = s => String(s ?? "").replace(/[&<>"']/g, c => (
  {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

/** Markdown -> sanitised HTML. Falls back to escaped text if the CDN libs didn't load. */
export const md = text => {
  if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(text));
  return `<pre style="white-space:pre-wrap">${esc(text)}</pre>`;
};

/* ---------------------------------------------------------------- timestamps
 * The store writes UTC (`datetime.now(timezone.utc).isoformat()`), which is right for a log
 * that may be read anywhere, but a reader wants their own clock. These parse the stored
 * string and render it in the browser's timezone, so a run at 13:14 in New York reads 13:14
 * rather than 17:14. Rows written before the timezone suffix existed are assumed to be UTC,
 * which is what they were.
 */
const asDate = ts => new Date(/[Z+]|-\d\d:\d\d$/.test(ts || "") ? ts : ts + "Z");

/** Clock time for one log line, e.g. "13:14:20". */
export const localTime = ts => {
  const d = asDate(ts);
  return isNaN(d) ? String(ts ?? "").slice(11, 19)
                  : d.toLocaleTimeString([], { hour12: false });
};

/** Date and time for a history row, e.g. "2026-09-24 13:14". */
export const localStamp = ts => {
  const d = asDate(ts);
  if (isNaN(d)) return String(ts ?? "").slice(0, 16).replace("T", " ");
  const p = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
         `${p(d.getHours())}:${p(d.getMinutes())}`;
};

/** The reader's timezone abbreviation, e.g. "EST" - shown once, not on every row. */
export const localZone = () => {
  try {
    return new Intl.DateTimeFormat([], { timeZoneName: "short" })
      .formatToParts(new Date()).find(p => p.type === "timeZoneName")?.value || "";
  } catch { return ""; }
};
