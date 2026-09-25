"""Render a finished report as a PDF the browser can simply save.

The browser's own print engine makes a better-looking document than this does, but it makes it
through the print dialog: the reader has to find "Save as PDF" in a printer menu, and on a Mac
that menu opens pointing at whatever printer they last used. Downloading the Markdown, next to
it, opens the ordinary Save panel. Two buttons that both say "download" should not behave that
differently, so the PDF is built here and served as a file.

Pure Python on purpose - xhtml2pdf and markdown are pip-installable with no system libraries,
which keeps `pip install -e '.[web]'` enough to run everything. A headless browser would
render more faithfully and would also mean shipping a browser.
"""

from __future__ import annotations

import io
from datetime import date

PAGE_CSS = """
@page { size: A4; margin: 22mm 20mm 20mm; }
body { font-family: Times; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; margin: 0 0 2mm; }
h2 { font-size: 12.5pt; margin: 7mm 0 2mm; border-bottom: 0.4pt solid #999; padding-bottom: 1mm; }
h3 { font-size: 11pt; margin: 5mm 0 1.5mm; }
p, li { font-size: 10.5pt; }
a { color: #14425c; }
table { width: 100%; margin: 3mm 0; }   /* xhtml2pdf has no border-collapse */
th { background: #eee; font-size: 8.5pt; text-align: left; padding: 1.5mm; border: 0.4pt solid #bbb; }
td { font-size: 8.5pt; padding: 1.5mm; border: 0.4pt solid #ccc; }
.title-block { margin-bottom: 9mm; border-bottom: 0.8pt solid #333; padding-bottom: 4mm; }
.title-block h1 { font-size: 20pt; }
.byline { font-size: 9.5pt; font-style: italic; color: #444; }
.stamp { font-size: 8pt; color: #555; }
"""


class PdfUnavailable(RuntimeError):
    """The optional dependencies are not installed. The caller falls back to printing."""


def available() -> bool:
    try:
        import markdown  # noqa: F401
        from xhtml2pdf import pisa  # noqa: F401
    except ImportError:
        return False
    return True


def render(report_markdown: str, *, title: str, session: str, today: date | None = None) -> bytes:
    """A PDF of ``report_markdown``, with the same title page the print stylesheet gives it."""
    try:
        import markdown
        from xhtml2pdf import pisa
    except ImportError as e:  # pragma: no cover - exercised by the endpoint's fallback
        raise PdfUnavailable("install the 'pdf' extra to download PDFs") from e

    # The report's own H1 becomes the title block, so it is not repeated in the body.
    body = report_markdown
    if body.startswith("# "):
        body = body.split("\n", 1)[1] if "\n" in body else ""

    when = (today or date.today()).strftime("%d %B %Y")
    html = markdown.markdown(body, extensions=["tables", "sane_lists"])
    page = (f"<html><head><meta charset='utf-8'><style>{PAGE_CSS}</style></head><body>"
            f"<div class='title-block'><h1>{_escape(title)}</h1>"
            f"<div class='byline'>Prepared by rootlogic, an agentic research assistant</div>"
            f"<div class='stamp'>{when} &middot; session {_escape(session)}</div></div>"
            f"{html}</body></html>")

    out = io.BytesIO()
    result = pisa.CreatePDF(page, dest=out, encoding="utf-8")
    if result.err:
        raise PdfUnavailable(f"the PDF renderer reported {result.err} error(s)")
    return out.getvalue()


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
