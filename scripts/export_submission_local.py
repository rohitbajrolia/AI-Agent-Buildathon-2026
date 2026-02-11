from __future__ import annotations

import html
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


ROOT_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT_DIR / "dist"


@dataclass(frozen=True)
class DocSpec:
    in_file: str
    out_base: str


DOCS: list[DocSpec] = [
    DocSpec("BUILDATHON_SUBMISSION.md", "Coverage_Concierge_Proposal"),
    DocSpec("PROJECT_BRIEF.md", "Coverage_Concierge_Project_Brief"),
    DocSpec("PITCH_90_SECONDS.md", "Coverage_Concierge_90_Second_Pitch"),
]


@dataclass(frozen=True)
class Block:
    kind: str
    text: str


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")


def parse_markdown(md: str) -> list[Block]:
    blocks: list[Block] = []
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def flush_paragraph(paragraph_lines: list[str]) -> None:
        text = " ".join(line.strip() for line in paragraph_lines).strip()
        if text:
            blocks.append(Block("p", text))

    paragraph_lines: list[str] = []
    in_code = False

    for raw in lines:
        line = raw.rstrip("\n")

        if line.strip().startswith("```"):
            if paragraph_lines:
                flush_paragraph(paragraph_lines)
                paragraph_lines = []
            in_code = not in_code
            continue

        if in_code:
            # Keep code blocks as preformatted paragraphs (text-only export).
            if line.strip() == "":
                blocks.append(Block("code", ""))
            else:
                blocks.append(Block("code", line))
            continue

        if line.strip() == "":
            if paragraph_lines:
                flush_paragraph(paragraph_lines)
                paragraph_lines = []
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            if paragraph_lines:
                flush_paragraph(paragraph_lines)
                paragraph_lines = []
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            blocks.append(Block(f"h{level}", text))
            continue

        bullet_match = _BULLET_RE.match(line)
        if bullet_match:
            if paragraph_lines:
                flush_paragraph(paragraph_lines)
                paragraph_lines = []
            blocks.append(Block("li", bullet_match.group(1).strip()))
            continue

        paragraph_lines.append(line)

    if paragraph_lines:
        flush_paragraph(paragraph_lines)

    # Collapse consecutive code lines into a single block for better readability.
    collapsed: list[Block] = []
    code_buf: list[str] = []
    for block in blocks:
        if block.kind == "code":
            code_buf.append(block.text)
            continue
        if code_buf:
            collapsed.append(Block("codeblock", "\n".join(code_buf).rstrip("\n")))
            code_buf = []
        collapsed.append(block)
    if code_buf:
        collapsed.append(Block("codeblock", "\n".join(code_buf).rstrip("\n")))

    return collapsed


def blocks_to_plain_text(blocks: Iterable[Block]) -> str:
    out: list[str] = []
    for b in blocks:
        if b.kind.startswith("h") and b.kind[1:].isdigit():
            out.append(b.text)
            out.append("")
        elif b.kind == "p":
            out.append(b.text)
            out.append("")
        elif b.kind == "li":
            out.append(f"- {b.text}")
        elif b.kind == "codeblock":
            out.append(b.text)
            out.append("")
        else:
            out.append(b.text)
    return "\n".join(out).strip() + "\n"


def render_html(title: str, blocks: Iterable[Block]) -> str:
    css = """
:root {
  --bg: #ffffff;
  --fg: #111827;
  --muted: #6b7280;
  --rule: #e5e7eb;
  --code-bg: #f3f4f6;
  --code-fg: #111827;
  --link: #2563eb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b0b0c;
    --fg: #e5e7eb;
    --muted: #9ca3af;
    --rule: #1f2937;
    --code-bg: #111827;
    --code-fg: #e5e7eb;
    --link: #60a5fa;
  }
}
body {
  background: var(--bg);
  color: var(--fg);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  margin: 40px auto;
  padding: 0 24px;
  max-width: 900px;
}
h1, h2, h3, h4 {
  line-height: 1.25;
  margin: 1.2em 0 0.5em;
}
hr {
  border: 0;
  border-top: 1px solid var(--rule);
  margin: 24px 0;
}
pre {
  background: var(--code-bg);
  color: var(--code-fg);
  padding: 12px 14px;
  border-radius: 10px;
  overflow-x: auto;
}
code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
.small { color: var(--muted); font-size: 12px; }
ul { margin: 0.25em 0 0.75em 1.25em; }
li { margin: 0.15em 0; }
""".strip()

    def esc(s: str) -> str:
        return html.escape(s, quote=True)

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append(f"<title>{esc(title)}</title>")
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<style>{css}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append(f"<h1>{esc(title)}</h1>")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"<p class=\"small\">Generated: {esc(generated_at)}</p>")
    parts.append("<hr>")

    in_ul = False
    for b in blocks:
        if b.kind == "li":
            if not in_ul:
                parts.append("<ul>")
                in_ul = True
            parts.append(f"<li>{esc(b.text)}</li>")
            continue
        if in_ul:
            parts.append("</ul>")
            in_ul = False

        if b.kind.startswith("h") and b.kind[1:].isdigit():
            level = int(b.kind[1:])
            level = min(max(level, 2), 4)  # keep headings compact
            parts.append(f"<h{level}>{esc(b.text)}</h{level}>")
        elif b.kind == "p":
            parts.append(f"<p>{esc(b.text)}</p>")
        elif b.kind == "codeblock":
            parts.append("<pre><code>")
            parts.append(esc(b.text))
            parts.append("</code></pre>")
        else:
            parts.append(f"<p>{esc(b.text)}</p>")

    if in_ul:
        parts.append("</ul>")

    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


def _xml_escape(text: str) -> str:
    # WordprocessingML requires XML escaping. We also preserve leading/trailing
    # spaces via xml:space when needed.
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _w_t(text: str) -> str:
    escaped = _xml_escape(text)
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        return f'<w:t xml:space="preserve">{escaped}</w:t>'
    return f"<w:t>{escaped}</w:t>"


def render_docx(title: str, blocks: Iterable[Block], out_path: Path) -> None:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def paragraph(text: str, *, bold: bool = False, size_half_points: int = 22) -> str:
        rpr = [f'<w:sz w:val="{size_half_points}"/>', f'<w:szCs w:val="{size_half_points}"/>']
        if bold:
            rpr.insert(0, "<w:b/>")
        rpr_xml = "".join(rpr)
        return (
            "<w:p>"
            "<w:r>"
            f"<w:rPr>{rpr_xml}</w:rPr>"
            f"{_w_t(text)}"
            "</w:r>"
            "</w:p>"
        )

    body_parts: list[str] = []
    body_parts.append(paragraph(title, bold=True, size_half_points=36))
    body_parts.append(paragraph("Generated: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), size_half_points=18))
    body_parts.append(paragraph(""))

    for b in blocks:
        if b.kind.startswith("h") and b.kind[1:].isdigit():
            level = int(b.kind[1:])
            size = 30 if level <= 2 else 26 if level == 3 else 24
            body_parts.append(paragraph(b.text, bold=True, size_half_points=size))
        elif b.kind == "p":
            body_parts.append(paragraph(b.text))
            body_parts.append(paragraph(""))
        elif b.kind == "li":
            body_parts.append(paragraph("• " + b.text))
        elif b.kind == "codeblock":
            for line in b.text.split("\n"):
                body_parts.append(paragraph(line, size_half_points=18))
            body_parts.append(paragraph(""))
        else:
            body_parts.append(paragraph(b.text))

    # Basic section properties (Letter).
    sect_pr = (
        "<w:sectPr>"
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
    )

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<w:body>"
        + "".join(body_parts)
        + sect_pr
        + "</w:body>"
        "</w:document>"
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        "</Types>"
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )

    doc_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"></Relationships>'
    )

    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{_xml_escape(title)}</dc:title>"
        "<dc:creator>Coverage Concierge</dc:creator>"
        "<cp:lastModifiedBy>Coverage Concierge</cp:lastModifiedBy>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{now_iso}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{now_iso}</dcterms:modified>"
        "</cp:coreProperties>"
    )

    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Coverage Concierge</Application>"
        "</Properties>"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", doc_rels_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def render_pdf(title: str, blocks: Iterable[Block], out_path: Path) -> None:
    # Text-only PDF with basic pagination. This is intentionally simple and
    # dependency-free, but is good enough to be opened and submitted.
    plain = blocks_to_plain_text(blocks)

    wrapped_lines: list[str] = []
    for line in plain.splitlines():
        if line.strip() == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=95, break_long_words=False, replace_whitespace=False))

    lines_per_page = 52
    pages: list[list[str]] = [
        wrapped_lines[i : i + lines_per_page] for i in range(0, len(wrapped_lines), lines_per_page)
    ]
    if not pages:
        pages = [[""]]

    # Build PDF objects with stable object numbers:
    # 1) Catalog
    # 2) Pages
    # 3) Font
    # 4..) Content + Page pairs
    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    # 1) Catalog (references Pages at 2 0 R)
    add_object(b"<< /Type /Catalog /Pages 2 0 R >>")

    # 2) Pages placeholder (we'll fill kids after building page objects)
    add_object(b"<< /Type /Pages /Count 0 /Kids [] >>")

    # 3) Font
    font_obj_num = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_nums: list[int] = []

    for page_lines in pages:
        text_lines = []
        text_lines.append("BT")
        text_lines.append("/F1 12 Tf")
        text_lines.append("72 740 Td")
        leading = 14
        for idx, line in enumerate(page_lines):
            if idx > 0:
                text_lines.append(f"0 -{leading} Td")
            if line == "":
                continue
            text_lines.append(f"({_pdf_escape(line)}) Tj")
        text_lines.append("ET")
        stream = ("\n".join(text_lines) + "\n").encode("utf-8")
        content_obj = (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"endstream"
        )
        content_num = add_object(content_obj)

        # Page size: US Letter 612x792
        page_stub = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
            f"/Contents {content_num} 0 R >>"
        ).encode("ascii")
        page_num = add_object(page_stub)
        page_obj_nums.append(page_num)

    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    pages_obj = f"<< /Type /Pages /Count {len(page_obj_nums)} /Kids [{kids}] >>".encode("ascii")
    objects[1] = pages_obj

    final_objects = objects

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets: list[int] = [0]
    body = bytearray()

    def write_obj(obj_num: int, data: bytes) -> None:
        offsets.append(len(header) + len(body))
        body.extend(f"{obj_num} 0 obj\n".encode("ascii"))
        body.extend(data)
        body.extend(b"\nendobj\n")

    for i, data in enumerate(final_objects, start=1):
        write_obj(i, data)

    xref_start = len(header) + len(body)
    xref = bytearray()
    xref.extend(f"xref\n0 {len(final_objects) + 1}\n".encode("ascii"))
    xref.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        xref.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    trailer = (
        "trailer\n"
        f"<< /Size {len(final_objects) + 1} /Root 1 0 R >>\n"
        "startxref\n"
        f"{xref_start}\n"
        "%%EOF\n"
    ).encode("ascii")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + body + xref + trailer)


def export_one(spec: DocSpec) -> None:
    in_path = ROOT_DIR / spec.in_file
    if not in_path.exists():
        return

    md = in_path.read_text(encoding="utf-8")
    blocks = parse_markdown(md)

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    html_out = DIST_DIR / f"{spec.out_base}.html"
    docx_out = DIST_DIR / f"{spec.out_base}.docx"
    pdf_out = DIST_DIR / f"{spec.out_base}.pdf"

    title = spec.out_base.replace("_", " ")

    html_out.write_text(render_html(title=title, blocks=blocks), encoding="utf-8")
    render_docx(title=title, blocks=blocks, out_path=docx_out)
    render_pdf(title=title, blocks=blocks, out_path=pdf_out)


def main() -> int:
    for spec in DOCS:
        export_one(spec)

    notes = DIST_DIR / "EXPORT_NOTES.txt"
    notes.write_text(
        "Exports generated without external dependencies.\n"
        "- .docx: open in Word/Google Docs\n"
        "- .pdf: text-only, suitable for submission\n"
        "- .html: use browser Print -> Save as PDF for a formatted PDF\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
