from __future__ import annotations

import hashlib
import html
import mimetypes
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .document import build_document


_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_LIST_RE = re.compile(r"^\s*(?:(?P<bullet>[-+*])|(?P<number>\d+)[.)])\s+(?P<text>.*)$")
_INLINE_TOKEN_RE = re.compile(r"(!?\[[^\]]*\]\([^)]+\)|`[^`]*`|(?<!\\)\$[^$\n]+(?<!\\)\$)")
_LINK_RE = re.compile(r"^(!?)\[([^\]]*)\]\(([^)]+)\)$")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_PIPE_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def build_markdown_document(
    lines: list[str],
    *,
    paper_id: str,
    source_path: Path,
    source_hash: str,
    sections: list[dict[str, Any]],
    equations: list[dict[str, Any]],
    pdf_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the shared rich-document contract from Markdown source.

    Markdown is first represented as conservative semantic HTML and then sent
    through the same document builder used for ar5iv and local HTML.  The
    representation keeps source-line spans and treats extracted display math
    as authoritative, so the compatibility equation view and rich view retain
    the same identities and PDF anchors.
    """

    rendered = _markdown_semantic_html(
        lines,
        source_path=source_path,
        sections=sections,
        equations=equations,
    )
    provider = "markdown-pdf" if pdf_path else "markdown"
    markdown_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    pdf_sha256 = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest() if pdf_path else ""
    return build_document(
        rendered,
        paper_id=paper_id,
        assets=_markdown_assets(lines, source_path=source_path),
        equations=equations,
        source_metadata={
            "provider": provider,
            "format": "markdown",
            "path": str(source_path),
            "source_sha256": markdown_sha256,
            "input_hash": source_hash,
            "pdf_path": str(pdf_path) if pdf_path else "",
            "pdf_sha256": pdf_sha256,
        },
    )


def _markdown_semantic_html(
    lines: list[str],
    *,
    source_path: Path,
    sections: list[dict[str, Any]],
    equations: list[dict[str, Any]],
) -> str:
    equation_by_start = {
        int(item["markdown_line_start"]): item
        for item in equations
        if isinstance(item.get("markdown_line_start"), int)
    }
    section_by_start = {
        int(item["markdown_line_start"]): item
        for item in sections
        if isinstance(item.get("markdown_line_start"), int)
    }
    has_source_heading = any(_HEADING_RE.match(line) for line in lines)
    parts = ['<article class="arc_markdown_document">']
    section_open = False
    if sections and not has_source_heading:
        parts.append(_section_open(sections[0], source_path=source_path))
        section_open = True

    index = 0
    while index < len(lines):
        line_number = index + 1
        equation = equation_by_start.get(line_number)
        if equation is not None:
            parts.append(_display_equation(equation, source_path=source_path))
            index = int(equation.get("markdown_line_end") or line_number)
            continue

        heading = _HEADING_RE.match(lines[index])
        if heading:
            if section_open:
                parts.append("</section>")
            section = section_by_start.get(line_number) or {
                "section_id": f"sec-line-{line_number}",
                "title": heading.group(2),
                "level": len(heading.group(1)),
            }
            parts.append(_section_open(section, source_path=source_path))
            section_open = True
            title = _strip_heading_markup(heading.group(2))
            level = min(6, max(1, len(heading.group(1))))
            parts.append(
                f'<h{level} id="{html.escape(str(section["section_id"]) + ".heading", quote=True)}"'
                f'{_line_attributes(source_path, line_number, line_number)}>{_inline_html(title)}</h{level}>'
            )
            index += 1
            continue

        fence = _FENCE_RE.match(lines[index])
        if fence:
            marker = fence.group(1)
            end = index + 1
            while end < len(lines) and not lines[end].lstrip().startswith(marker):
                end += 1
            content_end = min(end, len(lines))
            content = "\n".join(lines[index + 1 : content_end])
            end_line = min(end + 1, len(lines))
            parts.append(
                f'<pre id="md-line-{line_number}"{_line_attributes(source_path, line_number, end_line)}>'
                f"{html.escape(content)}</pre>"
            )
            index = min(end + 1, len(lines))
            continue

        if _IMAGE_RE.search(lines[index]):
            parts.extend(
                _image_line_blocks(
                    lines[index], source_path=source_path, line_number=line_number
                )
            )
            index += 1
            continue

        if index + 1 < len(lines) and _is_pipe_table(lines[index], lines[index + 1]):
            start = index
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and _is_pipe_row(lines[index]):
                table_lines.append(lines[index])
                index += 1
            parts.append(
                _pipe_table_html(
                    table_lines,
                    source_path=source_path,
                    line_start=start + 1,
                    line_end=index,
                )
            )
            continue

        list_match = _LIST_RE.match(lines[index])
        if list_match:
            ordered = list_match.group("number") is not None
            start = index
            items: list[tuple[int, re.Match[str]]] = []
            while index < len(lines):
                current = _LIST_RE.match(lines[index])
                if current is None or (current.group("number") is not None) != ordered:
                    break
                items.append((index + 1, current))
                index += 1
            tag = "ol" if ordered else "ul"
            start_attribute = (
                f' start="{html.escape(items[0][1].group("number") or "1", quote=True)}"'
                if ordered
                else ""
            )
            parts.append(
                f'<{tag} id="md-line-{start + 1}"{start_attribute}'
                f'{_line_attributes(source_path, start + 1, index)}>'
            )
            for item_line, item in items:
                parts.append(
                    f'<li id="md-line-{item_line}-item"{_line_attributes(source_path, item_line, item_line)}>'
                    f'{_inline_html(item.group("text"))}</li>'
                )
            parts.append(f"</{tag}>")
            continue

        if not lines[index].strip():
            index += 1
            continue

        start = index
        paragraph: list[str] = []
        while index < len(lines):
            current_line = index + 1
            if not lines[index].strip():
                break
            if current_line in equation_by_start or _HEADING_RE.match(lines[index]):
                break
            if _FENCE_RE.match(lines[index]) or _LIST_RE.match(lines[index]):
                break
            if _IMAGE_RE.search(lines[index]):
                break
            if index + 1 < len(lines) and _is_pipe_table(lines[index], lines[index + 1]):
                break
            paragraph.append(lines[index].strip())
            index += 1
        if paragraph:
            text = "\n".join(paragraph)
            parts.append(
                f'<p id="md-line-{start + 1}"{_line_attributes(source_path, start + 1, index)}>'
                f"{_inline_html(text)}</p>"
            )
        else:
            index += 1

    if section_open:
        parts.append("</section>")
    parts.append("</article>")
    return "\n".join(parts)


def _section_open(section: dict[str, Any], *, source_path: Path) -> str:
    section_id = html.escape(str(section.get("section_id") or "section"), quote=True)
    start = int(section.get("markdown_line_start") or 1)
    end = int(section.get("markdown_line_end") or start)
    return f'<section id="{section_id}"{_line_attributes(source_path, start, end)}>'


def _display_equation(equation: dict[str, Any], *, source_path: Path) -> str:
    equation_id = html.escape(str(equation.get("id") or "equation"), quote=True)
    start = int(equation.get("markdown_line_start") or 1)
    end = int(equation.get("markdown_line_end") or start)
    tex = str(equation.get("normalized_latex") or equation.get("equation") or "")
    number = str(equation.get("printed_equation_number") or "")
    number_cell = (
        f'<td class="ltx_eqn_eqno">({html.escape(number)})</td>' if number else ""
    )
    escaped_tex = html.escape(tex)
    return (
        f'<table class="ltx_equation" id="{equation_id}"'
        f'{_line_attributes(source_path, start, end)}><tr><td>'
        f'<math alttext="{html.escape(tex, quote=True)}"><semantics>'
        f'<annotation encoding="application/x-tex">{escaped_tex}</annotation>'
        f"</semantics></math></td>{number_cell}</tr></table>"
    )


def _line_attributes(source_path: Path, start: int, end: int) -> str:
    return (
        f' data-source-format="markdown" data-source-path="{html.escape(str(source_path), quote=True)}"'
        f' data-source-line-start="{start}" data-source-line-end="{end}"'
    )


def _strip_heading_markup(value: str) -> str:
    return re.sub(r"[*_`~]", "", value).strip()


def _inline_html(value: str) -> str:
    rendered: list[str] = []
    position = 0
    for match in _INLINE_TOKEN_RE.finditer(value):
        rendered.append(html.escape(value[position : match.start()]))
        token = match.group(0)
        link = _LINK_RE.match(token)
        if link:
            image_marker, label, href = link.groups()
            if image_marker:
                rendered.append(
                    f'<img class="arc_markdown_image" src="{html.escape(href, quote=True)}"'
                    f' alt="{html.escape(label, quote=True)}"/>'
                )
            elif href.startswith("#"):
                rendered.append(
                    f'<span class="arc_markdown_internal_link" data-href="{html.escape(href, quote=True)}">'
                    f"{html.escape(label)}</span>"
                )
            else:
                rendered.append(
                    f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'
                )
        elif token.startswith("`"):
            rendered.append(f"<code>{html.escape(token[1:-1])}</code>")
        else:
            tex = token[1:-1]
            rendered.append(
                f'<math alttext="{html.escape(tex, quote=True)}"><semantics>'
                f'<annotation encoding="application/x-tex">{html.escape(tex)}</annotation>'
                f"</semantics></math>"
            )
        position = match.end()
    rendered.append(html.escape(value[position:]))
    return "".join(rendered).replace("\n", "<br/>")


def _image_line_blocks(line: str, *, source_path: Path, line_number: int) -> list[str]:
    """Render Markdown images as first-class figure blocks without losing adjacent text."""

    parts: list[str] = []
    position = 0
    for image_index, match in enumerate(_IMAGE_RE.finditer(line), start=1):
        before = line[position : match.start()].strip()
        if before:
            parts.append(
                f'<p id="md-line-{line_number}-before-image-{image_index}"'
                f'{_line_attributes(source_path, line_number, line_number)}>{_inline_html(before)}</p>'
            )
        label, href = match.groups()
        figure_id = f"md-image-line-{line_number}-{image_index}"
        caption = f"<figcaption>{html.escape(label)}</figcaption>" if label else ""
        parts.append(
            f'<figure class="ltx_figure arc_markdown_figure" id="{figure_id}"'
            f'{_line_attributes(source_path, line_number, line_number)}>'
            f'<img src="{html.escape(href, quote=True)}" alt="{html.escape(label, quote=True)}"/>'
            f"{caption}</figure>"
        )
        position = match.end()
    after = line[position:].strip()
    if after:
        parts.append(
            f'<p id="md-line-{line_number}-after-image"'
            f'{_line_attributes(source_path, line_number, line_number)}>{_inline_html(after)}</p>'
        )
    return parts


def _markdown_assets(lines: list[str], *, source_path: Path) -> list[dict[str, Any]]:
    """Describe local Markdown image files in the shared asset contract.

    The original Markdown URL is retained verbatim while local relative paths
    are resolved against the Markdown file, never the process working directory.
    Missing or remote assets remain explicit records so integrity cannot report
    a complete/renderable document after silently dropping an image.
    """

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        for match in _IMAGE_RE.finditer(line):
            label, href = match.groups()
            href = href.strip()
            split = urlsplit(href)
            local_path: Path | None = None
            if split.scheme == "file":
                local_path = Path(unquote(split.path))
            elif not split.scheme and not split.netloc:
                raw_path = Path(unquote(split.path))
                local_path = raw_path if raw_path.is_absolute() else source_path.parent / raw_path
            record: dict[str, Any] = {
                "asset_id": "",
                "source_url": href,
                "original_url": href,
                "relative_path": href,
                "media_type": mimetypes.guess_type(split.path)[0] or "",
                "sha256": "",
                "bytes": 0,
                "cache_path": "",
                "status": "missing",
                "alt": label,
                "source_span": {
                    "format": "markdown",
                    "path": str(source_path),
                    "line_start": line_number,
                    "line_end": line_number,
                },
            }
            if local_path is not None and local_path.is_file():
                data = local_path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                record.update(
                    {
                        "asset_id": f"sha256:{digest}",
                        "sha256": digest,
                        "bytes": len(data),
                        "cache_path": str(local_path.resolve()),
                        "status": "cached",
                    }
                )
            records.append(record)
    return records


def _is_pipe_table(header: str, separator: str) -> bool:
    return _is_pipe_row(header) and bool(_PIPE_TABLE_SEPARATOR_RE.match(separator))


def _is_pipe_row(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and not _PIPE_TABLE_SEPARATOR_RE.match(stripped)


def _pipe_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]
    return [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def _pipe_table_html(
    lines: list[str], *, source_path: Path, line_start: int, line_end: int
) -> str:
    headers = _pipe_cells(lines[0])
    rows = [_pipe_cells(line) for line in lines[2:]]
    table_id = f"md-table-line-{line_start}"
    parts = [
        f'<table class="ltx_table arc_markdown_table" id="{table_id}"'
        f'{_line_attributes(source_path, line_start, line_end)}><thead><tr>'
    ]
    parts.extend(f"<th>{_inline_html(cell)}</th>" for cell in headers)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{_inline_html(cell)}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)
