from __future__ import annotations

import hashlib
import random
import re
import subprocess
from pathlib import Path
from typing import Any

from .ar5iv_html import parse_html
from .markdown_document import build_markdown_document
from .pdf_structure import parse_index_entries, read_embedded_outline
from .structure import build_structure, empty_index_entries, normalize_document_kind


PARSER_VERSION = 21
PDF_PAGE_MATCH_MIN_SCORE = 8
DISPLAY_ENVIRONMENTS = ("equation", "align", "gather", "multline", "eqnarray")
SECTION_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
EQUATION_NUMBER_PATTERN = r"[A-Za-z]?\d+(?:\.\d+)+|\d+(?:\.\d+)*[A-Za-z]?"
# Printed equation labels in scanned books are frequently OCR'd with spaces
# around dots or between digits, e.g. ``( 1.1 . 7)``.  Keep the canonical
# source-number grammar strict, but accept this layout variant when reading
# the PDF text layer and normalize it before matching source tags.
PRINTED_EQUATION_NUMBER_PATTERN = r"(?:[A-Za-z]?\s*\d+(?:\s*\.\s*\d+)+|\s*\d+(?:\s*\.\s*\d+)*[A-Za-z]?)\s*"
MATH_LIKE_CHARS = set("=+-*/^_<>≤≥∝⇒≡−•·√∂∇πρφΩωηε")
GREEK_TOKEN_NAMES = {
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b4": "delta",
    "\u03b5": "epsilon",
    "\u03b7": "eta",
    "\u03b8": "theta",
    "\u03bb": "lambda",
    "\u03bc": "mu",
    "\u03bd": "nu",
    "\u03c0": "pi",
    "\u03c1": "rho",
    "\u03c3": "sigma",
    "\u03c4": "tau",
    "\u03c6": "phi",
    "\u03c8": "psi",
    "\u03c9": "omega",
    "\u0394": "delta",
    "\u039b": "lambda",
    "\u03a9": "omega",
}


def parse_source_input(
    *,
    source_path: str | Path | None = None,
    source_id: str | None = None,
    html_path: str | Path | None = None,
    html_text: str | None = None,
    tex_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    include_document: bool = True,
    source_url: str = "",
    assets: list[dict[str, Any]] | None = None,
    document_kind: str = "auto",
) -> dict[str, Any]:
    document_kind = normalize_document_kind(document_kind)
    resolved = _resolve_inputs(
        source_path=source_path,
        html_path=html_path,
        tex_path=tex_path,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
    )
    paper_id = source_id or _generated_source_id()
    if html_text is not None:
        parsed = parse_html(
            html_text,
            paper_id=paper_id,
            include_document=include_document,
            source_url=source_url,
            assets=assets,
        )
        return _attach_structure(
            _canonical(parsed, paper_id=paper_id, source_hash=_sha256_text(html_text)),
            document_kind=document_kind,
        )
    if resolved["html_path"]:
        path = Path(resolved["html_path"])
        data = path.read_bytes()
        parsed = parse_html(data.decode("utf-8"), paper_id=paper_id, include_document=include_document)
        return _attach_structure(
            _canonical(parsed, paper_id=paper_id, source_hash=_sha256_bytes(data)),
            document_kind=document_kind,
        )
    if resolved["tex_path"]:
        return _attach_structure(
            parse_tex_document(Path(resolved["tex_path"]), paper_id=paper_id, pdf_path=resolved["pdf_path"]),
            document_kind=document_kind,
            pdf_path=resolved["pdf_path"],
            require_pdf_reconciliation=True,
        )
    if resolved["markdown_path"]:
        parsed = parse_markdown_document(
            Path(resolved["markdown_path"]),
            paper_id=paper_id,
            pdf_path=resolved["pdf_path"],
            include_document=include_document,
        )
        return _attach_structure(
            parsed,
            document_kind=document_kind,
            pdf_path=resolved["pdf_path"],
            require_pdf_reconciliation=bool(resolved["pdf_path"]),
        )
    if resolved["pdf_path"]:
        return _attach_structure(
            parse_pdf_document(Path(resolved["pdf_path"]), paper_id=paper_id),
            document_kind=document_kind,
            pdf_path=resolved["pdf_path"],
        )
    raise ValueError("parse_source_input requires an HTML, TeX, Markdown, PDF, or ar5iv source")


def parse_source_input_with_warnings(
    *,
    source_path: str | Path | None = None,
    source_id: str | None = None,
    html_path: str | Path | None = None,
    html_text: str | None = None,
    tex_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    include_document: bool = True,
    document_kind: str = "auto",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document_kind = normalize_document_kind(document_kind)
    resolved = _resolve_inputs(
        source_path=source_path,
        html_path=html_path,
        tex_path=tex_path,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
    )
    paper_id = source_id or _generated_source_id()
    if html_text is not None:
        parsed = parse_html(html_text, paper_id=paper_id, include_document=include_document)
        return _attach_structure(
            _canonical(parsed, paper_id=paper_id, source_hash=_sha256_text(html_text)),
            document_kind=document_kind,
        ), []
    if resolved["html_path"]:
        path = Path(resolved["html_path"])
        data = path.read_bytes()
        parsed = parse_html(data.decode("utf-8"), paper_id=paper_id, include_document=include_document)
        return _attach_structure(
            _canonical(parsed, paper_id=paper_id, source_hash=_sha256_bytes(data)),
            document_kind=document_kind,
        ), []
    if resolved["tex_path"]:
        if resolved["pdf_path"]:
            parsed, warnings = parse_tex_document_with_warnings(
                Path(resolved["tex_path"]), paper_id=paper_id, pdf_path=resolved["pdf_path"]
            )
            return _attach_structure(
                parsed,
                document_kind=document_kind,
                pdf_path=resolved["pdf_path"],
                require_pdf_reconciliation=True,
            ), warnings
        return _attach_structure(
            parse_tex_document(Path(resolved["tex_path"]), paper_id=paper_id),
            document_kind=document_kind,
        ), []
    if resolved["markdown_path"]:
        if resolved["pdf_path"]:
            parsed, warnings = parse_markdown_document_with_warnings(
                Path(resolved["markdown_path"]),
                paper_id=paper_id,
                pdf_path=resolved["pdf_path"],
                include_document=include_document,
            )
            return _attach_structure(
                parsed,
                document_kind=document_kind,
                pdf_path=resolved["pdf_path"],
                require_pdf_reconciliation=True,
            ), warnings
        return _attach_structure(
            parse_markdown_document(
                Path(resolved["markdown_path"]), paper_id=paper_id, include_document=include_document
            ),
            document_kind=document_kind,
        ), []
    if resolved["pdf_path"]:
        parsed, warnings = parse_pdf_document_with_warnings(Path(resolved["pdf_path"]), paper_id=paper_id)
        return _attach_structure(
            parsed, document_kind=document_kind, pdf_path=resolved["pdf_path"]
        ), warnings
    raise ValueError("parse_source_input requires an HTML, TeX, Markdown, PDF, or ar5iv source")


def source_input_hash(
    *,
    source_path: str | Path | None = None,
    html_path: str | Path | None = None,
    tex_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
) -> str:
    resolved = _resolve_inputs(
        source_path=source_path,
        html_path=html_path,
        tex_path=tex_path,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
    )
    if resolved["html_path"]:
        return _sha256_bytes(Path(resolved["html_path"]).read_bytes())
    if resolved["tex_path"]:
        paths = [Path(resolved["tex_path"])]
        if resolved["pdf_path"]:
            paths.append(Path(resolved["pdf_path"]))
        return _combined_hash(paths)
    if resolved["markdown_path"]:
        paths = [Path(resolved["markdown_path"])]
        if resolved["pdf_path"]:
            paths.append(Path(resolved["pdf_path"]))
        return _combined_hash(paths)
    if resolved["pdf_path"]:
        return _sha256_bytes(Path(resolved["pdf_path"]).read_bytes())
    raise ValueError("parse_source_input requires an HTML, TeX, Markdown, PDF, or ar5iv source")


def extract_pdf_pages(path: str | Path) -> list[str]:
    pages, _ = _extract_pdf_pages_with_warning(path)
    return pages


def _extract_pdf_pages_with_warning(path: str | Path) -> tuple[list[str], dict[str, Any] | None]:
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return [], {
            "code": "pdf_not_used",
            "message": "PDF input was provided but pdftotext is not installed; PDF was not used.",
            "pdf_path": str(path),
        }
    except subprocess.TimeoutExpired:
        return [], {
            "code": "pdf_not_used",
            "message": "PDF input was provided but pdftotext timed out; PDF was not used.",
            "pdf_path": str(path),
        }
    except subprocess.SubprocessError:
        return [], {
            "code": "pdf_not_used",
            "message": "PDF input was provided but pdftotext failed; PDF was not used.",
            "pdf_path": str(path),
        }
    except OSError:
        return [], {
            "code": "pdf_not_used",
            "message": "PDF input was provided but pdftotext could not run; PDF was not used.",
            "pdf_path": str(path),
        }
    raw_pages = completed.stdout.split("\f")
    if raw_pages and not raw_pages[-1].strip():
        raw_pages.pop()
    pages = [page.strip() for page in raw_pages]
    if not any(pages):
        return [], {
            "code": "pdf_not_used",
            "message": "PDF input was provided but pdftotext returned no text; PDF was not used.",
            "pdf_path": str(path),
        }
    return pages, None


def parse_tex_document(path: Path, *, paper_id: str, pdf_path: str | Path | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    pdf_pages = extract_pdf_pages(pdf_path) if pdf_path else []
    return _tex_document_from_pages(path, paper_id=paper_id, pdf_path=pdf_path, lines=lines, pdf_pages=pdf_pages)


def parse_tex_document_with_warnings(
    path: Path, *, paper_id: str, pdf_path: str | Path | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    pdf_pages: list[str] = []
    warnings: list[dict[str, Any]] = []
    if pdf_path:
        pdf_pages, warning = _extract_pdf_pages_with_warning(pdf_path)
        if warning:
            warnings.append(warning)
    return _tex_document_from_pages(path, paper_id=paper_id, pdf_path=pdf_path, lines=lines, pdf_pages=pdf_pages), warnings


def _tex_document_from_pages(
    path: Path,
    *,
    paper_id: str,
    pdf_path: str | Path | None,
    lines: list[str],
    pdf_pages: list[str],
) -> dict[str, Any]:
    active_lines = _tex_active_lines(lines)
    sections = _tex_sections(path, active_lines)
    equations = _tex_equations(path, active_lines, sections)
    if pdf_pages:
        _enrich_equations_from_pdf(equations, pdf_pages)
        _fill_section_pdf_pages(sections, equations)
    return {
        "paper_id": paper_id,
        "parser_version": PARSER_VERSION,
        "source_hash": _combined_hash([path, Path(pdf_path)] if pdf_path else [path]),
        "toc": [{"id": item["section_id"], "title": item["title"], "level": item["level"]} for item in sections],
        "sections": sections,
        "equations": equations,
    }


def parse_markdown_document(
    path: Path,
    *,
    paper_id: str,
    pdf_path: str | Path | None = None,
    include_document: bool = True,
) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    pdf_pages = extract_pdf_pages(pdf_path) if pdf_path else []
    return _markdown_document_from_pages(
        path,
        paper_id=paper_id,
        pdf_path=pdf_path,
        lines=lines,
        pdf_pages=pdf_pages,
        include_document=include_document,
    )


def parse_markdown_document_with_warnings(
    path: Path,
    *,
    paper_id: str,
    pdf_path: str | Path | None = None,
    include_document: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    pdf_pages: list[str] = []
    warnings: list[dict[str, Any]] = []
    if pdf_path:
        pdf_pages, warning = _extract_pdf_pages_with_warning(pdf_path)
        if warning:
            warnings.append(warning)
    return (
        _markdown_document_from_pages(
            path,
            paper_id=paper_id,
            pdf_path=pdf_path,
            lines=lines,
            pdf_pages=pdf_pages,
            include_document=include_document,
        ),
        warnings,
    )


def _markdown_document_from_pages(
    path: Path,
    *,
    paper_id: str,
    pdf_path: str | Path | None,
    lines: list[str],
    pdf_pages: list[str],
    include_document: bool,
) -> dict[str, Any]:
    sections = _markdown_sections(path, lines)
    equations = _markdown_equations(path, lines, sections)
    if pdf_pages:
        _enrich_equations_from_pdf(equations, pdf_pages)
        _fill_section_pdf_pages(sections, equations)
    source_hash = _combined_hash([path, Path(pdf_path)] if pdf_path else [path])
    parsed = {
        "paper_id": paper_id,
        "parser_version": PARSER_VERSION,
        "source_hash": source_hash,
        "toc": [{"id": item["section_id"], "title": item["title"], "level": item["level"]} for item in sections],
        "sections": sections,
        "equations": equations,
    }
    if include_document:
        parsed["document"] = build_markdown_document(
            lines,
            paper_id=paper_id,
            source_path=path,
            source_hash=source_hash,
            sections=sections,
            equations=equations,
            pdf_path=pdf_path,
        )
    return parsed


def parse_pdf_document(path: Path, *, paper_id: str) -> dict[str, Any]:
    pages = extract_pdf_pages(path)
    return _pdf_document_from_pages(path, paper_id=paper_id, pages=pages)


def parse_pdf_document_with_warnings(path: Path, *, paper_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pages, warning = _extract_pdf_pages_with_warning(path)
    warnings = [warning] if warning else []
    return _pdf_document_from_pages(path, paper_id=paper_id, pages=pages), warnings


def _pdf_document_from_pages(path: Path, *, paper_id: str, pages: list[str]) -> dict[str, Any]:
    sections = _pdf_sections(path, pages)
    equations = _pdf_equations(path, pages, sections)
    return {
        "paper_id": paper_id,
        "parser_version": PARSER_VERSION,
        "source_hash": _sha256_bytes(path.read_bytes()),
        "toc": [{"id": item["section_id"], "title": item["title"], "level": item["level"]} for item in sections],
        "sections": sections,
        "equations": equations,
    }


def _canonical(parsed: dict[str, Any], *, paper_id: str, source_hash: str) -> dict[str, Any]:
    result = {
        "paper_id": paper_id,
        "parser_version": PARSER_VERSION,
        "source_hash": source_hash,
        "toc": list(parsed.get("toc") or []),
        "sections": list(parsed.get("sections") or []),
        "equations": list(parsed.get("equations") or []),
    }
    if isinstance(parsed.get("document"), dict):
        result["document"] = dict(parsed["document"])
    return result


def _attach_structure(
    parsed: dict[str, Any],
    *,
    document_kind: str,
    pdf_path: str | Path | None = None,
    require_pdf_reconciliation: bool = False,
) -> dict[str, Any]:
    pdf_pages: list[str] = []
    index_entries = empty_index_entries()
    embedded_outline: list[dict[str, Any]] = []
    page_labels: list[str] = []
    if pdf_path:
        pdf_pages = extract_pdf_pages(pdf_path)
        if not pdf_pages and require_pdf_reconciliation:
            raise ValueError(
                "Paired PDF reconciliation requires an extractable PDF text layer; PDF extraction failed."
            )
        if pdf_pages:
            embedded_outline, page_labels = read_embedded_outline(pdf_path)
            index_entries = parse_index_entries(pdf_pages, page_labels=page_labels)
    parsed["structure"] = build_structure(
        parsed,
        requested_document_kind=document_kind,
        pdf_path=pdf_path,
        pdf_pages=pdf_pages,
        index_source_pages=index_entries.get("source_pages") or [],
        embedded_outline=embedded_outline,
        pdf_page_labels=page_labels,
    )
    parsed["index_entries"] = index_entries
    return parsed


def _resolve_inputs(
    *,
    source_path: str | Path | None,
    html_path: str | Path | None,
    tex_path: str | Path | None,
    markdown_path: str | Path | None,
    pdf_path: str | Path | None,
) -> dict[str, Path | None]:
    resolved = {
        "html_path": Path(html_path) if html_path else None,
        "tex_path": Path(tex_path) if tex_path else None,
        "markdown_path": Path(markdown_path) if markdown_path else None,
        "pdf_path": Path(pdf_path) if pdf_path else None,
    }
    if source_path:
        path = Path(source_path)
        suffix = path.suffix.lower()
        if suffix in {".html", ".htm"}:
            resolved["html_path"] = path
        elif suffix == ".tex":
            resolved["tex_path"] = path
        elif suffix in {".md", ".markdown"}:
            resolved["markdown_path"] = path
        elif suffix == ".pdf":
            resolved["pdf_path"] = path
        else:
            raise ValueError(f"Cannot infer parse source type from extension: {path}")
    return resolved


def _generated_source_id() -> str:
    return f"arc-{random.SystemRandom().randrange(100000000):08d}"


def _tex_active_lines(lines: list[str]) -> list[str]:
    active: list[str] = []
    in_comment = False
    begin_comment = re.compile(r"\\begin\{comment\*?\}")
    end_comment = re.compile(r"\\end\{comment\*?\}")
    for line in lines:
        stripped = _strip_comment(line)
        if in_comment:
            active.append("")
            if end_comment.search(stripped):
                in_comment = False
            continue
        if begin_comment.search(stripped):
            active.append("")
            if not end_comment.search(stripped):
                in_comment = True
            continue
        active.append(line)
    return active


def _tex_sections(path: Path, lines: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        active_line = _strip_comment(line)
        match = re.search(r"\\(section|subsection|subsubsection)\*?\s*\{([^{}]*)\}", active_line)
        if not match:
            continue
        command, title = match.groups()
        label = _line_label(lines[index]) if index < len(lines) else ""
        section_id = label or f"sec_{len(sections) + 1:04d}"
        sections.append(
            {
                "section_id": section_id,
                "title": _clean_text(title),
                "level": SECTION_LEVELS[command],
                "text": "",
                "source_path": str(path),
                "tex_line_start": index,
                "tex_line_end": len(lines),
                "pdf_page_start": None,
                "pdf_page_end": None,
            }
        )
    for index, section in enumerate(sections):
        if index + 1 < len(sections):
            section["tex_line_end"] = sections[index + 1]["tex_line_start"] - 1
        section["text"] = _section_text(lines, int(section["tex_line_start"]), int(section["tex_line_end"]))
    return sections


def _tex_equations(path: Path, lines: list[str], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    equations: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        env = _begin_environment(lines[index])
        if env:
            start = index
            end = _find_environment_end(lines, index, env)
            equations.append(_tex_equation_record(path, lines, start, end, env, len(equations), sections))
            index = end + 1
            continue
        display = _display_math_span(lines, index)
        if display:
            start, end, env = display
            equations.append(_tex_equation_record(path, lines, start, end, env, len(equations), sections))
            index = end + 1
            continue
        index += 1
    return equations


def _markdown_sections(path: Path, lines: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    in_fence = False
    for index, line in enumerate(lines, start=1):
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        hashes, title = match.groups()
        sections.append(
            {
                "section_id": f"sec_{len(sections) + 1:04d}",
                "title": _clean_markdown_text(title),
                "level": len(hashes),
                "text": "",
                "source_path": str(path),
                "markdown_line_start": index,
                "markdown_line_end": len(lines),
                "pdf_page_start": None,
                "pdf_page_end": None,
            }
        )
    if not sections and lines:
        sections.append(
            {
                "section_id": "sec_0001",
                "title": path.stem,
                "level": 1,
                "text": "",
                "source_path": str(path),
                "markdown_line_start": 1,
                "markdown_line_end": len(lines),
                "pdf_page_start": None,
                "pdf_page_end": None,
            }
        )
    for index, section in enumerate(sections):
        if index + 1 < len(sections):
            section["markdown_line_end"] = sections[index + 1]["markdown_line_start"] - 1
        start = int(section["markdown_line_start"])
        end = int(section["markdown_line_end"])
        section["text"] = _markdown_section_text(lines, start, end)
    return sections


def _markdown_equations(
    path: Path, lines: list[str], sections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    equations: list[dict[str, Any]] = []
    in_fence = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue
        span = _markdown_display_math_span(lines, index)
        if not span:
            index += 1
            continue
        start, end, environment = span
        raw_tex = "\n".join(lines[start : end + 1])
        equation = _normalize_latex(raw_tex, environment)
        section = _section_for_markdown_line(sections, start + 1)
        record: dict[str, Any] = {
            "id": f"eq_{len(equations) + 1:05d}",
            "equation": equation,
            "before": _nearby_markdown_text_before(lines, start),
            "after": _nearby_markdown_text_after(lines, end),
            "section_id": str(section.get("section_id") or ""),
            "section_title": str(section.get("title") or ""),
            "source_path": str(path),
            "markdown_line_start": start + 1,
            "markdown_line_end": end + 1,
            "tex_label": _equation_label(lines, start, end),
            "raw_tex": raw_tex,
            "normalized_latex": equation,
            "confidence": "high",
            "parser_warnings": [],
        }
        _set_source_equation_numbers(record, raw_tex)
        equations.append(record)
        index = end + 1
    return equations


def _markdown_display_math_span(lines: list[str], start: int) -> tuple[int, int, str] | None:
    stripped = lines[start].strip()
    if stripped.startswith("$$"):
        if stripped != "$$" and stripped.count("$$") >= 2:
            return start, start, "dollar_display"
        return start, _find_display_end(lines, start + 1, "$$"), "dollar_display"
    if stripped.startswith(r"\["):
        if stripped != r"\[" and r"\]" in stripped:
            return start, start, "display_math"
        return start, _find_display_end(lines, start + 1, r"\]"), "display_math"
    env = _begin_environment(lines[start])
    if env:
        return start, _find_environment_end(lines, start, env), env
    return None


def _section_for_markdown_line(sections: list[dict[str, Any]], line_number: int) -> dict[str, Any]:
    for section in reversed(sections):
        if int(section["markdown_line_start"]) <= line_number <= int(section["markdown_line_end"]):
            return section
    return {}


def _markdown_section_text(lines: list[str], start: int, end: int) -> str:
    return _clean_text("\n".join(_clean_markdown_text(line) for line in lines[start - 1 : end]))


def _nearby_markdown_text_before(lines: list[str], start: int) -> str:
    for index in range(start - 1, -1, -1):
        if text := _markdown_context_line(lines[index]):
            return text
    return ""


def _nearby_markdown_text_after(lines: list[str], end: int) -> str:
    for index in range(end + 1, len(lines)):
        if text := _markdown_context_line(lines[index]):
            return text
    return ""


def _markdown_context_line(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped in {"$$", r"\[", r"\]"}:
        return ""
    if re.match(r"^\s*(?:```|~~~|#{1,6}\s)", line):
        return ""
    return _clean_markdown_text(line)


def _clean_markdown_text(text: str) -> str:
    cleaned = re.sub(r"!?(?:\[([^\]]*)\])\([^)]*\)", r"\1", text)
    cleaned = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", cleaned)
    cleaned = re.sub(r"[*_`]|(?<!\\)~", "", cleaned)
    cleaned = cleaned.replace(r"\~", "~")
    return _clean_text(cleaned)


def _tex_equation_record(
    path: Path,
    lines: list[str],
    start: int,
    end: int,
    environment: str,
    equation_index: int,
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_tex = "\n".join(lines[start : end + 1])
    equation = _normalize_latex(raw_tex, environment)
    section = _section_for_line(sections, start + 1)
    record = {
        "id": f"eq_{equation_index + 1:05d}",
        "equation": equation,
        "before": _nearby_text_before(lines, start),
        "after": _nearby_text_after(lines, end),
        "section_id": str(section.get("section_id") or ""),
        "section_title": str(section.get("title") or ""),
        "source_path": str(path),
        "tex_line_start": start + 1,
        "tex_line_end": end + 1,
        "tex_label": _equation_label(lines, start, end),
        "raw_tex": raw_tex,
        "normalized_latex": equation,
        "confidence": "high",
        "parser_warnings": [],
    }
    _set_source_equation_numbers(record, raw_tex)
    return record


def _pdf_sections(path: Path, pages: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, start=1):
        title = _first_heading_like_line(page) or f"Page {page_index}"
        section_id = f"pdf_page_{page_index:04d}"
        sections.append(
            {
                "section_id": section_id,
                "title": title,
                "level": 1,
                "text": page,
                "source_path": str(path),
                "pdf_page_start": page_index,
                "pdf_page_end": page_index,
            }
        )
    return sections


def _pdf_equations(path: Path, pages: list[str], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    equations: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, start=1):
        lines = [line.strip() for line in page.splitlines()]
        for line_index, line in enumerate(lines):
            candidate = _pdf_equation_candidate(line)
            if not candidate:
                continue
            number = _line_equation_number(line)
            section = sections[page_index - 1] if page_index - 1 < len(sections) else {}
            equations.append(
                {
                    "id": f"eq_{len(equations) + 1:05d}",
                    "equation": candidate,
                    "before": _neighbor_pdf_text(lines, line_index, -1),
                    "after": _neighbor_pdf_text(lines, line_index, 1),
                    "section_id": str(section.get("section_id") or ""),
                    "section_title": str(section.get("title") or ""),
                    "source_path": str(path),
                    "printed_equation_number": number,
                    "pdf_page": page_index,
                    "confidence": "medium" if number else "low",
                    "parser_warnings": [
                        {"code": "pdf_only_candidate", "message": "PDF-only equation extraction is approximate."}
                    ],
                }
            )
    return equations


def _begin_environment(line: str) -> str:
    match = re.search(r"\\begin\{(" + "|".join(DISPLAY_ENVIRONMENTS) + r")\*?\}", _strip_comment(line))
    return match.group(1) if match else ""


def _find_environment_end(lines: list[str], start: int, env: str) -> int:
    pattern = re.compile(r"\\end\{" + re.escape(env) + r"\*?\}")
    for index in range(start, len(lines)):
        if pattern.search(_strip_comment(lines[index])):
            return index
    return start


def _display_math_span(lines: list[str], start: int) -> tuple[int, int, str] | None:
    line = _strip_comment(lines[start])
    if r"\[" in line:
        return start, _find_display_end(lines, start, r"\]"), "display_math"
    if "$$" in line:
        if line.count("$$") >= 2:
            return start, start, "dollar_display"
        return start, _find_display_end(lines, start + 1, "$$"), "dollar_display"
    return None


def _find_display_end(lines: list[str], start: int, token: str) -> int:
    for index in range(start, len(lines)):
        if token in _strip_comment(lines[index]):
            return index
    return start


def _equation_label(lines: list[str], start: int, end: int) -> str:
    for index in range(start, end + 1):
        if label := _line_label(lines[index]):
            return label
    if start > 0 and (label := _line_label(lines[start - 1])):
        return label
    if end + 1 < len(lines) and (label := _line_label(lines[end + 1])):
        return label
    return ""


def _line_label(line: str, default: str = "") -> str:
    match = re.search(r"\\label\{([^{}]+)\}", _strip_comment(line))
    return match.group(1) if match else default


def _normalize_latex(raw_tex: str, environment: str) -> str:
    text = raw_tex.strip()
    if environment in DISPLAY_ENVIRONMENTS:
        text = re.sub(r"^\\begin\{[^{}]+\*?\}", "", text).strip()
        text = re.sub(r"\\end\{[^{}]+\*?\}$", "", text).strip()
    elif environment == "display_math":
        text = text.replace(r"\[", "", 1)
        text = text.rsplit(r"\]", 1)[0]
    elif environment == "dollar_display":
        text = text.replace("$$", "", 1)
        text = text.rsplit("$$", 1)[0]
    cleaned = []
    for line in text.splitlines():
        stripped = _strip_comment(line).strip()
        if not stripped or re.fullmatch(r"\\label\{[^{}]+\}", stripped):
            continue
        cleaned.append(stripped)
    return " ".join(" ".join(cleaned).split())


def _section_for_line(sections: list[dict[str, Any]], line_number: int) -> dict[str, Any]:
    for section in reversed(sections):
        if int(section["tex_line_start"]) <= line_number <= int(section["tex_line_end"]):
            return section
    return {}


def _section_text(lines: list[str], start: int, end: int) -> str:
    parts = []
    for line in lines[start - 1 : end]:
        text = _text_context_line(line)
        if text:
            parts.append(text)
    return _clean_text("\n".join(parts))


def _nearby_text_before(lines: list[str], start: int) -> str:
    for index in range(start - 1, -1, -1):
        text = _text_context_line(lines[index])
        if text:
            return text
    return ""


def _nearby_text_after(lines: list[str], end: int) -> str:
    for index in range(end + 1, len(lines)):
        text = _text_context_line(lines[index])
        if text:
            return text
    return ""


def _text_context_line(line: str) -> str:
    stripped = _strip_comment(line).strip()
    if not stripped:
        return ""
    if re.match(r"\\(begin|end|label|section|subsection|subsubsection|documentclass|usepackage|newcommand|renewcommand)", stripped):
        return ""
    if stripped in {r"\[", r"\]", "$$"}:
        return ""
    return stripped


def _pdf_equation_candidate(line: str) -> str:
    cleaned = re.sub(rf"\(({PRINTED_EQUATION_NUMBER_PATTERN})\)\s*$", "", line).strip()
    if "=" not in cleaned:
        return ""
    tokens = _search_tokens(cleaned)
    if len(tokens) < 3:
        return ""
    return cleaned


def _line_equation_number(line: str) -> str | None:
    match = re.search(rf"\(({PRINTED_EQUATION_NUMBER_PATTERN})\)\s*$", line)
    return re.sub(r"\s+", "", match.group(1)) if match else None


def _first_heading_like_line(page: str) -> str:
    for raw in page.splitlines():
        line = raw.strip()
        if re.match(r"^\d+(?:\.\d+)*\s+\S+", line):
            return line
    return ""


def _neighbor_pdf_text(lines: list[str], index: int, direction: int) -> str:
    current = index + direction
    while 0 <= current < len(lines):
        line = lines[current].strip()
        if line and not _pdf_equation_candidate(line):
            return line
        current += direction
    return ""


def _enrich_equations_from_pdf(equations: list[dict[str, Any]], pages: list[str]) -> None:
    anchors = _monotonic_tag_anchors(equations, pages)
    anchor_pages = {equation_index: page for equation_index, page in anchors}
    for equation_index, page in anchor_pages.items():
        equations[equation_index]["pdf_page"] = page

    previous_page = 1
    for equation_index, equation in enumerate(equations):
        if equation_index in anchor_pages:
            previous_page = anchor_pages[equation_index]
            continue
        lower_page, upper_page = _page_bounds_for_equation(
            equation_index, anchors, page_count=len(pages), previous_page=previous_page
        )
        best_page = _best_pdf_page_in_bounds(equation, pages, lower_page, upper_page)
        if best_page is None:
            continue
        equation["pdf_page"] = best_page
        previous_page = best_page

    reserved_numbers_by_page: dict[int, set[str]] = {}
    for equation in equations:
        page_number = equation.get("pdf_page")
        if not isinstance(page_number, int):
            continue
        source_numbers = _source_equation_numbers(str(equation.get("raw_tex") or ""))
        if source_numbers:
            reserved_numbers_by_page.setdefault(page_number, set()).update(source_numbers)

    for equation in equations:
        page_number = equation.get("pdf_page")
        raw_tex = str(equation.get("raw_tex") or "")
        if (
            not isinstance(page_number, int)
            or _source_equation_numbers(raw_tex)
            or _source_equation_is_unnumbered(raw_tex)
        ):
            continue
        number, numbers = _best_equation_number_match(pages[page_number - 1], equation)
        # A source tag is authoritative for the equation that owns it.  Do not
        # infer the same printed number for a nearby unnumbered display on the
        # same PDF page merely because its context window reaches that tag.
        if number in reserved_numbers_by_page.get(page_number, set()):
            continue
        if number:
            equation["printed_equation_number"] = number
            equation["printed_equation_numbers"] = numbers or [number]


def _set_source_equation_numbers(record: dict[str, Any], raw_tex: str) -> None:
    directive_numbers, unnumbered, directive_warnings = _source_equation_number_directives(raw_tex)
    record.setdefault("parser_warnings", []).extend(directive_warnings)
    if unnumbered:
        record.pop("printed_equation_number", None)
        record.pop("printed_equation_numbers", None)
        equation = _remove_explicit_equation_tags(str(record.get("equation") or ""))
        record["equation"] = equation
        record["normalized_latex"] = equation
        return
    numbers = _source_equation_numbers(raw_tex)
    if not numbers:
        return
    record["printed_equation_number"] = numbers[0]
    record["printed_equation_numbers"] = numbers
    if not directive_numbers and not _explicit_equation_tags(raw_tex):
        equation = str(record.get("equation") or "")
        equation = _remove_recognized_ocr_trailing_equation_number(equation)
        record["equation"] = equation
        record["normalized_latex"] = equation


def _explicit_equation_tags(raw_tex: str) -> list[str]:
    numbers = [match.strip() for match in re.findall(r"\\tag\*?\s*\{([^{}]+)\}", raw_tex)]
    return list(dict.fromkeys(number for number in numbers if number))


def _remove_explicit_equation_tags(equation: str) -> str:
    """Remove visible TeX tags when an explicit unnumbered directive wins."""
    return re.sub(r"\s*\\tag\*?\s*\{[^{}]+\}", "", equation).strip()


def _source_equation_numbers(raw_tex: str) -> list[str]:
    directive_numbers, unnumbered, _ = _source_equation_number_directives(raw_tex)
    if unnumbered:
        return []
    if directive_numbers:
        return directive_numbers
    tags = _explicit_equation_tags(raw_tex)
    if tags:
        return tags
    match = _ocr_trailing_equation_number_match(raw_tex)
    if not match:
        return []
    suffix = re.sub(r"\s+", "", match.group(2))
    return [f"{match.group(1)}.{suffix}"] if suffix else []


def _ocr_trailing_equation_number_match(text: str) -> re.Match[str] | None:
    """Return a trailing OCR equation number only with layout evidence.

    A compact value such as ``f(x)=(2.18)`` is valid formula content and is
    therefore left untouched.  A standalone number, a visibly right-separated
    number, or OCR-spaced digits such as ``(2. 1 8)`` provide enough evidence
    to treat the suffix as a printed equation number.
    """
    match = re.search(
        r"\(\s*(\d+)\s*\.\s*((?:\d\s*)+)\)\s*(?:\$\$|\\\]|\\end\{[^{}]+\*?\})?\s*$",
        text,
    )
    if not match:
        return None
    line_prefix = text[: match.start()].rsplit("\n", 1)[-1]
    standalone = not line_prefix.strip()
    layout_gap = bool(re.search(r"[ \t]{2,}$", line_prefix))
    ocr_spaced_digits = bool(re.search(r"\d\s+\d", match.group(0)))
    return match if standalone or layout_gap or ocr_spaced_digits else None


def _remove_recognized_ocr_trailing_equation_number(equation: str) -> str:
    """Remove a suffix after the raw source has passed the layout check."""
    return re.sub(r"\s*\(\s*\d+\s*\.\s*(?:\d\s*)+\)\s*$", "", equation).rstrip()


def _source_equation_is_unnumbered(raw_tex: str) -> bool:
    _, unnumbered, _ = _source_equation_number_directives(raw_tex)
    return unnumbered


def _source_equation_number_directives(
    raw_tex: str,
) -> tuple[list[str], bool, list[dict[str, str]]]:
    """Read portable equation-number overrides from TeX comments.

    ``% arc:equation-number 2.3`` supplies authoritative printed numbers, while
    ``% arc:unnumbered`` prevents approximate PDF number inference.  An
    unnumbered directive wins conflicting declarations so that the parser does
    not invent a printed number.
    """
    numbers: list[str] = []
    unnumbered = False
    warnings: list[dict[str, str]] = []
    for line in raw_tex.splitlines():
        comment = _tex_comment(line).strip()
        if re.fullmatch(r"arc:unnumbered", comment, flags=re.IGNORECASE):
            unnumbered = True
            continue
        match = re.fullmatch(r"arc:equation-number\s+(.+?)\s*", comment, flags=re.IGNORECASE)
        if not match:
            continue
        candidates = [part for part in re.split(r"[\s,]+", match.group(1)) if part]
        invalid = [number for number in candidates if not re.fullmatch(EQUATION_NUMBER_PATTERN, number)]
        if not candidates or invalid:
            warnings.append(
                {
                    "code": "invalid_equation_number_directive",
                    "message": "Ignored an arc:equation-number directive with an invalid printed number.",
                }
            )
            continue
        numbers.extend(candidates)
    numbers = list(dict.fromkeys(numbers))
    if unnumbered and (numbers or _explicit_equation_tags(raw_tex)):
        warnings.append(
            {
                "code": "conflicting_equation_number_directives",
                "message": "arc:unnumbered overrides conflicting source equation numbers.",
            }
        )
        numbers = []
    return numbers, unnumbered, warnings


def _monotonic_tag_anchors(
    equations: list[dict[str, Any]], pages: list[str]
) -> list[tuple[int, int]]:
    pages_by_number: dict[str, list[int]] = {}
    for page_index, page in enumerate(pages, start=1):
        for number, _ in _printed_equation_number_candidates(page):
            page_numbers = pages_by_number.setdefault(number, [])
            if page_index not in page_numbers:
                page_numbers.append(page_index)

    # Equation numbers are not globally unique in books that restart numbering
    # by chapter (and short labels such as ``(1)`` repeat especially often).
    # Resolve them in source order instead of selecting only globally unique
    # labels.  The first candidate after the preceding anchor with formula or
    # nearby-prose evidence is the stable page anchor; this keeps subsequent
    # unnumbered displays within the correct local page interval.
    anchors: list[tuple[int, int]] = []
    previous_page = 0
    for equation_index, equation in enumerate(equations):
        tags = _source_equation_numbers(str(equation.get("raw_tex") or ""))
        if not tags:
            continue
        candidate_pages = pages_by_number.get(tags[0], [])
        if not candidate_pages:
            continue
        selected_page: int | None = None
        for page in candidate_pages:
            if page < previous_page:
                continue
            if _source_number_anchor_has_content_evidence(equation, pages[page - 1]):
                selected_page = page
                break
        if selected_page is None:
            continue
        anchors.append((equation_index, selected_page))
        previous_page = selected_page
    return anchors


def _source_number_anchor_has_content_evidence(equation: dict[str, Any], page: str) -> bool:
    """Require formula or exact nearby prose evidence in addition to a tag."""
    page_tokens = set(_search_tokens(page))
    equation_tokens = set(_search_tokens(_label_focused_equation_text(equation)))
    # Two non-numeric symbol tokens are enough when the printed number is
    # already unique.  This retains short rows (for example a single aligned
    # Mandelstam relation) whose surrounding Markdown context is empty.
    if _token_overlap_score(equation_tokens, page_tokens) >= 2:
        return True
    page_context = _context_match_text(page)
    for field in ("before", "after"):
        context = _context_match_text(str(equation.get(field) or ""))
        if _is_long_context(context) and _has_exact_context_fragment(context, page_context):
            return True
    return False


def _longest_nondecreasing_anchors(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not candidates:
        return []
    lengths = [1] * len(candidates)
    previous = [-1] * len(candidates)
    for index, (_, page) in enumerate(candidates):
        for prior_index in range(index):
            if candidates[prior_index][1] <= page and lengths[prior_index] + 1 > lengths[index]:
                lengths[index] = lengths[prior_index] + 1
                previous[index] = prior_index
    current = max(range(len(candidates)), key=lengths.__getitem__)
    selected: list[tuple[int, int]] = []
    while current >= 0:
        selected.append(candidates[current])
        current = previous[current]
    return list(reversed(selected))


def _page_bounds_for_equation(
    equation_index: int,
    anchors: list[tuple[int, int]],
    *,
    page_count: int,
    previous_page: int,
) -> tuple[int, int]:
    lower_page = max(1, previous_page)
    upper_page = page_count
    for anchor_index, anchor_page in anchors:
        if anchor_index < equation_index:
            lower_page = max(lower_page, anchor_page)
        elif anchor_index > equation_index:
            upper_page = anchor_page
            break
    return lower_page, upper_page


def _best_pdf_page_in_bounds(
    equation: dict[str, Any], pages: list[str], lower_page: int, upper_page: int
) -> int | None:
    best_page: int | None = None
    best_score = 0
    for page_number in range(lower_page, upper_page + 1):
        score = _pdf_match_score(equation, pages[page_number - 1])
        if score > best_score:
            best_score = score
            best_page = page_number
    return best_page if best_score >= PDF_PAGE_MATCH_MIN_SCORE else None


def _pdf_match_score(equation: dict[str, Any], page: str) -> int:
    haystack = _search_text(page)
    page_tokens = set(_search_tokens(page))
    score = 0
    for field in ("before", "after"):
        needle = _search_text(str(equation.get(field) or ""))
        if needle and needle in haystack:
            score += 10
        if needle_tokens := set(_search_tokens(str(equation.get(field) or ""))):
            score += min(_token_overlap_score(needle_tokens, page_tokens), 20)
    eq_tokens = set(_search_tokens(str(equation.get("equation") or "")))
    score += _token_overlap_score(eq_tokens, page_tokens)
    return score


def _best_equation_number(page: str, equation: dict[str, Any]) -> str | None:
    number, _ = _best_equation_number_match(page, equation)
    return number


def _best_equation_number_match(page: str, equation: dict[str, Any]) -> tuple[str | None, list[str]]:
    candidates = _printed_equation_number_candidates(page)
    if not candidates:
        return None, []
    focused_equation = _label_focused_equation_text(equation)
    equation_tokens = set(_search_tokens(focused_equation))
    if not equation_tokens:
        return candidates[0][0], [candidates[0][0]]
    before_tokens = set(_search_tokens(str(equation.get("before") or "")))
    after_tokens = set(_search_tokens(str(equation.get("after") or "")))
    line_spans = _line_spans(page)
    has_label_focus = focused_equation != str(equation.get("equation") or "")
    best_number = candidates[0][0]
    best_score = -1
    best_index = 0
    for candidate_index, (number, offset) in enumerate(candidates):
        if has_label_focus:
            equation_window = _line_window_for_offset(line_spans, offset, before_lines=3, after_lines=0)
        else:
            equation_window = _line_window_for_offset(line_spans, offset, before_lines=8, after_lines=4)
        before_window = _line_window_before_offset(line_spans, offset, max_lines=12)
        after_window = _line_window_after_offset(line_spans, offset, max_lines=12)
        before_score = _token_overlap_score(before_tokens, set(_search_tokens(before_window)))
        after_score = _token_overlap_score(after_tokens, set(_search_tokens(after_window)))
        equation_score = _token_overlap_score(equation_tokens, set(_search_tokens(equation_window)))
        if equation_score <= 0 or not _ordered_exact_context_evidence(
            equation, before_window=before_window, after_window=after_window
        ):
            continue
        bracket_bonus = 50 if before_score >= 6 and after_score >= 6 else 0
        if has_label_focus:
            score = bracket_bonus + (before_score * 2) + (after_score * 4) + (equation_score * 10)
        else:
            score = bracket_bonus + (before_score * 4) + (after_score * 4) + equation_score
        if score > best_score:
            best_score = score
            best_number = number
            best_index = candidate_index
    if best_score < 0:
        return None, []
    return best_number, _printed_equation_number_sequence(candidates, best_index, equation)


def _ordered_exact_context_evidence(
    equation: dict[str, Any], *, before_window: str, after_window: str
) -> bool:
    before = _context_match_text(str(equation.get("before") or ""))
    if _is_long_context(before):
        return _has_exact_context_fragment(before, _context_match_text(before_window))
    after = _context_match_text(str(equation.get("after") or ""))
    if _is_long_context(after):
        return _has_exact_context_fragment(after, _context_match_text(after_window))
    return False


def _is_long_context(text: str) -> bool:
    return len(text) >= 8


def _context_match_text(text: str) -> str:
    without_commands = re.sub(r"\\[A-Za-z]+\*?", "", str(text or ""))
    return "".join(_search_tokens(without_commands))


def _has_exact_context_fragment(context: str, window: str, *, maximum_fragment_length: int = 16) -> bool:
    fragment_length = min(maximum_fragment_length, len(context))
    if fragment_length < 8 or len(window) < fragment_length:
        return False
    return any(
        context[index : index + fragment_length] in window
        for index in range(len(context) - fragment_length + 1)
    )


def _printed_equation_number_candidates(text: str) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    offset = 0
    previous_nonempty = ""
    for raw_line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(raw_line)
        line = raw_line.rstrip()
        stripped = line.strip()
        match = re.search(rf"\(({PRINTED_EQUATION_NUMBER_PATTERN})\)\s*$", line)
        if not match:
            if stripped:
                previous_nonempty = stripped
            continue
        prefix = line[: match.start()].strip()
        if _looks_like_printed_number_prefix(prefix, previous_nonempty):
            candidates.append((re.sub(r"\s+", "", match.group(1)), line_start + match.start()))
        if stripped:
            previous_nonempty = stripped
    return candidates


def _printed_equation_number_sequence(
    candidates: list[tuple[str, int]], primary_index: int, equation: dict[str, Any]
) -> list[str]:
    if not candidates:
        return []
    row_count, label_index = _numbered_latex_row_info(equation)
    if row_count <= 1:
        return [candidates[primary_index][0]]
    start = primary_index - label_index
    end = start + row_count
    if start < 0:
        start = 0
        end = min(len(candidates), row_count)
    if end > len(candidates):
        end = len(candidates)
        start = max(0, end - row_count)
    numbers = [number for number, _ in candidates[start:end]]
    if candidates[primary_index][0] not in numbers:
        return [candidates[primary_index][0]]
    return list(dict.fromkeys(numbers))


def _numbered_latex_row_info(equation: dict[str, Any]) -> tuple[int, int]:
    raw_tex = str(equation.get("raw_tex") or "")
    if not raw_tex:
        return 1, 0
    rows = _latex_rows(raw_tex)
    if not rows:
        return 1, 0
    tex_label = str(equation.get("tex_label") or "")
    numbered_rows: list[str] = []
    label_numbered_index = 0
    for row in rows:
        if not _latex_row_is_numbered(row):
            continue
        if tex_label and re.search(r"\\label\{" + re.escape(tex_label) + r"\}", row):
            label_numbered_index = len(numbered_rows)
        numbered_rows.append(row)
    if not numbered_rows:
        return 1, 0
    return len(numbered_rows), min(label_numbered_index, len(numbered_rows) - 1)


def _latex_rows(raw_tex: str) -> list[str]:
    rows: list[str] = []
    current: list[str] = []
    for raw_line in raw_tex.splitlines():
        line = _strip_environment_commands(raw_line)
        leading_separator = re.match(r"^\s*\\\\\s*(.*)$", line)
        if leading_separator and current:
            current.append(r"\\")
            rows.append("\n".join(current))
            current = []
            remainder = leading_separator.group(1)
            if remainder.strip():
                current.append(remainder)
            continue
        if line.strip() or current:
            current.append(line)
        if _has_row_separator(line):
            rows.append("\n".join(current))
            current = []
    if current:
        rows.append("\n".join(current))
    return [row for row in rows if _latex_row_has_content(row)]


def _latex_row_is_numbered(row: str) -> bool:
    if re.search(r"\\(?:nonumber|notag)\b", row):
        return False
    return _latex_row_has_content(row)


def _latex_row_has_content(row: str) -> bool:
    row = re.sub(r"\\label\{[^{}]+\}", "", row)
    row = row.replace(r"\\", "")
    return _has_latex_content(row)


def _looks_like_printed_number_prefix(prefix: str, previous_nonempty: str) -> bool:
    if not prefix:
        return True
    if _has_math_like_text(prefix):
        return True
    # OCR sometimes turns the tail of a display into a short fragment such as
    # ``~( } r`` immediately before the label.  It is still a printed equation
    # label when the prefix is short and contains no prose-like word sequence.
    if len(prefix) <= 24 and len(re.findall(r"[A-Za-z]+", prefix)) <= 3:
        return True
    return bool(re.fullmatch(r"[\d\s.,]+", prefix) and _has_math_like_text(previous_nonempty))


def _has_math_like_text(text: str) -> bool:
    return any(char in MATH_LIKE_CHARS for char in text)


def _label_focused_equation_text(equation: dict[str, Any]) -> str:
    raw_tex = str(equation.get("raw_tex") or "")
    tex_label = str(equation.get("tex_label") or "")
    if raw_tex and tex_label:
        focused = _latex_row_near_label(raw_tex, tex_label)
        if focused:
            return focused
    return str(equation.get("equation") or "")


def _latex_row_near_label(raw_tex: str, tex_label: str) -> str:
    lines = raw_tex.splitlines()
    label_pattern = re.compile(r"\\label\{" + re.escape(tex_label) + r"\}")
    for label_index, line in enumerate(lines):
        match = label_pattern.search(line)
        if not match:
            continue
        start = _label_target_line(lines, label_index, match)
        if start is None:
            return ""
        return _normalize_latex(_collect_latex_row(lines, start, label_pattern), "label_focus")
    return ""


def _label_target_line(lines: list[str], label_index: int, label_match: re.Match[str]) -> int | None:
    line = lines[label_index]
    before_label = _strip_environment_commands(line[: label_match.start()]).replace(r"\\", "").strip()
    after_label = _strip_environment_commands(line[label_match.end() :]).strip()
    if _has_latex_content(after_label):
        return label_index
    if _has_latex_content(before_label):
        return _row_start_before(lines, label_index)
    for index in range(label_index + 1, len(lines)):
        if _has_latex_content(_strip_environment_commands(lines[index])):
            return index
    return None


def _row_start_before(lines: list[str], index: int) -> int:
    current = index
    while current > 0 and not _has_row_separator(lines[current - 1]):
        current -= 1
    return current


def _collect_latex_row(lines: list[str], start: int, label_pattern: re.Pattern[str]) -> str:
    row: list[str] = []
    for index in range(start, len(lines)):
        line = label_pattern.sub("", lines[index])
        if re.search(r"\\end\{[^{}]+\*?\}", line):
            break
        line = _strip_environment_commands(line)
        if _has_latex_content(line):
            row.append(line)
        if _has_row_separator(line):
            break
    return "\n".join(row)


def _strip_environment_commands(line: str) -> str:
    line = re.sub(r"\\begin\{[^{}]+\*?\}", "", line)
    line = re.sub(r"\\end\{[^{}]+\*?\}", "", line)
    return line


def _has_latex_content(line: str) -> bool:
    stripped = _strip_comment(line).strip()
    if not stripped:
        return False
    if re.fullmatch(r"\\label\{[^{}]+\}", stripped):
        return False
    if re.fullmatch(r"\\\\", stripped):
        return False
    return True


def _has_row_separator(line: str) -> bool:
    return bool(re.search(r"\\\\(?:\s|$|\[|%)", line))


def _token_overlap_score(left: set[str], right: set[str]) -> int:
    return sum(_token_weight(token) for token in left.intersection(right))


def _token_weight(token: str) -> int:
    if token.isdigit():
        return 0
    if len(token) == 1:
        return 1
    return 3


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        start = offset
        offset += len(raw_line)
        spans.append((start, offset, raw_line.strip()))
    if text and (not spans or spans[-1][1] < len(text)):
        spans.append((offset, len(text), text[offset:].strip()))
    return spans


def _line_window_for_offset(
    line_spans: list[tuple[int, int, str]], offset: int, *, before_lines: int = 2, after_lines: int = 2
) -> str:
    for index, (start, end, _) in enumerate(line_spans):
        if start <= offset < end:
            first = max(0, index - before_lines)
            last = min(len(line_spans), index + 1 + after_lines)
            return "\n".join(line for _, _, line in line_spans[first:last])
    return ""


def _line_window_before_offset(line_spans: list[tuple[int, int, str]], offset: int, *, max_lines: int = 8) -> str:
    for index, (start, end, _) in enumerate(line_spans):
        if start <= offset < end:
            first = max(0, index - max_lines)
            return "\n".join(line for _, _, line in line_spans[first:index])
    return ""


def _line_window_after_offset(line_spans: list[tuple[int, int, str]], offset: int, *, max_lines: int = 8) -> str:
    for index, (start, end, _) in enumerate(line_spans):
        if start <= offset < end:
            last = min(len(line_spans), index + 1 + max_lines)
            return "\n".join(line for _, _, line in line_spans[index + 1 : last])
    return ""


def _search_text(text: str) -> str:
    return " ".join(_search_tokens(text))


def _search_tokens(text: str) -> list[str]:
    normalized = (
        str(text or "")
        .replace("\\", " ")
        .replace("^", " ")
        .replace("_", " ")
        .replace("{", " ")
        .replace("}", " ")
    )
    for symbol, name in GREEK_TOKEN_NAMES.items():
        normalized = normalized.replace(symbol, f" {name} ")
    return [token.lower() for token in re.findall(r"\w+", normalized)]


def _fill_section_pdf_pages(sections: list[dict[str, Any]], equations: list[dict[str, Any]]) -> None:
    pages_by_section: dict[str, list[int]] = {}
    for equation in equations:
        page = equation.get("pdf_page")
        section_id = str(equation.get("section_id") or "")
        if isinstance(page, int) and section_id:
            pages_by_section.setdefault(section_id, []).append(page)
    for section in sections:
        pages = pages_by_section.get(str(section.get("section_id") or ""), [])
        if pages:
            section["pdf_page_start"] = min(pages)
            section["pdf_page_end"] = max(pages)


def _strip_comment(line: str) -> str:
    out = []
    escaped = False
    for char in line:
        if char == "%" and not escaped:
            break
        out.append(char)
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    return "".join(out)


def _tex_comment(line: str) -> str:
    escaped = False
    for index, char in enumerate(line):
        if char == "%" and not escaped:
            return line[index + 1 :]
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    return ""


def _clean_text(text: str) -> str:
    return " ".join((text or "").split())


def _combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
