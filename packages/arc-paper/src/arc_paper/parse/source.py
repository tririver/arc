from __future__ import annotations

import hashlib
import random
import re
import subprocess
from pathlib import Path
from typing import Any

from .ar5iv_html import parse_html


PARSER_VERSION = 9
DISPLAY_ENVIRONMENTS = ("equation", "align", "gather", "multline", "eqnarray")
SECTION_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
EQUATION_NUMBER_PATTERN = r"[A-Za-z]?\d+(?:\.\d+)+|\d+(?:\.\d+)*[A-Za-z]?"
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
    pdf_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = _resolve_inputs(source_path=source_path, html_path=html_path, tex_path=tex_path, pdf_path=pdf_path)
    paper_id = source_id or _generated_source_id()
    if html_text is not None:
        parsed = parse_html(html_text, paper_id=paper_id)
        return _canonical(parsed, paper_id=paper_id, source_hash=_sha256_text(html_text))
    if resolved["html_path"]:
        path = Path(resolved["html_path"])
        data = path.read_bytes()
        parsed = parse_html(data.decode("utf-8"), paper_id=paper_id)
        return _canonical(parsed, paper_id=paper_id, source_hash=_sha256_bytes(data))
    if resolved["tex_path"]:
        return parse_tex_document(Path(resolved["tex_path"]), paper_id=paper_id, pdf_path=resolved["pdf_path"])
    if resolved["pdf_path"]:
        return parse_pdf_document(Path(resolved["pdf_path"]), paper_id=paper_id)
    raise ValueError("parse_source_input requires an HTML, TeX, PDF, or ar5iv source")


def parse_source_input_with_warnings(
    *,
    source_path: str | Path | None = None,
    source_id: str | None = None,
    html_path: str | Path | None = None,
    html_text: str | None = None,
    tex_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = _resolve_inputs(source_path=source_path, html_path=html_path, tex_path=tex_path, pdf_path=pdf_path)
    paper_id = source_id or _generated_source_id()
    if html_text is not None:
        parsed = parse_html(html_text, paper_id=paper_id)
        return _canonical(parsed, paper_id=paper_id, source_hash=_sha256_text(html_text)), []
    if resolved["html_path"]:
        path = Path(resolved["html_path"])
        data = path.read_bytes()
        parsed = parse_html(data.decode("utf-8"), paper_id=paper_id)
        return _canonical(parsed, paper_id=paper_id, source_hash=_sha256_bytes(data)), []
    if resolved["tex_path"]:
        if resolved["pdf_path"]:
            return parse_tex_document_with_warnings(
                Path(resolved["tex_path"]), paper_id=paper_id, pdf_path=resolved["pdf_path"]
            )
        return parse_tex_document(Path(resolved["tex_path"]), paper_id=paper_id), []
    if resolved["pdf_path"]:
        return parse_pdf_document_with_warnings(Path(resolved["pdf_path"]), paper_id=paper_id)
    raise ValueError("parse_source_input requires an HTML, TeX, PDF, or ar5iv source")


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
    pages = [page.strip() for page in completed.stdout.split("\f") if page.strip()]
    if not pages:
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
    sections = _tex_sections(path, lines)
    equations = _tex_equations(path, lines, sections)
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
    return {
        "paper_id": paper_id,
        "parser_version": PARSER_VERSION,
        "source_hash": source_hash,
        "toc": list(parsed.get("toc") or []),
        "sections": list(parsed.get("sections") or []),
        "equations": list(parsed.get("equations") or []),
    }


def _resolve_inputs(
    *,
    source_path: str | Path | None,
    html_path: str | Path | None,
    tex_path: str | Path | None,
    pdf_path: str | Path | None,
) -> dict[str, Path | None]:
    resolved = {
        "html_path": Path(html_path) if html_path else None,
        "tex_path": Path(tex_path) if tex_path else None,
        "pdf_path": Path(pdf_path) if pdf_path else None,
    }
    if source_path:
        path = Path(source_path)
        suffix = path.suffix.lower()
        if suffix in {".html", ".htm"}:
            resolved["html_path"] = path
        elif suffix == ".tex":
            resolved["tex_path"] = path
        elif suffix == ".pdf":
            resolved["pdf_path"] = path
        else:
            raise ValueError(f"Cannot infer parse source type from extension: {path}")
    return resolved


def _generated_source_id() -> str:
    return f"arc-{random.SystemRandom().randrange(100000000):08d}"


def _tex_sections(path: Path, lines: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        match = re.search(r"\\(section|subsection|subsubsection)\*?\s*\{([^{}]*)\}", line)
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
    return {
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
    match = re.search(r"\\begin\{(" + "|".join(DISPLAY_ENVIRONMENTS) + r")\*?\}", line)
    return match.group(1) if match else ""


def _find_environment_end(lines: list[str], start: int, env: str) -> int:
    pattern = re.compile(r"\\end\{" + re.escape(env) + r"\*?\}")
    for index in range(start, len(lines)):
        if pattern.search(lines[index]):
            return index
    return start


def _display_math_span(lines: list[str], start: int) -> tuple[int, int, str] | None:
    line = lines[start]
    if r"\[" in line:
        return start, _find_display_end(lines, start, r"\]"), "display_math"
    if "$$" in line:
        if line.count("$$") >= 2:
            return start, start, "dollar_display"
        return start, _find_display_end(lines, start + 1, "$$"), "dollar_display"
    return None


def _find_display_end(lines: list[str], start: int, token: str) -> int:
    for index in range(start, len(lines)):
        if token in lines[index]:
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
    match = re.search(r"\\label\{([^{}]+)\}", line)
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
    cleaned = re.sub(rf"\(({EQUATION_NUMBER_PATTERN})\)\s*$", "", line).strip()
    if "=" not in cleaned:
        return ""
    tokens = _search_tokens(cleaned)
    if len(tokens) < 3:
        return ""
    return cleaned


def _line_equation_number(line: str) -> str | None:
    match = re.search(rf"\(({EQUATION_NUMBER_PATTERN})\)\s*$", line)
    return match.group(1) if match else None


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
    for equation in equations:
        best_page = None
        best_score = 0
        for page_index, page in enumerate(pages, start=1):
            score = _pdf_match_score(equation, page)
            if score > best_score:
                best_score = score
                best_page = (page_index, page)
        if best_page and best_score >= 4:
            equation["pdf_page"] = best_page[0]
            if number := _best_equation_number(best_page[1], equation):
                equation["printed_equation_number"] = number


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
    candidates = [(match.group(1), match.start()) for match in re.finditer(rf"\(({EQUATION_NUMBER_PATTERN})\)", page)]
    if not candidates:
        return None
    equation_tokens = set(_search_tokens(str(equation.get("equation") or "")))
    if not equation_tokens:
        return candidates[0][0]
    before_tokens = set(_search_tokens(str(equation.get("before") or "")))
    after_tokens = set(_search_tokens(str(equation.get("after") or "")))
    line_spans = _line_spans(page)
    best_number = candidates[0][0]
    best_score = -1
    for number, offset in candidates:
        equation_window = _line_window_for_offset(line_spans, offset)
        before_window = _line_window_before_offset(line_spans, offset)
        after_window = _line_window_after_offset(line_spans, offset)
        before_score = _token_overlap_score(before_tokens, set(_search_tokens(before_window)))
        after_score = _token_overlap_score(after_tokens, set(_search_tokens(after_window)))
        equation_score = _token_overlap_score(equation_tokens, set(_search_tokens(equation_window)))
        bracket_bonus = 50 if before_score and after_score else 0
        score = bracket_bonus + (before_score * 4) + (after_score * 4) + equation_score
        if score > best_score:
            best_score = score
            best_number = number
    return best_number


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


def _line_window_for_offset(line_spans: list[tuple[int, int, str]], offset: int) -> str:
    for index, (start, end, _) in enumerate(line_spans):
        if start <= offset < end:
            first = max(0, index - 2)
            last = min(len(line_spans), index + 2)
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
