from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


_PAGE_LABEL = r"(?:[ivxlcdm]+|\d+)(?:\s*[-–—]\s*(?:[ivxlcdm]+|\d+))?"
_ENTRY_RE = re.compile(
    rf"^(?P<indent>\s*)(?P<title>.+?)(?:\s*\.{{2,}}\s*|\s{{2,}}|,\s*)(?P<pages>{_PAGE_LABEL}(?:\s*,\s*{_PAGE_LABEL})*)\s*$",
    re.IGNORECASE,
)
_SEE_ALSO_RE = re.compile(r"^(?P<title>.+?),\s*see\s+also\s+(?P<targets>.+)$", re.IGNORECASE)
_SEE_RE = re.compile(r"^(?P<title>.+?),\s*see\s+(?P<targets>.+)$", re.IGNORECASE)


def title_fingerprint(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"^\s*(?:chapter|part|section)?\s*[\divxlcdm]+(?:\.\d+)*[.)\-:]?\s+", "", text)
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def read_embedded_outline(
    path: str | Path, *, reader_factory: Callable[[str | Path], Any] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a PDF outline and displayed page labels.

    pypdf is optional.  Tests and hosts without it can inject a compatible
    reader factory; absence simply permits the printed-TOC fallback.
    """

    if reader_factory is None:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            return [], []
        reader_factory = PdfReader
    try:
        reader = reader_factory(path)
    except Exception:
        return [], []
    page_labels = [str(value) for value in (getattr(reader, "page_labels", None) or [])]
    raw_outline = getattr(reader, "outline", None) or getattr(reader, "outlines", None) or []
    entries: list[dict[str, Any]] = []

    def visit(values: Iterable[Any], level: int) -> None:
        previous: dict[str, Any] | None = None
        for value in values:
            if isinstance(value, list):
                visit(value, level + 1)
                continue
            title = str(getattr(value, "title", "") or "").strip()
            if not title:
                continue
            try:
                physical_page = int(reader.get_destination_page_number(value)) + 1
            except Exception:
                physical_page = None
            previous = {
                "title": title,
                "level": level,
                "physical_page": physical_page,
                "printed_page": (
                    page_labels[physical_page - 1]
                    if physical_page and physical_page <= len(page_labels)
                    else ""
                ),
            }
            entries.append(previous)

    visit(raw_outline, 1)
    return entries, page_labels


def parse_printed_toc(
    pages: Iterable[str],
    *,
    page_labels: Iterable[str] | None = None,
    excluded_pages: Iterable[int] = (),
) -> tuple[list[dict[str, Any]], list[int]]:
    page_list = list(pages)
    labels = list(page_labels or [])
    excluded = set(excluded_pages)
    candidates: list[tuple[int, list[dict[str, Any]]]] = []
    for physical_page, page in enumerate(page_list, start=1):
        if physical_page in excluded:
            continue
        parsed = _parse_locator_lines(page, labels=labels, toc=True)
        numbered = sum(_looks_numbered_title(str(item.get("title") or "")) for item in parsed)
        if len(parsed) >= 3 and numbered * 2 >= len(parsed):
            candidates.append((physical_page, parsed))
    if not candidates:
        return [], []
    early_limit = max(3, (len(page_list) + 2) // 3)
    candidates = [candidate for candidate in candidates if candidate[0] <= early_limit]
    if not candidates:
        return [], []
    # A printed contents occupies a contiguous early run.  Stop at the first
    # gap so bibliography/index columns later in the PDF are not mistaken for it.
    run = [candidates[0]]
    for candidate in candidates[1:]:
        if candidate[0] != run[-1][0] + 1:
            break
        run.append(candidate)
    entries = [item for _, page_entries in run for item in page_entries]
    for item in entries:
        item.pop("_indent", None)
    return entries, [page for page, _ in run]


def parse_index_entries(
    pages: Iterable[str],
    *,
    page_labels: Iterable[str] | None = None,
    start_page: int | None = None,
) -> dict[str, Any]:
    page_list = list(pages)
    labels = list(page_labels or [])
    parsed_by_page = [
        _parse_locator_lines(page, labels=labels, toc=False)
        for page in page_list
    ]
    if start_page is None:
        headings = [
            index + 1 for index, page in enumerate(page_list)
            if _has_explicit_index_heading(page)
        ]
        if not headings:
            return {
                "schema_version": "arc.paper.index_entries.v1",
                "entries": [],
                "source_pages": [],
                "raw_lines": [],
            }
        # Automatic extraction requires explicit structural evidence. Locator
        # density alone also matches bibliographies and numbered appendices.
        source_pages = list(range(headings[-1], len(page_list) + 1))
    else:
        source_pages = list(range(max(1, start_page), len(page_list) + 1))
    flat: list[dict[str, Any]] = []
    for page in source_pages:
        lines = page_list[page - 1].splitlines()
        for line_index, raw in enumerate(lines):
            parsed_line = _parse_locator_lines(raw, labels=labels, toc=False)
            if parsed_line:
                flat.extend(parsed_line)
                continue
            stripped = raw.strip()
            indent = len(raw) - len(raw.lstrip())
            next_indent = next(
                (
                    len(candidate) - len(candidate.lstrip())
                    for candidate in lines[line_index + 1 :]
                    if candidate.strip()
                ),
                -1,
            )
            if stripped and next_indent > indent and not re.fullmatch(_PAGE_LABEL, stripped, re.IGNORECASE):
                flat.append({
                    "term": stripped,
                    "raw": raw,
                    "page_ranges": [],
                    "see": [],
                    "see_also": [],
                    "children": [],
                    "unlocated_parent": True,
                    "_indent": indent,
                })
    roots: list[dict[str, Any]] = []
    stack: list[tuple[int, dict[str, Any]]] = []
    for item in flat:
        indent = int(item.pop("_indent", 0))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack and indent > stack[-1][0]:
            stack[-1][1].setdefault("children", []).append(item)
        else:
            roots.append(item)
        stack.append((indent, item))
    return {
        "schema_version": "arc.paper.index_entries.v1",
        "entries": roots,
        "source_pages": source_pages,
        "raw_lines": [
            {"physical_page": page, "line": line_number, "text": line}
            for page in source_pages
            for line_number, line in enumerate(page_list[page - 1].splitlines(), start=1)
            if line.strip()
        ],
    }


def reconcile_headings_to_pages(
    headings: Iterable[dict[str, Any]],
    pages: Iterable[str],
    *,
    authority_entries: Iterable[dict[str, Any]] | None = None,
    excluded_pages: Iterable[int] = (),
) -> list[dict[str, Any]]:
    """Return the unique monotonic heading/page alignment or fail closed."""

    page_list = list(pages)
    excluded = set(excluded_pages)
    authority = list(authority_entries or [])
    candidates: list[list[int]] = []
    heading_list = list(headings)
    for heading in heading_list:
        fingerprint = title_fingerprint(str(heading.get("title") or ""))
        if not fingerprint:
            raise ValueError("Cannot reconcile an empty source heading to the PDF.")
        authoritative_pages = sorted({
            int(item["physical_page"])
            for item in authority
            if item.get("physical_page")
            and title_fingerprint(str(item.get("title") or "")) == fingerprint
        })
        if authoritative_pages:
            matches = authoritative_pages
        else:
            matches = [
                page_number
                for page_number, text in enumerate(page_list, start=1)
                if page_number not in excluded and _fingerprint_in_page(fingerprint, text)
            ]
        prose_matches = _heading_prose_pages(heading, page_list, excluded)
        if matches and prose_matches:
            refined = sorted(set(matches).intersection(prose_matches))
            if refined:
                matches = refined
        elif not matches and prose_matches:
            # Formula-heavy or OCR-normalized headings may not survive
            # verbatim, but their immediately following source prose can
            # still provide a unique PDF-side anchor.
            matches = prose_matches
        if not authoritative_pages and len(matches) != 1:
            page_hint = heading.get("pdf_page_start")
            if isinstance(page_hint, int) and (not matches or page_hint in matches):
                # Paired parsing can infer a section page from one of its
                # equations. Use that as a tie-breaker or last resort, not as
                # stronger evidence than a uniquely matched heading/prose
                # anchor: a section commonly begins before its first equation.
                matches = [page_hint]
        if not matches:
            raise ValueError(f"PDF reconciliation found no page for heading {heading.get('title')!r}.")
        candidates.append(matches)

    solutions: list[list[int]] = []

    def search(index: int, prior: int, selected: list[int]) -> None:
        if len(solutions) > 1:
            return
        if index == len(candidates):
            solutions.append(list(selected))
            return
        for page in candidates[index]:
            if page >= prior:
                search(index + 1, page, [*selected, page])

    search(0, 1, [])
    if len(solutions) != 1:
        raise ValueError("PDF reconciliation is ambiguous; heading/page alignment is not unique.")
    selected = solutions[0]
    return [
        {
            "section_id": str(heading.get("section_id") or ""),
            "title_fingerprint": title_fingerprint(str(heading.get("title") or "")),
            "pdf_page_start": page,
            "pdf_page_end": (
                max(page, selected[index + 1] - 1)
                if index + 1 < len(selected)
                else len(page_list)
            ),
        }
        for index, (heading, page) in enumerate(zip(heading_list, selected, strict=True))
    ]


def _heading_prose_pages(
    heading: dict[str, Any], pages: list[str], excluded: set[int]
) -> list[int]:
    text_tokens = title_fingerprint(str(heading.get("text") or "")).split()
    title_tokens = title_fingerprint(str(heading.get("title") or "")).split()
    if title_tokens:
        title_start = next(
            (
                index for index in range(min(8, len(text_tokens)))
                if text_tokens[index : index + len(title_tokens)] == title_tokens
            ),
            None,
        )
        if title_start is not None:
            text_tokens = text_tokens[title_start + len(title_tokens) :]
    # A short leading quotation or sentence is normally unique while staying
    # robust to line wrapping and minor rich-source/PDF whitespace changes.
    for length in (12, 8, 5):
        if len(text_tokens) < length:
            continue
        fingerprint = " ".join(text_tokens[:length])
        matches = [
            page_number for page_number, page in enumerate(pages, 1)
            if page_number not in excluded and _fingerprint_in_page(fingerprint, page)
        ]
        if matches:
            return matches
    return []


def reconcile_blocks_to_pages(
    blocks: Iterable[dict[str, Any]],
    pages: Iterable[str],
    *,
    section_anchors: Iterable[dict[str, Any]],
    equations: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Map every rich source block to one unique monotonic PDF start page."""

    block_list = list(blocks)
    page_list = list(pages)
    sections = {str(item.get("section_id") or ""): item for item in section_anchors}
    equations_by_id = {str(item.get("id") or ""): item for item in equations}
    candidates: list[list[int]] = []
    methods: list[str] = []
    in_transport_details = False
    for block in block_list:
        section = sections.get(str(block.get("section_id") or ""))
        lower = int((section or {}).get("pdf_page_start") or 1)
        upper = int((section or {}).get("pdf_page_end") or len(page_list))
        source_id = str(block.get("source_id") or block.get("block_id") or "")
        equation_page = (equations_by_id.get(source_id) or {}).get("pdf_page")
        raw_text = str(block.get("text") or "")
        transport_block = in_transport_details or bool(re.search(r"<\s*details\b", raw_text, re.IGNORECASE))
        if re.search(r"<\s*details\b", raw_text, re.IGNORECASE):
            in_transport_details = True
        if isinstance(equation_page, int) and lower <= equation_page <= upper:
            matches = [equation_page]
            method = "equation_anchor"
        elif str(block.get("kind") or "") == "heading" and section:
            matches = [lower]
            method = "section_heading_anchor"
        else:
            matches = _block_text_pages(raw_text, page_list, lower, upper)
            method = "text_fingerprint"
        if (
            len(matches) > 1
            and candidates
            and len(candidates[-1]) == 1
            and candidates[-1][0] in matches
            and str(block_list[len(candidates) - 1].get("section_id") or "")
            == str(block.get("section_id") or "")
        ):
            matches = list(candidates[-1])
            method = "preceding_source_anchor"
        if (
            matches
            and candidates
            and len(candidates[-1]) == 1
            and max(matches) < candidates[-1][0]
            and str(block_list[len(candidates) - 1].get("section_id") or "")
            == str(block.get("section_id") or "")
        ):
            # A repeated phrase can point backward inside the same section.
            # Preserve the PDF-authoritative monotonic source order and record
            # that the preceding block, not the repeated phrase, chose the page.
            matches = list(candidates[-1])
            method = "preceding_source_anchor"
        if (
            not matches
            and candidates
            and str(block_list[len(candidates) - 1].get("section_id") or "")
            == str(block.get("section_id") or "")
        ):
            # Some formula-heavy OCR prose and non-text figures have no stable
            # text-layer fingerprint. Source order binds them to the preceding
            # reconciled block within the same PDF-authoritative section; the
            # proof records this weaker method explicitly.
            matches = list(candidates[-1])
            method = "preceding_source_anchor"
        if re.search(r"<\s*/\s*details\s*>", raw_text, re.IGNORECASE):
            in_transport_details = False
        if not matches and lower == upper:
            matches = [lower]
            method = "single_page_section"
        if not matches:
            raise ValueError(
                f"PDF reconciliation found no unique evidence for block {block.get('block_id')!r}."
            )
        candidates.append(matches)
        methods.append(method)
    selected = _unique_monotonic_solution(candidates, label="block/page")
    return [
        {
            "block_id": str(block.get("block_id") or ""),
            "section_id": str(block.get("section_id") or ""),
            "source_fingerprint": title_fingerprint(str(block.get("text") or "")),
            "pdf_page_start": page,
            "pdf_page_end": page,
            "alignment_method": method,
        }
        for block, page, method in zip(block_list, selected, methods, strict=True)
    ]


def _parse_locator_lines(text: str, *, labels: list[str], toc: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw in str(text or "").splitlines():
        if not raw.strip():
            continue
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        see_also = _SEE_ALSO_RE.match(stripped)
        see = _SEE_RE.match(stripped) if see_also is None else None
        if not toc and (see_also or see):
            match = see_also or see
            assert match is not None
            entries.append({
                "term": match.group("title").strip(),
                "raw": raw,
                "page_ranges": [],
                "see": [] if see_also else _split_targets(match.group("targets")),
                "see_also": _split_targets(match.group("targets")) if see_also else [],
                "children": [],
                "_indent": indent,
            })
            continue
        match = _ENTRY_RE.match(raw)
        if match is None:
            continue
        title = match.group("title").strip()
        if not title or not title_fingerprint(title):
            continue
        ranges = [_page_range(value.strip(), labels) for value in match.group("pages").split(",")]
        base = {
            "raw": raw,
            "page_ranges": ranges,
            "_indent": len(match.group("indent")),
        }
        if toc:
            numbered_level = _numbered_title_level(title)
            base.update({
                "title": title,
                "level": numbered_level or (1 + min(5, len(match.group("indent")) // 2)),
                "printed_page": ranges[0]["printed_start"],
                "physical_page": ranges[0].get("physical_start"),
            })
        else:
            base.update({"term": title, "see": [], "see_also": [], "children": []})
        entries.append(base)
    return entries


def _page_range(value: str, labels: list[str]) -> dict[str, Any]:
    parts = re.split(r"\s*[-–—]\s*", value, maxsplit=1)
    start = parts[0]
    end = parts[1] if len(parts) == 2 else start
    return {
        "printed_start": start,
        "printed_end": end,
        "physical_start": _physical_page(start, labels),
        "physical_end": _physical_page(end, labels),
    }


def _physical_page(label: str, labels: list[str]) -> int | None:
    normalized = str(label).strip().casefold()
    matches = [index + 1 for index, value in enumerate(labels) if str(value).strip().casefold() == normalized]
    return matches[0] if len(matches) == 1 else None


def _split_targets(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"\s*;\s*|\s*,\s*", value) if item.strip()]


def _has_explicit_index_heading(page: str) -> bool:
    lines = [unicodedata.normalize("NFKC", line).strip().casefold() for line in str(page).splitlines()]
    first = [line for line in lines if line][:5]
    return any(line in {"index", "subject index", "author index"} for line in first)


def _fingerprint_in_page(fingerprint: str, page: str) -> bool:
    page_fingerprint = title_fingerprint(page)
    return f" {fingerprint} " in f" {page_fingerprint} "


def _block_text_pages(text: str, pages: list[str], lower: int, upper: int) -> list[int]:
    tokens = title_fingerprint(text).split()
    if not tokens:
        return []
    lengths = [min(12, len(tokens)), min(8, len(tokens)), min(4, len(tokens)), min(2, len(tokens))]
    for length in dict.fromkeys(value for value in lengths if value >= 2):
        offsets = list(dict.fromkeys([
            0,
            max(0, len(tokens) - length),
            *range(0, max(1, len(tokens) - length + 1), max(1, length // 2)),
        ]))
        evidence: list[list[int]] = []
        for offset in offsets[:80]:
            fingerprint = " ".join(tokens[offset : offset + length])
            matches = [
                page_number
                for page_number in range(max(1, lower), min(len(pages), upper) + 1)
                if _fingerprint_in_page(fingerprint, pages[page_number - 1])
            ]
            if matches:
                unique_matches = sorted(set(matches))
                if len(unique_matches) == 1:
                    return unique_matches
                evidence.append(unique_matches)
        if evidence:
            consensus = set(evidence[0])
            for match_set in evidence[1:]:
                consensus.intersection_update(match_set)
            if consensus:
                ordered = sorted(consensus)
                # Long source blocks may span a PDF page boundary; their
                # start-page mapping is the earliest page carrying the same
                # block evidence. Keep short repeated labels ambiguous.
                return [ordered[0]] if len(tokens) >= 12 else ordered
            uniquely_supported = {matches[0] for matches in evidence if len(matches) == 1}
            if len(uniquely_supported) == 1:
                return sorted(uniquely_supported)
    return []


def _unique_monotonic_solution(candidates: list[list[int]], *, label: str) -> list[int]:
    if not candidates:
        return []
    states: dict[int, int] = {}
    predecessors: list[dict[int, int | None]] = []
    for index, raw_pages in enumerate(candidates):
        layer: dict[int, int] = {}
        layer_predecessors: dict[int, int | None] = {}
        for page in sorted(set(raw_pages)):
            if index == 0:
                count, predecessor = 1, None
            else:
                eligible = [prior for prior, count in states.items() if prior <= page and count]
                count = min(2, sum(states[prior] for prior in eligible))
                predecessor = eligible[0] if count == 1 and len(eligible) == 1 and states[eligible[0]] == 1 else None
            if count:
                layer[page] = count
                layer_predecessors[page] = predecessor
        states = layer
        predecessors.append(layer_predecessors)
        if not states:
            break
    if sum(states.values()) != 1:
        raise ValueError(f"PDF reconciliation is ambiguous; {label} alignment is not unique.")
    page = next(value for value, count in states.items() if count == 1)
    selected = [page]
    for index in range(len(predecessors) - 1, 0, -1):
        prior = predecessors[index].get(page)
        if prior is None:
            raise ValueError(f"PDF reconciliation is ambiguous; {label} alignment is not unique.")
        selected.append(prior)
        page = prior
    return list(reversed(selected))


def _looks_numbered_title(value: str) -> bool:
    return bool(re.match(r"^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)\-:]?\s+\S", value, re.IGNORECASE))


def _numbered_title_level(value: str) -> int | None:
    match = re.match(r"^\s*(?P<number>\d+(?:\.\d+)*)[.)\-:]?\s+\S", value)
    if match:
        return match.group("number").count(".") + 1
    return 1 if re.match(r"^\s*[ivxlcdm]+[.)\-:]?\s+\S", value, re.IGNORECASE) else None
