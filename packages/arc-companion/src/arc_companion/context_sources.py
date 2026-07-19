from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable

from .evidence import arc_cache_descriptor, text_sha256, validate_evidence_record


CONTEXT_INDEX_VERSION = "arc.companion.context-index.v3"


class ContextSourceError(RuntimeError):
    """Raised when an explicitly requested local ARC context source is unavailable."""


def load_context_evidence(
    paper_ids: Iterable[str],
    *,
    parsed_getter: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load an auditable lightweight index exclusively from the local arc-paper cache.

    Selection and prompt-size limits are deliberately applied later, per source
    segment.  Pre-sampling here can permanently discard the only relevant block
    from a long reference before the segment query is known.
    """
    if parsed_getter is None:
        from arc_paper import service

        parsed_getter = service.get_parsed_source
    records: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        source_id = str(paper_id).strip()
        if not source_id:
            raise ContextSourceError("context paper IDs must not be empty")
        parsed = _local_parsed_source(source_id, parsed_getter=parsed_getter)
        pieces = _source_pieces(parsed)
        if not pieces:
            raise ContextSourceError(
                f"Local arc-paper cache for {source_id} contains no usable sections, equations, or blocks"
            )
        metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
        document = parsed.get("document") if isinstance(parsed.get("document"), dict) else {}
        document_metadata = (
            document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        )
        title = _source_title(parsed, metadata=metadata, fallback=source_id)
        authors = (
            metadata.get("authors") or parsed.get("authors")
            or document_metadata.get("authors") or document.get("authors") or []
        )
        year = (
            metadata.get("year") or parsed.get("year")
            or document_metadata.get("year") or document.get("year")
        )
        digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:20]
        record = {
            "evidence_id": f"context-{digest}",
            "relation": "context",
            "paper_id": source_id,
            "title": title,
            "authors": authors,
            "year": year,
            "evidence_level": "full_text",
            "abstract": str(metadata.get("abstract") or parsed.get("abstract") or ""),
            "blocks": pieces,
            "context_role": "explanation_and_conceptual_connections_only",
            "context_index": {
                "version": CONTEXT_INDEX_VERSION,
                "block_count": len(pieces),
                "indexed_chars": sum(len(item["text"]) for item in pieces),
            },
        }
        record["source_descriptor"] = arc_cache_descriptor(
            paper_id=source_id,
            title=title,
            authors=authors,
            year=year,
            evidence_level="full_text",
            content=pieces,
            document_hash=str(parsed.get("source_hash") or parsed.get("document_hash") or ""),
        )
        validate_evidence_record(record)
        records.append(record)
    return records


def _source_title(parsed: dict[str, Any], *, metadata: dict[str, Any], fallback: str) -> str:
    explicit = str(metadata.get("title") or parsed.get("title") or "").strip()
    if explicit:
        return explicit
    document = parsed.get("document") if isinstance(parsed.get("document"), dict) else {}
    document_metadata = (
        document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    )
    document_title = str(document_metadata.get("title") or document.get("title") or "").strip()
    if document_title:
        return document_title
    for owner, collection_name in (
        (parsed, "toc"), (document, "toc"), (parsed, "sections"), (document, "sections")
    ):
        collection = owner.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if isinstance(item, dict) and str(item.get("title") or "").strip():
                return str(item["title"]).strip()
    return fallback


def _local_parsed_source(
    paper_id: str, *, parsed_getter: Callable[..., dict[str, Any]]
) -> dict[str, Any]:
    """Use only the cache reader; never parse, fetch, or call MCP as a fallback."""
    result = parsed_getter(paper_id, include_document=True)
    if not (isinstance(result, dict) and result.get("ok")):
        result = parsed_getter(paper_id, include_document=False)
    if not (isinstance(result, dict) and result.get("ok")):
        error = result.get("error") if isinstance(result, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        raise ContextSourceError(
            f"Unable to load context source {paper_id} from the local arc-paper cache: "
            f"{message or 'cache entry not found'}"
        )
    parsed = result.get("data")
    if not isinstance(parsed, dict):
        raise ContextSourceError(f"Local arc-paper cache returned invalid data for {paper_id}")
    return parsed


def _source_pieces(parsed: dict[str, Any]) -> list[dict[str, str]]:
    document = parsed.get("document") if isinstance(parsed.get("document"), dict) else {}
    rich_blocks = document.get("blocks") if isinstance(document.get("blocks"), list) else []
    if rich_blocks:
        candidates: list[dict[str, str] | None] = []
        section_title = ""
        for index, value in enumerate(rich_blocks, 1):
            explicit = _block_section_title(value)
            if explicit:
                section_title = explicit
            candidates.append(
                _piece_from_block(value, index, section_title=section_title)
            )
        return [item for item in candidates if item]

    sections = parsed.get("sections") if isinstance(parsed.get("sections"), list) else []
    equations = parsed.get("equations") if isinstance(parsed.get("equations"), list) else []
    section_pieces = [
        item for index, value in enumerate(sections, 1)
        if (item := _piece_from_section(value, index))
    ]
    equation_pieces = [
        item for index, value in enumerate(equations, 1)
        if (item := _piece_from_equation(value, index))
    ]
    return [*section_pieces, *equation_pieces]


def _piece_from_block(
    value: Any, index: int, *, section_title: str = ""
) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    text = str(value.get("text") or value.get("title") or value.get("caption") or "").strip()
    locator = str(value.get("block_id") or value.get("id") or f"block-{index}")
    return _piece(locator, text, section_title=section_title)


def _piece_from_section(value: Any, index: int) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    title = str(value.get("title") or "").strip()
    body = str(value.get("text") or "").strip()
    text = "\n".join(item for item in (title, body) if item)
    locator = str(value.get("section_id") or f"section-{index}")
    return _piece(locator, text, section_title=title)


def _piece_from_equation(value: Any, index: int) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    before = str(value.get("before") or "").strip()
    equation = str(
        value.get("normalized_latex") or value.get("raw_tex") or value.get("equation") or ""
    ).strip()
    after = str(value.get("after") or "").strip()
    text = "\n".join(item for item in (before[-300:], equation, after[:300]) if item)
    locator = str(value.get("id") or value.get("equation_id") or f"equation-{index}")
    return _piece(locator, text)


def _piece(
    locator: str, text: str, *, section_title: str = ""
) -> dict[str, str] | None:
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    piece = {"block_id": locator, "text": cleaned, "sha256": text_sha256(cleaned)}
    normalized_title = " ".join(str(section_title or "").split())
    if normalized_title:
        piece["section_title"] = normalized_title
    return piece


def _block_section_title(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    explicit = value.get("section_title")
    if explicit:
        return " ".join(str(explicit).split())
    heading = value.get("heading") if isinstance(value.get("heading"), dict) else {}
    kind = str(value.get("kind") or value.get("type") or "").casefold()
    is_heading = bool(
        kind in {"heading", "section", "subsection", "subsubsection"}
        or value.get("heading_level")
        or heading
    )
    if not is_heading:
        return ""
    title = (
        value.get("title") or heading.get("title") or heading.get("text")
        or value.get("text") or ""
    )
    return " ".join(str(title).split())
