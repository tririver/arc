from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .evidence import arc_cache_descriptor, text_sha256, validate_evidence_record


class SourceError(RuntimeError):
    """Raised when arc-paper cannot provide a complete source document."""


@dataclass(frozen=True)
class SourceBundle:
    paper_id: str
    parsed: dict[str, Any]
    document: dict[str, Any]
    metadata: dict[str, Any]
    references: list[dict[str, Any]]
    citers: list[dict[str, Any]]
    diagnostics: tuple[dict[str, str], ...] = ()
    related_evidence: tuple[dict[str, Any], ...] = ()


def load_source_bundle(
    paper_id: str,
    *,
    refresh: bool = False,
    recache: bool = False,
    parse: Callable[..., dict[str, Any]] | None = None,
    metadata_getter: Callable[..., dict[str, Any]] | None = None,
    references_getter: Callable[..., dict[str, Any]] | None = None,
    citers_getter: Callable[..., dict[str, Any]] | None = None,
) -> SourceBundle:
    if parse is None or metadata_getter is None or references_getter is None or citers_getter is None:
        from arc_paper import service

        parse = parse or service.parse_source
        metadata_getter = metadata_getter or service.get_metadata
        references_getter = references_getter or service.get_references
        citers_getter = citers_getter or service.get_citers

    result = parse(
        source="ar5iv",
        paper_id=paper_id,
        include_document=True,
        refresh=refresh,
        recache=recache,
    )
    parsed = _unwrap(result, "complete parsed source")
    if not isinstance(parsed, dict):
        raise SourceError("arc-paper returned a non-object parsed source")
    document = parsed.get("document")
    if not isinstance(document, dict):
        raise SourceError("arc-paper did not return document; recache this paper with the current parser")
    validate_complete_document(document)

    metadata = _required_data(
        metadata_getter(paper_id, refresh=refresh),
        label="seed metadata",
        expected_type=dict,
    )
    references = _required_data(
        references_getter(paper_id, refresh=refresh, enrich=True),
        label="seed references",
        expected_type=list,
    )
    diagnostics: list[dict[str, str]] = []
    citers = _optional_data(
        citers_getter(paper_id, refresh=refresh, limit=100, sort="mostcited"),
        label="seed citers",
        expected_type=list,
        fallback=[],
        diagnostics=diagnostics,
    )
    canonical_id = str(parsed.get("paper_id") or paper_id)
    related_evidence, related_diagnostics = _load_related_evidence(
        references,
        citers,
        parse=parse,
        refresh=refresh,
        recache=recache,
    )
    diagnostics.extend(related_diagnostics)
    return SourceBundle(
        paper_id=canonical_id,
        parsed=parsed,
        document=document,
        metadata=metadata,
        references=references,
        citers=citers,
        diagnostics=tuple(diagnostics),
        related_evidence=tuple(related_evidence),
    )


def _load_related_evidence(
    references: list[dict[str, Any]],
    citers: list[dict[str, Any]],
    *,
    parse: Callable[..., dict[str, Any]],
    refresh: bool,
    recache: bool,
    limit_per_kind: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Cache a bounded, generic full-text evidence set through arc-paper's public API."""
    selected: list[tuple[str, int, dict[str, Any], str]] = []
    for kind, items in (("prior", references), ("later", citers)):
        candidates = sorted(
            enumerate(items),
            key=lambda pair: (-_citation_count(pair[1]), pair[0]),
        )[:limit_per_kind]
        for rank, (_, item) in enumerate(candidates, 1):
            paper_id = _arxiv_identifier(item)
            if paper_id:
                selected.append((kind, rank, item, paper_id))

    def load(record: tuple[str, int, dict[str, Any], str]) -> dict[str, Any]:
        kind, rank, item, paper_id = record
        result = parse(
            source="ar5iv",
            paper_id=paper_id,
            include_document=False,
            refresh=refresh,
            recache=recache,
        )
        parsed = None
        if isinstance(result, dict) and result.get("ok") and isinstance(result.get("data"), dict):
            parsed = result["data"]
        sections = list(parsed.get("sections") or []) if isinstance(parsed, dict) else []
        compact_blocks = [
            _compact_evidence_section(section, index=index)
            for index, section in enumerate(sections, 1)
            if _evidence_text(section)
        ]
        evidence_level = "full_text" if compact_blocks else "abstract_only"
        abstract = str(item.get("abstract") or "")
        document_hash = str((parsed or {}).get("source_hash") or "")
        value = {
            "evidence_id": f"{kind}-{rank:03d}",
            "relation": kind,
            "paper_id": paper_id,
            "title": str(item.get("title") or ""),
            "authors": item.get("authors") or [],
            "year": item.get("year"),
            "citation_count": item.get("citation_count"),
            "evidence_level": evidence_level,
            "abstract": abstract,
            "blocks": compact_blocks,
        }
        value["source_descriptor"] = arc_cache_descriptor(
            paper_id=paper_id,
            title=value["title"],
            authors=value["authors"],
            year=value["year"],
            evidence_level=evidence_level,
            content=compact_blocks if compact_blocks else abstract,
            document_hash=document_hash,
        )
        validate_evidence_record(value)
        return value

    output: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, str]] = []
    if selected:
        with ThreadPoolExecutor(max_workers=min(8, len(selected))) as executor:
            futures = {executor.submit(load, record): record for record in selected}
            for future in as_completed(futures):
                kind, rank, item, paper_id = futures[future]
                try:
                    value = future.result()
                except Exception as exc:
                    abstract = str(item.get("abstract") or "")
                    if not abstract.strip():
                        diagnostics.append({
                            "severity": "warning",
                            "code": "related_evidence_unavailable",
                            "source": "arc-paper",
                            "message": (
                                f"No recordable full text or verified abstract is available for "
                                f"{paper_id}: {exc}"
                            ),
                        })
                        continue
                    value = {
                        "evidence_id": f"{kind}-{rank:03d}",
                        "relation": kind,
                        "paper_id": paper_id,
                        "title": str(item.get("title") or ""),
                        "authors": item.get("authors") or [],
                        "year": item.get("year"),
                        "citation_count": item.get("citation_count"),
                        "evidence_level": "abstract_only",
                        "abstract": abstract,
                        "blocks": [],
                    }
                    value["source_descriptor"] = arc_cache_descriptor(
                        paper_id=paper_id,
                        title=value["title"],
                        authors=value["authors"],
                        year=value["year"],
                        evidence_level="abstract_only",
                        content=abstract,
                    )
                    validate_evidence_record(value)
                    diagnostics.append({
                        "severity": "warning",
                        "code": "related_full_text_unavailable",
                        "source": "arc-paper",
                        "message": f"Unable to cache related full text for {paper_id}: {exc}",
                    })
                output[value["evidence_id"]] = value
    ordered = [output[key] for key in sorted(output)]
    return ordered, diagnostics


def _arxiv_identifier(item: dict[str, Any]) -> str:
    value = item.get("arxiv_id") or item.get("arxiv")
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("value") or value.get("id")
    text = str(value or "").strip()
    if text:
        return text
    paper_id = str(item.get("paper_id") or "").strip()
    return paper_id if "arxiv" in paper_id.casefold() else ""


def _citation_count(item: dict[str, Any]) -> int:
    try:
        return int(item.get("citation_count") or 0)
    except (TypeError, ValueError):
        return 0


def _evidence_text(block: dict[str, Any]) -> str:
    return str(block.get("text") or block.get("title") or block.get("caption") or "").strip()


def _compact_evidence_section(section: dict[str, Any], *, index: int) -> dict[str, str]:
    """Project a lightweight parsed section into an auditable evidence block."""
    text = _evidence_text(section)[:2_000]
    return {
        "block_id": str(section.get("section_id") or f"section-{index}"),
        "type": "section",
        "text": text,
        "sha256": text_sha256(text),
    }


def validate_complete_document(document: dict[str, Any]) -> None:
    blocks = document.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise SourceError("arc-paper document has no ordered content blocks")
    ids = [block_id(block) for block in blocks]
    if any(not value for value in ids):
        raise SourceError("every arc-paper document block must have an id")
    if len(ids) != len(set(ids)):
        raise SourceError("arc-paper document contains duplicate block ids")

    integrity = document.get("integrity") or {}
    status = str(integrity.get("status") or "").lower()
    complete = integrity.get("complete")
    blocking = integrity.get("blocking_issues") or integrity.get("errors") or []
    if status not in {"complete", "ok", "passed"}:
        detail = status or "not explicitly complete"
        raise SourceError(f"arc-paper document integrity is {detail}")
    if complete is False:
        raise SourceError("arc-paper document integrity is incomplete")
    if blocking:
        raise SourceError(f"arc-paper document has blocking integrity issues: {blocking}")


def block_id(block: dict[str, Any]) -> str:
    return str(block.get("block_id") or block.get("id") or "")


def asset_path(asset: dict[str, Any]) -> Path | None:
    value = asset.get("cache_path") or asset.get("path") or asset.get("local_path")
    return Path(str(value)).expanduser() if value else None


def _unwrap(result: dict[str, Any], label: str) -> Any:
    if result.get("ok"):
        return result.get("data")
    error = result.get("error") or {}
    raise SourceError(f"Unable to load {label}: {error.get('message') or 'unknown arc-paper error'}")


def _required_data(
    result: dict[str, Any],
    *,
    label: str,
    expected_type: type,
) -> Any:
    if not isinstance(result, dict) or not result.get("ok"):
        error = result.get("error") if isinstance(result, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        raise SourceError(f"Unable to load {label}: {message or 'unknown arc-paper error'}")
    value = result.get("data")
    if not isinstance(value, expected_type):
        raise SourceError(
            f"Unable to load {label}: arc-paper returned {type(value).__name__}, "
            f"expected {expected_type.__name__}"
        )
    return value


def _optional_data(
    result: dict[str, Any],
    *,
    label: str,
    expected_type: type,
    fallback: Any,
    diagnostics: list[dict[str, str]],
) -> Any:
    value = result.get("data") if isinstance(result, dict) and result.get("ok") else None
    if isinstance(value, expected_type):
        return value
    error = result.get("error") if isinstance(result, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not message and isinstance(result, dict) and result.get("ok"):
        message = f"arc-paper returned {type(value).__name__}, expected {expected_type.__name__}"
    diagnostics.append(
        {
            "severity": "warning",
            "code": "citer_context_unavailable",
            "source": "arc-paper",
            "message": f"Unable to load optional {label}: {message or 'unknown arc-paper error'}",
        }
    )
    return fallback
