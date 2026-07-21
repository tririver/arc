from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .evidence import arc_cache_descriptor, validate_evidence_record


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
    document_kind: str = "auto",
    parse: Callable[..., dict[str, Any]] | None = None,
    metadata_getter: Callable[..., dict[str, Any]] | None = None,
    references_getter: Callable[..., dict[str, Any]] | None = None,
    citers_getter: Callable[..., dict[str, Any]] | None = None,
    parsed_getter: Callable[..., dict[str, Any]] | None = None,
) -> SourceBundle:
    if (
        parse is None
        or metadata_getter is None
        or references_getter is None
        or citers_getter is None
        or parsed_getter is None
    ):
        from arc_paper import service

        parse = parse or service.parse_source
        metadata_getter = metadata_getter or service.get_metadata
        references_getter = references_getter or service.get_references
        citers_getter = citers_getter or service.get_citers
        parsed_getter = parsed_getter or service.get_parsed_source

    cached_result = parsed_getter(paper_id, include_document=True)
    cached_available = bool(
        isinstance(cached_result, dict)
        and cached_result.get("ok")
        and isinstance(cached_result.get("data"), dict)
        and isinstance(cached_result["data"].get("document"), dict)
    )
    if _is_explicit_local_source_id(paper_id) or cached_available:
        cached = _unwrap(cached_result, "complete cached source")
        if not isinstance(cached, dict):
            raise SourceError("arc-paper returned a non-object parsed source")
        if _is_explicit_local_source_id(paper_id):
            parse_kwargs = _local_reparse_kwargs(cached, paper_id)
        else:
            parse_kwargs = {"source": "ar5iv", "paper_id": paper_id}
        result = parse(
            **parse_kwargs,
            include_document=True,
            refresh=refresh,
            recache=recache,
            document_kind=document_kind,
        )
        parsed = _unwrap(result, "current parsed source")
        if not isinstance(parsed, dict):
            raise SourceError("arc-paper returned a non-object parsed source")
        document = _validated_document(parsed, document_kind=document_kind)
        # A reparsed local source normally carries these values itself.  Retain
        # cached evidence metadata when ARC-Paper intentionally returns only
        # source structure and content.
        metadata = _cached_source_metadata(parsed)
        if metadata == {"_arc_companion_metadata_source": "unavailable"}:
            metadata = _cached_source_metadata(cached)
        references = _cached_source_list(parsed, "references") or _cached_source_list(cached, "references")
        citers = _cached_source_list(parsed, "citers") or _cached_source_list(cached, "citers")
        return SourceBundle(
            paper_id=str(parsed.get("paper_id") or paper_id),
            parsed=parsed,
            document=document,
            metadata=metadata,
            references=references,
            citers=citers,
        )

    result = parse(
        source="ar5iv",
        paper_id=paper_id,
        include_document=True,
        refresh=refresh,
        recache=recache,
        document_kind=document_kind,
    )
    parsed = _unwrap(result, "complete parsed source")
    if not isinstance(parsed, dict):
        raise SourceError("arc-paper returned a non-object parsed source")
    document = _validated_document(parsed, document_kind=document_kind)

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


def _is_explicit_local_source_id(source_id: str) -> bool:
    """Recognize namespaces whose public contract is local-cache-only.

    DOI and INSPIRE identifiers also contain a colon, but are resolvable remote
    identifiers and must not be accidentally reclassified as local documents.
    Other identifier forms still take the cache-only path when an actual rich
    cache entry was found by ``load_source_bundle``.
    """
    namespace, separator, _ = source_id.strip().partition(":")
    return bool(separator) and namespace.casefold() in {"local", "isbn"}


def _local_reparse_kwargs(parsed: dict[str, Any], source_id: str) -> dict[str, Any]:
    document = parsed.get("document") if isinstance(parsed.get("document"), dict) else {}
    source = document.get("source") if isinstance(document.get("source"), dict) else {}
    path_text = str(source.get("path") or "").strip()
    pdf_text = str(source.get("pdf_path") or "").strip()
    source_format = str(source.get("format") or "").strip().casefold()
    if not source_format and path_text:
        source_format = Path(path_text).suffix.lower().lstrip(".")
    if source_format in {"md", "markdown"} and path_text:
        return {
            "source": "markdown-pdf" if pdf_text else "markdown",
            "source_id": source_id,
            "markdown_path": path_text,
            **({"pdf_path": pdf_text} if pdf_text else {}),
        }
    if source_format == "tex" and path_text:
        return {
            "source": "tex-pdf" if pdf_text else "tex",
            "source_id": source_id,
            "tex_path": path_text,
            **({"pdf_path": pdf_text} if pdf_text else {}),
        }
    if source_format in {"html", "htm"} and path_text and not pdf_text:
        return {"source": "html", "source_id": source_id, "html_path": path_text}
    if source_format == "pdf" and (pdf_text or path_text):
        return {"source": "pdf", "source_id": source_id, "pdf_path": pdf_text or path_text}
    raise SourceError(
        "cached local source lacks reusable source provenance; recache it with the current arc-paper parser"
    )


def _validated_document(parsed: dict[str, Any], *, document_kind: str) -> dict[str, Any]:
    document = parsed.get("document")
    if not isinstance(document, dict):
        raise SourceError("arc-paper did not return document; recache this source with the current parser")
    validate_complete_document(document)
    structure = parsed.get("structure")
    if not isinstance(structure, dict) or structure.get("schema_version") != "arc.paper.structure.v1":
        raise SourceError("arc-paper source lacks the current structure contract; recache it")
    if structure.get("requested_document_kind") != document_kind:
        raise SourceError("arc-paper structure does not match the requested document kind; recache it")
    source = document.get("source") if isinstance(document.get("source"), dict) else {}
    pdf_path_text = str(source.get("pdf_path") or "").strip()
    source_pdf_hash = str(source.get("pdf_sha256") or "").strip()
    if pdf_path_text or source_pdf_hash:
        if not pdf_path_text:
            raise SourceError("paired PDF provenance has no path; recache it")
        pdf_path = Path(pdf_path_text)
        if not pdf_path.is_file():
            raise SourceError(f"paired PDF is unavailable: {pdf_path}")
        current_pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        if source_pdf_hash != current_pdf_hash:
            raise SourceError("paired PDF hash changed; recache the source before companion generation")
        try:
            from arc_paper import service as paper_service
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise SourceError("arc-paper structure validator is unavailable") from exc
        if not paper_service.has_reconciliation_proof(structure, pdf_hash=current_pdf_hash):
            raise SourceError("paired PDF lacks a current complete reconciliation proof; recache it")
        proof = structure.get("reconciliation") or {}
        if proof.get("source_hash") != parsed.get("source_hash"):
            raise SourceError("paired PDF reconciliation proof does not match the parsed source")
    return document


def _cached_source_metadata(parsed: dict[str, Any]) -> dict[str, Any]:
    metadata = parsed.get("metadata")
    output = dict(metadata) if isinstance(metadata, dict) else {}
    document = parsed.get("document") if isinstance(parsed.get("document"), dict) else {}
    document_metadata = (
        document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    )
    provenance: dict[str, str] = {
        key: "parsed.metadata"
        for key, value in output.items()
        if key != "_arc_companion_metadata_source" and value is not None
    }
    fields = ("title", "authors", "year", "abstract", "page_count")
    for key in fields:
        if output.get(key) is not None:
            continue
        for owner, label in (
            (parsed, "parsed"),
            (document_metadata, "document.metadata"),
            (document, "document"),
        ):
            if owner.get(key) is not None:
                output[key] = owner[key]
                provenance[key] = label
                break
    if not str(output.get("title") or "").strip():
        for owner, label in ((parsed, "parsed.toc"), (document, "document.toc")):
            toc = owner.get("toc")
            if not isinstance(toc, list):
                continue
            title = next(
                (
                    str(item.get("title") or "").strip()
                    for item in toc
                    if isinstance(item, dict) and str(item.get("title") or "").strip()
                ),
                "",
            )
            if title:
                output["title"] = title
                provenance["title"] = label
                break
    if provenance and set(provenance.values()) == {"parsed.metadata"}:
        output["_arc_companion_metadata_source"] = "parsed.metadata"
    else:
        output["_arc_companion_metadata_source"] = provenance or "unavailable"
    return output


def _cached_source_list(parsed: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = parsed.get(key)
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _load_related_evidence(
    references: list[dict[str, Any]],
    citers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Register verified abstracts without parsing or downloading related papers.

    The high-agent catalogs retain lightweight metadata for discovery.  Missing
    local evidence is resolved later through the bounded evidence-request loop;
    ordinary companion startup must not parse a citation-count shortlist.
    """
    output: list[dict[str, Any]] = []
    for kind, items in (("prior", references), ("later", citers)):
        for rank, item in enumerate(items, 1):
            paper_id = _arxiv_identifier(item)
            abstract = str(item.get("abstract") or "").strip()
            if not paper_id or not abstract:
                continue
            output.append(_abstract_evidence_record(
                kind=kind, rank=rank, item=item, paper_id=paper_id, abstract=abstract,
            ))
    return output, []


def _abstract_evidence_record(
    *, kind: str, rank: int, item: dict[str, Any], paper_id: str, abstract: str
) -> dict[str, Any]:
    value = {
        "evidence_id": f"{kind}-{rank:03d}",
        "relation": kind,
        "paper_id": paper_id,
        "arxiv_id": item.get("arxiv_id") or item.get("arxiv"),
        "doi": item.get("doi"),
        "inspire_id": item.get("inspire_id"),
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
    return value


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
