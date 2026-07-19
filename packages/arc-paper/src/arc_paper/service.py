from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from arc_llm.runner import resolve_llm_config

from .cache import (
    CachePaths,
    cache_root,
    now_iso,
    parsed_source_annotations_cache_path,
    parsed_source_cache_path,
    parsed_source_lock,
    read_paper_alias,
    read_json,
    rich_document_cache_path,
    text_query_cache_path,
    write_json,
)
from .ids import arxiv_path_id
from .ids import extract_paper_ids as _extract_paper_ids
from .ids import normalize_paper_id
from .ids import paper_ids_safe_dir_name as _paper_ids_safe_dir_name
from .parse.ar5iv_html import get_section as parsed_get_section
from .parse.equations import find_equation_context
from .parse.document import DOCUMENT_SCHEMA_VERSION, RICH_DOCUMENT_PARSER_VERSION
from .parse.source import PARSER_VERSION as SOURCE_PARSER_VERSION
from .parse.source import parse_source_input
from .parse.source import parse_source_input_with_warnings
from .parse.source import source_input_hash
from .providers import Ar5ivProvider, ArxivSourceProvider, InspireProvider
from .providers.ar5iv import ar5iv_url
from .providers.base import ProviderError
from .reference_inference import ReferenceInferenceError, infer_main_references
from .results import err, ok
from .search import FullTextSearchFile, search_parsed_full_text
from .summary.input_pack import build_input_pack
from .summary.model import DEFAULT_SUMMARY_MODEL_TIER
from .summary.providers.select import select_summary_provider
from .summary.schema import load_summary_prompt, load_summary_schema
from .summary.store import read_latest_summary, read_summary, store_summary


_inspire = InspireProvider()
_ar5iv = Ar5ivProvider()
_arxiv_source = ArxivSourceProvider()
ProgressCallback = Callable[[dict[str, Any]], None]
LEGACY_PARSED_SOURCE_KEYS = ("paper_id", "parser_version", "source_hash", "toc", "sections", "equations")
RICH_PARSER_VERSION = RICH_DOCUMENT_PARSER_VERSION


def extract_paper_ids(text: str) -> dict[str, Any]:
    return ok(_extract_paper_ids(text))


def paper_ids_safe_dir_name(ids: str | Iterable[str]) -> dict[str, Any]:
    raw_values = [ids] if isinstance(ids, str) else list(ids or [])
    values = [str(item) for item in raw_values if str(item).strip()]
    if not values:
        return err("paper_ids_required", "At least one paper id is required.")
    return ok(_paper_ids_safe_dir_name(values), provider="local")


def cache_arxiv_source(
    paper_id: str,
    *,
    version: int,
    refresh: bool = False,
    license_url: str = "",
) -> dict[str, Any]:
    normalized = normalize_paper_id(paper_id)
    if not arxiv_path_id(normalized):
        return err("not_arxiv_id", f"arXiv source requires an arXiv ID: {paper_id}")
    try:
        with parsed_source_lock(normalized, namespace=f"arxiv-source-v{version}"):
            manifest = _arxiv_source.cache_source(
                normalized,
                version=version,
                refresh=refresh,
                license_url=license_url,
            )
        return ok(manifest, provider="arxiv-source", cache="write" if refresh else "hit-or-write")
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except ValueError as exc:
        return err("arxiv_source_invalid", str(exc))
    except Exception as exc:
        return err("arxiv_source_cache_failed", str(exc))


def probe_arxiv_source(paper_id: str, *, version: int) -> dict[str, Any]:
    normalized = normalize_paper_id(paper_id)
    if not arxiv_path_id(normalized):
        return err("not_arxiv_id", f"arXiv source requires an arXiv ID: {paper_id}")
    try:
        manifest = _arxiv_source.probe_source(normalized, version=version)
        if manifest is None:
            return err(
                "arxiv_source_not_cached",
                f"No cached source found for {normalized}v{version}; run source-cache explicitly.",
            )
        return ok(manifest, provider="local-cache", cache="hit")
    except ValueError as exc:
        return err("arxiv_source_invalid", str(exc))
    except Exception as exc:
        return err("arxiv_source_probe_failed", str(exc))


def llm_infer_main_references(
    text: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    query_text = (text or "").strip()
    cache_path = text_query_cache_path("main-references", query_text)
    cached = read_json(cache_path) if not refresh else None

    explicit_ids = _extract_paper_ids(query_text)
    if explicit_ids:
        meta = {"provider": "local-parser", "llm_used": False}
        if (
            isinstance(cached, dict)
            and cached.get("paper_ids") == explicit_ids
            and (cached.get("meta") or {}).get("provider") == "local-parser"
        ):
            return ok(
                explicit_ids,
                **meta,
                cache="hit",
                cache_path=str(cache_path),
                cached_at=cached.get("created_at"),
            )
        _write_reference_query_cache(cache_path, query_text=query_text, paper_ids=explicit_ids, meta=meta)
        return ok(explicit_ids, **meta, cache="write", cache_path=str(cache_path))
    if isinstance(cached, dict) and isinstance(cached.get("paper_ids"), list):
        meta = dict(cached.get("meta") or {})
        return ok(
            cached["paper_ids"],
            **meta,
            cache="hit",
            cache_path=str(cache_path),
            cached_at=cached.get("created_at"),
        )
    try:
        result = infer_main_references(
            query_text,
            provider=provider,
            model=model,
            model_tier=model_tier,
            refresh=refresh,
            metadata_lookup=_inspire.get_metadata,
        )
    except ReferenceInferenceError as exc:
        return err(exc.code, exc.message)
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("reference_inference_failed", str(exc))
    meta = {
        "provider": result.get("provider"),
        "model": result.get("model"),
        "llm_used": True,
        "focus_scope": result.get("focus_scope"),
        "warnings": result.get("warnings", []),
        "verified_references": result.get("verified_references", []),
        "rejected_candidates": result.get("rejected_candidates", []),
        "raw_llm_response": result.get("raw_llm_response"),
    }
    _write_reference_query_cache(cache_path, query_text=query_text, paper_ids=result["paper_ids"], meta=meta)
    return ok(
        result["paper_ids"],
        **meta,
        cache="write",
        cache_path=str(cache_path),
    )


def _write_reference_query_cache(
    cache_path: Path,
    *,
    query_text: str,
    paper_ids: list[str],
    meta: dict[str, Any],
) -> None:
    write_json(
        cache_path,
        {
            "schema_version": "arc.paper.main_reference_query.v1",
            "query_text": query_text,
            "paper_ids": paper_ids,
            "meta": meta,
            "created_at": now_iso(),
        },
    )


def get_title(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _metadata_field(paper_id, "title", refresh=refresh))


def get_abstract(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _metadata_field(paper_id, "abstract", refresh=refresh))


def get_authors(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _metadata_field(paper_id, "authors", refresh=refresh))


def get_metadata(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _call(lambda: _inspire.get_metadata(paper_id, refresh=refresh), "inspire"))


def search_inspire(query: str, *, limit: int = 20) -> dict[str, Any]:
    return _call(lambda: _inspire.search_metadata(query, limit=limit), "inspire")


def get_references(ids: str | Iterable[str], *, refresh: bool = False, enrich: bool = False):
    return _map(
        ids,
        lambda paper_id: _call(
            lambda: _inspire.get_references(paper_id, refresh=refresh, enrich=enrich),
            "inspire",
        ),
    )


def get_citers(
    ids: str | Iterable[str],
    *,
    refresh: bool = False,
    limit: int = 1000,
    sort: str = "mostrecent",
):
    return _map(
        ids,
        lambda paper_id: _call(
            lambda: _inspire.get_citers(paper_id, refresh=refresh, limit=limit, sort=sort),
            "inspire",
        ),
    )


def get_citer_count(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _call(lambda: _inspire.get_citer_count(paper_id, refresh=refresh), "inspire"))


def get_toc(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _call(lambda: _parsed(paper_id, refresh=refresh)["toc"], "ar5iv"))


def get_section(ids: str | Iterable[str], section: str, *, refresh: bool = False):
    return _map(ids, lambda paper_id: _section_one(paper_id, section, refresh=refresh))


def get_equation_context(ids: str | Iterable[str], query: str, *, refresh: bool = False):
    return _map(ids, lambda paper_id: _equation_one(paper_id, query, refresh=refresh))


def search_full_text(
    ids: str | Iterable[str] | None,
    *,
    query: str,
    refresh: bool = False,
    limit: int = 20,
    context: int = 1,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    try:
        files, missing_papers = _full_text_search_files(ids, refresh=refresh)
        hits, meta = search_parsed_full_text(
            files,
            query,
            limit=limit,
            context=context,
            case_sensitive=case_sensitive,
        )
        hits = _enrich_search_hits_with_cached_metadata(hits)
    except ValueError as exc:
        return err("search_query_required", str(exc))
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("paper_search_error", str(exc))
    return ok(
        hits,
        provider="local-cache",
        query=(query or "").strip(),
        missing_papers=missing_papers,
        **meta,
    )


def parse_source(
    source_path: str | Path | None = None,
    *,
    source: str = "auto",
    source_id: str | None = None,
    paper_id: str | None = None,
    html_path: str | Path | None = None,
    tex_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    refresh: bool = False,
    include_document: bool = False,
    recache: bool = False,
) -> dict[str, Any]:
    try:
        if refresh and recache:
            raise ValueError("--refresh and --recache are mutually exclusive")
        warnings: list[dict[str, Any]] = []
        resolved_id = _parse_source_id(source_id=source_id, paper_id=paper_id)
        if source == "ar5iv" or paper_id:
            if source not in {"auto", "ar5iv"}:
                raise ValueError("--paper-id only supports source=auto or source=ar5iv")
            cache_id = _parsed_source_lookup_id(str(resolved_id or ""))
            if not cache_id:
                raise ValueError("ar5iv parsing requires paper_id")
            with parsed_source_lock(cache_id, namespace=f"light-v{SOURCE_PARSER_VERSION}"):
                path = parsed_source_cache_path(cache_id)
                cached = read_json(path) if not refresh and not recache else None
                if _is_current_light_cache(cached, cache_id):
                    if not include_document:
                        return ok(
                            _parsed_source_view(cached, include_document=False),
                            provider="local-cache",
                            cache="hit",
                            cache_path=str(path),
                        )
                    document = _read_rich_document(cached)
                    if document is not None:
                        return ok(
                            _parsed_source_with_document(cached, document),
                            provider="local-cache",
                            cache="hit",
                            cache_path=str(path),
                            rich_cache_path=str(_rich_cache_path(cached)),
                        )
                parsed = _parse_ar5iv_source(
                    resolved_id,
                    refresh=refresh,
                    include_document=include_document,
                )
                if (
                    _is_current_light_cache(cached, cache_id)
                    and not recache
                    and cached.get("source_hash") == parsed.get("source_hash")
                ):
                    path = parsed_source_cache_path(cache_id)
                    rich_path = _write_rich_cache(parsed) if include_document else None
                else:
                    path, rich_path = _write_parsed_caches(parsed, include_document=include_document)
        else:
            _validate_local_source(
                source,
                source_path=source_path,
                html_path=html_path,
                tex_path=tex_path,
                markdown_path=markdown_path,
                pdf_path=pdf_path,
            )
            current_hash = source_input_hash(
                source_path=source_path,
                html_path=html_path,
                tex_path=tex_path,
                markdown_path=markdown_path,
                pdf_path=pdf_path,
            )
            lock_id = str(resolved_id or current_hash)
            with parsed_source_lock(lock_id, namespace=f"light-v{SOURCE_PARSER_VERSION}"):
                path = parsed_source_cache_path(resolved_id) if resolved_id else None
                cached = read_json(path) if path is not None and not refresh and not recache else None
                if (
                    _is_current_light_cache(cached, resolved_id)
                    and cached.get("source_hash") == current_hash
                ):
                    if not include_document:
                        return ok(
                            _parsed_source_view(cached, include_document=False),
                            provider="local-cache",
                            cache="hit",
                            cache_path=str(path),
                        )
                    document = _read_rich_document(cached)
                    if document is not None:
                        return ok(
                            _parsed_source_with_document(cached, document),
                            provider="local-cache",
                            cache="hit",
                            cache_path=str(path),
                            rich_cache_path=str(_rich_cache_path(cached)),
                        )
                parsed, warnings = _parse_local_source(
                    source=source,
                    source_path=source_path,
                    source_id=resolved_id,
                    html_path=html_path,
                    tex_path=tex_path,
                    markdown_path=markdown_path,
                    pdf_path=pdf_path,
                    include_document=include_document,
                )
                path, rich_path = _write_parsed_caches(parsed, include_document=include_document)
        return ok(
            _parsed_source_view(parsed, include_document=include_document),
            provider="local-cache",
            cache="write",
            cache_path=str(path),
            rich_cache_path=str(rich_path) if rich_path is not None else None,
            warnings=warnings or None,
        )
    except FileNotFoundError as exc:
        return err("parse_source_not_found", str(exc))
    except ValueError as exc:
        return err("parse_source_invalid", str(exc))
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("parse_source_failed", str(exc))


def get_parsed_source(source_id: str, *, include_document: bool = False) -> dict[str, Any]:
    lookup_id = _parsed_source_lookup_id(source_id)
    parsed = _read_parsed_source(source_id)
    if parsed is None:
        return err("parsed_source_not_found", f"No parsed source found for {source_id}")
    rich_path: Path | None = None
    if include_document:
        document = _read_rich_document(parsed)
        if document is not None:
            parsed = _parsed_source_with_document(parsed, document)
            rich_path = _rich_cache_path(parsed)
        elif arxiv_path_id(lookup_id):
            try:
                parsed = _parsed(lookup_id, refresh=False, require_document=True)
                rich_path = _rich_cache_path(parsed)
            except (ProviderError, ValueError, OSError) as exc:
                return err("parsed_source_upgrade_failed", str(exc))
        else:
            with parsed_source_lock(lookup_id, namespace=f"rich-v{RICH_PARSER_VERSION}"):
                document = _read_rich_document(parsed)
                if document is None:
                    document = _rebuild_local_rich_document_from_stale_cache(parsed)
            if document is None:
                return err(
                    "parsed_source_document_not_found",
                    f"No complete parsed document found for {source_id}; recache the local source first.",
                )
            parsed = _parsed_source_with_document(parsed, document)
            rich_path = _rich_cache_path(parsed)
    return ok(
        _parsed_source_view(parsed, include_document=include_document),
        provider="local-cache",
        cache="hit",
        cache_path=str(parsed_source_cache_path(lookup_id)),
        rich_cache_path=str(rich_path) if rich_path is not None else None,
    )


def get_parsed_source_toc(source_id: str) -> dict[str, Any]:
    parsed = _read_parsed_source(source_id)
    if parsed is None:
        return err("parsed_source_not_found", f"No parsed source found for {source_id}")
    return ok(parsed.get("toc") or [], provider="local-cache", cache="hit")


def get_parsed_source_section(source_id: str, section: str) -> dict[str, Any]:
    parsed = _read_parsed_source(source_id)
    if parsed is None:
        return err("parsed_source_not_found", f"No parsed source found for {source_id}")
    result = parsed_get_section(parsed, section)
    if result.get("ok"):
        result["meta"] = {"provider": "local-cache", "cache": "hit"}
    return result


def get_parsed_source_equations(source_id: str) -> dict[str, Any]:
    parsed = _read_parsed_source(source_id)
    if parsed is None:
        return err("parsed_source_not_found", f"No parsed source found for {source_id}")
    annotations = _current_equation_annotations(source_id, parsed)
    equations = [_with_equation_annotations(equation, annotations) for equation in parsed.get("equations") or []]
    return ok(equations, provider="local-cache", cache="hit")


def get_parsed_source_equation(source_id: str, equation_id: str) -> dict[str, Any]:
    parsed = _read_parsed_source(source_id)
    if parsed is None:
        return err("parsed_source_not_found", f"No parsed source found for {source_id}")
    annotations = _current_equation_annotations(source_id, parsed)
    for equation in parsed.get("equations") or []:
        if str(equation.get("id") or "") == equation_id:
            return ok(_with_equation_annotations(equation, annotations), provider="local-cache", cache="hit")
    return err("parsed_source_equation_not_found", f"No equation {equation_id} found in {source_id}")


def mark_parsed_equation(
    source_id: str,
    equation_id: str,
    *,
    status: str = "problematic",
    reason: str = "",
) -> dict[str, Any]:
    source_id = str(source_id or "").strip()
    equation_id = str(equation_id or "").strip()
    status = str(status or "problematic").strip()
    reason = str(reason or "").strip()
    if not source_id:
        return err("parsed_source_id_required", "A parsed source id is required.")
    if not equation_id:
        return err("parsed_source_equation_id_required", "A parsed equation id is required.")
    if status not in {"problematic", "needs_recache", "resolved"}:
        return err("parsed_source_annotation_invalid", f"Unsupported parsed equation status {status!r}.")
    if not reason:
        return err("parsed_source_annotation_reason_required", "A reason is required when marking a parsed equation.")

    parsed = _read_parsed_source(source_id)
    if parsed is None:
        return err("parsed_source_not_found", f"No parsed source found for {source_id}")
    equation = _find_parsed_equation(parsed, equation_id)
    if equation is None:
        return err("parsed_source_equation_not_found", f"No equation {equation_id} found in {source_id}")

    annotation_source_id = _parsed_source_lookup_id(source_id)
    path = parsed_source_annotations_cache_path(annotation_source_id)
    sidecar = _read_parsed_source_annotations(annotation_source_id)
    annotations = [
        annotation
        for annotation in sidecar.get("annotations", [])
        if not (
            isinstance(annotation, dict)
            and annotation.get("target_kind") == "equation"
            and str(annotation.get("target_id") or "") == equation_id
        )
    ]
    existing = _annotation_for_target(sidecar.get("annotations", []), equation_id)
    now = now_iso()
    annotation = {
        "source_id": annotation_source_id,
        "target_kind": "equation",
        "target_id": equation_id,
        "status": status,
        "reason": reason,
        "source_hash": str(parsed.get("source_hash") or ""),
        "parser_version": int(parsed.get("parser_version") or 0),
        "equation_fingerprint": _equation_fingerprint(equation),
        "created_at": str(existing.get("created_at") or now) if existing else now,
        "updated_at": now,
    }
    annotations.append(annotation)
    write_json(
        path,
        {
            "schema_version": "arc.parsed_source.annotations.v1",
            "source_id": annotation_source_id,
            "annotations": annotations,
        },
    )
    return ok(annotation, provider="local-cache", cache="write", cache_path=str(path))


def search_parsed_source(
    source_id: str,
    *,
    query: str,
    limit: int = 20,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    parsed = _read_parsed_source(source_id)
    if parsed is None:
        return err("parsed_source_not_found", f"No parsed source found for {source_id}")
    if not (query or "").strip():
        return err("parsed_source_search_query_required", "A non-empty parsed source search query is required.")
    hits = _search_parsed_source_records(parsed, query, limit=max(1, min(int(limit), 200)), case_sensitive=case_sensitive)
    annotations = _current_equation_annotations(source_id, parsed)
    hits = [_with_equation_annotations(hit, annotations) if hit.get("kind") == "equation" else hit for hit in hits]
    return ok(hits, provider="local-cache", cache="hit", query=query, limit=limit, case_sensitive=case_sensitive)


def _read_parsed_source(source_id: str) -> dict[str, Any] | None:
    parsed = read_json(parsed_source_cache_path(_parsed_source_lookup_id(source_id)))
    return parsed if isinstance(parsed, dict) else None


def _read_parsed_source_annotations(source_id: str) -> dict[str, Any]:
    data = read_json(parsed_source_annotations_cache_path(source_id))
    if not isinstance(data, dict):
        return {"schema_version": "arc.parsed_source.annotations.v1", "source_id": source_id, "annotations": []}
    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        annotations = []
    return {
        "schema_version": str(data.get("schema_version") or "arc.parsed_source.annotations.v1"),
        "source_id": str(data.get("source_id") or source_id),
        "annotations": [annotation for annotation in annotations if isinstance(annotation, dict)],
    }


def _find_parsed_equation(parsed: dict[str, Any], equation_id: str) -> dict[str, Any] | None:
    for equation in parsed.get("equations") or []:
        if isinstance(equation, dict) and str(equation.get("id") or "") == equation_id:
            return equation
    return None


def _annotation_for_target(annotations: list[Any], equation_id: str) -> dict[str, Any]:
    for annotation in annotations:
        if (
            isinstance(annotation, dict)
            and annotation.get("target_kind") == "equation"
            and str(annotation.get("target_id") or "") == equation_id
        ):
            return annotation
    return {}


def _current_equation_annotations(source_id: str, parsed: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    source_hash = str(parsed.get("source_hash") or "")
    sidecar = _read_parsed_source_annotations(_parsed_source_lookup_id(source_id))
    current: dict[str, list[dict[str, Any]]] = {}
    for annotation in sidecar.get("annotations", []):
        if annotation.get("target_kind") != "equation":
            continue
        if str(annotation.get("source_hash") or "") != source_hash:
            continue
        target_id = str(annotation.get("target_id") or "")
        if not target_id:
            continue
        equation = _find_parsed_equation(parsed, target_id)
        if equation is None:
            continue
        if int(annotation.get("parser_version") or 0) != int(parsed.get("parser_version") or 0):
            continue
        if str(annotation.get("equation_fingerprint") or "") != _equation_fingerprint(equation):
            continue
        current.setdefault(target_id, []).append(dict(annotation))
    return current


def _equation_fingerprint(equation: dict[str, Any]) -> str:
    material = "\n".join(
        str(equation.get(key) or "")
        for key in (
            "id",
            "equation",
            "before",
            "after",
            "section_id",
            "section_title",
            "tex_label",
            "printed_equation_number",
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _with_equation_annotations(equation: dict[str, Any], annotations: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = dict(equation)
    target_annotations = annotations.get(str(equation.get("id") or ""))
    if target_annotations:
        out["annotations"] = target_annotations
    return out


def _search_parsed_source_records(
    parsed: dict[str, Any],
    query: str,
    *,
    limit: int,
    case_sensitive: bool,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for section in parsed.get("sections") or []:
        haystack = " ".join(str(section.get(field) or "") for field in ("section_id", "title", "text"))
        if _text_contains(haystack, query, case_sensitive=case_sensitive):
            hits.append({"kind": "section", **section})
        if len(hits) >= limit:
            return hits[:limit]
    for equation in parsed.get("equations") or []:
        haystack = " ".join(
            str(equation.get(field) or "")
            for field in (
                "id",
                "equation",
                "before",
                "after",
                "section_title",
                "tex_label",
                "printed_equation_number",
                "printed_equation_numbers",
            )
        )
        if _text_contains(haystack, query, case_sensitive=case_sensitive):
            hits.append({"kind": "equation", **equation})
        if len(hits) >= limit:
            break
    return hits[:limit]


def _text_contains(text: str, query: str, *, case_sensitive: bool) -> bool:
    if case_sensitive:
        return query in text
    return query.lower() in text.lower()


def _enrich_search_hits_with_cached_metadata(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_by_paper: dict[str, dict[str, str]] = {}
    enriched: list[dict[str, Any]] = []
    for hit in hits:
        paper_id = str(hit.get("paper_id") or "")
        if paper_id not in metadata_by_paper:
            metadata_by_paper[paper_id] = _cached_search_hit_metadata(paper_id)
        enriched_hit = dict(hit)
        enriched_hit.update(metadata_by_paper[paper_id])
        enriched.append(enriched_hit)
    return enriched


def _cached_search_hit_metadata(paper_id: str) -> dict[str, str]:
    cached = read_json(CachePaths.for_paper(paper_id).inspire_metadata)
    metadata = cached.get("metadata", cached) if isinstance(cached, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    authors = _metadata_authors(metadata.get("authors"))
    return {
        "title": _metadata_title(metadata),
        "authors": _format_author_list(authors),
    }


def _metadata_title(metadata: dict[str, Any]) -> str:
    if title := str(metadata.get("title") or "").strip():
        return title
    titles = metadata.get("titles")
    if isinstance(titles, list):
        for item in titles:
            if isinstance(item, dict) and (title := str(item.get("title") or "").strip()):
                return title
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def _metadata_authors(raw_authors: Any) -> list[str]:
    authors: list[str] = []
    if not isinstance(raw_authors, list):
        return authors
    for author in raw_authors:
        if isinstance(author, str):
            name = author
        elif isinstance(author, dict):
            name = str(author.get("full_name") or author.get("name") or author.get("display_name") or "")
        else:
            name = ""
        if name.strip():
            authors.append(name.strip())
    return authors


def _format_author_list(authors: list[str]) -> str:
    if not authors:
        return ""
    if len(authors) > 5:
        return f"{authors[0]} et al."
    return ", ".join(authors)


def get_llm_summary(
    ids: str | Iterable[str],
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    refresh: bool = False,
    progress_callback: ProgressCallback | None = None,
):
    return _map(
        ids,
        lambda paper_id: _get_or_generate_summary_one(
            paper_id,
            provider=provider,
            model=model,
            model_tier=model_tier,
            refresh=refresh,
            progress_callback=progress_callback,
        ),
    )


def get_cached_llm_summary(ids: str | Iterable[str]):
    return _map(ids, _cached_summary_one)


def generate_llm_summary(
    ids: str | Iterable[str],
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    refresh: bool = False,
    progress_callback: ProgressCallback | None = None,
):
    return _map(
        ids,
        lambda paper_id: _generate_summary_one(
            paper_id,
            provider=provider,
            model=model,
            model_tier=model_tier,
            refresh=refresh,
            progress_callback=progress_callback,
        ),
    )


def store_llm_summary(paper_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    try:
        path = store_summary(paper_id, summary)
    except Exception as exc:
        return err("summary_store_failed", str(exc))
    return ok({"summary_path": str(path), "summary": summary}, provider="local-cache", cache="write")


def doctor_cache(paper_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "cache_root": str(cache_root()),
        "env": {
            "ARC_PAPER_CACHE": os.environ.get("ARC_PAPER_CACHE"),
            "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
            "HOME": os.environ.get("HOME"),
        },
    }
    if paper_id:
        normalized = normalize_paper_id(paper_id)
        paths = CachePaths.for_paper(normalized)
        latest_summary_path = paths.paper_dir / "summaries" / "paper-summary-v1" / "latest.json"
        latest = read_latest_summary(normalized)
        data["paper"] = {
            "paper_id": normalized,
            "paper_dir": str(paths.paper_dir),
            "paper_dir_exists": paths.paper_dir.exists(),
            "latest_summary_path": str(latest_summary_path),
            "latest_summary_exists": latest_summary_path.exists(),
            "latest_summary_title": latest.get("title") if isinstance(latest, dict) else None,
            "latest_summary_source_hash": (
                (latest.get("provenance") or {}).get("source_hash") if isinstance(latest, dict) else None
            ),
        }
    return ok(data)


def list_cached_papers(
    *,
    ids: Iterable[str] | None = None,
    since: str | None = None,
    older_than: str | None = None,
) -> dict[str, Any]:
    try:
        items = _select_cached_papers(ids=ids, since=since, older_than=older_than)
    except ValueError as exc:
        return err("cache_filter_invalid", str(exc))
    return ok(
        {"items": items, "count": len(items)},
        provider="local-cache",
        cache_root=str(cache_root()),
        since=since,
        older_than=older_than,
    )


def remove_cached_papers(
    *,
    ids: Iterable[str] | None = None,
    since: str | None = None,
    older_than: str | None = None,
    all_items: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    if not all_items and not ids and not since and not older_than:
        return err("cache_remove_selector_required", "Use --id, --since, --older-than, or --all before removing cache entries.")
    selected = list_cached_papers(ids=ids, since=since, older_than=older_than)
    if not selected.get("ok"):
        return selected
    items = list((selected.get("data") or {}).get("items") or [])
    removed_paths: list[str] = []
    skipped_paths: list[dict[str, str]] = []
    if not dry_run:
        for path in _unique_selected_paths(items):
            safe_path = _safe_cache_path(Path(path))
            if safe_path is None:
                skipped_paths.append({"path": str(path), "reason": "outside cache root"})
                continue
            if not safe_path.exists():
                continue
            try:
                if safe_path.is_dir():
                    shutil.rmtree(safe_path)
                else:
                    safe_path.unlink()
                removed_paths.append(str(safe_path))
            except OSError as exc:
                skipped_paths.append({"path": str(safe_path), "reason": str(exc)})
    return ok(
        {
            "items": items,
            "count": len(items),
            "dry_run": dry_run,
            "removed_count": len(removed_paths),
            "removed_paths": removed_paths,
            "skipped_paths": skipped_paths,
        },
        provider="local-cache",
        cache_root=str(cache_root()),
        since=since,
        older_than=older_than,
    )


def _select_cached_papers(
    *,
    ids: Iterable[str] | None,
    since: str | None,
    older_than: str | None,
) -> list[dict[str, Any]]:
    items = _cached_paper_items()
    id_filter = _cache_id_filter(ids)
    if id_filter:
        items = [item for item in items if _cache_item_matches_id(item, id_filter)]
    now = time.time()
    if since:
        threshold = now - _parse_duration_seconds(since)
        items = [item for item in items if float(item.get("modified_time") or 0.0) >= threshold]
    if older_than:
        threshold = now - _parse_duration_seconds(older_than)
        items = [item for item in items if float(item.get("modified_time") or 0.0) <= threshold]
    return sorted(items, key=lambda item: (str(item.get("paper_id") or ""), str(item.get("modified_at") or "")))


def _cached_paper_items() -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    root = cache_root()
    for paper_dir in sorted((root / "papers").iterdir() if (root / "papers").is_dir() else []):
        if paper_dir.is_dir():
            _add_cache_path(grouped, unquote(paper_dir.name), "paper_dir", paper_dir)
    for source_path in sorted((root / "sources").glob("*.json")):
        paper_id = _paper_id_from_json(source_path, default=source_path.stem)
        _add_cache_path(grouped, paper_id, "source", source_path)
    for annotation_path in sorted((root / "source-annotations").glob("*.json")):
        paper_id = _source_id_from_annotation_json(annotation_path, default=annotation_path.stem)
        _add_cache_path(grouped, paper_id, "source_annotation", annotation_path)
    for alias_path in sorted((root / "paper-aliases").glob("*.json")):
        paper_id = _paper_id_from_json(alias_path, default=unquote(alias_path.stem))
        _add_cache_path(grouped, paper_id, "paper_alias", alias_path)
    return [_finalize_cache_item(item) for item in grouped.values()]


def _add_cache_path(grouped: dict[str, dict[str, Any]], paper_id: str, kind: str, path: Path) -> None:
    if not paper_id or not path.exists():
        return
    item = grouped.setdefault(
        paper_id,
        {
            "paper_id": paper_id,
            "kinds": [],
            "paths": [],
            "bytes": 0,
            "modified_time": 0.0,
        },
    )
    if kind not in item["kinds"]:
        item["kinds"].append(kind)
    modified_time = _path_modified_time(path)
    item["modified_time"] = max(float(item["modified_time"]), modified_time)
    item["bytes"] = int(item["bytes"]) + _path_size(path)
    item["paths"].append(
        {
            "kind": kind,
            "path": str(path),
            "bytes": _path_size(path),
            "modified_at": _iso_from_timestamp(modified_time),
        }
    )


def _finalize_cache_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["kinds"] = sorted(out.get("kinds") or [])
    out["paths"] = sorted(out.get("paths") or [], key=lambda path: (path.get("kind", ""), path.get("path", "")))
    out["modified_at"] = _iso_from_timestamp(float(out.get("modified_time") or 0.0))
    return out


def _paper_id_from_json(path: Path, *, default: str) -> str:
    data = read_json(path)
    if isinstance(data, dict):
        for key in ("paper_id", "source_id", "canonical_id"):
            value = str(data.get(key) or "").strip()
            if value:
                return value
    return default


def _source_id_from_annotation_json(path: Path, *, default: str) -> str:
    data = read_json(path)
    if isinstance(data, dict):
        value = str(data.get("source_id") or "").strip()
        if value:
            return value
    return default


def _cache_id_filter(ids: Iterable[str] | None) -> set[str]:
    out: set[str] = set()
    for paper_id in ids or []:
        raw = str(paper_id or "").strip()
        normalized = normalize_paper_id(raw)
        if raw:
            out.add(raw)
        if normalized:
            out.add(normalized)
    return out


def _cache_item_matches_id(item: dict[str, Any], id_filter: set[str]) -> bool:
    paper_id = str(item.get("paper_id") or "")
    return paper_id in id_filter or normalize_paper_id(paper_id) in id_filter


def _parse_duration_seconds(value: str) -> float:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    if text.startswith("past "):
        text = text[5:].strip()
    units = {
        "s": 1,
        "sec": 1,
        "second": 1,
        "seconds": 1,
        "m": 60,
        "min": 60,
        "minute": 60,
        "minutes": 60,
        "h": 3600,
        "hr": 3600,
        "hour": 3600,
        "hours": 3600,
        "d": 86400,
        "day": 86400,
        "days": 86400,
        "w": 7 * 86400,
        "week": 7 * 86400,
        "weeks": 7 * 86400,
    }
    if text in units:
        return float(units[text])
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([a-z]+)$", text)
    if not match:
        raise ValueError(f"Invalid duration {value!r}; use values like 1h, 1d, or past hour.")
    amount = float(match.group(1))
    unit = match.group(2)
    if unit not in units:
        raise ValueError(f"Invalid duration unit {unit!r}; use h, d, or week.")
    return amount * units[unit]


def _path_modified_time(path: Path) -> float:
    if path.is_dir():
        latest = path.stat().st_mtime
        for child in path.rglob("*"):
            try:
                latest = max(latest, child.stat().st_mtime)
            except OSError:
                continue
        return latest
    return path.stat().st_mtime


def _path_size(path: Path) -> int:
    if path.is_dir():
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
        return total
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _unique_selected_paths(items: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for item in items:
        for path_info in item.get("paths") or []:
            path = str(path_info.get("path") or "")
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _safe_cache_path(path: Path) -> Path | None:
    root = cache_root().resolve()
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def _metadata_field(paper_id: str, field: str, *, refresh: bool) -> dict[str, Any]:
    return _call(lambda: _inspire.get_metadata(paper_id, refresh=refresh).get(field), "inspire")


def _cached_summary_one(paper_id: str) -> dict[str, Any]:
    cached = read_latest_summary(paper_id)
    if cached:
        return ok(cached, provider="local-cache", cache="hit")
    return err("summary_not_available", f"No cached LLM summary is available for {paper_id}")


def _section_one(paper_id: str, section: str, *, refresh: bool) -> dict[str, Any]:
    try:
        parsed = _parsed(paper_id, refresh=refresh)
        result = parsed_get_section(parsed, section)
        if result["ok"]:
            result["meta"] = {"provider": "ar5iv"}
        return result
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("paper_query_error", str(exc))


def _equation_one(paper_id: str, query: str, *, refresh: bool) -> dict[str, Any]:
    try:
        parsed = _parsed(paper_id, refresh=refresh)
        return ok(find_equation_context(parsed.get("equations") or [], query), provider="ar5iv")
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("paper_query_error", str(exc))


def _parse_source_id(*, source_id: str | None, paper_id: str | None) -> str | None:
    if source_id and paper_id:
        raise ValueError("Use either --id or --paper-id, not both.")
    if paper_id:
        return normalize_paper_id(paper_id)
    return source_id


def _parse_local_source(
    *,
    source: str,
    source_path: str | Path | None,
    source_id: str | None,
    html_path: str | Path | None,
    tex_path: str | Path | None,
    markdown_path: str | Path | None,
    pdf_path: str | Path | None,
    include_document: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_local_source(
        source,
        source_path=source_path,
        html_path=html_path,
        tex_path=tex_path,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
    )
    return parse_source_input_with_warnings(
        source_path=source_path,
        source_id=source_id,
        html_path=html_path,
        tex_path=tex_path,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
        include_document=include_document,
    )


def _validate_local_source(
    source: str,
    *,
    source_path: str | Path | None,
    html_path: str | Path | None,
    tex_path: str | Path | None,
    markdown_path: str | Path | None,
    pdf_path: str | Path | None,
) -> None:
    if source not in {"auto", "html", "tex", "markdown", "pdf", "tex-pdf", "markdown-pdf"}:
        raise ValueError(f"Unsupported parse source: {source}")
    if source == "auto":
        return
    source_suffix = Path(source_path).suffix.lower() if source_path else ""
    has_html = bool(html_path) or source_suffix in {".html", ".htm"}
    has_tex = bool(tex_path) or source_suffix == ".tex"
    has_markdown = bool(markdown_path) or source_suffix in {".md", ".markdown"}
    has_pdf = bool(pdf_path) or source_suffix == ".pdf"
    if source == "html":
        if not has_html or has_tex or has_markdown or has_pdf:
            raise ValueError("source=html requires HTML input only")
    elif source == "tex":
        if not has_tex or has_html or has_markdown or has_pdf:
            raise ValueError("source=tex requires TeX input only; use source=tex-pdf with --pdf")
    elif source == "markdown":
        if not has_markdown or has_html or has_tex or has_pdf:
            raise ValueError("source=markdown requires Markdown input only; use source=markdown-pdf with --pdf")
    elif source == "pdf":
        if not has_pdf or has_html or has_tex or has_markdown:
            raise ValueError("source=pdf requires PDF input only")
    elif source == "tex-pdf":
        if not has_tex or not pdf_path or has_html or has_markdown or source_suffix == ".pdf":
            raise ValueError("source=tex-pdf requires TeX input plus --pdf")
    elif source == "markdown-pdf":
        if not has_markdown or not pdf_path or has_html or has_tex or source_suffix == ".pdf":
            raise ValueError("source=markdown-pdf requires Markdown input plus --pdf")


def _parse_ar5iv_source(
    paper_id: str | None,
    *,
    refresh: bool,
    include_document: bool = False,
) -> dict[str, Any]:
    if not paper_id:
        raise ValueError("ar5iv parsing requires paper_id")
    full_text_id = _full_text_paper_id(paper_id, refresh=refresh)
    html = _ar5iv.get_html(full_text_id, refresh=refresh)
    assets: list[dict[str, Any]] = []
    if include_document:
        cache_assets = getattr(_ar5iv, "cache_assets", None)
        if callable(cache_assets):
            assets = cache_assets(full_text_id, html, refresh=refresh)
    normalized = normalize_paper_id(full_text_id)
    return parse_source_input(
        html_text=html,
        source_id=normalized,
        include_document=include_document,
        source_url=ar5iv_url(normalized),
        assets=assets,
    )


def _parsed(paper_id: str, *, refresh: bool, require_document: bool = False) -> dict[str, Any]:
    full_text_id = _full_text_paper_id(paper_id, refresh=refresh)
    normalized = normalize_paper_id(full_text_id)
    with parsed_source_lock(normalized, namespace=f"light-v{SOURCE_PARSER_VERSION}"):
        path = parsed_source_cache_path(normalized)
        cached = read_json(path) if not refresh else None
        if _is_current_light_cache(cached, normalized):
            if not require_document:
                return cached
            document = _read_rich_document(cached)
            if document is not None:
                return _parsed_source_with_document(cached, document)
        parsed = _parse_ar5iv_source(
            normalized,
            refresh=refresh,
            include_document=require_document,
        )
        _write_parsed_caches(parsed, include_document=require_document)
        return parsed


def _is_current_light_cache(cached: Any, paper_id: str | None) -> bool:
    return bool(
        isinstance(cached, dict)
        and cached.get("paper_id") == paper_id
        and cached.get("parser_version") == SOURCE_PARSER_VERSION
        and cached.get("source_hash")
    )


def _rich_cache_path(parsed: dict[str, Any]) -> Path:
    return rich_document_cache_path(
        str(parsed.get("paper_id") or ""),
        str(parsed.get("source_hash") or ""),
        RICH_PARSER_VERSION,
    )


def _read_rich_document(parsed: dict[str, Any]) -> dict[str, Any] | None:
    cached = read_json(_rich_cache_path(parsed))
    if isinstance(cached, dict):
        document = cached.get("document")
        if (
            cached.get("paper_id") == parsed.get("paper_id")
            and cached.get("source_hash") == parsed.get("source_hash")
            and cached.get("rich_parser_version") == RICH_PARSER_VERSION
            and isinstance(document, dict)
            and document.get("schema_version") == DOCUMENT_SCHEMA_VERSION
            and document.get("parser_version") == RICH_PARSER_VERSION
        ):
            return document
    legacy_document = parsed.get("document")
    if (
        isinstance(legacy_document, dict)
        and legacy_document.get("schema_version") == DOCUMENT_SCHEMA_VERSION
        and legacy_document.get("parser_version") == RICH_PARSER_VERSION
    ):
        write_json(
            _rich_cache_path(parsed),
            {
                "paper_id": parsed.get("paper_id"),
                "source_hash": parsed.get("source_hash"),
                "rich_parser_version": RICH_PARSER_VERSION,
                "document": legacy_document,
            },
        )
        return legacy_document
    return None


def _rebuild_local_rich_document_from_stale_cache(parsed: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild a version-stale local Markdown document from cached provenance."""

    current_path = _rich_cache_path(parsed)
    source_hash = str(parsed.get("source_hash") or "")
    candidates = sorted(
        current_path.parent.parent.glob(f"v*/{source_hash}.json"),
        key=_rich_cache_version,
        reverse=True,
    )
    for candidate in candidates:
        if candidate == current_path:
            continue
        stale = read_json(candidate)
        if not isinstance(stale, dict):
            continue
        if stale.get("paper_id") != parsed.get("paper_id") or stale.get("source_hash") != source_hash:
            continue
        document = stale.get("document")
        source = document.get("source") if isinstance(document, dict) else None
        if not isinstance(source, dict) or source.get("format") != "markdown":
            continue
        markdown_path = Path(str(source.get("path") or ""))
        raw_pdf_path = str(source.get("pdf_path") or "")
        pdf_path = Path(raw_pdf_path) if raw_pdf_path else None
        if not markdown_path.is_file() or (pdf_path is not None and not pdf_path.is_file()):
            continue
        if source_input_hash(markdown_path=markdown_path, pdf_path=pdf_path) != source_hash:
            continue
        rebuilt = parse_source_input(
            markdown_path=markdown_path,
            pdf_path=pdf_path,
            source_id=str(parsed.get("paper_id") or ""),
            include_document=True,
        )
        if rebuilt.get("source_hash") != source_hash or not isinstance(rebuilt.get("document"), dict):
            continue
        _write_rich_cache(rebuilt)
        return rebuilt["document"]
    return None


def _rich_cache_version(path: Path) -> int:
    match = re.fullmatch(r"v(\d+)", path.parent.name)
    return int(match.group(1)) if match else -1


def _write_parsed_caches(
    parsed: dict[str, Any],
    *,
    include_document: bool,
) -> tuple[Path, Path | None]:
    paper_id = str(parsed["paper_id"])
    light = {key: parsed.get(key) for key in LEGACY_PARSED_SOURCE_KEYS}
    light_path = parsed_source_cache_path(paper_id)
    write_json(light_path, light)
    rich_path: Path | None = None
    document = parsed.get("document")
    if include_document and isinstance(document, dict):
        rich_path = _write_rich_cache(parsed)
    return light_path, rich_path


def _write_rich_cache(parsed: dict[str, Any]) -> Path:
    path = _rich_cache_path(parsed)
    write_json(
        path,
        {
            "paper_id": parsed.get("paper_id"),
            "source_hash": parsed.get("source_hash"),
            "rich_parser_version": RICH_PARSER_VERSION,
            "document": parsed["document"],
        },
    )
    return path


def _parsed_source_with_document(parsed: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    return {**_parsed_source_view(parsed, include_document=False), "document": document}


def _parsed_source_view(parsed: dict[str, Any], *, include_document: bool) -> dict[str, Any]:
    if include_document:
        document = parsed.get("document")
        if isinstance(document, dict):
            return _parsed_source_with_document(parsed, document)
    return {key: parsed.get(key) for key in LEGACY_PARSED_SOURCE_KEYS}


def _parsed_source_lookup_id(source_id: str) -> str:
    current = normalize_paper_id(source_id)
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        alias = normalize_paper_id(read_paper_alias(current))
        if not alias:
            return current
        current = alias
    return current or normalize_paper_id(source_id)


def _full_text_search_files(
    ids: str | Iterable[str] | None,
    *,
    refresh: bool,
) -> tuple[list[FullTextSearchFile], list[str]]:
    if ids is None:
        return _all_cached_full_text_files(), []
    raw_ids = [ids] if isinstance(ids, str) else list(ids or [])
    files_by_path: dict[Path, FullTextSearchFile] = {}
    missing: list[str] = []
    for raw in raw_ids:
        normalized_raw = normalize_paper_id(str(raw))
        local_path = parsed_source_cache_path(normalized_raw)
        if not refresh and not arxiv_path_id(normalized_raw) and local_path.exists():
            files_by_path[local_path] = FullTextSearchFile(normalized_raw, local_path)
            continue
        full_text_id = _full_text_paper_id(str(raw), refresh=refresh)
        path = parsed_source_cache_path(full_text_id)
        _parsed(full_text_id, refresh=refresh)
        if path.exists():
            files_by_path[path] = FullTextSearchFile(full_text_id, path)
        else:
            missing.append(full_text_id)
    return list(files_by_path.values()), missing


def _all_cached_full_text_files() -> list[FullTextSearchFile]:
    files = []
    for path in sorted((cache_root() / "sources").glob("*.json")):
        parsed = read_json(path)
        if isinstance(parsed, dict) and parsed.get("paper_id"):
            files.append(FullTextSearchFile(str(parsed["paper_id"]), path))
    return files


def _full_text_paper_id(paper_id: str, *, refresh: bool) -> str:
    normalized = normalize_paper_id(paper_id)
    if arxiv_path_id(normalized):
        return normalized
    metadata = _inspire.get_metadata(normalized, refresh=refresh)
    if arxiv_id := metadata.get("arxiv_id"):
        return normalize_paper_id(f"arXiv:{arxiv_id}")
    return normalized


def _summary_status(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    task = _build_summary_task(paper_id, refresh=refresh)
    summary_paper_id = str(task["input_pack"]["paper_id"])
    source_hash = task["input_pack"]["source_hash"]
    if not refresh and (cached := read_summary(summary_paper_id, source_hash=source_hash)):
        return ok(cached, provider="local-cache", cache="hit")
    return _needs_llm(summary_paper_id, task)


def _summary_status_for_generation(
    paper_id: str,
    *,
    refresh: bool,
    provider: str,
    model: str | None,
) -> dict[str, Any]:
    task = _build_summary_task(paper_id, refresh=refresh)
    summary_paper_id = str(task["input_pack"]["paper_id"])
    source_hash = task["input_pack"]["source_hash"]
    if not refresh and (
        cached := read_summary(summary_paper_id, source_hash=source_hash, provider=provider, model=model)
    ):
        return ok(cached, provider="local-cache", cache="hit")
    return _needs_llm(summary_paper_id, task)


def _get_or_generate_summary_one(
    paper_id: str,
    *,
    provider: str,
    model: str | None,
    model_tier: str | None,
    refresh: bool,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    config = None
    if provider != "auto" or model or model_tier:
        try:
            config = resolve_llm_config(provider=provider, model=model, model_tier=model_tier)
        except Exception as exc:
            return err("summary_generation_failed", str(exc))
        status = _summary_status_for_generation_or_error(
            paper_id,
            refresh=refresh,
            provider=config.provider,
            model=config.model,
        )
    else:
        status = _summary_status_or_error(paper_id, refresh=refresh)
    if status["ok"]:
        return status
    if status.get("status") != "needs_llm":
        return status
    return _generate_from_status(
        paper_id,
        status,
        config=config,
        provider=provider,
        model=model,
        model_tier=model_tier,
        progress_callback=progress_callback,
    )


def _generate_summary_one(
    paper_id: str,
    *,
    provider: str,
    model: str | None,
    model_tier: str | None,
    refresh: bool,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    model_tier = _summary_model_tier(model=model, model_tier=model_tier)
    try:
        config = resolve_llm_config(provider=provider, model=model, model_tier=model_tier)
    except Exception as exc:
        return err("summary_generation_failed", str(exc))
    status = _summary_status_for_generation_or_error(
        paper_id,
        refresh=refresh,
        provider=config.provider,
        model=config.model,
    )
    if status["ok"]:
        return status
    if status.get("status") != "needs_llm":
        return status
    return _generate_from_status(
        paper_id,
        status,
        config=config,
        model_tier=model_tier,
        progress_callback=progress_callback,
    )


def _summary_status_or_error(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    try:
        return _summary_status(paper_id, refresh=refresh)
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("paper_query_error", str(exc))


def _summary_status_for_generation_or_error(
    paper_id: str,
    *,
    refresh: bool,
    provider: str,
    model: str | None,
) -> dict[str, Any]:
    try:
        return _summary_status_for_generation(paper_id, refresh=refresh, provider=provider, model=model)
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("paper_query_error", str(exc))


def _generate_from_status(
    paper_id: str,
    status: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None,
    config: Any | None = None,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
) -> dict[str, Any]:
    try:
        model_tier = _summary_model_tier(model=model, model_tier=model_tier)
        if config is None:
            config = resolve_llm_config(provider=provider, model=model, model_tier=model_tier)
        selected = select_summary_provider(config.provider)
        if selected.name == "manual":
            return status
        summary = selected.generate_summary(
            status["llm_task"],
            model=config.model,
            model_tier=model_tier,
            progress_callback=progress_callback,
        )
        summary_paper_id = str(status.get("paper_id") or paper_id)
        path = store_summary(summary_paper_id, summary)
    except Exception as exc:
        return err("summary_generation_failed", str(exc))
    return ok(summary, provider=selected.name, cache="write", summary_path=str(path))


def _summary_model_tier(*, model: str | None, model_tier: str | None) -> str | None:
    if model_tier:
        return model_tier
    if model:
        return None
    return DEFAULT_SUMMARY_MODEL_TIER


def _build_summary_task(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    metadata = _inspire.get_metadata(paper_id, refresh=refresh)
    summary_paper_id = _summary_paper_id(paper_id, metadata)
    parsed = _parsed(summary_paper_id, refresh=refresh)
    input_pack = build_input_pack(
        summary_paper_id,
        metadata=metadata,
        parsed=parsed,
    )
    return {
        "task_type": "paper_summary",
        "pipeline": "section_then_paper",
        "refresh": refresh,
        "prompt_version": "paper-summary-v1",
        "system_prompt": load_summary_prompt(),
        "user_prompt": "Generate a paper summary JSON for the supplied input pack.",
        "input_pack": input_pack,
        "output_schema": load_summary_schema(),
    }


def _needs_llm(paper_id: str, task: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_paper_id(paper_id)
    return {
        "ok": False,
        "status": "needs_llm",
        "paper_id": normalized,
        "llm_task": task,
        "next": {
            "store_command": f"arc-paper store-llm-summary {normalized} --summary-json -"
        },
    }


def _summary_paper_id(requested_id: str, metadata: dict[str, Any]) -> str:
    if metadata_id := normalize_paper_id(str(metadata.get("paper_id") or "")):
        return metadata_id
    if arxiv_id := arxiv_path_id(str(metadata.get("arxiv_id") or "")):
        return normalize_paper_id(f"arXiv:{arxiv_id}")
    return normalize_paper_id(requested_id)


def _map(ids: str | Iterable[str] | None, func: Callable[[str], dict[str, Any]]):
    if ids is None:
        return err("paper_ids_required", "At least one paper id is required.")
    if isinstance(ids, str):
        return func(normalize_paper_id(ids))
    out: dict[str, dict[str, Any]] = {}
    for raw in ids:
        paper_id = normalize_paper_id(str(raw))
        if paper_id not in out:
            out[paper_id] = func(paper_id)
    return out


def _call(func: Callable[[], Any], provider: str) -> dict[str, Any]:
    try:
        return ok(func(), provider=provider)
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("paper_query_error", str(exc))
