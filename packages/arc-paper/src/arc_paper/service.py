from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .cache import (
    CachePaths,
    cache_root,
    now_iso,
    parsed_source_annotations_cache_path,
    parsed_source_cache_path,
    read_json,
    text_query_cache_path,
    write_json,
)
from .ids import arxiv_path_id
from .ids import extract_paper_ids as _extract_paper_ids
from .ids import normalize_paper_id
from .ids import paper_ids_safe_dir_name as _paper_ids_safe_dir_name
from .parse.ar5iv_html import get_section as parsed_get_section
from .parse.equations import find_equation_context
from .parse.source import PARSER_VERSION as SOURCE_PARSER_VERSION
from .parse.source import parse_source_input
from .parse.source import parse_source_input_with_warnings
from .providers import Ar5ivProvider, InspireProvider
from .providers.base import ProviderError
from .reference_inference import ReferenceInferenceError, infer_main_references
from .results import err, ok
from .search import FullTextSearchFile, search_parsed_full_text
from .summary.input_pack import build_input_pack
from .summary.providers.select import select_summary_provider
from .summary.schema import load_summary_prompt, load_summary_schema
from .summary.store import read_latest_summary, read_summary, store_summary


_inspire = InspireProvider()
_ar5iv = Ar5ivProvider()
ProgressCallback = Callable[[dict[str, Any]], None]


def extract_paper_ids(text: str) -> dict[str, Any]:
    return ok(_extract_paper_ids(text))


def paper_ids_safe_dir_name(ids: str | Iterable[str]) -> dict[str, Any]:
    raw_values = [ids] if isinstance(ids, str) else list(ids or [])
    values = [str(item) for item in raw_values if str(item).strip()]
    if not values:
        return err("paper_ids_required", "At least one paper id is required.")
    return ok(_paper_ids_safe_dir_name(values), provider="local")


def llm_infer_main_references(
    text: str,
    *,
    provider: str = "auto",
    model: str | None = None,
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
    pdf_path: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    try:
        warnings: list[dict[str, Any]] = []
        resolved_id = _parse_source_id(source_id=source_id, paper_id=paper_id)
        if source == "ar5iv" or paper_id:
            if source not in {"auto", "ar5iv"}:
                raise ValueError("--paper-id only supports source=auto or source=ar5iv")
            parsed = _parse_ar5iv_source(resolved_id, refresh=refresh)
        else:
            parsed, warnings = _parse_local_source(
                source=source,
                source_path=source_path,
                source_id=resolved_id,
                html_path=html_path,
                tex_path=tex_path,
                pdf_path=pdf_path,
            )
        path = parsed_source_cache_path(str(parsed["paper_id"]))
        write_json(path, parsed)
        return ok(parsed, provider="local-cache", cache="write", cache_path=str(path), warnings=warnings or None)
    except FileNotFoundError as exc:
        return err("parse_source_not_found", str(exc))
    except ValueError as exc:
        return err("parse_source_invalid", str(exc))
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("parse_source_failed", str(exc))


def get_parsed_source(source_id: str) -> dict[str, Any]:
    parsed = _read_parsed_source(source_id)
    if parsed is None:
        return err("parsed_source_not_found", f"No parsed source found for {source_id}")
    return ok(parsed, provider="local-cache", cache="hit", cache_path=str(parsed_source_cache_path(source_id)))


def get_parsed_source_toc(source_id: str) -> dict[str, Any]:
    parsed = _read_parsed_source(source_id)
    if parsed is None:
        return err("parsed_source_not_found", f"No parsed source found for {source_id}")
    return ok(parsed.get("toc") or [], provider="local-cache", cache="hit")


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
    if _find_parsed_equation(parsed, equation_id) is None:
        return err("parsed_source_equation_not_found", f"No equation {equation_id} found in {source_id}")

    path = parsed_source_annotations_cache_path(source_id)
    sidecar = _read_parsed_source_annotations(source_id)
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
        "source_id": source_id,
        "target_kind": "equation",
        "target_id": equation_id,
        "status": status,
        "reason": reason,
        "source_hash": str(parsed.get("source_hash") or ""),
        "created_at": str(existing.get("created_at") or now) if existing else now,
        "updated_at": now,
    }
    annotations.append(annotation)
    write_json(
        path,
        {
            "schema_version": "arc.parsed_source.annotations.v1",
            "source_id": source_id,
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


def validate_note_check(run_dir: str | Path) -> dict[str, Any]:
    base = Path(run_dir)
    missing = []
    required = [
        base / "note-check-triage.json",
        base / "plan.json",
        base / "foundation" / "latest.json",
        base / "consensus" / "config.json",
        base / "consensus" / "results.json",
    ]
    for path in required:
        if not path.is_file():
            missing.append(_display_run_path(base, path))
    triage_path = base / "note-check-triage.json"
    results_path = base / "consensus" / "results.json"
    triage = read_json(triage_path) if triage_path.is_file() else None
    consensus = read_json(results_path) if results_path.is_file() else None
    violations = []
    status_counts: dict[str, int] = {}
    allowed = {"verified", "reference_disagrees", "unresolved", "context_only"}
    if isinstance(triage, dict):
        for note in triage.get("notes") or []:
            parsed_path_raw = str(note.get("parsed_source_path") or "")
            if not parsed_path_raw:
                missing.append("parsed source JSON")
                continue
            parsed_path = Path(parsed_path_raw)
            if not parsed_path.is_file():
                missing.append(str(parsed_path) if str(parsed_path) else "parsed source JSON")
        consensus_by_step = _consensus_steps(consensus)
        for claim in triage.get("claims_to_check") or []:
            claim_id = str(claim.get("id") or claim.get("equation_id") or "<unknown>")
            status = str(claim.get("status") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status not in allowed:
                violations.append(f"{claim_id}: invalid status {status!r}")
            step_id = str(claim.get("consensus_step_id") or "")
            if not step_id:
                violations.append(f"{claim_id}: missing consensus_step_id")
                continue
            if step_id not in consensus_by_step:
                violations.append(f"{claim_id}: missing consensus result for {step_id}")
                continue
            if status == "verified" and consensus_by_step[step_id] != "all_agree":
                violations.append(f"{claim_id}: verified requires consensus all_agree for {step_id}")
    elif triage_path.is_file():
        violations.append("note-check-triage.json is not valid JSON")

    if missing or violations:
        result = err("note_check_validation_failed", "Note-check run is missing required artifacts or valid consensus statuses.")
        result["missing"] = missing
        result["violations"] = violations
        return result
    return ok(
        {
            "run_dir": str(base),
            "claims_checked": sum(status_counts.values()),
            "status_counts": status_counts,
        },
        provider="local-cache",
    )


def _read_parsed_source(source_id: str) -> dict[str, Any] | None:
    parsed = read_json(parsed_source_cache_path(source_id))
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
    sidecar = _read_parsed_source_annotations(source_id)
    current: dict[str, list[dict[str, Any]]] = {}
    for annotation in sidecar.get("annotations", []):
        if annotation.get("target_kind") != "equation":
            continue
        if str(annotation.get("source_hash") or "") != source_hash:
            continue
        target_id = str(annotation.get("target_id") or "")
        if not target_id:
            continue
        current.setdefault(target_id, []).append(dict(annotation))
    return current


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
            for field in ("id", "equation", "before", "after", "section_title", "tex_label", "printed_equation_number")
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


def _consensus_steps(consensus: Any) -> dict[str, str]:
    if not isinstance(consensus, dict):
        return {}
    steps = consensus.get("steps") or consensus.get("results") or []
    out = {}
    for step in steps:
        if isinstance(step, dict) and step.get("step_id"):
            out[str(step["step_id"])] = str(step.get("status") or "")
    return out


def _display_run_path(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


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
    refresh: bool = False,
    progress_callback: ProgressCallback | None = None,
):
    return _map(
        ids,
        lambda paper_id: _get_or_generate_summary_one(
            paper_id,
            provider=provider,
            model=model,
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
    refresh: bool = False,
    progress_callback: ProgressCallback | None = None,
):
    return _map(
        ids,
        lambda paper_id: _generate_summary_one(
            paper_id,
            provider=provider,
            model=model,
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
    pdf_path: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_local_source(source, source_path=source_path, html_path=html_path, tex_path=tex_path, pdf_path=pdf_path)
    return parse_source_input_with_warnings(
        source_path=source_path,
        source_id=source_id,
        html_path=html_path,
        tex_path=tex_path,
        pdf_path=pdf_path,
    )


def _validate_local_source(
    source: str,
    *,
    source_path: str | Path | None,
    html_path: str | Path | None,
    tex_path: str | Path | None,
    pdf_path: str | Path | None,
) -> None:
    if source not in {"auto", "html", "tex", "pdf", "tex-pdf"}:
        raise ValueError(f"Unsupported parse source: {source}")
    if source == "auto":
        return
    source_suffix = Path(source_path).suffix.lower() if source_path else ""
    has_html = bool(html_path) or source_suffix in {".html", ".htm"}
    has_tex = bool(tex_path) or source_suffix == ".tex"
    has_pdf = bool(pdf_path) or source_suffix == ".pdf"
    if source == "html":
        if not has_html or has_tex or has_pdf:
            raise ValueError("source=html requires HTML input only")
    elif source == "tex":
        if not has_tex or has_html or source_suffix == ".pdf":
            raise ValueError("source=tex requires TeX input")
    elif source == "pdf":
        if not has_pdf or has_html or has_tex:
            raise ValueError("source=pdf requires PDF input only")
    elif source == "tex-pdf":
        if not has_tex or not pdf_path or has_html or source_suffix == ".pdf":
            raise ValueError("source=tex-pdf requires TeX input plus --pdf")


def _parse_ar5iv_source(paper_id: str | None, *, refresh: bool) -> dict[str, Any]:
    if not paper_id:
        raise ValueError("ar5iv parsing requires paper_id")
    full_text_id = _full_text_paper_id(paper_id, refresh=refresh)
    html = _ar5iv.get_html(full_text_id, refresh=refresh)
    return parse_source_input(html_text=html, source_id=normalize_paper_id(full_text_id))


def _parsed(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    full_text_id = _full_text_paper_id(paper_id, refresh=refresh)
    normalized = normalize_paper_id(full_text_id)
    path = parsed_source_cache_path(normalized)
    if not refresh and (cached := read_json(path)):
        if cached.get("parser_version") == SOURCE_PARSER_VERSION:
            return cached
    parsed = _parse_ar5iv_source(normalized, refresh=refresh)
    write_json(path, parsed)
    return parsed


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


def _get_or_generate_summary_one(
    paper_id: str,
    *,
    provider: str,
    model: str | None,
    refresh: bool,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    status = _summary_status_or_error(paper_id, refresh=refresh)
    if status["ok"]:
        return status
    if status.get("status") != "needs_llm":
        return status
    return _generate_from_status(paper_id, status, provider=provider, model=model, progress_callback=progress_callback)


def _generate_summary_one(
    paper_id: str,
    *,
    provider: str,
    model: str | None,
    refresh: bool,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    status = _summary_status_or_error(paper_id, refresh=refresh)
    if status["ok"]:
        return status
    if status.get("status") != "needs_llm":
        return status
    return _generate_from_status(paper_id, status, provider=provider, model=model, progress_callback=progress_callback)


def _summary_status_or_error(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    try:
        return _summary_status(paper_id, refresh=refresh)
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("paper_query_error", str(exc))


def _generate_from_status(
    paper_id: str,
    status: dict[str, Any],
    *,
    provider: str,
    model: str | None,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    selected = select_summary_provider(provider)
    if selected.name == "manual":
        return status
    try:
        summary = selected.generate_summary(status["llm_task"], model=model, progress_callback=progress_callback)
        summary_paper_id = str(status.get("paper_id") or paper_id)
        path = store_summary(summary_paper_id, summary)
    except Exception as exc:
        return err("summary_generation_failed", str(exc))
    return ok(summary, provider=selected.name, cache="write", summary_path=str(path))


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


def _map(ids: str | Iterable[str], func: Callable[[str], dict[str, Any]]):
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
