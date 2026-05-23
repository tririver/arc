from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .cache import CachePaths, cache_root, now_iso, read_json, text_query_cache_path, write_json
from .ids import arxiv_path_id
from .ids import extract_paper_ids as _extract_paper_ids
from .ids import normalize_paper_id
from .ids import paper_ids_safe_dir_name as _paper_ids_safe_dir_name
from .parse.ar5iv_html import PARSER_VERSION
from .parse.ar5iv_html import get_section as parsed_get_section
from .parse.ar5iv_html import parse_html
from .parse.equations import find_equation_context
from .providers import Ar5ivProvider, InspireProvider
from .providers.base import ProviderError
from .reference_inference import ReferenceInferenceError, infer_main_references
from .results import err, ok
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
        full_text_id = _full_text_paper_id(paper_id, refresh=refresh)
        html = _ar5iv.get_html(full_text_id, refresh=refresh)
        return ok(find_equation_context(html, query), provider="ar5iv")
    except ProviderError as exc:
        return err(exc.code, exc.message)
    except Exception as exc:
        return err("paper_query_error", str(exc))


def _parsed(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    full_text_id = _full_text_paper_id(paper_id, refresh=refresh)
    paths = CachePaths.for_paper(full_text_id)
    if not refresh and (cached := read_json(paths.ar5iv_parsed)):
        if cached.get("parser_version") == PARSER_VERSION:
            return cached
    html = _ar5iv.get_html(full_text_id, refresh=refresh)
    parsed = parse_html(html, paper_id=normalize_paper_id(full_text_id))
    write_json(paths.ar5iv_parsed, parsed)
    return parsed


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
    source_hash = task["input_pack"]["source_hash"]
    if not refresh and (cached := read_summary(paper_id, source_hash=source_hash)):
        return ok(cached, provider="local-cache", cache="hit")
    return _needs_llm(paper_id, task)


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
        path = store_summary(paper_id, summary)
    except Exception as exc:
        return err("summary_generation_failed", str(exc))
    return ok(summary, provider=selected.name, cache="write", summary_path=str(path))


def _build_summary_task(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    metadata = _inspire.get_metadata(paper_id, refresh=refresh)
    parsed = _parsed(paper_id, refresh=refresh)
    input_pack = build_input_pack(
        paper_id,
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
