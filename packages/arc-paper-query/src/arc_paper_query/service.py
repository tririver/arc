from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .cache import CachePaths, read_json, write_json
from .ids import normalize_paper_id
from .parse.ar5iv_html import get_section as parsed_get_section
from .parse.ar5iv_html import parse_html
from .parse.equations import find_equation_context
from .providers import Ar5ivProvider, InspireProvider
from .providers.base import ProviderError
from .results import err, ok
from .summary.input_pack import build_input_pack
from .summary.providers.select import select_summary_provider
from .summary.schema import load_summary_prompt, load_summary_schema
from .summary.store import read_summary, store_summary


_inspire = InspireProvider()
_ar5iv = Ar5ivProvider()


def get_title(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _metadata_field(paper_id, "title", refresh=refresh))


def get_abstract(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _metadata_field(paper_id, "abstract", refresh=refresh))


def get_authors(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _metadata_field(paper_id, "authors", refresh=refresh))


def get_references(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _call(lambda: _inspire.get_references(paper_id, refresh=refresh), "inspire"))


def get_citers(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _call(lambda: _inspire.get_citers(paper_id, refresh=refresh), "inspire"))


def get_citer_count(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _call(lambda: _inspire.get_citer_count(paper_id, refresh=refresh), "inspire"))


def get_toc(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _call(lambda: _parsed(paper_id, refresh=refresh)["toc"], "ar5iv"))


def get_section(ids: str | Iterable[str], section: str, *, refresh: bool = False):
    return _map(ids, lambda paper_id: _section_one(paper_id, section, refresh=refresh))


def get_equation_context(ids: str | Iterable[str], query: str, *, refresh: bool = False):
    return _map(ids, lambda paper_id: _equation_one(paper_id, query, refresh=refresh))


def get_llm_summary(ids: str | Iterable[str], *, refresh: bool = False):
    return _map(ids, lambda paper_id: _summary_status(paper_id, refresh=refresh))


def generate_llm_summary(
    ids: str | Iterable[str],
    *,
    provider: str = "auto",
    model: str | None = None,
    refresh: bool = False,
):
    return _map(ids, lambda paper_id: _generate_summary_one(paper_id, provider=provider, model=model, refresh=refresh))


def store_llm_summary(paper_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    try:
        path = store_summary(paper_id, summary)
    except Exception as exc:
        return err("summary_store_failed", str(exc))
    return ok({"summary_path": str(path), "summary": summary}, provider="local-cache", cache="write")


def _metadata_field(paper_id: str, field: str, *, refresh: bool) -> dict[str, Any]:
    return _call(lambda: _inspire.get_metadata(paper_id, refresh=refresh).get(field), "inspire")


def _section_one(paper_id: str, section: str, *, refresh: bool) -> dict[str, Any]:
    parsed = _parsed(paper_id, refresh=refresh)
    result = parsed_get_section(parsed, section)
    if result["ok"]:
        result["meta"] = {"provider": "ar5iv"}
    return result


def _equation_one(paper_id: str, query: str, *, refresh: bool) -> dict[str, Any]:
    html = _ar5iv.get_html(paper_id, refresh=refresh)
    return ok(find_equation_context(html, query), provider="ar5iv")


def _parsed(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    paths = CachePaths.for_paper(paper_id)
    if not refresh and (cached := read_json(paths.ar5iv_parsed)):
        return cached
    html = _ar5iv.get_html(paper_id, refresh=refresh)
    parsed = parse_html(html, paper_id=normalize_paper_id(paper_id))
    write_json(paths.ar5iv_parsed, parsed)
    return parsed


def _summary_status(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    task = _build_summary_task(paper_id, refresh=refresh)
    source_hash = task["input_pack"]["source_hash"]
    if not refresh and (cached := read_summary(paper_id, source_hash=source_hash)):
        return ok(cached, provider="local-cache", cache="hit")
    return _needs_llm(paper_id, task)


def _generate_summary_one(paper_id: str, *, provider: str, model: str | None, refresh: bool) -> dict[str, Any]:
    status = _summary_status(paper_id, refresh=refresh)
    if status["ok"]:
        return status
    selected = select_summary_provider(provider)
    if selected.name == "manual":
        return status
    try:
        summary = selected.generate_summary(status["llm_task"], model=model)
        path = store_summary(paper_id, summary)
    except Exception as exc:
        return err("summary_generation_failed", str(exc))
    return ok(summary, provider=selected.name, cache="write", summary_path=str(path))


def _build_summary_task(paper_id: str, *, refresh: bool) -> dict[str, Any]:
    metadata = _inspire.get_metadata(paper_id, refresh=refresh)
    parsed = _parsed(paper_id, refresh=refresh)
    references = _inspire.get_references(paper_id, refresh=refresh)
    input_pack = build_input_pack(
        paper_id,
        metadata=metadata,
        parsed=parsed,
        references=references,
    )
    return {
        "task_type": "paper_summary",
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
            "store_command": f"arc-paper-query store-llm-summary {normalized} --summary-json -"
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
