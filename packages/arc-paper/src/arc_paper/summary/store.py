from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..cache import CachePaths, read_json, write_json
from ..ids import normalize_paper_id
from .schema import PROMPT_VERSION, validate_summary


class SummaryStoreError(RuntimeError):
    pass


def store_summary(paper_id: str, summary: dict[str, Any]) -> Path:
    normalized = normalize_paper_id(paper_id)
    validate_summary(summary)
    if normalize_paper_id(summary.get("paper_id", "")) != normalized:
        raise SummaryStoreError("summary paper_id does not match requested paper_id")
    provenance = summary["provenance"]
    prompt_version = provenance.get("prompt_version") or PROMPT_VERSION
    source_hash = provenance["source_hash"]
    path = CachePaths.for_paper(normalized).summary_path(
        prompt_version,
        source_hash,
        provider=_provenance_text(provenance, "method"),
        model=_provenance_text(provenance, "model"),
    )
    write_json(path, summary)
    write_json(_latest_summary_path(normalized, prompt_version), summary)
    return path


def read_summary(
    paper_id: str,
    *,
    prompt_version: str = PROMPT_VERSION,
    source_hash: str,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    if provider or model:
        path = CachePaths.for_paper(paper_id).summary_path(
            prompt_version,
            source_hash,
            provider=provider,
            model=model,
        )
        cached = _validated_summary_or_none(read_json(path))
        if cached:
            return cached
        if provider and model is None:
            return _matching_latest_summary(
                paper_id,
                prompt_version=prompt_version,
                source_hash=source_hash,
                provider=provider,
            )
        return None
    cached = _validated_summary_or_none(
        read_json(CachePaths.for_paper(paper_id).summary_path(prompt_version, source_hash))
    )
    if cached:
        return cached
    latest = read_latest_summary(paper_id, prompt_version=prompt_version)
    if latest and (latest.get("provenance") or {}).get("source_hash") == source_hash:
        return latest
    return None


def _matching_latest_summary(
    paper_id: str,
    *,
    prompt_version: str,
    source_hash: str,
    provider: str,
) -> dict[str, Any] | None:
    latest = read_latest_summary(paper_id, prompt_version=prompt_version)
    provenance = latest.get("provenance") if isinstance(latest, dict) else None
    if not isinstance(provenance, dict):
        return None
    if provenance.get("source_hash") != source_hash:
        return None
    if provenance.get("method") != provider:
        return None
    return latest


def read_latest_summary(
    paper_id: str,
    *,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any] | None:
    return _validated_summary_or_none(read_json(_latest_summary_path(paper_id, prompt_version)))


def store_section_summary(
    paper_id: str,
    *,
    prompt_version: str = PROMPT_VERSION,
    source_hash: str,
    provider: str | None = None,
    model: str | None = None,
    section_index: int,
    section_id: str,
    summary: dict[str, Any],
) -> Path:
    path = _section_summary_path(
        paper_id,
        prompt_version=prompt_version,
        source_hash=source_hash,
        provider=provider,
        model=model,
        section_index=section_index,
        section_id=section_id,
    )
    write_json(path, summary)
    return path


def read_section_summary(
    paper_id: str,
    *,
    prompt_version: str = PROMPT_VERSION,
    source_hash: str,
    provider: str | None = None,
    model: str | None = None,
    section_index: int,
    section_id: str,
) -> dict[str, Any] | None:
    return read_json(
        _section_summary_path(
            paper_id,
            prompt_version=prompt_version,
            source_hash=source_hash,
            provider=provider,
            model=model,
            section_index=section_index,
            section_id=section_id,
        )
    )


def _latest_summary_path(paper_id: str, prompt_version: str) -> Path:
    paths = CachePaths.for_paper(paper_id)
    return paths.paper_dir / "summaries" / prompt_version / "latest.json"


def _section_summary_path(
    paper_id: str,
    *,
    prompt_version: str,
    source_hash: str,
    provider: str | None,
    model: str | None,
    section_index: int,
    section_id: str,
) -> Path:
    paths = CachePaths.for_paper(paper_id)
    safe_section = quote(section_id or "section", safe="")
    filename = f"{section_index:04d}-{safe_section}.json"
    if provider or model:
        provider_dir = quote(provider or "unknown", safe="")
        model_dir = quote(model or "default", safe="")
        return (
            paths.paper_dir
            / "summaries"
            / prompt_version
            / "providers"
            / provider_dir
            / model_dir
            / f"{source_hash}.sections"
            / filename
        )
    return paths.paper_dir / "summaries" / prompt_version / f"{source_hash}.sections" / filename


def _provenance_text(provenance: dict[str, Any], key: str) -> str | None:
    value = provenance.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validated_summary_or_none(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    try:
        validate_summary(data)
    except Exception:
        return None
    return data
