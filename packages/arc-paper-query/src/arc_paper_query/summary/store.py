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
    path = CachePaths.for_paper(normalized).summary_path(prompt_version, source_hash)
    write_json(path, summary)
    write_json(_latest_summary_path(normalized, prompt_version), summary)
    return path


def read_summary(
    paper_id: str,
    *,
    prompt_version: str = PROMPT_VERSION,
    source_hash: str,
) -> dict[str, Any] | None:
    path = CachePaths.for_paper(paper_id).summary_path(prompt_version, source_hash)
    return _validated_summary_or_none(read_json(path))


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
    section_index: int,
    section_id: str,
    summary: dict[str, Any],
) -> Path:
    path = _section_summary_path(
        paper_id,
        prompt_version=prompt_version,
        source_hash=source_hash,
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
    section_index: int,
    section_id: str,
) -> dict[str, Any] | None:
    return read_json(
        _section_summary_path(
            paper_id,
            prompt_version=prompt_version,
            source_hash=source_hash,
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
    section_index: int,
    section_id: str,
) -> Path:
    paths = CachePaths.for_paper(paper_id)
    safe_section = quote(section_id or "section", safe="")
    filename = f"{section_index:04d}-{safe_section}.json"
    return paths.paper_dir / "summaries" / prompt_version / f"{source_hash}.sections" / filename


def _validated_summary_or_none(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    try:
        validate_summary(data)
    except Exception:
        return None
    return data
