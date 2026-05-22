from __future__ import annotations

from pathlib import Path
from typing import Any

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
    return path


def read_summary(
    paper_id: str,
    *,
    prompt_version: str = PROMPT_VERSION,
    source_hash: str,
) -> dict[str, Any] | None:
    path = CachePaths.for_paper(paper_id).summary_path(prompt_version, source_hash)
    return read_json(path)
