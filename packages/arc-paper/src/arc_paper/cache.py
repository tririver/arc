from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import get_ident
from typing import Any
from urllib.parse import quote

from .ids import normalize_paper_id


ONE_MONTH_SECONDS = 30 * 24 * 60 * 60


@dataclass(frozen=True)
class CachePaths:
    paper_id: str
    paper_dir: Path
    ar5iv_html: Path
    ar5iv_parsed: Path
    inspire_metadata: Path
    inspire_references: Path
    inspire_citers: Path

    @classmethod
    def for_paper(cls, paper_id: str) -> "CachePaths":
        normalized = normalize_paper_id(paper_id)
        paper_dir = cache_root() / "papers" / quote(normalized, safe="")
        return cls(
            paper_id=normalized,
            paper_dir=paper_dir,
            ar5iv_html=paper_dir / "ar5iv" / "fulltext.html",
            ar5iv_parsed=paper_dir / "ar5iv" / "parsed.json",
            inspire_metadata=paper_dir / "inspire" / "metadata.json",
            inspire_references=paper_dir / "inspire" / "references.json",
            inspire_citers=paper_dir / "inspire" / "citers.json",
        )

    def summary_path(self, prompt_version: str, source_hash: str) -> Path:
        return self.paper_dir / "summaries" / prompt_version / f"{source_hash}.json"


def cache_root() -> Path:
    if value := os.environ.get("ARC_PAPER_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("XDG_CACHE_HOME"):
        return Path(value).expanduser() / "arc" / "arc-paper"
    if project_root := _project_root():
        return project_root / "cache" / "arc-paper"
    return Path.home() / ".cache" / "arc" / "arc-paper"


def text_query_cache_path(namespace: str, text: str) -> Path:
    key = hashlib.sha1((text or "").strip().encode("utf-8")).hexdigest()
    return cache_root() / "queries" / namespace / f"{key}.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, *, ttl_seconds: int | None = None) -> Any | None:
    if not _is_fresh(path, ttl_seconds=ttl_seconds):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp_path(path)
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_text(path: Path, *, ttl_seconds: int | None = None) -> str | None:
    if not _is_fresh(path, ttl_seconds=ttl_seconds):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp_path(path)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _is_fresh(path: Path, *, ttl_seconds: int | None) -> bool:
    if not path.exists():
        return False
    if ttl_seconds is not None and ttl_seconds >= 0:
        age = time.time() - path.stat().st_mtime
        if age > ttl_seconds:
            return False
    return True


def _unique_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{get_ident()}.{time.time_ns()}.tmp")


def _project_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "packages" / "arc-paper" / "pyproject.toml").is_file():
            return parent
    return None
