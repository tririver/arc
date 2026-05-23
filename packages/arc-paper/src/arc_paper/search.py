from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class FullTextSearchFile:
    paper_id: str
    path: Path


def search_cached_full_text(
    files: list[FullTextSearchFile],
    query: str,
    *,
    limit: int = 20,
    context: int = 0,
    case_sensitive: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_query = (query or "").strip()
    if not normalized_query:
        raise ValueError("query is required")
    normalized_limit = max(1, min(int(limit), 200))
    normalized_context = max(0, min(int(context), 5))
    if not files:
        return [], {
            "search_backend": _backend_name(),
            "searched_files": 0,
            "limit": normalized_limit,
            "context": normalized_context,
            "case_sensitive": case_sensitive,
            "truncated": False,
        }

    if shutil.which("rg"):
        ripgrep_result = _search_with_ripgrep(
            files,
            normalized_query,
            limit=normalized_limit,
            context=normalized_context,
            case_sensitive=case_sensitive,
        )
        if ripgrep_result is not None:
            hits, truncated = ripgrep_result
            backend = "ripgrep"
        else:
            hits, truncated = _search_with_python(
                files,
                normalized_query,
                limit=normalized_limit,
                context=normalized_context,
                case_sensitive=case_sensitive,
            )
            backend = "python"
    else:
        hits, truncated = _search_with_python(
            files,
            normalized_query,
            limit=normalized_limit,
            context=normalized_context,
            case_sensitive=case_sensitive,
        )
        backend = "python"

    return hits, {
        "search_backend": backend,
        "searched_files": len(files),
        "limit": normalized_limit,
        "context": normalized_context,
        "case_sensitive": case_sensitive,
        "truncated": truncated,
    }


def _backend_name() -> str:
    return "ripgrep" if shutil.which("rg") else "python"


def _search_with_ripgrep(
    files: list[FullTextSearchFile],
    query: str,
    *,
    limit: int,
    context: int,
    case_sensitive: bool,
) -> tuple[list[dict[str, Any]], bool] | None:
    file_by_path = {str(item.path): item for item in files}
    cmd = [
        shutil.which("rg") or "rg",
        "--json",
        "--fixed-strings",
        "--line-number",
        "--color",
        "never",
        "--max-count",
        str(limit + 1),
    ]
    if not case_sensitive:
        cmd.append("--ignore-case")
    cmd.extend(["--", query])
    cmd.extend(str(item.path) for item in files)
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode not in (0, 1):
        return None

    hits: list[dict[str, Any]] = []
    truncated = False
    for raw_event in completed.stdout.splitlines():
        try:
            event = json.loads(raw_event)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data") or {}
        path_text = ((data.get("path") or {}).get("text") or "").strip()
        search_file = file_by_path.get(path_text)
        if search_file is None:
            continue
        line_number = int(data.get("line_number") or 0)
        line_text = str((data.get("lines") or {}).get("text") or "").rstrip("\n")
        if len(hits) >= limit:
            truncated = True
            break
        hits.append(_hit(search_file, line_number, line_text, context=context))
    return hits, truncated


def _search_with_python(
    files: list[FullTextSearchFile],
    query: str,
    *,
    limit: int,
    context: int,
    case_sensitive: bool,
) -> tuple[list[dict[str, Any]], bool]:
    hits: list[dict[str, Any]] = []
    truncated = False
    needle = query if case_sensitive else query.lower()
    for search_file in files:
        try:
            lines = search_file.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines, start=1):
            haystack = line if case_sensitive else line.lower()
            if needle not in haystack:
                continue
            if len(hits) >= limit:
                truncated = True
                return hits, truncated
            hits.append(_hit(search_file, index, line, context=context, lines=lines))
    return hits, truncated


def _hit(
    search_file: FullTextSearchFile,
    line_number: int,
    line_text: str,
    *,
    context: int,
    lines: list[str] | None = None,
) -> dict[str, Any]:
    if lines is None and context:
        try:
            lines = search_file.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
    context_before: list[str] = []
    context_after: list[str] = []
    if lines and context:
        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)
        context_before = [_clean_snippet(line) for line in lines[start : line_number - 1]]
        context_after = [_clean_snippet(line) for line in lines[line_number:end]]
    return {
        "paper_id": search_file.paper_id,
        "line_number": line_number,
        "snippet": _clean_snippet(line_text),
        "context_before": [item for item in context_before if item],
        "context_after": [item for item in context_after if item],
        "cache_path": str(search_file.path),
    }


def _clean_snippet(text: str) -> str:
    compact = " ".join(str(text or "").split())
    if "<" in compact and ">" in compact:
        compact = BeautifulSoup(compact, "lxml").get_text(" ", strip=True)
        compact = " ".join(compact.split())
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) > 500:
        return f"{compact[:497].rstrip()}..."
    return compact
