from __future__ import annotations

import json
import re
import shlex
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


def search_parsed_full_text(
    files: list[FullTextSearchFile],
    query: str,
    *,
    limit: int = 20,
    context: int = 1,
    case_sensitive: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_query = (query or "").strip()
    if not normalized_query:
        raise ValueError("query is required")
    normalized_limit = max(1, min(int(limit), 200))
    normalized_context = max(0, min(int(context), 5))
    backend = "python-parsed-json"
    candidate_files = files

    if files and shutil.which("rg"):
        matched_files = _parsed_files_with_ripgrep(files, normalized_query, case_sensitive=case_sensitive)
        if matched_files is not None:
            candidate_files = matched_files
            backend = "ripgrep-parsed-json"

    if not candidate_files:
        return [], {
            "search_backend": backend,
            "searched_files": len(files),
            "limit": normalized_limit,
            "context": normalized_context,
            "case_sensitive": case_sensitive,
            "truncated": False,
        }

    hits, truncated = _search_parsed_candidates(
        candidate_files,
        normalized_query,
        limit=normalized_limit,
        context=normalized_context,
        case_sensitive=case_sensitive,
    )
    return hits, {
        "search_backend": backend,
        "searched_files": len(files),
        "limit": normalized_limit,
        "context": normalized_context,
        "case_sensitive": case_sensitive,
        "truncated": truncated,
    }


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


def _parsed_files_with_ripgrep(
    files: list[FullTextSearchFile],
    query: str,
    *,
    case_sensitive: bool,
) -> list[FullTextSearchFile] | None:
    path_by_text = {str(item.path): item for item in files}
    cmd = [
        shutil.which("rg") or "rg",
        "--files-with-matches",
        "--color",
        "never",
    ]
    if not case_sensitive:
        cmd.append("--ignore-case")
    pattern = _json_search_pattern(query)
    if pattern == query:
        cmd.append("--fixed-strings")
    cmd.extend(["--", pattern])
    cmd.extend(str(item.path) for item in files)
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode not in (0, 1):
        return None
    matched = []
    seen: set[str] = set()
    for raw_path in completed.stdout.splitlines():
        path_text = raw_path.strip()
        if path_text in seen:
            continue
        seen.add(path_text)
        if item := path_by_text.get(path_text):
            matched.append(item)
    return matched


def _json_search_pattern(query: str) -> str:
    tokens = [token for token in re.split(r"\s+", query.strip()) if token]
    if len(tokens) <= 1:
        return query
    return r"(?:\\n|\s)+".join(re.escape(token) for token in tokens)


def _search_parsed_candidates(
    files: list[FullTextSearchFile],
    query: str,
    *,
    limit: int,
    context: int,
    case_sensitive: bool,
) -> tuple[list[dict[str, Any]], bool]:
    hits: list[dict[str, Any]] = []
    hit_by_snippet: dict[tuple[str, str], int] = {}
    normalized_query = _normalize_search_text(query, case_sensitive=case_sensitive)
    for search_file in files:
        parsed = _read_parsed_json(search_file.path)
        if not parsed:
            continue
        paper_id = str(parsed.get("paper_id") or search_file.paper_id)
        for section in parsed.get("sections") or []:
            if not isinstance(section, dict):
                continue
            hit = _section_search_hit(
                paper_id,
                section,
                normalized_query,
                case_sensitive=case_sensitive,
                context=context,
            )
            if not hit:
                continue
            dedupe_key = (hit["paper_id"], _normalize_search_text(hit["snippet"], case_sensitive=False))
            if dedupe_key in hit_by_snippet:
                existing_index = hit_by_snippet[dedupe_key]
                if _is_more_specific_section(hit, hits[existing_index]):
                    hits[existing_index] = hit
                continue
            if len(hits) >= limit:
                return hits, True
            hit_by_snippet[dedupe_key] = len(hits)
            hits.append(hit)
    return hits, False


def _read_parsed_json(path: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _section_search_hit(
    paper_id: str,
    section: dict[str, Any],
    query: str,
    *,
    case_sensitive: bool,
    context: int,
) -> dict[str, Any] | None:
    section_id = str(section.get("section_id") or "")
    section_title = str(section.get("title") or "")
    text = str(section.get("text") or "")
    if _contains_query(section_title, query, case_sensitive=case_sensitive):
        return _parsed_hit(
            paper_id,
            section_id=section_id,
            section_title=section_title,
            snippet=_title_snippet(section_title, text),
            matched_in="section_title",
        )
    if not _contains_query(text, query, case_sensitive=case_sensitive):
        return None
    return _parsed_hit(
        paper_id,
        section_id=section_id,
        section_title=section_title,
        snippet=_section_snippet(text, query, context=context, case_sensitive=case_sensitive),
        matched_in="section_text",
    )


def _parsed_hit(
    paper_id: str,
    *,
    section_id: str,
    section_title: str,
    snippet: str,
    matched_in: str,
) -> dict[str, Any]:
    section_selector = section_id or section_title
    section_mcp = f"get_section(paper_id={_mcp_string(paper_id)}, section={_mcp_string(section_selector)})"
    section_cli = f"arc-paper get-section {shlex.quote(paper_id)} --section {shlex.quote(section_selector)} --json"
    metadata_mcp = f"get_metadata(paper_id={_mcp_string(paper_id)})"
    metadata_cli = f"arc-paper get-metadata {shlex.quote(paper_id)} --json"
    return {
        "paper_id": paper_id,
        "section_id": section_id,
        "section_title": section_title,
        "matched_in": matched_in,
        "snippet": snippet,
        "next_steps": {
            "read_section": {"mcp": section_mcp, "cli": section_cli},
            "get_metadata": {"mcp": metadata_mcp, "cli": metadata_cli},
        },
    }


def _mcp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _is_more_specific_section(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_id = str(candidate.get("section_id") or "")
    current_id = str(current.get("section_id") or "")
    if current_id and candidate_id.startswith(f"{current_id}."):
        return True
    return candidate_id.count(".") > current_id.count(".")


def _contains_query(text: str, query: str, *, case_sensitive: bool) -> bool:
    return query in _normalize_search_text(text, case_sensitive=case_sensitive)


def _normalize_search_text(text: str, *, case_sensitive: bool) -> str:
    compact = " ".join(str(text or "").split())
    return compact if case_sensitive else compact.lower()


def _title_snippet(title: str, text: str) -> str:
    lines = _snippet_lines(text)
    if not lines:
        return _clean_snippet(title)
    return _clean_snippet("\n".join([title, lines[0]]))


def _section_snippet(text: str, query: str, *, context: int, case_sensitive: bool) -> str:
    lines = _snippet_lines(text)
    for index, line in enumerate(lines):
        if _contains_query(line, query, case_sensitive=case_sensitive):
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            return _clean_snippet("\n".join(lines[start:end]))
    compact = _clean_snippet(text, max_length=1200)
    match_at = _normalize_search_text(compact, case_sensitive=case_sensitive).find(query)
    if match_at < 0:
        return compact
    start = max(0, match_at - 300)
    end = min(len(compact), match_at + len(query) + 300)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(compact) else ""
    return f"{prefix}{compact[start:end].strip()}{suffix}"


def _snippet_lines(text: str) -> list[str]:
    return [line for line in (_clean_snippet(raw) for raw in str(text or "").splitlines()) if line]


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


def _clean_snippet(text: str, *, max_length: int = 900) -> str:
    compact = " ".join(str(text or "").split())
    if "<" in compact and ">" in compact:
        compact = BeautifulSoup(compact, "lxml").get_text(" ", strip=True)
        compact = " ".join(compact.split())
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) > max_length:
        return f"{compact[: max_length - 3].rstrip()}..."
    return compact
