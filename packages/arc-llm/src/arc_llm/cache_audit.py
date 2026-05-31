from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_run(run_root: Path | str) -> dict[str, Any]:
    root = Path(run_root)
    call_paths = _session_call_paths(root)
    calls = _read_calls(call_paths)
    total_input = sum(_input_tokens_from_usage(call.get("usage", {})) for call in calls)
    total_cached = sum(_cached_tokens_from_usage(call.get("usage", {})) for call in calls)
    total_output = sum(_int(call.get("usage", {}).get("output_tokens")) for call in calls)
    groups: dict[str, dict[str, Any]] = {}
    for call in calls:
        usage = call.get("usage", {})
        key = "|".join(
            [
                str(call.get("provider_used") or ""),
                str(call.get("model_used") or ""),
                str(call.get("runtime_fingerprint") or ""),
                str(call.get("static_prefix_sha256") or ""),
            ]
        )
        group = groups.setdefault(key, {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0})
        group["calls"] += 1
        group["input_tokens"] += _input_tokens_from_usage(usage)
        group["cached_input_tokens"] += _cached_tokens_from_usage(usage)
        group["output_tokens"] += _int(usage.get("output_tokens")) if isinstance(usage, dict) else 0
    worst = sorted(
        calls,
        key=lambda call: _input_tokens_from_usage(call.get("usage", {}))
        - _cached_tokens_from_usage(call.get("usage", {})),
        reverse=True,
    )[:20]
    return {
        "schema_version": "arc.llm.cache_audit.v1",
        "run_root": str(root),
        "session_call_paths": [str(path) for path in call_paths],
        "total_calls": len(calls),
        "total_input_tokens": total_input,
        "total_cached_input_tokens": total_cached,
        "total_output_tokens": total_output,
        "overall_cached_input_ratio": (total_cached / max(1, total_input)) if total_input else None,
        "groups": groups,
        "worst_cache_miss_calls": worst,
        "duplicate_context_warnings": _duplicate_context_warnings(root),
        "schema_change_warnings": _schema_change_warnings(calls),
    }


def first_difference(a: str, b: str) -> dict[str, Any]:
    left_lines = a.splitlines()
    right_lines = b.splitlines()
    for line_index in range(max(len(left_lines), len(right_lines))):
        left = left_lines[line_index] if line_index < len(left_lines) else ""
        right = right_lines[line_index] if line_index < len(right_lines) else ""
        if left == right:
            continue
        for col_index in range(max(len(left), len(right))):
            left_char = left[col_index] if col_index < len(left) else ""
            right_char = right[col_index] if col_index < len(right) else ""
            if left_char != right_char:
                return {
                    "line": line_index + 1,
                    "column": col_index + 1,
                    "left_snippet": left[max(0, col_index - 40) : col_index + 80],
                    "right_snippet": right[max(0, col_index - 40) : col_index + 80],
                }
    return {"line": None, "column": None, "left_snippet": "", "right_snippet": ""}


def _session_call_paths(root: Path) -> list[Path]:
    paths = [
        root / "sessions" / "calls.jsonl",
        root / "llm_sessions" / "calls.jsonl",
        root / "idea_loops" / "sessions" / "calls.jsonl",
        root / "attempt_batches" / "_sessions" / "calls.jsonl",
    ]
    config = _read_json(root / "config.json")
    session = config.get("session") if isinstance(config, dict) else None
    loop_sessions = (
        [
            loop.get("session")
            for loop in config.get("loops", [])
            if isinstance(loop, dict) and isinstance(loop.get("session"), dict)
        ]
        if isinstance(config, dict)
        else []
    )
    sessions = [item for item in [session, *loop_sessions] if isinstance(item, dict)]
    for session in sessions:
        session_root = session.get("root")
        if session_root:
            paths.append(Path(str(session_root)).expanduser() / "calls.jsonl")
        elif session.get("reuse_across_batch_calls"):
            run_dir = config.get("run_dir")
            if run_dir:
                paths.append(Path(str(run_dir)).expanduser() / "_sessions" / "calls.jsonl")
    if root.exists():
        recursive_count = 0
        for path in sorted(root.rglob("calls.jsonl")):
            if {"sessions", "_sessions", "llm_sessions"} & set(path.parts):
                paths.append(path)
                recursive_count += 1
                if recursive_count >= 100:
                    break
    return _dedupe_paths(paths)


def _read_calls(paths: list[Path]) -> list[dict[str, Any]]:
    calls = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                calls.append(payload)
    return calls


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _duplicate_context_warnings(root: Path) -> list[dict[str, Any]]:
    warnings = []
    for path in sorted(root.glob("loops/*/rounds/round_*/prompts/*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        caller_count = text.count("caller_context")
        if "## ARC Worker Context" in text and caller_count >= 2:
            warnings.append(
                {
                    "path": str(path),
                    "warning": "prompt appears to contain both inline caller_context and appended ARC Worker Context",
                    "caller_context_mentions": caller_count,
                }
            )
    return warnings


def _schema_change_warnings(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session: dict[str, set[str]] = {}
    for call in calls:
        session_key = call.get("session_key")
        schema_sha = call.get("schema_sha256")
        if not session_key or not schema_sha:
            continue
        by_session.setdefault(str(session_key), set()).add(str(schema_sha))
    return [
        {
            "session_key": session_key,
            "distinct_schema_sha256_count": len(values),
            "schema_sha256_values": sorted(values),
        }
        for session_key, values in sorted(by_session.items())
        if len(values) > 1
    ]


def _input_tokens_from_usage(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    if usage.get("total_input_tokens") is not None:
        return _int(usage.get("total_input_tokens"))
    if usage.get("cache_creation_input_tokens") is not None or usage.get("cache_read_input_tokens") is not None:
        return (
            _int(usage.get("input_tokens"))
            + _int(usage.get("cache_creation_input_tokens"))
            + _int(usage.get("cache_read_input_tokens"))
        )
    if usage.get("input_tokens") is not None:
        return _int(usage.get("input_tokens"))
    return 0


def _cached_tokens_from_usage(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    if usage.get("effective_cached_input_tokens") is not None:
        return _int(usage.get("effective_cached_input_tokens"))
    if usage.get("cache_creation_input_tokens") is not None or usage.get("cache_read_input_tokens") is not None:
        return _int(usage.get("cache_read_input_tokens"))
    if usage.get("cached_input_tokens") is not None:
        return _int(usage.get("cached_input_tokens"))
    return 0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
