from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


def read_json_template(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON template must contain an object: {path}")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def replace_placeholders(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for placeholder, replacement in replacements.items():
            result = result.replace(placeholder, replacement)
        return result
    if isinstance(value, list):
        return [replace_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_placeholders(item, replacements) for key, item in value.items()}
    return copy.deepcopy(value)


def materialize_worker(
    template: Mapping[str, Any],
    *,
    worker_id: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    replacements: Mapping[str, str] | None = None,
    output_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = replace_placeholders(template, replacements or {})
    if overrides:
        payload = deep_merge(payload, overrides)
    if worker_id is not None:
        payload["id"] = worker_id
    if output_schema is not None:
        payload["output_schema"] = copy.deepcopy(dict(output_schema))
    return payload


def materialize_loop(
    loop_template: Mapping[str, Any],
    *,
    loop_id: str,
    caller_context: Mapping[str, Any],
    proposers: list[Mapping[str, Any]],
    reviewers: list[Mapping[str, Any]],
    session: Mapping[str, Any] | None = None,
    cache_context: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(loop_template))
    if overrides:
        payload = deep_merge(payload, overrides)
    payload["loop_id"] = loop_id
    payload["caller_context"] = copy.deepcopy(dict(caller_context))
    payload["proposers"] = [copy.deepcopy(dict(item)) for item in proposers]
    payload["reviewers"] = [copy.deepcopy(dict(item)) for item in reviewers]
    if session is not None:
        payload["session"] = copy.deepcopy(dict(session))
    if cache_context is not None:
        payload["cache_context"] = copy.deepcopy(dict(cache_context))
    return payload


def materialize_batch(
    *,
    run_id: str,
    run_dir: Path | str,
    loops: list[Mapping[str, Any]],
    defaults: Mapping[str, Any] | None = None,
    artifact_options: Mapping[str, Any] | None = None,
    output_recovery: Mapping[str, Any] | None = None,
    session: Mapping[str, Any] | None = None,
    max_concurrent_loops: int = 1,
    fail_fast: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "max_concurrent_loops": int(max_concurrent_loops),
        "fail_fast": bool(fail_fast),
        "loops": [copy.deepcopy(dict(loop)) for loop in loops],
    }
    if defaults is not None:
        payload["defaults"] = copy.deepcopy(dict(defaults))
    if artifact_options is not None:
        payload["artifact_options"] = copy.deepcopy(dict(artifact_options))
    payload["output_recovery"] = copy.deepcopy(
        dict(
            output_recovery
            or {
                "enabled": True,
                "mode": "warn",
                "allow_natural_language": True,
            }
        )
    )
    if session is not None:
        payload["session"] = copy.deepcopy(dict(session))
    return payload
