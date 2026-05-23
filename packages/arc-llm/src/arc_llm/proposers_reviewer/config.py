from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


BATCH_CONFIG_SCHEMA = "arc.llm.proposers_reviewer_batch.config.v1"
REVIEW_ENVELOPE_SCHEMA = "arc.llm.review_envelope.v1"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PromptConfig:
    system: str
    template: str


@dataclass(frozen=True)
class WorkerConfig:
    id: str
    prompt: PromptConfig
    output_schema: dict[str, Any] | None
    provider: str
    model: str | None
    runtime: dict[str, Any]


@dataclass(frozen=True)
class LoopConfig:
    loop_id: str
    max_rounds: int
    early_stop_enabled: bool
    proposers: list[WorkerConfig]
    reviewers: list[WorkerConfig]
    caller_context: dict[str, Any]


@dataclass(frozen=True)
class BatchConfig:
    schema_version: str
    run_id: str
    run_dir: Path
    max_concurrent_loops: int
    existing_run_policy: str
    fail_fast: bool
    loops: list[LoopConfig]


def load_batch_config(payload: Mapping[str, Any]) -> BatchConfig:
    data = copy.deepcopy(dict(payload))
    schema_version = _required_text(data, "schema_version")
    if schema_version != BATCH_CONFIG_SCHEMA:
        raise ConfigError(f"schema_version must be {BATCH_CONFIG_SCHEMA}")

    run_id = _safe_id(_required_text(data, "run_id"), "run_id")
    run_dir = Path(_required_text(data, "run_dir")).expanduser()
    max_concurrent_loops = _positive_int(data.get("max_concurrent_loops", 1), "max_concurrent_loops")
    existing_run_policy = str(data.get("existing_run_policy", "fail")).strip() or "fail"
    if existing_run_policy not in {"fail", "append_new_loops"}:
        raise ConfigError("existing_run_policy must be fail or append_new_loops")
    fail_fast = bool(data.get("fail_fast", False))

    defaults = _dict(data.get("defaults", {}), "defaults")
    default_runtime = _dict(defaults.get("runtime", {}), "defaults.runtime")
    default_provider = str(defaults.get("provider", "auto") or "auto")
    default_model = defaults.get("model")
    if default_model is not None:
        default_model = str(default_model)

    raw_loops = data.get("loops")
    if not isinstance(raw_loops, list) or not raw_loops:
        raise ConfigError("loops must be a non-empty list")

    loops: list[LoopConfig] = []
    seen_loop_ids: set[str] = set()
    for raw_loop in raw_loops:
        loop = _parse_loop(
            raw_loop,
            default_provider=default_provider,
            default_model=default_model,
            default_runtime=default_runtime,
        )
        if loop.loop_id in seen_loop_ids:
            raise ConfigError(f"duplicate loop_id: {loop.loop_id}")
        seen_loop_ids.add(loop.loop_id)
        loops.append(loop)

    return BatchConfig(
        schema_version=schema_version,
        run_id=run_id,
        run_dir=run_dir,
        max_concurrent_loops=max_concurrent_loops,
        existing_run_policy=existing_run_policy,
        fail_fast=fail_fast,
        loops=loops,
    )


def worker_env(worker: WorkerConfig, *, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    runtime = worker.runtime
    if runtime.get("allow_internet"):
        env["ARC_CODEX_ALLOW_INTERNET"] = "true"
        env["ARC_CLAUDE_ALLOW_INTERNET"] = "true"
    if runtime.get("allow_mcp"):
        env["ARC_CODEX_ENABLE_MCP"] = "true"
        env["ARC_CLAUDE_ALLOW_MCP"] = "true"

    _put(env, "ARC_CODEX_SANDBOX", runtime.get("codex_sandbox"))
    _put(env, "ARC_CODEX_PROFILE", runtime.get("codex_profile"))
    _put(env, "ARC_CODEX_PROFILE_V2", runtime.get("codex_profile_v2"))
    _put(env, "ARC_CODEX_REASONING_EFFORT", runtime.get("codex_reasoning_effort"))
    _put(env, "ARC_CODEX_REASONING_SUMMARY", runtime.get("codex_reasoning_summary"))
    _put(env, "ARC_CODEX_MODEL_VERBOSITY", runtime.get("codex_model_verbosity"))
    _put(env, "ARC_CODEX_WEB_SEARCH", runtime.get("codex_web_search"))
    _put(env, "ARC_CODEX_NETWORK_ACCESS", runtime.get("codex_network_access"))
    _put(env, "ARC_CLAUDE_EFFORT", runtime.get("claude_effort"))
    _put(env, "ARC_CLAUDE_TOOLS", runtime.get("claude_tools"))
    _put(env, "ARC_CLAUDE_MAX_BUDGET_USD", runtime.get("claude_max_budget_usd"))
    _put(env, "ARC_CLAUDE_FALLBACK_MODEL", runtime.get("claude_fallback_model"))
    return env


def _parse_loop(
    raw_loop: Any,
    *,
    default_provider: str,
    default_model: str | None,
    default_runtime: Mapping[str, Any],
) -> LoopConfig:
    loop_data = _dict(raw_loop, "loop")
    loop_id = _safe_id(_required_text(loop_data, "loop_id"), "loop_id")
    max_rounds = _positive_int(loop_data.get("max_rounds"), f"{loop_id}.max_rounds")
    early_stop = _dict(loop_data.get("early_stop", {}), f"{loop_id}.early_stop")
    early_stop_enabled = bool(early_stop.get("enabled", False))
    proposers = _parse_workers(
        loop_data.get("proposers"),
        field_name=f"{loop_id}.proposers",
        default_provider=default_provider,
        default_model=default_model,
        default_runtime=default_runtime,
        duplicate_label="proposer",
    )
    reviewers = _parse_workers(
        loop_data.get("reviewers"),
        field_name=f"{loop_id}.reviewers",
        default_provider=default_provider,
        default_model=default_model,
        default_runtime=default_runtime,
        duplicate_label="reviewer",
    )
    if len(reviewers) != 1:
        raise ConfigError(f"{loop_id} must configure exactly one reviewer in v1")
    return LoopConfig(
        loop_id=loop_id,
        max_rounds=max_rounds,
        early_stop_enabled=early_stop_enabled,
        proposers=proposers,
        reviewers=reviewers,
        caller_context=_dict(loop_data.get("caller_context", {}), f"{loop_id}.caller_context"),
    )


def _parse_workers(
    raw_workers: Any,
    *,
    field_name: str,
    default_provider: str,
    default_model: str | None,
    default_runtime: Mapping[str, Any],
    duplicate_label: str,
) -> list[WorkerConfig]:
    if not isinstance(raw_workers, list) or not raw_workers:
        raise ConfigError(f"{field_name} must be a non-empty list")
    workers: list[WorkerConfig] = []
    seen_ids: set[str] = set()
    for raw_worker in raw_workers:
        worker = _parse_worker(
            raw_worker,
            field_name=field_name,
            default_provider=default_provider,
            default_model=default_model,
            default_runtime=default_runtime,
        )
        if worker.id in seen_ids:
            raise ConfigError(f"duplicate {duplicate_label} id: {worker.id}")
        seen_ids.add(worker.id)
        workers.append(worker)
    return workers


def _parse_worker(
    raw_worker: Any,
    *,
    field_name: str,
    default_provider: str,
    default_model: str | None,
    default_runtime: Mapping[str, Any],
) -> WorkerConfig:
    worker_data = _dict(raw_worker, field_name)
    worker_id = _safe_id(_required_text(worker_data, "id"), f"{field_name}.id")
    prompt_data = _dict(worker_data.get("prompt"), f"{field_name}.{worker_id}.prompt")
    prompt = PromptConfig(
        system=str(prompt_data.get("system", "")),
        template=str(prompt_data.get("template", "")),
    )
    if not prompt.template:
        raise ConfigError(f"{field_name}.{worker_id}.prompt.template is required")

    output_schema = worker_data.get("output_schema")
    if output_schema is not None and not isinstance(output_schema, dict):
        raise ConfigError(f"{field_name}.{worker_id}.output_schema must be an object")

    runtime = dict(default_runtime)
    runtime.update(_dict(worker_data.get("runtime", {}), f"{field_name}.{worker_id}.runtime"))
    provider = str(worker_data.get("provider", default_provider) or "auto")
    model = worker_data.get("model", default_model)
    if model is not None:
        model = str(model)
    return WorkerConfig(
        id=worker_id,
        prompt=prompt,
        output_schema=copy.deepcopy(output_schema),
        provider=provider,
        model=model,
        runtime=runtime,
    )


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        raise ConfigError(f"{key} is required")
    text = str(value).strip()
    if not text:
        raise ConfigError(f"{key} is required")
    return text


def _dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be an object")
    return copy.deepcopy(value)


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ConfigError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return parsed


def _safe_id(value: str, field_name: str) -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise ConfigError(f"{field_name} must contain only letters, numbers, dot, underscore, or dash")
    return value


def _put(env: dict[str, str], key: str, value: Any) -> None:
    if value is not None:
        env[key] = str(value)
