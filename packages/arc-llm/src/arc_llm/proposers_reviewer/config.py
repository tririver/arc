from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from arc_llm.call_record import allow_arc_llm_call_record
from arc_llm.model import VALID_MODEL_TIERS


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
    model_tier: str | None
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
class ArtifactOptions:
    save_prompts: bool


@dataclass(frozen=True)
class BatchConfig:
    schema_version: str
    run_id: str
    run_dir: Path
    max_concurrent_loops: int
    fail_fast: bool
    artifact_options: ArtifactOptions
    loops: list[LoopConfig]


def load_batch_config(payload: Mapping[str, Any]) -> BatchConfig:
    data = copy.deepcopy(dict(payload))
    schema_version = _required_text(data, "schema_version")
    if schema_version != BATCH_CONFIG_SCHEMA:
        raise ConfigError(f"schema_version must be {BATCH_CONFIG_SCHEMA}")

    run_id = _safe_id(_required_text(data, "run_id"), "run_id")
    run_dir = Path(_required_text(data, "run_dir")).expanduser()
    max_concurrent_loops = _positive_int(data.get("max_concurrent_loops", 1), "max_concurrent_loops")
    fail_fast = _bool(data.get("fail_fast", False), "fail_fast")
    artifact_options = _parse_artifact_options(data.get("artifact_options", {}))

    defaults = _dict(data.get("defaults", {}), "defaults")
    default_runtime = _dict(defaults.get("runtime", {}), "defaults.runtime")
    default_provider = str(defaults.get("provider", "auto") or "auto")
    default_model = defaults.get("model")
    if default_model is not None:
        default_model = str(default_model)
    _validate_exact_model_provider(default_model, default_provider, "defaults")
    default_model_tier = _model_tier(defaults.get("model_tier"), "defaults.model_tier")

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
            default_model_tier=default_model_tier,
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
        fail_fast=fail_fast,
        artifact_options=artifact_options,
        loops=loops,
    )


def worker_env(worker: WorkerConfig, *, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    runtime = worker.runtime
    allow_internet = _bool(runtime.get("allow_internet", False), "runtime.allow_internet")
    allow_mcp = _bool(runtime.get("allow_mcp", False), "runtime.allow_mcp")
    if allow_internet:
        env["ARC_CODEX_ALLOW_INTERNET"] = "true"
        env["ARC_CLAUDE_ALLOW_INTERNET"] = "true"
    else:
        env["ARC_CODEX_ALLOW_INTERNET"] = "false"
        env["ARC_CLAUDE_ALLOW_INTERNET"] = "false"
        env["ARC_CODEX_WEB_SEARCH"] = "disabled"
        env["ARC_CODEX_NETWORK_ACCESS"] = "false"
        env.pop("ARC_CLAUDE_TOOLS", None)
    if allow_mcp:
        mcp_mode = _mcp_mode(runtime.get("mcp_mode"), "runtime.mcp_mode")
        env["ARC_CODEX_ENABLE_MCP"] = "true"
        env["ARC_CLAUDE_ALLOW_MCP"] = "true"
        _put(env, "ARC_CODEX_MCP_MODE", mcp_mode)
        _put(env, "ARC_CLAUDE_MCP_MODE", mcp_mode)
    else:
        env["ARC_CODEX_ENABLE_MCP"] = "false"
        env["ARC_CLAUDE_ALLOW_MCP"] = "false"
        env.pop("ARC_CODEX_MCP_MODE", None)
        env.pop("ARC_CLAUDE_MCP_MODE", None)
        env.pop("ARC_CLAUDE_MCP_CONFIG", None)
        env.pop("ARC_CLAUDE_MCP_CONFIG_JSON", None)
    if worker.model_tier:
        env.setdefault("ARC_CODEX_REASONING_EFFORT", _codex_effort_for_model_tier(worker.model_tier))
        env.setdefault("ARC_CLAUDE_EFFORT", _claude_effort_for_model_tier(worker.model_tier))

    _put(env, "ARC_CODEX_SANDBOX", runtime.get("codex_sandbox"))
    _put(env, "ARC_CODEX_PROFILE", runtime.get("codex_profile"))
    _put(env, "ARC_CODEX_PROFILE_V2", runtime.get("codex_profile_v2"))
    _put(env, "ARC_CODEX_WORK_DIR", runtime.get("codex_work_dir"))
    _put_path_list(env, "ARC_CODEX_ADD_DIRS", runtime.get("codex_add_dirs"))
    _put(env, "ARC_CODEX_ARC_MCP_COMMAND", runtime.get("arc_mcp_command"))
    _put_json_object(env, "ARC_CODEX_ARC_MCP_ENV_JSON", runtime.get("arc_mcp_env"))
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
    default_model_tier: str | None,
    default_runtime: Mapping[str, Any],
) -> LoopConfig:
    loop_data = _dict(raw_loop, "loop")
    loop_id = _safe_id(_required_text(loop_data, "loop_id"), "loop_id")
    max_rounds = _positive_int(loop_data.get("max_rounds"), f"{loop_id}.max_rounds")
    early_stop = _dict(loop_data.get("early_stop", {}), f"{loop_id}.early_stop")
    early_stop_enabled = _bool(early_stop.get("enabled", False), f"{loop_id}.early_stop.enabled")
    proposers = _parse_workers(
        loop_data.get("proposers"),
        field_name=f"{loop_id}.proposers",
        default_provider=default_provider,
        default_model=default_model,
        default_model_tier=default_model_tier,
        default_runtime=default_runtime,
        duplicate_label="proposer",
    )
    reviewers = _parse_workers(
        loop_data.get("reviewers"),
        field_name=f"{loop_id}.reviewers",
        default_provider=default_provider,
        default_model=default_model,
        default_model_tier=default_model_tier,
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
    default_model_tier: str | None,
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
            default_model_tier=default_model_tier,
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
    default_model_tier: str | None,
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
    if output_schema is not None:
        output_schema = allow_arc_llm_call_record(output_schema)

    runtime = dict(default_runtime)
    runtime.update(_dict(worker_data.get("runtime", {}), f"{field_name}.{worker_id}.runtime"))
    provider = str(worker_data.get("provider", default_provider) or "auto")
    model = worker_data.get("model", default_model)
    if model is not None:
        model = str(model)
    _validate_exact_model_provider(model, provider, f"{field_name}.{worker_id}")
    model_tier = _model_tier(worker_data.get("model_tier", default_model_tier), f"{field_name}.{worker_id}.model_tier")
    return WorkerConfig(
        id=worker_id,
        prompt=prompt,
        output_schema=copy.deepcopy(output_schema),
        provider=provider,
        model=model,
        model_tier=model_tier,
        runtime=runtime,
    )


def _parse_artifact_options(raw_options: Any) -> ArtifactOptions:
    options = _dict(raw_options, "artifact_options")
    return ArtifactOptions(save_prompts=_bool(options.get("save_prompts", True), "artifact_options.save_prompts"))


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


def _bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"{field_name} must be a boolean")


def _safe_id(value: str, field_name: str) -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise ConfigError(f"{field_name} must contain only letters, numbers, dot, underscore, or dash")
    return value


def _put(env: dict[str, str], key: str, value: Any) -> None:
    if value is not None:
        env[key] = str(value)


def _put_path_list(env: dict[str, str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        items = value
    else:
        raise ConfigError(f"{key} must be a string or a list of strings")
    env[key] = json.dumps(items, ensure_ascii=False)


def _put_json_object(env: dict[str, str], key: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ConfigError(f"{key} must be an object with string keys")
    env[key] = json.dumps(value, ensure_ascii=False)


def _model_tier(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in VALID_MODEL_TIERS:
        raise ConfigError("model_tier must be one of: high, medium, low")
    return text


def _validate_exact_model_provider(model: str | None, provider: str, field_name: str) -> None:
    if model is not None and provider == "auto":
        raise ConfigError(f"{field_name}.model requires explicit provider")


def _mcp_mode(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text not in {"user-config", "arc-only"}:
        raise ConfigError(f"{field_name} must be one of: user-config, arc-only")
    return text


def _codex_effort_for_model_tier(tier: str) -> str:
    return {"low": "low", "medium": "medium", "high": "xhigh"}[tier]


def _claude_effort_for_model_tier(tier: str) -> str:
    return {"low": "low", "medium": "medium", "high": "high"}[tier]
