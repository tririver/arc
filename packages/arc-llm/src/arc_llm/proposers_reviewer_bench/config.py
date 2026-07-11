from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from arc_llm.model import VALID_MODEL_TIERS
from arc_llm.proposers_reviewer.config import BATCH_CONFIG_SCHEMA, ConfigError, load_batch_config


BENCH_CONFIG_SCHEMA = "arc.llm.proposers_reviewer_bench.config.v1"
IMPROVEMENT_SCHEMA = "arc.llm.proposers_reviewer_bench.improvement.v1"
SUGGESTED_IMPROVEMENT_FIELD = "suggested_improvement"
SUGGESTED_IMPROVEMENT_PROMPT = (
    "Benchmark prompt-improvement note: In your output JSON, add a top-level "
    "`suggested_improvement` object. Separate reusable workflow/prompt "
    "improvements from domain-specific advice. Reusable prompt suggestions "
    "should transfer across research domains. Domain-specific technical "
    "suggestions should be framed as advice for this proposer/reviewer exchange, "
    "not as global prompt-template edits."
)


@dataclass(frozen=True)
class BenchOptions:
    samples: int = 10
    sample_loop_id_prefix: str = "idea"
    sample_loop_id_start: int = 1
    max_rounds: int = 5
    max_iterations: int = 10
    patience: int = 3
    max_concurrent_loops: int = 100
    default_provider: str = "auto"
    sample_model_tier: str | None = "medium"
    improver_provider: str = "auto"
    improver_model: str | None = None
    improver_model_tier: str | None = "high"
    score_path: str = "review_payload.marks.total_score"
    min_delta: float = 0.15
    min_z: float = 0.5
    allow_reviewer_prompt_edits: bool = False
    improver_context_mode: str = "auto"
    improver_context_max_chars: int = 600_000


@dataclass(frozen=True)
class BenchConfig:
    batch_payload: dict[str, Any]
    options: BenchOptions


def load_bench_config(payload: Mapping[str, Any]) -> BenchConfig:
    data = copy.deepcopy(dict(payload))
    schema_version = str(data.get("schema_version", "")).strip()
    if schema_version not in {BATCH_CONFIG_SCHEMA, BENCH_CONFIG_SCHEMA}:
        raise ConfigError(f"schema_version must be {BATCH_CONFIG_SCHEMA} or {BENCH_CONFIG_SCHEMA}")

    raw_options = _dict(data.pop("bench", {}), "bench")
    data["schema_version"] = BATCH_CONFIG_SCHEMA
    load_batch_config(data)
    raw_loops = data.get("loops")
    if isinstance(raw_loops, list) and len(raw_loops) != 1:
        raise ConfigError(f"benchmark configs support exactly one loop template; found {len(raw_loops)}")
    return BenchConfig(batch_payload=data, options=_parse_options(raw_options))


def materialize_batch_payload(
    config: BenchConfig,
    *,
    iteration_index: int,
    candidate_id: str,
    base_payload_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if iteration_index < 0:
        raise ConfigError("iteration_index must be non-negative")
    if not candidate_id:
        raise ConfigError("candidate_id is required")

    options = config.options
    source = copy.deepcopy(dict(base_payload_override or config.batch_payload))
    source["schema_version"] = BATCH_CONFIG_SCHEMA
    source["run_id"] = f"{config.batch_payload['run_id']}_iter{iteration_index:03d}_{candidate_id}"
    source["max_concurrent_loops"] = options.max_concurrent_loops
    source["session"] = {
        "policy": "stateful",
        "history_mode": "delta",
        "scope_id": f"bench/{config.batch_payload['run_id']}/{candidate_id}",
        "reuse_across_batch_calls": False,
        "max_concurrent_same_prefix": 12,
    }
    source["defaults"] = _defaults_with_provider_and_model_tier(
        source.get("defaults"),
        default_provider=options.default_provider,
        sample_model_tier=options.sample_model_tier,
    )

    raw_loops = source.get("loops")
    if not isinstance(raw_loops, list) or not raw_loops:
        raise ConfigError("loops must be a non-empty list")
    template_loop = raw_loops[0]
    loops: list[dict[str, Any]] = []
    for index in range(options.samples):
        loop = copy.deepcopy(template_loop)
        loop["loop_id"] = f"{options.sample_loop_id_prefix}_{options.sample_loop_id_start + index:03d}"
        loop["max_rounds"] = options.max_rounds
        _apply_sample_model_tier(loop, options.sample_model_tier)
        _add_suggested_improvement_hint(loop)
        loops.append(loop)
    source["loops"] = loops
    load_batch_config(source)
    return source


def apply_improvement_edits(
    payload: Mapping[str, Any],
    improvement: Mapping[str, Any],
    *,
    allow_reviewer_prompt_edits: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if improvement.get("schema_version") != IMPROVEMENT_SCHEMA:
        raise ConfigError(f"improvement.schema_version must be {IMPROVEMENT_SCHEMA}")
    edits = improvement.get("edits", [])
    if not isinstance(edits, list):
        raise ConfigError("improvement.edits must be a list")

    updated = copy.deepcopy(dict(payload))
    applied: list[dict[str, Any]] = []
    for raw_edit in edits:
        edit = _dict(raw_edit, "improvement.edits[]")
        target = str(edit.get("target", ""))
        operation = str(edit.get("operation", ""))
        entry = {"target": target, "operation": operation}
        if target.startswith("reviewers.") and not allow_reviewer_prompt_edits:
            entry.update({"applied": False, "reason": "reviewer prompt edits are disabled"})
            applied.append(entry)
            continue
        if target not in {"proposers.*.prompt.template", "reviewers.*.prompt.template"}:
            entry.update({"applied": False, "reason": "unsupported target"})
            applied.append(entry)
            continue
        if operation not in {"append_paragraph", "replace"}:
            entry.update({"applied": False, "reason": "unsupported operation"})
            applied.append(entry)
            continue
        text = str(edit.get("text", ""))
        if not text.strip():
            entry.update({"applied": False, "reason": "text is required"})
            applied.append(entry)
            continue
        _apply_prompt_template_edit(updated, target=target, operation=operation, text=text)
        entry["applied"] = True
        applied.append(entry)
    load_batch_config(_as_batch_config_payload(updated))
    return updated, applied


def improvement_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "required": ["schema_version", "edits"],
        "properties": {
            "schema_version": {"const": IMPROVEMENT_SCHEMA},
            "rationale": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["target", "operation", "text"],
                    "properties": {
                        "target": {
                            "type": "string",
                            "enum": ["proposers.*.prompt.template", "reviewers.*.prompt.template"],
                        },
                        "operation": {"type": "string", "enum": ["append_paragraph", "replace"]},
                        "text": {"type": "string"},
                    },
                },
            },
        },
    }


def _add_suggested_improvement_hint(loop: dict[str, Any]) -> None:
    for worker_key in ("proposers", "reviewers"):
        workers = loop.get(worker_key, [])
        if not isinstance(workers, list):
            continue
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            _append_suggested_improvement_prompt(worker)
            _extend_output_schema(worker)


def _append_suggested_improvement_prompt(worker: dict[str, Any]) -> None:
    prompt = worker.setdefault("prompt", {})
    if not isinstance(prompt, dict):
        return
    template = str(prompt.get("template", ""))
    if SUGGESTED_IMPROVEMENT_FIELD in template:
        return
    prompt["template"] = f"{template.rstrip()}\n\n{SUGGESTED_IMPROVEMENT_PROMPT}" if template.strip() else SUGGESTED_IMPROVEMENT_PROMPT


def _extend_output_schema(worker: dict[str, Any]) -> None:
    schema = worker.get("output_schema")
    if not isinstance(schema, dict):
        return
    if schema.get("type") not in {None, "object"}:
        return
    schema.setdefault("type", "object")
    properties = schema.setdefault("properties", {})
    if not isinstance(properties, dict):
        return
    properties.setdefault(SUGGESTED_IMPROVEMENT_FIELD, _suggested_improvement_schema())


def _suggested_improvement_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "summary": {"type": "string"},
            "prompt": {"type": "string"},
            "context": {"type": "string"},
            "tooling": {"type": "string"},
            "scoring": {"type": "string"},
        },
    }


def _parse_options(raw: Mapping[str, Any]) -> BenchOptions:
    default_provider = _nonempty_text(raw.get("default_provider", "auto"), "bench.default_provider")
    improver_provider = _nonempty_text(raw.get("improver_provider", default_provider), "bench.improver_provider")
    improver_model = _optional_text(raw.get("improver_model"), "bench.improver_model")
    if improver_model is not None and improver_provider == "auto":
        raise ConfigError("bench.improver_model requires explicit provider")
    return BenchOptions(
        samples=_positive_int(raw.get("samples", 10), "bench.samples"),
        sample_loop_id_prefix=_nonempty_text(raw.get("sample_loop_id_prefix", "idea"), "bench.sample_loop_id_prefix"),
        sample_loop_id_start=_positive_int(raw.get("sample_loop_id_start", 1), "bench.sample_loop_id_start"),
        max_rounds=_positive_int(raw.get("max_rounds", 5), "bench.max_rounds"),
        max_iterations=_positive_int(raw.get("max_iterations", 10), "bench.max_iterations"),
        patience=_positive_int(raw.get("patience", 3), "bench.patience"),
        max_concurrent_loops=_positive_int(raw.get("max_concurrent_loops", 100), "bench.max_concurrent_loops"),
        default_provider=default_provider,
        sample_model_tier=_model_tier(raw.get("sample_model_tier", "medium"), "bench.sample_model_tier"),
        improver_provider=improver_provider,
        improver_model=improver_model,
        improver_model_tier=_model_tier(raw.get("improver_model_tier", "high"), "bench.improver_model_tier"),
        score_path=_nonempty_text(raw.get("score_path", "review_payload.marks.total_score"), "bench.score_path"),
        min_delta=_float(raw.get("min_delta", 0.15), "bench.min_delta"),
        min_z=_float(raw.get("min_z", 0.5), "bench.min_z"),
        allow_reviewer_prompt_edits=_bool(raw.get("allow_reviewer_prompt_edits", False), "bench.allow_reviewer_prompt_edits"),
        improver_context_mode=_context_mode(raw.get("improver_context_mode", "auto"), "bench.improver_context_mode"),
        improver_context_max_chars=_positive_int(
            raw.get("improver_context_max_chars", 600_000), "bench.improver_context_max_chars"
        ),
    )


def _defaults_with_provider_and_model_tier(
    raw_defaults: Any,
    *,
    default_provider: str,
    sample_model_tier: str | None,
) -> dict[str, Any]:
    defaults = _dict(raw_defaults, "defaults")
    provider = str(defaults.get("provider", "auto") or "auto")
    if provider == "auto":
        defaults["provider"] = default_provider
    if sample_model_tier and not defaults.get("model"):
        defaults["model_tier"] = sample_model_tier
    return defaults


def _apply_sample_model_tier(loop: dict[str, Any], sample_model_tier: str | None) -> None:
    if not sample_model_tier:
        return
    for worker_key in ("proposers", "reviewers"):
        workers = loop.get(worker_key, [])
        if not isinstance(workers, list):
            continue
        for worker in workers:
            if isinstance(worker, dict) and not worker.get("model"):
                worker["model_tier"] = sample_model_tier


def _apply_prompt_template_edit(payload: dict[str, Any], *, target: str, operation: str, text: str) -> None:
    worker_key = "proposers" if target.startswith("proposers.") else "reviewers"
    for loop in payload.get("loops", []):
        if not isinstance(loop, dict):
            continue
        workers = loop.get(worker_key, [])
        if not isinstance(workers, list):
            continue
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            prompt = worker.setdefault("prompt", {})
            if not isinstance(prompt, dict):
                continue
            if operation == "replace":
                prompt["template"] = text
            else:
                current = str(prompt.get("template", ""))
                prompt["template"] = f"{current.rstrip()}\n\n{text.strip()}" if current.strip() else text.strip()


def _as_batch_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(payload)
    data["schema_version"] = BATCH_CONFIG_SCHEMA
    return data


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


def _float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ConfigError(f"{field_name} must be a number") from exc


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


def _nonempty_text(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ConfigError(f"{field_name} is required")
    return text


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _model_tier(value: Any, field_name: str) -> str | None:
    text = _optional_text(value, field_name)
    if text is None:
        return None
    if text not in VALID_MODEL_TIERS:
        raise ConfigError(f"{field_name} must be one of: low, medium, high, xhigh")
    return text


def _context_mode(value: Any, field_name: str) -> str:
    text = str(value).strip().lower()
    if text not in {"auto", "paths", "expanded"}:
        raise ConfigError(f"{field_name} must be one of: auto, paths, expanded")
    return text
