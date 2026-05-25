from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


RESEARCH_IDEAS_CONFIG_SCHEMA = "arc.workflow.research_ideas.config.v1"
RESEARCH_IDEAS_VARIANT_SCHEMA = "arc.workflow.research_ideas.variant.v1"
GLOBAL_REVIEW_SCHEMA = "arc.workflow.research_ideas.global_review.v1"
VALID_MODEL_TIERS = frozenset({"low", "medium", "high"})
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
GLOBAL_REVIEW_DEFAULT_TEMPLATE = (
    "Review all proposed ideas on one common scale against the full user intent. "
    "Use caller_context.marking_scheme as the sole source for mark names, ranges, "
    "and scoring guidance. Use the supplied proposal "
    "records, novelty-check notes, and included context. Do not rely on memory for "
    "novelty confidence; use supplied search evidence and any internet evidence available "
    "under the configured runtime. Rank all ideas by total_score and use the same scale "
    "across variants. Return exactly one "
    f"JSON object with schema_version {GLOBAL_REVIEW_SCHEMA}. Input:\n{{caller_context_json}}"
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ContextPolicy:
    require_domain_markdown: bool
    attach_domain_markdown: bool
    attach_arc_paper_tool_notes: bool


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    path: Path
    loop_template: Path
    proposer_template: Path
    context_policy: ContextPolicy
    proposer_overrides: dict[str, Any]
    description: str


@dataclass(frozen=True)
class ReviewerConfig:
    provider: str
    model: str | None
    model_tier: str | None
    allow_tools: bool
    runtime: dict[str, Any]
    system: str
    template: str


@dataclass(frozen=True)
class ResearchIdeasConfig:
    schema_version: str
    run_id: str
    run_dir: Path
    project_dir: Path
    user_intent: str
    variant_config_dir: Path
    variant_glob: str
    loops_per_variant: int
    existing_run_policy: str
    save_prompts: bool
    variants: list[VariantConfig]
    reviewer: ReviewerConfig


def load_research_ideas_config(payload: Mapping[str, Any]) -> ResearchIdeasConfig:
    data = copy.deepcopy(dict(payload))
    schema_version = _required_text(data, "schema_version")
    if schema_version != RESEARCH_IDEAS_CONFIG_SCHEMA:
        raise ConfigError(f"schema_version must be {RESEARCH_IDEAS_CONFIG_SCHEMA}")

    run_id = _safe_id(_required_text(data, "run_id"), "run_id")
    run_dir = Path(_required_text(data, "run_dir")).expanduser()
    project_dir = Path(_required_text(data, "project_dir")).expanduser()
    user_intent = _required_text(data, "user_intent")
    variant_config_dir = Path(_required_text(data, "variant_config_dir")).expanduser()
    variant_glob = str(data.get("variant_glob", "suggest-ideas-*.variant.json") or "").strip()
    if not variant_glob:
        raise ConfigError("variant_glob is required")
    loops_per_variant = _positive_int(data.get("loops_per_variant", 5), "loops_per_variant")
    existing_run_policy = str(data.get("existing_run_policy", "fail")).strip() or "fail"
    if existing_run_policy not in {"fail", "overwrite"}:
        raise ConfigError("existing_run_policy must be fail or overwrite")
    artifact_options = _dict(data.get("artifact_options", {}), "artifact_options")
    variants = _discover_variants(variant_config_dir, variant_glob)
    if not variants:
        raise ConfigError(f"No enabled research-ideas variants found in {variant_config_dir} with {variant_glob}")

    return ResearchIdeasConfig(
        schema_version=schema_version,
        run_id=run_id,
        run_dir=run_dir,
        project_dir=project_dir,
        user_intent=user_intent,
        variant_config_dir=variant_config_dir,
        variant_glob=variant_glob,
        loops_per_variant=loops_per_variant,
        existing_run_policy=existing_run_policy,
        save_prompts=bool(artifact_options.get("save_prompts", True)),
        variants=variants,
        reviewer=_parse_reviewer(data.get("reviewer", {})),
    )


def _discover_variants(root: Path, pattern: str) -> list[VariantConfig]:
    if not root.exists():
        raise ConfigError(f"variant_config_dir does not exist: {root}")
    variants: list[VariantConfig] = []
    for path in sorted(root.glob(pattern)):
        if "_inactivated" in path.name or ".disabled." in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ConfigError(f"variant config must be an object: {path}")
        variant = _parse_variant(payload, path=path)
        if variant is not None:
            variants.append(variant)
    return variants


def _parse_variant(payload: Mapping[str, Any], *, path: Path) -> VariantConfig | None:
    schema_version = str(payload.get("schema_version", "")).strip()
    if not schema_version:
        raise ConfigError(f"{path}.schema_version is required")
    if schema_version != RESEARCH_IDEAS_VARIANT_SCHEMA:
        raise ConfigError(f"{path}.schema_version must be {RESEARCH_IDEAS_VARIANT_SCHEMA}")
    if payload.get("enabled", True) is False:
        return None
    variant_id = _safe_id(_variant_required_text(payload, "variant_id", path), f"{path}.variant_id")
    base = path.parent
    loop_template = _relative_path(base, _variant_required_text(payload, "loop_template", path))
    proposer_template = _relative_path(base, _variant_required_text(payload, "proposer_template", path))
    if not loop_template.exists():
        raise ConfigError(f"loop_template does not exist: {loop_template}")
    if not proposer_template.exists():
        raise ConfigError(f"proposer_template does not exist: {proposer_template}")
    return VariantConfig(
        variant_id=variant_id,
        path=path,
        loop_template=loop_template,
        proposer_template=proposer_template,
        context_policy=_parse_context_policy(payload.get("context_policy", {}), path=path),
        proposer_overrides=_dict(payload.get("proposer", {}), f"{path}.proposer"),
        description=str(payload.get("description", "")),
    )


def _parse_context_policy(raw: Any, *, path: Path) -> ContextPolicy:
    data = _dict(raw, f"{path}.context_policy")
    attach_domain = _bool(data.get("attach_domain_markdown", False), f"{path}.context_policy.attach_domain_markdown")
    return ContextPolicy(
        require_domain_markdown=_bool(
            data.get("require_domain_markdown", attach_domain),
            f"{path}.context_policy.require_domain_markdown",
        ),
        attach_domain_markdown=attach_domain,
        attach_arc_paper_tool_notes=_bool(
            data.get("attach_arc_paper_tool_notes", attach_domain),
            f"{path}.context_policy.attach_arc_paper_tool_notes",
        ),
    )


def _parse_reviewer(raw: Any) -> ReviewerConfig:
    data = _dict(raw, "reviewer")
    allow_tools = _bool(data.get("allow_tools", False), "reviewer.allow_tools")
    runtime = _dict(data.get("runtime", {}), "reviewer.runtime")
    if not allow_tools:
        runtime["allow_mcp"] = False
    prompt = _dict(data.get("prompt", {}), "reviewer.prompt")
    return ReviewerConfig(
        provider=str(data.get("provider", "deepseek") or "deepseek"),
        model=_optional_text(data.get("model"), "reviewer.model"),
        model_tier=_model_tier(data.get("model_tier", "high"), "reviewer.model_tier"),
        allow_tools=allow_tools,
        runtime=runtime,
        system=str(
            prompt.get(
                "system",
                "You are a skeptical but constructive theoretical-physics reviewer comparing ARC research ideas.",
            )
        ),
        template=str(
            prompt.get(
                "template",
                GLOBAL_REVIEW_DEFAULT_TEMPLATE,
            )
        ),
    )


def _relative_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base / path


def _required_text(data: Mapping[str, Any], field_name: str) -> str:
    text = str(data.get(field_name, "")).strip()
    if not text:
        raise ConfigError(f"{field_name} is required")
    return text


def _variant_required_text(data: Mapping[str, Any], key: str, path: Path) -> str:
    text = str(data.get(key, "")).strip()
    if not text:
        raise ConfigError(f"{path}.{key} is required")
    return text


def _safe_id(value: str, field_name: str) -> str:
    if not SAFE_ID_RE.match(value):
        raise ConfigError(f"{field_name} must match {SAFE_ID_RE.pattern}")
    return value


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
        raise ConfigError(f"{field_name} must be one of: high, medium, low")
    return text
