from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


IDEAS_CONFIG_SCHEMA = "arc.workflow.ideas.config.v1"
IDEAS_VARIANT_SCHEMA = "arc.workflow.ideas.variant.v1"
DOMAIN_MANIFEST_SCHEMA = "arc.workflow.domain_manifest.v1"
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RESEARCH_SCOPES = {"single_domain", "cross_domain"}


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
    reviewer_template: Path
    reviewer_output_schema: Path
    marking_scheme: Path
    research_scope: str
    context_policy: ContextPolicy
    proposer_overrides: dict[str, Any]
    description: str


@dataclass(frozen=True)
class IdeasConfig:
    schema_version: str
    run_id: str
    run_dir: Path
    project_dir: Path
    user_intent: str
    variant_config_dir: Path
    variant_glob: str
    loops_per_variant: int
    save_prompts: bool
    variants: list[VariantConfig]
    domain_manifest_path: Path
    domain_manifest: dict[str, Any] | None
    research_scope: str
    exploration_profiles: list[dict[str, str]]
    routing_warnings: list[str]


def load_ideas_config(payload: Mapping[str, Any]) -> IdeasConfig:
    data = copy.deepcopy(dict(payload))
    schema_version = _required_text(data, "schema_version")
    if schema_version != IDEAS_CONFIG_SCHEMA:
        raise ConfigError(f"schema_version must be {IDEAS_CONFIG_SCHEMA}")

    run_id = _safe_id(_required_text(data, "run_id"), "run_id")
    run_dir = Path(_required_text(data, "run_dir")).expanduser()
    project_dir = Path(_required_text(data, "project_dir")).expanduser()
    user_intent = _required_text(data, "user_intent")
    variant_config_dir = Path(_required_text(data, "variant_config_dir")).expanduser()
    _validate_strict_variant_config_dir(variant_config_dir)
    variant_glob = str(data.get("variant_glob", "ideas-*.variant.json") or "").strip()
    if not variant_glob:
        raise ConfigError("variant_glob is required")
    loops_per_variant = _positive_int(data.get("loops_per_variant", 5), "loops_per_variant")
    artifact_options = _dict(data.get("artifact_options", {}), "artifact_options")
    domain_manifest_path, manifest_was_explicit = _configured_manifest_path(data, project_dir=project_dir)
    domain_manifest, research_scope, routing_warnings = _load_domain_manifest(
        domain_manifest_path,
        required=manifest_was_explicit,
    )
    variants = [
        variant
        for variant in _discover_variants(variant_config_dir, variant_glob)
        if variant.research_scope == research_scope
    ]
    if not variants:
        raise ConfigError(
            f"No enabled {research_scope} ideas variants found in {variant_config_dir} with {variant_glob}"
        )
    exploration_profiles = _exploration_profiles(data.get("exploration_profiles"))
    if research_scope == "cross_domain":
        if exploration_profiles and len(exploration_profiles) != loops_per_variant:
            raise ConfigError("exploration_profiles must contain exactly one profile per cross-domain loop")
        if not exploration_profiles and loops_per_variant != 5:
            raise ConfigError(
                "cross-domain ideas use five default exploration profiles; provide exactly loops_per_variant "
                "exploration_profiles when loops_per_variant is not 5"
            )

    return IdeasConfig(
        schema_version=schema_version,
        run_id=run_id,
        run_dir=run_dir,
        project_dir=project_dir,
        user_intent=user_intent,
        variant_config_dir=variant_config_dir,
        variant_glob=variant_glob,
        loops_per_variant=loops_per_variant,
        save_prompts=_bool(artifact_options.get("save_prompts", True), "artifact_options.save_prompts"),
        variants=variants,
        domain_manifest_path=domain_manifest_path,
        domain_manifest=domain_manifest,
        research_scope=research_scope,
        exploration_profiles=exploration_profiles,
        routing_warnings=routing_warnings,
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
    if schema_version != IDEAS_VARIANT_SCHEMA:
        raise ConfigError(f"{path}.schema_version must be {IDEAS_VARIANT_SCHEMA}")
    if payload.get("enabled", True) is False:
        return None
    variant_id = _safe_id(_variant_required_text(payload, "variant_id", path), f"{path}.variant_id")
    base = path.parent
    loop_template = _relative_path(base, _variant_required_text(payload, "loop_template", path))
    proposer_template = _relative_path(base, _variant_required_text(payload, "proposer_template", path))
    reviewer_template = _relative_path(base, str(payload.get("reviewer_template", "ideas-reviewer.template.json")))
    reviewer_output_schema = _relative_path(
        base,
        str(payload.get("reviewer_output_schema", "ideas-reviewer-output.schema.json")),
    )
    marking_scheme = _relative_path(base, str(payload.get("marking_scheme", "ideas-marking-scheme.json")))
    research_scope = str(payload.get("research_scope", "single_domain")).strip()
    if research_scope not in RESEARCH_SCOPES:
        raise ConfigError(f"{path}.research_scope must be one of {sorted(RESEARCH_SCOPES)}")
    if not loop_template.exists():
        raise ConfigError(f"loop_template does not exist: {loop_template}")
    if not proposer_template.exists():
        raise ConfigError(f"proposer_template does not exist: {proposer_template}")
    if not reviewer_template.exists():
        raise ConfigError(f"reviewer_template does not exist: {reviewer_template}")
    if not reviewer_output_schema.exists():
        raise ConfigError(f"reviewer_output_schema does not exist: {reviewer_output_schema}")
    if not marking_scheme.exists():
        raise ConfigError(f"marking_scheme does not exist: {marking_scheme}")
    return VariantConfig(
        variant_id=variant_id,
        path=path,
        loop_template=loop_template,
        proposer_template=proposer_template,
        reviewer_template=reviewer_template,
        reviewer_output_schema=reviewer_output_schema,
        marking_scheme=marking_scheme,
        research_scope=research_scope,
        context_policy=_parse_context_policy(payload.get("context_policy", {}), path=path),
        proposer_overrides=_dict(payload.get("proposer", {}), f"{path}.proposer"),
        description=str(payload.get("description", "")),
    )


def _configured_manifest_path(data: Mapping[str, Any], *, project_dir: Path) -> tuple[Path, bool]:
    raw = str(data.get("domain_manifest_path", "") or "").strip()
    if not raw:
        return project_dir / "domain" / "domain-manifest.json", False
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else project_dir / path), True


def _validate_strict_variant_config_dir(path: Path) -> None:
    required_root = str(os.environ.get("ARC_REQUIRE_REPO_ROOT", "")).strip()
    if not required_root:
        return
    root = Path(required_root).expanduser().resolve()
    expected = (root / "plugins/arc/skills/arc/workflows/json").resolve()
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"strict ARC source mode cannot resolve variant_config_dir: {path}") from exc
    if resolved != expected:
        raise ConfigError(
            "strict ARC source mode requires variant_config_dir from the required checkout: "
            f"expected {expected}, got {resolved}"
        )


def _load_domain_manifest(path: Path, *, required: bool) -> tuple[dict[str, Any] | None, str, list[str]]:
    if not path.is_file():
        if required:
            raise ConfigError(f"domain_manifest_path does not exist: {path}")
        return (
            None,
            "single_domain",
            [
                "domain_manifest_unavailable: Domain manifest was unavailable; "
                "using the legacy single-domain ideas path."
            ],
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not read domain manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"domain manifest must be an object: {path}")
    if payload.get("schema_version") != DOMAIN_MANIFEST_SCHEMA:
        raise ConfigError(f"{path}.schema_version must be {DOMAIN_MANIFEST_SCHEMA}")
    domains = payload.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ConfigError(f"{path}.domains must be a non-empty array")
    domain_ids: set[str] = set()
    for index, domain in enumerate(domains):
        if not isinstance(domain, dict):
            raise ConfigError(f"{path}.domains[{index}] must be an object")
        domain_id = str(domain.get("domain_id", "")).strip()
        if not domain_id:
            raise ConfigError(f"{path}.domains[{index}].domain_id is required")
        domain_ids.add(domain_id)
    return payload, "cross_domain" if len(domain_ids) >= 2 else "single_domain", []


def _exploration_profiles(raw: Any) -> list[dict[str, str]]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ConfigError("exploration_profiles must be a non-empty array")
    profiles: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"exploration_profiles[{index}] must be an object")
        profile_id = _safe_id(str(item.get("profile_id", "")).strip(), f"exploration_profiles[{index}].profile_id")
        mission = str(item.get("mission", "")).strip()
        if not mission:
            raise ConfigError(f"exploration_profiles[{index}].mission is required")
        if profile_id in seen:
            raise ConfigError(f"exploration_profiles contains duplicate profile_id: {profile_id}")
        seen.add(profile_id)
        profiles.append({"profile_id": profile_id, "mission": mission})
    return profiles


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
