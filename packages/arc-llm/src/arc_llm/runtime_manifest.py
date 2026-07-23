from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from .schema_cache import canonical_json, sha256_text
from .nested_shell_capability import capability_runtime_identity
from .paper_access_policy import resolve_arc_paper_access


RUNTIME_MANIFEST_VERSION = "arc.llm.runtime_manifest.v2"

_SHARED_RECIPE_KEYS = {
    "ARC_PAPER_ACCESS",
    "ARC_PAPER_DIRECT_SHELL",
    "ARC_LLM_INHERIT_HOST_TOOLS",
    "ARC_LLM_HOST_TOOLS_RISK",
    "ARC_LLM_WORKER_CONTEXT",
    "ARC_PAPER_CACHE",
    "ARC_PAPER_WORKER_BASE_CACHE",
    "ARC_PAPER_WORKER_SESSION_DIR",
    "ARC_PAPER_WORKER_TOMBSTONE_DIR",
    "ARC_PAPER_WORKER_SESSION_ID",
    "ARC_PAPER_WORKER_ALLOWED_OPERATIONS_JSON",
    "ARC_PAPER_WORKER_ALLOWED_TARGETS_JSON",
    "ARC_PAPER_WORKER_READ_POLICY_SCHEMA",
    "ARC_PAPER_WORKER_READ_POLICY_SHA256",
}

_PROVIDER_RECIPE_KEYS = {
    "codex-cli": {
        "ARC_CODEX_SANDBOX", "ARC_CODEX_CONFIG", "ARC_CODEX_CONFIG_JSON",
        "ARC_CODEX_HISTORY_PERSISTENCE", "ARC_CODEX_EPHEMERAL",
        "ARC_CODEX_WORK_DIR", "ARC_CODEX_ADD_DIRS", "ARC_CODEX_PROFILE",
        "ARC_CODEX_PROFILE_V2", "ARC_CODEX_ENABLE_MCP", "ARC_CODEX_MCP_MODE",
        "ARC_CODEX_ARC_MCP_COMMAND", "ARC_CODEX_ARC_MCP_ENV_JSON",
        "ARC_CODEX_ALLOW_INTERNET", "ARC_CODEX_NETWORK_ACCESS",
        "ARC_CODEX_WEB_SEARCH", "ARC_CODEX_REASONING_EFFORT",
        "ARC_CODEX_REASONING_SUMMARY", "ARC_CODEX_MODEL_VERBOSITY",
        "ARC_CODEX_IGNORE_USER_CONFIG", "ARC_CODEX_IGNORE_RULES",
    },
    "claude-cli": {
        "ARC_CLAUDE_TOOLS", "ARC_CLAUDE_ALLOWED_TOOLS", "ARC_CLAUDE_ALLOW_MCP",
        "ARC_CLAUDE_MCP_MODE", "ARC_CLAUDE_MCP_CONFIG",
        "ARC_CLAUDE_MCP_CONFIG_JSON", "ARC_CLAUDE_STRICT_MCP_CONFIG",
        "ARC_CLAUDE_ARC_MCP_COMMAND", "ARC_CLAUDE_ARC_MCP_ARGS_JSON",
        "ARC_CLAUDE_ARC_MCP_ENV_JSON", "ARC_CLAUDE_ARC_MCP_CONFIG_PATH",
        "ARC_CLAUDE_TEXT_OUTPUT_FORMAT_JSON", "ARC_CLAUDE_EFFORT",
        "ARC_CLAUDE_BARE", "ARC_CLAUDE_EXCLUDE_DYNAMIC_SYSTEM_PROMPT_SECTIONS",
        "ARC_CLAUDE_ALLOW_INTERNET",
    },
}

_BOOLEAN_KEYS = {
    "ARC_CODEX_EPHEMERAL", "ARC_CODEX_ENABLE_MCP", "ARC_CODEX_ALLOW_INTERNET",
    "ARC_CODEX_NETWORK_ACCESS", "ARC_CODEX_WEB_SEARCH",
    "ARC_CODEX_IGNORE_USER_CONFIG", "ARC_CODEX_IGNORE_RULES",
    "ARC_CLAUDE_ALLOW_MCP", "ARC_CLAUDE_STRICT_MCP_CONFIG", "ARC_CLAUDE_BARE",
    "ARC_CLAUDE_EXCLUDE_DYNAMIC_SYSTEM_PROMPT_SECTIONS", "ARC_CLAUDE_ALLOW_INTERNET",
    "ARC_LLM_INHERIT_HOST_TOOLS", "ARC_LLM_WORKER_CONTEXT",
    "ARC_PAPER_DIRECT_SHELL",
}


def runtime_manifest(
    *,
    provider: str,
    model: str | None,
    model_tier: str | None,
    env: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Return the normalized recipe that may affect this provider request.

    Host discovery and audit-only values (for example ``process_chain``) are
    deliberately outside this manifest.  Callers may record them separately,
    but they do not change native-session compatibility.
    """

    source = os.environ if env is None and provider == "kimi-code-cli" else (env or {})
    access = resolve_arc_paper_access(env=source)
    keys = _PROVIDER_RECIPE_KEYS.get(provider, set()) | _SHARED_RECIPE_KEYS
    recipe_env = {
        key: normalized
        for key in sorted(keys)
        if (normalized := _normalize_env_value(key, source.get(key))) is not None
    }
    recipe_env["ARC_PAPER_ACCESS"] = access.access
    if access.access == "none":
        for key in (
            "ARC_LLM_WORKER_CONTEXT", "ARC_PAPER_CACHE", "ARC_PAPER_WORKER_BASE_CACHE",
            "ARC_PAPER_WORKER_SESSION_DIR", "ARC_PAPER_WORKER_TOMBSTONE_DIR",
            "ARC_PAPER_WORKER_SESSION_ID", "ARC_PAPER_WORKER_READ_POLICY_SCHEMA",
            "ARC_PAPER_WORKER_READ_POLICY_SHA256",
        ):
            recipe_env.pop(key, None)

    recipe: dict[str, Any] = {
        "env": recipe_env,
        "capabilities": capability_runtime_identity(source),
    }
    file_hashes = _provider_file_hashes(provider, source)
    if file_hashes:
        recipe["file_hashes"] = file_hashes
    if provider == "kimi-code-cli":
        recipe["provider_runtime"] = _kimi_recipe(source, model_tier=model_tier)
    return {
        "schema_version": RUNTIME_MANIFEST_VERSION,
        "provider": provider,
        "model": model,
        "request_recipe": recipe,
    }


def runtime_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(dict(manifest)))


def _normalize_env_value(key: str, value: str | None) -> Any:
    if key in _BOOLEAN_KEYS:
        normalized = str(value or "").strip().lower()
        if normalized in {"", "0", "false", "no", "off", "none"}:
            return None
        if normalized in {"1", "true", "yes", "on"}:
            return True
        return normalized
    if value is None or not str(value).strip():
        return None
    if key == "ARC_LLM_HOST_TOOLS_RISK" and str(value).strip().lower() == "none":
        return None
    return str(value).strip()


def _kimi_recipe(env: Mapping[str, str], *, model_tier: str | None) -> dict[str, Any]:
    binary = (env.get("ARC_KIMI_BIN") or "kimi").strip() or "kimi"
    provider_timeout = (env.get("ARC_KIMI_IDLE_TIMEOUT_SECONDS") or "").strip()
    timeout = provider_timeout or (env.get("ARC_LLM_IDLE_TIMEOUT_SECONDS") or "").strip() or None
    tier = (model_tier or "").strip().lower()
    selected_mapping = env.get(f"ARC_LLM_KIMI_{tier.upper()}_MODEL") if tier else None
    return {
        "binary": binary,
        "resolved_binary": shutil.which(binary, path=env.get("PATH", os.defpath)),
        "work_dir": str(Path(env.get("ARC_KIMI_WORK_DIR") or os.getcwd()).expanduser().resolve(strict=False)),
        "kimi_code_home": env.get("KIMI_CODE_HOME") or None,
        "selected_tier_mapping": selected_mapping or None,
        "idle_timeout_seconds": timeout,
    }


def _provider_file_hashes(provider: str, env: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if provider == "claude-cli":
        paths = _newline_paths(env.get("ARC_CLAUDE_MCP_CONFIG"))
        if paths:
            result["ARC_CLAUDE_MCP_CONFIG"] = [_file_hash(path) for path in paths]
        json_paths = _json_paths(env.get("ARC_CLAUDE_MCP_CONFIG_JSON"))
        if json_paths:
            result["ARC_CLAUDE_MCP_CONFIG_JSON"] = [_file_hash(path) for path in json_paths]
    if _normalize_env_value("ARC_LLM_INHERIT_HOST_TOOLS", env.get("ARC_LLM_INHERIT_HOST_TOOLS")):
        paths: list[Path] = []
        if provider == "codex-cli":
            paths.append(Path(env.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml")
        elif provider == "claude-cli":
            paths.append(Path(env.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "settings.json")
        existing = [path for path in paths if path.is_file()]
        if existing:
            result["inherited_host_config"] = [_file_hash(str(path)) for path in existing]
    return result


def _newline_paths(value: str | None) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def _json_paths(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "")
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, str):
        return [parsed] if parsed.strip() else []
    return [item for item in parsed if isinstance(item, str) and item.strip()] if isinstance(parsed, list) else []


def _file_hash(path_text: str) -> dict[str, str | None]:
    path = Path(path_text).expanduser()
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"path": str(path), "sha256": None, "error": type(exc).__name__}
    return {"path": str(path), "sha256": sha256_text(content), "error": None}
