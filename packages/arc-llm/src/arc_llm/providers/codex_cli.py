from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .base import LLMWorkerError


class CodexCliProvider:
    name = "codex-cli"

    def __init__(self, *, env: Mapping[str, str] | None = None) -> None:
        self.env = os.environ if env is None else env

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        schema = schema or {"type": "object"}
        with tempfile.TemporaryDirectory(prefix="arc-llm-") as tmp:
            tmpdir = Path(tmp)
            schema_path = tmpdir / "output.schema.json"
            output_path = tmpdir / "output.json"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

            cmd = [
                *_base_cmd(self.env),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
            ]
            if model:
                cmd.extend(["-m", model])
            cmd.append("-")

            try:
                result = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=dict(self.env),
                    timeout=_timeout_seconds(self.env, "ARC_CODEX_TIMEOUT_SECONDS"),
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMWorkerError(f"codex exec timed out after {exc.timeout} seconds") from exc
            if result.returncode != 0:
                raise LLMWorkerError(result.stderr or result.stdout or "codex exec failed")
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LLMWorkerError(f"Could not read Codex JSON output: {exc}") from exc
            if not isinstance(payload, dict):
                raise LLMWorkerError("Codex JSON output was not an object")
            return payload

    def generate_text(self, prompt: str, *, model: str | None = None) -> str:
        with tempfile.TemporaryDirectory(prefix="arc-llm-") as tmp:
            output_path = Path(tmp) / "output.txt"
            cmd = [
                *_base_cmd(self.env),
                "--output-last-message",
                str(output_path),
            ]
            if model:
                cmd.extend(["-m", model])
            cmd.append("-")

            try:
                result = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=dict(self.env),
                    timeout=_timeout_seconds(self.env, "ARC_CODEX_TIMEOUT_SECONDS"),
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMWorkerError(f"codex exec timed out after {exc.timeout} seconds") from exc
            if result.returncode != 0:
                raise LLMWorkerError(result.stderr or result.stdout or "codex exec failed")
            try:
                return output_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise LLMWorkerError(f"Could not read Codex text output: {exc}") from exc


def _base_cmd(env: Mapping[str, str]) -> list[str]:
    profile = _env_text(env, "ARC_CODEX_PROFILE", "")
    profile_v2 = _env_text(env, "ARC_CODEX_PROFILE_V2", "")
    enable_mcp = _env_bool(env, "ARC_CODEX_ENABLE_MCP", False)
    mcp_mode = _env_text(env, "ARC_CODEX_MCP_MODE", "user-config" if enable_mcp else "")
    if enable_mcp and mcp_mode not in {"user-config", "arc-only"}:
        raise LLMWorkerError("ARC_CODEX_MCP_MODE must be one of: user-config, arc-only")
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--color",
        _env_text(env, "ARC_CODEX_COLOR", "never"),
        "--sandbox",
        _env_text(env, "ARC_CODEX_SANDBOX", "read-only"),
    ]
    if work_dir := _env_text(env, "ARC_CODEX_WORK_DIR", ""):
        cmd.extend(["--cd", work_dir])
    for add_dir in _env_list(env, "ARC_CODEX_ADD_DIRS"):
        cmd.extend(["--add-dir", add_dir])
    if profile:
        cmd.extend(["--profile", profile])
    if profile_v2:
        cmd.extend(["--profile-v2", profile_v2])
    if _env_bool(env, "ARC_CODEX_EPHEMERAL", True):
        cmd.append("--ephemeral")
    ignore_user_config_default = not (enable_mcp or profile or profile_v2)
    if enable_mcp and mcp_mode == "arc-only":
        ignore_user_config_default = True
    if _env_bool(env, "ARC_CODEX_IGNORE_USER_CONFIG", ignore_user_config_default):
        cmd.append("--ignore-user-config")
    if _env_bool(env, "ARC_CODEX_IGNORE_RULES", True):
        cmd.append("--ignore-rules")

    for key, value in _codex_config_overrides(env):
        cmd.extend(["-c", f"{key}={value}"])
    for key, value in _arc_only_mcp_config_overrides(env):
        cmd.extend(["-c", f"{key}={value}"])
    for override in _extra_config_overrides(env):
        cmd.extend(["-c", override])
    return cmd


def _codex_config_overrides(env: Mapping[str, str]) -> list[tuple[str, str]]:
    allow_internet = _env_bool(env, "ARC_CODEX_ALLOW_INTERNET", False)
    overrides = [
        ("model_reasoning_effort", _toml_string(_env_text(env, "ARC_CODEX_REASONING_EFFORT", "low"))),
        ("model_reasoning_summary", _toml_string(_env_text(env, "ARC_CODEX_REASONING_SUMMARY", "none"))),
        ("model_verbosity", _toml_string(_env_text(env, "ARC_CODEX_MODEL_VERBOSITY", "low"))),
        ("hide_agent_reasoning", _toml_bool(_env_bool(env, "ARC_CODEX_HIDE_AGENT_REASONING", True))),
        ("history.persistence", _toml_string(_env_text(env, "ARC_CODEX_HISTORY_PERSISTENCE", "none"))),
        ("web_search", _toml_string(_env_text(env, "ARC_CODEX_WEB_SEARCH", "live" if allow_internet else "disabled"))),
    ]
    if env.get("ARC_CODEX_NETWORK_ACCESS") is not None or allow_internet:
        overrides.append(
            (
                "sandbox_workspace_write.network_access",
                _toml_bool(_env_bool(env, "ARC_CODEX_NETWORK_ACCESS", allow_internet)),
            )
        )
    return [(key, value) for key, value in overrides if value]


def _arc_only_mcp_config_overrides(env: Mapping[str, str]) -> list[tuple[str, str]]:
    if not _env_bool(env, "ARC_CODEX_ENABLE_MCP", False):
        return []
    if _env_text(env, "ARC_CODEX_MCP_MODE", "user-config") != "arc-only":
        return []

    mcp_env = {
        "ARC_AGENT_HOST": "codex",
    }
    for key in ("ARC_PAPER_CACHE", "ARC_DOMAIN_CACHE", "ARC_MCP_CACHE"):
        if value := env.get(key):
            mcp_env[key] = value
    if raw := env.get("ARC_CODEX_ARC_MCP_ENV_JSON"):
        try:
            extra_env = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMWorkerError(f"ARC_CODEX_ARC_MCP_ENV_JSON was not valid JSON: {exc}") from exc
        if not isinstance(extra_env, dict) or not all(isinstance(key, str) for key in extra_env):
            raise LLMWorkerError("ARC_CODEX_ARC_MCP_ENV_JSON must be a JSON object with string keys")
        mcp_env.update({key: str(value) for key, value in extra_env.items()})

    overrides = [
        ("mcp_servers.arc.command", _toml_string(_arc_mcp_command(env))),
        ("mcp_servers.arc.default_tools_approval_mode", _toml_string("approve")),
    ]
    overrides.extend((f"mcp_servers.arc.env.{key}", _toml_string(value)) for key, value in sorted(mcp_env.items()))
    return overrides


def _arc_mcp_command(env: Mapping[str, str]) -> str:
    if command := _env_text(env, "ARC_CODEX_ARC_MCP_COMMAND", ""):
        return command
    if command := shutil.which("arc-mcp"):
        return command
    sibling = Path(sys.executable).with_name("arc-mcp")
    if sibling.exists():
        return str(sibling)
    return "arc-mcp"


def _extra_config_overrides(env: Mapping[str, str]) -> list[str]:
    values = []
    if raw_json := env.get("ARC_CODEX_CONFIG_JSON"):
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise LLMWorkerError(f"ARC_CODEX_CONFIG_JSON was not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMWorkerError("ARC_CODEX_CONFIG_JSON must be a JSON object")
        values.extend(f"{key}={_toml_value(value)}" for key, value in payload.items())
    if raw := env.get("ARC_CODEX_CONFIG"):
        values.extend(line.strip() for line in raw.splitlines() if line.strip())
    return values


def _env_text(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key)
    if value is None:
        return default
    return value.strip()


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _timeout_seconds(env: Mapping[str, str], provider_key: str) -> float | None:
    key = provider_key if env.get(provider_key) not in {None, ""} else "ARC_LLM_TIMEOUT_SECONDS"
    value = env.get(key)
    if value is None or not value.strip():
        return None
    try:
        timeout = float(value)
    except ValueError as exc:
        raise LLMWorkerError(f"{key} must be a positive number") from exc
    if timeout <= 0:
        raise LLMWorkerError(f"{key} must be a positive number")
    return timeout


def _env_list(env: Mapping[str, str], key: str) -> list[str]:
    value = env.get(key)
    if value is None or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    raise LLMWorkerError(f"{key} must be a JSON string, JSON array of strings, or newline-separated strings")


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_string(value: str) -> str:
    if not value:
        return ""
    return json.dumps(value)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return _toml_bool(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _toml_string(value)
    return json.dumps(value)
