from __future__ import annotations

import json
import os
import subprocess
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
        with tempfile.TemporaryDirectory(prefix="arc-llm-worker-") as tmp:
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

            result = subprocess.run(cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        with tempfile.TemporaryDirectory(prefix="arc-llm-worker-") as tmp:
            output_path = Path(tmp) / "output.txt"
            cmd = [
                *_base_cmd(self.env),
                "--output-last-message",
                str(output_path),
            ]
            if model:
                cmd.extend(["-m", model])
            cmd.append("-")

            result = subprocess.run(cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--color",
        _env_text(env, "ARC_CODEX_COLOR", "never"),
        "--sandbox",
        _env_text(env, "ARC_CODEX_SANDBOX", "read-only"),
    ]
    if profile:
        cmd.extend(["--profile", profile])
    if profile_v2:
        cmd.extend(["--profile-v2", profile_v2])
    if _env_bool(env, "ARC_CODEX_EPHEMERAL", True):
        cmd.append("--ephemeral")
    if _env_bool(env, "ARC_CODEX_IGNORE_USER_CONFIG", not (enable_mcp or profile or profile_v2)):
        cmd.append("--ignore-user-config")
    if _env_bool(env, "ARC_CODEX_IGNORE_RULES", True):
        cmd.append("--ignore-rules")

    for key, value in _codex_config_overrides(env):
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
