from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Mapping

from .base import LLMWorkerError


class ClaudeCliProvider:
    name = "claude-cli"

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
        cmd = [
            *_base_cmd(self.env),
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, ensure_ascii=False),
        ]
        if model:
            cmd.extend(["--model", model])

        result = subprocess.run(cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise LLMWorkerError(result.stderr or result.stdout or "claude -p failed")
        return _extract_json(result.stdout)

    def generate_text(self, prompt: str, *, model: str | None = None) -> str:
        cmd = _base_cmd(self.env)
        if model:
            cmd.extend(["--model", model])

        result = subprocess.run(cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise LLMWorkerError(result.stderr or result.stdout or "claude -p failed")
        return result.stdout


def _extract_json(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LLMWorkerError(f"Claude output was not JSON: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("result"), str):
        try:
            nested = json.loads(payload["result"])
        except json.JSONDecodeError as exc:
            raise LLMWorkerError(f"Claude result field was not JSON: {exc}") from exc
        if not isinstance(nested, dict):
            raise LLMWorkerError("Claude result JSON was not an object")
        return nested
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        return payload["result"]
    if isinstance(payload, dict):
        return payload
    raise LLMWorkerError("Claude JSON output was not an object")


def _base_cmd(env: Mapping[str, str]) -> list[str]:
    cmd = ["claude", "-p"]
    mcp_configs = _mcp_configs(env)
    allow_mcp = _env_bool(env, "ARC_CLAUDE_ALLOW_MCP", bool(mcp_configs))
    if _env_bool(env, "ARC_CLAUDE_BARE", not allow_mcp or bool(mcp_configs)):
        cmd.append("--bare")
    tools = _claude_tools(env, allow_mcp=allow_mcp)
    if tools is not None:
        cmd.extend(["--tools", tools])
    effort = _env_text(env, "ARC_CLAUDE_EFFORT", "low")
    if effort:
        cmd.extend(["--effort", effort])
    if _env_bool(env, "ARC_CLAUDE_NO_SESSION_PERSISTENCE", True):
        cmd.append("--no-session-persistence")
    if _env_bool(env, "ARC_CLAUDE_EXCLUDE_DYNAMIC_SYSTEM_PROMPT_SECTIONS", True):
        cmd.append("--exclude-dynamic-system-prompt-sections")
    if budget := _env_text(env, "ARC_CLAUDE_MAX_BUDGET_USD", ""):
        cmd.extend(["--max-budget-usd", budget])
    if fallback := _env_text(env, "ARC_CLAUDE_FALLBACK_MODEL", ""):
        cmd.extend(["--fallback-model", fallback])
    if mcp_configs:
        cmd.append("--mcp-config")
        cmd.extend(mcp_configs)
        if _env_bool(env, "ARC_CLAUDE_STRICT_MCP_CONFIG", True):
            cmd.append("--strict-mcp-config")
    return cmd


def _claude_tools(env: Mapping[str, str], *, allow_mcp: bool) -> str | None:
    if "ARC_CLAUDE_TOOLS" in env:
        return env["ARC_CLAUDE_TOOLS"]
    if allow_mcp:
        return "default"
    if _env_bool(env, "ARC_CLAUDE_ALLOW_INTERNET", False):
        return "WebSearch,WebFetch"
    return ""


def _mcp_configs(env: Mapping[str, str]) -> list[str]:
    values = []
    if raw_json := env.get("ARC_CLAUDE_MCP_CONFIG_JSON"):
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise LLMWorkerError(f"ARC_CLAUDE_MCP_CONFIG_JSON was not valid JSON: {exc}") from exc
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise LLMWorkerError("ARC_CLAUDE_MCP_CONFIG_JSON must be a JSON array of strings")
        values.extend(payload)
    if raw := env.get("ARC_CLAUDE_MCP_CONFIG"):
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
