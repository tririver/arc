from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

from arc_llm.sessions import LLMSessionRef
from arc_llm.schema_cache import canonical_json, sha256_text
from arc_llm.structured_recovery import parse_json_object_relaxed, structured_metadata
from arc_llm.usage import LLMProviderResponse, LLMUsage

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
        return self.generate_json_result(prompt, schema=schema, model=model).value

    def generate_json_result(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        session: LLMSessionRef | None = None,
        session_policy: str = "stateless",
        schema_cache_dir: Path | None = None,
        artifact_dir: Path | None = None,
        output_recovery: str = "strict",
    ) -> LLMProviderResponse[dict[str, Any]]:
        schema = schema or {"type": "object"}
        stateful = session_policy == "stateful" and session is not None
        mode = _json_schema_mode(self.env, model=model)
        cmd = [
            *_base_cmd(self.env, stateful=stateful),
            "--output-format",
            "json",
        ]
        effective_prompt = prompt
        if mode == "provider":
            cmd.extend(["--json-schema", json.dumps(schema, ensure_ascii=False)])
        else:
            effective_prompt = _with_json_schema_contract(prompt, schema)
        native_id = session.native_session_id if stateful else None
        if stateful:
            if native_id:
                cmd.extend(["--resume", native_id])
            else:
                native_id = str(uuid.uuid4())
                cmd.extend(["--session-id", native_id])
        if model:
            cmd.extend(["--model", model])

        try:
            result = subprocess.run(
                cmd,
                input=effective_prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(self.env),
                timeout=_timeout_seconds(self.env, "ARC_CLAUDE_TIMEOUT_SECONDS"),
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMWorkerError(f"claude -p timed out after {exc.timeout} seconds") from exc
        _write_raw_artifacts(artifact_dir, stdout=result.stdout, stderr=result.stderr)
        if result.returncode != 0:
            recovered = _maybe_recover_claude_error_envelope(result.stdout, output_recovery=output_recovery)
            if recovered is not None:
                value, usage, returned_session_id, structured_output = recovered
                return LLMProviderResponse(
                    value,
                    usage=usage,
                    native_session_id=returned_session_id or native_id,
                    raw_output=result.stdout,
                    prompt_sent_sha256=sha256_text(effective_prompt),
                    structured_output=structured_output,
                )
            raise LLMWorkerError(result.stderr or result.stdout or "claude -p failed")
        value, usage, returned_session_id, structured_output = _extract_claude_metadata(
            result.stdout,
            output_recovery=output_recovery,
        )
        return LLMProviderResponse(
            value,
            usage=usage,
            native_session_id=returned_session_id or native_id,
            raw_output=result.stdout,
            prompt_sent_sha256=sha256_text(effective_prompt),
            structured_output=structured_output,
        )

    def generate_text(self, prompt: str, *, model: str | None = None) -> str:
        return self.generate_text_result(prompt, model=model).value

    def generate_text_result(
        self,
        prompt: str,
        *,
        model: str | None = None,
        session: LLMSessionRef | None = None,
        session_policy: str = "stateless",
        artifact_dir: Path | None = None,
    ) -> LLMProviderResponse[str]:
        stateful = session_policy == "stateful" and session is not None
        cmd = _base_cmd(self.env, stateful=stateful)
        json_output = stateful or _env_bool(self.env, "ARC_CLAUDE_TEXT_OUTPUT_FORMAT_JSON", False)
        if json_output:
            cmd.extend(["--output-format", "json"])
        native_id = session.native_session_id if stateful else None
        if stateful:
            if native_id:
                cmd.extend(["--resume", native_id])
            else:
                native_id = str(uuid.uuid4())
                cmd.extend(["--session-id", native_id])
        if model:
            cmd.extend(["--model", model])

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(self.env),
                timeout=_timeout_seconds(self.env, "ARC_CLAUDE_TIMEOUT_SECONDS"),
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMWorkerError(f"claude -p timed out after {exc.timeout} seconds") from exc
        if result.returncode != 0:
            raise LLMWorkerError(result.stderr or result.stdout or "claude -p failed")
        if json_output:
            value, usage, returned_session_id = _extract_claude_text_metadata(result.stdout)
            return LLMProviderResponse(
                value,
                usage=usage,
                native_session_id=returned_session_id or native_id,
                raw_output=result.stdout,
            )
        return LLMProviderResponse(result.stdout, native_session_id=native_id, raw_output=result.stdout)


def _extract_json(stdout: str, *, output_recovery: str = "strict") -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LLMWorkerError(f"Claude output was not JSON: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("result"), str):
        try:
            nested = json.loads(payload["result"])
        except json.JSONDecodeError as exc:
            if output_recovery != "warn":
                raise LLMWorkerError(f"Claude result field was not JSON: {exc}") from exc
            extracted, warnings = parse_json_object_relaxed(payload["result"])
            if isinstance(extracted, dict):
                return extracted, structured_metadata(
                    severity="minor",
                    warnings=["Claude result field was a string; extracted JSON object from it.", *warnings],
                    raw_text=payload["result"],
                    strategy="extract_json",
                )
            return {}, structured_metadata(
                severity="major",
                warnings=["Claude result field was not JSON; using schema recovery.", *warnings],
                raw_text=payload["result"],
                strategy="natural_language_fallback",
            )
        if not isinstance(nested, dict):
            raise LLMWorkerError("Claude result JSON was not an object")
        return nested, None
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        return payload["result"], None
    if isinstance(payload, dict):
        return payload, None
    raise LLMWorkerError("Claude JSON output was not an object")


def _extract_claude_metadata(stdout: str, *, output_recovery: str = "strict") -> tuple[dict[str, Any], LLMUsage, str | None, dict[str, Any] | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LLMWorkerError(f"Claude output was not JSON: {exc}") from exc
    value, structured_output = _extract_json(stdout, output_recovery=output_recovery)
    return value, _usage_from_payload(payload), _session_id_from_payload(payload), structured_output


def _extract_claude_text_metadata(stdout: str) -> tuple[str, LLMUsage, str | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LLMWorkerError(f"Claude output was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMWorkerError("Claude text output JSON was not an object")
    usage = _usage_from_payload(payload)
    result = payload.get("result", "")
    if not isinstance(result, str):
        raise LLMWorkerError("Claude text result field was not a string")
    native_session_id = payload.get("session_id") or payload.get("sessionId")
    return result, usage, str(native_session_id) if native_session_id is not None else None


def _maybe_recover_claude_error_envelope(
    stdout: str,
    *,
    output_recovery: str,
) -> tuple[dict[str, Any], LLMUsage, str | None, dict[str, Any] | None] | None:
    if output_recovery != "warn":
        return None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("subtype") != "error_max_structured_output_retries":
        return None
    return (
        {},
        _usage_from_payload(payload),
        _session_id_from_payload(payload),
        structured_metadata(
            severity="major",
            warnings=["Claude structured output failed after provider retries."],
            raw_text=stdout,
            strategy="schema_default",
            provider_error_type="error_max_structured_output_retries",
        ),
    )


def _usage_from_payload(payload: Any) -> LLMUsage:
    usage_source = payload.get("usage") or payload.get("current_usage") or {} if isinstance(payload, dict) else {}
    if not isinstance(usage_source, dict):
        usage_source = {}
    return LLMUsage(
        input_tokens=_int_or_none(usage_source.get("input_tokens")),
        output_tokens=_int_or_none(usage_source.get("output_tokens")),
        cache_creation_input_tokens=_int_or_none(usage_source.get("cache_creation_input_tokens")),
        cache_read_input_tokens=_int_or_none(usage_source.get("cache_read_input_tokens")),
        raw=usage_source,
    )


def _session_id_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    native_session_id = payload.get("session_id") or payload.get("sessionId")
    return str(native_session_id) if native_session_id is not None else None


def _base_cmd(env: Mapping[str, str], *, stateful: bool = False) -> list[str]:
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
    if env.get("ARC_CLAUDE_NO_SESSION_PERSISTENCE") is not None:
        no_session_persistence = _env_bool(env, "ARC_CLAUDE_NO_SESSION_PERSISTENCE", True)
    else:
        no_session_persistence = False if stateful else True
    if no_session_persistence:
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


def _json_schema_mode(env: Mapping[str, str], *, model: str | None) -> str:
    raw = _env_text(env, "ARC_CLAUDE_JSON_SCHEMA_MODE", "auto").strip().lower()
    if raw not in {"auto", "provider", "prompt"}:
        raise LLMWorkerError("ARC_CLAUDE_JSON_SCHEMA_MODE must be auto, provider, or prompt")
    if raw != "auto":
        return raw
    model_text = (model or env.get("ARC_CLAUDE_MODEL") or "").lower()
    prompt_markers = _env_text(env, "ARC_CLAUDE_JSON_SCHEMA_PROMPT_MODELS", _default_prompt_schema_model_markers())
    if any(marker and marker.lower() in model_text for marker in prompt_markers.split(",")):
        return "prompt"
    return "provider"


def _default_prompt_schema_model_markers() -> str:
    return "".join(("deep", "seek"))


def _with_json_schema_contract(prompt: str, schema: Mapping[str, Any]) -> str:
    return (
        prompt.rstrip()
        + "\n\n## JSON output contract\n"
        + "Return exactly one JSON object and no surrounding prose.\n"
        + "The JSON object must satisfy the supplied schema. Every required field must be present.\n"
        + "Do not wrap the object in Markdown. Do not put the JSON object inside a string field such as result.\n"
        + "If uncertain, use a short explanatory string for required string fields.\n"
        + "Use null only when the schema explicitly allows null.\n"
        + canonical_json(dict(schema))
        + "\n"
    )


def _write_raw_artifacts(artifact_dir: Path | None, *, stdout: str, stderr: str) -> None:
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "raw_stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
    (artifact_dir / "raw_stderr.txt").write_text(stderr, encoding="utf-8", errors="replace")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _claude_tools(env: Mapping[str, str], *, allow_mcp: bool) -> str | None:
    if "ARC_CLAUDE_TOOLS" in env:
        return env["ARC_CLAUDE_TOOLS"]
    allow_internet = _env_bool(env, "ARC_CLAUDE_ALLOW_INTERNET", False)
    if allow_mcp:
        if not allow_internet:
            raise LLMWorkerError(
                "Claude MCP with ARC_CLAUDE_ALLOW_INTERNET=false requires explicit ARC_CLAUDE_TOOLS. "
                "Refusing to use --tools default because that may enable non-ARC built-in tools."
            )
        return "default"
    if allow_internet:
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
    mcp_mode = _env_text(env, "ARC_CLAUDE_MCP_MODE", "user-config")
    if mcp_mode not in {"", "user-config", "arc-only"}:
        raise LLMWorkerError("ARC_CLAUDE_MCP_MODE must be one of: user-config, arc-only")
    if mcp_mode == "arc-only" and not values:
        values.append(str(_write_arc_only_mcp_config(env)))
    return values


def _write_arc_only_mcp_config(env: Mapping[str, str]) -> Path:
    path = _arc_only_mcp_config_path(env)
    command, args = _arc_mcp_command_and_args(env)
    mcp_env = {"ARC_AGENT_HOST": "claude"}
    for key in ("ARC_PAPER_CACHE", "ARC_DOMAIN_CACHE", "ARC_MCP_CACHE"):
        if value := env.get(key):
            mcp_env[key] = value
    if raw := env.get("ARC_CLAUDE_ARC_MCP_ENV_JSON"):
        try:
            extra_env = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMWorkerError(f"ARC_CLAUDE_ARC_MCP_ENV_JSON was not valid JSON: {exc}") from exc
        if not isinstance(extra_env, dict) or not all(isinstance(key, str) for key in extra_env):
            raise LLMWorkerError("ARC_CLAUDE_ARC_MCP_ENV_JSON must be a JSON object with string keys")
        mcp_env.update({key: str(value) for key, value in extra_env.items()})
    payload = {"mcpServers": {"arc": {"command": command, "args": args, "env": mcp_env}}}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _arc_only_mcp_config_path(env: Mapping[str, str]) -> Path:
    if value := _env_text(env, "ARC_CLAUDE_ARC_MCP_CONFIG_PATH", ""):
        return Path(value).expanduser()
    cache_home = env.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return base / "arc-llm" / "mcp" / "arc-claude-mcp.json"


def _arc_mcp_command_and_args(env: Mapping[str, str]) -> tuple[str, list[str]]:
    if command := _env_text(env, "ARC_CLAUDE_ARC_MCP_COMMAND", ""):
        return command, _env_json_string_list(env, "ARC_CLAUDE_ARC_MCP_ARGS_JSON")
    if command := shutil.which("arc-mcp"):
        return command, []
    sibling = Path(sys.executable).with_name("arc-mcp")
    if sibling.exists():
        return str(sibling), []
    return "arc-mcp", []


def _env_json_string_list(env: Mapping[str, str], key: str) -> list[str]:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMWorkerError(f"{key} was not valid JSON: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LLMWorkerError(f"{key} must be a JSON array of strings")
    return value


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
