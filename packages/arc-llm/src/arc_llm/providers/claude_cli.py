from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from arc_llm.sessions import LLMSessionRef
from arc_llm.attempt_diagnostics import current_attempt_diagnostics
from arc_llm.failure_classification import classify_provider_diagnostic, disposition_error_kwargs
from arc_llm.paths import llm_cache_root
from arc_llm.schema_cache import canonical_json, sha256_text
from arc_llm.response_candidates import has_complete_candidate, material_from_claude
from arc_llm.structured_recovery import parse_json_object_relaxed, structured_metadata
from arc_llm.usage import LLMProviderResponse, LLMUsage

from .base import LLMFailureCategory, LLMSubmissionState, LLMWorkerError
from .activity import ActivityTracker, resolve_idle_timeout_seconds
from .lifecycle import run_streaming_process_group


_STREAM_JSON_SUPPORT_CACHE: dict[tuple[str, str | None], bool] = {}
_STREAM_JSON_SUPPORT_LOCK = threading.Lock()


def _run_claude(cmd, prompt, *, env, idle_timeout_seconds, cancel_check, progress_callback):
    _require_stream_json_support(env)
    activity = ActivityTracker(
        provider="claude-cli",
        idle_timeout_seconds=resolve_idle_timeout_seconds(
            idle_timeout_seconds, env=env, provider="claude-cli"
        ),
        progress_callback=progress_callback,
    )
    return run_streaming_process_group(
        cmd, input_text=prompt, env=env, activity=activity,
        stdout_line_callback=lambda line: _record_claude_activity(line, activity),
        cancel_check=cancel_check,
    )


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
        response = self.generate_json_result(prompt, schema=schema, model=model)
        if response.deferred_output_error is not None:
            raise response.deferred_output_error
        return response.value

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
        defer_output_errors: bool = False,
        schema_transport: str = "auto",
        cancel_check=None,
        idle_timeout_seconds: float | None = None,
        progress_callback=None,
    ) -> LLMProviderResponse[dict[str, Any]]:
        schema = schema or {"type": "object"}
        stateful = session_policy == "stateful" and session is not None
        if schema_transport not in {"auto", "prompt"}:
            raise LLMWorkerError("schema_transport must be auto or prompt")
        mode = (
            "prompt"
            if schema_transport == "prompt"
            else _json_schema_mode(self.env, model=model, output_recovery=output_recovery)
        )
        cmd = [
            *_base_cmd(self.env, stateful=stateful),
            "--output-format",
            "stream-json",
            "--verbose",
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

        result = _run_claude(
            cmd, effective_prompt, env=self.env, idle_timeout_seconds=idle_timeout_seconds,
            cancel_check=cancel_check, progress_callback=progress_callback,
        )
        _write_raw_artifacts(artifact_dir, stdout=result.stdout, stderr=result.stderr)
        if result.returncode != 0:
            raise _claude_failure(result.stderr or result.stdout or "claude -p failed")
        terminal_output = _claude_terminal_json(result.stdout)
        candidate_material = material_from_claude(result.stdout)
        deferred_output_error: BaseException | None = None
        try:
            value, usage, returned_session_id, structured_output = _extract_claude_metadata(
                terminal_output,
                output_recovery=output_recovery,
            )
        except LLMWorkerError as exc:
            terminal_payload = json.loads(terminal_output)
            if (
                terminal_payload.get("subtype") == "error_max_structured_output_retries"
                or not defer_output_errors
                or not has_complete_candidate(candidate_material)
            ):
                raise
            value = {}
            usage = _usage_from_payload(terminal_payload)
            returned_session_id = _session_id_from_payload(terminal_payload)
            structured_output = None
            deferred_output_error = exc
        return LLMProviderResponse(
            value,
            usage=usage,
            native_session_id=returned_session_id or native_id,
            raw_output=result.stdout,
            raw_model_output=_claude_model_text(terminal_output),
            prompt_sent_sha256=sha256_text(effective_prompt),
            structured_output=structured_output,
            candidate_material=candidate_material,
            deferred_output_error=deferred_output_error,
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
        cancel_check=None,
        idle_timeout_seconds: float | None = None,
        progress_callback=None,
    ) -> LLMProviderResponse[str]:
        stateful = session_policy == "stateful" and session is not None
        cmd = _base_cmd(self.env, stateful=stateful)
        json_output = True
        cmd.extend(["--output-format", "stream-json", "--verbose"])
        native_id = session.native_session_id if stateful else None
        if stateful:
            if native_id:
                cmd.extend(["--resume", native_id])
            else:
                native_id = str(uuid.uuid4())
                cmd.extend(["--session-id", native_id])
        if model:
            cmd.extend(["--model", model])

        result = _run_claude(
            cmd, prompt, env=self.env, idle_timeout_seconds=idle_timeout_seconds,
            cancel_check=cancel_check, progress_callback=progress_callback,
        )
        _write_raw_artifacts(artifact_dir, stdout=result.stdout, stderr=result.stderr)
        if result.returncode != 0:
            raise _claude_failure(result.stderr or result.stdout or "claude -p failed")
        if json_output:
            value, usage, returned_session_id = _extract_claude_text_metadata(
                _claude_terminal_json(result.stdout)
            )
            return LLMProviderResponse(
                value,
                usage=usage,
                native_session_id=returned_session_id or native_id,
                raw_output=result.stdout,
                raw_model_output=value,
            )
        return LLMProviderResponse(
            result.stdout,
            native_session_id=native_id,
            raw_output=result.stdout,
            raw_model_output=result.stdout,
        )


def _claude_failure(message: str) -> LLMWorkerError:
    disposition = classify_provider_diagnostic(
        message,
        submission_state=LLMSubmissionState.SUBMITTED,
    )
    if disposition is not None:
        return LLMWorkerError(message, **disposition_error_kwargs(disposition))
    return LLMWorkerError(
        message,
        retryable=False,
        category=LLMFailureCategory.PROVIDER_INTERNAL,
        submission_state=LLMSubmissionState.SUBMITTED,
    )


def _require_stream_json_support(env: Mapping[str, str]) -> None:
    override = env.get("ARC_CLAUDE_STREAM_JSON_SUPPORT")
    if override is not None:
        supported = _env_bool(env, "ARC_CLAUDE_STREAM_JSON_SUPPORT", False)
    else:
        key = (env.get("ARC_CLAUDE_BIN", "claude"), env.get("PATH"))
        with _STREAM_JSON_SUPPORT_LOCK:
            cached = _STREAM_JSON_SUPPORT_CACHE.get(key)
        if cached is None:
            try:
                result = subprocess.run(
                    [key[0], "-p", "--help"], text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, env=dict(env), timeout=5,
                )
                supported = result.returncode == 0 and "stream-json" in (result.stdout + result.stderr)
            except (OSError, subprocess.SubprocessError):
                supported = False
            with _STREAM_JSON_SUPPORT_LOCK:
                _STREAM_JSON_SUPPORT_CACHE[key] = supported
        else:
            supported = cached
    if not supported:
        raise LLMWorkerError(
            "Installed Claude CLI does not advertise stream-json; upgrade Claude Code before "
            "starting a paid ARC call",
            retryable=False,
            category=LLMFailureCategory.INVALID_REQUEST,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        )


def _record_claude_activity(line: str, activity: ActivityTracker) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(event, dict):
        return
    event_type = str(event.get("type") or "")
    if event_type == "system":
        native_session_id = event.get("session_id") or event.get("sessionId")
        if native_session_id:
            activity.record_metadata(
                "session",
                text="provider session established",
                detail={"native_session_id": str(native_session_id), "resumable": True},
            )
        return
    if event_type == "assistant":
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    activity.record("assistant", text=block["text"])
                elif block.get("type") == "tool_use":
                    activity.record_tool_state(
                        tool_type="claude_tool",
                        status="running",
                        tool_id=str(block.get("id") or "anonymous"),
                    )
        return
    if event_type == "stream_event":
        inner = event.get("event")
        if not isinstance(inner, dict):
            return
        inner_type = str(inner.get("type") or "")
        # Text deltas are fragments, not classifiable progress messages.  The
        # completed assistant event above records their aggregated content.
        if inner_type == "content_block_start":
            block = inner.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                activity.record_tool_state(
                    tool_type="claude_tool",
                    status="running",
                    tool_id=str(block.get("id") or inner.get("index") or "anonymous"),
                )


def _claude_terminal_json(stdout: str) -> str:
    terminal: dict[str, Any] | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            terminal = event
    if terminal is None:
        raise _submitted_output_error("Claude stream-json output contained no terminal result event")
    return json.dumps(terminal, ensure_ascii=False)


def _submitted_output_error(message: str) -> LLMWorkerError:
    return LLMWorkerError(
        message,
        retryable=False,
        category=LLMFailureCategory.OUTPUT_INVALID,
        submission_state=LLMSubmissionState.SUBMITTED,
    )


def _claude_model_text(stdout: str) -> str:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout
    if isinstance(payload, Mapping):
        result = payload.get("result")
        if isinstance(result, str):
            return result
        structured = payload.get("structured_output")
        if isinstance(structured, Mapping):
            return json.dumps(structured, ensure_ascii=False, sort_keys=True, default=str)
    return ""


def _extract_json(stdout: str, *, output_recovery: str = "strict") -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise _submitted_output_error(f"Claude output was not JSON: {exc}") from exc
    if (
        isinstance(payload, dict)
        and payload.get("type") == "result"
        and isinstance(payload.get("structured_output"), dict)
    ):
        return payload["structured_output"], None
    if isinstance(payload, dict) and isinstance(payload.get("result"), str):
        try:
            nested = json.loads(payload["result"])
        except json.JSONDecodeError as exc:
            extracted, warnings = parse_json_object_relaxed(payload["result"])
            if isinstance(extracted, dict):
                return extracted, structured_metadata(
                    severity="minor",
                    warnings=["Claude result field was a string; extracted JSON object from it.", *warnings],
                    raw_text=payload["result"],
                    strategy="extract_json",
                )
            if output_recovery != "warn":
                raise _submitted_output_error(f"Claude result field was not JSON: {exc}") from exc
            return {}, structured_metadata(
                severity="major",
                warnings=["Claude result field was not JSON; using schema recovery.", *warnings],
                raw_text=payload["result"],
                strategy="natural_language_fallback",
            )
        if not isinstance(nested, dict):
            if output_recovery == "warn":
                return {}, structured_metadata(
                    severity="major",
                    warnings=[
                        "Claude result field decoded as JSON but was not an object; using local recovery.",
                        f"decoded_type={type(nested).__name__}",
                    ],
                    raw_text=payload["result"],
                    strategy="natural_language_fallback",
                    provider_error_type=type(nested).__name__,
                )
            raise _submitted_output_error("Claude result JSON was not an object")
        return nested, None
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        return payload["result"], None
    if isinstance(payload, dict):
        return payload, None
    raise _submitted_output_error("Claude JSON output was not an object")


def _extract_claude_metadata(stdout: str, *, output_recovery: str = "strict") -> tuple[dict[str, Any], LLMUsage, str | None, dict[str, Any] | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        if output_recovery != "warn":
            raise _submitted_output_error(f"Claude output was not JSON: {exc}") from exc
        return (
            {},
            LLMUsage(),
            None,
            structured_metadata(
                severity="major",
                warnings=[
                    f"Claude output was not a JSON envelope; accepted as natural language warning: {exc}"
                ],
                raw_text=stdout,
                strategy="natural_language_fallback",
                provider_error_type="JSONDecodeError",
            ),
        )
    if isinstance(payload, dict) and payload.get("subtype") == "error_max_structured_output_retries":
        raise LLMWorkerError(
            "Claude provider exhausted structured-output retries",
            retryable=False,
            category=LLMFailureCategory.OUTPUT_INVALID,
            submission_state=LLMSubmissionState.SUBMITTED,
        )
    value, structured_output = _extract_json(stdout, output_recovery=output_recovery)
    return value, _usage_from_payload(payload), _session_id_from_payload(payload), structured_output


def _extract_claude_text_metadata(stdout: str) -> tuple[str, LLMUsage, str | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise _submitted_output_error(f"Claude output was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _submitted_output_error("Claude text output JSON was not an object")
    usage = _usage_from_payload(payload)
    result = payload.get("result", "")
    if not isinstance(result, str):
        raise _submitted_output_error("Claude text result field was not a string")
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
    allowed_tools = _claude_allowed_tools(env, allow_mcp=allow_mcp)
    if allowed_tools is not None:
        cmd.extend(["--allowedTools", allowed_tools])
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


def _json_schema_mode(env: Mapping[str, str], *, model: str | None, output_recovery: str = "strict") -> str:
    raw = _env_text(env, "ARC_CLAUDE_JSON_SCHEMA_MODE", "auto").strip().lower()
    if raw not in {"auto", "provider", "prompt"}:
        raise LLMWorkerError("ARC_CLAUDE_JSON_SCHEMA_MODE must be auto, provider, or prompt")
    if raw != "auto":
        return raw
    if output_recovery == "warn":
        warn_mode = _env_text(env, "ARC_CLAUDE_WARN_JSON_SCHEMA_MODE", "prompt").strip().lower()
        if warn_mode not in {"provider", "prompt"}:
            raise LLMWorkerError("ARC_CLAUDE_WARN_JSON_SCHEMA_MODE must be provider or prompt")
        return warn_mode
    return "prompt"


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
    if current_attempt_diagnostics() is not None:
        # Streaming lifecycle capture already owns immutable, bounded attempt
        # diagnostics. Do not recreate mutable call-root raw files.
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
        return "WebSearch,WebFetch" if allow_internet else ""
    if allow_internet:
        return "WebSearch,WebFetch"
    return ""


def _claude_allowed_tools(env: Mapping[str, str], *, allow_mcp: bool) -> str | None:
    if "ARC_CLAUDE_ALLOWED_TOOLS" in env:
        return env["ARC_CLAUDE_ALLOWED_TOOLS"]
    if allow_mcp and _env_text(env, "ARC_CLAUDE_MCP_MODE", "user-config") == "arc-only":
        return "mcp__arc__*"
    return None


def _mcp_configs(env: Mapping[str, str]) -> list[str]:
    mcp_mode = _env_text(env, "ARC_CLAUDE_MCP_MODE", "user-config")
    if mcp_mode not in {"", "user-config", "arc-only"}:
        raise LLMWorkerError("ARC_CLAUDE_MCP_MODE must be one of: user-config, arc-only")
    if mcp_mode == "arc-only":
        if (env.get("ARC_CLAUDE_MCP_CONFIG") or env.get("ARC_CLAUDE_MCP_CONFIG_JSON")) and not _env_bool(
            env, "ARC_CLAUDE_ARC_ONLY_ALLOW_EXTRA_CONFIGS", False
        ):
            raise LLMWorkerError(
                "ARC_CLAUDE_MCP_MODE=arc-only cannot be combined with ARC_CLAUDE_MCP_CONFIG "
                "or ARC_CLAUDE_MCP_CONFIG_JSON. Use user-config mode or unset the extra configs."
            )
        return [str(_write_arc_only_mcp_config(env))]
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


def _write_arc_only_mcp_config(env: Mapping[str, str]) -> Path:
    path = _arc_only_mcp_config_path(env)
    command, args = _arc_mcp_command_and_args(env)
    mcp_env = {"ARC_AGENT_HOST": "claude"}
    for key in ("ARC_PAPER_CACHE", "ARC_DOMAIN_CACHE", "ARC_JOBS_CACHE"):
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
    return llm_cache_root(env) / "mcp" / "arc-claude-mcp.json"


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
