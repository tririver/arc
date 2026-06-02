from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from arc_llm.json_schema import to_provider_json_schema
from arc_llm.schema_cache import canonical_json, sha256_text, write_schema_cache_file
from arc_llm.sessions import LLMSessionRef
from arc_llm.structured_recovery import parse_json_object_relaxed, structured_metadata
from arc_llm.usage import LLMProviderResponse, LLMUsage

from .base import LLMWorkerError


_RESUME_SCHEMA_SUPPORT_CACHE: dict[tuple[str, str | None], bool] = {}
_RESUME_SCHEMA_SUPPORT_LOCK = threading.Lock()


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
        provider_schema = to_provider_json_schema(schema)
        stateful = session_policy == "stateful" and session is not None
        with tempfile.TemporaryDirectory(prefix="arc-llm-") as tmp:
            tmpdir = Path(tmp)
            output_path = tmpdir / "output.json"
            schema_path: Path | None = None
            if provider_schema is not None:
                schema_path = write_schema_cache_file(
                    provider_schema,
                    cache_dir=schema_cache_dir or _default_schema_cache_dir(self.env),
                )
            resume_id = session.native_session_id if stateful else None
            use_schema = provider_schema is not None
            effective_prompt = _with_json_object_contract(prompt) if provider_schema is None else prompt
            if provider_schema is not None and resume_id and not _codex_resume_supports_output_schema(self.env):
                use_schema = False
                effective_prompt = _with_json_schema_contract(prompt, provider_schema)

            cmd = _codex_exec_cmd(
                self.env,
                stateful=stateful,
                resume_session_id=resume_id,
                output_path=output_path,
                schema_path=schema_path if use_schema else None,
                model=model,
                json_events=stateful,
            )

            try:
                result = subprocess.run(
                    cmd,
                    input=effective_prompt,
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
            native_session_id, usage, raw_events = _parse_codex_json_events(result.stdout) if stateful else (None, LLMUsage(), ())
            if stateful and not resume_id and not native_session_id:
                raise LLMWorkerError("stateful Codex call did not report thread/session id")
            payload, structured_output = _read_json_payload(output_path, result.stdout, output_recovery=output_recovery)
            return LLMProviderResponse(
                payload,
                usage=usage,
                native_session_id=native_session_id or (session.native_session_id if session else None),
                raw_events=raw_events,
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
        with tempfile.TemporaryDirectory(prefix="arc-llm-") as tmp:
            output_path = Path(tmp) / "output.txt"
            cmd = _codex_exec_cmd(
                self.env,
                stateful=stateful,
                resume_session_id=session.native_session_id if stateful else None,
                output_path=output_path,
                schema_path=None,
                model=model,
                json_events=stateful,
            )

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
            native_session_id, usage, raw_events = _parse_codex_json_events(result.stdout) if stateful else (None, LLMUsage(), ())
            if stateful and not session.native_session_id and not native_session_id:
                raise LLMWorkerError("stateful Codex call did not report thread/session id")
            try:
                value = output_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise LLMWorkerError(f"Could not read Codex text output: {exc}") from exc
            return LLMProviderResponse(
                value,
                usage=usage,
                native_session_id=native_session_id or (session.native_session_id if session else None),
                raw_events=raw_events,
                raw_output=result.stdout,
                prompt_sent_sha256=sha256_text(prompt),
            )


def _base_cmd(env: Mapping[str, str], *, stateful: bool = False) -> list[str]:
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
    if _env_bool(env, "ARC_CODEX_EPHEMERAL", False if stateful else True):
        cmd.append("--ephemeral")
    ignore_user_config_default = not (enable_mcp or profile or profile_v2)
    if enable_mcp and mcp_mode == "arc-only":
        ignore_user_config_default = True
    if _env_bool(env, "ARC_CODEX_IGNORE_USER_CONFIG", ignore_user_config_default):
        cmd.append("--ignore-user-config")
    if _env_bool(env, "ARC_CODEX_IGNORE_RULES", True):
        cmd.append("--ignore-rules")

    for key, value in _codex_config_overrides(env, stateful=stateful):
        cmd.extend(["-c", f"{key}={value}"])
    for key, value in _arc_only_mcp_config_overrides(env):
        cmd.extend(["-c", f"{key}={value}"])
    for override in _extra_config_overrides(env, mcp_mode=mcp_mode):
        cmd.extend(["-c", override])
    return cmd


def _codex_config_overrides(env: Mapping[str, str], *, stateful: bool = False) -> list[tuple[str, str]]:
    allow_internet = _env_bool(env, "ARC_CODEX_ALLOW_INTERNET", False)
    overrides = [
        ("model_reasoning_effort", _toml_string(_env_text(env, "ARC_CODEX_REASONING_EFFORT", "low"))),
        ("model_reasoning_summary", _toml_string(_env_text(env, "ARC_CODEX_REASONING_SUMMARY", "none"))),
        ("model_verbosity", _toml_string(_env_text(env, "ARC_CODEX_MODEL_VERBOSITY", "low"))),
        ("hide_agent_reasoning", _toml_bool(_env_bool(env, "ARC_CODEX_HIDE_AGENT_REASONING", True))),
        ("web_search", _toml_string(_env_text(env, "ARC_CODEX_WEB_SEARCH", "live" if allow_internet else "disabled"))),
    ]
    if env.get("ARC_CODEX_HISTORY_PERSISTENCE") is not None:
        overrides.append(("history.persistence", _toml_string(_env_text(env, "ARC_CODEX_HISTORY_PERSISTENCE", ""))))
    elif not stateful:
        overrides.append(("history.persistence", _toml_string("none")))
    if env.get("ARC_CODEX_NETWORK_ACCESS") is not None or allow_internet:
        overrides.append(
            (
                "sandbox_workspace_write.network_access",
                _toml_bool(_env_bool(env, "ARC_CODEX_NETWORK_ACCESS", allow_internet)),
            )
        )
    return [(key, value) for key, value in overrides if value]


def _codex_exec_cmd(
    env: Mapping[str, str],
    *,
    stateful: bool,
    resume_session_id: str | None,
    output_path: Path,
    schema_path: Path | None,
    model: str | None,
    json_events: bool,
) -> list[str]:
    cmd = _base_cmd(env, stateful=stateful)
    if json_events:
        cmd.append("--json")
    if schema_path is not None:
        cmd.extend(["--output-schema", str(schema_path)])
    cmd.extend(["--output-last-message", str(output_path)])
    if model:
        cmd.extend(["-m", model])
    if resume_session_id:
        cmd.extend(["resume", resume_session_id, "-"])
    else:
        cmd.append("-")
    return cmd


def _default_schema_cache_dir(env: Mapping[str, str]) -> Path:
    if value := _env_text(env, "ARC_LLM_SCHEMA_CACHE_DIR", ""):
        return Path(value).expanduser()
    return Path(tempfile.gettempdir()) / "arc-llm-schema-cache"


def _codex_resume_supports_output_schema(env: Mapping[str, str]) -> bool:
    override = env.get("ARC_CODEX_RESUME_SUPPORTS_OUTPUT_SCHEMA")
    if override is not None:
        return _env_bool(env, "ARC_CODEX_RESUME_SUPPORTS_OUTPUT_SCHEMA", False)
    key = (env.get("ARC_CODEX_BIN", "codex"), env.get("PATH"))
    with _RESUME_SCHEMA_SUPPORT_LOCK:
        cached = _RESUME_SCHEMA_SUPPORT_CACHE.get(key)
        if cached is not None:
            return cached
    supported = _probe_codex_resume_schema_support(env)
    with _RESUME_SCHEMA_SUPPORT_LOCK:
        _RESUME_SCHEMA_SUPPORT_CACHE[key] = supported
    return supported


def _probe_codex_resume_schema_support(env: Mapping[str, str]) -> bool:
    try:
        result = subprocess.run(
            [env.get("ARC_CODEX_BIN", "codex"), "exec", "resume", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            timeout=_timeout_seconds(env, "ARC_CODEX_HELP_TIMEOUT_SECONDS") or 5,
        )
    except Exception:
        return False
    return result.returncode == 0 and "--output-schema" in result.stdout


def _with_json_schema_contract(prompt: str, schema: dict[str, Any]) -> str:
    return (
        prompt.rstrip()
        + "\n\n## JSON output contract for this turn\n"
        + "Return exactly one JSON object. Do not wrap it in Markdown. It must conform to this JSON Schema:\n"
        + canonical_json(schema)
        + "\n"
    )


def _with_json_object_contract(prompt: str) -> str:
    return (
        prompt.rstrip()
        + "\n\n## JSON output contract for this turn\n"
        + "Return exactly one JSON object. Do not wrap it in Markdown.\n"
    )


def _parse_codex_json_events(stdout: str) -> tuple[str | None, LLMUsage, tuple[dict[str, Any], ...]]:
    events: list[dict[str, Any]] = []
    thread_id = None
    usage = LLMUsage()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        if event.get("type") == "thread.started":
            thread_id = str(event.get("thread_id") or event.get("session_id") or "") or thread_id
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            raw = event["usage"]
            usage = LLMUsage(
                input_tokens=_int_or_none(raw.get("input_tokens")),
                cached_input_tokens=_int_or_none(raw.get("cached_input_tokens")),
                output_tokens=_int_or_none(raw.get("output_tokens")),
                reasoning_output_tokens=_int_or_none(raw.get("reasoning_output_tokens")),
                raw=raw,
            )
    return thread_id, usage, tuple(events)


def _read_json_payload(
    output_path: Path,
    stdout: str,
    *,
    output_recovery: str = "strict",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        text = output_path.read_text(encoding="utf-8")
    except OSError:
        text = stdout
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if output_recovery != "warn":
            payload = json.loads(_extract_first_json_object(text))
        else:
            try:
                payload = json.loads(_extract_first_json_object(text))
            except Exception as exc:
                extracted, warnings = parse_json_object_relaxed(text)
                if isinstance(extracted, dict):
                    return extracted, structured_metadata(
                        severity="minor",
                        warnings=["Codex output was not direct JSON; extracted JSON object from text.", *warnings],
                        raw_text=text,
                        strategy="extract_json",
                        provider_error_type=type(exc).__name__,
                    )
                return {}, structured_metadata(
                    severity="major",
                    warnings=["Codex output did not contain a JSON object; using local recovery.", *warnings],
                    raw_text=text,
                    strategy="natural_language_fallback",
                    provider_error_type=type(exc).__name__,
                )
    if not isinstance(payload, dict):
        if output_recovery != "warn":
            raise LLMWorkerError("Codex JSON output was not an object")
        return {}, structured_metadata(
            severity="major",
            warnings=["Codex JSON output was not an object; using local recovery."],
            raw_text=text,
            strategy="natural_language_fallback",
            provider_error_type=type(payload).__name__,
        )
    return payload, None


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise LLMWorkerError("Codex JSON output did not contain an object")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise LLMWorkerError("Codex JSON output contained an unterminated object")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        ("mcp_servers.arc.required", _toml_bool(True)),
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


def _extra_config_overrides(env: Mapping[str, str], *, mcp_mode: str = "user-config") -> list[str]:
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
    if mcp_mode == "arc-only":
        for item in values:
            key = item.split("=", 1)[0].strip()
            if key.startswith("mcp_servers."):
                raise LLMWorkerError(
                    "ARC_CODEX_MCP_MODE=arc-only cannot be combined with ARC_CODEX_CONFIG/JSON "
                    "entries under mcp_servers.*"
                )
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
