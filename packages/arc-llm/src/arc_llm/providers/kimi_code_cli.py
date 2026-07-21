from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Mapping

from arc_llm.retryable_output import ProviderOutputClass, classify_provider_output_text
from arc_llm.failure_classification import classify_provider_diagnostic, disposition_error_kwargs
from arc_llm.schema_cache import canonical_json, sha256_text
from arc_llm.sessions import LLMSessionRef
from arc_llm.structured_recovery import parse_json_object_relaxed, structured_metadata
from arc_llm.usage import LLMProviderResponse, LLMUsage

from .base import (
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerCancelled,
    LLMWorkerError,
    LLMWorkerTimeout,
)
from .kimi_safety import resolve_kimi_retry_safety
from .lifecycle import (
    resolve_worker_call_timeout_seconds,
    start_process_group_watchdog,
    stop_process_group_watchdog,
    terminate_process_group,
)


EXPERIMENTAL_WARNING = (
    "kimi-code-cli is experimental and inherits Kimi Code configuration, instructions, skills, hooks, "
    "plugins, MCP, tool permissions, and persistent sessions; it may access the network, run commands, "
    "and modify files."
)
MINIMUM_KIMI_VERSION = (0, 28, 0)
_WARNING_LOCK = threading.Lock()
_WARNING_EMITTED = False


class KimiCodeCliProvider:
    """Kimi Code provider using its ACP stdio transport.

    Each ARC call gets a fresh ACP process. Stateful ARC sessions resume the
    provider-owned Kimi session by its native session id; stateless calls still
    create the provider's normal persistent session on disk.
    """

    name = "kimi-code-cli"

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
        timeout_seconds: float | None = None,
        deadline: float | None = None,
        cancel_check=None,
    ) -> LLMProviderResponse[dict[str, Any]]:
        del schema_cache_dir
        effective_prompt = _with_json_schema_contract(prompt, schema or {"type": "object"})
        response = self._generate(
            effective_prompt,
            model=model,
            session=session,
            session_policy=session_policy,
            artifact_dir=artifact_dir,
            timeout_seconds=timeout_seconds,
            deadline=deadline,
            cancel_check=cancel_check,
        )
        value, recovery = _parse_json_response(response.value, output_recovery=output_recovery)
        return LLMProviderResponse(
            value,
            usage=response.usage,
            native_session_id=response.native_session_id,
            raw_events=response.raw_events,
            raw_output=response.raw_output,
            prompt_sent_sha256=response.prompt_sent_sha256,
            structured_output=recovery,
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
        timeout_seconds: float | None = None,
        deadline: float | None = None,
        cancel_check=None,
    ) -> LLMProviderResponse[str]:
        return self._generate(
            prompt,
            model=model,
            session=session,
            session_policy=session_policy,
            artifact_dir=artifact_dir,
            timeout_seconds=timeout_seconds,
            deadline=deadline,
            cancel_check=cancel_check,
        )

    def _generate(
        self,
        prompt: str,
        *,
        model: str | None,
        session: LLMSessionRef | None,
        session_policy: str,
        artifact_dir: Path | None,
        timeout_seconds: float | None,
        deadline: float | None,
        cancel_check,
    ) -> LLMProviderResponse[str]:
        _warn_experimental_once()
        stateful = session_policy == "stateful" and session is not None
        resume_id = session.native_session_id if stateful else None
        work_dir = _work_dir(self.env)
        timeout = resolve_worker_call_timeout_seconds(timeout_seconds, env=self.env, provider=self.name)
        effective_deadline = deadline if deadline is not None else (
            time.monotonic() + timeout if timeout is not None else None
        )
        client = _AcpProcess(
            self.env,
            artifact_dir=artifact_dir,
            deadline=effective_deadline,
            cancel_check=cancel_check,
        )
        native_session_id: str | None = None
        try:
            client.start()
            initialized = client.request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {"readTextFile": False, "writeTextFile": False},
                    },
                },
            )
            _validate_initialize(initialized)
            client.request("authenticate", {"methodId": "login"})
            if resume_id:
                resumed = client.request(
                    "session/resume",
                    {"sessionId": resume_id, "cwd": str(work_dir), "mcpServers": []},
                )
                native_session_id = resume_id
                if isinstance(resumed, dict) and resumed.get("sessionId"):
                    native_session_id = str(resumed["sessionId"])
            else:
                created = client.request(
                    "session/new",
                    {"cwd": str(work_dir), "mcpServers": []},
                )
                if not isinstance(created, dict) or not created.get("sessionId"):
                    raise _worker_error("Kimi ACP session/new did not return a sessionId", retryable=False)
                native_session_id = str(created["sessionId"])

            if model and model != "default_model":
                client.request(
                    "session/set_config_option",
                    {"sessionId": native_session_id, "configId": "model", "value": model},
                )
            client.current_session_id = native_session_id
            result = client.request(
                "session/prompt",
                {"sessionId": native_session_id, "prompt": [{"type": "text", "text": prompt}]},
            )
            if isinstance(result, dict) and result.get("stopReason") in {"cancelled", "refusal"}:
                raise _worker_error(
                    f"Kimi ACP prompt stopped with {result.get('stopReason')}",
                    retryable=result.get("stopReason") == "cancelled",
                )
            text = "".join(client.message_chunks)
            if not text:
                # A clean end_turn means the Kimi CLI already completed its own
                # request lifecycle. Recreating an ARC stateless session cannot
                # distinguish a transient provider failure from exhausted quota
                # and can multiply provider-side retries, so leave recovery to a
                # later workflow resume instead of retrying immediately.
                raise _worker_error(
                    "Kimi ACP prompt returned no agent message text",
                    retryable=False,
                    abort_batch=True,
                )
            return LLMProviderResponse(
                text,
                usage=LLMUsage(),
                native_session_id=native_session_id,
                raw_events=tuple(client.events),
                raw_output=text,
                raw_model_output=text,
                prompt_sent_sha256=sha256_text(prompt),
            )
        except (_AcpTimeout, LLMWorkerCancelled) as exc:
            client.cancel_and_stop(native_session_id)
            if isinstance(exc, LLMWorkerCancelled):
                raise
            raise LLMWorkerTimeout("kimi acp timed out") from exc
        finally:
            client.close()


class _AcpTimeout(TimeoutError):
    pass


class _AcpProcess:
    def __init__(
        self,
        env: Mapping[str, str],
        *,
        artifact_dir: Path | None,
        deadline: float | None = None,
        cancel_check=None,
    ) -> None:
        self.env = env
        self.artifact_dir = artifact_dir
        self.process: subprocess.Popen[str] | None = None
        self.watchdog: subprocess.Popen[object] | None = None
        self.stdout_queue: queue.Queue[str | None] = queue.Queue()
        self.stderr_lines: list[str] = []
        self.stdout_lines: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.message_chunks: list[str] = []
        self.current_session_id: str | None = None
        self._next_id = 1
        self._write_lock = threading.Lock()
        fallback_timeout = _timeout_seconds(env)
        self._deadline = deadline if deadline is not None else (
            time.monotonic() + fallback_timeout if fallback_timeout is not None else None
        )
        self._cancel_check = cancel_check
        self._threads: list[threading.Thread] = []
        self._stopped = False
        self._fatal_error: LLMWorkerError | None = None
        self._fatal_lock = threading.Lock()

    def start(self) -> None:
        retry_safety = resolve_kimi_retry_safety(self.env)
        command = list(retry_safety.command)
        if retry_safety.warning:
            warnings.warn(retry_safety.warning, RuntimeWarning, stacklevel=2)
        child_env = dict(self.env)
        child_env.update(
            {
                "KIMI_CODE_NO_AUTO_UPDATE": "1",
                "KIMI_DISABLE_TELEMETRY": "1",
                "KIMI_DISABLE_CRON": "1",
            }
        )
        kwargs: dict[str, Any] = {"start_new_session": True}
        if os.name == "nt":
            kwargs = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_env,
                **kwargs,
            )
        except FileNotFoundError as exc:
            raise _worker_error(
                f"Kimi Code binary not found: {command[0]}. Install @moonshot-ai/kimi-code >=0.28.0.",
                retryable=False,
            ) from exc
        self.watchdog = start_process_group_watchdog(self.process, grace_seconds=0.35)
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        stdout_thread = threading.Thread(target=self._read_stdout, name="arc-kimi-stdout", daemon=True)
        stderr_thread = threading.Thread(target=self._read_stderr, name="arc-kimi-stderr", daemon=True)
        self._threads = [stdout_thread, stderr_thread]
        for thread in self._threads:
            thread.start()

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.stdout_lines.append(line)
                self.stdout_queue.put(line)
        finally:
            self.stdout_queue.put(None)

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_lines.append(line)
            disposition = classify_provider_diagnostic(
                line,
                submission_state=LLMSubmissionState.UNKNOWN,
            )
            if disposition is not None and disposition.abort_scope.value == "provider":
                with self._fatal_lock:
                    if self._fatal_error is None:
                        self._fatal_error = LLMWorkerError(
                            f"Kimi provider reported {disposition.category.value}",
                            **disposition_error_kwargs(disposition),
                        )

    def request(self, method: str, params: dict[str, Any]) -> Any:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            with self._fatal_lock:
                fatal_error = self._fatal_error
            if fatal_error is not None:
                raise fatal_error
            if self._cancel_check is not None and self._cancel_check():
                raise LLMWorkerCancelled("Kimi ACP call was cancelled")
            remaining = None if self._deadline is None else self._deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise _AcpTimeout()
            try:
                line = self.stdout_queue.get(
                    timeout=0.1 if remaining is None else min(remaining, 0.1)
                )
            except queue.Empty:
                continue
            if line is None:
                return_code = self.process.poll() if self.process is not None else None
                diagnostic = "".join(self.stderr_lines).strip()
                message = f"kimi acp exited before replying to {method}"
                if return_code is not None:
                    message += f" (exit {return_code})"
                if diagnostic:
                    message += f": {diagnostic}"
                retryable, abort_batch = _diagnostic_disposition(diagnostic, default=True)
                raise _worker_error(message, retryable=retryable, abort_batch=abort_batch)
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _worker_error(f"Kimi ACP emitted invalid JSON: {exc}", retryable=False) from exc
            if not isinstance(event, dict):
                raise _worker_error("Kimi ACP emitted a non-object JSON-RPC message", retryable=False)
            self.events.append(event)
            self._capture_update(event)
            if "method" in event and "id" in event:
                self._handle_reverse_request(event)
                continue
            if event.get("id") != request_id:
                continue
            if "error" in event:
                raise _rpc_error(method, event["error"])
            if "result" not in event:
                raise _worker_error(f"Kimi ACP response to {method} had no result", retryable=False)
            return event.get("result")

    def _capture_update(self, event: dict[str, Any]) -> None:
        if event.get("method") != "session/update":
            return
        params = event.get("params")
        if not isinstance(params, dict):
            return
        if self.current_session_id and params.get("sessionId") != self.current_session_id:
            return
        update = params.get("update")
        if not isinstance(update, dict) or update.get("sessionUpdate") != "agent_message_chunk":
            return
        content = update.get("content")
        if isinstance(content, dict) and content.get("type") == "text" and isinstance(content.get("text"), str):
            self.message_chunks.append(content["text"])

    def _handle_reverse_request(self, event: dict[str, Any]) -> None:
        method = str(event.get("method") or "")
        if method == "session/request_permission":
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": event["id"],
                    "result": {"outcome": {"outcome": "cancelled"}},
                }
            )
            return
        if method in {"fs/read_text_file", "fs/write_text_file"}:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": event["id"],
                    "error": {"code": -32001, "message": "ARC denies ACP reverse filesystem access"},
                }
            )
            return
        self._send(
            {
                "jsonrpc": "2.0",
                "id": event["id"],
                "error": {"code": -32601, "message": f"ARC does not implement reverse method {method}"},
            }
        )

    def _send(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise _worker_error("Kimi ACP stdin is unavailable", retryable=True)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            with self._write_lock:
                self.process.stdin.write(line)
                self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            diagnostic = "".join(self.stderr_lines).strip()
            message = "Kimi ACP transport closed while writing"
            if diagnostic:
                message += f": {diagnostic}"
            retryable, abort_batch = _diagnostic_disposition(diagnostic, default=True)
            raise _worker_error(
                message,
                retryable=retryable,
                abort_batch=abort_batch,
            ) from exc

    def cancel_and_stop(self, session_id: str | None) -> None:
        if self._stopped:
            return
        if session_id and self.process is not None and self.process.poll() is None:
            try:
                self._send(
                    {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": session_id}}
                )
            except Exception:
                pass
        self._close_stdin()
        self._wait_or_signal()

    def close(self) -> None:
        if not self._stopped:
            self._close_stdin()
            self._wait_or_signal()
        for thread in self._threads:
            thread.join(timeout=0.2)
        self._write_artifacts()

    def _close_stdin(self) -> None:
        if self.process is not None and self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass

    def _wait_or_signal(self) -> None:
        process = self.process
        if process is None:
            self._stopped = True
            return
        try:
            process.wait(timeout=0.35)
        except subprocess.TimeoutExpired:
            pass
        # Cleanup must also inspect the process group after the ACP leader has
        # exited; helpers otherwise survive a successful or cancelled call.
        terminate_process_group(process, grace_seconds=0.35)
        stop_process_group_watchdog(self.watchdog)
        self.watchdog = None
        self._stopped = True

    def _write_artifacts(self) -> None:
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        raw_events = "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in self.events
        )
        (self.artifact_dir / "raw_events.jsonl").write_text(raw_events, encoding="utf-8")
        (self.artifact_dir / "raw_stdout.txt").write_text(
            "".join(self.stdout_lines), encoding="utf-8", errors="replace"
        )
        (self.artifact_dir / "raw_stderr.txt").write_text(
            "".join(self.stderr_lines), encoding="utf-8", errors="replace"
        )


def _validate_initialize(result: Any) -> None:
    if not isinstance(result, dict):
        raise _worker_error("Kimi ACP initialize returned a non-object result", retryable=False)
    version_value = (result.get("agentInfo") or {}).get("version") if isinstance(result.get("agentInfo"), dict) else None
    version = _parse_version(version_value)
    if version is None:
        raise _worker_error("Kimi ACP initialize did not report a valid agent version", retryable=False)
    if version < MINIMUM_KIMI_VERSION:
        rendered = str(version_value or "unknown")
        raise _worker_error(
            f"Kimi Code {rendered} is incompatible; kimi-code-cli requires >=0.28.0",
            retryable=False,
        )


def _parse_version(value: Any) -> tuple[int, int, int] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)", str(value or ""))
    if not match:
        return None
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def _rpc_error(method: str, raw_error: Any) -> LLMWorkerError:
    error = raw_error if isinstance(raw_error, dict) else {}
    code = error.get("code")
    message = str(error.get("message") or raw_error or "unknown JSON-RPC error")
    if method == "authenticate":
        return _worker_error(
            f"Kimi Code authentication is unavailable; run `kimi login` first: {message}",
            retryable=False,
            abort_batch=True,
        )
    if code in {-32600, -32601, -32602, -32603, -32700}:
        return _worker_error(f"Kimi ACP protocol error during {method}: {message}", retryable=False)
    retryable, abort_batch = _diagnostic_disposition(message, default=True)
    return _worker_error(
        f"Kimi ACP error during {method}: {message}",
        retryable=retryable,
        abort_batch=abort_batch,
    )


def _diagnostic_disposition(diagnostic: str, *, default: bool) -> tuple[bool, bool]:
    """Return retryability and batch-abort policy for a Kimi diagnostic."""
    normalized = re.sub(r"[-_\s]+", " ", str(diagnostic or "").strip().lower())
    if any(
        phrase in normalized
        for phrase in (
            "usage limit",
            "quota exhausted",
            "quota exceeded",
            "insufficient quota",
            "billing hard limit",
        )
    ):
        return False, True
    if any(
        phrase in normalized
        for phrase in (
            "too many requests",
            "rate limit",
            "authentication failed",
            "invalid api key",
            "login required",
            "not logged in",
            "unauthorized",
            "forbidden",
        )
    ):
        return False, True
    if re.search(r"(?<!\d)(?:401|403|429)(?!\d)", normalized):
        return False, True
    classification = classify_provider_output_text(diagnostic)
    if classification.classification == ProviderOutputClass.FATAL_PROVIDER_FAILURE:
        return False, False
    if classification.classification == ProviderOutputClass.RETRYABLE_PROVIDER_FAILURE:
        return True, False
    return default, False


def _worker_error(
    message: str,
    *,
    retryable: bool,
    abort_batch: bool = False,
    submission_state: LLMSubmissionState = LLMSubmissionState.UNKNOWN,
) -> LLMWorkerError:
    disposition = classify_provider_diagnostic(
        message,
        submission_state=submission_state,
    )
    if disposition is not None:
        return LLMWorkerError(message, **disposition_error_kwargs(disposition))
    return LLMWorkerError(
        message,
        retryable=False,
        abort_batch=abort_batch,
        category=(
            LLMFailureCategory.PROVIDER_INTERNAL
            if retryable or abort_batch
            else LLMFailureCategory.OUTPUT_INVALID
        ),
        submission_state=submission_state,
    )


def _timeout_seconds(env: Mapping[str, str]) -> float | None:
    return resolve_worker_call_timeout_seconds(None, env=env, provider="kimi-code-cli")


def _work_dir(env: Mapping[str, str]) -> Path:
    raw = env.get("ARC_KIMI_WORK_DIR")
    path = Path(raw).expanduser() if raw and raw.strip() else Path.cwd()
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise _worker_error(f"Kimi work directory is unavailable: {path}: {exc}", retryable=False) from exc


def _with_json_schema_contract(prompt: str, schema: Mapping[str, Any]) -> str:
    return (
        prompt.rstrip()
        + "\n\n## JSON output contract for this turn\n"
        + "Return exactly one JSON object and no surrounding prose. Do not wrap it in Markdown. "
        + "The object must conform to this canonical JSON Schema:\n"
        + canonical_json(dict(schema))
        + "\n"
    )


def _parse_json_response(
    text: str,
    *,
    output_recovery: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as direct_error:
        value, parser_warnings = parse_json_object_relaxed(text)
        if isinstance(value, dict):
            return value, structured_metadata(
                severity="minor",
                warnings=["Kimi output was not direct JSON; extracted a JSON object from text.", *parser_warnings],
                raw_text=text,
                strategy="extract_json",
                provider_error_type=type(direct_error).__name__,
            )
        if output_recovery == "warn":
            return {}, structured_metadata(
                severity="major",
                warnings=["Kimi output did not contain a JSON object; using local recovery.", *parser_warnings],
                raw_text=text,
                strategy="natural_language_fallback",
                provider_error_type=type(direct_error).__name__,
            )
        raise _worker_error(
            f"Kimi output was not JSON: {direct_error}",
            retryable=False,
            submission_state=LLMSubmissionState.SUBMITTED,
        ) from direct_error
    if not isinstance(value, dict):
        if output_recovery == "warn":
            return {}, structured_metadata(
                severity="major",
                warnings=["Kimi JSON output was not an object; using local recovery."],
                raw_text=text,
                strategy="natural_language_fallback",
                provider_error_type=type(value).__name__,
            )
        raise _worker_error(
            "Kimi JSON output was not an object",
            retryable=False,
            submission_state=LLMSubmissionState.SUBMITTED,
        )
    return value, None


def _warn_experimental_once() -> None:
    global _WARNING_EMITTED
    with _WARNING_LOCK:
        if _WARNING_EMITTED:
            return
        _WARNING_EMITTED = True
    warnings.warn(EXPERIMENTAL_WARNING, RuntimeWarning, stacklevel=3)


__all__ = ["EXPERIMENTAL_WARNING", "KimiCodeCliProvider"]
