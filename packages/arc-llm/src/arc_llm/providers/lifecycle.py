from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence

from .base import LLMWorkerCancelled, LLMWorkerError, LLMWorkerTimeout


_PROVIDER_TIMEOUT_KEYS = {
    "codex-cli": "ARC_CODEX_TIMEOUT_SECONDS",
    "claude-cli": "ARC_CLAUDE_TIMEOUT_SECONDS",
    "kimi-code-cli": "ARC_KIMI_TIMEOUT_SECONDS",
}


def resolve_worker_call_timeout_seconds(
    explicit: float | int | None, *, env: Mapping[str, str] | None, provider: str | None = None
) -> float | None:
    if explicit is not None:
        return _positive_timeout(explicit, "worker_call_timeout_seconds")
    material = os.environ if env is None else env
    provider_key = _PROVIDER_TIMEOUT_KEYS.get(provider or "")
    if provider_key and material.get(provider_key) not in {None, ""}:
        return _positive_timeout(material[provider_key], provider_key)
    if material.get("ARC_LLM_TIMEOUT_SECONDS") not in {None, ""}:
        return _positive_timeout(material["ARC_LLM_TIMEOUT_SECONDS"], "ARC_LLM_TIMEOUT_SECONDS")
    return None


def remaining_seconds(deadline: float | None) -> float:
    return float("inf") if deadline is None else max(0.0, deadline - time.monotonic())


def check_lifecycle(deadline: float | None, cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise LLMWorkerCancelled("LLM worker call was cancelled")
    if deadline is not None and remaining_seconds(deadline) <= 0:
        raise LLMWorkerTimeout("LLM worker call timed out")


def run_process_group(
    command: Sequence[str], *, input_text: str, env: Mapping[str, str], deadline: float | None,
    cancel_check: Callable[[], bool] | None = None, poll_interval_seconds: float = 0.1,
    terminate_grace_seconds: float = 0.5,
) -> subprocess.CompletedProcess[str]:
    check_lifecycle(deadline, cancel_check)
    kwargs: dict[str, object] = {"start_new_session": True}
    if os.name == "nt":
        kwargs = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    try:
        process = subprocess.Popen(
            list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=dict(env), **kwargs,
        )
    except FileNotFoundError as exc:
        raise LLMWorkerError(f"LLM provider binary not found: {command[0]}", retryable=False) from exc
    pending_input: str | None = input_text
    try:
        while True:
            try:
                check_lifecycle(deadline, cancel_check)
            except (LLMWorkerCancelled, LLMWorkerTimeout):
                terminate_process_group(process, grace_seconds=terminate_grace_seconds)
                raise
            try:
                stdout, stderr = process.communicate(
                    input=pending_input,
                    timeout=min(poll_interval_seconds, remaining_seconds(deadline)),
                )
            except subprocess.TimeoutExpired:
                # Popen retains partially written stdin and captured output;
                # subsequent communicate calls continue both operations.
                pending_input = None
                continue
            return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)
    except BaseException:
        if process.poll() is None:
            terminate_process_group(process, grace_seconds=terminate_grace_seconds)
        raise


def terminate_process_group(process: subprocess.Popen[object], *, grace_seconds: float = 0.5) -> None:
    if process.poll() is not None:
        return
    _signal_group(process, force=False)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_group(process, force=True)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _signal_group(process: subprocess.Popen[object], *, force: bool) -> None:
    try:
        if os.name == "nt":
            process.kill() if force else process.terminate()
        else:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill() if force else process.terminate()
        except OSError:
            pass


def _positive_timeout(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMWorkerError(f"{name} must be a positive number", retryable=False) from exc
    if result <= 0:
        raise LLMWorkerError(f"{name} must be a positive number", retryable=False)
    return result
