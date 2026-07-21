from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence

from .base import LLMFailureCategory, LLMSubmissionState, LLMWorkerCancelled, LLMWorkerError, LLMWorkerTimeout


_PROVIDER_TIMEOUT_KEYS = {
    "codex-cli": "ARC_CODEX_TIMEOUT_SECONDS",
    "claude-cli": "ARC_CLAUDE_TIMEOUT_SECONDS",
    "kimi-code-cli": "ARC_KIMI_TIMEOUT_SECONDS",
}

DEFAULT_WORKER_CALL_TIMEOUT_SECONDS = 60 * 60


def resolve_worker_call_timeout_seconds(
    explicit: float | int | None, *, env: Mapping[str, str] | None, provider: str | None = None
) -> float:
    if explicit is not None:
        return _positive_timeout(explicit, "worker_call_timeout_seconds")
    material = os.environ if env is None else env
    provider_key = _PROVIDER_TIMEOUT_KEYS.get(provider or "")
    if provider_key and material.get(provider_key) not in {None, ""}:
        return _positive_timeout(material[provider_key], provider_key)
    if material.get("ARC_LLM_TIMEOUT_SECONDS") not in {None, ""}:
        return _positive_timeout(material["ARC_LLM_TIMEOUT_SECONDS"], "ARC_LLM_TIMEOUT_SECONDS")
    return float(DEFAULT_WORKER_CALL_TIMEOUT_SECONDS)


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
        raise LLMWorkerError(
            f"LLM provider binary not found: {command[0]}", retryable=False,
            category=LLMFailureCategory.INVALID_REQUEST,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        ) from exc
    watchdog = start_process_group_watchdog(process, grace_seconds=terminate_grace_seconds)
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
            # A provider executable can exit after spawning helpers which keep
            # running in the provider's process group.  Reap those helpers
            # before releasing the call slot or reporting completion.
            terminate_process_group(process, grace_seconds=terminate_grace_seconds)
            return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)
    except BaseException:
        terminate_process_group(process, grace_seconds=terminate_grace_seconds)
        raise
    finally:
        stop_process_group_watchdog(watchdog)


def start_process_group_watchdog(
    process: subprocess.Popen[object], *, grace_seconds: float = 0.5
) -> subprocess.Popen[object] | None:
    """Start a detached POSIX watchdog which survives the caller's SIGKILL."""
    if os.name != "posix":
        # Windows callers retain CREATE_NEW_PROCESS_GROUP cleanup.  A future
        # Job Object implementation can replace this interface without
        # changing providers.
        return None
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "arc_llm.providers.process_watchdog",
                str(os.getpid()),
                str(process.pid),
                str(grace_seconds),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        terminate_process_group(process, grace_seconds=grace_seconds)
        raise LLMWorkerError(
            f"Could not start provider process watchdog: {exc}", retryable=False,
            category=LLMFailureCategory.LOCAL_IO,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        ) from exc


def stop_process_group_watchdog(watchdog: subprocess.Popen[object] | None) -> None:
    if watchdog is None:
        return
    try:
        watchdog.wait(timeout=0.5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        watchdog.terminate()
        watchdog.wait(timeout=0.2)
    except (OSError, subprocess.TimeoutExpired):
        try:
            watchdog.kill()
        except OSError:
            pass


def terminate_process_group(process: subprocess.Popen[object], *, grace_seconds: float = 0.5) -> None:
    # Do not use ``process.poll()`` as an early return on POSIX.  The group
    # leader may have exited while descendants (including a paid provider
    # helper) remain alive in the same process group.
    process.poll()
    if os.name != "posix" and process.returncode is not None:
        return
    _signal_group(process, force=False)
    if _wait_for_process_group_exit(process, timeout=grace_seconds):
        return
    _signal_group(process, force=True)
    _wait_for_process_group_exit(process, timeout=grace_seconds)


def _wait_for_process_group_exit(process: subprocess.Popen[object], *, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        process.poll()
        if os.name != "posix":
            if process.returncode is not None:
                return True
        elif not _process_group_exists(process.pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(process: subprocess.Popen[object], *, force: bool) -> None:
    try:
        if os.name != "posix":
            process.kill() if force else process.terminate()
        else:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        if process.poll() is not None:
            return
        try:
            process.kill() if force else process.terminate()
        except OSError:
            pass


def _positive_timeout(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMWorkerError(
            f"{name} must be a positive number", retryable=False,
            category=LLMFailureCategory.INVALID_REQUEST,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        ) from exc
    if result <= 0:
        raise LLMWorkerError(
            f"{name} must be a positive number", retryable=False,
            category=LLMFailureCategory.INVALID_REQUEST,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        )
    return result
