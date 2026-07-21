from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import queue
import threading
from collections.abc import Callable, Mapping, Sequence

from .base import (
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerCancelled,
    LLMWorkerError,
    LLMWorkerTimeout,
)
from .activity import ActivityTracker

def run_streaming_process_group(
    command: Sequence[str],
    *,
    input_text: str,
    env: Mapping[str, str],
    activity: ActivityTracker,
    stdout_line_callback: Callable[[str], None],
    cancel_check: Callable[[], bool] | None = None,
    poll_interval_seconds: float = 0.1,
    terminate_grace_seconds: float = 0.5,
) -> subprocess.CompletedProcess[str]:
    """Run a JSONL-style provider while enforcing activity-based timeout."""
    kwargs: dict[str, object] = {"start_new_session": True}
    if os.name == "nt":
        kwargs = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    try:
        process = subprocess.Popen(
            list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=dict(env), **kwargs,
        )
    except FileNotFoundError as exc:
        raise LLMWorkerError(
            f"LLM provider binary not found: {command[0]}", retryable=False,
            category=LLMFailureCategory.INVALID_REQUEST,
            submission_state=LLMSubmissionState.NOT_SUBMITTED,
        ) from exc

    watchdog = start_process_group_watchdog(process, grace_seconds=terminate_grace_seconds)
    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def read_stream(name: str, stream: object) -> None:
        try:
            for line in stream:  # type: ignore[union-attr]
                output_queue.put((name, line))
        finally:
            output_queue.put((name, None))

    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    readers = [
        threading.Thread(target=read_stream, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=read_stream, args=("stderr", process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    input_submitted = threading.Event()
    def write_input() -> None:
        try:
            process.stdin.write(input_text)
            process.stdin.flush()
            activity.submitted()
            input_submitted.set()
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    writer = threading.Thread(target=write_input, name="arc-provider-stdin", daemon=True)
    try:
        transport_started = time.monotonic()
        writer.start()
        closed: set[str] = set()
        while process.poll() is None or len(closed) < 2:
            if (
                not input_submitted.is_set()
                and time.monotonic() - transport_started >= activity.idle_timeout_seconds
            ):
                raise LLMWorkerTimeout(
                    f"{activity.provider} produced no meaningful output while delivering input for "
                    f"{activity.idle_timeout_seconds:g} seconds",
                    submission_state=LLMSubmissionState.NOT_SUBMITTED,
                )
            if cancel_check is not None and cancel_check():
                raise LLMWorkerCancelled(
                    "LLM worker call was cancelled",
                    submission_state=(
                        LLMSubmissionState.SUBMITTED
                        if input_submitted.is_set()
                        else LLMSubmissionState.NOT_SUBMITTED
                    ),
                )
            activity.check()
            try:
                source, line = output_queue.get(timeout=poll_interval_seconds)
            except queue.Empty:
                continue
            if line is None:
                closed.add(source)
                continue
            if source == "stdout":
                stdout_parts.append(line)
                stdout_line_callback(line)
            else:
                # Diagnostics are captured but never count as meaningful work.
                stderr_parts.append(line)
        terminate_process_group(process, grace_seconds=terminate_grace_seconds)
        return subprocess.CompletedProcess(
            list(command), process.returncode, "".join(stdout_parts), "".join(stderr_parts)
        )
    except BaseException:
        terminate_process_group(process, grace_seconds=terminate_grace_seconds)
        raise
    finally:
        stop_process_group_watchdog(watchdog)
        for reader in readers:
            reader.join(timeout=0.2)
        writer.join(timeout=0.2)


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
