from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .jobs import (
    JobCancelled,
    JobPaths,
    TERMINAL_STATUSES,
    _process_record,
    acquire_worker_lock,
    append_event,
    find_job_paths,
    finish_job,
    is_cancel_requested,
    persist_result,
    read_job,
    read_json,
    record_progress,
    release_worker_lock,
    restored_environment,
    set_error,
    start_running,
    open_private_binary,
    update_status,
    validate_arc_argv,
)


MAX_PROGRESS_LINE_BYTES = 256 * 1024
MAX_PROGRESS_FILE_BYTES = 16 * 1024 * 1024
_SIGNAL_CANCEL_REQUESTED = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one persisted ARC CLI job")
    parser.add_argument("job_id")
    args = parser.parse_args(argv)
    previous_handlers: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _request_signal_cancel)
    try:
        return run_job(args.job_id)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def run_job(job_id: str) -> int:
    if not acquire_worker_lock(job_id):
        return 0
    paths = find_job_paths(job_id) or JobPaths.for_job(job_id)
    try:
        current = read_json(paths.status, {})
        if isinstance(current, dict) and current.get("status") in TERMINAL_STATUSES:
            return 0
        start_running(job_id)
        if is_cancel_requested(job_id):
            raise JobCancelled("ARC job cancellation was requested before command launch.")
        job = read_job(job_id)
        argv = job.get("argv")
        if not isinstance(argv, list):
            raise ValueError("persisted process job does not contain argv")
        normalized_argv, command = validate_arc_argv(argv)
        cwd = _validated_cwd(job.get("cwd"))
        persisted_command = job.get("command")
        if persisted_command and Path(str(persisted_command)).resolve() != Path(command).resolve():
            raise ValueError("persisted command no longer resolves to the active Python runtime")
        result, exit_code = _run_command(
            job_id,
            normalized_argv,
            command,
            cwd=cwd,
            paths=paths,
            environment=restored_environment(job.get("environment")),
        )
        persist_result(job_id, result, paths=paths)
        output = result.get("output") if isinstance(result, dict) else None
        reported_status = output.get("status") if isinstance(output, dict) else None
        if reported_status == "cancelled":
            set_error(
                job_id,
                "job_cancelled",
                "ARC CLI reported cancellation.",
                cancelled=True,
                paths=paths,
            )
            return 1
        if reported_status == "failed":
            set_error(
                job_id,
                "job_command_reported_failure",
                "ARC CLI reported a failed terminal status.",
                details={"exit_code": exit_code},
                paths=paths,
            )
            return 1
        if exit_code != 0:
            set_error(
                job_id,
                "job_command_failed",
                f"ARC CLI exited with status {exit_code}.",
                details={"exit_code": exit_code, "stderr_tail": _tail_text(paths.stderr)},
                paths=paths,
            )
            return 1
        if reported_status in {"done", "completed", "degraded", "stopped", "needs_llm"}:
            finish_job(job_id, result, str(reported_status))
            return 0
        if isinstance(output, dict) and output.get("ok") is False:
            if output.get("status") == "needs_llm":
                finish_job(job_id, result, "needs_llm")
                return 0
            set_error(
                job_id,
                "job_command_reported_failure",
                "ARC CLI reported an unsuccessful JSON result.",
                details={"exit_code": exit_code},
                paths=paths,
            )
            return 1
        status = "needs_llm" if isinstance(output, dict) and output.get("status") == "needs_llm" else "done"
        finish_job(job_id, result, status)
        return 0
    except JobCancelled as exc:
        set_error(job_id, "job_cancelled", str(exc), cancelled=True, paths=paths)
        return 0
    except Exception as exc:
        set_error(job_id, "job_failed", str(exc), paths=paths)
        return 1
    finally:
        release_worker_lock(job_id)


def _run_command(
    job_id: str,
    argv: list[str],
    command: str,
    *,
    cwd: str,
    paths: JobPaths,
    environment: dict[str, str],
) -> tuple[dict[str, Any], int]:
    launch_argv = [command, *argv[1:]]
    paths.progress_sidechannel.unlink(missing_ok=True)
    environment = dict(environment)
    environment["ARC_JOB_PROGRESS_FILE"] = str(paths.progress_sidechannel)
    progress_offset, progress_buffer = 0, b""
    with open_private_binary(paths.stdout) as stdout, open_private_binary(paths.stderr) as stderr:
        process = subprocess.Popen(
            launch_argv,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
            shell=False,
            cwd=cwd,
            env=environment,
        )
        watchdog = _start_process_watchdog(process)
        update_status(
            job_id,
            paths=paths,
            phase="command_running",
            process={**_process_record(process.pid), "argv": argv, "command": command},
        )
        append_event(job_id, {"event": "command_started", "pid": process.pid}, paths=paths)
        try:
            while process.poll() is None:
                progress_offset, progress_buffer = _drain_progress(
                    job_id, paths, offset=progress_offset, buffer=progress_buffer
                )
                if is_cancel_requested(job_id) or _SIGNAL_CANCEL_REQUESTED:
                    raise JobCancelled(
                        "ARC job cancellation was requested; command was terminated."
                    )
                time.sleep(0.1)
            exit_code = int(process.returncode or 0)
            # A successful CLI leader can still leave helpers in its process
            # group.  Clear the group before finalizing the persisted job.
            _terminate_process(process)
            _drain_progress(
                job_id, paths, offset=progress_offset, buffer=progress_buffer, final=True
            )
        except BaseException:
            _terminate_process(process)
            raise
        finally:
            _stop_process_watchdog(watchdog)

    output = _read_json_output(paths.stdout)
    result: dict[str, Any] = {
        "argv": argv,
        "command": command,
        "cwd": cwd,
        "exit_code": exit_code,
        "stdout_path": str(paths.stdout),
        "stderr_path": str(paths.stderr),
        "stdout_bytes": _file_size(paths.stdout),
        "stderr_bytes": _file_size(paths.stderr),
    }
    if output is not None:
        result["output"] = output
    append_event(job_id, {"event": "command_finished", "exit_code": exit_code}, paths=paths)
    return result, exit_code


def _request_signal_cancel(signum: int, frame: Any) -> None:
    del signum, frame
    global _SIGNAL_CANCEL_REQUESTED
    _SIGNAL_CANCEL_REQUESTED = True


def _drain_progress(
    job_id: str,
    paths: JobPaths,
    *,
    offset: int,
    buffer: bytes,
    final: bool = False,
) -> tuple[int, bytes]:
    try:
        size = paths.progress_sidechannel.stat().st_size
    except OSError:
        return offset, buffer
    if size > MAX_PROGRESS_FILE_BYTES or size < offset:
        raise ValueError("ARC job progress side-channel is invalid or too large")
    with paths.progress_sidechannel.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
    offset += len(chunk)
    lines = (buffer + chunk).split(b"\n")
    buffer = lines.pop()
    if final and buffer:
        lines.append(buffer)
        buffer = b""
    for raw in lines:
        if not raw.strip():
            continue
        if len(raw) > MAX_PROGRESS_LINE_BYTES:
            raise ValueError("ARC job progress event exceeded its size limit")
        try:
            event = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("ARC job progress event is not valid JSON") from exc
        record_progress(job_id, _validated_progress_event(event))
    return offset, buffer


def _validated_progress_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("ARC job progress event must be an object")
    event = value.get("event")
    if not isinstance(event, str) or not event or len(event) > 128:
        raise ValueError("ARC job progress event requires a short event name")
    schema_version = value.get("schema_version")
    if schema_version not in {None, "arc.llm.proposers_reviewer.progress.v1"}:
        raise ValueError("ARC job progress event has an unsupported schema_version")
    forbidden = {"job_id", "status", "environment", "argv", "command"}
    if forbidden & value.keys():
        raise ValueError("ARC job progress event contains reserved fields")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("ARC job progress event must contain JSON values") from exc
    if len(encoded.encode("utf-8")) > MAX_PROGRESS_LINE_BYTES:
        raise ValueError("ARC job progress event exceeded its size limit")
    sanitized = dict(value)
    sanitized.pop("schema_version", None)
    sanitized.pop("updated_at", None)
    return sanitized


def _validated_cwd(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("persisted process job does not contain cwd")
    try:
        cwd = Path(value).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"persisted job cwd does not exist: {value}") from exc
    if not cwd.is_dir():
        raise ValueError(f"persisted job cwd is not a directory: {cwd}")
    return str(cwd)


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    process.poll()
    if os.name != "posix" and process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except OSError:
        pass
    if _wait_for_process_group_exit(process, timeout=0.25):
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except OSError:
        pass
    _wait_for_process_group_exit(process, timeout=0.25)


def _wait_for_process_group_exit(process: subprocess.Popen[Any], *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        process.poll()
        if os.name != "posix":
            if process.returncode is not None:
                return True
        else:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _start_process_watchdog(process: subprocess.Popen[Any]) -> subprocess.Popen[Any] | None:
    if os.name != "posix":
        return None
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "arc_jobs.process_watchdog",
                str(os.getpid()),
                str(process.pid),
                "0.25",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        _terminate_process(process)
        raise


def _stop_process_watchdog(watchdog: subprocess.Popen[Any] | None) -> None:
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


def _read_json_output(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> Any:
    try:
        if path.stat().st_size > max_bytes:
            return None
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            return None
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _tail_text(path: Path, *, limit: int = 4096) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
