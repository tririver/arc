from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
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
    release_worker_lock,
    set_error,
    start_running,
    open_private_binary,
    update_status,
    validate_arc_argv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one persisted ARC CLI job")
    parser.add_argument("job_id")
    args = parser.parse_args(argv)
    return run_job(args.job_id)


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
        result, exit_code = _run_command(job_id, normalized_argv, command, cwd=cwd, paths=paths)
        persist_result(job_id, result, paths=paths)
        if exit_code != 0:
            set_error(
                job_id,
                "job_command_failed",
                f"ARC CLI exited with status {exit_code}.",
                details={"exit_code": exit_code, "stderr_tail": _tail_text(paths.stderr)},
                paths=paths,
            )
            return 1
        output = result.get("output") if isinstance(result, dict) else None
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
) -> tuple[dict[str, Any], int]:
    launch_argv = [command, *argv[1:]]
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
        )
        update_status(
            job_id,
            paths=paths,
            phase="command_running",
            process={**_process_record(process.pid), "argv": argv, "command": command},
        )
        append_event(job_id, {"event": "command_started", "pid": process.pid}, paths=paths)
        while process.poll() is None:
            if is_cancel_requested(job_id):
                _terminate_process(process)
                raise JobCancelled("ARC job cancellation was requested; command was terminated.")
            time.sleep(0.1)
        exit_code = int(process.returncode or 0)

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
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
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
