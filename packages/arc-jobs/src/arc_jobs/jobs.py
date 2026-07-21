from __future__ import annotations

import json
import os
import re
import signal
import shlex
import sqlite3
import stat
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock, get_ident
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]
JobRunner = Callable[[ProgressCallback, CancelCheck], Any]
StatusResolver = Callable[[Any], str]

ALLOWED_COMMANDS = frozenset(
    {
        "arc-paper",
        "arc-domain",
        "arc-llm",
        "arc-typeset",
        "arc-companion",
    }
)
SUCCESS_TERMINAL_STATUSES = frozenset(
    {
        "done",
        "completed",
        "degraded",
        "stopped",
        "needs_llm",
        "first_chapter_ready",
        "needs_supervision",
    }
)
FAILURE_TERMINAL_STATUSES = frozenset({"failed", "cancelled"})
TERMINAL_STATUSES = SUCCESS_TERMINAL_STATUSES | FAILURE_TERMINAL_STATUSES
MAX_WORKER_LAUNCH_ATTEMPTS = 3
MAX_EVENT_TAIL_BYTES = 1024 * 1024
WORKER_LAUNCH_GRACE_SECONDS = 1.0
RESERVED_STATUS_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "job_id",
        "job_type",
        "status",
        "phase",
        "progress",
        "payload",
        "argv",
        "command",
        "cancel_requested",
        "started_at",
        "updated_at",
        "finished_at",
        "worker",
        "process",
        "error",
        "error_path",
        "result_path",
        "execution_mode",
        "ok",
        "errors",
        "meta",
        "environment",
        "review_sequence",
        "last_activity_at",
        "last_substantive_excerpt",
        "last_substantive_at",
        "artifact_paths",
    }
)
_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_STATUS_UPDATE_LOCK = RLock()


class JobCancelled(RuntimeError):
    """Raised when a cooperative job notices a cancellation request."""


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    job_dir: Path
    job: Path
    status: Path
    events: Path
    result: Path
    error: Path
    heartbeat: Path
    worker_process: Path
    worker_lock: Path
    recovery_lock: Path
    cancel_request: Path
    stdout: Path
    stderr: Path
    worker_stdout: Path
    worker_stderr: Path
    progress_sidechannel: Path

    @classmethod
    def for_job(cls, job_id: str, *, root: Path | None = None) -> "JobPaths":
        safe_id = safe_job_id(job_id)
        job_dir = (root or jobs_root()) / safe_id
        return cls(
            job_id=safe_id,
            job_dir=job_dir,
            job=job_dir / "job.json",
            status=job_dir / "status.json",
            events=job_dir / "events.jsonl",
            result=job_dir / "result.json",
            error=job_dir / "error.json",
            heartbeat=job_dir / "heartbeat.json",
            worker_process=job_dir / "worker.json",
            worker_lock=job_dir / "worker.lock",
            recovery_lock=job_dir / "recovery.lock",
            cancel_request=job_dir / "cancel.request",
            stdout=job_dir / "stdout.log",
            stderr=job_dir / "stderr.log",
            worker_stdout=job_dir / "worker.stdout.log",
            worker_stderr=job_dir / "worker.stderr.log",
            progress_sidechannel=job_dir / "progress.jsonl",
        )


class JobManager:
    """Persist and execute ARC CLI jobs independently of any agent protocol."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        event_limit: int = 100,
        worker_mode: str | None = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="arc-job")
        self._event_limit = event_limit
        self._worker_mode = worker_mode
        self._futures: dict[str, Future[Any]] = {}
        self._lock = Lock()

    def start(
        self,
        *,
        job_type: str,
        payload: Mapping[str, Any] | None,
        runner: JobRunner | None = None,
        status_resolver: StatusResolver | None = None,
        argv: Sequence[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        """Create a job, using a thread runner or an allowlisted CLI argv."""
        payload_dict = dict(payload or {})
        _validate_status_payload(payload_dict)
        normalized_argv: list[str] | None = None
        command: str | None = None
        if argv is not None:
            normalized_argv, command = validate_arc_argv(argv)
        normalized_cwd = _normalize_cwd(cwd)
        environment_snapshot = snapshot_environment(overrides=environment)

        use_thread_worker = self._use_thread_worker() and runner is not None
        if not use_thread_worker and normalized_argv is None:
            raise ValueError("process jobs require an allowlisted ARC CLI argv")

        job_id = uuid4().hex
        paths = JobPaths.for_job(job_id)
        now = now_iso()
        job: dict[str, Any] = {
            "schema_version": "arc.job.v1",
            "job_id": job_id,
            "job_type": job_type,
            "payload": payload_dict,
            "created_at": now,
            "execution_mode": "thread" if use_thread_worker else "process",
            "environment": environment_snapshot,
        }
        if normalized_argv is not None:
            job["argv"] = normalized_argv
            job["command"] = command
            job["cwd"] = normalized_cwd
        _ensure_private_dir(paths.job_dir.parent)
        paths.job_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        write_json(paths.job, job)
        status: dict[str, Any] = {
            "schema_version": "arc.job_status.v1",
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "phase": "queued",
            "progress": {},
            "review_sequence": 0,
            "last_activity_at": None,
            "last_substantive_excerpt": None,
            "last_substantive_at": None,
            "artifact_paths": [],
            "payload": payload_dict,
            "cancel_requested": False,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "execution_mode": "thread" if use_thread_worker else "process",
            **payload_dict,
        }
        if normalized_argv is not None:
            status["argv"] = normalized_argv
            status["command"] = command
            status["cwd"] = normalized_cwd
        write_json(paths.status, status)
        append_event(job_id, {"event": "job_queued"})

        if use_thread_worker:
            assert runner is not None
            future = self._executor.submit(
                self._run_thread_worker,
                job_id,
                runner,
                status_resolver or _default_status,
            )
            with self._lock:
                self._futures[job_id] = future
        else:
            self._launch_worker(job_id)
        return job_id

    def submit(
        self,
        argv: Sequence[str],
        *,
        job_type: str = "cli",
        payload: Mapping[str, Any] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        return self.start(
            job_type=job_type, payload=payload, argv=argv, cwd=cwd, environment=environment
        )

    def wait(self, job_id: str, *, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            status = self.status(job_id)
            if status.get("status") in TERMINAL_STATUSES or status.get("status") == "job_unknown":
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.1, remaining))

    def status(self, job_id: str) -> dict[str, Any]:
        paths = find_job_paths(job_id)
        status = read_json(paths.status) if paths is not None else None
        if not isinstance(status, dict):
            return unknown_job(job_id)
        self._reconcile(paths, status)
        status = read_json(paths.status, status)
        if paths.cancel_request.exists() and status.get("status") not in TERMINAL_STATUSES:
            status["cancel_requested"] = True
        status["ok"] = True
        status.setdefault("errors", [])
        status.setdefault("meta", {})
        status["events"] = tail_events(paths.events, limit=self._event_limit)
        status["eta"] = estimate_eta(status)
        return status

    def result(self, job_id: str) -> dict[str, Any]:
        status = self.status(job_id)
        if status.get("status") == "job_unknown":
            return status
        paths = find_job_paths(job_id)
        assert paths is not None
        stored = read_json(paths.result)
        result = _unwrap_result(stored)
        if status.get("status") in SUCCESS_TERMINAL_STATUSES:
            return {
                "ok": True,
                "status": status["status"],
                "job_id": job_id,
                "job_type": status.get("job_type"),
                "result": result,
                "meta": {"job": job_meta(status, paths=paths)},
            }
        if status.get("status") in {"failed", "cancelled", "cancel_requested"}:
            error = read_json(paths.error) or status.get("error")
            response = {
                "ok": False,
                "status": status.get("status"),
                "job_id": job_id,
                "job_type": status.get("job_type"),
                "error": error
                or {
                    "code": str(status.get("status")),
                    "message": f"ARC job {status.get('status')}",
                },
                "errors": [],
                "meta": {"job": job_meta(status, paths=paths)},
            }
            if stored is not None:
                response["result"] = result
            return response
        return {
            "ok": False,
            "status": status.get("status", "running"),
            "job_id": job_id,
            "job_type": status.get("job_type"),
            "message": "ARC job is not complete yet.",
            "next": {
                "cli_command": arc_jobs_cli_command("status", job_id, "--json"),
                "poll_after_seconds": 5,
            },
            "meta": {"job": job_meta(status, paths=paths)},
        }

    def list_jobs(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for root in readable_job_roots():
            if not root.exists():
                continue
            for path in root.iterdir():
                if not path.is_dir() or path.name in seen:
                    continue
                try:
                    status = self.status(path.name)
                except ValueError:
                    continue
                if status.get("status") != "job_unknown":
                    items.append(status)
                    seen.add(path.name)
        items.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
        return {"ok": True, "data": {"jobs": items}, "errors": [], "meta": {}}

    def cancel(self, job_id: str) -> dict[str, Any]:
        paths = find_job_paths(job_id)
        if paths is None or not paths.status.exists():
            return unknown_job(job_id)
        current = read_json(paths.status, {})
        if isinstance(current, dict) and current.get("status") in TERMINAL_STATUSES:
            return self.status(job_id)
        # Publish the status before the marker. Workers act on the marker, so they
        # cannot finalize cancellation before the request status is persisted.
        update_status(
            job_id,
            paths=paths,
            status="cancel_requested",
            phase="cancel_requested",
            cancel_requested=True,
            error={
                "code": "job_cancel_requested",
                "message": "Cancellation was requested; the running command is being terminated.",
            },
        )
        write_private_text(paths.cancel_request, now_iso())
        append_event(job_id, {"event": "job_cancel_requested"}, paths=paths)
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None and future.cancel():
            set_error(job_id, "job_cancelled", "ARC job was cancelled before it started.", cancelled=True, paths=paths)
        return self.status(job_id)

    def _use_thread_worker(self) -> bool:
        mode = self._worker_mode or os.environ.get("ARC_JOBS_WORKER_MODE", "process")
        return mode == "thread"

    def _launch_worker(self, job_id: str) -> None:
        paths = find_job_paths(job_id) or JobPaths.for_job(job_id)
        current = read_json(paths.status, {})
        if not isinstance(current, dict) or current.get("status") in TERMINAL_STATUSES:
            return
        attempts = int(current.get("worker_launch_attempts") or 0) + 1
        update_status(
            job_id,
            paths=paths,
            phase="worker_launching",
            worker_launch_attempts=attempts,
        )
        command = [sys.executable, "-m", "arc_jobs.worker", job_id]
        job = read_json(paths.job, {})
        snapshot = job.get("environment") if isinstance(job, Mapping) else None
        try:
            worker_environment = restored_environment(snapshot)
        except ValueError as exc:
            set_error(job_id, "job_environment_invalid", str(exc), paths=paths)
            return
        stdout = open_private_binary(paths.worker_stdout, append=True)
        stderr = open_private_binary(paths.worker_stderr, append=True)
        try:
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    close_fds=True,
                    env=worker_environment,
                )
            except Exception as exc:
                set_error(job_id, "job_worker_launch_failed", f"Could not launch ARC job worker: {exc}")
                return
        finally:
            stdout.close()
            stderr.close()
        write_json(
            paths.worker_process,
            {**_process_record(process.pid), "command": command, "launched_at": now_iso()},
        )

    def _reconcile(self, paths: JobPaths, status: Mapping[str, Any]) -> None:
        """Recover safe pre-command failures and terminalize orphaned commands."""
        if status.get("status") in TERMINAL_STATUSES:
            return
        execution_mode = str(
            status.get("execution_mode")
            or ("process" if isinstance(status.get("argv"), list) else "thread")
        )
        worker = status.get("worker")
        if isinstance(worker, Mapping) and _pid_record_alive(worker):
            return
        launched_worker = read_json(paths.worker_process, {})
        if isinstance(launched_worker, Mapping) and _pid_record_alive(launched_worker):
            return
        lock_owner = read_json(paths.worker_lock, {})
        if isinstance(lock_owner, Mapping) and _pid_record_alive(lock_owner):
            return
        latest = read_json(paths.status, {})
        if isinstance(latest, Mapping):
            if latest.get("status") in TERMINAL_STATUSES:
                return
            status = latest
        process = status.get("process")
        process_started = isinstance(process, Mapping) and isinstance(process.get("pid"), int)
        process_alive = process_started and (
            _pid_record_alive(process) or _recorded_process_group_alive(process)
        )
        if paths.cancel_request.exists() or status.get("status") == "cancel_requested":
            if process_alive:
                _terminate_recorded_process(process)
            set_error(
                paths.job_id,
                "job_cancelled",
                "ARC job was cancelled after its worker stopped.",
                cancelled=True,
                paths=paths,
            )
            return
        if process_started:
            termination_attempted = bool(
                process_alive and _terminate_recorded_process(process)
            )
            set_error(
                paths.job_id,
                "job_worker_lost",
                "ARC job worker exited after launching the command; the command cannot be safely resumed.",
                details={
                    "orphaned_process_alive": bool(process_alive),
                    "termination_attempted": termination_attempted,
                    "process": dict(process),
                },
                paths=paths,
            )
            return
        if execution_mode != "process":
            # A live in-process executor may still have this queued. Once it has
            # recorded a worker, loss is terminal because its callable is not persisted.
            if isinstance(worker, Mapping):
                set_error(
                    paths.job_id,
                    "job_worker_lost",
                    "In-process ARC job worker exited before completing the job.",
                    paths=paths,
                )
            return
        attempts = int(status.get("worker_launch_attempts") or 0)
        if attempts == 0 and not paths.worker_process.exists():
            updated = _parse_time(status.get("updated_at") or status.get("started_at"))
            if updated is not None and time.time() - updated < WORKER_LAUNCH_GRACE_SECONDS:
                return
        if attempts >= MAX_WORKER_LAUNCH_ATTEMPTS:
            set_error(
                paths.job_id,
                "job_worker_unavailable",
                f"ARC job worker failed to start after {attempts} attempts.",
                paths=paths,
            )
            return
        if not acquire_recovery_lock(paths.job_id):
            return
        try:
            latest = read_json(paths.status, {})
            if not isinstance(latest, dict) or latest.get("status") in TERMINAL_STATUSES:
                return
            latest_worker = latest.get("worker")
            if isinstance(latest_worker, Mapping) and _pid_record_alive(latest_worker):
                return
            append_event(paths.job_id, {"event": "job_worker_restarting"}, paths=paths)
            self._launch_worker(paths.job_id)
        finally:
            release_recovery_lock(paths.job_id)

    def _run_thread_worker(self, job_id: str, runner: JobRunner, status_resolver: StatusResolver) -> None:
        if not acquire_worker_lock(job_id):
            return
        try:
            start_running(job_id)

            def cancel_requested() -> bool:
                return is_cancel_requested(job_id)

            def progress(event: dict[str, Any]) -> None:
                if cancel_requested():
                    raise JobCancelled("ARC job cancellation was requested.")
                record_progress(job_id, event)

            try:
                result = runner(progress, cancel_requested)
                if cancel_requested():
                    raise JobCancelled("ARC job cancellation was requested.")
                finish_job(job_id, result, status_resolver(result))
            except JobCancelled as exc:
                set_error(job_id, "job_cancelled", str(exc), cancelled=True)
            except Exception as exc:
                set_error(job_id, "job_failed", str(exc))
            except BaseException as exc:
                set_error(job_id, "job_worker_lost", f"In-process worker stopped: {exc}")
                raise
        finally:
            release_worker_lock(job_id)


def cache_root() -> Path:
    if value := os.environ.get("ARC_JOBS_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("ARC_HOME"):
        return Path(value).expanduser() / "cache" / "arc-jobs"
    if value := os.environ.get("XDG_CACHE_HOME"):
        return Path(value).expanduser() / "arc" / "arc-jobs"
    return Path.home() / ".cache" / "arc" / "arc-jobs"


def jobs_root() -> Path:
    if value := os.environ.get("ARC_JOBS_DIR"):
        return Path(value).expanduser()
    if value := os.environ.get("ARC_JOBS_CACHE"):
        return Path(value).expanduser() / "jobs"
    if value := os.environ.get("ARC_HOME"):
        return Path(value).expanduser() / "jobs"
    return cache_root() / "jobs"


_SNAPSHOT_ENV_KEYS = frozenset(
    {
        "ARC_HOME",
        "ARC_RUNTIME_HOME",
        "ARC_AGENT_HOST",
        "ARC_PAPER_CACHE",
        "ARC_DOMAIN_CACHE",
        "ARC_LLM_CACHE",
        "ARC_JOBS_CACHE",
        "ARC_JOBS_DIR",
        "ARC_LLM_TMP_DIR",
        "ARC_LLM_SCHEMA_CACHE_DIR",
        "ARC_LLM_IDLE_TIMEOUT_SECONDS",
        "ARC_LLM_MAX_CONCURRENCY",
        "ARC_CODEX_IDLE_TIMEOUT_SECONDS",
        "ARC_CODEX_MAX_CONCURRENCY",
        "ARC_CODEX_REASONING_EFFORT",
        "ARC_CLAUDE_IDLE_TIMEOUT_SECONDS",
        "ARC_CLAUDE_MAX_CONCURRENCY",
        "ARC_CLAUDE_EFFORT",
        "ARC_KIMI_IDLE_TIMEOUT_SECONDS",
        "ARC_KIMI_MAX_CONCURRENCY",
        "ARC_KIMI_ALLOW_INTERNAL_RETRIES",
        "ARC_KIMI_WORK_DIR",
        "ARC_CODEX_BIN",
        "ARC_CLAUDE_BIN",
        "ARC_KIMI_BIN",
        "KIMI_CODE_HOME",
        "ARC_LLM_CODEX_LOW_MODEL",
        "ARC_LLM_CODEX_MEDIUM_MODEL",
        "ARC_LLM_CODEX_HIGH_MODEL",
        "ARC_LLM_CODEX_MAX_MODEL",
        "ARC_LLM_CLAUDE_LOW_MODEL",
        "ARC_LLM_CLAUDE_MEDIUM_MODEL",
        "ARC_LLM_CLAUDE_HIGH_MODEL",
        "ARC_LLM_CLAUDE_MAX_MODEL",
        "ARC_LLM_KIMI_LOW_MODEL",
        "ARC_LLM_KIMI_MEDIUM_MODEL",
        "ARC_LLM_KIMI_HIGH_MODEL",
        "ARC_LLM_KIMI_MAX_MODEL",
    }
)
_PROCESS_ENV_KEYS = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "LANG", "TZ", "SYSTEMROOT", "COMSPEC", "PATHEXT"}
)
_SECRET_MARKERS = ("TOKEN", "API_KEY", "APIKEY", "PASSWORD", "SECRET", "CREDENTIAL")
_REMOVED_LLM_TIMEOUT_ENV_KEYS = frozenset(
    {
        "ARC_LLM_TIMEOUT_SECONDS",
        "ARC_CODEX_TIMEOUT_SECONDS",
        "ARC_CLAUDE_TIMEOUT_SECONDS",
        "ARC_KIMI_TIMEOUT_SECONDS",
    }
)


def _reject_removed_llm_timeout_environment(environment: Mapping[str, Any]) -> None:
    present = [
        key
        for key in sorted(_REMOVED_LLM_TIMEOUT_ENV_KEYS)
        if str(environment.get(key) or "").strip()
    ]
    if not present:
        return
    replacements = ", ".join(
        key.replace("_TIMEOUT_SECONDS", "_IDLE_TIMEOUT_SECONDS") for key in present
    )
    raise ValueError(
        "LLM total-timeout environment variables were removed; use idle timeout "
        f"variables instead: {replacements}"
    )


def snapshot_environment(
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Capture only portable, non-secret ARC context for a detached job."""
    source = os.environ if env is None else env
    _reject_removed_llm_timeout_environment(source)
    override_values = dict(overrides or {})
    _reject_removed_llm_timeout_environment(override_values)
    result = {
        key: str(source[key])
        for key in _SNAPSHOT_ENV_KEYS
        if source.get(key) not in {None, ""}
    }
    if "ARC_AGENT_HOST" not in result:
        try:
            from .runtime import detect_agent_host

            host = detect_agent_host(source)
            if host != "unknown":
                result["ARC_AGENT_HOST"] = host
        except (ImportError, RuntimeError):
            pass
    for key, value in override_values.items():
        if key in _REMOVED_LLM_TIMEOUT_ENV_KEYS and isinstance(value, str) and not value:
            continue
        if any(marker in key.upper() for marker in _SECRET_MARKERS):
            raise ValueError(f"job environment must not contain secrets: {key}")
        if key not in _SNAPSHOT_ENV_KEYS:
            raise ValueError(f"job environment key is not allowlisted: {key}")
        if not isinstance(value, str):
            raise ValueError(f"job environment value must be a string: {key}")
        if value:
            result[key] = value
        else:
            result.pop(key, None)
    return dict(sorted(result.items()))


def restored_environment(
    snapshot: Any, *, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build a minimal child environment and reject tampered persisted context."""
    base = os.environ if base is None else base
    result = {
        key: str(base[key])
        for key in _PROCESS_ENV_KEYS
        if base.get(key) not in {None, ""}
    }
    result.update(
        {key: str(value) for key, value in base.items() if key.startswith("LC_") and value}
    )
    if snapshot is None:
        return result
    if not isinstance(snapshot, Mapping):
        raise ValueError("persisted job environment must be an object")
    _reject_removed_llm_timeout_environment(snapshot)
    for key, value in snapshot.items():
        if key not in _SNAPSHOT_ENV_KEYS or not isinstance(value, str):
            raise ValueError(f"persisted job environment is invalid: {key}")
        if any(marker in key.upper() for marker in _SECRET_MARKERS):
            raise ValueError(f"persisted job environment contains a secret: {key}")
        if value:
            result[key] = value
    if result.get("ARC_LLM_TMP_DIR"):
        result["TMPDIR"] = result["ARC_LLM_TMP_DIR"]
    return result


def readable_job_roots() -> list[Path]:
    return [jobs_root()]


def stats_db_path() -> Path:
    if (os.environ.get("ARC_HOME") or os.environ.get("ARC_JOBS_DIR")) and not os.environ.get(
        "ARC_JOBS_CACHE"
    ):
        return jobs_root() / ".stats" / "jobs.sqlite"
    return cache_root() / "stats" / "jobs.sqlite"


def safe_job_id(value: str) -> str:
    candidate = value.strip()
    if not candidate or _JOB_ID_PATTERN.fullmatch(candidate) is None:
        raise ValueError("job_id must contain only letters, digits, '-' or '_'")
    return candidate


def runtime_script_dirs() -> tuple[Path, ...]:
    """Return executable directories belonging to the active Python runtime."""
    candidates = [Path(sys.executable).resolve().parent]
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidates.append(Path(scripts).expanduser().resolve())
    result: list[Path] = []
    for candidate in candidates:
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def arc_jobs_cli_argv(*args: str, executable: str | os.PathLike[str] | None = None) -> list[str]:
    """Return a stable invocation from the active runtime, independent of PATH."""
    name = "arc-jobs.exe" if os.name == "nt" else "arc-jobs"
    # Preserve a virtualenv's symlink path. Resolving it to the base interpreter
    # would lose the adjacent ARC console scripts and the environment site-packages.
    python = Path(executable or sys.executable).expanduser().absolute()
    adjacent = python.with_name(name)
    if adjacent.is_file() and (os.name == "nt" or os.access(adjacent, os.X_OK)):
        return [str(adjacent), *args]
    return [str(python), "-m", "arc_jobs.cli", *args]


def arc_jobs_cli_command(
    *args: str,
    executable: str | os.PathLike[str] | None = None,
) -> str:
    return " ".join(
        shlex.quote(item) for item in arc_jobs_cli_argv(*args, executable=executable)
    )


def validate_arc_argv(argv: Sequence[str]) -> tuple[list[str], str]:
    if isinstance(argv, (str, bytes)):
        raise ValueError("argv must be a sequence of strings, not a shell command string")
    normalized = list(argv)
    if not normalized:
        raise ValueError("argv must include an ARC CLI command")
    if any(not isinstance(arg, str) for arg in normalized):
        raise ValueError("every argv item must be a string")
    if any("\x00" in arg for arg in normalized):
        raise ValueError("argv items must not contain NUL bytes")

    requested = Path(normalized[0]).expanduser()
    requested_name = requested.name
    command_name = requested_name[:-4] if requested_name.lower().endswith(".exe") else requested_name
    if command_name not in ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(ALLOWED_COMMANDS))
        raise ValueError(f"command {command_name!r} is not allowed; expected one of: {allowed}")

    resolved = resolve_arc_command(command_name)
    if requested.parent != Path("."):
        try:
            explicit = requested.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"command path does not exist: {requested}") from exc
        if explicit != resolved:
            raise ValueError("explicit command path is not the allowlisted executable in this Python runtime")
    normalized[0] = command_name
    return normalized, str(resolved)


def _normalize_cwd(value: str | os.PathLike[str] | None) -> str:
    candidate = Path.cwd() if value is None else Path(value).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"job cwd does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise ValueError(f"job cwd is not a directory: {resolved}")
    return str(resolved)


def resolve_arc_command(command_name: str) -> Path:
    if command_name not in ALLOWED_COMMANDS:
        raise ValueError(f"command {command_name!r} is not allowlisted")
    suffixes = (".exe", "") if os.name == "nt" else ("",)
    for directory in runtime_script_dirs():
        directory_resolved = directory.resolve()
        for suffix in suffixes:
            candidate = directory / f"{command_name}{suffix}"
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            if resolved.parent != directory_resolved or not resolved.is_file():
                continue
            if os.name != "nt" and not os.access(resolved, os.X_OK):
                continue
            return resolved
    locations = ", ".join(str(path) for path in runtime_script_dirs())
    raise ValueError(f"{command_name} is not installed in the active Python runtime ({locations})")


def find_job_paths(job_id: str) -> JobPaths | None:
    safe_id = safe_job_id(job_id)
    for root in readable_job_roots():
        paths = JobPaths.for_job(safe_id, root=root)
        if paths.job.exists() or paths.status.exists():
            _secure_job_storage(paths)
            return paths
    return None


def read_job(job_id: str) -> dict[str, Any]:
    paths = find_job_paths(job_id)
    data = read_json(paths.job) if paths is not None else None
    if not isinstance(data, dict):
        raise FileNotFoundError(f"Unknown ARC job: {job_id}")
    return data


def acquire_worker_lock(job_id: str) -> bool:
    paths = find_job_paths(job_id) or JobPaths.for_job(job_id)
    return _acquire_pid_lock(paths.worker_lock, job_id=job_id)


def release_worker_lock(job_id: str) -> None:
    paths = find_job_paths(job_id) or JobPaths.for_job(job_id)
    _release_pid_lock(paths.worker_lock)


def acquire_recovery_lock(job_id: str) -> bool:
    paths = find_job_paths(job_id) or JobPaths.for_job(job_id)
    return _acquire_pid_lock(paths.recovery_lock, job_id=job_id)


def release_recovery_lock(job_id: str) -> None:
    paths = find_job_paths(job_id) or JobPaths.for_job(job_id)
    _release_pid_lock(paths.recovery_lock)


def start_running(job_id: str) -> None:
    now = now_iso()
    paths = find_job_paths(job_id) or JobPaths.for_job(job_id)
    update_status(
        job_id,
        paths=paths,
        status="running",
        phase="running",
        worker=_process_record(os.getpid()),
        worker_started_at=now,
    )
    write_json(paths.heartbeat, {"job_id": job_id, "pid": os.getpid(), "heartbeat_at": now})
    write_json(paths.worker_process, {**_process_record(os.getpid()), "started_at": now})
    append_event(job_id, {"event": "job_started"}, paths=paths)


def record_progress(job_id: str, event: Mapping[str, Any]) -> None:
    if is_cancel_requested(job_id):
        raise JobCancelled("ARC job cancellation was requested.")
    paths = find_job_paths(job_id) or JobPaths.for_job(job_id)
    normalized_event = dict(event)
    normalized_event.pop("schema_version", None)
    normalized_event.pop("updated_at", None)
    provider_review_sequence = normalized_event.pop("review_sequence", None)
    if provider_review_sequence is not None:
        normalized_event["provider_review_sequence"] = provider_review_sequence
    with _STATUS_UPDATE_LOCK:
        event_name = str(normalized_event.get("event") or "")
        if event_name == "review_due":
            current = read_json(paths.status, {})
            current_cursor = current.get("review_sequence", 0) if isinstance(current, dict) else 0
            if not isinstance(current_cursor, int) or isinstance(current_cursor, bool):
                current_cursor = 0
            normalized_event["review_sequence"] = max(0, current_cursor) + 1
        timestamped = append_event(job_id, normalized_event, paths=paths)
        progress = _progress_from_event(timestamped)
        updates: dict[str, Any] = {"phase": event_name or None, "progress": progress}
        status_fields = {
            "round",
            "round_number",
            "phase",
            "run_id",
            "loop_id",
            "worker_id",
            "role",
            "worker_status",
            "active_workers",
            "completed_workers",
            "failed_workers",
            "evidence_round_number",
            "evidence_status",
            "request_count",
            "sections_completed",
            "sections_total",
            "current",
            "title",
            "section_id",
            "chapter_id",
            "segment_id",
            "lane",
            "generation",
            "block_status",
            "warning",
            "warnings",
            "failure_count",
            "review_sequence",
            "provider_review_sequence",
            "last_activity_at",
            "artifact_paths",
        }
        for key, value in timestamped.items():
            if key in status_fields:
                updates[key] = value
        excerpt = _substantive_progress_excerpt(timestamped)
        if timestamped.get("substantive") is True:
            updates["last_activity_at"] = timestamped["at"]
        if excerpt is not None:
            updates["last_substantive_excerpt"] = excerpt
            updates["last_substantive_at"] = timestamped["at"]
        update_status(
            job_id,
            paths=paths,
            **{key: value for key, value in updates.items() if value is not None},
        )
        write_json(
            paths.heartbeat,
            {
                "job_id": job_id,
                "pid": os.getpid(),
                "heartbeat_at": timestamped["at"],
                "phase": event_name,
            },
        )


def _substantive_progress_excerpt(event: Mapping[str, Any]) -> str | None:
    """Return a bounded user-visible progress excerpt, never a heartbeat."""
    if event.get("substantive") is not True:
        return None
    for key in ("summary", "excerpt", "message"):
        value = event.get(key)
        if isinstance(value, str):
            compact = " ".join(value.split())
            if compact:
                return compact[:2000]
    return None


def append_event(
    job_id: str,
    event: Mapping[str, Any],
    *,
    paths: JobPaths | None = None,
) -> dict[str, Any]:
    paths = paths or find_job_paths(job_id) or JobPaths.for_job(job_id)
    timestamped = {
        "schema_version": "arc.job_event.v1",
        "event_id": uuid4().hex,
        "at": now_iso(),
        **dict(event),
    }
    _ensure_private_dir(paths.events.parent)
    fd = os.open(paths.events, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(timestamped, ensure_ascii=False) + "\n")
    return timestamped


def persist_result(job_id: str, result: Any, *, paths: JobPaths | None = None) -> None:
    paths = paths or find_job_paths(job_id) or JobPaths.for_job(job_id)
    job = read_json(paths.job, {})
    write_json(
        paths.result,
        {
            "schema_version": "arc.job_result.v1",
            "job_id": job_id,
            "job_type": job.get("job_type") if isinstance(job, dict) else None,
            "result": result,
            "created_at": now_iso(),
        },
    )


def finish_job(job_id: str, result: Any, status: str) -> None:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"invalid terminal job status: {status}")
    paths = find_job_paths(job_id) or JobPaths.for_job(job_id)
    persist_result(job_id, result, paths=paths)
    finished_at = now_iso()
    update_status(
        job_id,
        paths=paths,
        status=status,
        phase=status,
        finished_at=finished_at,
        result_path=str(paths.result),
    )
    append_event(job_id, {"event": f"job_{status}"}, paths=paths)
    record_history(job_id, status=status, paths=paths)


def set_error(
    job_id: str,
    code: str,
    message: str,
    *,
    cancelled: bool = False,
    details: Mapping[str, Any] | None = None,
    paths: JobPaths | None = None,
) -> None:
    paths = paths or find_job_paths(job_id) or JobPaths.for_job(job_id)
    error = {
        "schema_version": "arc.job_error.v1",
        "code": code,
        "message": message,
        **dict(details or {}),
    }
    write_json(paths.error, error)
    status = "cancelled" if cancelled else "failed"
    update_status(
        job_id,
        paths=paths,
        status=status,
        phase=status,
        finished_at=now_iso(),
        error=error,
        error_path=str(paths.error),
    )
    append_event(job_id, {"event": f"job_{status}", "error_code": code}, paths=paths)
    record_history(job_id, status=status, paths=paths)


def is_cancel_requested(job_id: str) -> bool:
    paths = find_job_paths(job_id)
    return paths is not None and paths.cancel_request.exists()


def update_status(job_id: str, *, paths: JobPaths | None = None, **fields: Any) -> dict[str, Any]:
    with _STATUS_UPDATE_LOCK:
        paths = paths or find_job_paths(job_id) or JobPaths.for_job(job_id)
        status = read_json(paths.status)
        if not isinstance(status, dict):
            status = {"schema_version": "arc.job_status.v1", "job_id": job_id}
        status.update({key: value for key, value in fields.items() if value is not None})
        status["updated_at"] = now_iso()
        write_json(paths.status, status)
        return status


def tail_events(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    lines = _tail_lines(path, limit=limit, max_bytes=MAX_EVENT_TAIL_BYTES)
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def record_history(job_id: str, *, status: str, paths: JobPaths | None = None) -> None:
    paths = paths or find_job_paths(job_id) or JobPaths.for_job(job_id)
    job = read_json(paths.job)
    current = read_json(paths.status)
    if not isinstance(job, dict) or not isinstance(current, dict):
        return
    started = _parse_time(current.get("worker_started_at") or current.get("started_at"))
    finished = _parse_time(current.get("finished_at"))
    if started is None or finished is None:
        return
    argv = job.get("argv") if isinstance(job.get("argv"), list) else []
    command = str(argv[0]) if argv else str(job.get("job_type") or "")
    duration = max(0.0, finished - started)
    db_path = stats_db_path()
    _ensure_private_dir(db_path.parent)
    _ensure_private_file(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS job_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                command TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            INSERT INTO job_history (job_type, command, status, duration_seconds, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(job.get("job_type") or ""), command, status, duration, now_iso()),
        )
    _secure_sqlite_files(db_path)


def estimate_eta(status: Mapping[str, Any]) -> dict[str, Any]:
    if status.get("status") in TERMINAL_STATUSES:
        return {"available": False, "reason": "terminal_status"}
    job_type = str(status.get("job_type") or "")
    argv = status.get("argv") if isinstance(status.get("argv"), list) else []
    command = str(argv[0]) if argv else job_type
    samples = _history_samples(job_type, command)
    if len(samples) < 3:
        return {"available": False, "reason": "insufficient_history", "samples": len(samples)}
    sorted_samples = sorted(samples)
    started = _parse_time(status.get("worker_started_at") or status.get("started_at"))
    elapsed = max(0.0, time.time() - started) if started is not None else 0.0
    low = sorted_samples[0] if len(sorted_samples) < 10 else _quantile(sorted_samples, 0.025)
    high = sorted_samples[-1] if len(sorted_samples) < 10 else _quantile(sorted_samples, 0.975)
    median = _quantile(sorted_samples, 0.5)
    return {
        "available": True,
        "samples": len(sorted_samples),
        "basis": f"history:{job_type}:{command}:{len(sorted_samples)}",
        "total_seconds_p50": round(median, 1),
        "total_seconds_low": round(low, 1),
        "total_seconds_high": round(high, 1),
        "remaining_seconds_p50": round(max(0.0, median - elapsed), 1),
        "remaining_seconds_low": round(max(0.0, low - elapsed), 1),
        "remaining_seconds_high": round(max(0.0, high - elapsed), 1),
    }


def write_json(path: Path, data: Any) -> None:
    _ensure_private_dir(path.parent)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{get_ident()}.{time.time_ns()}.tmp")
    write_private_text(temporary, json.dumps(data, indent=2, ensure_ascii=False), exclusive=True)
    temporary.replace(path)
    _chmod_private_file(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def job_meta(status: Mapping[str, Any], *, paths: JobPaths | None = None) -> dict[str, Any]:
    paths = paths or (
        find_job_paths(str(status.get("job_id") or "")) if status.get("job_id") else None
    )
    return {
        "job_id": status.get("job_id"),
        "job_type": status.get("job_type"),
        "status": status.get("status"),
        "phase": status.get("phase"),
        "started_at": status.get("started_at"),
        "updated_at": status.get("updated_at"),
        "finished_at": status.get("finished_at"),
        "status_path": str(paths.status) if paths is not None else None,
    }


def unknown_job(job_id: str) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "job_unknown",
        "error": {"code": "job_unknown", "message": f"Unknown ARC job: {job_id}"},
        "errors": [],
        "meta": {},
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_private_text(path: Path, text: str, *, exclusive: bool = False) -> None:
    _ensure_private_dir(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    _chmod_private_file(path)


def open_private_binary(path: Path, *, append: bool = False):
    _ensure_private_dir(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    _chmod_private_file(path)
    return os.fdopen(fd, "ab" if append else "wb")


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        if not path.is_symlink():
            path.chmod(0o700)
    except OSError:
        pass


def _chmod_private_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            path.chmod(0o600)
    except OSError:
        pass


def _ensure_private_file(path: Path) -> None:
    _ensure_private_dir(path.parent)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(fd)
    _chmod_private_file(path)


def _secure_job_storage(paths: JobPaths) -> None:
    _ensure_private_dir(paths.job_dir.parent.parent)
    _ensure_private_dir(paths.job_dir.parent)
    if paths.job_dir.exists():
        _ensure_private_dir(paths.job_dir)
    for path in (
        paths.job,
        paths.status,
        paths.events,
        paths.result,
        paths.error,
        paths.heartbeat,
        paths.worker_process,
        paths.worker_lock,
        paths.recovery_lock,
        paths.cancel_request,
        paths.stdout,
        paths.stderr,
        paths.worker_stdout,
        paths.worker_stderr,
        paths.progress_sidechannel,
    ):
        _chmod_private_file(path)


def _process_record(pid: int) -> dict[str, Any]:
    record: dict[str, Any] = {"pid": pid}
    identity = _process_identity(pid)
    if identity is not None:
        record["start_id"] = identity[0]
        record["state"] = identity[1]
    return record


def _process_identity(pid: int) -> tuple[str, str] | None:
    if pid <= 0:
        return None
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        raw = proc_stat.read_text(encoding="utf-8")
    except OSError:
        return None
    close = raw.rfind(")")
    fields = raw[close + 2 :].split() if close >= 0 else []
    if len(fields) <= 19:
        return None
    return fields[19], fields[0]


def _pid_record_alive(record: Mapping[str, Any]) -> bool:
    pid = record.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    identity = _process_identity(pid)
    if identity is not None:
        start_id, state = identity
        if state == "Z":
            return False
        expected = record.get("start_id")
        return expected is None or str(expected) == start_id
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_pid_lock(path: Path, *, job_id: str) -> bool:
    _ensure_private_dir(path.parent)
    payload = {"job_id": job_id, **_process_record(os.getpid()), "created_at": now_iso()}
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            owner = read_json(path, {})
            if isinstance(owner, Mapping) and _pid_record_alive(owner):
                return False
            try:
                age = max(0.0, time.time() - path.stat().st_mtime)
            except OSError:
                continue
            if not owner and age < 5.0:
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                return False
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _chmod_private_file(path)
        return True
    return False


def _release_pid_lock(path: Path) -> None:
    owner = read_json(path, {})
    if not isinstance(owner, Mapping) or owner.get("pid") != os.getpid():
        return
    expected = owner.get("start_id")
    current = _process_identity(os.getpid())
    if expected is not None and current is not None and str(expected) != current[0]:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _terminate_recorded_process(record: Mapping[str, Any]) -> bool:
    raw_pid = record.get("pid")
    if not isinstance(raw_pid, int) or raw_pid <= 0:
        return False
    pid = raw_pid
    # Never signal a historic process group after its recorded leader identity
    # disappeared: the numeric PGID may already belong to an unrelated process.
    # The worker watchdog owns descendant cleanup after a verified leader exits.
    if not _pid_record_alive(record):
        return False
    try:
        if os.name == "posix":
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline:
        if not _recorded_process_group_alive(record):
            return True
        time.sleep(0.01)
    try:
        if os.name == "posix":
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return True


def _recorded_process_group_alive(record: Mapping[str, Any]) -> bool:
    if os.name != "posix":
        return _pid_record_alive(record)
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tail_lines(path: Path, *, limit: int, max_bytes: int) -> list[bytes]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            data = b""
            while position > 0 and data.count(b"\n") <= limit and len(data) < max_bytes:
                size = min(64 * 1024, position, max_bytes - len(data))
                if size <= 0:
                    break
                position -= size
                handle.seek(position)
                data = handle.read(size) + data
    except OSError:
        return []
    lines = data.splitlines()
    if position > 0 and lines:
        lines = lines[1:]
    return lines[-limit:]


def _secure_sqlite_files(path: Path) -> None:
    for candidate in path.parent.glob(f"{path.name}*"):
        _chmod_private_file(candidate)


def _validate_status_payload(payload: Mapping[str, Any]) -> None:
    conflicts = sorted(set(payload) & RESERVED_STATUS_PAYLOAD_KEYS)
    if conflicts:
        raise ValueError(f"payload contains reserved job status keys: {', '.join(conflicts)}")


def _unwrap_result(stored: Any) -> Any:
    if (
        isinstance(stored, dict)
        and stored.get("schema_version") == "arc.job_result.v1"
        and "result" in stored
    ):
        return stored["result"]
    return stored


def _history_samples(job_type: str, command: str) -> list[float]:
    path = stats_db_path()
    if not path.exists():
        return []
    _chmod_private_file(path)
    try:
        with sqlite3.connect(path) as db:
            rows = db.execute(
                """
                SELECT duration_seconds FROM job_history
                WHERE job_type = ? AND command = ? AND status = 'done'
                ORDER BY id DESC LIMIT 100
                """,
                (job_type, command),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [float(row[0]) for row in rows]


def _progress_from_event(event: Mapping[str, Any]) -> dict[str, Any]:
    progress: dict[str, Any] = {}
    if event.get("sections_completed") is not None:
        progress["completed"] = event.get("sections_completed")
    if event.get("sections_total") is not None:
        progress["total"] = event.get("sections_total")
    if event.get("title") or event.get("section_id"):
        progress["current"] = " ".join(
            str(event.get(key) or "") for key in ("section_id", "title")
        ).strip()
    return progress


def _default_status(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("status") in TERMINAL_STATUSES:
            return str(result["status"])
        if result.get("ok") is False:
            return "failed"
        if result.get("ok") is True:
            return "done"
    return "done"


def _project_root() -> Path | None:
    """Compatibility hook; source-checkout caches are intentionally never selected."""
    return None


def _parse_time(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1 - fraction) + values[high] * fraction
