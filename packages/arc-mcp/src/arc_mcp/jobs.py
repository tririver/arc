from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import tomllib
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, get_ident
from typing import Any, Callable, Mapping
from uuid import uuid4


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]
JobRunner = Callable[[ProgressCallback, CancelCheck], Any]
StatusResolver = Callable[[Any], str]
TERMINAL_STATUSES = {"done", "failed", "cancelled", "needs_llm"}


class MCPJobCancelled(RuntimeError):
    """Raised when a cooperative MCP job notices a cancellation request."""


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
    worker_lock: Path
    cancel_request: Path
    stdout: Path
    stderr: Path

    @classmethod
    def for_job(cls, job_id: str) -> "JobPaths":
        job_dir = jobs_root() / safe_job_id(job_id)
        return cls(
            job_id=safe_job_id(job_id),
            job_dir=job_dir,
            job=job_dir / "job.json",
            status=job_dir / "status.json",
            events=job_dir / "events.jsonl",
            result=job_dir / "result.json",
            error=job_dir / "error.json",
            heartbeat=job_dir / "heartbeat.json",
            worker_lock=job_dir / "worker.lock",
            cancel_request=job_dir / "cancel.request",
            stdout=job_dir / "worker.stdout.log",
            stderr=job_dir / "worker.stderr.log",
        )


class MCPJobManager:
    def __init__(
        self,
        *,
        max_workers: int = 1,
        event_limit: int = 100,
        worker_mode: str | None = None,
    ):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="arc-mcp-job")
        self._event_limit = event_limit
        self._worker_mode = worker_mode
        self._futures: dict[str, Future] = {}
        self._lock = Lock()

    def start(
        self,
        *,
        job_type: str,
        payload: Mapping[str, Any] | None,
        runner: JobRunner | None = None,
        status_resolver: StatusResolver | None = None,
    ) -> str:
        job_id = uuid4().hex
        paths = JobPaths.for_job(job_id)
        now = now_iso()
        job = {
            "schema_version": "arc.mcp_job.v1",
            "job_id": job_id,
            "job_type": job_type,
            "payload": dict(payload or {}),
            "created_at": now,
        }
        paths.job_dir.mkdir(parents=True, exist_ok=True)
        write_json(paths.job, job)
        write_json(
            paths.status,
            {
                "schema_version": "arc.mcp_job_status.v1",
                "job_id": job_id,
                "job_type": job_type,
                "status": "queued",
                "phase": "queued",
                "progress": {},
                "cancel_requested": False,
                "started_at": now,
                "updated_at": now,
                "finished_at": None,
                **dict(payload or {}),
            },
        )
        if self._use_thread_worker() and runner is not None:
            future = self._executor.submit(self._run_thread_worker, job_id, runner, status_resolver or _default_status)
            with self._lock:
                self._futures[job_id] = future
        else:
            self._launch_worker(job_id)
        return job_id

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
        paths = JobPaths.for_job(job_id)
        status = read_json(paths.status)
        if not isinstance(status, dict):
            return {
                "ok": False,
                "status": "job_unknown",
                "error": {"code": "job_unknown", "message": f"Unknown MCP job: {job_id}"},
                "errors": [],
                "meta": {},
            }
        if paths.cancel_request.exists() and status.get("status") not in TERMINAL_STATUSES:
            status["cancel_requested"] = True
        status["events"] = tail_events(paths.events, limit=self._event_limit)
        status["eta"] = estimate_eta(status)
        return status

    def result(self, job_id: str) -> dict[str, Any]:
        status = self.status(job_id)
        if status.get("status") == "job_unknown":
            return status
        if status.get("status") in {"done", "needs_llm"}:
            result = read_json(JobPaths.for_job(job_id).result)
            return {
                "ok": True,
                "status": status["status"],
                "job_id": job_id,
                "job_type": status.get("job_type"),
                "result": result,
                "meta": {"job": job_meta(status)},
            }
        if status.get("status") in {"failed", "cancelled", "cancel_requested"}:
            error = read_json(JobPaths.for_job(job_id).error) or status.get("error")
            return {
                "ok": False,
                "status": status.get("status"),
                "job_id": job_id,
                "job_type": status.get("job_type"),
                "error": error or {"code": str(status.get("status")), "message": f"MCP job {status.get('status')}"},
                "errors": [],
                "meta": {"job": job_meta(status)},
            }
        return {
            "ok": False,
            "status": status.get("status", "running"),
            "job_id": job_id,
            "job_type": status.get("job_type"),
            "message": "MCP job is not complete yet.",
            "next": {"tool": "job_status", "arguments": {"job_id": job_id}, "poll_after_seconds": 5},
            "meta": {"job": job_meta(status)},
        }

    def list_jobs(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        root = jobs_root()
        if root.exists():
            for path in root.iterdir():
                if path.is_dir():
                    status = self.status(path.name)
                    if status.get("status") != "job_unknown":
                        items.append(status)
        items.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
        return {"ok": True, "data": {"jobs": items}, "errors": [], "meta": {}}

    def cancel(self, job_id: str) -> dict[str, Any]:
        paths = JobPaths.for_job(job_id)
        if not paths.status.exists():
            return {
                "ok": False,
                "status": "job_unknown",
                "error": {"code": "job_unknown", "message": f"Unknown MCP job: {job_id}"},
                "errors": [],
                "meta": {},
            }
        paths.cancel_request.write_text(now_iso(), encoding="utf-8")
        requested = update_status(
            job_id,
            status="cancel_requested",
            phase="cancel_requested",
            cancel_requested=True,
            error={
                "code": "job_cancel_requested",
                "message": "Cancellation was requested. Running subprocesses may finish before the job can stop.",
            },
        )
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None and future.cancel():
            set_error(job_id, "job_cancelled", "MCP job was cancelled before it started running.", cancelled=True)
            return self.status(job_id)
        requested["events"] = tail_events(paths.events, limit=self._event_limit)
        requested["eta"] = estimate_eta(requested)
        return requested

    def _use_thread_worker(self) -> bool:
        mode = self._worker_mode or os.environ.get("ARC_MCP_WORKER_MODE", "process")
        return mode == "thread"

    def _launch_worker(self, job_id: str) -> None:
        paths = JobPaths.for_job(job_id)
        command = [sys.executable, "-m", "arc_mcp.worker", job_id]
        stdout = paths.stdout.open("ab")
        stderr = paths.stderr.open("ab")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            stdout.close()
            stderr.close()
        update_status(
            job_id,
            worker={"pid": process.pid, "command": command},
        )

    def _run_thread_worker(self, job_id: str, runner: JobRunner, status_resolver: StatusResolver) -> None:
        if not acquire_worker_lock(job_id):
            return
        start_running(job_id)

        def cancel_requested() -> bool:
            return is_cancel_requested(job_id)

        def progress(event: dict[str, Any]) -> None:
            if cancel_requested():
                raise MCPJobCancelled("MCP job cancellation was requested.")
            record_progress(job_id, event)

        try:
            result = runner(progress, cancel_requested)
            if cancel_requested():
                raise MCPJobCancelled("MCP job cancellation was requested.")
            finish_job(job_id, result, status_resolver(result))
        except MCPJobCancelled as exc:
            set_error(job_id, "job_cancelled", str(exc), cancelled=True)
        except Exception as exc:
            set_error(job_id, "job_failed", str(exc))


def cache_root() -> Path:
    if value := os.environ.get("ARC_MCP_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("XDG_CACHE_HOME"):
        return Path(value).expanduser() / "arc" / "arc-mcp"
    if project_root := _project_root():
        return project_root / "cache" / "arc-mcp"
    return Path.home() / ".cache" / "arc" / "arc-mcp"


def jobs_root() -> Path:
    return cache_root() / "jobs"


def stats_db_path() -> Path:
    return cache_root() / "stats" / "jobs.sqlite"


def safe_job_id(value: str) -> str:
    safe = "".join(char for char in value.strip() if char.isalnum() or char in "-_")
    if not safe:
        raise ValueError("job_id is required")
    return safe


def read_job(job_id: str) -> dict[str, Any]:
    data = read_json(JobPaths.for_job(job_id).job)
    if not isinstance(data, dict):
        raise FileNotFoundError(f"Unknown MCP job: {job_id}")
    return data


def acquire_worker_lock(job_id: str) -> bool:
    paths = JobPaths.for_job(job_id)
    payload = {
        "job_id": job_id,
        "pid": os.getpid(),
        "created_at": now_iso(),
    }
    try:
        fd = os.open(paths.worker_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return True


def start_running(job_id: str) -> None:
    now = now_iso()
    update_status(
        job_id,
        status="running",
        phase="running",
        worker={"pid": os.getpid()},
        worker_started_at=now,
    )
    write_json(JobPaths.for_job(job_id).heartbeat, {"job_id": job_id, "pid": os.getpid(), "heartbeat_at": now})


def record_progress(job_id: str, event: Mapping[str, Any]) -> None:
    if is_cancel_requested(job_id):
        raise MCPJobCancelled("MCP job cancellation was requested.")
    paths = JobPaths.for_job(job_id)
    timestamped = {"at": now_iso(), **dict(event)}
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    with paths.events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(timestamped, ensure_ascii=False) + "\n")
    event_name = str(timestamped.get("event") or "")
    progress = _progress_from_event(timestamped)
    updates: dict[str, Any] = {
        "phase": event_name or None,
        "progress": progress,
    }
    for key, value in timestamped.items():
        if key not in {"at", "event"}:
            updates[key] = value
    update_status(job_id, **{key: value for key, value in updates.items() if value is not None})
    write_json(
        paths.heartbeat,
        {"job_id": job_id, "pid": os.getpid(), "heartbeat_at": timestamped["at"], "phase": event_name},
    )


def finish_job(job_id: str, result: Any, status: str) -> None:
    paths = JobPaths.for_job(job_id)
    write_json(paths.result, result)
    finished_at = now_iso()
    update_status(job_id, status=status, phase=status, finished_at=finished_at, result_path=str(paths.result))
    record_history(job_id, status=status)


def set_error(job_id: str, code: str, message: str, *, cancelled: bool = False) -> None:
    paths = JobPaths.for_job(job_id)
    error = {"code": code, "message": message}
    write_json(paths.error, error)
    status = "cancelled" if cancelled else "failed"
    finished_at = now_iso()
    update_status(job_id, status=status, phase=status, finished_at=finished_at, error=error, error_path=str(paths.error))
    record_history(job_id, status=status)


def is_cancel_requested(job_id: str) -> bool:
    return JobPaths.for_job(job_id).cancel_request.exists()


def update_status(job_id: str, **fields: Any) -> dict[str, Any]:
    paths = JobPaths.for_job(job_id)
    status = read_json(paths.status)
    if not isinstance(status, dict):
        status = {"job_id": job_id}
    status.update({key: value for key, value in fields.items() if value is not None})
    status["updated_at"] = now_iso()
    write_json(paths.status, status)
    return status


def tail_events(path: Path, *, limit: int) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def record_history(job_id: str, *, status: str) -> None:
    job = read_json(JobPaths.for_job(job_id).job)
    current = read_json(JobPaths.for_job(job_id).status)
    if not isinstance(job, dict) or not isinstance(current, dict):
        return
    started = _parse_time(current.get("worker_started_at") or current.get("started_at"))
    finished = _parse_time(current.get("finished_at"))
    if started is None or finished is None:
        return
    duration = max(0.0, finished - started)
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    key = stats_key(str(job.get("job_type") or ""), payload)
    path = stats_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS job_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                stats_key TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                status TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            INSERT INTO job_history
              (job_type, stats_key, provider, model, status, duration_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(job.get("job_type") or ""),
                key,
                _string_or_none(payload.get("provider")),
                _string_or_none(payload.get("model")),
                status,
                duration,
                now_iso(),
            ),
        )


def estimate_eta(status: Mapping[str, Any]) -> dict[str, Any]:
    job_type = str(status.get("job_type") or "")
    if status.get("status") in TERMINAL_STATUSES:
        return {"available": False, "reason": "terminal_status"}
    key = stats_key(job_type, status)
    samples = _history_samples(job_type, key)
    if len(samples) < 3:
        return {"available": False, "reason": "insufficient_history", "samples": len(samples)}
    started = _parse_time(status.get("worker_started_at") or status.get("started_at"))
    elapsed = max(0.0, time.time() - started) if started is not None else 0.0
    sorted_samples = sorted(samples)
    if len(sorted_samples) < 10:
        low = sorted_samples[0]
        high = sorted_samples[-1]
        basis = f"history:{job_type}:{len(sorted_samples)}:minmax"
    else:
        low = _quantile(sorted_samples, 0.025)
        high = _quantile(sorted_samples, 0.975)
        basis = f"history:{job_type}:{len(sorted_samples)}:central95"
    p50 = _quantile(sorted_samples, 0.5)
    return {
        "available": True,
        "samples": len(sorted_samples),
        "basis": basis,
        "total_seconds_p50": round(p50, 1),
        "total_seconds_low": round(low, 1),
        "total_seconds_high": round(high, 1),
        "remaining_seconds_p50": round(max(0.0, p50 - elapsed), 1),
        "remaining_seconds_low": round(max(0.0, low - elapsed), 1),
        "remaining_seconds_high": round(max(0.0, high - elapsed), 1),
    }


def stats_key(job_type: str, payload: Mapping[str, Any]) -> str:
    provider = str(payload.get("provider") or "")
    model = str(payload.get("model") or "")
    model_tier = str(payload.get("model_tier") or "")
    if job_type == "paper_summary":
        paper_ids = payload.get("paper_ids")
        count = len(paper_ids) if isinstance(paper_ids, list) else 1
        return f"paper_summary:papers={_bucket(count)}:provider={provider}:model={model}:tier={model_tier}"
    if job_type == "domain_build":
        workers = payload.get("workers")
        return f"domain_build:workers={workers}:provider={provider}:model={model}"
    if job_type == "summary_batch_run":
        max_items = payload.get("max_items")
        count = int(max_items) if isinstance(max_items, int) and max_items > 0 else 0
        return f"summary_batch_run:items={_bucket(count)}:provider={provider}:model={model}:tier={model_tier}"
    return f"{job_type}:provider={provider}:model={model}"


def resolve_inline_wait_seconds(
    *,
    env: Mapping[str, str] | None = None,
    server_name: str = "arc",
    default: float = 90.0,
) -> float:
    env = env if env is not None else os.environ
    explicit = _float_env(env, "ARC_MCP_INLINE_WAIT_SEC")
    if explicit is not None:
        return max(0.0, explicit)
    margin = _float_env(env, "ARC_MCP_BACKGROUND_MARGIN_SEC")
    if margin is None:
        margin = 10.0
    timeout = _float_env(env, "ARC_MCP_TOOL_TIMEOUT_SEC")
    if timeout is None:
        timeout = _codex_mcp_tool_timeout(env=env, server_name=server_name)
    if timeout is None:
        return max(0.0, default)
    return max(0.0, timeout - margin)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{get_ident()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def job_meta(status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": status.get("job_id"),
        "job_type": status.get("job_type"),
        "status": status.get("status"),
        "phase": status.get("phase"),
        "started_at": status.get("started_at"),
        "updated_at": status.get("updated_at"),
        "finished_at": status.get("finished_at"),
        "status_path": str(JobPaths.for_job(str(status.get("job_id") or "")).status)
        if status.get("job_id")
        else None,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _history_samples(job_type: str, key: str) -> list[float]:
    path = stats_db_path()
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as db:
            rows = db.execute(
                """
                SELECT duration_seconds
                FROM job_history
                WHERE job_type = ? AND stats_key = ? AND status = 'done'
                ORDER BY id DESC
                LIMIT 100
                """,
                (job_type, key),
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
        progress["current"] = " ".join(str(event.get(key) or "") for key in ("section_id", "title")).strip()
    return progress


def _default_status(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("status") == "needs_llm":
            return "needs_llm"
        if result.get("ok") is False:
            return "failed"
        if result.get("ok") is True:
            return "done"
    return "done"


def _codex_mcp_tool_timeout(*, env: Mapping[str, str], server_name: str) -> float | None:
    config_path = Path(env.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    server = servers.get(server_name)
    if not isinstance(server, dict):
        return None
    value = server.get("tool_timeout_sec")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _project_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "packages" / "arc-mcp").is_dir() and (parent / "cache").is_dir():
            return parent
    return None


def _float_env(env: Mapping[str, str], key: str) -> float | None:
    value = env.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
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
    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low
    return values[low] * (1 - frac) + values[high] * frac


def _bucket(count: int) -> str:
    if count <= 0:
        return "unknown"
    if count <= 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    if count <= 50:
        return "21-50"
    if count <= 100:
        return "51-100"
    return "100+"


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
