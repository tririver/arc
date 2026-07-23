from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import get_ident
from typing import Any, Iterator


class LockConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.run_dir / self.run_id

    @property
    def config(self) -> Path:
        return self.run_root / "config.json"

    @property
    def manifest(self) -> Path:
        return self.run_root / "manifest.json"

    @property
    def state(self) -> Path:
        return self.run_root / "state.json"

    @property
    def lock(self) -> Path:
        return self.run_root / "run.lock"

    @property
    def sessions_root(self) -> Path:
        return self.run_root / "sessions"

    @property
    def call_records(self) -> Path:
        return self.run_root / "call_records.jsonl"

    @property
    def evidence_journal_root(self) -> Path:
        return self.run_root / "evidence-journal"

    def loop(self, loop_id: str) -> "LoopPaths":
        return LoopPaths(run_root=self.run_root, loop_id=loop_id)


@dataclass(frozen=True)
class LoopPaths:
    run_root: Path
    loop_id: str

    @property
    def loop_root(self) -> Path:
        return self.run_root / "loops" / self.loop_id

    @property
    def lock(self) -> Path:
        return self.loop_root / "lock.json"

    @property
    def config(self) -> Path:
        return self.loop_root / "loop_config.json"

    @property
    def state(self) -> Path:
        return self.loop_root / "state.json"

    @property
    def transcript(self) -> Path:
        return self.loop_root / "transcript.jsonl"

    def round(self, round_number: int) -> "RoundPaths":
        return RoundPaths(loop_root=self.loop_root, round_number=round_number)


@dataclass(frozen=True)
class RoundPaths:
    loop_root: Path
    round_number: int

    @property
    def round_root(self) -> Path:
        return self.loop_root / "rounds" / f"round_{self.round_number:03d}"

    @property
    def context_dir(self) -> Path:
        return self.round_root / "context"

    @property
    def prompt_dir(self) -> Path:
        return self.round_root / "prompts"

    @property
    def proposer_output_dir(self) -> Path:
        return self.round_root / "proposer_outputs"

    @property
    def review_dir(self) -> Path:
        return self.round_root / "reviews"

    @property
    def error_dir(self) -> Path:
        return self.round_root / "errors"

    def proposer_context(self, worker_id: str) -> Path:
        return self.context_dir / f"{worker_id}.json"

    def reviewer_context(self, worker_id: str) -> Path:
        return self.context_dir / f"{worker_id}.json"

    def prompt(self, worker_id: str, *, kind: str = "prompt") -> Path:
        if kind != "prompt":
            return self.prompt_dir / f"{worker_id}.{kind}.md"
        return self.prompt_dir / f"{worker_id}.md"

    def proposer_output(self, worker_id: str) -> Path:
        return self.proposer_output_dir / f"{worker_id}.json"

    def review(self, worker_id: str) -> Path:
        return self.review_dir / f"{worker_id}.json"

    def worker_error(self, worker_id: str) -> Path:
        return self.error_dir / f"{worker_id}.json"


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    _fsync_parent(path)


@contextmanager
def acquire_lock(lock_path: Path, *, run_id: str, loop_id: str | None = None) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "loop_id": loop_id or "",
        "pid": os.getpid(),
        "thread_id": get_ident(),
        "host": _hostname(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(lock_path, flags, 0o644)
    except FileExistsError as exc:
        if _recover_dead_process_lock(lock_path):
            try:
                fd = os.open(lock_path, flags, 0o644)
            except FileExistsError as retry_exc:
                raise LockConflictError(f"lock already exists: {lock_path}") from retry_exc
        else:
            raise LockConflictError(f"lock already exists: {lock_path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def append_jsonl(path: Path, item: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{get_ident()}.{time.time_ns()}.tmp")


def _fsync_parent(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return ""


def _recover_dead_process_lock(lock_path: Path) -> bool:
    payload = _read_lock_payload(lock_path)
    if payload.get("host") != _hostname():
        return False
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    if _pid_exists(pid):
        return False
    try:
        lock_path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _read_lock_payload(lock_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True
