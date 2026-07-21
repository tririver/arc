from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import socket
from datetime import datetime, timezone
from typing import IO, Any


class BuildInProgressError(RuntimeError):
    """Raised when another process already owns a companion project build."""


@dataclass
class ProjectBuildLock:
    """A non-blocking, process-scoped lock for one companion project.

    The small JSON payload is diagnostic only. Ownership is enforced by the OS
    lock, so a crashed process cannot leave a permanently live lock behind.
    """

    path: Path
    _handle: IO[str] | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            _lock_nonblocking(handle)
        except OSError as exc:
            handle.close()
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            owner = _read_owner(self.path)
            detail = f" ({owner})" if owner else ""
            raise BuildInProgressError(
                f"another companion build is already running for this project{detail}"
            ) from exc
        self._handle = handle
        try:
            payload = {
                "schema_version": "arc.companion.build-lock.v1",
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "process_start_identity": _process_start_identity(os.getpid()),
            }
            handle.seek(0)
            handle.truncate()
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            _unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> ProjectBuildLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def _lock_nonblocking(handle: IO[str]) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        handle.seek(0)
        if not handle.read(1):
            handle.seek(0)
            handle.write("\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: IO[str]) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_owner(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ""
    if not isinstance(value, dict):
        return ""
    host = str(value.get("host") or "unknown-host")
    pid = str(value.get("pid") or "unknown-pid")
    return f"owner {host}:{pid}"


def inspect_lock(path: Path) -> dict[str, Any] | None:
    """Return diagnostic owner data and whether the OS lock is currently held."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    active = _is_locked(path)
    pid = value.get("pid")
    recorded_identity = value.get("process_start_identity")
    identity_matches = None
    if value.get("host") == socket.gethostname() and isinstance(pid, int):
        current_identity = _process_start_identity(pid)
        identity_matches = bool(
            current_identity and recorded_identity and current_identity == recorded_identity
        )
    return {**value, "active": active, "process_identity_matches": identity_matches}


def _is_locked(path: Path) -> bool:
    try:
        handle = path.open("a+", encoding="utf-8")
    except OSError:
        return False
    try:
        try:
            _lock_nonblocking(handle)
        except OSError as exc:
            return exc.errno in {errno.EACCES, errno.EAGAIN}
        _unlock(handle)
        return False
    finally:
        handle.close()


def _process_start_identity(pid: int) -> str | None:
    """Return the kernel process birth token where the host exposes one."""

    if os.name != "posix":
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except OSError:
        return None
    return fields[21] if len(fields) > 21 else None
