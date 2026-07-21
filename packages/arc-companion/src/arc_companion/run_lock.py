from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import socket
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
