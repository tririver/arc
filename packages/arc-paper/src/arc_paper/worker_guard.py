"""Process-local and bearer checks for paper CLI use inside model workers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from typing import Iterator


WORKER_GUARD_SCHEMA_VERSION = "arc.paper.worker-guard.v1"
_WRAPPER_AUTHORIZED: ContextVar[bool] = ContextVar("arc_paper_wrapper_authorized", default=False)


def in_worker_context() -> bool:
    return os.environ.get("ARC_LLM_WORKER_CONTEXT", "").strip().lower() == "true"


def wrapper_call_authorized() -> bool:
    return _WRAPPER_AUTHORIZED.get()


@contextmanager
def authorized_wrapper_call() -> Iterator[None]:
    token = _WRAPPER_AUTHORIZED.set(True)
    try:
        yield
    finally:
        _WRAPPER_AUTHORIZED.reset(token)


def validate_worker_guard_if_required() -> None:
    if not in_worker_context():
        return
    guard_value = os.environ.get("ARC_PAPER_WORKER_GUARD", "")
    token = os.environ.get("ARC_PAPER_WORKER_TOKEN", "")
    if not guard_value or len(token) < 32:
        raise PermissionError("worker paper guard and strong token are required")
    guard_path = Path(guard_value).expanduser()
    if not guard_path.is_absolute():
        raise PermissionError("worker paper guard path must be absolute")
    info = guard_path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PermissionError("worker paper guard must be a regular non-symlink file")
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("worker paper guard must be owner-only")
    value = json.loads(guard_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != WORKER_GUARD_SCHEMA_VERSION:
        raise PermissionError("worker paper guard schema is invalid")
    expected = {
        "session_id": os.environ.get("ARC_PAPER_WORKER_SESSION_ID", ""),
        "run_root": _resolved_env("ARC_PAPER_WORKER_SESSION_DIR"),
        "base_root": _resolved_env("ARC_PAPER_WORKER_BASE_CACHE"),
        "overlay_root": _resolved_env("ARC_PAPER_CACHE"),
    }
    if not all(expected.values()) or any(value.get(key) != item for key, item in expected.items()):
        raise PermissionError("worker paper guard does not match the active session")
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(str(value.get("token_sha256") or ""), token_hash):
        raise PermissionError("worker paper token is invalid")


def _resolved_env(name: str) -> str:
    value = os.environ.get(name, "")
    return str(Path(value).expanduser().resolve()) if value else ""
