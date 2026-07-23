from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_CURRENT_SESSION: ContextVar[Any | None] = ContextVar(
    "arc_paper_current_worker_session", default=None,
)
_CURRENT_CALL_ID: ContextVar[str | None] = ContextVar(
    "arc_paper_current_worker_call_id", default=None,
)


def current_worker_session() -> Any | None:
    return _CURRENT_SESSION.get()


def current_worker_call_id() -> str | None:
    return _CURRENT_CALL_ID.get()


@contextmanager
def worker_session_context(session: Any) -> Iterator[None]:
    token = _CURRENT_SESSION.set(session)
    try:
        yield
    finally:
        _CURRENT_SESSION.reset(token)


@contextmanager
def worker_call_context(call_id: str) -> Iterator[None]:
    token = _CURRENT_CALL_ID.set(call_id)
    try:
        yield
    finally:
        _CURRENT_CALL_ID.reset(token)
