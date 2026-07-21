"""Process-local routing checks for paper CLI use inside model workers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
from typing import Iterator


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
