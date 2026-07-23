"""Controller-owned cooperative execution signals for managed paper jobs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]

_PROGRESS_CALLBACK: ContextVar[ProgressCallback | None] = ContextVar(
    "arc_paper_progress_callback", default=None,
)
_CANCEL_CHECK: ContextVar[CancelCheck | None] = ContextVar(
    "arc_paper_cancel_check", default=None,
)


class ManagedExecutionCancelled(RuntimeError):
    """Stop a managed batch without claiming another provider call."""

    code = "paper_broker_job_cancelled"
    abort_batch = True
    submission_state = "not_submitted"


@contextmanager
def managed_execution_scope(
    *,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> Iterator[None]:
    """Propagate job progress/cancellation through nested calls and threads."""

    progress_token = _PROGRESS_CALLBACK.set(progress_callback)
    cancel_token = _CANCEL_CHECK.set(cancel_check)
    try:
        yield
    finally:
        _CANCEL_CHECK.reset(cancel_token)
        _PROGRESS_CALLBACK.reset(progress_token)


def current_progress_callback() -> ProgressCallback | None:
    return _PROGRESS_CALLBACK.get()


def current_cancel_check() -> CancelCheck | None:
    return _CANCEL_CHECK.get()


def check_cancelled() -> None:
    cancel_check = current_cancel_check()
    if cancel_check is not None and cancel_check():
        raise ManagedExecutionCancelled(
            "Managed ARC-paper job cancellation was requested.",
        )
