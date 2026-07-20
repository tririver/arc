from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


CancelCheck = Callable[[], bool]


@contextmanager
def install_signal_cancel_chain(cancel_check: CancelCheck | None = None) -> Iterator[CancelCheck]:
    """Combine SIGINT/SIGTERM with an optional caller-provided cancel check.

    Direct batch callers can pass the yielded check to
    ``run_proposers_reviewer_batch``. Signal handlers are installed only on
    the main thread and are always restored when the scope exits.
    """

    requested = threading.Event()

    def combined() -> bool:
        return requested.is_set() or bool(cancel_check is not None and cancel_check())

    previous: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signum, _frame: requested.set())
    try:
        yield combined
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
