from __future__ import annotations

import os
import signal
import sys
import time


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        return 2
    parent_pid, process_group_id = (int(args[0]), int(args[1]))
    grace_seconds = max(0.0, float(args[2]))
    while _group_exists(process_group_id):
        # After an unexpected parent exit POSIX reparents this watchdog.  It
        # then owns cleanup independently of job/status reconciliation.
        if os.getppid() != parent_pid:
            _signal_group(process_group_id, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline and _group_exists(process_group_id):
                time.sleep(0.02)
            if _group_exists(process_group_id):
                _signal_group(process_group_id, signal.SIGKILL)
            return 0
        time.sleep(0.05)
    return 0


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
