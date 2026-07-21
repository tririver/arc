from __future__ import annotations

import os
import signal
import sys
import time


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        return 2
    parent_pid, pgid = int(args[0]), int(args[1])
    grace_seconds = max(0.0, float(args[2]))
    while _group_exists(pgid):
        if os.getppid() != parent_pid:
            _signal(pgid, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline and _group_exists(pgid):
                time.sleep(0.02)
            if _group_exists(pgid):
                _signal(pgid, signal.SIGKILL)
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


def _signal(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
