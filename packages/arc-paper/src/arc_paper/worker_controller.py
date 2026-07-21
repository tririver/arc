"""Trusted controller-only finalization for worker cache overlays."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

from .results import err, ok
from .worker_session import WorkerCacheSession


GUARD_SCHEMA_VERSION = "arc.paper.controller-guard.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize an ARC paper worker overlay")
    sub = parser.add_subparsers(dest="command", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--run-root", required=True)
    finalize.add_argument("--base-root", required=True)
    finalize.add_argument("--session-id", required=True)
    finalize.add_argument("--worker-id", required=True)
    finalize.add_argument("--call-id", required=True)
    finalize.add_argument("--status", choices=["success", "failed", "cancelled"], required=True)
    args = parser.parse_args(argv)

    try:
        _validate_controller_guard(
            run_root=Path(args.run_root),
            base_root=Path(args.base_root),
            session_id=args.session_id,
        )
        session = WorkerCacheSession(
            base_root=args.base_root,
            run_root=args.run_root,
            session_id=args.session_id,
        )
        promotion = session.promote()
        session.audit(
            worker_id=args.worker_id,
            call_id=args.call_id,
            operation="controller finalize",
            status=args.status,
            source={"entrypoint": "arc-paper-worker-controller"},
            promotion=promotion,
            promotion_status="complete",
        )
        result = ok(
            {
                "schema_version": "arc.paper.worker-finalize.v1",
                "session_id": args.session_id,
                "promotion": promotion.as_dict(),
                "promotion_status": "complete",
            }
        )
    except Exception as exc:
        result = err("worker_controller_forbidden", str(exc) or exc.__class__.__name__)
        result["status"] = "error"
        result["error"]["type"] = exc.__class__.__name__
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") is True else 1


def _validate_controller_guard(*, run_root: Path, base_root: Path, session_id: str) -> None:
    if os.environ.get("ARC_PAPER_CONTROLLER_MODE") != "trusted":
        raise PermissionError("trusted controller mode is required")
    guard_value = os.environ.get("ARC_PAPER_CONTROLLER_GUARD", "")
    token = os.environ.get("ARC_PAPER_CONTROLLER_TOKEN", "")
    if not guard_value or len(token) < 32:
        raise PermissionError("controller guard and strong token are required")
    guard_path = Path(guard_value).expanduser()
    if not guard_path.is_absolute():
        raise PermissionError("controller guard path must be absolute")
    info = guard_path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PermissionError("controller guard must be a regular non-symlink file")
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("controller guard must be owner-only")
    value: Any = json.loads(guard_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != GUARD_SCHEMA_VERSION:
        raise PermissionError("controller guard schema is invalid")
    expected = {
        "session_id": session_id,
        "run_root": str(run_root.expanduser().resolve()),
        "base_root": str(base_root.expanduser().resolve()),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise PermissionError("controller guard does not match the requested session")
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(str(value.get("token_sha256") or ""), token_hash):
        raise PermissionError("controller token is invalid")


if __name__ == "__main__":
    raise SystemExit(main())
