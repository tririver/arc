"""Controller finalization for worker cache overlays."""

from __future__ import annotations

import argparse
import json
import sys

from .results import err, ok
from .worker_session import WorkerCacheSession


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

if __name__ == "__main__":
    raise SystemExit(main())
