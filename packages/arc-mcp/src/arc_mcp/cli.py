from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from . import server
from .jobs import MCPJobManager, cache_root
from .worker import run_job


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "jobs":
        return _jobs(argv[1:])
    if argv and argv[0] in {"status", "result", "watch", "cancel", "list", "root", "-h", "--help"}:
        return _jobs(argv)
    if argv and argv[0] == "worker":
        parser = argparse.ArgumentParser(description="Run ARC MCP worker")
        parser.add_argument("job_id")
        args = parser.parse_args(argv[1:])
        return run_job(args.job_id)
    return server.run_mcp_server()


def _jobs(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Inspect ARC MCP background jobs")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status")
    status.add_argument("job_id")
    status.add_argument("--json", action="store_true")

    result = sub.add_parser("result")
    result.add_argument("job_id")
    result.add_argument("--json", action="store_true")

    watch = sub.add_parser("watch")
    watch.add_argument("job_id")
    watch.add_argument("--interval", type=float, default=5.0)
    watch.add_argument("--json", action="store_true")
    watch.add_argument("--progress-jsonl", action="store_true")

    cancel = sub.add_parser("cancel")
    cancel.add_argument("job_id")
    cancel.add_argument("--json", action="store_true")

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--json", action="store_true")

    root = sub.add_parser("root")
    root.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    manager = MCPJobManager()
    if args.command == "status":
        return _emit(manager.status(args.job_id), json_output=args.json)
    if args.command == "result":
        return _emit(manager.result(args.job_id), json_output=args.json)
    if args.command == "cancel":
        return _emit(manager.cancel(args.job_id), json_output=args.json)
    if args.command == "list":
        return _emit(manager.list_jobs(), json_output=args.json)
    if args.command == "root":
        return _emit({"ok": True, "data": {"cache_root": str(cache_root())}, "errors": [], "meta": {}}, json_output=args.json)
    if args.command == "watch":
        return _watch(manager, args.job_id, interval=args.interval, json_output=args.json, progress_jsonl=args.progress_jsonl)
    raise AssertionError(f"Unhandled jobs command: {args.command}")


def _watch(
    manager: MCPJobManager,
    job_id: str,
    *,
    interval: float,
    json_output: bool,
    progress_jsonl: bool,
) -> int:
    seen_events = 0
    while True:
        status = manager.status(job_id)
        if progress_jsonl:
            events = status.get("events") if isinstance(status.get("events"), list) else []
            for event in events[seen_events:]:
                print(json.dumps({"job_id": job_id, **event}, ensure_ascii=False), flush=True)
            seen_events = len(events)
        elif not json_output:
            _print_human_status(status)
        if status.get("status") in {"done", "failed", "cancelled", "needs_llm", "job_unknown"}:
            if status.get("status") in {"done", "needs_llm"}:
                return _emit(manager.result(job_id), json_output=json_output)
            return _emit(status, json_output=json_output)
        time.sleep(max(0.1, interval))


def _emit(data: Any, *, json_output: bool) -> int:
    if json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    else:
        _print_human(data)
    return 0


def _print_human(data: Any) -> None:
    if isinstance(data, dict) and "status" in data:
        _print_human_status(data)
        return
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _print_human_status(status: dict[str, Any]) -> None:
    job_id = status.get("job_id", "")
    job_type = status.get("job_type", "")
    phase = status.get("phase") or status.get("status")
    eta = status.get("eta") if isinstance(status.get("eta"), dict) else {}
    if eta.get("available"):
        eta_text = (
            f", ETA {eta.get('remaining_seconds_low')}s-"
            f"{eta.get('remaining_seconds_high')}s ({eta.get('basis')})"
        )
    else:
        eta_text = ""
    print(f"{job_id} {job_type} {status.get('status')} {phase}{eta_text}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
