from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Any

from .jobs import JobManager, TERMINAL_STATUSES, arc_jobs_cli_command


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manager = JobManager()
        if args.command == "submit":
            command_argv = list(args.argv)
            if command_argv[:1] == ["--"]:
                command_argv = command_argv[1:]
            job_id = manager.submit(command_argv, job_type=args.job_type, cwd=args.cwd)
            response = {
                "ok": True,
                "status": "job_running",
                "job_id": job_id,
                "job_type": args.job_type,
                "argv": command_argv,
                "cwd": manager.status(job_id).get("cwd"),
                "next": {
                    "cli_command": arc_jobs_cli_command("watch", job_id, "--json"),
                    "poll_after_seconds": 5,
                },
                "errors": [],
                "meta": {},
            }
            return _emit(response, json_output=args.json)
        if args.command == "list":
            return _emit(manager.list_jobs(), json_output=args.json)
        if args.command == "status":
            response = manager.status(args.job_id)
            return _emit(
                response,
                json_output=args.json,
                failure=response.get("status") in {"job_unknown", "failed", "cancelled"},
            )
        if args.command == "result":
            response = manager.result(args.job_id)
            return _emit(response, json_output=args.json, failure=response.get("ok") is not True)
        if args.command == "cancel":
            response = manager.cancel(args.job_id)
            return _emit(response, json_output=args.json, failure=_is_unknown(response))
        if args.command == "watch":
            return _watch(
                manager,
                args.job_id,
                interval=args.interval,
                json_output=args.json,
                progress_jsonl=args.progress_jsonl,
            )
    except ValueError as exc:
        response = {
            "ok": False,
            "status": "invalid_request",
            "error": {"code": "invalid_request", "message": str(exc)},
            "errors": [],
            "meta": {},
        }
        if getattr(args, "progress_jsonl", False):
            return _emit_jsonl(response, failure=True)
        return _emit(response, json_output=getattr(args, "json", False), failure=True)
    except Exception as exc:
        response = {
            "ok": False,
            "status": "internal_error",
            "error": {"code": "internal_error", "message": str(exc)},
            "errors": [],
            "meta": {},
        }
        if getattr(args, "progress_jsonl", False):
            return _emit_jsonl(response, failure=True)
        return _emit(response, json_output=getattr(args, "json", False), failure=True)
    raise AssertionError(f"Unhandled command: {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run persistent jobs for allowlisted ARC CLIs")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Submit argv after '--'; shell command strings are not accepted")
    submit.add_argument("--job-type", default="cli")
    submit.add_argument("--cwd", default=".", help="Working directory for the ARC command")
    submit.add_argument("--json", action="store_true")
    submit.add_argument("argv", nargs=argparse.REMAINDER)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--json", action="store_true")

    for name in ("status", "result", "cancel"):
        command = sub.add_parser(name)
        command.add_argument("job_id")
        command.add_argument("--json", action="store_true")

    watch = sub.add_parser("watch")
    watch.add_argument("job_id")
    watch.add_argument("--interval", type=float, default=5.0)
    watch.add_argument("--json", action="store_true")
    watch.add_argument("--progress-jsonl", action="store_true")
    return parser


def _watch(
    manager: JobManager,
    job_id: str,
    *,
    interval: float,
    json_output: bool,
    progress_jsonl: bool,
) -> int:
    seen_event_keys: set[tuple[Any, Any, Any]] = set()
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_watch(signum, frame):
        del signum, frame
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_watch)
    try:
        while True:
            status = manager.status(job_id)
            if progress_jsonl:
                events = status.get("events") if isinstance(status.get("events"), list) else []
                for event in events:
                    key = (event.get("event_id"), event.get("at"), event.get("event"))
                    if key not in seen_event_keys:
                        _emit_jsonl({"job_id": job_id, **event})
                        seen_event_keys.add(key)
            elif not json_output:
                _print_human_status(status)
            if status.get("status") in TERMINAL_STATUSES or status.get("status") == "job_unknown":
                response = manager.result(job_id)
                if progress_jsonl:
                    return _emit_jsonl(response, failure=response.get("ok") is not True)
                return _emit(response, json_output=json_output, failure=response.get("ok") is not True)
            time.sleep(max(0.1, interval))
    except KeyboardInterrupt:
        manager.cancel(job_id)
        manager.wait(job_id, timeout=5.0)
        response = manager.result(job_id)
        if progress_jsonl:
            return _emit_jsonl(response, failure=True)
        return _emit(response, json_output=json_output, failure=True)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def _emit_jsonl(data: Any, *, failure: bool = False) -> int:
    print(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str), flush=True)
    return 1 if failure else 0


def _emit(data: Any, *, json_output: bool, failure: bool = False) -> int:
    if json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    else:
        _print_human(data)
    return 1 if failure else 0


def _print_human(data: Any) -> None:
    if isinstance(data, dict) and "status" in data:
        _print_human_status(data)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _print_human_status(status: dict[str, Any]) -> None:
    job_id = status.get("job_id", "")
    job_type = status.get("job_type", "")
    phase = status.get("phase") or status.get("status")
    print(f"{job_id} {job_type} {status.get('status')} {phase}", flush=True)


def _is_unknown(response: dict[str, Any]) -> bool:
    return response.get("status") == "job_unknown"


if __name__ == "__main__":
    raise SystemExit(main())
