from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .host import detect_host, select_llm_provider
from .runner import resolve_llm_config, run_json, run_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reusable ARC host LLM worker")
    sub = parser.add_subparsers(dest="command", required=True)

    run_json_parser = sub.add_parser("run-json")
    run_json_parser.add_argument("--prompt", default="-")
    run_json_parser.add_argument("--schema", default=None)
    run_json_parser.add_argument("--provider", default="auto")
    run_json_parser.add_argument("--model", default=None)
    run_json_parser.add_argument("--json", action="store_true")

    run_text_parser = sub.add_parser("run-text")
    run_text_parser.add_argument("--prompt", default="-")
    run_text_parser.add_argument("--provider", default="auto")
    run_text_parser.add_argument("--model", default=None)

    doctor = sub.add_parser("doctor")
    doctor_sub = doctor.add_subparsers(dest="doctor_command", required=True)
    doctor_sub.add_parser("host")
    doctor_sub.add_parser("provider")
    doctor_sub.add_parser("config")

    args = parser.parse_args(argv)
    result = _dispatch(args)
    if isinstance(result, str):
        print(result, end="" if result.endswith("\n") else "\n")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _dispatch(args: argparse.Namespace) -> Any:
    if args.command == "doctor":
        if args.doctor_command == "host":
            return detect_host().__dict__
        if args.doctor_command == "provider":
            selected = select_llm_provider()
            return {
                "provider": selected.provider,
                "host": selected.host.host,
                "signals": selected.signals,
            }
        config = resolve_llm_config()
        return {
            "provider": config.provider,
            "model": config.model,
            "host": config.host.host,
            "signals": config.signals,
        }
    if args.command == "run-json":
        return run_json(
            _read_prompt(args.prompt),
            schema=_read_schema(args.schema),
            provider=args.provider,
            model=args.model,
        )
    if args.command == "run-text":
        return run_text(_read_prompt(args.prompt), provider=args.provider, model=args.model)
    raise AssertionError(f"Unhandled command: {args.command}")


def _read_prompt(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    return Path(value).read_text(encoding="utf-8")


def _read_schema(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    return json.loads(Path(value).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
