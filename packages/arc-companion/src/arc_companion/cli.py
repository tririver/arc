from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .io import safe_name
from .package import package_project
from .pipeline import (
    DEFAULT_LANGUAGE,
    DEFAULT_WORKERS,
    LANGUAGE_NOTICE,
    BuildOptions,
    build_companion,
    read_status,
    validate_project,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate source-faithful annotated paper companions")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build or resume a paper companion")
    build.add_argument("paper_id")
    build.add_argument("--project-dir", default=None)
    build.add_argument("--annotation-language", default=None)
    build.add_argument("--provider", default="auto")
    build.add_argument("--model", default=None)
    build.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"total LLM-call concurrency budget shared by all active stages and lanes "
            f"(default: {DEFAULT_WORKERS})"
        ),
    )
    cache = build.add_mutually_exclusive_group()
    cache.add_argument("--recache", action="store_true")
    cache.add_argument("--refresh", action="store_true")
    domain = build.add_mutually_exclusive_group()
    domain.add_argument("--domain-id", default=None)
    domain.add_argument("--domain-manifest", default=None)
    build.add_argument(
        "--no-internet",
        action="store_true",
        help="disable internet access independently of MCP access",
    )
    build.add_argument(
        "--skip-translation",
        action="store_true",
        help="omit the translation lane when source and target languages already match",
    )
    build.add_argument(
        "--context-paper-id",
        action="append",
        default=[],
        help=(
            "repeatable arc-paper ID loaded only from the local cache as bounded "
            "explanatory context"
        ),
    )
    build.add_argument("--force", action="store_true")
    build.add_argument(
        "--stop-after-first-chapter",
        action="store_true",
        help=(
            "return successfully after the first chapter artifact passes source, "
            "compile, and PDF validation; rerun without this flag to resume"
        ),
    )
    build.add_argument(
        "--document-kind",
        choices=("auto", "article", "book"),
        default="auto",
    )
    build.add_argument("--idle-timeout-seconds", type=float, default=None)
    build.add_argument("--regenerate-commentary", action="store_true")
    build.add_argument(
        "--legacy-checkpoint",
        default=None,
        help=(
            "read-only legacy checkpoint file or directory used only to seed "
            "eligible cuts, glossary data, and validated translations"
        ),
    )
    build.add_argument("--json", action="store_true")

    resume = sub.add_parser("resume", help="Resume a supervised companion generation")
    resume.add_argument("--project-dir", required=True)
    resume.add_argument(
        "--action", required=True, choices=("resume-native", "restart-generation")
    )
    resume.add_argument(
        "--confirm-possible-duplicate-charge",
        action="store_true",
        help="required when restarting a generation because a submitted call may be billed twice",
    )
    resume.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="Show checkpoint/build status")
    status.add_argument("--project-dir", required=True)
    status.add_argument("--json", action="store_true")

    validate = sub.add_parser("validate", help="Re-run final PDF validation")
    validate.add_argument("--project-dir", required=True)
    validate.add_argument("--json", action="store_true")

    package = sub.add_parser("package", help="Package a completed companion")
    package.add_argument("--project-dir", required=True)
    package.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except Exception as exc:
        if not getattr(args, "json", False):
            raise
        result = _exception_result(exc)
    _emit(result, json_output=args.json)
    return 0 if result.get("ok") else 1


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "build":
        defaulted = args.annotation_language is None
        language = args.annotation_language or DEFAULT_LANGUAGE
        if defaulted:
            print(LANGUAGE_NOTICE, file=sys.stderr)
        project_dir = (
            Path(args.project_dir)
            if args.project_dir
            else Path.cwd() / "arc-tests" / "companion" / safe_name(args.paper_id)
        )
        try:
            options = BuildOptions(
                paper_id=args.paper_id,
                project_dir=project_dir,
                annotation_language=language,
                language_was_defaulted=defaulted,
                provider=args.provider,
                model=args.model,
                workers=args.workers,
                refresh=args.refresh,
                recache=args.recache,
                force=args.force,
                domain_id=args.domain_id,
                domain_manifest=Path(args.domain_manifest) if args.domain_manifest else None,
                allow_internet=not args.no_internet,
                skip_translation=args.skip_translation,
                context_paper_ids=tuple(args.context_paper_id),
                stop_after_first_chapter=args.stop_after_first_chapter,
                document_kind=args.document_kind,
                idle_timeout_seconds=args.idle_timeout_seconds,
                regenerate_commentary=args.regenerate_commentary,
                legacy_checkpoint=(
                    Path(args.legacy_checkpoint) if args.legacy_checkpoint else None
                ),
            )
        except ValueError as exc:
            result = {"ok": False, "data": None, "error": {"code": "invalid_options", "message": str(exc)}, "errors": []}
        else:
            result = build_companion(options)
    elif args.command == "resume":
        from .pipeline import resume_companion

        result = resume_companion(
            Path(args.project_dir),
            action=args.action,
            confirm_possible_duplicate_charge=args.confirm_possible_duplicate_charge,
        )
    elif args.command == "status":
        result = read_status(Path(args.project_dir))
    elif args.command == "validate":
        result = validate_project(Path(args.project_dir))
    elif args.command == "package":
        result = package_project(Path(args.project_dir))
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    return result


def _exception_result(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "error",
        "error": {
            "code": "command_failed",
            "message": str(exc) or exc.__class__.__name__,
            "type": exc.__class__.__name__,
        },
        "errors": [],
        "meta": {},
    }


def _emit(result: dict[str, Any], *, json_output: bool) -> None:
    meta = result.get("meta") or {}
    diagnostics = meta.get("diagnostics") or []
    emitted_failures: set[str] = set()
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        severity = str(diagnostic.get("severity") or "").lower()
        if severity not in {"warning", "error"}:
            continue
        message = str(diagnostic.get("message") or diagnostic.get("code") or "companion warning")
        print(f"WARNING: {message}", file=sys.stderr)
        if severity == "error":
            emitted_failures.add(message)
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if result.get("ok"):
        data = result.get("data") or {}
        print(
            data.get("output_pdf")
            or data.get("preview_pdf")
            or data.get("archive_path")
            or data.get("status")
            or data
        )
        return
    error = result.get("error") or {}
    message = str(error.get("message") or "companion command failed")
    if message not in emitted_failures:
        print(f"WARNING: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
