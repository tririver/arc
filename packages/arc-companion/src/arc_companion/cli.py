from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from .io import safe_name
from .package import package_project


DEFAULT_LANGUAGE = "zh-CN"
DEFAULT_WORKERS = 24
LANGUAGE_NOTICE = "默认使用中文生成伴读；如需切换伴读语言，请直接指定目标语言。"
# Test/controller injection seam without importing the generation pipeline for
# render-only commands.
build_companion = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate source-faithful annotated paper companions")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build or resume a paper companion")
    build.add_argument("paper_id")
    build.add_argument("--project-dir", default=None)
    build.add_argument("--annotation-language", default=None)
    build.add_argument(
        "--source-language",
        default=None,
        help=(
            "BCP-47 language tag established by source sampling; ARC does not "
            "detect the source language automatically"
        ),
    )
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
        "--arc-paper-access",
        choices=("none", "full"),
        default=None,
        help="ARC-paper access; full uses the Controller broker by default.",
    )
    legacy_paper = build.add_mutually_exclusive_group()
    legacy_paper.add_argument(
        "--arc-paper-cli",
        dest="arc_paper_cli_access",
        action="store_const",
        const="full",
        default=None,
        help="Deprecated alias for --arc-paper-access full.",
    )
    legacy_paper.add_argument(
        "--no-arc-paper-cli",
        dest="arc_paper_cli_access",
        action="store_const",
        const="none",
        help="Deprecated alias for --arc-paper-access none.",
    )
    build.add_argument(
        "--arc-paper-direct-shell",
        action="store_true",
        help="Explicitly request trusted nested-shell ARC-paper access.",
    )
    build.add_argument(
        "--arc-paper-child-llm-max-calls",
        type=int,
        default=None,
        help=(
            "opt in to managed ARC-paper child LLM operations with this finite "
            "run-shared call limit; requires full paper access and both token "
            "flags (default: managed child calls disabled); the persisted budget "
            "is reused during recovery"
        ),
    )
    build.add_argument(
        "--arc-paper-child-llm-max-tokens",
        type=int,
        default=None,
        help=(
            "finite run-shared token limit; must be set with the two other "
            "managed child budget flags"
        ),
    )
    build.add_argument(
        "--arc-paper-child-llm-output-reserve-tokens",
        type=int,
        default=None,
        help=(
            "per-call output token reservation; must be set with the two other "
            "managed child budget flags"
        ),
    )
    build.add_argument(
        "--inherit-host-tools",
        action="store_true",
        help=(
            "HIGH RISK: inherit host rules, skills, plugins, MCP, and extra tools "
            "for this run"
        ),
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
            "repeatable authorized arc-paper cache ID; intent-guided runs expose "
            "only metadata/TOC before exact on-demand section reads"
        ),
    )
    build.add_argument(
        "--reference-translation-id",
        default=None,
        metavar="SOURCE_ID",
        help=(
            "cached source ID for an existing translation used only by the "
            "translation lane"
        ),
    )
    build.add_argument(
        "--reference-translation-map",
        action="append",
        default=[],
        metavar="SOURCE_CHAPTER=REFERENCE_CHAPTER",
        help="repeat to map every source chapter explicitly in source order",
    )
    build.add_argument(
        "--user-intent",
        default=None,
        help=(
            "exact frozen user intent used to generate one shared content-lane "
            "guidance artifact"
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
    build.add_argument(
        "--recovery-policy",
        choices=("auto", "manual"),
        default="auto",
        help="recover eligible blocked generation lanes automatically (default: auto)",
    )
    build.add_argument(
        "--max-auto-replacements",
        type=int,
        default=3,
        help="maximum fresh replacement generations per blocked lane group (default: 3)",
    )
    build.add_argument(
        "--regenerate", action="append", default=[],
        choices=("segmentation", "glossary", "guide", "translation", "commentary", "review", "all"),
        help="repeat to explicitly regenerate only selected content lanes",
    )
    build.add_argument("--confirm-expensive-regeneration", action="store_true")
    build.add_argument(
        "--regenerate-segment",
        action="append",
        default=[],
        metavar="LANE:SEGMENT_ID",
        help="repeat to regenerate one translation or commentary segment",
    )
    build.add_argument(
        "--regenerate-commentary", action="store_true",
        help="deprecated alias for --regenerate commentary",
    )
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
        "--action",
        choices=("auto", "resume-native", "restart-generation"),
        default="auto",
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

    render_web = sub.add_parser(
        "render-web", help="Render or refresh the self-contained web reader"
    )
    render_web.add_argument("--project-dir", required=True)
    render_web.add_argument("--json", action="store_true")

    render = sub.add_parser(
        "render", help="Render immutable reviewed content without model calls"
    )
    render.add_argument("--project-dir", required=True)
    render.add_argument("--format", choices=("pdf", "web", "all"), default="all")
    render.add_argument("--content-sha", default=None)
    render.add_argument("--json", action="store_true")

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
        from .pipeline import BuildOptions, build_companion as pipeline_build_companion
        from arc_llm.paper_access_policy import resolve_arc_paper_access

        defaulted = args.annotation_language is None
        language = args.annotation_language or DEFAULT_LANGUAGE
        if defaulted:
            print(LANGUAGE_NOTICE, file=sys.stderr)
        if args.inherit_host_tools:
            print(
                "WARNING: --inherit-host-tools exposes host rules, skills, plugins, MCP, "
                "and extra tools to companion workers.",
                file=sys.stderr,
            )
        project_dir = (
            Path(args.project_dir)
            if args.project_dir
            else Path.cwd() / "arc-tests" / "companion" / safe_name(args.paper_id)
        )
        try:
            if args.force:
                raise ValueError(
                    "--force no longer regenerates companion content; use --regenerate LANE"
                )
            paper_access = resolve_arc_paper_access(
                {
                    "arc_paper_access": args.arc_paper_access,
                    "arc_paper_cli_access": args.arc_paper_cli_access,
                },
                os.environ,
            )
            for warning in paper_access.warnings:
                print(
                    f"WARNING: {warning}: use --arc-paper-access instead of the "
                    "deprecated ARC-paper CLI access alias.",
                    file=sys.stderr,
                )
            options = BuildOptions(
                paper_id=args.paper_id,
                project_dir=project_dir,
                annotation_language=language,
                source_language=args.source_language,
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
                arc_paper_access=paper_access.access,
                arc_paper_direct_shell=args.arc_paper_direct_shell,
                arc_paper_child_llm_max_calls=(
                    args.arc_paper_child_llm_max_calls
                ),
                arc_paper_child_llm_max_tokens=(
                    args.arc_paper_child_llm_max_tokens
                ),
                arc_paper_child_llm_output_reserve_tokens=(
                    args.arc_paper_child_llm_output_reserve_tokens
                ),
                inherit_host_tools=args.inherit_host_tools,
                skip_translation=args.skip_translation,
                context_paper_ids=tuple(args.context_paper_id),
                reference_translation_id=args.reference_translation_id,
                reference_translation_mappings=tuple(
                    args.reference_translation_map
                ),
                user_intent=args.user_intent,
                stop_after_first_chapter=args.stop_after_first_chapter,
                document_kind=args.document_kind,
                idle_timeout_seconds=args.idle_timeout_seconds,
                recovery_policy=args.recovery_policy,
                max_auto_replacements=args.max_auto_replacements,
                regenerate_lanes=tuple(args.regenerate),
                regenerate_segments=tuple(args.regenerate_segment),
                confirm_expensive_regeneration=args.confirm_expensive_regeneration,
                regenerate_commentary=args.regenerate_commentary,
                legacy_checkpoint=(
                    Path(args.legacy_checkpoint) if args.legacy_checkpoint else None
                ),
            )
        except ValueError as exc:
            result = {"ok": False, "data": None, "error": {"code": "invalid_options", "message": str(exc)}, "errors": []}
        else:
            runner = build_companion or pipeline_build_companion
            result = runner(options)
    elif args.command == "resume":
        from .pipeline import resume_companion

        result = resume_companion(
            Path(args.project_dir),
            action=args.action,
            confirm_possible_duplicate_charge=args.confirm_possible_duplicate_charge,
        )
    elif args.command == "status":
        from .pipeline import read_status

        result = read_status(Path(args.project_dir))
    elif args.command == "validate":
        from .pipeline import validate_project

        result = validate_project(Path(args.project_dir))
    elif args.command == "render-web":
        from .render import render_content

        result = render_content(Path(args.project_dir), format="web")
    elif args.command == "render":
        from .render import render_content

        result = render_content(
            Path(args.project_dir), format=args.format, content_sha256=args.content_sha,
        )
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
            data.get("output_run_pdf")
            or data.get("output_pdf")
            or data.get("preview_pdf")
            or data.get("output_html")
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
