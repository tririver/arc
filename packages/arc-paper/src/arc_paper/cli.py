"""Protocol-only command line interface for arc-paper operations and workflows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from typing import Any

from ac_jobs import (
    CommandError,
    CommandResult,
    CommandStatus,
    CommandWarning,
    command_result_json,
    command_result_from_snapshot,
    run_control_main,
)

from .registry import dispatch_operation, to_json_value
from .workflows.keywords import KeywordExtractionPaused


class _UsageError(ValueError):
    pass


class _HelpRequested(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status == 0:
            raise _HelpRequested
        super().exit(status, message)


class _DocumentTargetAction(argparse.Action):
    """Append mixed document targets in command-line order."""

    def __call__(self, parser, namespace, values, option_string=None):
        targets = list(getattr(namespace, self.dest, None) or [])
        if option_string == "--reference":
            targets.append({"kind": "reference", "reference": values})
        else:
            targets.append({"kind": "document", "document": _json_object(values)})
        setattr(namespace, self.dest, targets)


def _parser() -> _Parser:
    parser = _Parser(
        prog="arc-paper",
        description=(
            "Read, search, import, and cache research papers through a "
            "stable JSON command protocol."
        ),
        epilog=(
            "Durable keyword runs can be inspected with arc-paper status, "
            "stopped with arc-paper stop, and checked with arc-paper validate."
        ),
    )
    commands = parser.add_subparsers(dest="command")

    extract = commands.add_parser(
        "extract-paper-ids",
        help="extract normalized paper identifiers from text",
        description="Extract normalized arXiv and INSPIRE paper identifiers from text.",
    )
    extract.add_argument("text", nargs="+", help="text containing one or more paper IDs")

    safe = commands.add_parser(
        "safe-dir-name",
        help="derive a filesystem-safe name from paper identifiers",
        description="Derive a deterministic filesystem-safe name from paper identifiers.",
    )
    safe.add_argument("ids", nargs="+", help="paper identifiers to include")

    metadata_commands = {
        "get-title": "read a paper title",
        "get-abstract": "read a paper abstract",
        "get-authors": "read a paper author list",
        "get-metadata": "read normalized paper metadata",
        "get-citer-count": "read a paper citation count",
    }
    for name, summary in metadata_commands.items():
        command = commands.add_parser(name, help=summary, description=summary.capitalize() + ".")
        _paper_arguments(command)

    references = commands.add_parser(
        "get-references",
        help="list references cited by a paper",
        description="List references cited by a paper, optionally with enriched metadata.",
    )
    _paper_arguments(references)
    references.add_argument(
        "--enrich", action="store_true", help="include available metadata for each reference"
    )

    citers = commands.add_parser(
        "get-citers",
        help="list papers that cite a paper",
        description="List papers that cite a paper.",
    )
    _paper_arguments(citers)
    citers.add_argument("--limit", type=int, default=1000, help="maximum results (default: 1000)")
    citers.add_argument(
        "--sort",
        choices=("mostrecent", "mostcited"),
        default="mostrecent",
        help="result ordering (default: mostrecent)",
    )

    citer_search = commands.add_parser(
        "search-citers",
        help="shortlist direct citers by title and abstract phrases",
        description=(
            "Scan direct citers and match normalized literal OR phrases in "
            "their titles and abstracts."
        ),
    )
    _paper_arguments(citer_search)
    citer_search.add_argument(
        "--term",
        action="append",
        required=True,
        help="specific search phrase; repeat for synonyms",
    )
    citer_search.add_argument(
        "--scan-limit",
        type=int,
        default=1000,
        help="maximum citers to scan (default: 1000)",
    )
    citer_search.add_argument(
        "--limit",
        type=int,
        default=50,
        help="maximum matching records to return (default: 50)",
    )

    search = commands.add_parser(
        "search-metadata",
        help="search the paper metadata index",
        description="Search paper titles, abstracts, authors, and identifiers.",
    )
    search.add_argument("query", nargs="+", help="metadata search query")
    search.add_argument("--limit", type=int, default=20, help="maximum results (default: 20)")

    full_text = commands.add_parser(
        "search-full-text",
        help="search cached corpus or selected papers",
        description=(
            "Search cached corpus when no target is supplied, or repeat mixed "
            "reference and document targets for a focused search."
        ),
    )
    _search_document_target_arguments(full_text)
    full_text.add_argument(
        "--term", action="append", required=True, help="search phrase; repeat for alternatives"
    )
    full_text.add_argument(
        "--limit", type=int, default=100, help="maximum matches (default: 100)"
    )
    full_text.add_argument(
        "--context-lines", type=int, default=0, help="context lines around each match"
    )
    full_text.add_argument(
        "--case-sensitive", action="store_true", help="match letter case exactly"
    )

    toc = commands.add_parser(
        "get-table-of-contents",
        help="read one paper table of contents",
        description="Resolve one reference or exact cached document and list its sections.",
    )
    _single_document_target_arguments(toc)

    reconstruct_structure = commands.add_parser(
        "reconstruct-cached-structure",
        help="rebuild Markdown hierarchy from an independently cached PDF outline",
        description=(
            "Create a content-addressed structure overlay without changing "
            "either cached document."
        ),
    )
    _cached_document_arguments(reconstruct_structure)
    reconstruct_structure.add_argument(
        "--outline-document-ref",
        required=True,
        type=_json_object,
        help="PDF CachedDocumentRef JSON object",
    )

    section = commands.add_parser(
        "get-section",
        help="read one paper section",
        description="Resolve one reference or exact cached document and select a section.",
    )
    _single_document_target_arguments(section)
    section_selector = section.add_mutually_exclusive_group(required=True)
    section_selector.add_argument(
        "selector", nargs="?", help="section ID or title selector"
    )
    section_selector.add_argument(
        "--ordinal", type=_section_ordinal, help="zero-based section ordinal"
    )

    cached_range = commands.add_parser(
        "read-cached-source-range",
        help="read a verified line range from one cached text source",
        description=(
            "Read one-based inclusive source lines without fetching any provider."
        ),
    )
    _cached_document_arguments(cached_range)
    cached_range.add_argument("start_line", type=int, help="first one-based line")
    cached_range.add_argument("end_line", type=int, help="last one-based line")
    cached_range.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "for Markdown, omit standalone figure markup and recognized "
            "extraction sidecars while preserving other selected lines"
        ),
    )

    lookup_reference = commands.add_parser(
        "lookup-reference",
        help="look up one exact reference identity in the shared cache",
        description="Perform an exact cache-only DOI, arXiv, URL, or title lookup.",
    )
    _reference_identity_arguments(lookup_reference)
    lookup_reference.add_argument(
        "--cache-root", help="override the paper cache directory"
    )

    acquire_reference = commands.add_parser(
        "acquire-reference",
        help="cache-first acquisition of one exact reference",
        description="Acquire one DOI, arXiv, or URL reference and cache it.",
    )
    _reference_identity_arguments(acquire_reference, allow_title=False)
    acquire_reference.add_argument(
        "--refresh", action="store_true", help="refresh upstream data"
    )
    acquire_reference.add_argument(
        "--cache-root", help="override the paper cache directory"
    )

    admit_reference = commands.add_parser(
        "admit-reference",
        help="admit an already downloaded reference file into verified cache",
        description="Cache a local file under one exact reference identity.",
    )
    admit_reference.add_argument("path", help="local reference file")
    _reference_identity_arguments(admit_reference)
    admit_reference.add_argument(
        "--media-type", help="normalized media type override"
    )
    admit_reference.add_argument(
        "--cache-root", help="override the paper cache directory"
    )

    materialize_reference = commands.add_parser(
        "materialize-reference",
        help="write one verified cached resource to an explicit output path",
        description="Verify cached bytes and atomically materialize them.",
    )
    materialize_reference.add_argument(
        "--resource-ref",
        required=True,
        type=_json_object,
        help="CachedResourceRef JSON object",
    )
    materialize_reference.add_argument(
        "--output", required=True, help="explicit output file path"
    )
    materialize_reference.add_argument(
        "--cache-root", help="override the paper cache directory"
    )

    equations = commands.add_parser(
        "search-equations",
        help="search equations in selected papers",
        description="Search equation labels, math, and nearby text across mixed targets.",
    )
    _search_document_target_arguments(equations)
    equations.add_argument(
        "--term", action="append", required=True, help="label, math, or context phrase; repeat for OR"
    )
    equations.add_argument("--limit", type=int, default=20, help="maximum matches (default: 20)")
    equations.add_argument(
        "--context-lines", type=int, default=8, help="PDF layout lines around each match"
    )
    equations.add_argument(
        "--case-sensitive", action="store_true", help="match letter case exactly"
    )

    fetch_commands = {
        "fetch-arxiv-auto": "fetch and cache the best available arXiv source",
        "fetch-arxiv-pdf": "fetch and cache an arXiv PDF",
        "fetch-arxiv-html-bundle": (
            "fetch and cache arXiv HTML with authored image dependencies"
        ),
    }
    for name, summary in fetch_commands.items():
        command = commands.add_parser(name, help=summary, description=summary.capitalize() + ".")
        _paper_arguments(command)
        command.add_argument("--cache-root", help="override the paper cache directory")

    exported_html = commands.add_parser(
        "export-arxiv-html-bundle",
        help="export arXiv HTML with safe authored dependency paths",
        description=(
            "Fetch the preferred arXiv HTML bundle and materialize it for "
            "network-free local document parsing."
        ),
    )
    _paper_arguments(exported_html)
    exported_html.add_argument(
        "--output-dir",
        required=True,
        help="new or empty source-bundle directory",
    )
    exported_html.add_argument(
        "--cache-root", help="override the paper cache directory"
    )

    imported = commands.add_parser(
        "import-source",
        help="import a local source into the content-addressed cache",
        description="Import a local HTML, Markdown, TeX, or PDF source.",
    )
    imported.add_argument("path", help="local source path")
    imported.add_argument(
        "--format",
        dest="source_format",
        choices=("html", "markdown", "tex", "pdf"),
        help="source format override",
    )
    imported.add_argument("--cache-root", help="override the paper cache directory")

    parsed = commands.add_parser(
        "parse-local",
        help="parse a local source with optional validators",
        description="Parse a primary local source and compare optional validator documents.",
    )
    parsed.add_argument("primary_path", help="primary local source path")
    parsed.add_argument(
        "--validator", action="append", default=[], help="validator source path; repeat as needed"
    )
    parsed.add_argument(
        "--format",
        dest="primary_format",
        choices=("html", "markdown", "tex", "pdf"),
        help="primary source format override",
    )
    parsed.add_argument(
        "--validator-format",
        action="append",
        default=[],
        choices=("html", "markdown", "tex", "pdf"),
        help="format for the corresponding --validator",
    )
    parsed.add_argument(
        "--policy",
        choices=("none", "deterministic_only", "visual_all_pages"),
        help="validation policy",
    )
    parsed.add_argument("--cache-root", help="override the paper cache directory")

    exported_rich = commands.add_parser(
        "export-rich-document",
        help="export a portable rich-document workspace for downstream rendering",
        description=(
            "Parse a local Markdown, HTML, or flattened TeX source and export "
            "a portable RichDocument workspace with verified resources."
        ),
    )
    exported_rich.add_argument("source", help="local rich source path")
    exported_rich.add_argument(
        "--output-dir",
        required=True,
        help="new or empty output directory",
    )
    exported_rich.add_argument(
        "--validator",
        help="optional PDF validator path",
    )
    exported_rich.add_argument(
        "--format",
        dest="source_format",
        choices=("html", "markdown", "tex"),
        help="rich source format override",
    )
    exported_rich.add_argument(
        "--cache-root", help="override the paper cache directory"
    )

    keywords = commands.add_parser(
        "extract-keywords",
        help="build an approximate keyword inventory",
        description="Build a durable, approximate keyword inventory for a verified source.",
    )
    keywords.add_argument("source", help="paper identifier or local source")
    keywords.add_argument("--project-dir", required=True, help="durable workflow directory")
    keywords.add_argument(
        "--approx-count", type=_approx_count, default=50, help="target term count (default: 50)"
    )
    keywords.add_argument("--cache-root", help="override the paper cache directory")
    keywords.add_argument("--refresh", action="store_true", help="refresh remote paper data")
    keywords.add_argument("--llm-provider", default="auto", help="LLM provider (default: auto)")
    keywords.add_argument("--model", help="provider-specific model name")
    keywords.add_argument(
        "--model-tier",
        choices=("low", "medium", "high", "xhigh"),
        default="medium",
        help="model reasoning tier (default: medium)",
    )
    keywords.add_argument("--run-id", help="explicit durable run identifier")
    keywords.add_argument("--resume-input", type=_json_object, help="resume response JSON object")
    keywords.add_argument(
        "--structure-ref",
        type=_json_object,
        help="optional CachedDocumentStructureRef JSON object",
    )
    keywords.add_argument(
        "--section-id",
        action="append",
        dest="section_ids",
        help="overlay content section ID to include; repeat as needed",
    )
    keywords.add_argument(
        "--host-authority",
        choices=("unknown", "restricted", "unrestricted"),
        default="unknown",
        help="host permission attestation; unrestricted must be explicit",
    )

    cache = commands.add_parser(
        "cache",
        help="inspect or maintain cached paper data",
        description="Inspect or maintain the local paper cache.",
    )
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_list = cache_commands.add_parser(
        "list",
        help="list cache entries",
        description="List cache entries selected by paper ID, entry ID, or age.",
    )
    _cache_selector_arguments(cache_list)
    cache_list.add_argument(
        "--since", help="only entries updated within a duration such as 12h or 7d"
    )
    cache_remove = cache_commands.add_parser(
        "remove",
        help="preview or remove cache entries",
        description="Preview selected cache removals; pass --yes to apply them.",
    )
    _cache_selector_arguments(cache_remove)
    cache_remove.add_argument("--yes", action="store_true", help="apply the removal")
    cache_update = cache_commands.add_parser(
        "update",
        help="refresh selected cache entries",
        description="Refresh selected cache entries from their upstream sources.",
    )
    _cache_selector_arguments(cache_update)
    cache_export = cache_commands.add_parser(
        "export",
        help="export selected or all cache entries",
        description=(
            "Export exact cache-list entry IDs and their dependencies, or the "
            "whole cache, as a verified tar.gz archive."
        ),
    )
    cache_export.add_argument(
        "entry_ids", nargs="*", help="exact entry ID returned by cache list"
    )
    cache_export.add_argument("--all", action="store_true", dest="all_entries")
    cache_export.add_argument("--output", required=True, help="new .tar.gz archive path")
    cache_export.add_argument("--cache-root", help="override the paper cache directory")
    cache_import = cache_commands.add_parser(
        "import",
        help="import a cache archive",
        description="Validate and merge a cache tar.gz archive.",
    )
    cache_import.add_argument("archive", help="cache tar.gz archive")
    cache_import.add_argument(
        "--replace-conflicts",
        action="store_true",
        help="replace differing destination files after successful preflight",
    )
    cache_import.add_argument("--cache-root", help="override the paper cache directory")

    for name, summary in {
        "status": "inspect a durable keyword run through ac-jobs",
        "stop": "request a durable keyword run stop through ac-jobs",
        "validate": "validate a durable keyword run through ac-jobs",
    }.items():
        commands.add_parser(name, help=summary, description=summary.capitalize() + ".")

    return parser


def _paper_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paper_id", help="arXiv, INSPIRE, or normalized paper identifier")
    parser.add_argument("--refresh", action="store_true", help="refresh cached remote data")


def _arxiv_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("arxiv_id", help="arXiv identifier")
    parser.add_argument("--refresh", action="store_true", help="refresh cached arXiv data")


def _cached_document_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--document-ref",
        required=True,
        type=_json_object,
        help="CachedDocumentRef JSON object",
    )
    parser.add_argument("--cache-root", help="override the paper cache directory")


def _single_document_target_arguments(parser: argparse.ArgumentParser) -> None:
    _search_document_target_arguments(parser)
    _cached_structure_argument(parser)


def _search_document_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(targets=[])
    parser.add_argument(
        "--reference",
        dest="targets",
        action=_DocumentTargetAction,
        help="exact arXiv, DOI, INSPIRE, URL, or cached-title reference",
    )
    parser.add_argument(
        "--document-ref",
        dest="targets",
        action=_DocumentTargetAction,
        help="CachedDocumentRef JSON object",
    )
    parser.add_argument(
        "--source-format",
        choices=("html", "markdown", "tex", "pdf"),
        help="explicit representation for a reference target",
    )
    parser.add_argument("--refresh", action="store_true", help="refresh a reference target")
    parser.add_argument("--cache-root", help="override the paper cache directory")


def _cached_structure_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--structure-ref",
        type=_json_object,
        help="optional CachedDocumentStructureRef JSON object",
    )


def _reference_identity_arguments(
    parser: argparse.ArgumentParser, *, allow_title: bool = True
) -> None:
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--doi", help="exact DOI")
    identity.add_argument("--arxiv-id", help="exact arXiv identifier")
    identity.add_argument("--url", help="exact HTTP(S) URL")
    if allow_title:
        identity.add_argument("--title", help="exact normalized title")


def _cache_selector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--id", dest="paper_ids", action="append", default=[], help="paper ID; repeat as needed"
    )
    parser.add_argument(
        "--entry-id",
        dest="entry_ids",
        action="append",
        default=[],
        help="exact cache entry ID",
    )
    parser.add_argument("--cache-root", help="override the paper cache directory")


def _section_ordinal(value: str) -> int:
    try:
        ordinal = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--ordinal must be a non-negative integer"
        ) from exc
    if ordinal < 0:
        raise argparse.ArgumentTypeError(
            "--ordinal must be a non-negative integer"
        )
    return ordinal


def _approx_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--approx-count must be an integer between 1 and 200"
        ) from exc
    if not 1 <= count <= 200:
        raise argparse.ArgumentTypeError(
            "--approx-count must be an integer between 1 and 200"
        )
    return count


def _parameters(args: argparse.Namespace) -> dict[str, Any]:
    command = args.command
    if command == "extract-paper-ids":
        return {"text": " ".join(args.text)}
    if command == "safe-dir-name":
        return {"ids": args.ids}
    if command in {
        "get-title",
        "get-abstract",
        "get-authors",
        "get-metadata",
        "get-citer-count",
    }:
        return {"paper_id": args.paper_id, "refresh": args.refresh}
    if command == "get-references":
        return {
            "paper_id": args.paper_id,
            "refresh": args.refresh,
            "enrich": args.enrich,
        }
    if command == "get-citers":
        return {
            "paper_id": args.paper_id,
            "refresh": args.refresh,
            "limit": args.limit,
            "sort": args.sort,
        }
    if command == "search-citers":
        return {
            "paper_id": args.paper_id,
            "terms": args.term,
            "refresh": args.refresh,
            "scan_limit": args.scan_limit,
            "limit": args.limit,
        }
    if command == "search-metadata":
        return {"query": " ".join(args.query), "limit": args.limit}
    if command == "search-full-text":
        return {
            "targets": args.targets,
            "terms": args.term,
            "source_format": args.source_format,
            "refresh": args.refresh,
            "limit": args.limit,
            "context_lines": args.context_lines,
            "case_sensitive": args.case_sensitive,
            "cache_root": args.cache_root,
        }
    if command == "get-table-of-contents":
        if len(args.targets) != 1:
            raise _UsageError("get-table-of-contents requires exactly one target")
        return {
            "target": args.targets[0],
            "structure": args.structure_ref,
            "source_format": args.source_format,
            "refresh": args.refresh,
            "cache_root": args.cache_root,
        }
    if command == "reconstruct-cached-structure":
        return {
            "document": args.document_ref,
            "outline_document": args.outline_document_ref,
            "cache_root": args.cache_root,
        }
    if command == "get-section":
        if len(args.targets) != 1:
            raise _UsageError("get-section requires exactly one target")
        return {
            "target": args.targets[0],
            "structure": args.structure_ref,
            "selector": (
                args.ordinal if args.ordinal is not None else args.selector
            ),
            "source_format": args.source_format,
            "refresh": args.refresh,
            "cache_root": args.cache_root,
        }
    if command in {"lookup-reference", "acquire-reference"}:
        values = {
            "doi": args.doi,
            "arxiv_id": args.arxiv_id,
            "url": args.url,
            "cache_root": args.cache_root,
        }
        if command == "acquire-reference":
            values["refresh"] = args.refresh
        else:
            values["title"] = args.title
        return values
    if command == "admit-reference":
        return {
            "path": args.path,
            "doi": args.doi,
            "arxiv_id": args.arxiv_id,
            "url": args.url,
            "title": args.title,
            "media_type": args.media_type,
            "cache_root": args.cache_root,
        }
    if command == "materialize-reference":
        return {
            "resource": args.resource_ref,
            "output": args.output,
            "cache_root": args.cache_root,
        }
    if command == "read-cached-source-range":
        return {
            "document": args.document_ref,
            "start_line": args.start_line,
            "end_line": args.end_line,
            "text_only": args.text_only,
            "cache_root": args.cache_root,
        }
    if command == "search-equations":
        if not args.targets:
            raise _UsageError("search-equations requires at least one target")
        return {
            "targets": args.targets,
            "terms": args.term,
            "source_format": args.source_format,
            "refresh": args.refresh,
            "limit": args.limit,
            "context_lines": args.context_lines,
            "case_sensitive": args.case_sensitive,
            "cache_root": args.cache_root,
        }
    if command in {
        "fetch-arxiv-auto",
        "fetch-arxiv-pdf",
        "fetch-arxiv-html-bundle",
    }:
        return {
            "paper_id": args.paper_id,
            "refresh": args.refresh,
            "cache_root": args.cache_root,
        }
    if command == "export-arxiv-html-bundle":
        return {
            "paper_id": args.paper_id,
            "refresh": args.refresh,
            "output_dir": args.output_dir,
            "cache_root": args.cache_root,
        }
    if command == "import-source":
        return {
            "path": args.path,
            "source_format": args.source_format,
            "cache_root": args.cache_root,
        }
    if command == "parse-local":
        return {
            "primary_path": args.primary_path,
            "validator_paths": args.validator,
            "validator_formats": args.validator_format,
            "primary_format": args.primary_format,
            "policy": args.policy,
            "cache_root": args.cache_root,
        }
    if command == "export-rich-document":
        return {
            "source": args.source,
            "output_dir": args.output_dir,
            "validator": args.validator,
            "source_format": args.source_format,
            "cache_root": args.cache_root,
        }
    if command == "extract-keywords":
        return {
            "source": args.source,
            "project_dir": args.project_dir,
            "approx_count": args.approx_count,
            "cache_root": args.cache_root,
            "refresh": args.refresh,
            "llm_provider": args.llm_provider,
            "model": args.model,
            "model_tier": args.model_tier,
            "run_id": args.run_id,
            "resume_input": args.resume_input,
            "structure_ref": args.structure_ref,
            "section_ids": args.section_ids,
            "host_authority": args.host_authority,
        }
    if command == "cache":
        if args.cache_command == "export":
            if args.all_entries == bool(args.entry_ids):
                raise _UsageError(
                    "cache export requires either --all or at least one exact entry ID"
                )
            return {
                "output": args.output,
                "entry_ids": args.entry_ids,
                "all_entries": args.all_entries,
                "cache_root": args.cache_root,
            }
        if args.cache_command == "import":
            return {
                "archive": args.archive,
                "replace_conflicts": args.replace_conflicts,
                "cache_root": args.cache_root,
            }
        values: dict[str, Any] = {
            "paper_ids": args.paper_ids,
            "entry_ids": args.entry_ids,
            "cache_root": args.cache_root,
        }
        if args.cache_command == "list":
            values["since_seconds"] = _duration_seconds(args.since)
        elif args.cache_command == "remove":
            values["dry_run"] = not args.yes
        return values
    raise _UsageError(f"unsupported command: {command}")


def _duration_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.fullmatch(r"([1-9][0-9]*)([smhdw])", value)
    if match is None:
        raise _UsageError(
            "--since must be a positive integer followed by s, m, h, d, or w"
        )
    factors = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    seconds = int(match.group(1)) * factors[match.group(2)]
    if seconds > 36500 * 86400:
        raise _UsageError("--since cannot exceed 36500d")
    return seconds


def _json_object(value: str) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--resume-input must be a JSON object: {exc.msg}"
        ) from exc
    if not isinstance(document, dict):
        raise argparse.ArgumentTypeError(
            "--resume-input must be a JSON object"
        )
    return document


def _result_data(value: Any) -> Mapping[str, Any]:
    encoded = to_json_value(value)
    if isinstance(encoded, Mapping):
        return dict(encoded)
    return {"result": encoded}


def _emit(result: CommandResult, *, exit_code: int) -> int:
    sys.stdout.write(command_result_json(result) + "\n")
    return exit_code


def _help_command(arguments: list[str]) -> str:
    parts = ["arc-paper"]
    commands = _parser()._subparsers._group_actions[0].choices
    if arguments and arguments[0] in commands:
        parts.append(arguments[0])
        if arguments[0] == "cache" and len(arguments) > 1:
            cache_commands = commands["cache"]._subparsers._group_actions[0].choices
            if arguments[1] in cache_commands:
                parts.append(arguments[1])
    return " ".join((*parts, "--help"))


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"status", "stop", "validate"}:
        # ac-jobs is the sole implementation of generic durable-run controls.
        return run_control_main(arguments, prog="arc-paper")
    parser = _parser()
    try:
        args = parser.parse_args(arguments)
        if args.command is None:
            parser.print_help()
            return 0
        operation = (
            f"cache-{args.cache_command}"
            if args.command == "cache"
            else args.command
        )
        value = dispatch_operation(operation, _parameters(args))
        warnings: tuple[CommandWarning, ...] = ()
        raw_warnings = (
            value.get("warnings", ())
            if isinstance(value, Mapping)
            else getattr(value, "warnings", ())
        )
        if raw_warnings:
            warnings = tuple(
                CommandWarning("paper_warning", str(item)) for item in raw_warnings
            )
        return _emit(
            CommandResult(
                CommandStatus.COMPLETED,
                data=_result_data(value),
                warnings=warnings,
            ),
            exit_code=0,
        )
    except KeywordExtractionPaused as exc:
        return _emit(
            command_result_from_snapshot(exc.snapshot),
            exit_code=0,
        )
    except _HelpRequested:
        return 0
    except _UsageError as exc:
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError(
                    "invalid_request",
                    str(exc),
                    {"help_command": _help_command(arguments)},
                ),
            ),
            exit_code=2,
        )
    except OSError as exc:
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError("local_io_error", str(exc)),
            ),
            exit_code=1,
        )
    except Exception as exc:
        raw_code = getattr(exc, "code", None)
        code = (
            raw_code
            if isinstance(raw_code, str) and raw_code
            else "internal_error"
        )
        message = str(getattr(exc, "message", str(exc)))
        raw_paths = getattr(exc, "paths", ())
        details = (
            {"paths": list(raw_paths)}
            if isinstance(raw_paths, tuple) and raw_paths
            else {}
        )
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError(code, message, details),
            ),
            exit_code=1,
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
