from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import service
from .batch.db import BatchDB
from .batch.runner import export_batch, prefetch_batch, run_batch
from .host import detect_host, select_llm_provider
from .results import err, ok
from .summary.model import DEFAULT_SUMMARY_MODEL_TIER
from .worker_guard import in_worker_context, wrapper_call_authorized


def main(argv: list[str] | None = None) -> int:
    if in_worker_context() and not wrapper_call_authorized():
        result = err(
            "paper_worker_wrapper_required",
            "Model workers must use arc-paper-worker instead of the unrestricted arc-paper entry point",
        )
        result["status"] = "error"
        _print_json(result)
        return 1
    parser = argparse.ArgumentParser(description="Cache-first ar5iv and INSPIRE paper query tools")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract-paper-ids", aliases=["extract-ids"])
    extract.add_argument("text", nargs="*")
    extract.add_argument("--file", default=None)
    extract.add_argument("--json", action="store_true")

    safe_dir = sub.add_parser("safe-dir-name", aliases=["paper-ids-safe-dir-name"])
    safe_dir.add_argument("paper_ids", nargs="+")
    safe_dir.add_argument("--json", action="store_true")

    infer_refs = sub.add_parser("llm-infer-main-references", aliases=["infer-main-references"])
    infer_refs.add_argument("text", nargs="*")
    infer_refs.add_argument("--file", default=None)
    infer_refs.add_argument("--provider", default="auto")
    infer_refs.add_argument("--model", default=None)
    infer_refs.add_argument("--refresh", action="store_true")
    infer_refs.add_argument("--json", action="store_true")

    _paper_command(sub, "get-title")
    _paper_command(sub, "get-abstract")
    _paper_command(sub, "get-authors")
    _paper_command(sub, "get-metadata")
    references = _paper_command(sub, "get-references")
    references.add_argument("--enrich", action="store_true")
    citers = _paper_command(sub, "get-citers")
    citers.add_argument("--limit", type=int, default=1000)
    citers.add_argument("--sort", default="mostrecent", choices=["mostrecent", "mostcited"])
    _paper_command(sub, "get-citer-count")
    _paper_command(sub, "get-toc")
    llm_summary = _paper_command(sub, "get-llm-summary")
    llm_summary.add_argument("--provider", default="auto")
    llm_summary.add_argument("--model", default=None)
    llm_summary.add_argument(
        "--model-tier", choices=["max", "high", "medium", "low"], default=DEFAULT_SUMMARY_MODEL_TIER
    )
    llm_summary_prefixed = _paper_command(sub, "llm-summary")
    llm_summary_prefixed.add_argument("--provider", default="auto")
    llm_summary_prefixed.add_argument("--model", default=None)
    llm_summary_prefixed.add_argument(
        "--model-tier", choices=["max", "high", "medium", "low"], default=DEFAULT_SUMMARY_MODEL_TIER
    )

    generate = sub.add_parser("generate-llm-summary")
    generate.add_argument("paper_ids", nargs="+")
    generate.add_argument("--provider", default="auto")
    generate.add_argument("--model", default=None)
    generate.add_argument(
        "--model-tier", choices=["max", "high", "medium", "low"], default=DEFAULT_SUMMARY_MODEL_TIER
    )
    generate.add_argument("--refresh", action="store_true")
    generate.add_argument("--json", action="store_true")
    llm_generate = sub.add_parser("llm-generate-summary")
    llm_generate.add_argument("paper_ids", nargs="+")
    llm_generate.add_argument("--provider", default="auto")
    llm_generate.add_argument("--model", default=None)
    llm_generate.add_argument(
        "--model-tier", choices=["max", "high", "medium", "low"], default=DEFAULT_SUMMARY_MODEL_TIER
    )
    llm_generate.add_argument("--refresh", action="store_true")
    llm_generate.add_argument("--json", action="store_true")

    store = sub.add_parser("store-llm-summary")
    store.add_argument("paper_id")
    store.add_argument("--summary-json", required=True)
    store.add_argument("--json", action="store_true")

    section = sub.add_parser("get-section")
    section.add_argument("paper_ids", nargs="+")
    section.add_argument("--section", required=True)
    section.add_argument("--refresh", action="store_true")
    section.add_argument("--json", action="store_true")

    search_full_text = sub.add_parser("search-full-text")
    search_full_text.add_argument("paper_ids", nargs="*")
    search_full_text.add_argument("--query", required=True)
    search_full_text.add_argument("--refresh", action="store_true")
    search_full_text.add_argument("--limit", type=int, default=20)
    search_full_text.add_argument("--context", type=int, default=1)
    search_full_text.add_argument("--case-sensitive", action="store_true")
    search_full_text.add_argument("--json", action="store_true")

    equation = sub.add_parser("get-equation-context")
    equation.add_argument("paper_ids", nargs="+")
    equation.add_argument("--query", required=True)
    equation.add_argument("--refresh", action="store_true")
    equation.add_argument("--json", action="store_true")

    source_cache = sub.add_parser("source-cache")
    source_cache.add_argument("paper_id")
    source_cache.add_argument("--version", type=int, required=True)
    source_cache.add_argument("--license", dest="license_url", default="")
    source_cache.add_argument("--refresh", action="store_true")
    source_cache.add_argument("--json", action="store_true")
    source_probe = sub.add_parser("source-probe")
    source_probe.add_argument("paper_id")
    source_probe.add_argument("--version", type=int, required=True)
    source_probe.add_argument("--json", action="store_true")

    parse = sub.add_parser("parse")
    parse.add_argument("source_path", nargs="?")
    parse.add_argument(
        "--source",
        default="auto",
        choices=["auto", "ar5iv", "html", "tex", "markdown", "pdf", "tex-pdf", "markdown-pdf"],
    )
    parse.add_argument("--id", dest="source_id", default=None)
    parse.add_argument("--paper-id", default=None)
    parse.add_argument("--html", default=None)
    parse.add_argument("--tex", default=None)
    parse.add_argument("--markdown", "--md", dest="markdown", default=None)
    parse.add_argument("--pdf", default=None)
    parse.add_argument("--refresh", action="store_true")
    parse.add_argument("--recache", action="store_true")
    parse.add_argument("--include-document", action="store_true")
    parse.add_argument(
        "--document-kind", choices=["auto", "article", "book"], default="auto"
    )
    parse.add_argument("--json", action="store_true")
    get_parsed = sub.add_parser("get-parsed")
    get_parsed.add_argument("source_id")
    get_parsed.add_argument("--include-document", action="store_true")
    get_parsed.add_argument("--json", action="store_true")
    get_parsed_toc = sub.add_parser("get-parsed-toc")
    get_parsed_toc.add_argument("source_id")
    get_parsed_toc.add_argument("--json", action="store_true")
    get_parsed_section = sub.add_parser("get-parsed-section")
    get_parsed_section.add_argument("source_id")
    get_parsed_section.add_argument("--section", required=True)
    get_parsed_section.add_argument("--json", action="store_true")
    get_parsed_equations = sub.add_parser("get-parsed-equations")
    get_parsed_equations.add_argument("source_id")
    get_parsed_equations.add_argument("--json", action="store_true")
    get_parsed_equation = sub.add_parser("get-parsed-equation")
    get_parsed_equation.add_argument("source_id")
    get_parsed_equation.add_argument("--equation-id", required=True)
    get_parsed_equation.add_argument("--json", action="store_true")
    mark_parsed_equation = sub.add_parser("mark-parsed-equation")
    mark_parsed_equation.add_argument("source_id")
    mark_parsed_equation.add_argument("--equation-id", required=True)
    mark_parsed_equation.add_argument(
        "--status",
        default="problematic",
        choices=["problematic", "needs_recache", "resolved"],
    )
    mark_parsed_equation.add_argument("--reason", required=True)
    mark_parsed_equation.add_argument("--json", action="store_true")
    search_parsed = sub.add_parser("search-parsed")
    search_parsed.add_argument("source_id")
    search_parsed.add_argument("--query", required=True)
    search_parsed.add_argument("--limit", type=int, default=20)
    search_parsed.add_argument("--case-sensitive", action="store_true")
    search_parsed.add_argument("--json", action="store_true")
    cache_cmd = sub.add_parser("cache")
    cache_sub = cache_cmd.add_subparsers(dest="cache_command", required=True)
    cache_list = cache_sub.add_parser("list")
    _cache_filter_args(cache_list, include_all=False)
    cache_list.add_argument("--json", action="store_true")
    cache_remove = cache_sub.add_parser("remove")
    _cache_filter_args(cache_remove, include_all=True)
    cache_remove.add_argument("--dry-run", action="store_true")
    cache_remove.add_argument("--yes", action="store_true")
    cache_remove.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor")
    doctor_sub = doctor.add_subparsers(dest="doctor_command", required=True)
    for name in ("host", "provider"):
        p = doctor_sub.add_parser(name)
        p.add_argument("--json", action="store_true")
    cache = doctor_sub.add_parser("cache")
    cache.add_argument("paper_id", nargs="?")
    cache.add_argument("--json", action="store_true")

    batch = sub.add_parser("summary-batch")
    batch_sub = batch.add_subparsers(dest="batch_command", required=True)
    create = batch_sub.add_parser("create")
    create.add_argument("papers_file")
    create.add_argument("--name", required=True)
    create.add_argument("--json", action="store_true")
    status = batch_sub.add_parser("status")
    status.add_argument("name")
    status.add_argument("--json", action="store_true")
    retry = batch_sub.add_parser("retry-failed")
    retry.add_argument("name")
    retry.add_argument("--json", action="store_true")
    prefetch = batch_sub.add_parser("prefetch")
    prefetch.add_argument("name")
    prefetch.add_argument("--workers", type=int, default=4)
    prefetch.add_argument("--json", action="store_true")
    run = batch_sub.add_parser("run")
    run.add_argument("name")
    run.add_argument("--provider", default="auto")
    run.add_argument("--model", default=None)
    run.add_argument(
        "--model-tier", choices=["max", "high", "medium", "low"], default=DEFAULT_SUMMARY_MODEL_TIER
    )
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-items", type=int, default=None)
    run.add_argument("--json", action="store_true")
    export = batch_sub.add_parser("export")
    export.add_argument("name")
    export.add_argument("--output", required=True)
    export.add_argument("--format", default="jsonl", choices=["jsonl"])
    export.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except Exception as exc:
        if not getattr(args, "json", False):
            raise
        result = _exception_result(exc)
    _print_warnings(result)
    _print_json(result)
    return _exit_code(result)


def _paper_command(sub, name: str) -> argparse.ArgumentParser:
    parser = sub.add_parser(name)
    parser.add_argument("paper_ids", nargs="+")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _exit_code(result: Any) -> int:
    """Map ARC result envelopes to a process status for agent callers."""
    if isinstance(result, dict) and result.get("ok") is False:
        if result.get("status") == "needs_llm":
            return 0
        return 1
    return 0


def _exception_result(exc: Exception) -> dict[str, Any]:
    """Return the stable failure envelope promised by ``--json`` commands."""
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


def _cache_filter_args(parser: argparse.ArgumentParser, *, include_all: bool) -> None:
    parser.add_argument("--id", dest="ids", action="append", default=None)
    parser.add_argument("--since", default=None)
    parser.add_argument("--older-than", default=None)
    parser.add_argument("--past-hour", dest="since", action="store_const", const="1h")
    parser.add_argument("--past-day", dest="since", action="store_const", const="1d")
    if include_all:
        parser.add_argument("--all", dest="all_items", action="store_true")


def _dispatch(args: argparse.Namespace) -> Any:
    command = args.command
    if command == "cache":
        if args.cache_command == "list":
            return service.list_cached_papers(ids=args.ids, since=args.since, older_than=args.older_than)
        if args.cache_command == "remove":
            if args.dry_run:
                result = service.remove_cached_papers(
                    ids=args.ids,
                    since=args.since,
                    older_than=args.older_than,
                    all_items=args.all_items,
                    dry_run=True,
                )
                _print_cache_remove_preview(result)
                return result
            if args.yes:
                result = service.remove_cached_papers(
                    ids=args.ids,
                    since=args.since,
                    older_than=args.older_than,
                    all_items=args.all_items,
                    dry_run=False,
                )
                _print_cache_remove_preview(result)
                return result
            preview = service.remove_cached_papers(
                ids=args.ids,
                since=args.since,
                older_than=args.older_than,
                all_items=args.all_items,
                dry_run=True,
            )
            _print_cache_remove_preview(preview)
            if not preview.get("ok"):
                return preview
            if not (preview.get("data") or {}).get("items"):
                return preview
            if not _confirm_cache_remove():
                return ok(
                    {
                        "cancelled": True,
                        "items": (preview.get("data") or {}).get("items") or [],
                        "removed_count": 0,
                        "removed_paths": [],
                    },
                    provider="local-cache",
                    confirmed=False,
                )
            return service.remove_cached_papers(
                ids=args.ids,
                since=args.since,
                older_than=args.older_than,
                all_items=args.all_items,
                dry_run=False,
            )
        return err("cache_command_invalid", f"Unknown cache command {args.cache_command!r}")
    if command == "doctor":
        if args.doctor_command == "host":
            detected = detect_host()
            return ok(
                {
                    "host": detected.host,
                    "confidence": detected.confidence,
                    "signals": detected.signals,
                }
            )
        if args.doctor_command == "cache":
            return service.doctor_cache(args.paper_id)
        selected = select_llm_provider()
        return ok(
            {
                "provider": selected.provider,
                "host": selected.host.host,
                "confidence": selected.host.confidence,
                "signals": selected.signals,
            }
        )
    if command == "summary-batch":
        db = BatchDB.default()
        if args.batch_command == "create":
            paper_ids = _read_papers_file(args.papers_file)
            db.create_batch(args.name, paper_ids, "paper-summary-v1")
            return ok({"batch": args.name, "counts": db.status_counts(args.name)})
        if args.batch_command == "status":
            return ok({"batch": args.name, "counts": db.status_counts(args.name)})
        if args.batch_command == "retry-failed":
            db.retry_failed(args.name)
            return ok({"batch": args.name, "counts": db.status_counts(args.name)})
        if args.batch_command == "prefetch":
            return ok(prefetch_batch(args.name, workers=args.workers, db=db))
        if args.batch_command == "run":
            return ok(
                run_batch(
                    args.name,
                    provider=args.provider,
                    model=args.model,
                    model_tier=args.model_tier,
                    concurrency=args.concurrency,
                    max_items=args.max_items,
                    db=db,
                )
            )
        if args.batch_command == "export":
            return ok(export_batch(args.name, output=Path(args.output), db=db))
    if command == "store-llm-summary":
        return service.store_llm_summary(args.paper_id, _read_summary_arg(args.summary_json))
    if command in {"extract-paper-ids", "extract-ids"}:
        return service.extract_paper_ids(_read_text_arg(args))
    if command in {"safe-dir-name", "paper-ids-safe-dir-name"}:
        return service.paper_ids_safe_dir_name(args.paper_ids)
    if command in {"llm-infer-main-references", "infer-main-references"}:
        return service.llm_infer_main_references(
            _read_text_arg(args),
            provider=args.provider,
            model=args.model,
            refresh=args.refresh,
        )
    if command == "parse":
        kwargs = {
            "source": args.source,
            "source_id": args.source_id,
            "paper_id": args.paper_id,
            "html_path": args.html,
            "tex_path": args.tex,
            "pdf_path": args.pdf,
            "refresh": args.refresh,
        }
        if args.document_kind != "auto":
            kwargs["document_kind"] = args.document_kind
        if args.markdown:
            kwargs["markdown_path"] = args.markdown
        if args.recache:
            kwargs["recache"] = True
        if args.include_document:
            kwargs["include_document"] = True
        return service.parse_source(args.source_path, **kwargs)
    if command == "source-cache":
        return service.cache_arxiv_source(
            args.paper_id,
            version=args.version,
            refresh=args.refresh,
            license_url=args.license_url,
        )
    if command == "source-probe":
        return service.probe_arxiv_source(args.paper_id, version=args.version)
    if command == "get-parsed":
        if args.include_document:
            return service.get_parsed_source(args.source_id, include_document=True)
        return service.get_parsed_source(args.source_id)
    if command == "get-parsed-toc":
        return service.get_parsed_source_toc(args.source_id)
    if command == "get-parsed-section":
        return service.get_parsed_source_section(args.source_id, args.section)
    if command == "get-parsed-equations":
        return service.get_parsed_source_equations(args.source_id)
    if command == "get-parsed-equation":
        return service.get_parsed_source_equation(args.source_id, args.equation_id)
    if command == "mark-parsed-equation":
        return service.mark_parsed_equation(
            args.source_id,
            args.equation_id,
            status=args.status,
            reason=args.reason,
        )
    if command == "search-parsed":
        return service.search_parsed_source(
            args.source_id,
            query=args.query,
            limit=args.limit,
            case_sensitive=args.case_sensitive,
        )
    paper_ids = args.paper_ids[0] if len(args.paper_ids) == 1 else args.paper_ids
    if command == "get-title":
        return service.get_title(paper_ids, refresh=args.refresh)
    if command == "get-abstract":
        return service.get_abstract(paper_ids, refresh=args.refresh)
    if command == "get-authors":
        return service.get_authors(paper_ids, refresh=args.refresh)
    if command == "get-metadata":
        return service.get_metadata(paper_ids, refresh=args.refresh)
    if command == "get-references":
        return service.get_references(paper_ids, refresh=args.refresh, enrich=args.enrich)
    if command == "get-citers":
        return service.get_citers(paper_ids, refresh=args.refresh, limit=args.limit, sort=args.sort)
    if command == "get-citer-count":
        return service.get_citer_count(paper_ids, refresh=args.refresh)
    if command == "get-toc":
        return service.get_toc(paper_ids, refresh=args.refresh)
    if command == "get-section":
        return service.get_section(paper_ids, args.section, refresh=args.refresh)
    if command == "search-full-text":
        search_ids = None
        if args.paper_ids:
            search_ids = args.paper_ids[0] if len(args.paper_ids) == 1 else args.paper_ids
        return service.search_full_text(
            search_ids,
            query=args.query,
            refresh=args.refresh,
            limit=args.limit,
            context=args.context,
            case_sensitive=args.case_sensitive,
        )
    if command == "get-equation-context":
        return service.get_equation_context(paper_ids, args.query, refresh=args.refresh)
    if command in {"get-llm-summary", "llm-summary"}:
        return service.get_llm_summary(
            paper_ids,
            provider=args.provider,
            model=args.model,
            model_tier=args.model_tier,
            refresh=args.refresh,
        )
    if command in {"generate-llm-summary", "llm-generate-summary"}:
        return service.generate_llm_summary(
            paper_ids,
            provider=args.provider,
            model=args.model,
            model_tier=args.model_tier,
            refresh=args.refresh,
        )
    raise AssertionError(f"Unhandled command: {command}")


def _read_summary_arg(value: str) -> dict[str, Any]:
    if value == "-":
        return json.loads(sys.stdin.read())
    with open(value, encoding="utf-8") as handle:
        return json.load(handle)


def _read_text_arg(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            parts.append(handle.read())
    if args.text:
        parts.append(" ".join(args.text))
    if not parts and not sys.stdin.isatty():
        parts.append(sys.stdin.read())
    return "\n".join(parts)


def _read_papers_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _print_cache_remove_preview(result: Any) -> None:
    if not isinstance(result, dict):
        return
    data = result.get("data")
    if not isinstance(data, dict):
        return
    items = data.get("items") or []
    if not items:
        print("No cached papers selected for removal.", file=sys.stderr)
        return
    print("Cached papers selected for removal:", file=sys.stderr)
    for item in items:
        paper_id = str(item.get("paper_id") or "<unknown>")
        kinds = ", ".join(str(kind) for kind in item.get("kinds") or [])
        print(f"- {paper_id} [{kinds}]", file=sys.stderr)
        for path_info in item.get("paths") or []:
            path = str(path_info.get("path") or "")
            kind = str(path_info.get("kind") or "cache")
            if path:
                print(f"  - {kind}: {path}", file=sys.stderr)


def _confirm_cache_remove() -> bool:
    print("Remove cached papers? [y/N] ", end="", file=sys.stderr, flush=True)
    answer = sys.stdin.readline().strip().lower()
    return answer == "y"


def _print_warnings(data: Any) -> None:
    if not isinstance(data, dict):
        return
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return
    for warning in meta.get("warnings") or []:
        if not isinstance(warning, dict):
            continue
        message = str(warning.get("message") or warning.get("code") or "").strip()
        if not message:
            continue
        pdf_path = str(warning.get("pdf_path") or "").strip()
        suffix = f" ({pdf_path})" if pdf_path else ""
        print(f"WARNING: {message}{suffix}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
