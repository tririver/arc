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
from .results import ok


def main(argv: list[str] | None = None) -> int:
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
    llm_summary_prefixed = _paper_command(sub, "llm-summary")
    llm_summary_prefixed.add_argument("--provider", default="auto")
    llm_summary_prefixed.add_argument("--model", default=None)

    generate = sub.add_parser("generate-llm-summary")
    generate.add_argument("paper_ids", nargs="+")
    generate.add_argument("--provider", default="auto")
    generate.add_argument("--model", default=None)
    generate.add_argument("--refresh", action="store_true")
    generate.add_argument("--json", action="store_true")
    llm_generate = sub.add_parser("llm-generate-summary")
    llm_generate.add_argument("paper_ids", nargs="+")
    llm_generate.add_argument("--provider", default="auto")
    llm_generate.add_argument("--model", default=None)
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
    search_full_text.add_argument("--context", type=int, default=0)
    search_full_text.add_argument("--case-sensitive", action="store_true")
    search_full_text.add_argument("--json", action="store_true")

    equation = sub.add_parser("get-equation-context")
    equation.add_argument("paper_ids", nargs="+")
    equation.add_argument("--query", required=True)
    equation.add_argument("--refresh", action="store_true")
    equation.add_argument("--json", action="store_true")

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
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-items", type=int, default=None)
    run.add_argument("--json", action="store_true")
    export = batch_sub.add_parser("export")
    export.add_argument("name")
    export.add_argument("--output", required=True)
    export.add_argument("--format", default="jsonl", choices=["jsonl"])
    export.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    result = _dispatch(args)
    _print_json(result)
    return 0


def _paper_command(sub, name: str) -> argparse.ArgumentParser:
    parser = sub.add_parser(name)
    parser.add_argument("paper_ids", nargs="+")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _dispatch(args: argparse.Namespace) -> Any:
    command = args.command
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
        return service.get_llm_summary(paper_ids, provider=args.provider, model=args.model, refresh=args.refresh)
    if command in {"generate-llm-summary", "llm-generate-summary"}:
        return service.generate_llm_summary(
            paper_ids,
            provider=args.provider,
            model=args.model,
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


if __name__ == "__main__":
    raise SystemExit(main())
