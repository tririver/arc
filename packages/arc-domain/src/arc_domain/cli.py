from __future__ import annotations

import argparse
import json
from typing import Any

from . import service


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ARC research-domain artifacts from a seed paper")
    sub = parser.add_subparsers(dest="command", required=True)

    init = _seed_command(sub, "init")
    foundation = _seed_command(sub, "identify-foundation")
    _llm_args(foundation)
    llm_foundation = _seed_command(sub, "llm-identify-foundation")
    _llm_args(llm_foundation)
    network = _seed_command(sub, "build-network")
    _llm_args(network)
    llm_network = _seed_command(sub, "llm-build-network")
    _llm_args(llm_network)
    evidence = _seed_command(sub, "build-evidence")
    summary = _seed_command(sub, "summarize")
    _llm_args(summary)
    llm_summary = _seed_command(sub, "llm-summarize")
    _llm_args(llm_summary)
    build = _seed_command(sub, "build")
    _llm_args(build)
    llm_build = _seed_command(sub, "llm-build")
    _llm_args(llm_build)
    status = sub.add_parser("status")
    status.add_argument("seed_paper", nargs="?")
    status.add_argument("--intent", default="")
    status.add_argument("--domain-id", default=None)
    status.add_argument("--json", action="store_true")
    get_summary = sub.add_parser("get-summary")
    get_summary.add_argument("seed_paper", nargs="?")
    get_summary.add_argument("--intent", default="")
    get_summary.add_argument("--domain-id", default=None)
    get_summary.add_argument("--json", action="store_true")
    get_graph = sub.add_parser("get-graph")
    get_graph.add_argument("seed_paper", nargs="?")
    get_graph.add_argument("--intent", default="")
    get_graph.add_argument("--domain-id", default=None)
    get_graph.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    result = _dispatch(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _seed_command(sub, name: str) -> argparse.ArgumentParser:
    parser = sub.add_parser(name)
    parser.add_argument("seed_paper")
    parser.add_argument("--intent", default="")
    parser.add_argument("--domain-id", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser


def _llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--model", default=None)


def _dispatch(args: argparse.Namespace) -> Any:
    if args.command == "init":
        return service.init_domain(args.seed_paper, intent=args.intent, domain_id=args.domain_id)
    if args.command in {"identify-foundation", "llm-identify-foundation"}:
        return service.identify_foundation(
            args.seed_paper,
            intent=args.intent,
            domain_id=args.domain_id,
            provider=args.provider,
            model=args.model,
            refresh=args.refresh,
            workers=args.workers,
        )
    if args.command in {"build-network", "llm-build-network"}:
        return service.build_network(
            args.seed_paper,
            intent=args.intent,
            domain_id=args.domain_id,
            provider=args.provider,
            model=args.model,
            refresh=args.refresh,
            workers=args.workers,
        )
    if args.command == "build-evidence":
        return service.build_evidence_pack(
            args.seed_paper,
            intent=args.intent,
            domain_id=args.domain_id,
            refresh=args.refresh,
            workers=args.workers,
        )
    if args.command in {"summarize", "llm-summarize"}:
        return service.summarize_domain(
            args.seed_paper,
            intent=args.intent,
            domain_id=args.domain_id,
            provider=args.provider,
            model=args.model,
        )
    if args.command in {"build", "llm-build"}:
        return service.build_domain(
            args.seed_paper,
            intent=args.intent,
            domain_id=args.domain_id,
            provider=args.provider,
            model=args.model,
            refresh=args.refresh,
            workers=args.workers,
        )
    if args.command == "status":
        return service.status(args.seed_paper, intent=args.intent, domain_id=args.domain_id)
    if args.command == "get-summary":
        return service.get_domain_summary(args.seed_paper, intent=args.intent, domain_id=args.domain_id)
    if args.command == "get-graph":
        return service.get_domain_graph(args.seed_paper, intent=args.intent, domain_id=args.domain_id)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
