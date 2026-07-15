#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "arc.workflow.domain_manifest.v1"
SUMMARY_SUFFIX = "_domain_summary.json"


class ManifestError(ValueError):
    pass


def build_domain_manifest(project_dir: Path) -> dict[str, Any]:
    project_dir = project_dir.expanduser().resolve()
    context_path = project_dir / "context.json"
    domain_dir = project_dir / "domain"
    context = _read_object(context_path)
    seed_by_domain = _seed_by_domain(context)
    if not domain_dir.is_dir():
        raise ManifestError(f"domain directory does not exist: {domain_dir}")

    domains: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    seen: dict[str, Path] = {}
    for summary_path in sorted(domain_dir.glob(f"*{SUMMARY_SUFFIX}")):
        summary = _read_object(summary_path)
        domain_id = _required_text(summary, "domain_id", summary_path)
        if domain_id in seen:
            duplicates.append(
                {
                    "domain_id": domain_id,
                    "kept_summary_json_path": _relative(project_dir, seen[domain_id]),
                    "duplicate_summary_json_path": _relative(project_dir, summary_path),
                }
            )
            continue

        prefix = summary_path.name[: -len(SUMMARY_SUFFIX)]
        markdown_path = domain_dir / f"{prefix}_domain_summary.md"
        paper_pack_path = domain_dir / f"{prefix}_paper_json_pack.json"
        for required_path in (markdown_path, paper_pack_path):
            if not required_path.is_file():
                raise ManifestError(f"required domain artifact does not exist: {required_path}")

        foundation = summary.get("foundation_paper")
        if not isinstance(foundation, dict):
            foundation = {}
        seed_paper = seed_by_domain.get(domain_id, "")
        if not seed_paper:
            seed_paper = str(foundation.get("paper_id", "")).strip()
        if not seed_paper:
            seed_paper = prefix

        domains.append(
            {
                "domain_id": domain_id,
                "seed_paper": seed_paper,
                "title": _required_text(summary, "domain_title", summary_path),
                "summary_json_path": _relative(project_dir, summary_path),
                "summary_markdown_path": _relative(project_dir, markdown_path),
                "paper_json_pack_path": _relative(project_dir, paper_pack_path),
            }
        )
        seen[domain_id] = summary_path

    if not domains:
        raise ManifestError(f"no {SUMMARY_SUFFIX} files found in {domain_dir}")

    requested = context.get("seed_paper_list", [])
    if not isinstance(requested, list):
        requested = []
    requested_strings = [str(item) for item in requested]
    requested_order = {seed: index for index, seed in enumerate(requested_strings)}
    domains.sort(
        key=lambda item: (
            requested_order.get(item["seed_paper"], len(requested_order)),
            item["domain_id"],
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "user_intent": str(context.get("user_intent", "")).strip(),
        "requested_seed_papers": requested_strings,
        "domain_count": len(domains),
        "domains": domains,
        "duplicates": duplicates,
    }


def write_domain_manifest(project_dir: Path, output: Path | None = None) -> Path:
    project_dir = project_dir.expanduser().resolve()
    destination = output.expanduser().resolve() if output else project_dir / "domain" / "domain-manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_domain_manifest(project_dir)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"required JSON file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"JSON root must be an object: {path}")
    return payload


def _required_text(payload: dict[str, Any], key: str, path: Path) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ManifestError(f"{path} is missing required field {key}")
    return value


def _seed_by_domain(context: dict[str, Any]) -> dict[str, str]:
    raw_records = context.get("domain_records", [])
    if not isinstance(raw_records, list):
        raise ManifestError("context.json domain_records must be an array")
    result: dict[str, str] = {}
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ManifestError(f"context.json domain_records[{index}] must be an object")
        domain_id = str(record.get("domain_id", "")).strip()
        seed_paper = str(record.get("seed_paper", "")).strip()
        if not domain_id or not seed_paper:
            raise ManifestError(
                f"context.json domain_records[{index}] requires domain_id and seed_paper"
            )
        if domain_id in result and result[domain_id] != seed_paper:
            raise ManifestError(f"conflicting requested seeds recorded for domain {domain_id}")
        result[domain_id] = seed_paper
    return result


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ManifestError(f"domain artifact must be inside project directory: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a project-local ARC domain manifest.")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        destination = write_domain_manifest(
            Path(args.project_dir),
            Path(args.output) if args.output else None,
        )
        payload = json.loads(destination.read_text(encoding="utf-8"))
        result = {
            "status": "completed",
            "manifest_path": str(destination),
            "domain_count": payload["domain_count"],
            "duplicate_count": len(payload["duplicates"]),
        }
        print(json.dumps(result, ensure_ascii=False) if args.json else str(destination))
        return 0
    except ManifestError as exc:
        result = {"status": "failed", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False) if args.json else f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
