#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "arc.workflow.domain_manifest.v2"
GROUPING_SCHEMA_VERSION = "arc.workflow.domain_field_grouping.v1"
HARD_SEPARATION_CONFIDENCE = 0.80
SUMMARY_SUFFIX = "_domain_summary.json"


class ManifestError(ValueError):
    pass


class GroupingConstraintError(ManifestError):
    pass


def build_domain_manifest(project_dir: Path, *, grouping_result: dict[str, Any] | None = None) -> dict[str, Any]:
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

        paper_pack = _read_object(paper_pack_path)
        papers = paper_pack.get("papers", [])
        paper_ids = sorted({
            str(item.get("paper_id", "")).strip()
            for item in papers if isinstance(item, dict) and item.get("paper_id")
        }) if isinstance(papers, list) else []
        citation_edges = sorted({
            (str(item.get("paper_id", "")).strip(), str(reference.get("paper_id", "")).strip())
            for item in papers if isinstance(item, dict)
            for reference in item.get("references", []) if isinstance(item.get("references"), list) and isinstance(reference, dict)
            if item.get("paper_id") and reference.get("paper_id")
        }) if isinstance(papers, list) else []
        domains.append(
            {
                "domain_package_id": domain_id,
                "seed_paper": seed_paper,
                "title": _required_text(summary, "domain_title", summary_path),
                "overview": str(summary.get("overview") or summary.get("brief_introduction") or ""),
                "task_focus": summary.get("task_focus", {}),
                "methodology": summary.get("methodology", []),
                "known_solved_cases": summary.get("known_solved_cases", []),
                "open_axes_for_new_work": summary.get("open_axes_for_new_work", []),
                "mathematical_opportunities": summary.get("mathematical_opportunities", {"well_defined_problems": []}),
                "summary_schema_version": str(summary.get("schema_version", "")),
                "foundation_paper_ids": sorted({seed_paper, str(foundation.get("paper_id", "")).strip()} - {""}),
                "paper_ids": paper_ids,
                "citation_edges": [list(edge) for edge in citation_edges],
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
            item["domain_package_id"],
        )
    )
    warning = ""
    try:
        pairs = _validate_grouping(grouping_result, domains)
        grouping_method = "llm_semantic_pair_classification"
    except ManifestError as exc:
        pairs = []
        grouping_method = "conservative_fallback"
        warning = f"field_grouping_degraded: {exc}; merged all domain packages into one field"
    field_groups = _build_field_groups(domains, pairs, intent=str(context.get("user_intent", "")), force_single=bool(warning))
    grouping_payload = {
        "schema_version": GROUPING_SCHEMA_VERSION,
        "grouping_method": grouping_method,
        "hard_separation_confidence": HARD_SEPARATION_CONFIDENCE,
        "pair_classifications": pairs,
        "field_groups": [
            {
                key: item[key]
                for key in ("field_id", "domain_package_ids", "confidence", "reason", "evidence")
            }
            for item in field_groups
        ],
        "warnings": [warning] if warning else [],
    }
    grouping_path = domain_dir / "field-grouping.json"
    grouping_path.write_text(json.dumps(grouping_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "user_intent": str(context.get("user_intent", "")).strip(),
        "research_scope": "single_domain" if len(field_groups) == 1 else "cross_domain",
        "requested_seed_papers": requested_strings,
        "package_count": len(domains),
        "domain_packages": domains,
        "field_count": len(field_groups),
        "field_groups": field_groups,
        "grouping_method": grouping_method,
        "grouping_artifact": _relative(project_dir, grouping_path),
        "grouping_warnings": grouping_payload["warnings"],
        "duplicates": duplicates,
    }


def write_domain_manifest(project_dir: Path, output: Path | None = None) -> Path:
    project_dir = project_dir.expanduser().resolve()
    destination = output.expanduser().resolve() if output else project_dir / "domain" / "domain-manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        preliminary = build_domain_manifest(project_dir)
        if preliminary["package_count"] == 1:
            payload = build_domain_manifest(project_dir, grouping_result={"pairs": []})
        else:
            grouping_result = _llm_grouping(preliminary["domain_packages"], preliminary["user_intent"])
            payload = build_domain_manifest(project_dir, grouping_result=grouping_result)
    except Exception:
        payload = build_domain_manifest(project_dir)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def _validate_grouping(payload: dict[str, Any] | None, packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(packages) == 1 and payload is None:
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ManifestError("semantic field grouping was unavailable")
    expected = set(itertools.combinations(sorted(item["domain_package_id"] for item in packages), 2))
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload["pairs"]:
        if not isinstance(item, dict): raise ManifestError("grouping pairs must be objects")
        pair = tuple(sorted((str(item.get("package_a", "")), str(item.get("package_b", "")))))
        label = str(item.get("classification", ""))
        try: confidence = float(item.get("confidence"))
        except (TypeError, ValueError) as exc: raise ManifestError(f"invalid confidence for pair {pair}") from exc
        if pair not in expected or pair in found or label not in {"same_field", "distinct_field", "uncertain"} or not 0 <= confidence <= 1:
            raise ManifestError(f"invalid or duplicate grouping pair {pair}")
        if not isinstance(item.get("evidence"), dict): raise ManifestError(f"pair {pair} requires evidence")
        found[pair] = {"package_a": pair[0], "package_b": pair[1], "classification": label,
                       "confidence": confidence, "reason": str(item.get("reason", "")), "evidence": item["evidence"]}
    if set(found) != expected: raise ManifestError("grouping must classify every package pair")
    ordered = [found[pair] for pair in sorted(found)]
    _validate_pair_constraints(ordered, sorted(item["domain_package_id"] for item in packages))
    return ordered


def _validate_pair_constraints(pairs: list[dict[str, Any]], package_ids: list[str]) -> None:
    """Require conservative mergeability to be an equivalence relation.

    Every pair below the hard-distinct threshold is a conservative merge edge. If
    such edges transitively connect a hard-distinct pair, any split would depend on
    iteration order rather than model-supported evidence, so reject the grouping.
    """
    parent = {item: item for item in package_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    hard: list[dict[str, Any]] = []
    for item in pairs:
        if item["classification"] == "distinct_field" and item["confidence"] >= HARD_SEPARATION_CONFIDENCE:
            hard.append(item)
        else:
            union(item["package_a"], item["package_b"])
    conflicts = [
        item for item in hard if find(item["package_a"]) == find(item["package_b"])
    ]
    if conflicts:
        formatted = ", ".join(
            f"{item['package_a']}–{item['package_b']}" for item in conflicts
        )
        raise GroupingConstraintError(
            "contradictory/non-transitive semantic grouping: hard-distinct pair(s) "
            f"{formatted} are transitively connected by conservative same-field relations"
        )


def _build_field_groups(packages: list[dict[str, Any]], pairs: list[dict[str, Any]], *, intent: str, force_single: bool) -> list[dict[str, Any]]:
    package_ids = sorted(item["domain_package_id"] for item in packages)
    parent = {item: item for item in package_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    if force_single:
        for package_id in package_ids[1:]:
            union(package_ids[0], package_id)
    else:
        for item in pairs:
            hard = item["classification"] == "distinct_field" and item["confidence"] >= HARD_SEPARATION_CONFIDENCE
            if not hard:
                union(item["package_a"], item["package_b"])
    by_root: dict[str, list[str]] = {}
    for package_id in package_ids:
        by_root.setdefault(find(package_id), []).append(package_id)
    bins = sorted((sorted(items) for items in by_root.values()), key=lambda items: tuple(items))
    by_id = {item["domain_package_id"]: item for item in packages}
    intent_hash = hashlib.sha256(intent.encode()).hexdigest()
    result = []
    for ids in bins:
        members = [by_id[item] for item in ids]
        digest = hashlib.sha256(("\n".join(ids) + "\n" + intent_hash).encode()).hexdigest()[:16]
        relevant_pairs = [item for item in pairs if item["package_a"] in ids or item["package_b"] in ids]
        internal_pairs = [item for item in relevant_pairs if item["package_a"] in ids and item["package_b"] in ids]
        confidence_values = [float(item["confidence"]) for item in internal_pairs]
        if not confidence_values:
            confidence_values = [
                float(item["confidence"])
                for item in relevant_pairs
                if item["classification"] == "distinct_field"
                and item["confidence"] >= HARD_SEPARATION_CONFIDENCE
            ]
        confidence = min(confidence_values) if confidence_values else (0.0 if force_single else 1.0)
        reasons = [str(item["reason"]).strip() for item in relevant_pairs if str(item["reason"]).strip()]
        result.append({
            "field_id": f"field-{digest}",
            "domain_package_ids": ids,
            "confidence": confidence,
            "reason": (
                "Conservative fallback merged all packages because semantic grouping was unavailable."
                if force_single else "; ".join(reasons) or "Single package field; no pairwise merge evidence required."
            ),
            "evidence": [
                {
                    "package_a": item["package_a"],
                    "package_b": item["package_b"],
                    "classification": item["classification"],
                    "confidence": item["confidence"],
                    "evidence": item["evidence"],
                }
                for item in relevant_pairs
            ],
            "field_card": {
            "seed_papers": [item["seed_paper"] for item in members],
            "titles": [item["title"] for item in members],
            "overviews": [item["overview"] for item in members if item["overview"]],
            "task_focus": [item["task_focus"] for item in members if item["task_focus"]],
            "methodology": [method for item in members if isinstance(item["methodology"], list) for method in item["methodology"]],
            "known_solved_cases": [case for item in members if isinstance(item["known_solved_cases"], list) for case in item["known_solved_cases"]],
            "open_axes_for_new_work": [axis for item in members if isinstance(item["open_axes_for_new_work"], list) for axis in item["open_axes_for_new_work"]],
            "mathematical_opportunities": {"well_defined_problems": [problem for item in members
                for problem in item["mathematical_opportunities"].get("well_defined_problems", [])
                if isinstance(item["mathematical_opportunities"], dict)]},
            "summary_schema_versions": [item["summary_schema_version"] for item in members],
            "summary_json_paths": [item["summary_json_path"] for item in members],
            "summary_markdown_paths": [item["summary_markdown_path"] for item in members],
            "paper_json_pack_paths": [item["paper_json_pack_path"] for item in members],
            "paper_ids": sorted({paper for item in members for paper in item["paper_ids"]}),
            "citation_edges": sorted({tuple(edge) for item in members for edge in item["citation_edges"]}),
        }})
    return result


def _llm_grouping(packages: list[dict[str, Any]], intent: str) -> dict[str, Any]:
    from arc_llm import run_json
    schema = {"type": "object", "additionalProperties": False, "required": ["pairs"], "properties": {"pairs": {
        "type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["package_a", "package_b", "classification", "confidence", "reason", "evidence"], "properties": {
            "package_a": {"type": "string"}, "package_b": {"type": "string"},
            "classification": {"type": "string", "enum": ["same_field", "distinct_field", "uncertain"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "reason": {"type": "string"},
            "evidence": {"type": "object", "additionalProperties": False, "required": ["semantic", "paper_overlap", "citation_overlap"],
                "properties": {"semantic": {"type": "string"}, "paper_overlap": {"type": "string"}, "citation_overlap": {"type": "string"}}}
        }}}}}
    compact = [{key: item[key] for key in ("domain_package_id", "seed_paper", "foundation_paper_ids", "title", "overview", "task_focus", "methodology", "paper_ids", "citation_edges")} for item in packages]
    prompt = f"Classify every unordered package pair as same_field, distinct_field, or uncertain. Exact intent: {intent}\nPackages: {json.dumps(compact, ensure_ascii=False)}"
    return run_json(prompt, schema=schema, provider="auto", model_tier="medium")


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
            "package_count": payload["package_count"],
            "field_count": payload["field_count"],
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
