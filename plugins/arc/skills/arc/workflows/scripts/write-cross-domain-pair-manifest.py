#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "arc.workflow.cross_domain_pair_manifest.v1"


class PairManifestError(ValueError):
    pass


def build_pair_manifest(
    domain_manifest_path: Path,
    partner_selection_path: Path,
    *,
    source_provenance_path: Path,
) -> dict[str, Any]:
    domain_manifest_path = domain_manifest_path.expanduser().resolve()
    partner_selection_path = partner_selection_path.expanduser().resolve()
    source_provenance_path = source_provenance_path.expanduser().resolve()
    provenance_record = _read_object(source_provenance_path)
    repo_root = _validated_provenance_root(provenance_record)
    for path in (domain_manifest_path, partner_selection_path, source_provenance_path):
        _require_fresh_repo_path(path, repo_root=repo_root)
    domain_manifest = _read_object(domain_manifest_path)
    selection = _read_object(partner_selection_path)
    if domain_manifest.get("schema_version") != "arc.workflow.domain_manifest.v1":
        raise PairManifestError("domain manifest has the wrong schema_version")
    if selection.get("schema_version") != "arc.workflow.cross_domain_partner_selection.v1":
        raise PairManifestError("partner selection has the wrong schema_version")
    domains = domain_manifest.get("domains")
    if not isinstance(domains, list):
        raise PairManifestError("domain manifest domains must be an array")
    distinct = {str(item.get("domain_id", "")).strip() for item in domains if isinstance(item, dict)}
    distinct.discard("")
    if len(distinct) != 2:
        raise PairManifestError("a frozen cross-domain benchmark pair requires exactly two distinct domains")

    anchor_data = selection.get("anchor")
    selected = selection.get("selected_candidate")
    if not isinstance(anchor_data, dict) or not isinstance(selected, dict):
        raise PairManifestError("partner selection is missing anchor or selected_candidate")
    anchor_id = str(anchor_data.get("domain_id", "")).strip()
    partner_seed = str(selected.get("representative_seed", "")).strip()
    if not anchor_id or not partner_seed:
        raise PairManifestError("partner selection does not identify the anchor domain and selected seed")

    project_dir = domain_manifest_path.parent.parent
    anchor_entry = next(
        (item for item in domains if isinstance(item, dict) and item.get("domain_id") == anchor_id),
        None,
    )
    partner_entry = next(
        (
            item
            for item in domains
            if isinstance(item, dict)
            and _paper_id_key(str(item.get("seed_paper", ""))) == _paper_id_key(partner_seed)
        ),
        None,
    )
    if anchor_entry is None:
        raise PairManifestError(f"anchor domain {anchor_id!r} is absent from domain manifest")
    if partner_entry is None:
        raise PairManifestError(f"selected partner seed {partner_seed!r} is absent from domain manifest")
    if anchor_entry.get("domain_id") == partner_entry.get("domain_id"):
        raise PairManifestError("anchor and partner resolve to the same domain")

    provenance = {
        "path": str(source_provenance_path),
        "sha256": _sha256(source_provenance_path),
        "repo_root": str(repo_root),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        "user_intent": str(domain_manifest.get("user_intent", "")),
        "domain_manifest": {"path": str(domain_manifest_path), "sha256": _sha256(domain_manifest_path)},
        "partner_selection": {"path": str(partner_selection_path), "sha256": _sha256(partner_selection_path)},
        "source_provenance": provenance,
        "anchor": _frozen_domain(project_dir, anchor_entry, repo_root=repo_root),
        "partner": _frozen_domain(project_dir, partner_entry, repo_root=repo_root),
    }


def _frozen_domain(project_dir: Path, entry: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    artifacts = {}
    for field in ("summary_json_path", "summary_markdown_path", "paper_json_pack_path"):
        raw = str(entry.get(field, "")).strip()
        if not raw:
            raise PairManifestError(f"domain {entry.get('domain_id')!r} is missing {field}")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = project_dir / path
        path = path.resolve()
        if not path.is_file():
            raise PairManifestError(f"frozen domain artifact does not exist: {path}")
        _require_fresh_repo_path(path, repo_root=repo_root)
        artifacts[field] = {"path": str(path), "sha256": _sha256(path)}
    return {
        "domain_id": str(entry.get("domain_id", "")),
        "seed_paper": str(entry.get("seed_paper", "")),
        "title": str(entry.get("title", "")),
        "artifacts": artifacts,
    }


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairManifestError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PairManifestError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paper_id_key(value: str) -> str:
    text = value.strip().lower()
    if text.startswith("arxiv:"):
        text = text[6:]
    if "v" in text:
        stem, version = text.rsplit("v", 1)
        if version.isdigit():
            text = stem
    return text


def _validated_provenance_root(record: dict[str, Any]) -> Path:
    if record.get("schema_version") != "arc.workflow.source_provenance.v1":
        raise PairManifestError("source provenance has the wrong schema_version")
    raw_root = str(record.get("repo_root", "")).strip()
    if not raw_root:
        raise PairManifestError("source provenance is missing repo_root")
    root = Path(raw_root).expanduser().resolve()
    modules = record.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise PairManifestError("source provenance is missing verified ARC modules")
    for module_name, item in modules.items():
        if not isinstance(item, dict):
            raise PairManifestError(f"invalid source provenance for module {module_name}")
        module_file = Path(str(item.get("file", ""))).expanduser().resolve()
        if not module_file.is_relative_to(root / "packages"):
            raise PairManifestError(f"module {module_name} is outside provenance repo_root")
    return root


def _require_fresh_repo_path(path: Path, *, repo_root: Path) -> None:
    if not path.is_relative_to(repo_root):
        raise PairManifestError(f"benchmark input is outside provenance repo_root: {path}")
    relative_parts = path.relative_to(repo_root).parts
    if "0_ref" in relative_parts or relative_parts[:2] == ("arc-tests", "prev"):
        raise PairManifestError(f"historical ARC artifact is forbidden in a frozen pair: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze one AI-selected cross-domain benchmark pair.")
    parser.add_argument("--domain-manifest", required=True)
    parser.add_argument("--partner-selection", required=True)
    parser.add_argument("--source-provenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = build_pair_manifest(
            Path(args.domain_manifest),
            Path(args.partner_selection),
            source_provenance_path=Path(args.source_provenance),
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result = {"status": "completed", "output": str(output)}
        print(json.dumps(result) if args.json else str(output))
        return 0
    except PairManifestError as exc:
        result = {"status": "failed", "error": str(exc)}
        print(json.dumps(result) if args.json else f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
