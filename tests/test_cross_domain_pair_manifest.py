from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "plugins/arc/skills/arc/workflows/scripts/write-cross-domain-pair-manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("write_cross_domain_pair_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def test_pair_manifest_freezes_selected_domains_and_provenance(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    domain_dir = project / "domain"
    domain_dir.mkdir(parents=True)
    domains = []
    for prefix, domain_id, seed in (("anchor", "anchor-id", "seed:anchor"), ("partner", "partner-id", "seed:partner")):
        for suffix in ("domain_summary.json", "domain_summary.md", "paper_json_pack.json"):
            (domain_dir / f"{prefix}_{suffix}").write_text(f"{prefix} {suffix}\n", encoding="utf-8")
        domains.append(
            {
                "domain_id": domain_id,
                "seed_paper": seed,
                "title": prefix,
                "summary_json_path": f"domain/{prefix}_domain_summary.json",
                "summary_markdown_path": f"domain/{prefix}_domain_summary.md",
                "paper_json_pack_path": f"domain/{prefix}_paper_json_pack.json",
            }
        )
    domain_manifest = domain_dir / "domain-manifest.json"
    domain_manifest.write_text(
        json.dumps(
            {
                "schema_version": "arc.workflow.domain_manifest.v1",
                "user_intent": "intent",
                "domains": domains,
            }
        ),
        encoding="utf-8",
    )
    selection = project / "partner-selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": "arc.workflow.cross_domain_partner_selection.v1",
                "anchor": {"domain_id": "anchor-id"},
                "selected_candidate": {"representative_seed": "seed:partner"},
            }
        ),
        encoding="utf-8",
    )
    provenance = project / "source-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "arc.workflow.source_provenance.v1",
                "repo_root": str(tmp_path),
                "modules": {
                    "arc_llm": {"file": str(tmp_path / "packages/arc-llm/src/arc_llm/__init__.py")},
                    "arc_paper": {"file": str(tmp_path / "packages/arc-paper/src/arc_paper/__init__.py")},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = module.build_pair_manifest(
        domain_manifest,
        selection,
        source_provenance_path=provenance,
    )

    assert payload["frozen"] is True
    assert payload["anchor"]["domain_id"] == "anchor-id"
    assert payload["partner"]["seed_paper"] == "seed:partner"
    assert len(payload["anchor"]["artifacts"]["summary_json_path"]["sha256"]) == 64
    assert payload["source_provenance"]["repo_root"] == str(tmp_path)


def test_pair_manifest_rejects_historical_inputs(tmp_path: Path) -> None:
    module = _load_module()
    provenance = tmp_path / "source-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "arc.workflow.source_provenance.v1",
                "repo_root": str(tmp_path),
                "modules": {
                    "arc_llm": {"file": str(tmp_path / "packages/arc-llm/src/arc_llm/__init__.py")}
                },
            }
        ),
        encoding="utf-8",
    )
    historical = tmp_path / "arc-tests" / "prev" / "domain-manifest.json"
    historical.parent.mkdir(parents=True)
    historical.write_text("{}\n", encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")

    with pytest.raises(module.PairManifestError, match="historical ARC artifact"):
        module.build_pair_manifest(historical, selection, source_provenance_path=provenance)
