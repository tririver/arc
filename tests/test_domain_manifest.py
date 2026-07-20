from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "plugins/arc/skills/arc/workflows/scripts/write-domain-manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("write_domain_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def _write_domain(
    project: Path,
    prefix: str,
    domain_id: str,
    seed: str,
    *,
    schema_version: str = "arc.domain_summary.v4",
) -> None:
    domain = project / "domain"
    domain.mkdir(parents=True, exist_ok=True)
    (domain / f"{prefix}_domain_summary.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "domain_id": domain_id,
                "domain_title": f"Domain {domain_id}",
                "foundation_paper": {"paper_id": seed},
            }
        ),
        encoding="utf-8",
    )
    (domain / f"{prefix}_domain_summary.md").write_text("# Domain\n", encoding="utf-8")
    (domain / f"{prefix}_paper_json_pack.json").write_text("{}\n", encoding="utf-8")


def test_manifest_uses_distinct_domain_ids_and_relative_paths(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "context.json").write_text(
        json.dumps({"user_intent": "cross fields", "seed_paper_list": ["seed:a", "seed:b"]}),
        encoding="utf-8",
    )
    _write_domain(project, "a", "domain-a", "seed:a")
    _write_domain(project, "b", "domain-b", "seed:b")
    _write_domain(project, "duplicate", "domain-a", "seed:a2")

    payload = module.build_domain_manifest(project)

    assert payload["schema_version"] == "arc.workflow.domain_manifest.v2"
    assert payload["package_count"] == 2
    assert payload["field_count"] == 1
    assert payload["research_scope"] == "single_domain"
    assert [item["domain_package_id"] for item in payload["domain_packages"]] == ["domain-a", "domain-b"]
    assert payload["domain_packages"][0]["summary_json_path"] == "domain/a_domain_summary.json"
    assert payload["domain_packages"][0]["seed_paper"] == "seed:a"
    assert payload["duplicates"] == [
        {
            "domain_id": "domain-a",
            "kept_summary_json_path": "domain/a_domain_summary.json",
            "duplicate_summary_json_path": "domain/duplicate_domain_summary.json",
        }
    ]


def test_manifest_preserves_requested_seed_order(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "context.json").write_text(
        json.dumps({"user_intent": "cross fields", "seed_paper_list": ["seed:z", "seed:a"]}),
        encoding="utf-8",
    )
    _write_domain(project, "a", "domain-a", "seed:a")
    _write_domain(project, "z", "domain-z", "seed:z")

    payload = module.build_domain_manifest(project)

    assert [item["seed_paper"] for item in payload["domain_packages"]] == ["seed:z", "seed:a"]


def test_manifest_indexes_mixed_v4_v5_summaries_without_rewriting_them(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "context.json").write_text(
        json.dumps({"seed_paper_list": ["seed:a", "seed:b"]}),
        encoding="utf-8",
    )
    _write_domain(project, "a", "domain-a", "seed:a", schema_version="arc.domain_summary.v4")
    _write_domain(project, "b", "domain-b", "seed:b", schema_version="arc.domain_summary.v5")

    payload = module.build_domain_manifest(project)

    assert [item["domain_package_id"] for item in payload["domain_packages"]] == ["domain-a", "domain-b"]
    assert json.loads((project / "domain/a_domain_summary.json").read_text())["schema_version"] == (
        "arc.domain_summary.v4"
    )
    assert json.loads((project / "domain/b_domain_summary.json").read_text())["schema_version"] == (
        "arc.domain_summary.v5"
    )


def test_manifest_prefers_requested_seed_domain_records_over_foundation(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "context.json").write_text(
        json.dumps(
            {
                "seed_paper_list": ["arXiv:1234.5678"],
                "domain_records": [
                    {"domain_id": "domain-a", "seed_paper": "arXiv:1234.5678"}
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_domain(project, "a", "domain-a", "arXiv:9999.0001")

    payload = module.build_domain_manifest(project)

    assert payload["domain_packages"][0]["seed_paper"] == "arXiv:1234.5678"


def test_manifest_hard_separates_only_high_confidence_distinct_fields(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "context.json").write_text(json.dumps({"user_intent": "bridge", "seed_paper_list": ["seed:a", "seed:b"]}))
    _write_domain(project, "a", "domain-a", "seed:a")
    _write_domain(project, "b", "domain-b", "seed:b")
    pair = {"package_a": "domain-a", "package_b": "domain-b", "classification": "distinct_field",
            "confidence": 0.8, "reason": "different methods", "evidence": {"semantic": "x"}}

    payload = module.build_domain_manifest(project, grouping_result={"pairs": [pair]})

    assert payload["field_count"] == 2
    assert payload["research_scope"] == "cross_domain"
    assert all(item["field_id"].startswith("field-") for item in payload["field_groups"])


def test_manifest_low_confidence_or_failed_grouping_merges_conservatively(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "context.json").write_text(json.dumps({"user_intent": "same area"}))
    _write_domain(project, "a", "domain-a", "seed:a")
    _write_domain(project, "b", "domain-b", "seed:b")
    pair = {"package_a": "domain-a", "package_b": "domain-b", "classification": "distinct_field",
            "confidence": 0.79, "reason": "weak", "evidence": {}}

    low = module.build_domain_manifest(project, grouping_result={"pairs": [pair]})
    failed = module.build_domain_manifest(project, grouping_result={"pairs": []})

    assert low["field_count"] == 1
    assert failed["field_count"] == 1
    assert failed["grouping_method"] == "conservative_fallback"
    assert failed["grouping_warnings"]


def test_manifest_three_package_grouping_is_deterministic_and_evidence_backed(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "context.json").write_text(json.dumps({"user_intent": "bridge"}), encoding="utf-8")
    for suffix in ("a", "b", "c"):
        _write_domain(project, suffix, f"domain-{suffix}", f"seed:{suffix}")
    pairs = [
        {"package_a": "domain-a", "package_b": "domain-b", "classification": "same_field", "confidence": 0.91, "reason": "same methods", "evidence": {"semantic": "shared"}},
        {"package_a": "domain-a", "package_b": "domain-c", "classification": "distinct_field", "confidence": 0.94, "reason": "different objects", "evidence": {"semantic": "distinct"}},
        {"package_a": "domain-b", "package_b": "domain-c", "classification": "distinct_field", "confidence": 0.88, "reason": "different objects", "evidence": {"semantic": "distinct"}},
    ]

    first = module.build_domain_manifest(project, grouping_result={"pairs": pairs})
    second = module.build_domain_manifest(project, grouping_result={"pairs": list(reversed(pairs))})

    assert first["field_count"] == 2
    assert [item["field_id"] for item in first["field_groups"]] == [item["field_id"] for item in second["field_groups"]]
    merged = next(item for item in first["field_groups"] if len(item["domain_package_ids"]) == 2)
    assert merged["confidence"] == 0.91
    assert merged["reason"]
    assert merged["evidence"]


def test_manifest_falls_back_on_nontransitive_grouping_across_hard_distinct_pair(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "context.json").write_text(json.dumps({"user_intent": "bridge"}), encoding="utf-8")
    for suffix in ("a", "b", "c"):
        _write_domain(project, suffix, f"domain-{suffix}", f"seed:{suffix}")
    pairs = [
        {"package_a": "domain-a", "package_b": "domain-b", "classification": "same_field", "confidence": 0.9, "reason": "same", "evidence": {}},
        {"package_a": "domain-b", "package_b": "domain-c", "classification": "uncertain", "confidence": 0.7, "reason": "uncertain", "evidence": {}},
        {"package_a": "domain-a", "package_b": "domain-c", "classification": "distinct_field", "confidence": 0.95, "reason": "hard split", "evidence": {}},
    ]

    payload = module.build_domain_manifest(project, grouping_result={"pairs": pairs})

    assert payload["field_count"] == 1
    assert payload["research_scope"] == "single_domain"
    assert payload["grouping_method"] == "conservative_fallback"
    assert "non-transitive" in payload["grouping_warnings"][0]
    grouping = json.loads((project / payload["grouping_artifact"]).read_text(encoding="utf-8"))
    assert grouping["warnings"] == payload["grouping_warnings"]


def test_manifest_requires_companion_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    project = tmp_path / "project"
    (project / "domain").mkdir(parents=True)
    (project / "context.json").write_text("{}\n", encoding="utf-8")
    (project / "domain/x_domain_summary.json").write_text(
        json.dumps({"domain_id": "x", "domain_title": "X"}),
        encoding="utf-8",
    )

    with pytest.raises(module.ManifestError, match="required domain artifact"):
        module.build_domain_manifest(project)
