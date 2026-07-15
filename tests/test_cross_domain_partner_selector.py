from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "plugins/arc/skills/arc/workflows/scripts/select-cross-domain-partner.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("select_cross_domain_partner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        sys.path.remove(str(SCRIPT.parent))
    return module


def test_partner_selection_is_open_world_verified_and_ranked(tmp_path: Path) -> None:
    module = _load_module()
    anchor = tmp_path / "fresh" / "anchor_domain_summary.json"
    anchor.parent.mkdir()
    anchor.write_text(json.dumps({"domain_id": "anchor", "domain_title": "Anchor"}), encoding="utf-8")
    calls: list[str] = []

    def fake_json(prompt: str, **_: Any) -> dict[str, Any]:
        calls.append(prompt)
        if len(calls) == 1:
            return {
                "candidates": [
                    {
                        "domain_label": f"Domain {index}",
                        "representative_seed": _seed(index),
                        "role_for_anchor": "either",
                        "transferred_ingredient": f"method {index}",
                        "target_capability_gap": "gap",
                        "translation_map": "map",
                        "first_calculation": "bounded calculation",
                        "compatibility_risks": ["risk"],
                        "semantic_distance_diagnostic": "distinct",
                    }
                    for index in range(6)
                ]
            }
        return {
            "ranked_candidates": [
                {
                    "representative_seed": _seed(index),
                    "bridge_physical_feasibility": 30 - index,
                    "transferred_ingredient_specificity": 20,
                    "substantive_target_opportunity": 20,
                    "semantic_distinctness": 10,
                    "hard_gate_passed": index != 1,
                    "hard_gate_failures": [] if index != 1 else ["same subfield"],
                    "reasoning": "audited",
                }
                for index in range(6)
            ]
        }

    result = module.select_partner(
        anchor,
        user_intent="find the best partner",
        json_runner=fake_json,
        metadata_fetcher=lambda seed: {"paper_id": seed, "title": f"Paper {seed}", "abstract": "A"},
    )

    assert len(result["verified_candidates"]) == 6
    assert result["selected_candidate"]["representative_seed"] == _seed(0)
    assert [item["representative_seed"] for item in result["fallback_candidates"]] == [_seed(2), _seed(3)]
    assert result["selection_policy"]["history_blind"] is True
    assert result["selection_policy"]["cache_blind"] is True
    assert "do not rank" in calls[0].lower()
    assert "exact identifier that arc paper tools can resolve" in calls[0].lower()
    assert "do not return a paper title" in calls[0].lower()
    assert "independently audit" in calls[1].lower()


def test_selector_schema_requires_exact_arc_resolvable_identifier_forms() -> None:
    schema_path = ROOT / "plugins/arc/skills/arc/workflows/json/cross-domain-partner-selector.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    seed_schema = schema["properties"]["candidates"]["items"]["properties"]["representative_seed"]
    pattern = re.compile(seed_schema["pattern"])

    for seed in (
        "arXiv:2401.12345",
        "arXiv:hep-th/0601001",
        "arXiv:alg-geom/9501001",
        "arXiv:atom-ph/9601001",
        "arXiv:chem-ph/9401001",
        "arXiv:solv-int/9701001",
        "doi:10.1103/PhysRevD.99.123456",
        "inspire:1234567",
    ):
        assert pattern.fullmatch(seed), seed
    for invalid in (
        "2401.12345",
        "https://arxiv.org/abs/2401.12345",
        "A famous representative paper",
        "seed:quantum-gravity",
        "arXiv:hep_th/0601001",
        "arXiv:/0601001",
    ):
        assert pattern.fullmatch(invalid) is None, invalid


def test_metadata_verification_failure_lists_every_rejected_seed_and_reason(tmp_path: Path) -> None:
    module = _load_module()
    anchor = tmp_path / "fresh" / "anchor_domain_summary.json"
    anchor.parent.mkdir()
    anchor.write_text(json.dumps({"domain_id": "anchor", "domain_title": "Anchor"}), encoding="utf-8")
    seeds = [_seed(index) for index in range(6)]

    def fake_json(_prompt: str, **_: Any) -> dict[str, Any]:
        return {
            "candidates": [
                {
                    "domain_label": f"Domain {index}",
                    "representative_seed": seed,
                    "role_for_anchor": "either",
                    "transferred_ingredient": f"method {index}",
                    "target_capability_gap": "gap",
                    "translation_map": "map",
                    "first_calculation": "bounded calculation",
                    "compatibility_risks": ["risk"],
                    "semantic_distance_diagnostic": "distinct",
                }
                for index, seed in enumerate(seeds)
            ]
        }

    def fake_metadata(seed: str) -> dict[str, str]:
        if seed in seeds[:2]:
            return {"paper_id": seed, "title": f"Paper {seed}"}
        if seed == seeds[3]:
            return {"paper_id": seed, "title": ""}
        raise RuntimeError(f"not found: {seed}")

    with pytest.raises(module.PartnerSelectionError) as exc_info:
        module.select_partner(
            anchor,
            user_intent="find the best partner",
            json_runner=fake_json,
            metadata_fetcher=fake_metadata,
        )

    message = str(exc_info.value)
    assert "(2/6 verified)" in message
    for seed in seeds[2:]:
        assert seed in message
    assert f"{seeds[2]!r}: metadata_error: not found: {seeds[2]}" in message
    assert f"{seeds[3]!r}: metadata_has_no_title" in message
    assert f"{seeds[4]!r}: metadata_error: not found: {seeds[4]}" in message
    assert f"{seeds[5]!r}: metadata_error: not found: {seeds[5]}" in message


def test_partner_selector_rejects_historical_anchor(tmp_path: Path) -> None:
    module = _load_module()
    anchor = tmp_path / "arc-tests" / "prev" / "anchor.json"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("{}\n", encoding="utf-8")

    with pytest.raises(module.PartnerSelectionError, match="historical ARC artifact"):
        module.select_partner(anchor, user_intent="intent")


def test_live_partner_selector_requires_strict_source_mode(tmp_path: Path, monkeypatch: Any) -> None:
    module = _load_module()
    monkeypatch.delenv("ARC_REQUIRE_REPO_ROOT", raising=False)
    anchor = tmp_path / "fresh" / "anchor.json"
    anchor.parent.mkdir()
    anchor.write_text(json.dumps({"domain_id": "anchor"}), encoding="utf-8")

    with pytest.raises(module.PartnerSelectionError, match="ARC_REQUIRE_REPO_ROOT is required"):
        module.select_partner(anchor, user_intent="intent")


def _seed(index: int) -> str:
    return f"arXiv:2401.{index + 1:05d}"
