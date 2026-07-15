from __future__ import annotations

import importlib.util
import json
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
                        "representative_seed": f"seed:{index}",
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
                    "representative_seed": f"seed:{index}",
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
    assert result["selected_candidate"]["representative_seed"] == "seed:0"
    assert [item["representative_seed"] for item in result["fallback_candidates"]] == ["seed:2", "seed:3"]
    assert result["selection_policy"]["history_blind"] is True
    assert result["selection_policy"]["cache_blind"] is True
    assert "do not rank" in calls[0].lower()
    assert "independently audit" in calls[1].lower()


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
