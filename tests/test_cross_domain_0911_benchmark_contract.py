from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "tests/fixtures/cross_domain_0911_benchmark.json"
CROSS_DOMAIN_LOOP_TEMPLATE = (
    ROOT / "plugins/arc/skills/arc/workflows/json/ideas-cross-domain-loop.template.json"
)


def test_0911_benchmark_uses_current_checkout_and_ai_selected_partner() -> None:
    payload = json.loads(SPEC.read_text(encoding="utf-8"))
    selection = payload["partner_selection"]
    artifact_root = ROOT / payload["artifact_root"]

    assert payload["source_mode"] == "required_repo_root"
    assert (ROOT / payload["repo_root"]).resolve() == ROOT
    assert artifact_root.is_relative_to(ROOT / "arc-tests")
    assert payload["anchor_seed"] == "arXiv:0911.3380"
    assert selection["mode"] == "ai_open_world_history_blind"
    assert selection["partner_seed"] is None
    assert selection["candidate_count"] == 6
    assert selection["select_once_before_ideas"] is True
    assert selection["freeze_pair_for_both_arms"] is True
    assert selection["forbidden_context_paths"] == ["0_ref", "arc-tests/prev"]
    assert selection["forbid_cache_hit_signals"] is True


def test_0911_benchmark_budget_and_acceptance_are_frozen() -> None:
    payload = json.loads(SPEC.read_text(encoding="utf-8"))

    assert payload["ideas_run"] == {
        "loops_per_variant": 5,
        "rounds_per_loop": 3,
        "matched_replicates_per_arm": 2,
        "save_prompts": True,
    }
    assert payload["acceptance"] == {
        "minimum_qualified_loops": 3,
        "top_three_all_qualified": True,
        "minimum_transfer_signature_clusters": 3,
        "maximum_same_central_mechanism": 2,
        "minimum_blind_preference_rate": 0.6,
        "maximum_feasibility_drop": 0.25,
        "require_single_domain_prompt_regression_tests": True,
    }


def test_0911_benchmark_round_budget_matches_current_arc_template() -> None:
    payload = json.loads(SPEC.read_text(encoding="utf-8"))
    loop_template = json.loads(CROSS_DOMAIN_LOOP_TEMPLATE.read_text(encoding="utf-8"))

    assert payload["ideas_run"]["rounds_per_loop"] == loop_template["max_rounds"] == 3
    assert payload["ideas_run"]["loops_per_variant"] * payload["ideas_run"]["rounds_per_loop"] == 15
