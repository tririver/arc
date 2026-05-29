from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_llm.proposers_reviewer.config import ConfigError
from arc_llm.proposers_reviewer.consensus import (
    load_consensus_config,
    run_proposers_reviewer_consensus,
)


def minimal_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": "arc.llm.proposers_reviewer_consensus.config.v1",
        "run_id": "calc_001",
        "run_dir": str(tmp_path),
        "steps": [{"step_id": "step_001", "prompt": "derive x"}],
    }
    config.update(overrides)
    return config


def test_consensus_config_defaults_to_two_proposers_and_two_recalculations(tmp_path):
    config = load_consensus_config(minimal_config(tmp_path))

    assert config.proposer_count == 2
    assert config.max_recalculations == 2
    assert config.human_gate["enabled"] is False


def test_consensus_allows_zero_recalculations(tmp_path):
    config = load_consensus_config(minimal_config(tmp_path, max_recalculations=0))

    assert config.max_recalculations == 0


def test_consensus_exact_model_requires_explicit_provider(tmp_path):
    with pytest.raises(ConfigError, match="defaults.model requires explicit provider"):
        load_consensus_config(minimal_config(tmp_path, defaults={"provider": "auto", "model": "gpt-5.5"}))


def test_consensus_rejects_foundation_check_kind(tmp_path):
    with pytest.raises(ConfigError, match="step.kind must be new_calculation"):
        load_consensus_config(
            minimal_config(
                tmp_path,
                steps=[
                    {
                        "step_id": "check_eq_target",
                        "kind": "foundation_check",
                        "prompt": "check target",
                    }
                ],
            )
        )


@pytest.mark.parametrize("legacy_key", ["foundation_file", "allowed_foundation", "target_equation_id"])
def test_consensus_rejects_legacy_allowed_context_keys(tmp_path, legacy_key):
    with pytest.raises(ConfigError, match=f"allowed_context.{legacy_key} is no longer supported"):
        load_consensus_config(
            minimal_config(
                tmp_path,
                steps=[
                    {
                        "step_id": "derive_target",
                        "prompt": "derive target",
                        "allowed_context": {legacy_key: "legacy"},
                    }
                ],
            )
        )


def test_consensus_accepts_all_agree_on_first_attempt(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"
    assert result["steps"][0]["accepted_output"] == {"result": "x"}
    assert fake.active_proposers_by_call == [["proposer_001", "proposer_002"]]
    reviewer_schema = fake.calls[0]["loops"][0]["reviewers"][0]["output_schema"]
    assert "schema_version" in reviewer_schema["required"]
    assert "proposer_messages" in reviewer_schema["required"]
    assert "review_payload" in reviewer_schema["required"]
    assert reviewer_schema["properties"]["schema_version"]["const"] == "arc.llm.review_envelope.v1"
    assert reviewer_schema["properties"]["proposer_messages"]["required"] == [
        "proposer_001",
        "proposer_002",
    ]
    reviewer = fake.calls[0]["loops"][0]["reviewers"][0]
    reviewer_template = reviewer["prompt"]["template"]
    proposer = fake.calls[0]["loops"][0]["proposers"][0]
    proposer_template = proposer["prompt"]["template"]
    caller_context = fake.calls[0]["loops"][0]["caller_context"]
    assert "Scientific Integrity Notice" in caller_context["integrity_reference"]["content"]
    assert "integrity_reference" in proposer_template
    assert "integrity_reference" in reviewer_template
    assert "very clearly step by step" in proposer_template
    assert "never skip a step" in proposer_template
    assert "LaTeX" in proposer_template
    assert "validity_scope" in proposer_template
    assert "work_note_assessment" in proposer_template
    assert "plan_foundation_assessment" not in proposer_template
    assert "reliable_until" not in proposer_template
    assert "work_note_assessment" in proposer["output_schema"]["required"]
    assert "plan_foundation_assessment" not in proposer["output_schema"]["required"]
    issue_types = proposer["output_schema"]["properties"]["work_note_assessment"]["properties"]["issue_type"]["enum"]
    assert issue_types == [
        "none",
        "work_note_inadequate",
        "work_note_conflict",
        "plan_wrong",
        "step_too_coarse",
        "target_ambiguous",
        "source_mapping_error",
        "human_needed",
        "other",
    ]
    assert proposer["runtime"]["allow_internet"] is True
    assert proposer["runtime"]["allow_mcp"] is True
    assert proposer["runtime"]["mcp_mode"] == "arc-only"
    assert "You may use ARC paper MCP tools" in proposer_template
    assert "Internet search is allowed" in proposer_template
    assert "validation-only final formulas" in proposer_template
    lower_proposer_template = proposer_template.lower()
    assert "strictly derive from the supplied work note" in lower_proposer_template
    assert "external sources may inspire methods" in lower_proposer_template
    assert "do not directly use any result" in lower_proposer_template
    assert "different conventions" in lower_proposer_template
    assert "old coordinates" in lower_proposer_template
    assert "newly introduced symbols" in lower_proposer_template
    assert reviewer["runtime"]["allow_mcp"] is False
    assert "physics and mathematics judgment" in reviewer_template
    assert "optional tools when useful" in reviewer_template
    assert "special limits are sanity checks" in reviewer_template.lower()
    assert "pairwise_symbolic_checks" not in reviewer_template
    assert "A-B" not in reviewer_template
    assert "B-C" not in reviewer_template
    assert "A-C" not in reviewer_template
    assert "source-declared old/new variable definitions" in reviewer_template
    assert "raw variable-name differences" in reviewer_template
    assert "proportionalities" in reviewer_template
    assert "constant quotient" in reviewer_template
    consensus_properties = reviewer_schema["properties"]["review_payload"]["properties"]["consensus"]["properties"]
    assert "agreement_assessment" in consensus_properties
    assert "pairwise_symbolic_checks" not in consensus_properties
    assert "best_written_proposer_id" in consensus_properties
    assert "best_written_selection_reason" in consensus_properties
    assert "validity_scope" in consensus_properties
    assert "workflow_action" in consensus_properties
    assert "workflow_action" in reviewer_schema["properties"]["review_payload"]["properties"]["consensus"]["required"]
    workflow_schema = consensus_properties["workflow_action"]
    assert workflow_schema["properties"]["action"]["enum"] == [
        "continue",
        "pause_for_human",
        "revise_plan",
        "split_step",
        "retry",
    ]
    assert workflow_schema["properties"]["issue_type"]["enum"] == [
        "none",
        "work_note_inadequate",
        "work_note_conflict",
        "plan_wrong",
        "step_too_coarse",
        "target_ambiguous",
        "source_mapping_error",
        "calculation_disagreement",
        "reference_disagreement",
        "worker_failure",
        "other",
    ]
    assert "reliable_until" not in consensus_properties
    agreement_schema = consensus_properties["agreement_assessment"]
    assert agreement_schema["required"] == [
        "target_quantity_match",
        "convention_match",
        "declared_scope_match",
        "agreement_covers_full_target",
        "comparison_summary",
        "accepted_by_reviewer_judgment",
    ]
    assert set(agreement_schema["properties"]) >= {
        "tool_checks",
        "sanity_checks",
        "special_limit_only",
        "notes",
    }


def test_blind_reference_check_disables_proposer_source_access_by_default(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    proposer = fake.calls[0]["loops"][0]["proposers"][0]
    proposer_template = proposer["prompt"]["template"]
    assert proposer["runtime"]["allow_internet"] is False
    assert proposer["runtime"]["allow_mcp"] is False
    assert proposer["runtime"]["codex_sandbox"] == "read-only"
    assert "Do not use internet search" in proposer_template
    assert "Do not use ARC paper MCP tools" in proposer_template
    assert "Do not read paper source sections" in proposer_template


def test_consensus_allows_step_opt_in_proposer_source_access_for_blind_check(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "proposer_runtime": {"allow_internet": True, "allow_mcp": True},
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    proposer = fake.calls[0]["loops"][0]["proposers"][0]
    proposer_template = proposer["prompt"]["template"]
    assert proposer["runtime"]["allow_internet"] is True
    assert proposer["runtime"]["allow_mcp"] is True
    assert proposer["runtime"]["mcp_mode"] == "arc-only"
    assert "You may use ARC paper MCP tools" in proposer_template
    assert "Internet search is allowed" in proposer_template


def test_reviewer_reference_claim_is_not_shared_with_proposers(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
            )
        ]
    )
    reference_claim = {
        "id": "ref_eq_001",
        "label": "target reference equation",
        "latex": "x = y + z",
        "source": {"paper_id": "arXiv:1234.5678", "section": "S2"},
    }
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "kind": "new_calculation",
                "prompt": "Derive x in terms of y and z from the supplied definitions.",
                "allowed_context": {"definitions": ["x, y, z are scalar symbols"]},
                "reviewer_reference_claim": reference_claim,
            }
        ],
    )

    run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    loop = fake.calls[0]["loops"][0]
    caller_context_json = json.dumps(loop["caller_context"])
    proposer_json = json.dumps(loop["proposers"])
    reviewer_json = json.dumps(loop["reviewers"])
    assert "x = y + z" not in caller_context_json
    assert "x = y + z" not in proposer_json
    assert "x = y + z" in reviewer_json
    assert "reviewer_reference_claim" in reviewer_json


def test_reviewer_prompt_selects_best_written_for_reference_disagrees(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x = y - z", "reference_claim_status": "disagrees"},
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    reviewer_template = fake.calls[0]["loops"][0]["reviewers"][0]["prompt"]["template"]
    assert "set status to all_agree, two_agree, all_disagree, unresolved, or reference_disagrees" in reviewer_template
    assert "When status is reference_disagrees" in reviewer_template
    assert "choose best_written_proposer_id from the agreeing blind proposer ids" in reviewer_template


def test_reference_disagrees_accepts_when_two_blind_proposers_agree(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x = y - z", "reference_claim_status": "disagrees"},
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"
    assert result["steps"][0]["reviewer_consensus"]["status"] == "reference_disagrees"
    assert result["steps"][0]["accepted_output"]["reference_claim_status"] == "disagrees"


def test_human_gate_blocks_reference_disagrees(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x = y - z", "reference_claim_status": "disagrees"},
                workflow_action={
                    "action": "pause_for_human",
                    "requires_human": True,
                    "issue_type": "reference_disagreement",
                    "reason": "blind derivation differs from the note formula",
                    "expert_question": "Which formula is intended?",
                },
            ),
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "later"},
            ),
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        human_gate={"enabled": True},
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            },
            {"step_id": "later_step", "prompt": "Derive later."},
        ],
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "blocked_for_user"
    assert result["steps"][0]["status"] == "blocked_for_user"
    assert result["steps"][0]["blocked_output"]["reason"] == "human_gate"
    assert result["steps"][0]["blocked_output"]["trigger_status"] == "reference_disagrees"
    assert result["steps"][0]["blocked_output"]["requires_human"] is True
    assert result["steps"][0]["blocked_output"]["workflow_action"]["issue_type"] == "reference_disagreement"
    assert len(result["steps"]) == 1
    assert len(fake.calls) == 1


def test_human_gate_stops_for_agreed_plan_revision_without_human(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_disagree",
                workflow_action={
                    "action": "revise_plan",
                    "requires_human": False,
                    "issue_type": "plan_wrong",
                    "proposed_revision": {"split_step": "derive intermediate q first"},
                    "reason": "all proposer assessments identify the same missing intermediate",
                    "expert_question": "",
                },
            ),
        ]
    )
    config = minimal_config(
        tmp_path,
        human_gate={"enabled": True},
        max_recalculations=2,
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "blocked_for_revision"
    assert result["steps"][0]["status"] == "blocked_for_revision"
    assert result["steps"][0]["blocked_output"]["requires_human"] is False
    assert result["steps"][0]["blocked_output"]["workflow_action"]["action"] == "revise_plan"
    assert len(result["steps"][0]["attempts"]) == 1
    assert len(fake.calls) == 1


def test_human_gate_blocks_worker_failure(tmp_path):
    calls: list[dict[str, Any]] = []

    def failing_batch_runner(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(config)
        return {"schema_version": "arc.llm.proposers_reviewer_batch.result.v1", "status": "failed"}

    result = run_proposers_reviewer_consensus(
        minimal_config(tmp_path, human_gate={"enabled": True}),
        batch_runner=failing_batch_runner,
        base_env={},
    )

    assert result["status"] == "blocked_for_user"
    assert result["steps"][0]["status"] == "blocked_for_user"
    assert result["steps"][0]["blocked_output"]["trigger_status"] == "failed"
    assert result["steps"][0]["blocked_output"]["workflow_action"]["issue_type"] == "worker_failure"
    assert result["steps"][0]["error"] == "attempt batch did not complete"
    assert len(calls) == 1


def test_reference_disagrees_requires_agreement_assessment(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement=False,
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "agreement_assessment" in result["steps"][0]["error"]


def test_reference_disagrees_requires_two_agreeing_blind_proposer_ids(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001"],
                accepted={"result": "x"},
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "reference_disagrees requires two agreeing blind proposer ids" in result["steps"][0]["error"]


def test_reference_disagrees_requires_target_and_convention_match(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={"convention_match": False},
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "convention_match" in result["steps"][0]["error"]


def test_consensus_fails_when_integrity_reference_is_missing(tmp_path):
    fake = FakeBatchRunner([])
    config = minimal_config(
        tmp_path,
        defaults={"integrity_reference_path": str(tmp_path / "missing-integrity.md")},
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "integrity" in result["steps"][0]["error"].lower()
    assert fake.calls == []


def test_consensus_two_agree_recalculates_only_likely_wrong_proposer(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "two_agree",
                agreed=["proposer_001", "proposer_002"],
                likely_wrong=["proposer_003"],
                recalculate=["proposer_003"],
            ),
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "fixed"},
            ),
        ]
    )

    result = run_proposers_reviewer_consensus(
        minimal_config(tmp_path, proposer_count=3),
        batch_runner=fake,
        base_env={},
    )

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"
    assert fake.active_proposers_by_call == [
        ["proposer_001", "proposer_002", "proposer_003"],
        ["proposer_003"],
    ]
    assert fake.caller_context_by_call[1]["locked_outputs"].keys() == {"proposer_001", "proposer_002"}


def test_consensus_passes_accepted_prior_step_outputs_to_later_steps(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
            ),
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "y"},
            ),
        ]
    )
    config = minimal_config(
        tmp_path,
        steps=[
            {"step_id": "step_001", "prompt": "derive x"},
            {"step_id": "step_002", "prompt": "derive y from x"},
        ],
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert fake.caller_context_by_call[0]["accepted_prior_step_outputs"] == {}
    assert fake.caller_context_by_call[1]["accepted_prior_step_outputs"] == {
        "step_001": {"result": "x"}
    }


def test_consensus_two_agree_without_isolated_wrong_proposer_recalculates_all(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review("two_agree", agreed=["proposer_001", "proposer_002"]),
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "fixed"},
            ),
        ]
    )

    run_proposers_reviewer_consensus(
        minimal_config(tmp_path, proposer_count=3),
        batch_runner=fake,
        base_env={},
    )

    assert fake.active_proposers_by_call == [
        ["proposer_001", "proposer_002", "proposer_003"],
        ["proposer_001", "proposer_002", "proposer_003"],
    ]


def test_consensus_all_disagree_recalculates_all(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review("all_disagree"),
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "resolved"},
            ),
        ]
    )

    run_proposers_reviewer_consensus(
        minimal_config(tmp_path, proposer_count=3, max_recalculations=1),
        batch_runner=fake,
        base_env={},
    )

    assert fake.active_proposers_by_call == [
        ["proposer_001", "proposer_002", "proposer_003"],
        ["proposer_001", "proposer_002", "proposer_003"],
    ]


def test_consensus_blocks_for_user_at_recalculation_limit(tmp_path):
    fake = FakeBatchRunner([consensus_review("all_disagree"), consensus_review("all_disagree")])

    result = run_proposers_reviewer_consensus(
        minimal_config(tmp_path, max_recalculations=1),
        batch_runner=fake,
        base_env={},
    )

    assert result["status"] == "blocked_for_user"
    assert result["steps"][0]["status"] == "blocked_for_user"
    assert result["steps"][0]["blocked_output"]["analysis"] == "review analysis"
    assert len(result["steps"][0]["attempts"]) == 2


def test_consensus_rejects_all_agree_without_agreement_assessment(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement=False,
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert result["steps"][0]["status"] == "failed"
    assert "agreement_assessment" in result["steps"][0]["error"]


def test_consensus_rejects_all_agree_without_best_written_selection(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                best_written=False,
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "best_written_proposer_id" in result["steps"][0]["error"]


def test_consensus_rejects_all_agree_with_best_written_outside_agreed_proposers(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                best_written_proposer_id="proposer_003",
            )
        ]
    )

    result = run_proposers_reviewer_consensus(
        minimal_config(tmp_path, proposer_count=3),
        batch_runner=fake,
        base_env={},
    )

    assert result["status"] == "failed"
    assert "agreed_proposer_ids" in result["steps"][0]["error"]


def test_consensus_rejects_all_agree_without_required_true_agreement_field(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={"agreement_covers_full_target": False},
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "agreement_covers_full_target" in result["steps"][0]["error"]


def test_consensus_rejects_visual_or_string_similarity_summary(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={
                    "comparison_summary": "The formulas look identical apart from spacing and formatting.",
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "formatting, spacing, visual similarity, looks identical, or string equality" in result["steps"][0]["error"]


def test_consensus_accepts_negated_visual_comparison_summary(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={
                    "comparison_summary": (
                        "Equivalence follows from matching target quantity, conventions, "
                        "and full derivation coverage, not by visual comparison."
                    ),
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


def test_consensus_accepts_negated_visual_inspection_summary(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={
                    "comparison_summary": (
                        "Equivalence follows from matching target quantity, conventions, "
                        "and full derivation coverage, not by visual inspection."
                    ),
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


def test_consensus_accepts_without_visual_inspection_summary(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={
                    "comparison_summary": (
                        "Equivalence follows from matching target quantity and conventions "
                        "without visual inspection."
                    ),
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


def test_consensus_accepts_not_relying_on_visual_inspection_summary(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={
                    "comparison_summary": (
                        "The reviewer matched the target and conventions, not relying on "
                        "visual inspection."
                    ),
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


@pytest.mark.parametrize(
    "comparison_summary",
    [
        "The formulas do not look identical; algebra matches.",
        "Equivalence is established not by string equality but by matched derivation.",
        "Never rely on visual inspection; algebra matches.",
    ],
)
def test_consensus_accepts_negated_weak_marker_summary(tmp_path, comparison_summary):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={"comparison_summary": comparison_summary},
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


@pytest.mark.parametrize(
    "comparison_summary",
    [
        "Accepted by visual inspection of the displayed formulas.",
        "Accepted based on visual inspection of the displayed formulas.",
        "Accepted by string equality of the displayed formulas.",
    ],
)
def test_consensus_rejects_visual_inspection_reliance_summary(tmp_path, comparison_summary):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={"comparison_summary": comparison_summary},
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "formatting, spacing, visual similarity, looks identical, or string equality" in result["steps"][0]["error"]


def test_consensus_rejects_special_limit_only_acceptance(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={"special_limit_only": True},
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "special_limit_only" in result["steps"][0]["error"]


def test_consensus_accepts_reviewer_judgment_without_sympy(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                agreement_overrides={
                    "tool_checks": [],
                    "comparison_summary": (
                        "Both derivations compute the same target quantity with matching "
                        "conventions and the full declared scope."
                    ),
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"
    assert "main_agent_agreement_check" not in result["steps"][0]["reviewer_consensus"]


def test_consensus_dry_run_does_not_call_batch_runner(tmp_path):
    def fail_batch_runner(*args, **kwargs):
        raise AssertionError("dry-run must not call the batch runner")

    result = run_proposers_reviewer_consensus(
        minimal_config(tmp_path),
        batch_runner=fail_batch_runner,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["proposer_count"] == 2
    assert result["max_recalculations"] == 2
    assert result["human_gate"]["enabled"] is False
    assert result["steps"] == [{"step_id": "step_001", "kind": "new_calculation"}]


class FakeBatchRunner:
    def __init__(
        self,
        reviews: list[dict[str, Any]],
        *,
        proposer_outputs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.reviews = list(reviews)
        self.proposer_outputs = proposer_outputs or {}
        self.calls: list[dict[str, Any]] = []

    @property
    def active_proposers_by_call(self) -> list[list[str]]:
        return [[item["id"] for item in call["loops"][0]["proposers"]] for call in self.calls]

    @property
    def caller_context_by_call(self) -> list[dict[str, Any]]:
        return [call["loops"][0]["caller_context"] for call in self.calls]

    def __call__(self, config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        call_number = len(self.calls) + 1
        self.calls.append(json.loads(json.dumps(config)))
        loop = config["loops"][0]
        run_root = Path(config["run_dir"]) / config["run_id"]
        round_root = run_root / "loops" / loop["loop_id"] / "rounds" / "round_001"
        for proposer in loop["proposers"]:
            path = round_root / "proposer_outputs" / f"{proposer['id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.proposer_outputs.get(
                proposer["id"],
                {"proposer_id": proposer["id"], "call_number": call_number},
            )
            path.write_text(
                json.dumps(payload) + "\n",
                encoding="utf-8",
            )
        review = self.reviews.pop(0)
        review_path = round_root / "reviews" / "reviewer_001.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps(review) + "\n", encoding="utf-8")
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": config["run_id"],
            "run_root": str(run_root),
            "loops": [{"loop_id": loop["loop_id"], "status": "completed"}],
        }


def consensus_review(
    status: str,
    *,
    agreed: list[str] | None = None,
    likely_wrong: list[str] | None = None,
    recalculate: list[str] | None = None,
    accepted: dict[str, Any] | None = None,
    agreement: bool = True,
    agreement_overrides: dict[str, Any] | None = None,
    best_written: bool = True,
    best_written_proposer_id: str | None = "proposer_001",
    workflow_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    consensus: dict[str, Any] = {
        "status": status,
        "accepted_result": accepted,
        "agreed_proposer_ids": agreed or [],
        "likely_wrong_proposer_ids": likely_wrong or [],
        "recalculate_proposer_ids": recalculate or [],
        "validity_scope": "valid under stated assumptions and foundation conventions",
        "analysis": "review analysis",
        "workflow_action": workflow_action or default_workflow_action_for_status(status),
    }
    if best_written:
        consensus["best_written_proposer_id"] = best_written_proposer_id
        consensus["best_written_selection_reason"] = "clearest logic and most complete derivation"
    if agreement:
        agreement_payload = {
            "target_quantity_match": True,
            "convention_match": True,
            "declared_scope_match": True,
            "agreement_covers_full_target": True,
            "comparison_summary": "Reviewer judged the calculations equivalent across the full target.",
            "accepted_by_reviewer_judgment": True,
            "tool_checks": [],
            "sanity_checks": [],
            "special_limit_only": False,
            "notes": "",
        }
        if agreement_overrides:
            agreement_payload.update(agreement_overrides)
        consensus["agreement_assessment"] = agreement_payload
    return {
        "schema_version": "arc.llm.review_envelope.v1",
        "controller": {"message": "reviewed", "stop_requested": False},
        "proposer_messages": {},
        "review_payload": {"consensus": consensus},
    }


def default_workflow_action_for_status(status: str) -> dict[str, Any]:
    if status == "all_agree":
        return {
            "action": "continue",
            "requires_human": False,
            "issue_type": "none",
            "proposed_revision": None,
            "reason": "all proposers agree",
            "expert_question": "",
        }
    issue_type = "reference_disagreement" if status == "reference_disagrees" else "calculation_disagreement"
    return {
        "action": "retry",
        "requires_human": False,
        "issue_type": issue_type,
        "proposed_revision": None,
        "reason": "fake retry action",
        "expert_question": "",
    }
