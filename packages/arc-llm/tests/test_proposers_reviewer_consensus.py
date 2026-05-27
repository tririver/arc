from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def test_consensus_config_defaults_to_three_proposers_and_three_recalculations(tmp_path):
    config = load_consensus_config(minimal_config(tmp_path))

    assert config.proposer_count == 3
    assert config.max_recalculations == 3


def test_consensus_accepts_all_agree_on_first_attempt(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"
    assert result["steps"][0]["accepted_output"] == {"result": "x"}
    assert fake.active_proposers_by_call == [["proposer_001", "proposer_002", "proposer_003"]]
    reviewer_schema = fake.calls[0]["loops"][0]["reviewers"][0]["output_schema"]
    assert "schema_version" in reviewer_schema["required"]
    assert "proposer_messages" in reviewer_schema["required"]
    assert "review_payload" in reviewer_schema["required"]
    assert reviewer_schema["properties"]["schema_version"]["const"] == "arc.llm.review_envelope.v1"
    assert reviewer_schema["properties"]["proposer_messages"]["required"] == [
        "proposer_001",
        "proposer_002",
        "proposer_003",
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
    assert "reliable_until" not in proposer_template
    assert proposer["runtime"]["allow_internet"] is True
    assert proposer["runtime"]["allow_mcp"] is True
    assert proposer["runtime"]["mcp_mode"] == "arc-only"
    assert "You may use ARC paper MCP tools" in proposer_template
    assert "Internet search is allowed" in proposer_template
    assert "validation-only final formulas" in proposer_template
    lower_proposer_template = proposer_template.lower()
    assert "strictly derive from the foundation" in lower_proposer_template
    assert "external sources may inspire methods" in lower_proposer_template
    assert "do not directly use any result" in lower_proposer_template
    assert "different conventions" in lower_proposer_template
    assert reviewer["runtime"]["allow_mcp"] is False
    assert "SymPy" in reviewer_template
    assert "expand" in reviewer_template
    assert "simplify" in reviewer_template
    assert "sympy_code" in reviewer_template
    assert "substitutions" in reviewer_template
    assert "A-B" in reviewer_template
    assert "B-C" in reviewer_template
    assert "A-C" in reviewer_template
    assert "10 randomly selected data points" in reviewer_template
    assert "relative error" in reviewer_template
    assert "check history" in reviewer_template
    assert "at least two" in reviewer_template
    assert "Never mark all_agree" in reviewer_template
    consensus_properties = reviewer_schema["properties"]["review_payload"]["properties"]["consensus"]["properties"]
    assert "pairwise_symbolic_checks" in consensus_properties
    assert "best_written_proposer_id" in consensus_properties
    assert "best_written_selection_reason" in consensus_properties
    assert "validity_scope" in consensus_properties
    assert "reliable_until" not in consensus_properties
    pairwise_properties = consensus_properties["pairwise_symbolic_checks"]["properties"]
    assert "used_sympy" in pairwise_properties


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
                pairwise_check_overrides={
                    "A_minus_B_zero": True,
                    "B_minus_C_zero": False,
                    "A_minus_C_zero": False,
                    "true_count": 1,
                },
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
    assert "When status is reference_disagrees" in reviewer_template
    assert "choose best_written_proposer_id from the agreeing blind proposer ids" in reviewer_template


def test_reference_disagrees_accepts_when_two_blind_proposers_agree(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x = y - z", "reference_claim_status": "disagrees"},
                pairwise_check_overrides={
                    "A_minus_B_zero": True,
                    "B_minus_C_zero": False,
                    "A_minus_C_zero": False,
                    "true_count": 1,
                    "sympy_code": (
                        "simplify(expand(A-B)); "
                        "simplify(expand(B-C)); "
                        "simplify(expand(A-C))"
                    ),
                    "check_history": [
                        "A-B reduces to 0.",
                        "B-C reduces to 2*z.",
                        "A-C reduces to 2*z.",
                    ],
                },
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


def test_reference_disagrees_requires_blind_proposer_agreement(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "A_minus_B_zero": False,
                    "B_minus_C_zero": False,
                    "A_minus_C_zero": False,
                    "true_count": 0,
                    "sympy_code": (
                        "simplify(expand(A-B)); "
                        "simplify(expand(B-C)); "
                        "simplify(expand(A-C))"
                    ),
                },
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
    assert "reference_disagrees requires A-B=0" in result["steps"][0]["error"]


def test_reference_disagrees_requires_two_agreeing_blind_proposer_ids(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "A_minus_B_zero": True,
                    "B_minus_C_zero": False,
                    "A_minus_C_zero": False,
                    "true_count": 1,
                    "sympy_code": (
                        "simplify(expand(A-B)); "
                        "simplify(expand(B-C)); "
                        "simplify(expand(A-C))"
                    ),
                },
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


def test_blind_reference_all_agree_requires_pairwise_checks_with_reference_claim(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                pairwise_checks=False,
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
    assert "pairwise_symbolic_checks" in result["steps"][0]["error"]


def test_foundation_check_context_exposes_only_axiom_checked_and_target(tmp_path):
    foundation = {
        "schema_version": "arc.research_foundation.v1",
        "run_id": "run_001",
        "version": 1,
        "conventions": [
            {"id": "conv_checked", "check_status": "checked", "consistency_status": "normalized"},
            {"id": "conv_unchecked", "check_status": "not_checked", "consistency_status": "normalized"},
        ],
        "equations": [
            {
                "id": "eq_axiom",
                "axiom_status": "axiom",
                "check_status": "not_checked",
                "sources": [{"paper_id": "arXiv:1", "mcp": "get_section(...)", "cli": "arc-paper ..."}],
            },
            {
                "id": "eq_checked",
                "axiom_status": "not_axiom",
                "check_status": "checked_analytic",
                "sources": [{"paper_id": "arXiv:2", "mcp": "get_section(...)", "cli": "arc-paper ..."}],
            },
            {
                "id": "eq_target",
                "axiom_status": "not_axiom",
                "check_status": "not_checked",
                "sources": [{"paper_id": "arXiv:3", "mcp": "get_section(...)", "cli": "arc-paper ..."}],
            },
            {"id": "eq_unchecked", "axiom_status": "not_axiom", "check_status": "not_checked"},
        ],
    }
    foundation_path = tmp_path / "foundation.json"
    foundation_path.write_text(json.dumps(foundation), encoding="utf-8")
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        steps=[
            {
                "step_id": "check_eq_target",
                "kind": "foundation_check",
                "prompt": "check target",
                "allowed_context": {
                    "foundation_file": str(foundation_path),
                    "target_equation_id": "eq_target",
                    "source_commands": [
                        "get_section(paper_id=\"arXiv:1\", section=\"S1\")",
                        "arc-paper get-section arXiv:1 --section S1 --json",
                    ],
                },
            }
        ],
    )

    run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    foundation_context = fake.calls[0]["loops"][0]["caller_context"]["foundation_context"]
    assert foundation_context["target_equation"]["id"] == "eq_target"
    assert [item["id"] for item in foundation_context["allowed_equations"]] == [
        "eq_axiom",
        "eq_checked",
    ]
    assert [item["id"] for item in foundation_context["allowed_conventions"]] == ["conv_checked"]
    assert "eq_unchecked" in foundation_context["omitted_equation_ids"]
    assert "source_path" not in foundation_context
    assert "sources" not in json.dumps(foundation_context)
    assert "arc-paper" not in json.dumps(foundation_context)
    allowed_context = fake.calls[0]["loops"][0]["caller_context"]["allowed_context"]
    assert "foundation_file" not in allowed_context
    assert "source_commands" not in allowed_context
    assert "arc-paper" not in json.dumps(allowed_context)


def test_foundation_check_fails_when_target_equation_is_missing(tmp_path):
    foundation_path = tmp_path / "foundation.json"
    foundation_path.write_text(
        json.dumps(
            {
                "schema_version": "arc.research_foundation.v1",
                "run_id": "run_001",
                "version": 1,
                "conventions": [],
                "equations": [{"id": "eq_other", "axiom_status": "axiom", "check_status": "not_checked"}],
            }
        ),
        encoding="utf-8",
    )
    fake = FakeBatchRunner([])
    config = minimal_config(
        tmp_path,
        steps=[
            {
                "step_id": "check_missing",
                "kind": "foundation_check",
                "prompt": "check target",
                "allowed_context": {
                    "foundation_file": str(foundation_path),
                    "target_equation_id": "eq_missing",
                },
            }
        ],
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "target_equation_id eq_missing was not found" in result["steps"][0]["error"]
    assert fake.calls == []


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

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

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

    run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

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
        minimal_config(tmp_path, max_recalculations=1),
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


def test_consensus_rejects_all_agree_without_pairwise_symbolic_checks(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_checks=False,
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert result["steps"][0]["status"] == "failed"
    assert "pairwise_symbolic_checks" in result["steps"][0]["error"]


def test_consensus_rejects_all_agree_without_best_written_selection(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
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

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "agreed_proposer_ids" in result["steps"][0]["error"]


def test_consensus_rejects_numerical_all_agree_with_too_few_samples(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "check_method": "numerical",
                    "sample_count": 9,
                    "numerical_relative_error": 1e-8,
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "at least 10" in result["steps"][0]["error"]


def test_consensus_rejects_all_agree_without_check_method(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={"check_method": ""},
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "check_method" in result["steps"][0]["error"]


def test_consensus_rejects_mixed_all_agree_with_too_few_samples(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "check_method": "mixed",
                    "sample_count": 9,
                    "numerical_relative_error": 1e-8,
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "at least 10" in result["steps"][0]["error"]


def test_consensus_rejects_sympy_all_agree_without_expand_and_simplify(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={"sympy_code": "simplify(A-B)"},
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "expand" in result["steps"][0]["error"]


def test_consensus_rejects_manual_all_agree_by_string_or_spacing_comparison(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "used_sympy": False,
                    "sympy_code": "",
                    "notes": "Manual comparison found only spacing differences.",
                    "check_method": "analytic",
                    "check_history": [
                        "Compared A and B: only difference is spacing.",
                        "Compared B and C: identical.",
                        "Compared A and C: identical.",
                    ],
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "manual all_agree" in result["steps"][0]["error"]


def test_consensus_accepts_main_agent_sympy_fallback_for_bad_reviewer_evidence(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "used_sympy": False,
                    "sympy_code": "",
                    "notes": "Manual comparison found only spacing differences.",
                    "check_method": "analytic",
                    "check_history": [
                        "Compared A and B as strings.",
                        "Compared B and C by visual inspection.",
                        "Compared A and C: identical.",
                    ],
                },
            )
        ],
        proposer_outputs={
            "proposer_001": {"final_result": "W_gt_kernel = f_eta*fstar_etap\nW_lt_kernel = g_eta*gstar_etap"},
            "proposer_002": {
                "final_result": {
                    "W_gt_kernel": "f_eta * fstar_etap",
                    "W_lt_kernel": "g_eta * gstar_etap",
                }
            },
            "proposer_003": {
                "final_result": "W_gt_kernel = f_eta*fstar_etap\nW_lt_kernel = g_eta*gstar_etap"
            },
        },
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    checks = result["steps"][0]["reviewer_consensus"]["pairwise_symbolic_checks"]
    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"
    assert checks["used_sympy"] is True
    assert checks["fallback_source"] == "main_agent_sympy"


def test_consensus_accepts_manual_all_agree_with_explicit_differences(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "used_sympy": False,
                    "sympy_code": "",
                    "notes": "Expressions are identical after explicit differences: A-B=x-x=0; B-C=x-x=0; A-C=x-x=0.",
                    "check_method": "analytic",
                    "check_history": [
                        "A-B=x-x=0",
                        "B-C=x-x=0",
                        "A-C=x-x=0",
                    ],
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


def test_consensus_accepts_manual_all_agree_with_spaced_difference_labels(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "used_sympy": False,
                    "sympy_code": "",
                    "notes": "Explicit differences are zero.",
                    "check_method": "analytic",
                    "check_history": [
                        "A - B: x - x = 0",
                        "B - C: x - x = 0",
                        "A - C: x - x = 0",
                    ],
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


def test_consensus_accepts_manual_all_agree_when_pairwise_differences_reduce_to_zero(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "used_sympy": False,
                    "sympy_code": "",
                    "notes": "Expressions differ only by notation; all pairwise differences reduce to zero symbolically.",
                    "check_method": "analytic",
                    "check_history": [
                        "Compared proposer_001 and proposer_002: differences reduce to zero.",
                        "Compared proposer_002 and proposer_003: differences reduce to zero.",
                        "Compared proposer_001 and proposer_003: differences reduce to zero.",
                    ],
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


def test_consensus_accepts_manual_all_agree_with_term_by_term_check(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "used_sympy": False,
                    "sympy_code": "",
                    "notes": "All expressions identical after notation changes. Explicit term-by-term comparison shows every term matches.",
                    "check_method": "analytic",
                    "check_history": [
                        "Checked term 1 in all expressions.",
                        "Checked term 2 in all expressions.",
                        "Checked the overall factor.",
                    ],
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


def test_consensus_accepts_manual_all_agree_when_named_differences_are_zero(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002", "proposer_003"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "used_sympy": False,
                    "sympy_code": "",
                    "notes": "The differences A-B, B-C, A-C are zero because expressions match term-by-term.",
                    "check_method": "analytic",
                    "check_history": ["Concluded all differences zero."],
                },
            )
        ]
    )

    result = run_proposers_reviewer_consensus(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


def test_consensus_dry_run_does_not_call_batch_runner(tmp_path):
    def fail_batch_runner(*args, **kwargs):
        raise AssertionError("dry-run must not call the batch runner")

    result = run_proposers_reviewer_consensus(
        minimal_config(tmp_path),
        batch_runner=fail_batch_runner,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["proposer_count"] == 3
    assert result["max_recalculations"] == 3
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
    pairwise_checks: bool = True,
    pairwise_check_overrides: dict[str, Any] | None = None,
    best_written: bool = True,
    best_written_proposer_id: str | None = "proposer_001",
) -> dict[str, Any]:
    consensus: dict[str, Any] = {
        "status": status,
        "accepted_result": accepted,
        "agreed_proposer_ids": agreed or [],
        "likely_wrong_proposer_ids": likely_wrong or [],
        "recalculate_proposer_ids": recalculate or [],
        "validity_scope": "valid under stated assumptions and foundation conventions",
        "analysis": "review analysis",
    }
    if best_written:
        consensus["best_written_proposer_id"] = best_written_proposer_id
        consensus["best_written_selection_reason"] = "clearest logic and most complete derivation"
    if pairwise_checks:
        pairwise_payload = {
            "used_sympy": True,
            "A_minus_B_zero": True,
            "B_minus_C_zero": True,
            "A_minus_C_zero": True,
            "true_count": 3,
            "sympy_code": "simplify(expand(A-B)); simplify(expand(B-C)); simplify(expand(A-C))",
            "notes": "fake pairwise checks",
            "check_method": "analytic",
            "numerical_relative_error": None,
            "sample_count": 0,
            "check_history": ["expanded and simplified"],
        }
        if pairwise_check_overrides:
            pairwise_payload.update(pairwise_check_overrides)
        consensus["pairwise_symbolic_checks"] = pairwise_payload
    return {
        "schema_version": "arc.llm.review_envelope.v1",
        "controller": {"message": "reviewed", "stop_requested": False},
        "proposer_messages": {},
        "review_payload": {"consensus": consensus},
    }
