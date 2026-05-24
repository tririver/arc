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
    assert proposer["runtime"]["allow_internet"] is False
    assert proposer["runtime"]["allow_mcp"] is False
    assert reviewer["runtime"]["allow_mcp"] is False
    assert "SymPy" in reviewer_template
    assert "expand" in reviewer_template
    assert "simplify" in reviewer_template
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
    pairwise_properties = consensus_properties["pairwise_symbolic_checks"]["properties"]
    assert "used_sympy" in pairwise_properties


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
    assert "foundation_file" not in fake.calls[0]["loops"][0]["caller_context"]["allowed_context"]


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
    def __init__(self, reviews: list[dict[str, Any]]) -> None:
        self.reviews = list(reviews)
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
            path.write_text(
                json.dumps({"proposer_id": proposer["id"], "call_number": call_number}) + "\n",
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
) -> dict[str, Any]:
    consensus: dict[str, Any] = {
        "status": status,
        "accepted_result": accepted,
        "agreed_proposer_ids": agreed or [],
        "likely_wrong_proposer_ids": likely_wrong or [],
        "recalculate_proposer_ids": recalculate or [],
        "reliable_until": "attempt boundary",
        "analysis": "review analysis",
    }
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
