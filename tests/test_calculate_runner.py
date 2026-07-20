from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "plugins/arc/skills/arc/workflows"
WJ = WF / "json"
WS = WF / "scripts"


def load_calculate_runner():
    spec = importlib.util.spec_from_file_location("calculate_runner", WS / "calculate_runner.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["calculate_runner"] = module
    assert spec.loader is not None
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def minimal_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "arc.workflow.calculate.config.v1",
        "run_id": "calc_001",
        "run_dir": str(tmp_path / "execute"),
        "workflow_json_dir": str(WJ),
        "steps": [{"step_id": "step_001", "prompt": "derive x"}],
    }
    payload.update(overrides)
    return payload


def test_calculate_runner_uses_templates_and_hides_reviewer_reference_claim(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner([calculate_review("all_agree", agreed=["proposer_001", "proposer_002"])])
    reference_claim = {"id": "ref_eq_001", "latex": "x = y + z"}
    config = minimal_config(
        tmp_path,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "derive x",
                "reviewer_reference_claim": reference_claim,
            }
        ],
    )

    result = runner.run_calculation(config, batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    batch = fake.calls[0]
    loop = batch["loops"][0]
    caller_context = loop["caller_context"]
    assert batch["schema_version"] == "arc.llm.proposers_reviewer_batch.config.v1"
    assert "Scientific Integrity Notice" in caller_context["integrity_reference"]["content"]
    assert "reviewer_reference_claim" not in json.dumps(caller_context)
    assert "reviewer_reference_claim" in loop["reviewers"][0]["prompt"]["template"]
    assert "reviewer_reference_claim" not in loop["proposers"][0]["prompt"]["template"]
    assert loop["proposers"][0]["runtime"]["allow_internet"] is False
    assert loop["proposers"][0]["runtime"]["allow_mcp"] is False
    assert result["warnings_summary"]["structured_output_warning_count"] == 0


def test_calculate_runner_recalculates_only_isolated_wrong_proposer(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "two_agree",
                agreed=["proposer_001", "proposer_002"],
                likely_wrong=["proposer_003"],
                recalculate=["proposer_003"],
            ),
            calculate_review("all_agree", agreed=["proposer_003"], best_written="proposer_001"),
        ]
    )

    result = runner.run_calculation(
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
    assert fake.calls[0]["session"]["scope_id"] == "calculate/calc_001/step_001"
    assert fake.calls[1]["session"]["scope_id"] == "calculate/calc_001/step_001"
    assert fake.calls[0]["session"]["root"] == fake.calls[1]["session"]["root"]
    assert fake.calls[0]["session"]["reuse_across_batch_calls"] is True
    assert fake.calls[0]["loops"][0]["cache_context"]["static_caller_context_keys"] == [
        "step_id",
        "step_kind",
        "step_prompt",
        "allowed_context",
        "accepted_prior_step_outputs",
        "max_recalculations",
        "integrity_reference",
        "consensus_instruction",
    ]
    assert sorted(fake.calls[1]["loops"][0]["caller_context"]["locked_outputs"]) == [
        "proposer_001",
        "proposer_002",
    ]


def test_major_recovered_reviewer_output_cannot_accept_calculation(tmp_path):
    runner = load_calculate_runner()
    review = calculate_review("all_agree", agreed=["proposer_001", "proposer_002"])
    review["arc_llm_call_record"] = {
        "structured_output": {
            "mode": "recovered",
            "severity": "major",
            "recovery_strategy": "natural_language_fallback",
        }
    }
    fake = FakeBatchRunner([review])

    result = runner.run_calculation(
        minimal_config(tmp_path, max_recalculations=0),
        batch_runner=fake,
        base_env={},
    )

    step = result["steps"][0]
    assert step["status"] == "blocked_for_user"
    assert step["reviewer_consensus"]["status"] == "unresolved"
    assert step["reviewer_consensus"]["accepted_result"] is None


def test_major_recovered_reviewer_blocks_without_full_retry(tmp_path):
    runner = load_calculate_runner()
    review = calculate_review("all_agree", agreed=["proposer_001", "proposer_002"])
    review["arc_llm_call_record"] = {
        "structured_output": {
            "mode": "recovered",
            "severity": "major",
            "recovery_strategy": "peer_visible_reviewer_fallback",
        }
    }
    fake = FakeBatchRunner(
        [
            review,
            calculate_review("all_agree", agreed=["proposer_001", "proposer_002"]),
        ]
    )

    result = runner.run_calculation(minimal_config(tmp_path), batch_runner=fake, base_env={})

    step = result["steps"][0]
    assert len(fake.calls) == 1
    assert step["status"] == "blocked_for_user"
    assert step["accepted_output"] is None
    assert step["blocked_output"]["reason"] == "reviewer_structured_output_recovery"
    assert step["reviewer_consensus"]["status"] == "unresolved"


def test_human_gate_does_not_preempt_retry_budget_for_retryable_status(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "unresolved",
                likely_wrong=["proposer_002"],
                recalculate=["proposer_002"],
                action="pause_for_human",
                requires_human=True,
            ),
            calculate_review("all_agree", agreed=["proposer_001", "proposer_002"]),
        ]
    )

    result = runner.run_calculation(
        minimal_config(
            tmp_path,
            human_gate={
                "enabled": True,
                "pause_on_statuses": ["unresolved"],
            },
        ),
        batch_runner=fake,
        base_env={},
    )

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"
    assert len(fake.calls) == 2


def test_reviewer_feedback_is_available_to_retry_attempt(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "unresolved",
                proposer_messages={
                    "proposer_001": "State the source notation explicitly.",
                    "proposer_002": "Map the coefficient labels back before final answer.",
                },
            ),
            calculate_review("all_agree", agreed=["proposer_001", "proposer_002"]),
        ]
    )

    result = runner.run_calculation(
        minimal_config(tmp_path),
        batch_runner=fake,
        base_env={},
    )

    assert result["status"] == "completed"
    retry_context = fake.calls[1]["loops"][0]["caller_context"]["retry_feedback"]
    assert retry_context[0]["status"] == "unresolved"
    assert (
        retry_context[0]["proposer_messages"]["proposer_001"]["message"]
        == "State the source notation explicitly."
    )
    assert (
        retry_context[0]["proposer_messages"]["proposer_002"]["message"]
        == "Map the coefficient labels back before final answer."
    )


def test_reference_disagreement_retries_before_human_gate(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                target_quantity_match=False,
                accepted_by_reviewer_judgment=False,
                proposer_messages={
                    "proposer_001": "Recheck the source momentum label.",
                    "proposer_002": "Recheck the source momentum label.",
                },
            ),
            calculate_review("all_agree", agreed=["proposer_001", "proposer_002"]),
        ]
    )

    result = runner.run_calculation(
        minimal_config(
            tmp_path,
            human_gate={
                "enabled": True,
                "pause_on_statuses": ["reference_disagrees"],
            },
            steps=[
                {
                    "step_id": "blind_ref_eq_001",
                    "prompt": "derive x",
                    "reviewer_reference_claim": {"id": "target", "latex": "x"},
                }
            ],
        ),
        batch_runner=fake,
        base_env={},
    )

    assert result["status"] == "completed"
    assert len(fake.calls) == 2
    retry_context = fake.calls[1]["loops"][0]["caller_context"]["retry_feedback"]
    assert retry_context[0]["status"] == "reference_disagrees"
    assert (
        retry_context[0]["proposer_messages"]["proposer_001"]["message"]
        == "Recheck the source momentum label."
    )


def test_calculate_runner_blocks_on_reference_disagreement_without_failing_validation(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                best_written="proposer_001",
                target_quantity_match=False,
                accepted_by_reviewer_judgment=False,
                special_limit_only=True,
            )
        ]
    )

    result = runner.run_calculation(
        minimal_config(
            tmp_path,
            max_recalculations=0,
            steps=[
                {
                    "step_id": "blind_ref_eq_001",
                    "prompt": "derive x",
                    "reviewer_reference_claim": {"id": "target", "latex": "x"},
                }
            ],
        ),
        batch_runner=fake,
        base_env={},
    )

    assert result["status"] == "blocked_for_user"
    step = result["steps"][0]
    assert step["status"] == "blocked_for_user"
    assert step["blocked_output"]["trigger_status"] == "reference_disagrees"
    assert "error" not in step


def test_reference_disagreement_can_be_convention_mismatch(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                best_written="proposer_001",
                accepted_by_reviewer_judgment=False,
                convention_match=False,
            )
        ]
    )

    result = runner.run_calculation(
        minimal_config(
            tmp_path,
            max_recalculations=0,
            steps=[
                {
                    "step_id": "blind_ref_eq_001",
                    "prompt": "derive x",
                    "reviewer_reference_claim": {"id": "target", "latex": "x"},
                }
            ],
        ),
        batch_runner=fake,
        base_env={},
    )

    assert result["status"] == "blocked_for_user"
    step = result["steps"][0]
    assert step["blocked_output"]["trigger_status"] == "reference_disagrees"
    assert "error" not in step


def test_reference_disagreement_can_be_scope_or_coverage_mismatch(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                best_written="proposer_001",
                target_quantity_match=True,
                convention_match=True,
                declared_scope_match=False,
                agreement_covers_full_target=False,
                accepted_by_reviewer_judgment=False,
            )
        ]
    )

    result = runner.run_calculation(
        minimal_config(
            tmp_path,
            max_recalculations=0,
            steps=[
                {
                    "step_id": "blind_ref_eq_001",
                    "prompt": "derive x",
                    "reviewer_reference_claim": {"id": "target", "latex": "x"},
                }
            ],
        ),
        batch_runner=fake,
        base_env={},
    )

    assert result["status"] == "blocked_for_user"
    step = result["steps"][0]
    assert step["blocked_output"]["trigger_status"] == "reference_disagrees"
    assert "error" not in step


def test_all_agree_likely_source_error_blocks_for_human(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                source_discrepancy_status="likely_source_error",
                source_discrepancy_confidence_reason="derivation disagrees with source but convention may differ",
                reviewer_says_no_human_convention_choice_needed=False,
            )
        ]
    )

    result = runner.run_calculation(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "blocked_for_user"
    step = result["steps"][0]
    assert step["status"] == "blocked_for_user"
    assert step["blocked_output"]["trigger_status"] == "all_agree"
    assert step["blocked_output"]["reason"] == "source_discrepancy_requires_human"
    assert "Human expert" not in step["blocked_output"]["expert_question"]
    assert step["blocked_output"]["source_discrepancies"][0]["item_id"] == "source_discrepancy"


def test_all_agree_blocks_on_every_nonconfirmed_source_discrepancy(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                source_discrepancies=[
                    {
                        "item_id": "eq_00008",
                        "status": "likely_source_error",
                        "source_claim": "eq_00008 uses H^2/sqrt(2k^3)",
                        "derived_result": "canonical normalization gives H/sqrt(2k^3)",
                        "confidence_reason": "both proposers and reviewer find an extra H",
                        "reviewer_says_no_human_convention_choice_needed": False,
                        "decision_question": (
                            "Is eq_00008 a source normalization typo, or is an unstated "
                            "field rescaling intended?"
                        ),
                    },
                    {
                        "item_id": "eq_00009",
                        "status": "ambiguous_convention",
                        "source_claim": "eq_00009 omits the SK branch factor",
                        "derived_result": "ordinary SK propagators give a branch-signed contact term",
                        "confidence_reason": "source may use an implicit signed SK metric",
                        "reviewer_says_no_human_convention_choice_needed": False,
                        "decision_question": (
                            "Should eq_00009 use an implicit signed SK metric, or should "
                            "downstream collapse signs remain pending?"
                        ),
                    },
                ],
            )
        ]
    )

    result = runner.run_calculation(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "blocked_for_user"
    blocked = result["steps"][0]["blocked_output"]
    assert blocked["reason"] == "source_discrepancy_requires_human"
    assert [item["item_id"] for item in blocked["source_discrepancies"]] == ["eq_00008", "eq_00009"]
    assert "eq_00008" in blocked["expert_question"]
    assert "eq_00009" in blocked["expert_question"]


def test_old_single_source_discrepancy_output_is_rejected(tmp_path):
    runner = load_calculate_runner()
    review = calculate_review("all_agree", agreed=["proposer_001", "proposer_002"])
    consensus = review["review_payload"]["consensus"]
    consensus.pop("source_discrepancies")
    consensus["source_discrepancy"] = {
        "status": "likely_source_error",
        "source_claim": "old source claim",
        "derived_result": "old derived result",
        "confidence_reason": "old schema",
        "reviewer_says_no_human_convention_choice_needed": False,
    }
    fake = FakeBatchRunner([review])

    result = runner.run_calculation(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "source_discrepancies" in result["steps"][0]["error"]


def test_legacy_source_discrepancy_field_is_rejected_even_with_new_array(tmp_path):
    runner = load_calculate_runner()
    review = calculate_review("all_agree", agreed=["proposer_001", "proposer_002"])
    consensus = review["review_payload"]["consensus"]
    consensus["source_discrepancy"] = {
        "status": "likely_source_error",
        "source_claim": "old source claim",
        "derived_result": "old derived result",
        "confidence_reason": "old schema",
        "reviewer_says_no_human_convention_choice_needed": False,
    }
    fake = FakeBatchRunner([review])

    result = runner.run_calculation(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "source_discrepancy" in result["steps"][0]["error"]


def test_all_agree_confirmed_source_error_can_continue(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                source_discrepancy_status="confirmed_source_error",
                source_discrepancy_confidence_reason=(
                    "blind proposers agree, reviewer agrees, accepted premises only, "
                    "not convention-dependent, no human convention choice needed"
                ),
                reviewer_says_no_human_convention_choice_needed=True,
            )
        ]
    )

    result = runner.run_calculation(minimal_config(tmp_path), batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"


def test_calculate_runner_dry_run_does_not_call_batch_runner(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner([])

    result = runner.run_calculation(
        minimal_config(tmp_path),
        batch_runner=fake,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert fake.calls == []


def test_calculate_templates_are_external_to_workflow_doc() -> None:
    calculate = (WF / "calculate.md").read_text(encoding="utf-8")
    proposer = json.loads((WJ / "calculate-proposer.template.json").read_text(encoding="utf-8"))
    reviewer = json.loads((WJ / "calculate-reviewer.template.json").read_text(encoding="utf-8"))

    assert "arc.llm.proposers_reviewer_batch.config.v1" not in calculate
    assert "calculate-proposer.template.json" in calculate
    assert "calculate-reviewer.template.json" in calculate
    assert "work_note_assessment" in proposer["prompt"]["template"]
    assert "agreement_assessment" in reviewer["prompt"]["template"]
    assert "arc_llm_call_record.structured_output" in reviewer["prompt"]["template"]
    assert "major" in reviewer["prompt"]["template"]
    assert "Do not mark consensus as accepted/all_agree based solely on major/fatal recovered proposer output." in reviewer[
        "prompt"
    ]["template"]


def test_calculate_template_sets_high_reasoning_defaults() -> None:
    data = json.loads((WJ / "calculate.config.template.json").read_text(encoding="utf-8"))
    defaults = data["defaults"]
    runtime = defaults["runtime"]

    assert defaults["model_tier"] == "high"
    assert runtime["codex_reasoning_effort"] == "high"
    assert runtime["codex_model_verbosity"] == "medium"
    assert runtime["claude_effort"] == "high"


def test_attempt_batch_config_carries_reasoning_defaults(tmp_path) -> None:
    runner = load_calculate_runner()
    template = json.loads((WJ / "calculate.config.template.json").read_text(encoding="utf-8"))
    config = runner.load_calculation_config(minimal_config(tmp_path, defaults=template["defaults"]))
    step = config.steps[0]

    batch = runner._attempt_batch_config(  # noqa: SLF001
        config,
        step,
        attempt_number=1,
        active_proposer_ids=["proposer_001", "proposer_002"],
        locked_outputs={},
        retry_feedback=[],
        run_root=tmp_path / "execute" / "calc_001",
        accepted_step_outputs={},
    )

    assert batch["defaults"] == template["defaults"]


def test_attempt_batch_config_uses_peer_visible_no_retry(tmp_path) -> None:
    runner = load_calculate_runner()
    config = runner.load_calculation_config(minimal_config(tmp_path))

    batch = runner._attempt_batch_config(  # noqa: SLF001
        config,
        config.steps[0],
        attempt_number=1,
        active_proposer_ids=["proposer_001", "proposer_002"],
        locked_outputs={},
        retry_feedback=[],
        run_root=tmp_path / "execute" / "calc_001",
        accepted_step_outputs={},
    )

    assert batch["output_recovery"]["schema_violation_policy"] == "peer_visible"
    assert batch["output_recovery"]["reviewer_validation_retries"] == 0


def test_calculate_parses_string_false_as_false(tmp_path) -> None:
    runner = load_calculate_runner()
    payload = minimal_config(
        tmp_path,
        human_gate={"enabled": "false"},
        artifact_options={"save_prompts": "false"},
    )

    config = runner.load_calculation_config(payload)

    assert config.human_gate["enabled"] is False
    assert config.artifact_options["save_prompts"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"human_gate": {"enabled": "maybe"}},
        {"artifact_options": {"save_prompts": "maybe"}},
    ],
)
def test_calculate_rejects_invalid_config_bool_strings(tmp_path, override: dict[str, Any]) -> None:
    runner = load_calculate_runner()

    with pytest.raises(runner.ConfigError, match="must be a boolean"):
        runner.load_calculation_config(minimal_config(tmp_path, **override))


def test_proposer_source_policy_parses_string_false() -> None:
    runner = load_calculate_runner()

    policy = runner._proposer_source_policy({"allow_mcp": "false", "allow_internet": "false"})  # noqa: SLF001

    assert "Do not use internet search" in policy
    assert "Do not use ARC paper MCP tools" in policy


def test_new_calculation_worker_defaults_to_no_mcp(tmp_path: Path) -> None:
    runner = load_calculate_runner()
    config = runner.load_calculation_config(minimal_config(tmp_path))

    runtime = runner._proposer_runtime(config, config.steps[0])  # noqa: SLF001

    assert runtime["allow_internet"] is True
    assert runtime["allow_mcp"] is False
    assert "mcp_mode" not in runtime


def test_calculate_worker_schemas_are_codex_strict(tmp_path) -> None:
    runner = load_calculate_runner()
    config = runner.load_calculation_config(minimal_config(tmp_path))
    step = config.steps[0]
    proposer_schema = runner._proposer_config(  # noqa: SLF001
        config,
        "proposer_001",
        runtime={"allow_internet": False, "allow_mcp": False},
    )["output_schema"]
    reviewer_schema = runner._reviewer_config(  # noqa: SLF001
        config,
        ["proposer_001", "proposer_002"],
        ["proposer_001", "proposer_002"],
        reviewer_reference_claim=step.reviewer_reference_claim,
        human_gate=config.human_gate,
    )["output_schema"]

    assert_codex_strict_objects(proposer_schema)
    assert_codex_strict_objects(reviewer_schema)
    accepted_result_schema = reviewer_schema["properties"]["review_payload"]["properties"]["consensus"]["properties"][
        "accepted_result"
    ]
    assert "object" in accepted_result_schema["type"]
    consensus_schema = reviewer_schema["properties"]["review_payload"]["properties"]["consensus"]
    assert "source_discrepancies" in consensus_schema["properties"]
    assert "source_discrepancy" not in consensus_schema["properties"]


def assert_codex_strict_objects(schema: Any) -> None:
    if isinstance(schema, dict):
        if "const" in schema:
            assert "type" in schema
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
            if "properties" in schema:
                assert set(schema.get("required", [])) == set(schema["properties"])
        if "object" in schema.get("type", []):
            assert schema.get("additionalProperties") is False
            if "properties" in schema:
                assert set(schema.get("required", [])) == set(schema["properties"])
        for value in schema.values():
            assert_codex_strict_objects(value)
    elif isinstance(schema, list):
        for item in schema:
            assert_codex_strict_objects(item)


def test_allowed_context_preserves_inert_source_provenance() -> None:
    runner = load_calculate_runner()
    context = runner._sanitize_caller_allowed_context(  # noqa: SLF001
        {
            "sources": [{"paper_id": "arXiv:0911.3380", "source_path": "cache/source.json"}],
            "cache_path": "cache/paper.json",
            "source_path": "source.tex",
            "source_commands": ["curl example"],
            "shell_commands": ["python script.py"],
            "nested": {"cli_invocations": ["arc-paper get"], "section": "2"},
        }
    )

    assert context["sources"][0]["source_path"] == "cache/source.json"
    assert context["cache_path"] == "cache/paper.json"
    assert context["source_path"] == "source.tex"
    assert "source_commands" not in context
    assert "shell_commands" not in context
    assert "cli_invocations" not in context["nested"]


def test_human_gate_respects_nonhuman_continue_action(tmp_path: Path) -> None:
    runner = load_calculate_runner()
    config = minimal_config(
        tmp_path,
        human_gate={"enabled": True, "pause_statuses": ["reference_disagrees"]},
        max_recalculations=0,
    )
    config["steps"][0]["reviewer_reference_claim"] = {"id": "target", "latex": "x"}
    fake = FakeBatchRunner(
        [
            calculate_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                target_quantity_match=False,
                accepted_by_reviewer_judgment=False,
                action="continue",
                requires_human=False,
            )
        ]
    )

    result = runner.run_calculation(config, batch_runner=fake, base_env={})

    assert result["status"] == "blocked_for_revision"
    assert result["steps"][0]["blocked_output"]["requires_human"] is False


class FakeBatchRunner:
    def __init__(self, reviews: list[dict[str, Any]], warnings_summary: dict[str, Any] | None = None) -> None:
        self.reviews = list(reviews)
        self.calls: list[dict[str, Any]] = []
        self.active_proposers_by_call: list[list[str]] = []
        self.warnings_summary = warnings_summary or {
            "structured_output_warning_count": 0,
            "structured_output_warnings_path": "",
            "cache_warning_count": 0,
            "cache_warnings_path": "",
        }

    def __call__(self, config: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        payload = json.loads(json.dumps(config))
        self.calls.append(payload)
        loop = payload["loops"][0]
        proposer_ids = [item["id"] for item in loop["proposers"]]
        self.active_proposers_by_call.append(proposer_ids)
        paths = attempt_paths(payload)
        paths["review_path"].parent.mkdir(parents=True, exist_ok=True)
        proposer_root = paths["round_root"] / "proposer_outputs"
        proposer_root.mkdir(parents=True, exist_ok=True)
        for proposer_id in proposer_ids:
            (proposer_root / f"{proposer_id}.json").write_text(
                json.dumps({"proposer_id": proposer_id, "final_result": proposer_id}),
                encoding="utf-8",
            )
        review = self.reviews.pop(0)
        paths["review_path"].write_text(json.dumps(review), encoding="utf-8")
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": payload["run_id"],
            "run_root": str(paths["run_root"]),
            "warnings_summary": self.warnings_summary,
            "loops": [{"loop_id": loop["loop_id"], "status": "completed"}],
        }


def attempt_paths(batch_config: Mapping[str, Any]) -> dict[str, Path]:
    run_root = Path(str(batch_config["run_dir"])) / str(batch_config["run_id"])
    loop_id = str(batch_config["loops"][0]["loop_id"])
    round_root = run_root / "loops" / loop_id / "rounds" / "round_001"
    return {
        "run_root": run_root,
        "round_root": round_root,
        "review_path": round_root / "reviews" / "reviewer_001.json",
    }


def calculate_review(
    status: str,
    *,
    agreed: list[str] | None = None,
    likely_wrong: list[str] | None = None,
    recalculate: list[str] | None = None,
    best_written: str | None = None,
    special_limit_only: bool = False,
    target_quantity_match: bool = True,
    convention_match: bool = True,
    declared_scope_match: bool = True,
    agreement_covers_full_target: bool = True,
    accepted_by_reviewer_judgment: bool | None = None,
    action: str | None = None,
    requires_human: bool | None = None,
    proposer_messages: dict[str, str] | None = None,
    source_discrepancy_status: str = "none",
    source_discrepancy_confidence_reason: str = "no source discrepancy",
    reviewer_says_no_human_convention_choice_needed: bool = False,
    source_discrepancies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    workflow_action = action or ("continue" if status == "all_agree" else "retry")
    consensus = {
        "status": status,
        "accepted_result": {"result": "x"} if status == "all_agree" else None,
        "agreed_proposer_ids": agreed or [],
        "likely_wrong_proposer_ids": likely_wrong or [],
        "recalculate_proposer_ids": recalculate or [],
        "validity_scope": "declared scope",
        "analysis": "review analysis",
        "best_written_proposer_id": best_written
        if best_written is not None
        else ((agreed or [None])[0] if status in {"all_agree", "reference_disagrees"} else None),
        "best_written_selection_reason": "clearest derivation"
        if status in {"all_agree", "reference_disagrees"}
        else "",
        "agreement_assessment": {
            "target_quantity_match": target_quantity_match,
            "convention_match": convention_match,
            "declared_scope_match": declared_scope_match,
            "agreement_covers_full_target": agreement_covers_full_target,
            "comparison_summary": "explicit algebraic comparison",
            "accepted_by_reviewer_judgment": bool(
                status == "all_agree" if accepted_by_reviewer_judgment is None else accepted_by_reviewer_judgment
            ),
            "special_limit_only": special_limit_only,
        },
        "workflow_action": {
            "action": workflow_action,
            "requires_human": bool(requires_human) if requires_human is not None else False,
            "issue_type": "none" if status == "all_agree" else "calculation_disagreement",
            "reason": "test",
        },
        "source_discrepancies": source_discrepancies
        if source_discrepancies is not None
        else (
            []
            if source_discrepancy_status == "none"
            else [
                {
                    "item_id": "source_discrepancy",
                    "status": source_discrepancy_status,
                    "source_claim": "source claim",
                    "derived_result": "derived result",
                    "confidence_reason": source_discrepancy_confidence_reason,
                    "reviewer_says_no_human_convention_choice_needed": (
                        reviewer_says_no_human_convention_choice_needed
                    ),
                    "decision_question": "How should ARC resolve this source discrepancy?",
                }
            ]
        ),
    }
    messages = {
        proposer_id: {"message": message}
        for proposer_id, message in (proposer_messages or {}).items()
    }
    return {
        "schema_version": "arc.llm.review_envelope.v1",
        "controller": {"message": "done", "stop_requested": False},
        "proposer_messages": {
            proposer_id: messages.get(proposer_id, {"message": ""})
            for proposer_id in ["proposer_001", "proposer_002", "proposer_003"]
        },
        "review_payload": {"consensus": consensus},
    }
