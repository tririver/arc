from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "skills/arc/workflows"
WJ = WF / "json"
WS = WF / "scripts"


def load_calculate_runner():
    spec = importlib.util.spec_from_file_location("calculate_runner", WS / "calculate_runner.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["calculate_runner"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
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
    assert sorted(fake.calls[1]["loops"][0]["caller_context"]["locked_outputs"]) == [
        "proposer_001",
        "proposer_002",
    ]


def test_calculate_runner_blocks_on_reference_disagreement_without_failing_validation(tmp_path):
    runner = load_calculate_runner()
    fake = FakeBatchRunner(
        [
            calculate_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                best_written="proposer_001",
                special_limit_only=True,
            )
        ]
    )

    result = runner.run_calculation(
        minimal_config(
            tmp_path,
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
                convention_match=False,
            )
        ]
    )

    result = runner.run_calculation(
        minimal_config(
            tmp_path,
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


def assert_codex_strict_objects(schema: Any) -> None:
    if isinstance(schema, dict):
        if "const" in schema:
            assert "type" in schema
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
        if "object" in schema.get("type", []):
            assert schema.get("additionalProperties") is False
        for value in schema.values():
            assert_codex_strict_objects(value)
    elif isinstance(schema, list):
        for item in schema:
            assert_codex_strict_objects(item)


class FakeBatchRunner:
    def __init__(self, reviews: list[dict[str, Any]]) -> None:
        self.reviews = list(reviews)
        self.calls: list[dict[str, Any]] = []
        self.active_proposers_by_call: list[list[str]] = []

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
    convention_match: bool = True,
) -> dict[str, Any]:
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
            "target_quantity_match": True,
            "convention_match": convention_match,
            "declared_scope_match": True,
            "agreement_covers_full_target": True,
            "comparison_summary": "explicit algebraic comparison",
            "accepted_by_reviewer_judgment": True,
            "special_limit_only": special_limit_only,
        },
        "workflow_action": {
            "action": "continue" if status == "all_agree" else "retry",
            "requires_human": False,
            "issue_type": "none" if status == "all_agree" else "calculation_disagreement",
            "reason": "test",
        },
    }
    return {
        "schema_version": "arc.llm.review_envelope.v1",
        "controller": {"message": "done", "stop_requested": False},
        "proposer_messages": {
            proposer_id: {"message": ""}
            for proposer_id in ["proposer_001", "proposer_002", "proposer_003"]
        },
        "review_payload": {"consensus": consensus},
    }
