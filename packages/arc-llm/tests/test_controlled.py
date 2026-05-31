from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from arc_llm.proposers_reviewer.controlled import JsonPolicyReviewController, run_controlled_proposers_reviewer


def test_controlled_runner_uses_shared_session_and_passes_attempt_artifacts(tmp_path):
    seen_batches: list[dict[str, Any]] = []
    seen_controller_inputs: list[dict[str, Any]] = []

    class Controller:
        def initial_state(self) -> dict[str, Any]:
            return {}

        def build_attempt_context(self, *, attempt_number: int, state: Mapping[str, Any]) -> dict[str, Any]:
            return {"attempt_number": attempt_number}

        def select_active_proposers(self, *, attempt_number: int, state: Mapping[str, Any]) -> list[str]:
            return ["proposer_001"]

        def on_attempt_result(
            self,
            *,
            attempt_number: int,
            batch_result: Mapping[str, Any],
            proposer_outputs: Mapping[str, Any],
            review: Mapping[str, Any],
            state: Mapping[str, Any],
        ) -> dict[str, Any]:
            seen_controller_inputs.append(
                {
                    "proposer_outputs": dict(proposer_outputs),
                    "review": dict(review),
                }
            )
            return {"action": "accept", "accepted_output": review["review_payload"]["accepted"]}

    def fake_batch_runner(batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        seen_batches.append(batch)
        run_root = Path(batch["run_dir"]) / batch["run_id"]
        loop_id = batch["loops"][0]["loop_id"]
        round_root = run_root / "loops" / loop_id / "rounds" / "round_001"
        proposer_dir = round_root / "proposer_outputs"
        review_dir = round_root / "reviews"
        proposer_dir.mkdir(parents=True)
        review_dir.mkdir(parents=True)
        (proposer_dir / "proposer_001.json").write_text(json.dumps({"value": 7}) + "\n", encoding="utf-8")
        (review_dir / "reviewer_001.json").write_text(
            json.dumps({"review_payload": {"accepted": {"value": 7}}}) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "run_root": str(run_root),
            "loops": [{"loop_id": loop_id, "status": "completed", "rounds_completed": 1}],
        }

    result = run_controlled_proposers_reviewer(
        run_id="calc_001",
        run_dir=tmp_path / "calc",
        controller=Controller(),
        loop_template={"max_rounds": 1, "early_stop": {"enabled": False}},
        proposer_templates={"proposer_001": {"id": "proposer_001", "prompt": {"system": "s", "template": "t"}}},
        reviewer_template={"id": "reviewer_001", "prompt": {"system": "s", "template": "t"}},
        max_attempts=1,
        session_scope_id="calculate/run/step",
        batch_runner=fake_batch_runner,
    )

    assert result["status"] == "accept"
    assert seen_batches[0]["session"]["reuse_across_batch_calls"] is True
    assert seen_batches[0]["session"]["scope_id"] == "calculate/run/step"
    assert seen_batches[0]["session"]["root"] == str(tmp_path / "calc/llm_sessions")
    assert seen_controller_inputs == [
        {
            "proposer_outputs": {"proposer_001": {"value": 7}},
            "review": {"review_payload": {"accepted": {"value": 7}}},
        }
    ]


def test_json_policy_controller_retries_then_accepts_with_locked_outputs():
    controller = JsonPolicyReviewController(
        {
            "status_path": "review_payload.consensus.status",
            "accept_statuses": ["all_agree"],
            "retry_statuses": ["two_agree"],
            "accepted_output_path": "review_payload.consensus.accepted_result",
            "retry_feedback_paths": {"analysis": "review_payload.consensus.analysis"},
            "next_active_proposer_ids_path": "review_payload.consensus.recalculate_proposer_ids",
            "lock_outputs_when_status": {"two_agree": {"ids_path": "review_payload.consensus.agreed_proposer_ids"}},
        }
    )
    retry_review = {
        "review_payload": {
            "consensus": {
                "status": "two_agree",
                "analysis": "rerun one proposer",
                "recalculate_proposer_ids": ["proposer_003"],
                "agreed_proposer_ids": ["proposer_001"],
            }
        }
    }

    retry = controller.on_attempt_result(
        attempt_number=1,
        batch_result={"status": "completed"},
        proposer_outputs={"proposer_001": {"value": 1}, "proposer_003": {"value": 3}},
        review=retry_review,
        state={},
    )
    accept = controller.on_attempt_result(
        attempt_number=2,
        batch_result={"status": "completed"},
        proposer_outputs={},
        review={
            "review_payload": {
                "consensus": {"status": "all_agree", "accepted_result": {"value": 1}},
            }
        },
        state=retry["next_state"],
    )

    assert retry["action"] == "retry"
    assert retry["next_state"]["next_active_proposer_ids"] == ["proposer_003"]
    assert retry["next_state"]["locked_outputs"] == {"proposer_001": {"value": 1}}
    assert accept["action"] == "accept"
    assert accept["accepted_output"] == {"value": 1}
