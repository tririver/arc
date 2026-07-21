from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch


PROPOSER_IDS = ("proposer_001", "proposer_002", "proposer_003")
START_VALUES = {"proposer_001": 1, "proposer_002": 2, "proposer_003": 3}
INCREMENTS = {"proposer_001": 1, "proposer_002": 2, "proposer_003": 3}
REVIEW_MESSAGES = {
    proposer_id: f"Add {increment} to your current number next round."
    for proposer_id, increment in INCREMENTS.items()
}
EXPECTED_VALUES_BY_ROUND = {
    1: {"proposer_001": 1, "proposer_002": 2, "proposer_003": 3},
    2: {"proposer_001": 2, "proposer_002": 4, "proposer_003": 6},
    3: {"proposer_001": 3, "proposer_002": 6, "proposer_003": 9},
}


pytestmark = pytest.mark.skipif(
    os.environ.get("ARC_RUN_LLM_TESTS") != "1" or os.environ.get("ARC_RUN_NET_TESTS") != "1",
    reason="true LLM integration test; set ARC_RUN_LLM_TESTS=1 and ARC_RUN_NET_TESTS=1 to run",
)


def test_true_llm_three_proposers_follow_targeted_reviews_for_three_rounds(tmp_path):
    config = _true_llm_targeted_addition_config(tmp_path)

    result = run_proposers_reviewer_batch(config)

    outputs_by_round = _read_outputs_by_round(tmp_path / "llm-runs" / "true_llm_targeted_additions")
    values_by_round = {
        round_number: {proposer_id: output["value"] for proposer_id, output in outputs.items()}
        for round_number, outputs in outputs_by_round.items()
    }
    received_messages_by_round = {
        round_number: {proposer_id: output["received_reviewer_message"] for proposer_id, output in outputs.items()}
        for round_number, outputs in outputs_by_round.items()
    }
    assert result["status"] == "completed"
    assert result["loops"][0]["rounds_completed"] == 3
    assert values_by_round == EXPECTED_VALUES_BY_ROUND
    assert received_messages_by_round == {
        1: {"proposer_001": "none", "proposer_002": "none", "proposer_003": "none"},
        2: REVIEW_MESSAGES,
        3: REVIEW_MESSAGES,
    }


def _true_llm_targeted_addition_config(tmp_path: Path) -> dict[str, Any]:
    provider = os.environ.get("ARC_LLM_TEST_PROVIDER", "codex-cli")
    model = os.environ.get("ARC_LLM_TEST_MODEL")
    defaults: dict[str, Any] = {
        "provider": provider,
        "runtime": {
            "allow_internet": False,
            "allow_mcp": False,
            "codex_reasoning_effort": os.environ.get("ARC_LLM_TEST_CODEX_REASONING_EFFORT", "low"),
            "codex_model_verbosity": "low",
            "claude_effort": os.environ.get("ARC_LLM_TEST_CLAUDE_EFFORT", "low"),
        },
    }
    if model:
        defaults["model"] = model

    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "true_llm_targeted_additions",
        "run_dir": str(tmp_path / "llm-runs"),
        "max_concurrent_loops": 1,
        "worker_idle_timeout_seconds": float(
            os.environ.get("ARC_LLM_TEST_IDLE_TIMEOUT_SECONDS", "1800")
        ),
        "defaults": defaults,
        "loops": [
            {
                "loop_id": "loop_001",
                "max_rounds": 3,
                "early_stop": {"enabled": False},
                "caller_context": {
                    "task": (
                        "This is a deterministic integration test for reviewer-to-proposer routing. "
                        "The proposers start at different values and must use only the latest "
                        "reviewer message addressed to their own worker_id."
                    )
                },
                "proposers": [_proposer_config(proposer_id) for proposer_id in PROPOSER_IDS],
                "reviewers": [_reviewer_config()],
            }
        ],
    }


def _proposer_config(worker_id: str) -> dict[str, Any]:
    return {
        "id": worker_id,
        "prompt": {
            "system": "You are a deterministic arithmetic worker in a test. Follow instructions exactly.",
            "template": (
                "Return JSON only. Your worker_id is {worker_id}. The current round is {round_number}.\n"
                "Round 1 rule: set value to your assigned starting value: proposer_001 starts at 1, "
                "proposer_002 starts at 2, and proposer_003 starts at 3. In round 1 return "
                "previous_value=0, applied_increment=0, received_reviewer_message='none', and "
                "followed_instruction=true.\n"
                "Later-round rule: inspect the ARC Worker Context correspondence. Find the latest "
                "proposer_message whose worker_id exactly equals your worker_id. Ignore messages for "
                "other worker_ids. Copy that message string exactly into received_reviewer_message. It "
                "will be one of: 'Add 1 to your current number next round.', 'Add 2 to your current "
                "number next round.', or 'Add 3 to your current number next round.'. Extract the "
                "number N from your own message, find your latest prior proposer_output value, set "
                "previous_value to that prior value, set applied_increment to N, and set value to "
                "previous_value + N. Do not use round_number to compute value."
            ),
        },
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "worker_id": {"type": "string"},
                "round": {"type": "integer"},
                "value": {"type": "integer"},
                "previous_value": {"type": "integer"},
                "applied_increment": {"type": "integer"},
                "received_reviewer_message": {"type": "string"},
                "followed_instruction": {"type": "boolean"},
            },
            "required": [
                "worker_id",
                "round",
                "value",
                "previous_value",
                "applied_increment",
                "received_reviewer_message",
                "followed_instruction",
            ],
        },
    }


def _reviewer_config() -> dict[str, Any]:
    return {
        "id": "reviewer_001",
        "prompt": {
            "system": "You are a deterministic arithmetic reviewer in a test. Follow instructions exactly.",
            "template": (
                "Return JSON only using the required review envelope. Inspect current_proposer_outputs. "
                "For each proposer id, send exactly the message assigned to that proposer: proposer_001 "
                "gets 'Add 1 to your current number next round.', proposer_002 gets 'Add 2 to your "
                "current number next round.', and proposer_003 gets 'Add 3 to your current number next "
                "round.'. Never request early stop. Put current values in review_payload.current_values "
                "and the exact message map in review_payload.requests."
            ),
        },
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "string", "const": "arc.llm.review_envelope.v1"},
                "controller": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "message": {"type": "string"},
                        "stop_requested": {"type": "boolean"},
                        "stop_reason": {"type": "string"},
                    },
                    "required": ["message", "stop_requested", "stop_reason"],
                },
                "proposer_messages": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "proposer_001": _review_message_schema(REVIEW_MESSAGES["proposer_001"]),
                        "proposer_002": _review_message_schema(REVIEW_MESSAGES["proposer_002"]),
                        "proposer_003": _review_message_schema(REVIEW_MESSAGES["proposer_003"]),
                    },
                    "required": list(PROPOSER_IDS),
                },
                "review_payload": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "requests": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                proposer_id: {"type": "string", "const": message}
                                for proposer_id, message in REVIEW_MESSAGES.items()
                            },
                            "required": list(PROPOSER_IDS),
                        },
                        "current_values": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                proposer_id: {"type": "integer"} for proposer_id in PROPOSER_IDS
                            },
                            "required": list(PROPOSER_IDS),
                        },
                    },
                    "required": ["requests", "current_values"],
                },
            },
            "required": ["schema_version", "controller", "proposer_messages", "review_payload"],
        },
    }


def _review_message_schema(message: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"message": {"type": "string", "const": message}},
        "required": ["message"],
    }


def _read_outputs_by_round(run_root: Path) -> dict[int, dict[str, dict[str, Any]]]:
    outputs_by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for round_number in (1, 2, 3):
        round_root = run_root / "loops" / "loop_001" / "rounds" / f"round_{round_number:03d}"
        outputs_by_round[round_number] = {
            proposer_id: json.loads((round_root / "proposer_outputs" / f"{proposer_id}.json").read_text())
            for proposer_id in PROPOSER_IDS
        }
    return outputs_by_round
