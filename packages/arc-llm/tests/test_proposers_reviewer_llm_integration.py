from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch


pytestmark = pytest.mark.skipif(
    os.environ.get("ARC_RUN_LLM_TESTS") != "1" or os.environ.get("ARC_RUN_NET_TESTS") != "1",
    reason="true LLM integration test; set ARC_RUN_LLM_TESTS=1 and ARC_RUN_NET_TESTS=1 to run",
)


def test_true_llm_two_proposers_follow_add_one_review_for_three_rounds(tmp_path):
    config = _true_llm_add_one_config(tmp_path)

    result = run_proposers_reviewer_batch(config)

    values_by_round = _read_values_by_round(tmp_path / "llm-runs" / "true_llm_add_one")
    assert result["status"] == "completed"
    assert result["loops"][0]["rounds_completed"] == 3
    assert values_by_round == {
        1: {"proposer_001": 1, "proposer_002": 1},
        2: {"proposer_001": 2, "proposer_002": 2},
        3: {"proposer_001": 3, "proposer_002": 3},
    }


def _true_llm_add_one_config(tmp_path: Path) -> dict[str, Any]:
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
        "run_id": "true_llm_add_one",
        "run_dir": str(tmp_path / "llm-runs"),
        "max_concurrent_loops": 1,
        "defaults": defaults,
        "loops": [
            {
                "loop_id": "loop_001",
                "max_rounds": 3,
                "early_stop": {"enabled": False},
                "caller_context": {
                    "task": (
                        "This is a deterministic integration test. Each proposer starts at 1. "
                        "After each reviewer request to add 1, that proposer must add exactly 1."
                    )
                },
                "proposers": [
                    _proposer_config("proposer_001"),
                    _proposer_config("proposer_002"),
                ],
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
                "Rule: in round 1, set value to 1. In later rounds, inspect the ARC Worker Context "
                "correspondence for the latest proposer_message addressed to your worker_id. If it says "
                "'Add 1 to your current number next round.', find your latest prior proposer_output value "
                "and set value to that value plus 1. Do not add anything else."
            ),
        },
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "worker_id": {"type": "string"},
                "round": {"type": "integer"},
                "value": {"type": "integer"},
                "followed_instruction": {"type": "boolean"},
            },
            "required": ["worker_id", "round", "value", "followed_instruction"],
        },
    }


def _reviewer_config() -> dict[str, Any]:
    return {
        "id": "reviewer_001",
        "prompt": {
            "system": "You are a deterministic arithmetic reviewer in a test. Follow instructions exactly.",
            "template": (
                "Return JSON only using the required review envelope. Inspect current_proposer_outputs. "
                "For every proposer id present, send exactly this proposer message: "
                "'Add 1 to your current number next round.' Never request early stop. Put current values "
                "in review_payload.current_values and set review_payload.request to 'add 1'."
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
                        "proposer_001": _review_message_schema(),
                        "proposer_002": _review_message_schema(),
                    },
                    "required": ["proposer_001", "proposer_002"],
                },
                "review_payload": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "request": {"type": "string", "const": "add 1"},
                        "current_values": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "proposer_001": {"type": "integer"},
                                "proposer_002": {"type": "integer"},
                            },
                            "required": ["proposer_001", "proposer_002"],
                        },
                    },
                    "required": ["request", "current_values"],
                },
            },
            "required": ["schema_version", "controller", "proposer_messages", "review_payload"],
        },
    }


def _review_message_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }


def _read_values_by_round(run_root: Path) -> dict[int, dict[str, int]]:
    values_by_round: dict[int, dict[str, int]] = {}
    for round_number in (1, 2, 3):
        round_root = run_root / "loops" / "loop_001" / "rounds" / f"round_{round_number:03d}"
        values_by_round[round_number] = {
            proposer_id: json.loads((round_root / "proposer_outputs" / f"{proposer_id}.json").read_text())["value"]
            for proposer_id in ("proposer_001", "proposer_002")
        }
    return values_by_round
