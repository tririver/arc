from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch


def base_config(tmp_path: Path, *, max_rounds: int = 2, early_stop: bool = False) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "run_001",
        "run_dir": str(tmp_path / "suggest-ideas"),
        "max_concurrent_loops": 2,
        "defaults": {"provider": "manual", "model": "fake-model"},
        "loops": [
            {
                "loop_id": "loop_001",
                "max_rounds": max_rounds,
                "early_stop": {"enabled": early_stop},
                "proposers": [
                    {
                        "id": "proposer_001",
                        "prompt": {
                            "system": "proposer system",
                            "template": "propose {worker_id} round {round_number}\n{correspondence_json}",
                        },
                        "output_schema": {"type": "object"},
                        "runtime": {"allow_internet": True},
                    },
                    {
                        "id": "proposer_002",
                        "prompt": {
                            "system": "proposer system",
                            "template": "propose {worker_id} round {round_number}\n{correspondence_json}",
                        },
                        "output_schema": {"type": "object"},
                    },
                ],
                "reviewers": [
                    {
                        "id": "reviewer_001",
                        "prompt": {
                            "system": "reviewer system",
                            "template": "review {worker_id} round {round_number}\n{current_proposer_outputs_json}\n{correspondence_json}",
                        },
                        "output_schema": {"type": "object"},
                        "runtime": {"allow_mcp": True},
                    }
                ],
                "caller_context": {"user_intent": "intent"},
            }
        ],
    }


class FakeJsonRunner:
    def __init__(self, *, stop_round: int | None = None, fail_loop: str | None = None) -> None:
        self.stop_round = stop_round
        self.fail_loop = fail_loop
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None,
        provider: str,
        model: str | None,
        env: dict[str, str],
    ) -> dict[str, Any]:
        context = _context_from_prompt(prompt)
        self.calls.append(
            {
                "worker_id": context["worker_id"],
                "loop_id": context["loop_id"],
                "round_number": context["round_number"],
                "provider": provider,
                "model": model,
                "env": dict(env),
            }
        )
        if self.fail_loop and context["loop_id"] == self.fail_loop:
            raise RuntimeError("simulated provider failure")
        worker_id = context["worker_id"]
        round_number = context["round_number"]
        loop_id = context["loop_id"]
        if worker_id.startswith("reviewer"):
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {
                    "message": f"controller-{loop_id}-round-{round_number}",
                    "stop_requested": self.stop_round == round_number,
                    "stop_reason": "agreement" if self.stop_round == round_number else "",
                },
                "proposer_messages": {
                    "proposer_001": {"message": f"review-to-proposer_001-round-{round_number}"},
                    "proposer_002": {"message": f"review-to-proposer_002-round-{round_number}"},
                },
                "review_payload": {"round": round_number, "loop_id": loop_id},
            }
        return {
            "worker_id": worker_id,
            "round": round_number,
            "loop_id": loop_id,
            "content": f"output-from-{worker_id}-round-{round_number}",
        }


class AddOneReviewRunner:
    def __call__(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None,
        provider: str,
        model: str | None,
        env: dict[str, str],
    ) -> dict[str, Any]:
        context = _context_from_prompt(prompt)
        worker_id = context["worker_id"]
        round_number = context["round_number"]
        if worker_id.startswith("reviewer"):
            values = {
                proposer_id: output["value"]
                for proposer_id, output in context["current_proposer_outputs"].items()
            }
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {
                    "message": "Both proposers must add 1 next round.",
                    "stop_requested": False,
                    "stop_reason": "",
                },
                "proposer_messages": {
                    proposer_id: {"message": "Add 1 to your current number next round."}
                    for proposer_id in values
                },
                "review_payload": {
                    "round": round_number,
                    "current_values": values,
                    "request": "add 1",
                },
            }

        previous = _latest_proposer_value(context["correspondence"], worker_id)
        review_message = _latest_proposer_message(context["correspondence"], worker_id)
        if previous is None:
            value = 1
            followed_review = False
        elif review_message == "Add 1 to your current number next round.":
            value = previous + 1
            followed_review = True
        else:
            value = previous
            followed_review = False
        return {
            "worker_id": worker_id,
            "round": round_number,
            "value": value,
            "followed_review": followed_review,
        }


def test_runner_records_full_past_correspondence_without_current_round_cross_talk(tmp_path):
    fake = FakeJsonRunner()

    result = run_proposers_reviewer_batch(base_config(tmp_path), json_runner=fake, base_env={})

    assert result["status"] == "completed"
    run_root = tmp_path / "suggest-ideas" / "run_001"
    round1_p1_prompt = (run_root / "loops/loop_001/rounds/round_001/prompts/proposer_001.md").read_text(
        encoding="utf-8"
    )
    round2_p1_prompt = (run_root / "loops/loop_001/rounds/round_002/prompts/proposer_001.md").read_text(
        encoding="utf-8"
    )
    round1_review_prompt = (run_root / "loops/loop_001/rounds/round_001/prompts/reviewer_001.md").read_text(
        encoding="utf-8"
    )

    assert "output-from-proposer_002-round-1" not in round1_p1_prompt
    assert "output-from-proposer_001-round-1" in round2_p1_prompt
    assert "output-from-proposer_002-round-1" in round2_p1_prompt
    assert "propose proposer_001 round 1" in round2_p1_prompt
    assert "review-to-proposer_001-round-1" in round2_p1_prompt
    assert "output-from-proposer_001-round-1" in round1_review_prompt
    assert "output-from-proposer_002-round-1" in round1_review_prompt


def test_two_proposers_follow_add_one_reviewer_request_for_three_rounds(tmp_path):
    result = run_proposers_reviewer_batch(
        base_config(tmp_path, max_rounds=3),
        json_runner=AddOneReviewRunner(),
        base_env={},
    )

    run_root = tmp_path / "suggest-ideas" / "run_001"
    values_by_round = {}
    for round_number in (1, 2, 3):
        round_root = run_root / "loops" / "loop_001" / "rounds" / f"round_{round_number:03d}"
        values_by_round[round_number] = {
            "proposer_001": json.loads((round_root / "proposer_outputs" / "proposer_001.json").read_text())["value"],
            "proposer_002": json.loads((round_root / "proposer_outputs" / "proposer_002.json").read_text())["value"],
        }
        review = json.loads((round_root / "reviews" / "reviewer_001.json").read_text())
        assert review["review_payload"]["request"] == "add 1"

    assert result["status"] == "completed"
    assert result["loops"][0]["rounds_completed"] == 3
    assert values_by_round == {
        1: {"proposer_001": 1, "proposer_002": 1},
        2: {"proposer_001": 2, "proposer_002": 2},
        3: {"proposer_001": 3, "proposer_002": 3},
    }


def test_two_loops_run_in_isolated_directories(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    second = json.loads(json.dumps(config["loops"][0]))
    second["loop_id"] = "loop_002"
    config["loops"].append(second)

    result = run_proposers_reviewer_batch(config, json_runner=FakeJsonRunner(), base_env={})

    assert result["status"] == "completed"
    run_root = tmp_path / "suggest-ideas" / "run_001"
    assert (run_root / "loops" / "loop_001" / "state.json").exists()
    assert (run_root / "loops" / "loop_002" / "state.json").exists()
    assert sorted(item["loop_id"] for item in result["loops"]) == ["loop_001", "loop_002"]


def test_max_rounds_controls_round_count(tmp_path):
    result = run_proposers_reviewer_batch(base_config(tmp_path, max_rounds=3), json_runner=FakeJsonRunner(), base_env={})

    loop = result["loops"][0]

    assert loop["status"] == "completed"
    assert loop["rounds_completed"] == 3


def test_early_stop_is_honored_when_enabled(tmp_path):
    result = run_proposers_reviewer_batch(
        base_config(tmp_path, max_rounds=5, early_stop=True),
        json_runner=FakeJsonRunner(stop_round=2),
        base_env={},
    )

    loop = result["loops"][0]

    assert loop["status"] == "stopped"
    assert loop["rounds_completed"] == 2
    assert loop["stop_reason"] == "agreement"


def test_early_stop_request_is_recorded_but_ignored_when_disabled(tmp_path):
    result = run_proposers_reviewer_batch(
        base_config(tmp_path, max_rounds=3, early_stop=False),
        json_runner=FakeJsonRunner(stop_round=1),
        base_env={},
    )

    loop = result["loops"][0]
    review = json.loads(
        (tmp_path / "suggest-ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json").read_text(
            encoding="utf-8"
        )
    )

    assert loop["status"] == "completed"
    assert loop["rounds_completed"] == 3
    assert review["controller"]["stop_requested"] is True


def test_failed_loop_does_not_corrupt_successful_loop(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    second = json.loads(json.dumps(config["loops"][0]))
    second["loop_id"] = "loop_bad"
    config["loops"].append(second)

    result = run_proposers_reviewer_batch(config, json_runner=FakeJsonRunner(fail_loop="loop_bad"), base_env={})

    statuses = {item["loop_id"]: item["status"] for item in result["loops"]}
    assert result["status"] == "failed"
    assert statuses["loop_001"] == "completed"
    assert statuses["loop_bad"] == "failed"
    assert (tmp_path / "suggest-ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json").exists()


def test_review_envelope_requires_review_payload(tmp_path):
    def invalid_reviewer(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "missing payload", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
            }
        return {
            "worker_id": context["worker_id"],
            "round": context["round_number"],
            "content": "proposal",
        }

    result = run_proposers_reviewer_batch(base_config(tmp_path, max_rounds=1), json_runner=invalid_reviewer, base_env={})

    assert result["status"] == "failed"
    assert result["loops"][0]["status"] == "failed"
    assert "review.review_payload must be an object" in result["loops"][0]["error"]


def test_worker_envs_are_isolated_and_os_environ_is_not_mutated(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_CODEX_ALLOW_INTERNET", raising=False)
    monkeypatch.delenv("ARC_CODEX_ENABLE_MCP", raising=False)
    fake = FakeJsonRunner()

    run_proposers_reviewer_batch(base_config(tmp_path, max_rounds=1), json_runner=fake, base_env={})

    proposer_call = next(call for call in fake.calls if call["worker_id"] == "proposer_001")
    reviewer_call = next(call for call in fake.calls if call["worker_id"] == "reviewer_001")
    assert proposer_call["env"]["ARC_CODEX_ALLOW_INTERNET"] == "true"
    assert "ARC_CODEX_ENABLE_MCP" not in proposer_call["env"]
    assert reviewer_call["env"]["ARC_CODEX_ENABLE_MCP"] == "true"
    assert "ARC_CODEX_ALLOW_INTERNET" not in os.environ
    assert "ARC_CODEX_ENABLE_MCP" not in os.environ


def _context_from_prompt(prompt: str) -> dict[str, Any]:
    match = re.search(r"^## ARC Worker Context\n(?P<context>\{.*?\})\n## Prompt", prompt, re.S | re.M)
    assert match, prompt
    return json.loads(match.group("context"))


def _latest_proposer_value(correspondence: list[dict[str, Any]], worker_id: str) -> int | None:
    for event in reversed(correspondence):
        if event.get("type") == "proposer_output" and event.get("worker_id") == worker_id:
            return int(event["output"]["value"])
    return None


def _latest_proposer_message(correspondence: list[dict[str, Any]], worker_id: str) -> str:
    for event in reversed(correspondence):
        if event.get("type") == "proposer_message" and event.get("worker_id") == worker_id:
            return str(event.get("message", {}).get("message", ""))
    return ""
