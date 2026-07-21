from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from arc_llm import runner as core_runner
from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD
from arc_llm.proposers_reviewer.config import load_batch_config
from arc_llm.proposers_reviewer import runner as runner_module
from arc_llm.proposers_reviewer.artifacts import RunPaths, acquire_lock, atomic_write_json
from arc_llm.proposers_reviewer.dialogue import (
    render_initial_worker_prompt,
    render_proposer_delta_prompt,
    render_reviewer_delta_prompt,
)
from arc_llm.proposers_reviewer.runner import PrefixConcurrencyLimiter, run_proposers_reviewer_batch
from arc_llm.proposers_reviewer.template_materializer import materialize_batch
from arc_llm.usage import LLMProviderResponse, LLMUsage


def base_config(tmp_path: Path, *, max_rounds: int = 2, early_stop: bool = False) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "run_001",
        "run_dir": str(tmp_path / "ideas"),
        "max_concurrent_loops": 2,
        "session": {"policy": "stateless", "history_mode": "full"},
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


def _fallback_output_recovery(legacy_reviewer_validation_retries: int | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "enabled": True,
        "mode": "warn",
        "allow_natural_language": True,
        "schema_violation_policy": "fallback",
    }
    if legacy_reviewer_validation_retries is not None:
        options["reviewer_validation_retries"] = legacy_reviewer_validation_retries
    return options


def test_batch_config_defaults_output_recovery_to_warn(tmp_path: Path):
    batch = load_batch_config(base_config(tmp_path, max_rounds=1))

    assert batch.output_recovery.enabled is True
    assert batch.output_recovery.mode == "warn"
    assert batch.output_recovery.allow_natural_language is True
    assert batch.output_recovery.schema_violation_policy == "peer_visible"
    assert batch.output_recovery.schema_formatter_enabled is True


def test_materialize_batch_defaults_to_peer_visible_schema_formatter(tmp_path: Path):
    cfg = materialize_batch(run_id="r", run_dir=tmp_path, loops=[])

    assert cfg["output_recovery"]["schema_violation_policy"] == "peer_visible"
    assert "reviewer_validation_retries" not in cfg["output_recovery"]
    assert cfg["output_recovery"]["schema_formatter"]["enabled"] is True


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
        model_tier: str | None = None,
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
                "model_tier": model_tier,
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


TARGETED_START_VALUES = {"proposer_001": 1, "proposer_002": 2, "proposer_003": 3}
TARGETED_INCREMENTS = {"proposer_001": 1, "proposer_002": 2, "proposer_003": 3}
TARGETED_REVIEW_MESSAGES = {
    proposer_id: f"Add {increment} to your current number next round."
    for proposer_id, increment in TARGETED_INCREMENTS.items()
}


class TargetedArithmeticReviewRunner:
    def __call__(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None,
        provider: str,
        model: str | None,
        model_tier: str | None = None,
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
                    "message": "Each proposer must use only its addressed increment next round.",
                    "stop_requested": False,
                    "stop_reason": "",
                },
                "proposer_messages": {
                    proposer_id: {"message": TARGETED_REVIEW_MESSAGES[proposer_id]}
                    for proposer_id in values
                },
                "review_payload": {
                    "round": round_number,
                    "current_values": values,
                    "requests": {proposer_id: TARGETED_REVIEW_MESSAGES[proposer_id] for proposer_id in values},
                },
            }

        previous = _latest_proposer_value(context["correspondence"], worker_id)
        review_message = _latest_proposer_message(context["correspondence"], worker_id)
        if previous is None:
            value = TARGETED_START_VALUES[worker_id]
            increment = 0
            followed_review = False
        elif review_message == TARGETED_REVIEW_MESSAGES[worker_id]:
            increment = TARGETED_INCREMENTS[worker_id]
            value = previous + increment
            followed_review = True
        else:
            increment = 0
            value = previous
            followed_review = False
        return {
            "worker_id": worker_id,
            "round": round_number,
            "received_reviewer_message": review_message or "none",
            "applied_increment": increment,
            "value": value,
            "followed_review": followed_review,
        }


def test_runner_sends_only_reply_correspondence_while_saving_prompt_artifacts(tmp_path):
    fake = FakeJsonRunner()

    result = run_proposers_reviewer_batch(base_config(tmp_path), json_runner=fake, base_env={})

    assert result["status"] == "completed"
    run_root = tmp_path / "ideas" / "run_001"
    round1_p1_prompt_path = run_root / "loops/loop_001/rounds/round_001/prompts/proposer_001.md"
    round1_p1_prompt = round1_p1_prompt_path.read_text(
        encoding="utf-8"
    )
    round2_p1_prompt = (run_root / "loops/loop_001/rounds/round_002/prompts/proposer_001.md").read_text(
        encoding="utf-8"
    )
    round1_review_prompt = (run_root / "loops/loop_001/rounds/round_001/prompts/reviewer_001.md").read_text(
        encoding="utf-8"
    )

    round2_p1_context = json.loads(
        (run_root / "loops/loop_001/rounds/round_002/context/proposer_001.json").read_text(encoding="utf-8")
    )
    correspondence_types = {event["type"] for event in round2_p1_context["correspondence"]}

    assert round1_p1_prompt_path.exists()
    assert "output-from-proposer_002-round-1" not in round1_p1_prompt
    assert "output-from-proposer_001-round-1" in round2_p1_prompt
    assert "output-from-proposer_002-round-1" in round2_p1_prompt
    assert "propose proposer_001 round 1" not in round2_p1_prompt
    assert "review-to-proposer_001-round-1" in round2_p1_prompt
    assert "output-from-proposer_001-round-1" in round1_review_prompt
    assert "output-from-proposer_002-round-1" in round1_review_prompt
    assert correspondence_types == {"controller_message", "proposer_message", "proposer_output", "review"}
    assert all(event["type"] not in {"proposer_prompt", "reviewer_prompt"} for event in round2_p1_context["correspondence"])

    transcript_events = [
        json.loads(line)
        for line in (run_root / "loops/loop_001/transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(event["type"] not in {"proposer_prompt", "reviewer_prompt"} for event in transcript_events)


def test_prompt_artifacts_can_be_disabled(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["artifact_options"] = {"save_prompts": False}

    result = run_proposers_reviewer_batch(config, json_runner=FakeJsonRunner(), base_env={})

    run_root = tmp_path / "ideas" / "run_001"
    assert result["status"] == "completed"
    assert not (run_root / "loops/loop_001/rounds/round_001/prompts/proposer_001.md").exists()
    assert not (run_root / "loops/loop_001/rounds/round_001/prompts/reviewer_001.md").exists()
    assert (run_root / "loops/loop_001/rounds/round_001/proposer_outputs/proposer_001.json").exists()


def test_prompt_artifacts_put_instructions_before_variable_context_for_cache_reuse(tmp_path):
    prompts = []

    def static_runner(prompt, **_):
        prompts.append(prompt)
        if "reviewer system" in prompt:
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    run_proposers_reviewer_batch(base_config(tmp_path, max_rounds=1), json_runner=static_runner, base_env={})

    first_prompt = prompts[0]
    assert first_prompt.startswith("## ARC Worker Instructions\n")
    assert first_prompt.index("### System") < first_prompt.index("## ARC Worker Context")
    assert first_prompt.index("### Task") < first_prompt.index("## ARC Worker Context")


def test_strict_mode_worker_call_errors_are_saved_as_debug_artifacts(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = {"enabled": True, "mode": "strict"}

    result = run_proposers_reviewer_batch(config, json_runner=FakeJsonRunner(fail_loop="loop_001"), base_env={})

    error_path = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/errors/proposer_001.json"
    error = json.loads(error_path.read_text(encoding="utf-8"))

    assert result["status"] == "failed"
    assert error["worker_id"] == "proposer_001"
    assert error["error_type"] == "RuntimeError"
    assert error["message"] == "simulated provider failure"


def test_proposer_exception_degrades_warn_mode_loop_without_fabricating_output(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    calls: list[str] = []

    def failing_proposer(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        calls.append(context["worker_id"])
        if context["worker_id"] == "proposer_001":
            raise RuntimeError("simulated proposer failure")
        if context["worker_id"].startswith("reviewer"):
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False, "stop_reason": ""},
                "proposer_messages": {
                    "proposer_001": {"message": "inspect failure"},
                    "proposer_002": {"message": "ok"},
                },
                "review_payload": {"ok": True},
            }
        return {"worker_id": context["worker_id"], "content": "ok"}

    result = run_proposers_reviewer_batch(config, json_runner=failing_proposer, base_env={})

    run_root = tmp_path / "ideas/run_001"
    failed_output = run_root / "loops/loop_001/rounds/round_001/proposer_outputs/proposer_001.json"
    error = json.loads(
        (run_root / "loops/loop_001/rounds/round_001/errors/proposer_001.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "degraded"
    assert result["loops"][0]["status"] == "degraded"
    assert "reviewer_001" in calls
    assert not failed_output.exists()
    assert error["call_status"] == "provider_error"
    assert error["message"] == "simulated proposer failure"


def test_worker_env_maps_claude_json_schema_runtime_options(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["loops"][0]["proposers"][0]["runtime"].update(
        {
            "claude_json_schema_mode": "auto",
            "claude_warn_json_schema_mode": "provider",
            "claude_json_schema_prompt_models": "deepseek,qwen",
        }
    )
    batch = load_batch_config(config)

    env = runner_module.worker_env(batch.loops[0].proposers[0], base_env={})

    assert env["ARC_CLAUDE_JSON_SCHEMA_MODE"] == "auto"
    assert env["ARC_CLAUDE_WARN_JSON_SCHEMA_MODE"] == "provider"
    assert env["ARC_CLAUDE_JSON_SCHEMA_PROMPT_MODELS"] == "deepseek,qwen"


def test_three_proposers_follow_targeted_reviewer_requests_for_three_rounds(tmp_path):
    config = base_config(tmp_path, max_rounds=3)
    third_proposer = json.loads(json.dumps(config["loops"][0]["proposers"][0]))
    third_proposer["id"] = "proposer_003"
    config["loops"][0]["proposers"].append(third_proposer)

    result = run_proposers_reviewer_batch(
        config,
        json_runner=TargetedArithmeticReviewRunner(),
        base_env={},
    )

    run_root = tmp_path / "ideas" / "run_001"
    values_by_round = {}
    received_messages_by_round = {}
    for round_number in (1, 2, 3):
        round_root = run_root / "loops" / "loop_001" / "rounds" / f"round_{round_number:03d}"
        proposer_outputs = {
            proposer_id: json.loads((round_root / "proposer_outputs" / f"{proposer_id}.json").read_text())
            for proposer_id in TARGETED_START_VALUES
        }
        values_by_round[round_number] = {proposer_id: output["value"] for proposer_id, output in proposer_outputs.items()}
        received_messages_by_round[round_number] = {
            proposer_id: output["received_reviewer_message"] for proposer_id, output in proposer_outputs.items()
        }
        review = json.loads((round_root / "reviews" / "reviewer_001.json").read_text())
        assert review["review_payload"]["requests"] == TARGETED_REVIEW_MESSAGES

    assert result["status"] == "completed"
    assert result["loops"][0]["rounds_completed"] == 3
    assert values_by_round == {
        1: {"proposer_001": 1, "proposer_002": 2, "proposer_003": 3},
        2: {"proposer_001": 2, "proposer_002": 4, "proposer_003": 6},
        3: {"proposer_001": 3, "proposer_002": 6, "proposer_003": 9},
    }
    assert received_messages_by_round == {
        1: {"proposer_001": "none", "proposer_002": "none", "proposer_003": "none"},
        2: TARGETED_REVIEW_MESSAGES,
        3: TARGETED_REVIEW_MESSAGES,
    }


def test_two_loops_run_in_isolated_directories(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    second = json.loads(json.dumps(config["loops"][0]))
    second["loop_id"] = "loop_002"
    config["loops"].append(second)

    result = run_proposers_reviewer_batch(config, json_runner=FakeJsonRunner(), base_env={})

    assert result["status"] == "completed"
    run_root = tmp_path / "ideas" / "run_001"
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
        (tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json").read_text(
            encoding="utf-8"
        )
    )

    assert loop["status"] == "completed"
    assert loop["rounds_completed"] == 3
    assert review["controller"]["stop_requested"] is True


def test_failed_loop_does_not_corrupt_successful_loop(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = {"enabled": True, "mode": "strict"}
    second = json.loads(json.dumps(config["loops"][0]))
    second["loop_id"] = "loop_bad"
    config["loops"].append(second)

    result = run_proposers_reviewer_batch(config, json_runner=FakeJsonRunner(fail_loop="loop_bad"), base_env={})

    statuses = {item["loop_id"]: item["status"] for item in result["loops"]}
    assert result["status"] == "degraded"
    assert statuses["loop_001"] == "completed"
    assert statuses["loop_bad"] == "failed"
    assert (tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json").exists()


def test_loop_failure_state_is_not_written_after_lock_loss(tmp_path):
    paths = RunPaths(run_dir=tmp_path / "ideas", run_id="run_001").loop("loop_001")
    atomic_write_json(paths.state, {"status": "running", "loop_id": "loop_001"})
    failure = {
        "loop_id": "loop_001",
        "status": "failed",
        "rounds_completed": 0,
        "error": "boom",
        "loop_root": str(paths.loop_root),
    }

    with acquire_lock(paths.lock, run_id="run_001", loop_id="loop_001"):
        runner_module._write_loop_failure_state(paths, run_id="run_001", result=failure)

    assert json.loads(paths.state.read_text(encoding="utf-8"))["status"] == "running"
    diagnostics = list((paths.loop_root / "errors").glob("failure_after_lock_lost.*.json"))
    assert len(diagnostics) == 1
    payload = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert payload["reason"] == "failed_to_reacquire_loop_lock"
    assert payload["failure_result"]["error"] == "boom"


def test_fail_fast_stops_scheduling_new_loops(monkeypatch, tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["fail_fast"] = True
    config["max_concurrent_loops"] = 1
    for loop_id in ["loop_002", "loop_003"]:
        loop = json.loads(json.dumps(config["loops"][0]))
        loop["loop_id"] = loop_id
        config["loops"].append(loop)
    started = []

    def fake_run_loop(loop, paths, run_id, artifact_options, *args):
        started.append(loop.loop_id)
        if loop.loop_id == "loop_001":
            return {
                "loop_id": loop.loop_id,
                "status": "failed",
                "rounds_completed": 0,
                "error": "boom",
                "loop_root": str(paths.loop_root),
            }
        return {
            "loop_id": loop.loop_id,
            "status": "completed",
            "rounds_completed": 1,
            "stop_reason": "",
            "loop_root": str(paths.loop_root),
        }

    monkeypatch.setattr(runner_module, "_run_loop", fake_run_loop)

    result = run_proposers_reviewer_batch(config, json_runner=FakeJsonRunner(), base_env={})

    assert started == ["loop_001"]
    statuses = {loop["loop_id"]: loop["status"] for loop in result["loops"]}
    assert statuses == {"loop_001": "failed", "loop_002": "skipped", "loop_003": "skipped"}


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

    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = {"enabled": True, "mode": "strict"}

    result = run_proposers_reviewer_batch(config, json_runner=invalid_reviewer, base_env={})

    assert result["status"] == "failed"
    assert result["loops"][0]["status"] == "failed"
    assert "review.review_payload must be an object" in result["loops"][0]["error"]


def test_invalid_reviewer_fails_warn_mode_without_fabricating_recovery(tmp_path):
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
        return {"ok": True}

    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = _fallback_output_recovery(1)

    result = run_proposers_reviewer_batch(config, json_runner=invalid_reviewer, base_env={})

    round_root = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001"
    error = json.loads((round_root / "errors/reviewer_001.json").read_text(encoding="utf-8"))

    assert result["status"] == "failed"
    assert result["loops"][0]["status"] == "failed"
    assert not (round_root / "reviews/reviewer_001.json").exists()
    assert error["call_status"] == "provider_error"
    assert "review.review_payload must be an object" in error["message"]


def test_review_envelope_rejects_unexpected_proposer_ids(tmp_path):
    def reviewer_with_extra_target(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "extra target", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                    "proposer_999": {"message": "not in this loop"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = {"enabled": True, "mode": "strict"}

    result = run_proposers_reviewer_batch(config, json_runner=reviewer_with_extra_target, base_env={})

    assert result["status"] == "failed"
    assert "review.proposer_messages unexpected: proposer_999" in result["loops"][0]["error"]


def test_invalid_reviewer_envelope_fails_without_retry_or_fabricated_review(tmp_path):
    calls = []

    def invalid_then_valid_reviewer(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        calls.append({"worker_id": context["worker_id"], "prompt": prompt})
        if context["worker_id"].startswith("reviewer"):
            reviewer_calls = [call for call in calls if call["worker_id"].startswith("reviewer")]
            if len(reviewer_calls) == 1:
                return {"message": "not an envelope"}
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "valid after retry", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {
            "worker_id": context["worker_id"],
            "round": context["round_number"],
            "content": "proposal",
        }

    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = _fallback_output_recovery(1)

    result = run_proposers_reviewer_batch(config, json_runner=invalid_then_valid_reviewer, base_env={})

    reviewer_prompts = [call["prompt"] for call in calls if call["worker_id"].startswith("reviewer")]
    round_root = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001"

    assert result["status"] == "failed"
    assert len(reviewer_prompts) == 1
    assert not (round_root / "prompts/reviewer_001.retry_001.md").exists()
    assert not (round_root / "reviews/reviewer_001.json").exists()


def test_reviewer_schema_failure_ignores_legacy_retry_count_and_fails(tmp_path):
    calls = 0

    def invalid_twice_then_valid(prompt, **kwargs):
        nonlocal calls
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            calls += 1
            if calls < 3:
                return {"message": f"invalid {calls}"}
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "valid after second retry", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = _fallback_output_recovery(2)

    result = run_proposers_reviewer_batch(config, json_runner=invalid_twice_then_valid, base_env={})

    round_root = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001"
    assert result["status"] == "failed"
    assert calls == 1
    assert not (round_root / "prompts/reviewer_001.retry_001.md").exists()
    assert not (round_root / "prompts/reviewer_001.retry_002.md").exists()
    assert not (round_root / "reviews/reviewer_001.json").exists()


def test_invalid_reviewer_plain_text_fails_without_retry_or_warning_output(tmp_path):
    calls = 0

    def invalid_reviewer(prompt, **kwargs):
        nonlocal calls
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            calls += 1
            return "Referee says derivation is unclear; ask both proposers to recompute the boundary term."
        return {"ok": True}

    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = {
        "enabled": True,
        "mode": "warn",
        "allow_natural_language": True,
        "schema_violation_policy": "peer_visible",
        "reviewer_validation_retries": 0,
    }

    result = run_proposers_reviewer_batch(config, json_runner=invalid_reviewer, base_env={})

    round_root = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001"
    assert result["status"] == "failed"
    assert calls == 1
    assert not (round_root / "prompts/reviewer_001.retry_001.md").exists()
    assert not (round_root / "reviews/reviewer_001.json").exists()


def test_invalid_idea_reviewer_no_longer_fabricates_zero_marks(tmp_path):
    def invalid_idea_reviewer(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            return "Malformed review, but all ideas should receive zero fallback marks."
        return {"title": "idea", "idea_summary": "summary"}

    config = base_config(tmp_path, max_rounds=1)
    config["loops"][0]["reviewers"][0]["output_schema"] = _idea_reviewer_schema()

    result = run_proposers_reviewer_batch(config, json_runner=invalid_idea_reviewer, base_env={})

    review_path = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json"
    assert result["status"] == "failed"
    assert not review_path.exists()


def test_custom_idea_reviewer_rich_format_failure_no_longer_uses_reviewer_formatter(tmp_path):
    calls = {"reviewer": 0, "formatter": 0}

    def rich_malformed_reviewer(prompt, **kwargs):
        if "Schema Formatter" in prompt:
            calls["formatter"] += 1
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {
                    "message": "formatted rich review",
                    "stop_requested": True,
                    "stop_reason": "ready",
                },
                "proposer_messages": {
                    "proposer_001": {"message": "accepted"},
                    "proposer_002": {"message": "accepted"},
                },
                "review_payload": {
                    "marks": {
                        "user_intent_relevance": 25,
                        "novelty": 13,
                        "confidence_of_novelty": 13,
                        "scientific_value": 13,
                        "planning": 14,
                        "problem_well_definedness": 14,
                        "total_score": 92,
                    }
                },
            }
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            calls["reviewer"] += 1
            return (
                "Final referee review: ready for execution. "
                "Scores: user_intent_relevance 25, novelty 13, confidence_of_novelty 13, "
                "scientific_value 13, planning 14, problem_well_definedness 14, total_score 92."
            )
        return {"title": "idea", "idea_summary": "summary"}

    config = base_config(tmp_path, max_rounds=1, early_stop=True)
    config["loops"][0]["reviewers"][0]["output_schema"] = _idea_reviewer_schema()

    result = run_proposers_reviewer_batch(config, json_runner=rich_malformed_reviewer, base_env={})

    review_path = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json"
    assert result["status"] == "failed"
    assert calls == {"reviewer": 1, "formatter": 0}
    assert not review_path.exists()


def test_reviewer_exception_fails_warn_mode_loop_without_fabricated_output(tmp_path):
    def reviewer_raises(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            raise RuntimeError("reviewer crashed")
        return {"title": "idea", "idea_summary": "summary"}

    config = base_config(tmp_path, max_rounds=1)
    config["loops"][0]["reviewers"][0]["output_schema"] = _idea_reviewer_schema()

    result = run_proposers_reviewer_batch(config, json_runner=reviewer_raises, base_env={})

    round_root = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001"
    error = json.loads((round_root / "errors/reviewer_001.json").read_text(encoding="utf-8"))

    assert result["status"] == "failed"
    assert not (round_root / "reviews/reviewer_001.json").exists()
    assert error["error_type"] == "RuntimeError"


def test_peer_visible_proposer_plain_text_is_wrapped_for_reviewer_context_and_warned(tmp_path):
    reviewer_seen: dict[str, Any] = {}

    def plain_text_proposer(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"] == "proposer_001":
            return "Proposal: compute the saddle correction and compare the two asymptotic limits."
        if context["worker_id"].startswith("reviewer"):
            reviewer_seen.update(context["current_proposer_outputs"])
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "expand"},
                    "proposer_002": {"message": "expand"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = {
        "enabled": True,
        "mode": "warn",
        "allow_natural_language": True,
        "schema_violation_policy": "peer_visible",
        "reviewer_validation_retries": 0,
    }

    result = run_proposers_reviewer_batch(config, json_runner=plain_text_proposer, base_env={})

    wrapped = reviewer_seen["proposer_001"]
    warnings = [
        json.loads(line)
        for line in (tmp_path / "ideas/run_001/structured_output_warnings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result["status"] == "completed"
    assert wrapped["schema_version"] == "arc.llm.unstructured_output.v1"
    assert wrapped["raw_text"].startswith("Proposal: compute")
    assert warnings[-1]["worker_id"] == "proposer_001"
    assert warnings[-1]["recovery_strategy"] == "peer_visible_unstructured_output"


def test_peer_visible_short_reviewer_text_fails_without_fabricated_warning_output(tmp_path):
    def service_unavailable_reviewer(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            return "Service unavailable"
        return {"ok": True}

    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = {
        "enabled": True,
        "mode": "warn",
        "allow_natural_language": True,
        "schema_violation_policy": "peer_visible",
        "reviewer_validation_retries": 0,
    }

    result = run_proposers_reviewer_batch(config, json_runner=service_unavailable_reviewer, base_env={})

    round_root = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001"
    assert result["status"] == "failed"
    assert result["warnings_summary"]["structured_output_warning_count"] == 0
    assert not (round_root / "reviews/reviewer_001.json").exists()


def test_warn_mode_fatal_mcp_failure_still_raises(tmp_path):
    def fatal_reviewer(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            raise RuntimeError("MCP server failed")
        return {"ok": True}

    config = base_config(tmp_path, max_rounds=1)

    result = run_proposers_reviewer_batch(config, json_runner=fatal_reviewer, base_env={})

    assert result["status"] == "failed"
    assert "MCP server failed" in result["loops"][0]["error"]


def test_valid_json_batch_has_empty_warning_summary(tmp_path):
    result = run_proposers_reviewer_batch(base_config(tmp_path, max_rounds=1), json_runner=FakeJsonRunner(), base_env={})

    summary = result["warnings_summary"]

    assert result["status"] == "completed"
    assert summary["structured_output_warning_count"] == 0
    assert summary["cache_warning_count"] == 0
    assert not Path(summary["structured_output_warnings_path"]).exists()
    assert not Path(summary["cache_warnings_path"]).exists()


def _strict_reviewer_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "controller", "proposer_messages", "review_payload"],
        "properties": {
            "schema_version": {"type": "string", "const": "arc.llm.review_envelope.v1"},
            "controller": {
                "type": "object",
                "additionalProperties": False,
                "required": ["message", "stop_requested", "stop_reason"],
                "properties": {
                    "message": {"type": "string"},
                    "stop_requested": {"type": "boolean"},
                    "stop_reason": {"type": ["string", "null"]},
                },
            },
            "proposer_messages": {
                "type": "object",
                "additionalProperties": False,
                "required": ["proposer_001", "proposer_002"],
                "properties": {
                    "proposer_001": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["message"],
                        "properties": {"message": {"type": "string"}},
                    },
                    "proposer_002": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["message"],
                        "properties": {"message": {"type": "string"}},
                    },
                },
            },
            "review_payload": {
                "type": "object",
                "required": ["consensus"],
                "properties": {
                    "consensus": {
                        "type": "object",
                        "required": ["status", "workflow_action"],
                        "properties": {
                            "status": {"type": "string", "enum": ["accepted", "unresolved"]},
                            "workflow_action": {
                                "type": "object",
                                "required": ["action"],
                                "properties": {
                                    "action": {"type": "string", "enum": ["finalize", "retry", "pause_for_human"]},
                                    "requires_human": {"type": "boolean"},
                                    "issue_type": {"type": "string", "enum": ["none", "worker_failure"]},
                                    "proposed_revision": {"type": ["string", "null"]},
                                    "reason": {"type": "string"},
                                    "expert_question": {"type": ["string", "null"]},
                                },
                                "additionalProperties": True,
                            },
                        },
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": True,
            },
        },
    }


def _idea_reviewer_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "controller", "proposer_messages", "review_payload"],
        "properties": {
            "schema_version": {"type": "string", "const": "arc.llm.review_envelope.v1"},
            "controller": {
                "type": "object",
                "additionalProperties": False,
                "required": ["message", "stop_requested", "stop_reason"],
                "properties": {
                    "message": {"type": "string"},
                    "stop_requested": {"type": "boolean"},
                    "stop_reason": {"type": ["string", "null"]},
                },
            },
            "proposer_messages": {
                "type": "object",
                "additionalProperties": False,
                "required": ["proposer_001", "proposer_002"],
                "properties": {
                    "proposer_001": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["message"],
                        "properties": {"message": {"type": "string"}},
                    },
                    "proposer_002": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["message"],
                        "properties": {"message": {"type": "string"}},
                    },
                },
            },
            "review_payload": {
                "type": "object",
                "additionalProperties": False,
                "required": ["marks"],
                "properties": {
                    "marks": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "user_intent_relevance",
                            "novelty",
                            "confidence_of_novelty",
                            "scientific_value",
                            "planning",
                            "problem_well_definedness",
                            "total_score",
                        ],
                        "properties": {
                            "user_intent_relevance": {"type": "number"},
                            "novelty": {"type": "number"},
                            "confidence_of_novelty": {"type": "number"},
                            "scientific_value": {"type": "number"},
                            "planning": {"type": "number"},
                            "problem_well_definedness": {"type": "number"},
                            "total_score": {"type": "number"},
                        },
                    }
                },
            },
        },
    }


def _reviewer_schema_config(tmp_path: Path, *, output_recovery: dict[str, Any] | None = None) -> dict[str, Any]:
    config = base_config(tmp_path, max_rounds=1)
    config["loops"][0]["reviewers"][0]["output_schema"] = _strict_reviewer_schema()
    if output_recovery is not None:
        config["output_recovery"] = output_recovery
    return config


def _valid_strict_review(message: str = "valid") -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.review_envelope.v1",
        "controller": {"message": message, "stop_requested": False, "stop_reason": None},
        "proposer_messages": {
            "proposer_001": {"message": "revise"},
            "proposer_002": {"message": "revise"},
        },
        "review_payload": {
            "consensus": {
                "status": "unresolved",
                "workflow_action": {"action": "retry"},
            }
        },
    }


def test_raw_output_text_prefers_structured_raw_excerpt():
    output = {
        ARC_LLM_CALL_RECORD_FIELD: {
            "structured_output": {
                "mode": "recovered",
                "raw_text_excerpt": "real raw model text",
            }
        }
    }

    assert runner_module._raw_output_text(output) == "real raw model text"  # noqa: SLF001


def _shallow_valid_nested_invalid_review() -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.review_envelope.v1",
        "controller": {"message": "missing nested fields", "stop_requested": False, "stop_reason": None},
        "proposer_messages": {
            "proposer_001": {"message": "revise"},
            "proposer_002": {"message": "revise"},
        },
        "review_payload": {},
    }


def test_reviewer_valid_schema_does_not_retry_or_recover(tmp_path):
    calls = 0

    def valid_reviewer(prompt, **kwargs):
        nonlocal calls
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            calls += 1
            return _valid_strict_review("valid first try")
        return {"ok": True}

    result = run_proposers_reviewer_batch(
        _reviewer_schema_config(tmp_path),
        json_runner=valid_reviewer,
        base_env={},
    )

    review_path = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert calls == 1
    assert not (tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/prompts/reviewer_001.retry_001.md").exists()
    assert not (tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/errors/reviewer_001.validation_001.json").exists()
    record = review.get(ARC_LLM_CALL_RECORD_FIELD, {})
    assert record.get("structured_output") in (None, {})


def test_reviewer_full_schema_failure_fails_without_retry_or_output(tmp_path):
    calls = 0

    def invalid_then_valid_reviewer(prompt, **kwargs):
        nonlocal calls
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            calls += 1
            if calls == 1:
                return _shallow_valid_nested_invalid_review()
            return _valid_strict_review("valid after full schema retry")
        return {"ok": True}

    result = run_proposers_reviewer_batch(
        _reviewer_schema_config(tmp_path, output_recovery=_fallback_output_recovery(1)),
        json_runner=invalid_then_valid_reviewer,
        base_env={},
    )

    retry_prompt = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/prompts/reviewer_001.retry_001.md"
    review_path = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json"
    assert result["status"] == "failed"
    assert calls == 1
    assert not retry_prompt.exists()
    assert not review_path.exists()


def test_reviewer_schema_failure_fails_warn_mode_without_fabricated_output(tmp_path):
    calls = 0

    def invalid_reviewer(prompt, **kwargs):
        nonlocal calls
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            calls += 1
            return _shallow_valid_nested_invalid_review()
        return {"ok": True}

    result = run_proposers_reviewer_batch(
        _reviewer_schema_config(tmp_path, output_recovery=_fallback_output_recovery(1)),
        json_runner=invalid_reviewer,
        base_env={},
    )

    review_path = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json"
    assert result["status"] == "failed"
    assert calls == 1
    assert not review_path.exists()


def test_reviewer_schema_failure_cannot_preserve_approving_retry(tmp_path):
    calls = 0

    def invalid_then_approving_invalid_reviewer(prompt, **kwargs):
        nonlocal calls
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            calls += 1
            if calls == 1:
                return _shallow_valid_nested_invalid_review()
            review = _valid_strict_review("invalid approving retry")
            review["review_payload"]["consensus"]["status"] = "accepted"
            review["review_payload"]["consensus"]["workflow_action"]["action"] = "finalize"
            review["review_payload"]["consensus"]["workflow_action"]["requires_human"] = False
            review["review_payload"]["consensus"]["accepted_result"] = {"value": "unsafe"}
            review["extra_property_breaks_schema"] = True
            return review
        return {"ok": True}

    result = run_proposers_reviewer_batch(
        _reviewer_schema_config(tmp_path, output_recovery=_fallback_output_recovery(1)),
        json_runner=invalid_then_approving_invalid_reviewer,
        base_env={},
    )

    review_path = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json"
    assert result["status"] == "failed"
    assert calls == 1
    assert not review_path.exists()


def test_reviewer_schema_failure_raises_in_strict_mode(tmp_path):
    def invalid_reviewer(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            return _shallow_valid_nested_invalid_review()
        return {"ok": True}

    result = run_proposers_reviewer_batch(
        _reviewer_schema_config(tmp_path, output_recovery={"enabled": True, "mode": "strict"}),
        json_runner=invalid_reviewer,
        base_env={},
    )

    assert result["status"] == "failed"
    assert result["loops"][0]["status"] == "failed"
    assert "reviewer output failed JSON schema validation" in result["loops"][0]["error"]


def test_reviewer_schema_validation_is_requested_from_custom_runner(tmp_path):
    calls = []

    def invalid_then_valid_reviewer(prompt, *, validate_schema=True, **kwargs):
        is_reviewer = "reviewer system" in prompt
        if is_reviewer:
            calls.append(validate_schema)
            return {"message": "not an envelope"}
        assert validate_schema is True
        return {"ok": True}

    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = _fallback_output_recovery(1)

    result = run_proposers_reviewer_batch(config, json_runner=invalid_then_valid_reviewer, base_env={})

    assert result["status"] == "failed"
    assert calls == [True]


def test_reviewer_validation_artifact_is_not_saved_after_reviewer_fallback_removal(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["artifact_options"] = {"save_prompts": False}
    config["output_recovery"] = _fallback_output_recovery(1)
    calls = 0

    def invalid_then_valid_reviewer(prompt, **kwargs):
        nonlocal calls
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            calls += 1
            if calls == 1:
                return {"message": "not an envelope"}
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "valid after retry", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    result = run_proposers_reviewer_batch(config, json_runner=invalid_then_valid_reviewer, base_env={})

    validation_error = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/errors/reviewer_001.validation_001.json"
    assert result["status"] == "failed"
    assert calls == 1
    assert not validation_error.exists()


def test_worker_envs_are_isolated_and_os_environ_is_not_mutated(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_CODEX_ALLOW_INTERNET", raising=False)
    monkeypatch.delenv("ARC_CODEX_ENABLE_MCP", raising=False)
    fake = FakeJsonRunner()

    run_proposers_reviewer_batch(base_config(tmp_path, max_rounds=1), json_runner=fake, base_env={})

    proposer_call = next(call for call in fake.calls if call["worker_id"] == "proposer_001")
    reviewer_call = next(call for call in fake.calls if call["worker_id"] == "reviewer_001")
    assert proposer_call["env"]["ARC_CODEX_ALLOW_INTERNET"] == "true"
    assert proposer_call["env"]["ARC_CODEX_ENABLE_MCP"] == "false"
    assert reviewer_call["env"]["ARC_CODEX_ENABLE_MCP"] == "true"
    assert "ARC_CODEX_ALLOW_INTERNET" not in os.environ
    assert "ARC_CODEX_ENABLE_MCP" not in os.environ


def test_worker_model_tier_is_passed_as_runner_argument(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["defaults"].pop("model")
    config["defaults"]["provider"] = "auto"
    config["defaults"]["model_tier"] = "high"
    fake = FakeJsonRunner()

    run_proposers_reviewer_batch(config, json_runner=fake, base_env={})

    assert {call["model_tier"] for call in fake.calls} == {"high"}
    assert all("ARC_LLM_MODEL_TIER" not in call["env"] for call in fake.calls)


def test_custom_json_runner_receives_process_chain_when_supported(tmp_path):
    calls = []

    def fake(prompt, *, schema, provider, model, model_tier=None, env, process_chain):
        context = _context_from_prompt(prompt)
        calls.append({"worker_id": context["worker_id"], "process_chain": process_chain})
        if context["worker_id"].startswith("reviewer"):
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    run_proposers_reviewer_batch(
        base_config(tmp_path, max_rounds=1),
        json_runner=fake,
        base_env={},
        process_chain=["codex", "bash"],
    )

    assert calls
    assert {tuple(call["process_chain"]) for call in calls} == {("codex", "bash")}


def test_stateful_runner_uses_initial_then_delta_prompts_and_stable_session_keys(tmp_path):
    config = base_config(tmp_path, max_rounds=2)
    config["session"] = {"policy": "stateful", "history_mode": "delta", "max_concurrent_same_prefix": 2}
    calls: list[dict[str, Any]] = []

    def fake(prompt, *, schema, provider, model, model_tier=None, env, session_policy, session_key, call_label, **kwargs):
        calls.append(
            {
                "prompt": prompt,
                "session_policy": session_policy,
                "session_key": session_key,
                "call_label": call_label,
            }
        )
        if "/reviewer/" in session_key:
            round_number = 1 if "round_001" in call_label else 2
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": f"reviewed {round_number}", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": f"revise p1 {round_number}"},
                    "proposer_002": {"message": f"revise p2 {round_number}"},
                },
                "review_payload": {"round": round_number},
                ARC_LLM_CALL_RECORD_FIELD: {"native_session_id": f"native:{session_key}"},
            }
        worker_id = "proposer_001" if session_key.endswith("/proposer_001") else "proposer_002"
        round_number = 1 if "round_001" in call_label else 2
        return {
            "worker_id": worker_id,
            "round": round_number,
            "content": f"output-from-{worker_id}-round-{round_number}",
            ARC_LLM_CALL_RECORD_FIELD: {"native_session_id": f"native:{session_key}"},
        }

    result = run_proposers_reviewer_batch(config, json_runner=fake, base_env={})

    run_root = tmp_path / "ideas" / "run_001"
    p1_calls = [call for call in calls if call["session_key"].endswith("/proposer/proposer_001")]
    reviewer_calls = [call for call in calls if call["session_key"].endswith("/reviewer/reviewer_001")]
    round2_p1_prompt = p1_calls[1]["prompt"]
    round2_context = json.loads(
        (run_root / "loops/loop_001/rounds/round_002/context/proposer_001.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "completed"
    assert len(p1_calls) == 2
    assert {call["session_key"] for call in p1_calls} == {"run_001/loop_001/proposer/proposer_001"}
    assert {call["session_key"] for call in reviewer_calls} == {"run_001/loop_001/reviewer/reviewer_001"}
    assert "## ARC-LLM Worker Session ABI v2" in p1_calls[0]["prompt"]
    assert "## ARC-LLM Worker Delta Turn v2" in round2_p1_prompt
    assert "JSON Schema/output contract for this turn may differ" in reviewer_calls[1]["prompt"]
    assert "output-from-proposer_002-round-1" not in round2_p1_prompt
    assert "revise p1 1" in round2_p1_prompt
    assert round2_context["turn_kind"] == "delta"
    assert round2_context["session_key"] == "run_001/loop_001/proposer/proposer_001"


def test_locked_turn_kind_uses_initial_when_calls_exist_but_native_session_missing(tmp_path):
    config = base_config(tmp_path, max_rounds=2)
    config["session"] = {"policy": "stateful", "history_mode": "delta"}
    batch = load_batch_config(config)
    loop = batch.loops[0]
    manager = core_runner.LLMSessionManager(tmp_path / "sessions")
    session_key = "run_001/loop_001/proposer/proposer_001"
    manager.record_turn(
        session_key,
        call_label="prior",
        prompt_sha256="prompt",
        static_prefix_sha256=None,
        schema_sha256=None,
        usage={},
        provider_used="codex-cli",
        model_used="m",
        native_session_id=None,
    )
    prompt_options = {
        "initial": runner_module.WorkerPromptOption("initial", "initial", {}, None, tmp_path / "initial.md"),
        "delta": runner_module.WorkerPromptOption("delta", "delta", {}, None, tmp_path / "delta.md"),
    }

    assert runner_module._locked_turn_kind(loop, manager, session_key, prompt_options) == "initial"  # noqa: SLF001


def test_initial_prompt_static_prefix_stays_shared_until_variable_context(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    batch = load_batch_config(config)
    loop = batch.loops[0]

    prompt_1, _context_1, static_prefix_1 = render_initial_worker_prompt(
        loop=loop,
        worker=loop.proposers[0],
        role="proposer",
        round_number=1,
    )
    prompt_2, _context_2, static_prefix_2 = render_initial_worker_prompt(
        loop=loop,
        worker=loop.proposers[1],
        role="proposer",
        round_number=1,
    )
    variable_index = prompt_1.index("## Variable Initial Context")

    assert static_prefix_1 == static_prefix_2
    assert prompt_1.startswith(static_prefix_1)
    assert prompt_2.startswith(static_prefix_2)
    assert prompt_1[:variable_index] == prompt_2[:variable_index]
    assert prompt_1 != prompt_2


def test_initial_prompt_explains_split_caller_context(tmp_path):
    batch = load_batch_config(base_config(tmp_path, max_rounds=1))
    loop = batch.loops[0]

    prompt, _context, _static_prefix = render_initial_worker_prompt(
        loop=loop,
        worker=loop.proposers[0],
        role="proposer",
        round_number=1,
    )

    assert "union of Shared Static Task Context.caller_context" in prompt
    assert "Variable Initial Context.caller_context" in prompt


def test_proposer_delta_tells_worker_to_recompute_not_patch(tmp_path):
    batch = load_batch_config(base_config(tmp_path, max_rounds=2))
    loop = batch.loops[0]

    prompt, _context = render_proposer_delta_prompt(
        loop=loop,
        worker=loop.proposers[0],
        round_number=2,
        correspondence=[],
    )

    assert "remembered static caller_context" in prompt
    assert "caller_context_delta" in prompt
    assert "tentative scratch work, not as facts" in prompt
    assert "recompute" in prompt
    assert "merely patching" in prompt


def test_reviewer_delta_requires_independent_current_review(tmp_path):
    batch = load_batch_config(base_config(tmp_path, max_rounds=2))
    loop = batch.loops[0]

    prompt, _context = render_reviewer_delta_prompt(
        loop=loop,
        worker=loop.reviewers[0],
        round_number=2,
        current_proposer_outputs={"proposer_001": {"ok": True}},
    )

    assert "remembered static caller_context" in prompt
    assert "caller_context_delta" in prompt
    assert "independently" in prompt
    assert "Previous review history is background only" in prompt
    assert "current active_proposer_ids" in prompt


def test_custom_json_runner_that_calls_run_json_does_not_double_record_turns(tmp_path, monkeypatch):
    config = base_config(tmp_path, max_rounds=1)
    config["session"] = {"policy": "stateful", "history_mode": "delta"}

    class Provider:
        name = "codex-cli"

        def generate_json_result(
            self,
            prompt,
            *,
            schema=None,
            model=None,
            session=None,
            session_policy="stateless",
            schema_cache_dir=None,
            artifact_dir=None,
        ):
            if session and "/reviewer/" in session.key:
                value = {
                    "schema_version": "arc.llm.review_envelope.v1",
                    "controller": {"message": "reviewed", "stop_requested": False},
                    "proposer_messages": {
                        "proposer_001": {"message": "revise"},
                        "proposer_002": {"message": "revise"},
                    },
                    "review_payload": {"ok": True},
                }
            else:
                value = {"ok": True}
            value["arc_evidence_requests"] = []
            return LLMProviderResponse(
                value,
                usage=LLMUsage(input_tokens=10, cached_input_tokens=8),
                native_session_id=f"native-{session.key}" if session else None,
            )

    def custom_runner(
        prompt,
        *,
        schema,
        provider,
        model,
        model_tier=None,
        env,
        process_chain=None,
        session_policy,
        session_manager,
        session_key,
        call_label,
        artifact_dir,
        static_prefix=None,
    ):
        return core_runner.run_json(
            prompt,
            schema=schema,
            provider=provider,
            model=model,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
            session_policy=session_policy,
            session_manager=session_manager,
            session_key=session_key,
            call_label=call_label,
                artifact_dir=artifact_dir,
                static_prefix=static_prefix,
                idempotency_key=call_label,
            )

    monkeypatch.setattr(core_runner, "select_provider", lambda provider, **kwargs: Provider())

    result = run_proposers_reviewer_batch(config, json_runner=custom_runner, base_env={})
    calls_path = tmp_path / "ideas/run_001/sessions/calls.jsonl"
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]

    assert result["status"] == "completed"
    assert len(calls) == 3
    assert {call["call_label"] for call in calls} == {
        "loop/loop_001/round_001/proposer_001",
        "loop/loop_001/round_001/proposer_002",
        "loop/loop_001/round_001/reviewer_001",
    }


def test_non_reuse_scope_id_still_isolates_sessions_by_loop(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["session"] = {"policy": "stateful", "history_mode": "delta", "scope_id": "bench/run/current"}
    second_loop = json.loads(json.dumps(config["loops"][0]))
    second_loop["loop_id"] = "loop_002"
    config["loops"].append(second_loop)
    session_keys = set()

    def fake(prompt, *, schema, provider, model, model_tier=None, env, session_policy, session_key, call_label, **kwargs):
        session_keys.add(session_key)
        if "/reviewer/" in session_key:
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    result = run_proposers_reviewer_batch(config, json_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert "bench/run/current/loop_001/proposer/proposer_001" in session_keys
    assert "bench/run/current/loop_002/proposer/proposer_001" in session_keys
    assert "bench/run/current/proposer/proposer_001" not in session_keys


def test_loop_session_root_is_used_for_stateful_turn_records(tmp_path):
    shared_root = tmp_path / "shared_sessions"
    config = base_config(tmp_path, max_rounds=1)
    config["session"] = {"policy": "stateless", "history_mode": "full"}
    config["loops"][0]["session"] = {
        "policy": "stateful",
        "history_mode": "delta",
        "scope_id": "shared/scope",
        "reuse_across_batch_calls": True,
        "root": str(shared_root),
    }

    def fake(prompt, *, schema, provider, model, model_tier=None, env, session_policy, session_key, call_label, **kwargs):
        if "/reviewer/" in session_key:
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    result = run_proposers_reviewer_batch(config, json_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert (shared_root / "calls.jsonl").exists()
    assert not (tmp_path / "ideas/run_001/sessions/calls.jsonl").exists()


def test_custom_json_runner_with_var_kwargs_uses_legacy_full_prompts_by_default(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config.pop("session")
    prompts = []

    def fake(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        prompts.append(prompt)
        if context["worker_id"].startswith("reviewer"):
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    result = run_proposers_reviewer_batch(config, json_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert prompts
    assert all("## ARC Worker Context" in prompt for prompt in prompts)
    assert all("## ARC-LLM Worker Session ABI v2" not in prompt for prompt in prompts)


def test_custom_json_runner_receives_output_recovery_when_supported(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = {
        "enabled": True,
        "mode": "warn",
        "schema_formatter": {"enabled": False},
    }
    seen: list[tuple[str, bool, str | None]] = []

    def fake(
        prompt,
        *,
        schema,
        provider,
        model,
        model_tier=None,
        env,
        session_policy,
        session_key,
        call_label,
        output_recovery,
        schema_formatter_enabled,
        role_hint,
        **kwargs,
    ):
        seen.append((output_recovery, schema_formatter_enabled, role_hint))
        if role_hint == "reviewer":
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    result = run_proposers_reviewer_batch(config, json_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert ("warn", False, "proposer") in seen
    assert ("warn", False, "reviewer") in seen


def test_output_recovery_disables_natural_language_when_configured(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["output_recovery"] = {"enabled": True, "mode": "warn", "allow_natural_language": False}
    seen: list[str] = []

    def fake(
        prompt,
        *,
        schema,
        provider,
        model,
        model_tier=None,
        env,
        output_recovery,
        role_hint,
        **kwargs,
    ):
        seen.append(output_recovery)
        if role_hint == "reviewer":
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    result = run_proposers_reviewer_batch(config, json_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert seen
    assert set(seen) == {"strict"}


def test_stateful_reviewer_validation_failure_is_not_retried(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["session"] = {"policy": "stateful", "history_mode": "delta"}
    config["output_recovery"] = _fallback_output_recovery(1)
    reviewer_prompts = []

    def invalid_then_valid(prompt, *, schema, provider, model, model_tier=None, env, session_policy, session_key, call_label, **kwargs):
        if "/reviewer/" in session_key:
            reviewer_prompts.append(prompt)
            if len(reviewer_prompts) == 1:
                return {"message": "not an envelope"}
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "valid after retry", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "revise"},
                    "proposer_002": {"message": "revise"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    result = run_proposers_reviewer_batch(config, json_runner=invalid_then_valid, base_env={})

    assert result["status"] == "failed"
    assert len(reviewer_prompts) == 1


def test_cache_guard_writes_warning_after_warmup(tmp_path):
    config = base_config(tmp_path, max_rounds=2)
    config["session"] = {
        "policy": "stateful",
        "history_mode": "delta",
        "cache_guard": {
            "enabled": True,
            "mode": "warn",
            "warmup_calls": 1,
            "min_cached_input_ratio": 0.70,
        },
    }

    def fake(prompt, *, schema, provider, model, model_tier=None, env, session_policy, session_key, call_label, **kwargs):
        if "/reviewer/" in session_key:
            round_number = 1 if "round_001" in call_label else 2
            result = {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": f"reviewed {round_number}", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": f"revise p1 {round_number}"},
                    "proposer_002": {"message": f"revise p2 {round_number}"},
                },
                "review_payload": {"round": round_number},
            }
        else:
            worker_id = "proposer_001" if session_key.endswith("/proposer_001") else "proposer_002"
            round_number = 1 if "round_001" in call_label else 2
            result = {"worker_id": worker_id, "round": round_number, "content": f"proposal {round_number}"}
        result[ARC_LLM_CALL_RECORD_FIELD] = {
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "cached_input_ratio": 0.0,
            }
        }
        return result

    result = run_proposers_reviewer_batch(config, json_runner=fake, base_env={})

    warning_path = tmp_path / "ideas/run_001/cache_warnings.jsonl"
    warnings = [json.loads(line) for line in warning_path.read_text(encoding="utf-8").splitlines()]

    assert result["status"] == "completed"
    assert warnings
    assert {warning["turn_count"] for warning in warnings} == {2}
    assert all("round_002" in warning["call_label"] for warning in warnings)
    assert {warning["cached_input_ratio"] for warning in warnings} == {0.0}


def test_prefix_concurrency_limiter_caps_threads_per_prefix():
    limiter = PrefixConcurrencyLimiter(default_limit=2)
    active = 0
    max_active = 0
    guard = threading.Lock()

    def worker():
        nonlocal active, max_active
        with limiter.acquire("shared-prefix"):
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)
            with guard:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active <= 2


def _context_from_prompt(prompt: str) -> dict[str, Any]:
    match = re.search(r"^## ARC Worker Context\n(?P<context>\{.*\})\s*$", prompt, re.S | re.M)
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
