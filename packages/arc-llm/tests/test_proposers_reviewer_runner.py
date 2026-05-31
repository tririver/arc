from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD
from arc_llm.proposers_reviewer import runner as runner_module
from arc_llm.proposers_reviewer.artifacts import RunPaths, acquire_lock, atomic_write_json
from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch


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


def test_worker_call_errors_are_saved_as_debug_artifacts(tmp_path):
    config = base_config(tmp_path, max_rounds=1)

    result = run_proposers_reviewer_batch(config, json_runner=FakeJsonRunner(fail_loop="loop_001"), base_env={})

    error_path = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/errors/proposer_001.json"
    error = json.loads(error_path.read_text(encoding="utf-8"))

    assert result["status"] == "failed"
    assert error["worker_id"] == "proposer_001"
    assert error["error_type"] == "RuntimeError"
    assert error["message"] == "simulated provider failure"


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
    second = json.loads(json.dumps(config["loops"][0]))
    second["loop_id"] = "loop_bad"
    config["loops"].append(second)

    result = run_proposers_reviewer_batch(config, json_runner=FakeJsonRunner(fail_loop="loop_bad"), base_env={})

    statuses = {item["loop_id"]: item["status"] for item in result["loops"]}
    assert result["status"] == "failed"
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

    result = run_proposers_reviewer_batch(base_config(tmp_path, max_rounds=1), json_runner=invalid_reviewer, base_env={})

    assert result["status"] == "failed"
    assert result["loops"][0]["status"] == "failed"
    assert "review.review_payload must be an object" in result["loops"][0]["error"]


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

    result = run_proposers_reviewer_batch(base_config(tmp_path, max_rounds=1), json_runner=reviewer_with_extra_target, base_env={})

    assert result["status"] == "failed"
    assert "review.proposer_messages unexpected: proposer_999" in result["loops"][0]["error"]


def test_invalid_reviewer_envelope_is_retried_once_with_validation_feedback(tmp_path):
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

    result = run_proposers_reviewer_batch(
        base_config(tmp_path, max_rounds=1),
        json_runner=invalid_then_valid_reviewer,
        base_env={},
    )

    reviewer_prompts = [call["prompt"] for call in calls if call["worker_id"].startswith("reviewer")]
    review = json.loads(
        (tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/reviews/reviewer_001.json").read_text(
            encoding="utf-8"
        )
    )
    original_prompt = (
        tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/prompts/reviewer_001.md"
    ).read_text(encoding="utf-8")
    retry_prompt = (
        tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/prompts/reviewer_001.retry_001.md"
    ).read_text(encoding="utf-8")

    assert result["status"] == "completed"
    assert len(reviewer_prompts) == 2
    assert "Previous reviewer response failed validation" in reviewer_prompts[1]
    assert "review schema_version must be arc.llm.review_envelope.v1" in reviewer_prompts[1]
    assert review["controller"]["message"] == "valid after retry"
    assert "Previous reviewer response failed validation" not in original_prompt
    assert "Previous reviewer response failed validation" in retry_prompt


def test_reviewer_validation_artifact_is_saved_when_prompts_are_disabled(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["artifact_options"] = {"save_prompts": False}
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
    assert result["status"] == "completed"
    assert validation_error.exists()
    payload = json.loads(validation_error.read_text(encoding="utf-8"))
    assert payload["message"] == "review schema_version must be arc.llm.review_envelope.v1"
    assert payload["original_prompt_path"] == ""
    assert payload["retry_prompt_path"] == ""


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
            }
        worker_id = "proposer_001" if session_key.endswith("/proposer_001") else "proposer_002"
        round_number = 1 if "round_001" in call_label else 2
        return {"worker_id": worker_id, "round": round_number, "content": f"output-from-{worker_id}-round-{round_number}"}

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
    assert "output-from-proposer_002-round-1" not in round2_p1_prompt
    assert "revise p1 1" in round2_p1_prompt
    assert round2_context["turn_kind"] == "delta"
    assert round2_context["session_key"] == "run_001/loop_001/proposer/proposer_001"


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


def test_stateful_reviewer_validation_retry_is_compact_delta(tmp_path):
    config = base_config(tmp_path, max_rounds=1)
    config["session"] = {"policy": "stateful", "history_mode": "delta"}
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

    assert result["status"] == "completed"
    assert len(reviewer_prompts) == 2
    assert "## ARC-LLM Reviewer Validation Retry v2" in reviewer_prompts[1]
    assert "reviewer system" not in reviewer_prompts[1]
    assert "Current Proposer Outputs" not in reviewer_prompts[1]


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
