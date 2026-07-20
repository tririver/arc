from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from arc_llm.evidence import EVIDENCE_REQUESTS_FIELD, EvidenceResponse
from arc_llm.proposers_reviewer.runner import run_proposers_reviewer_batch


def _config(tmp_path: Path, *, max_rounds: int) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "evidence_run",
        "run_dir": str(tmp_path),
        "session": {"policy": "stateless", "history_mode": "full"},
        "defaults": {"provider": "manual", "model": "fake"},
        "loops": [
            {
                "loop_id": "loop_001",
                "max_rounds": max_rounds,
                "early_stop": {"enabled": False},
                "proposers": [
                    {
                        "id": "proposer_001",
                        "prompt": {"system": "proposer", "template": "Propose in round {round_number}."},
                        "output_schema": {"type": "object", "additionalProperties": False, "properties": {}},
                        "runtime": {"allow_mcp": False},
                    }
                ],
                "reviewers": [
                    {
                        "id": "reviewer_001",
                        "prompt": {"system": "reviewer", "template": "Review in round {round_number}."},
                        "output_schema": {"type": "object"},
                        "runtime": {"allow_mcp": False},
                    }
                ],
                "caller_context": {"user_intent": "test evidence mediation"},
            }
        ],
    }


class EvidenceRunner:
    def __init__(self, proposer_requests) -> None:
        self.proposer_requests = proposer_requests
        self.proposer_prompts: list[str] = []
        self.proposer_schemas: list[dict[str, Any]] = []

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        round_number = int(re.findall(r'"round_number":\s*(\d+)', prompt)[-1])
        if "### System\nreviewer" in prompt:
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "continue", "stop_requested": False},
                "proposer_messages": {"proposer_001": {"message": "revise"}},
                "review_payload": {"round": round_number},
            }
        self.proposer_prompts.append(prompt)
        self.proposer_schemas.append(kwargs["schema"])
        output = {"proposal": f"round {round_number}"}
        requests = self.proposer_requests(round_number)
        if requests is not None:
            output[EVIDENCE_REQUESTS_FIELD] = requests
        return output


def test_runner_resolves_requests_and_injects_provenance_next_round(tmp_path: Path) -> None:
    runner = EvidenceRunner(
        lambda round_number: [
            {
                "request_id": "proposer_001-nearby-work",
                "operation": "paper.search",
                "arguments": {"query": "nearby work"},
                "reason": "check novelty",
            }
        ]
        if round_number == 1
        else []
    )
    calls = []

    def controller(requests, *, round_number):
        calls.append((requests, round_number))
        assert requests[0].worker_id == "proposer_001"
        assert requests[0].role == "proposer"
        return (
            EvidenceResponse(
                requests[0].request_id,
                True,
                {"matches": ["arXiv:1234.5678"]},
                provenance={"source": "arc-paper", "query": "nearby work"},
            ),
        )

    result = run_proposers_reviewer_batch(
        _config(tmp_path, max_rounds=2),
        json_runner=runner,
        evidence_controller=controller,
        base_env={},
    )

    assert result["status"] == "completed"
    assert len(calls) == 1
    assert calls[0][1] == 1
    assert EVIDENCE_REQUESTS_FIELD in runner.proposer_schemas[0]["properties"]
    assert '"source": "arc-paper"' in runner.proposer_prompts[1]
    assert '"matches": [' in runner.proposer_prompts[1]
    loop = result["loops"][0]
    assert loop["evidence_rounds_completed"] == 1
    assert loop["evidence_request_count"] == 1
    transcript = Path(loop["loop_root"]) / "transcript.jsonl"
    evidence_events = [
        json.loads(line)
        for line in transcript.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("type") == "evidence_responses"
    ]
    assert evidence_events[0]["exchanges"][0]["response"]["provenance"]["source"] == "arc-paper"


def test_empty_request_list_is_a_no_op(tmp_path: Path) -> None:
    runner = EvidenceRunner(lambda _round_number: [])
    controller_calls = 0

    def controller(_requests, *, round_number):
        nonlocal controller_calls
        controller_calls += 1
        return ()

    result = run_proposers_reviewer_batch(
        _config(tmp_path, max_rounds=1),
        json_runner=runner,
        evidence_controller=controller,
        base_env={},
    )

    assert result["status"] == "completed"
    assert controller_calls == 0
    assert "evidence_rounds_completed" not in result["loops"][0]


def test_disabled_evidence_is_not_advertised_or_resolved(tmp_path: Path) -> None:
    config = _config(tmp_path, max_rounds=2)
    config["evidence"] = {"enabled": False}
    config["loops"][0]["proposers"][0]["output_schema"] = {"type": "object"}
    runner = EvidenceRunner(
        lambda _round_number: [
            {
                "request_id": "proposer_001-forbidden",
                "operation": "paper.metadata",
                "arguments": {"paper_id": "0911.3380"},
            }
        ]
    )
    controller_calls = 0

    def controller(_requests, *, round_number):
        nonlocal controller_calls
        controller_calls += 1
        return ()

    result = run_proposers_reviewer_batch(
        config,
        json_runner=runner,
        evidence_controller=controller,
        base_env={},
    )

    assert result["status"] == "completed"
    assert controller_calls == 0
    assert all(
        EVIDENCE_REQUESTS_FIELD not in schema.get("properties", {})
        for schema in runner.proposer_schemas
    )
    assert all("controller evidence" not in prompt for prompt in runner.proposer_prompts)
    assert "evidence_request_count" not in result["loops"][0]
    transcript = Path(result["loops"][0]["loop_root"]) / "transcript.jsonl"
    assert all(
        json.loads(line).get("type") != "evidence_responses"
        for line in transcript.read_text(encoding="utf-8").splitlines()
    )


def test_worker_evidence_override_never_reaches_controller(tmp_path: Path) -> None:
    config = _config(tmp_path, max_rounds=2)
    config["loops"][0]["proposers"][0]["evidence"] = {"enabled": False}
    config["loops"][0]["proposers"][0]["output_schema"] = {"type": "object"}
    runner = EvidenceRunner(
        lambda _round_number: [
            {
                "request_id": "proposer_001-forbidden",
                "operation": "paper.metadata",
                "arguments": {"paper_id": "0911.3380"},
            }
        ]
    )
    controller_calls = 0

    def controller(_requests, *, round_number):
        nonlocal controller_calls
        controller_calls += 1
        return ()

    result = run_proposers_reviewer_batch(
        config,
        json_runner=runner,
        evidence_controller=controller,
        base_env={},
    )

    assert result["status"] == "completed"
    assert controller_calls == 0
    assert all("controller evidence" not in prompt for prompt in runner.proposer_prompts)


def test_reviewer_receives_its_controller_response_on_the_next_turn(tmp_path: Path) -> None:
    reviewer_prompts: list[str] = []

    def json_runner(prompt: str, **_kwargs: Any) -> dict[str, Any]:
        round_number = int(re.findall(r'"round_number":\s*(\d+)', prompt)[-1])
        if "### System\nreviewer" not in prompt:
            return {"proposal": f"round {round_number}"}
        reviewer_prompts.append(prompt)
        output = {
            "schema_version": "arc.llm.review_envelope.v1",
            "controller": {"message": "continue", "stop_requested": False},
            "proposer_messages": {"proposer_001": {"message": "revise"}},
            "review_payload": {"round": round_number},
        }
        if round_number == 1:
            output[EVIDENCE_REQUESTS_FIELD] = [
                {
                    "request_id": "reviewer_001-section",
                    "operation": "paper.section",
                    "arguments": {"paper_id": "0911.3380", "section": "S2"},
                }
            ]
        return output

    def controller(requests, *, round_number):
        assert requests[0].role == "reviewer"
        return (
            EvidenceResponse(
                requests[0].request_id,
                True,
                {"text": "normalized equation"},
                provenance={"source": "arc-paper", "section": "S2"},
            ),
        )

    result = run_proposers_reviewer_batch(
        _config(tmp_path, max_rounds=2),
        json_runner=json_runner,
        evidence_controller=controller,
        base_env={},
    )

    assert result["status"] == "completed"
    assert len(reviewer_prompts) == 2
    assert '"section": "S2"' in reviewer_prompts[1]
    assert '"text": "normalized equation"' in reviewer_prompts[1]


def test_malformed_worker_request_fails_the_loop_deterministically(tmp_path: Path) -> None:
    runner = EvidenceRunner(lambda _round_number: [{"request_id": "proposer_001-bad"}])

    result = run_proposers_reviewer_batch(
        _config(tmp_path, max_rounds=1),
        json_runner=runner,
        evidence_controller=lambda _requests, **_kwargs: (),
        base_env={},
    )

    assert result["status"] == "failed"
    assert "evidence operation is required" in result["loops"][0]["error"]


def test_final_round_request_is_not_resolved_without_a_followup_turn(tmp_path: Path) -> None:
    runner = EvidenceRunner(
        lambda _round_number: [
            {
                "request_id": "proposer_001-final",
                "operation": "paper.metadata",
                "arguments": {"paper_id": "0911.3380"},
            }
        ]
    )
    controller_calls = 0

    def controller(_requests, *, round_number):
        nonlocal controller_calls
        controller_calls += 1
        return ()

    result = run_proposers_reviewer_batch(
        _config(tmp_path, max_rounds=1),
        json_runner=runner,
        evidence_controller=controller,
        base_env={},
    )

    assert result["status"] == "completed"
    assert controller_calls == 0
    assert result["loops"][0]["evidence_rounds_completed"] == 0
    transcript = Path(result["loops"][0]["loop_root"]) / "transcript.jsonl"
    event = next(
        event
        for event in (json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines())
        if event.get("type") == "evidence_responses"
    )
    assert event["status"] == "no_followup_round"
    assert event["exchanges"][0]["response"]["ok"] is False


def test_controller_is_never_called_after_three_evidence_rounds(tmp_path: Path) -> None:
    runner = EvidenceRunner(
        lambda round_number: [
            {
                "request_id": f"proposer_001-round-{round_number}",
                "operation": "paper.search",
                "arguments": {"round": round_number},
            }
        ]
    )
    controller_rounds: list[int] = []

    def controller(requests, *, round_number):
        controller_rounds.append(round_number)
        return tuple(EvidenceResponse(request.request_id, True, {"round": round_number}) for request in requests)

    result = run_proposers_reviewer_batch(
        _config(tmp_path, max_rounds=5),
        json_runner=runner,
        evidence_controller=controller,
        base_env={},
    )

    assert result["status"] == "completed"
    assert controller_rounds == [1, 2, 3]
    assert result["loops"][0]["evidence_rounds_completed"] == 3
    assert result["loops"][0]["evidence_request_count"] == 5
    transcript = Path(result["loops"][0]["loop_root"]) / "transcript.jsonl"
    statuses = [
        event["status"]
        for event in (json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines())
        if event.get("type") == "evidence_responses"
    ]
    assert statuses == ["resolved", "resolved", "resolved", "round_limit_reached", "no_followup_round"]
