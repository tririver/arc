from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest

from arc_llm import runner as core_runner
from arc_llm.evidence import EVIDENCE_REQUESTS_FIELD, allow_evidence_requests, evidence_requests_schema
from arc_llm.json_schema import CodexSchemaError, to_provider_json_schema, validate_codex_strict_schema
from arc_llm.proposers_reviewer.runner import _batch_status, run_proposers_reviewer_batch
from arc_llm.providers.lifecycle import resolve_worker_call_timeout_seconds, run_process_group
from arc_llm.providers.base import LLMSchemaError, LLMWorkerCancelled, LLMWorkerError, LLMWorkerTimeout

from .test_proposers_reviewer_runner import FakeJsonRunner, _context_from_prompt, base_config


def test_evidence_schema_is_codex_strict_required() -> None:
    schema = allow_evidence_requests({"type": "object", "properties": {}})
    assert schema is not None
    assert EVIDENCE_REQUESTS_FIELD in schema["required"]
    assert evidence_requests_schema()["items"]["required"] == [
        "request_id",
        "operation",
        "arguments",
        "reason",
    ]
    validate_codex_strict_schema(to_provider_json_schema(schema))


def test_recursive_codex_validator_rejects_optional_nested_property() -> None:
    schema = to_provider_json_schema(
        {
            "type": "object",
            "required": ["nested"],
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"required_by_codex": {"type": "string"}},
                    "required": [],
                }
            },
        }
    )
    with pytest.raises(CodexSchemaError, match=r"nested.*required.*missing"):
        validate_codex_strict_schema(schema)


def test_worker_timeout_precedence_and_unlimited_default() -> None:
    assert resolve_worker_call_timeout_seconds(None, env={}, provider="codex-cli") is None
    assert resolve_worker_call_timeout_seconds(None, env={"ARC_LLM_TIMEOUT_SECONDS": "22"}, provider="codex-cli") == 22
    assert resolve_worker_call_timeout_seconds(
        None,
        env={"ARC_LLM_TIMEOUT_SECONDS": "22", "ARC_CODEX_TIMEOUT_SECONDS": "11"},
        provider="codex-cli",
    ) == 11
    assert resolve_worker_call_timeout_seconds(
        7,
        env={"ARC_CODEX_TIMEOUT_SECONDS": "11"},
        provider="codex-cli",
    ) == 7


def test_process_group_deadline_covers_blocked_stdin_delivery() -> None:
    started = time.monotonic()
    with pytest.raises(LLMWorkerTimeout, match="timed out"):
        run_process_group(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            input_text="x" * (10 * 1024 * 1024),
            env=os.environ,
            deadline=time.monotonic() + 0.2,
            poll_interval_seconds=0.02,
            terminate_grace_seconds=0.2,
        )
    assert time.monotonic() - started < 2


def test_process_group_allows_no_deadline() -> None:
    result = run_process_group(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
        input_text="unlimited",
        env=os.environ,
        deadline=None,
        poll_interval_seconds=0.02,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "unlimited"


def test_process_group_cancellation_remains_active_without_deadline() -> None:
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(LLMWorkerCancelled, match="cancelled"):
            run_process_group(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                input_text="",
                env=os.environ,
                deadline=None,
                cancel_check=cancelled.is_set,
                poll_interval_seconds=0.02,
                terminate_grace_seconds=0.2,
            )
    finally:
        timer.cancel()

    assert time.monotonic() - started < 2


def test_runner_passes_no_deadline_when_timeout_is_unset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Provider:
        def generate_text_result(self, prompt, *, deadline=None, cancel_check=None, **_kwargs):
            captured["deadline"] = deadline
            captured["cancel_check"] = cancel_check
            return core_runner.LLMProviderResponse(value="ok")

    monkeypatch.setattr(core_runner, "select_provider", lambda *_args, **_kwargs: Provider())

    assert core_runner.run_text("prompt", provider="codex-cli", env={}, process_chain=[]) == "ok"
    assert captured == {"deadline": None, "cancel_check": None}


def test_retry_loop_uses_one_total_monotonic_deadline(monkeypatch) -> None:
    clock = {"now": 100.0}

    class Provider:
        attempts = 0

        def generate_json(self, prompt, *, schema=None, model=None):
            self.attempts += 1
            clock["now"] += 6
            raise LLMWorkerError("transient")

    provider = Provider()
    monkeypatch.setattr(core_runner, "select_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(core_runner.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr("arc_llm.providers.lifecycle.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(core_runner.time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))

    with pytest.raises(LLMWorkerTimeout, match="timed out"):
        core_runner.run_json(
            "prompt",
            provider="codex-cli",
            env={},
            process_chain=[],
            timeout_seconds=10,
        )
    assert provider.attempts == 1


def test_auto_manual_fails_before_provider_attempts(monkeypatch) -> None:
    def should_not_select(*_args, **_kwargs):
        raise AssertionError("manual provider must not be invoked")

    monkeypatch.setattr(core_runner, "select_provider", should_not_select)

    with pytest.raises(core_runner.LLMNeedsLLM, match="resolved to manual"):
        core_runner.run_json("prompt", provider="auto", env={}, process_chain=[])


def test_schema_error_type_is_preserved_without_retry(monkeypatch) -> None:
    class Provider:
        attempts = 0

        def generate_json(self, prompt, *, schema=None, model=None):
            self.attempts += 1
            raise LLMSchemaError("invalid provider schema")

    provider = Provider()
    monkeypatch.setattr(core_runner, "select_provider", lambda *_args, **_kwargs: provider)

    with pytest.raises(LLMSchemaError, match="invalid provider schema") as caught:
        core_runner.run_json("prompt", provider="codex-cli", env={}, process_chain=[])

    assert caught.value.retryable is False
    assert provider.attempts == 1


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["completed"], "completed"),
        (["stopped"], "stopped"),
        (["completed", "failed"], "degraded"),
        (["degraded"], "degraded"),
        (["failed"], "failed"),
        (["cancelled"], "cancelled"),
    ],
)
def test_batch_status_matrix(statuses: list[str], expected: str) -> None:
    assert _batch_status([{"status": status} for status in statuses]) == expected


def test_auto_manual_batch_returns_needs_llm_before_creating_run(tmp_path) -> None:
    config = base_config(tmp_path, max_rounds=1)
    config["defaults"]["provider"] = "auto"
    config["defaults"].pop("model", None)
    for loop in config["loops"]:
        for worker in [*loop["proposers"], *loop["reviewers"]]:
            worker.pop("provider", None)
            worker.pop("model", None)

    result = run_proposers_reviewer_batch(config, base_env={}, process_chain=[])

    assert result["status"] == "needs_llm"
    assert result["llm_task"]["provider_resolved"] == "manual"
    assert result["loops"] == []
    assert not (tmp_path / "ideas/run_001").exists()


def test_batch_timeout_passes_none_for_env_resolution_and_honors_worker_override(tmp_path) -> None:
    def execute(config):
        seen: dict[str, float | None] = {}

        def runner(prompt, *, timeout_seconds, **kwargs):
            context = _context_from_prompt(prompt)
            seen[context["worker_id"]] = timeout_seconds
            if context["worker_id"].startswith("reviewer"):
                return {
                    "schema_version": "arc.llm.review_envelope.v1",
                    "controller": {"message": "reviewed", "stop_requested": False},
                    "proposer_messages": {
                        "proposer_001": {"message": "ok"},
                        "proposer_002": {"message": "ok"},
                    },
                    "review_payload": {"ok": True},
                }
            return {"ok": True}

        run_proposers_reviewer_batch(config, json_runner=runner, base_env={})
        return seen

    unspecified = base_config(tmp_path / "unspecified", max_rounds=1)
    assert execute(unspecified) == {
        "proposer_001": None,
        "proposer_002": None,
        "reviewer_001": None,
    }

    explicit = base_config(tmp_path / "explicit", max_rounds=1)
    explicit["worker_call_timeout_seconds"] = 12
    explicit["loops"][0]["proposers"][0]["worker_call_timeout_seconds"] = 4
    assert execute(explicit) == {
        "proposer_001": 4,
        "proposer_002": 12,
        "reviewer_001": 12,
    }


def test_in_progress_state_updates_while_worker_call_is_still_running(tmp_path) -> None:
    config = base_config(tmp_path, max_rounds=1)
    slow_started = threading.Event()
    release_slow = threading.Event()
    completed: list[dict] = []
    errors: list[BaseException] = []

    def runner(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        worker_id = context["worker_id"]
        if worker_id == "proposer_002":
            slow_started.set()
            if not release_slow.wait(timeout=5):
                raise RuntimeError("test did not release slow proposer")
        if worker_id.startswith("reviewer"):
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "ok"},
                    "proposer_002": {"message": "ok"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    def execute() -> None:
        try:
            completed.append(run_proposers_reviewer_batch(config, json_runner=runner, base_env={}))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    try:
        assert slow_started.wait(timeout=3)
        run_state_path = tmp_path / "ideas/run_001/state.json"
        loop_state_path = tmp_path / "ideas/run_001/loops/loop_001/state.json"
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
            loop_state = json.loads(loop_state_path.read_text(encoding="utf-8"))
            if (
                loop_state.get("completed_workers") == 1
                and run_state.get("completed_workers") == 1
            ):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("progress state did not record the completed fast proposer")

        for state in (run_state, loop_state):
            assert state["status"] == "running"
            assert state["phase"] == "proposers"
            assert state["loop_id"] == "loop_001"
            assert state["round_number"] == 1
            assert state["active_workers"] == 1
            assert state["completed_workers"] == 1
            assert state["failed_workers"] == 0
            assert state["updated_at"]
        assert run_state["loop_progress"]["loop_001"]["active_workers"] == 1
    finally:
        release_slow.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert completed[0]["status"] == "completed"
    terminal_loop_state = json.loads(loop_state_path.read_text(encoding="utf-8"))
    assert terminal_loop_state["phase"] == "finished"
    assert terminal_loop_state["active_workers"] == 0


def test_partial_proposer_failure_is_degraded_without_fake_output(tmp_path) -> None:
    config = base_config(tmp_path, max_rounds=1)

    def runner(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"] == "proposer_001":
            raise RuntimeError("provider unavailable")
        if context["worker_id"].startswith("reviewer"):
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "failed"},
                    "proposer_002": {"message": "ok"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    result = run_proposers_reviewer_batch(config, json_runner=runner, base_env={})

    assert result["status"] == "degraded"
    loop = result["loops"][0]
    assert loop["status"] == "degraded"
    assert loop["worker_failures"][0]["call_status"] == "provider_error"
    failed_output = tmp_path / "ideas/run_001/loops/loop_001/rounds/round_001/proposer_outputs/proposer_001.json"
    assert not failed_output.exists()


def test_typed_timeout_is_preserved_in_worker_failure_status(tmp_path) -> None:
    config = base_config(tmp_path, max_rounds=1)

    def runner(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"] == "proposer_001":
            raise LLMWorkerTimeout("kimi acp timed out")
        if context["worker_id"].startswith("reviewer"):
            return {
                "schema_version": "arc.llm.review_envelope.v1",
                "controller": {"message": "reviewed", "stop_requested": False},
                "proposer_messages": {
                    "proposer_001": {"message": "failed"},
                    "proposer_002": {"message": "ok"},
                },
                "review_payload": {"ok": True},
            }
        return {"ok": True}

    result = run_proposers_reviewer_batch(config, json_runner=runner, base_env={})

    assert result["status"] == "degraded"
    assert result["loops"][0]["worker_failures"][0]["call_status"] == "timeout"


def test_reviewer_failure_is_failed_and_progress_is_append_only(tmp_path) -> None:
    config = base_config(tmp_path, max_rounds=1)
    events: list[dict[str, object]] = []

    def runner(prompt, **kwargs):
        context = _context_from_prompt(prompt)
        if context["worker_id"].startswith("reviewer"):
            raise RuntimeError("reviewer unavailable")
        return {"ok": True}

    result = run_proposers_reviewer_batch(
        config,
        json_runner=runner,
        base_env={},
        progress_callback=events.append,
    )

    assert result["status"] == "failed"
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"
    progress_path = tmp_path / "ideas/run_001/progress.jsonl"
    persisted = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert len(persisted) == len(events)


def test_cancel_before_scheduling_is_cancelled(tmp_path) -> None:
    result = run_proposers_reviewer_batch(
        base_config(tmp_path, max_rounds=1),
        json_runner=FakeJsonRunner(),
        base_env={},
        cancel_check=lambda: True,
    )
    assert result["status"] == "cancelled"
    assert result["loops"][0]["status"] == "cancelled"
