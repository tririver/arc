from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from arc_llm import runner
from arc_llm.progress_prompt import RUNTIME_PROGRESS_CONTRACT_MARKER, RUNTIME_PROGRESS_SESSION_MARKER
from arc_llm.runner import run_json_result
from arc_llm.sessions import LLMSessionManager
from arc_llm.usage import LLMProviderResponse, LLMUsage
from arc_llm.call_checkpoint import LLMCallRetryExhausted
from arc_llm.providers.base import LLMSubmissionState, LLMWorkerTimeout


class CountingProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate_json_result(self, prompt, **_kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        return LLMProviderResponse(
            {"ok": True},
            usage=LLMUsage(input_tokens=4, output_tokens=1),
            native_session_id="native-1",
        )


def _kwargs(tmp_path, manager, key="logical-1"):
    existing = manager.get_existing("chapter/translation")
    generation = existing.generation if existing is not None else 1
    return {
        "schema": {"type": "object", "required": ["ok"]},
        "provider": "codex-cli",
        "model": "m",
        "env": {"ARC_PAPER_ACCESS": "none"},
        "process_chain": [],
        "session_policy": "stateful",
        "session_manager": manager,
        "session_key": "chapter/translation",
        "artifact_dir": tmp_path / "artifacts",
        "call_label": key,
        "idempotency_key": key,
        "progress_contract_scope": "session",
        "initial_native_authorization": (
            str((tmp_path / "control-ledger.json").resolve()),
            "chapter/translation",
            "segment-1",
            generation,
            key,
        ),
    }


def _resume_authorization(tmp_path, key="logical-1"):
    return (
        str((tmp_path / "control-ledger.json").resolve()),
        "chapter/translation",
        "segment-1",
        1,
        key,
    )


def test_stateful_idempotency_replays_after_record_turn_crash(tmp_path, monkeypatch):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    real_validated = runner.record_validated
    monkeypatch.setattr(runner, "record_validated", lambda _prepared: (_ for _ in ()).throw(RuntimeError("crash")))

    with pytest.raises(RuntimeError, match="crash"):
        run_json_result("prompt", **_kwargs(tmp_path, manager))
    monkeypatch.setattr(runner, "record_validated", real_validated)
    outcome = run_json_result("prompt", **_kwargs(tmp_path, manager))

    assert provider.calls == 1
    assert manager.turn_count("chapter/translation") == 1
    assert outcome.logical_receipt["replayed"] is True


def test_stateful_idempotency_replays_after_response_before_turn(tmp_path, monkeypatch):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    real_record_turn = manager.record_turn
    failed = False

    def crash_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("turn crash")
        return real_record_turn(*args, **kwargs)

    monkeypatch.setattr(manager, "record_turn", crash_once)
    with pytest.raises(RuntimeError, match="turn crash"):
        run_json_result("prompt", **_kwargs(tmp_path, manager))
    outcome = run_json_result("prompt", **_kwargs(tmp_path, manager))

    assert provider.calls == 1
    assert outcome.logical_receipt["replayed"] is True


def test_stateful_idempotency_replays_after_response_persist_crash(tmp_path, monkeypatch):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    real_record_response = runner.record_response
    failed = False

    def persist_then_crash(prepared, response):
        nonlocal failed
        real_record_response(prepared, response)
        if not failed:
            failed = True
            raise RuntimeError("response checkpoint crash")

    monkeypatch.setattr(runner, "record_response", persist_then_crash)
    with pytest.raises(RuntimeError, match="response checkpoint crash"):
        run_json_result("prompt", **_kwargs(tmp_path, manager))
    outcome = run_json_result("prompt", **_kwargs(tmp_path, manager))

    assert provider.calls == 1
    assert outcome.logical_receipt["replayed"] is True


def test_successful_call_replays_when_caller_checkpoint_was_not_written(tmp_path, monkeypatch):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)

    first = run_json_result("prompt", **_kwargs(tmp_path, manager))
    second = run_json_result("prompt", **_kwargs(tmp_path, manager))

    assert first.logical_receipt["replayed"] is False
    assert second.logical_receipt["replayed"] is True
    assert provider.calls == 1
    replay_attempt = second.call_record["attempts"][0]
    assert replay_attempt["status"] == "replayed"
    replay_record = json.loads(
        ((tmp_path / "artifacts") / replay_attempt["diagnostic_path"]).read_text()
    )
    assert replay_record["outcome"] == "replayed"
    assert replay_record["submission_state"] == "not_submitted"
    assert replay_record["parsed_response_candidates"][0]["source"] == (
        "checkpoint_replayed_response"
    )
    replay_record_path = (tmp_path / "artifacts") / replay_attempt["diagnostic_path"]
    timeline = (replay_record_path.parent / replay_record["timeline"]["path"]).read_text()
    assert '"event": "checkpoint_replayed"' in timeline


def test_paid_stateful_logical_key_replays_before_rebuilt_prompt_digest(tmp_path, monkeypatch):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    kwargs = _kwargs(tmp_path, manager)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: run_json_result("prompt", **kwargs), range(2)))
    assert provider.calls == 1
    assert all(outcome.value["ok"] for outcome in outcomes)
    replay = run_json_result("rebuilt delta after receipt advanced", **kwargs)
    assert replay.logical_receipt["replayed"] is True
    assert provider.calls == 1


def test_receipt_persisted_before_caller_acceptance_replays_rebuilt_stream(
    tmp_path, monkeypatch,
):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    real_validated = runner.record_validated

    def crash_after_receipt(prepared):
        real_validated(prepared)
        raise RuntimeError("caller ledger crash")

    monkeypatch.setattr(runner, "record_validated", crash_after_receipt)
    with pytest.raises(RuntimeError, match="caller ledger crash"):
        run_json_result("generation bootstrap", **_kwargs(tmp_path, manager, "turn-crash"))
    assert manager.turn_count("chapter/translation") == 1

    monkeypatch.setattr(runner, "record_validated", real_validated)
    replay = run_json_result(
        "delta rebuilt after restart", **_kwargs(tmp_path, manager, "turn-crash")
    )

    assert replay.value == {"ok": True}
    assert replay.logical_receipt["replayed"] is True
    assert provider.calls == 1


def test_session_progress_contract_bootstrap_delta_and_rotate(tmp_path, monkeypatch):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)

    run_json_result("one", **_kwargs(tmp_path, manager, "turn-1"))
    run_json_result("two", **_kwargs(tmp_path, manager, "turn-2"))
    manager.rotate("chapter/translation", reason="rollover")
    run_json_result("three", **_kwargs(tmp_path, manager, "turn-3"))

    assert RUNTIME_PROGRESS_CONTRACT_MARKER in provider.prompts[0]
    assert RUNTIME_PROGRESS_SESSION_MARKER in provider.prompts[1]
    assert RUNTIME_PROGRESS_CONTRACT_MARKER not in provider.prompts[1]
    assert RUNTIME_PROGRESS_CONTRACT_MARKER in provider.prompts[2]


def test_json_result_exposes_receipt_usage_generation_and_prompt_bytes(tmp_path, monkeypatch):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)

    outcome = run_json_result("prompt", **_kwargs(tmp_path, manager))

    assert outcome.value == {"ok": True}
    assert outcome.usage.status == "known"
    assert outcome.generation == 1
    assert outcome.prompt_bytes and outcome.prompt_bytes > len("prompt")
    assert outcome.logical_receipt["idempotency_key"] == "logical-1"
    assert outcome.call_record["logical_receipt"] == outcome.logical_receipt


def test_submitted_timeout_only_explicitly_resumes_native_session(tmp_path, monkeypatch):
    class TimeoutThenReconcileProvider(CountingProvider):
        def generate_json_result(self, prompt, **kwargs):
            self.calls += 1
            self.prompts.append(prompt)
            assert kwargs["session"].native_session_id == "native-existing"
            assert tuple(kwargs["initial_native_authorization"]) == (
                str((tmp_path / "control-ledger.json").resolve()),
                "chapter/translation", "segment-1", 1, "supervised-turn",
            )
            if self.calls == 1:
                assert "supervised_native_resume" not in kwargs
                raise LLMWorkerTimeout(
                    "lost response",
                    submission_state=LLMSubmissionState.SUBMITTED,
                )
            assert tuple(kwargs["supervised_native_resume"]) == _resume_authorization(
                tmp_path, "supervised-turn"
            )
            return LLMProviderResponse(
                {"ok": True},
                usage=LLMUsage(input_tokens=2, output_tokens=1),
                native_session_id="native-existing",
            )

    provider = TimeoutThenReconcileProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    manager.get_or_create(
        key="chapter/translation",
        provider="codex-cli",
        model="m",
        runtime_fingerprint=runner._runtime_fp(
            provider_used="codex-cli", model="m", model_tier=None,
                env={"ARC_PAPER_ACCESS": "none"}, process_chain=[],
        ),
        metadata={
            "arc_runtime_capabilities": runner._runtime_capabilities(
                {"ARC_PAPER_ACCESS": "none"}
            ),
            "arc_runtime_manifest_version": runner.RUNTIME_MANIFEST_VERSION,
        },
    )
    manager.update_native_session_id("chapter/translation", "native-existing")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    kwargs = _kwargs(tmp_path, manager, "supervised-turn")

    with pytest.raises(LLMWorkerTimeout):
        run_json_result("original paid request", **kwargs)
    assert provider.calls == 1

    with pytest.raises(LLMCallRetryExhausted):
        run_json_result("original paid request", **kwargs)
    assert provider.calls == 1

    outcome = run_json_result(
        "rebuilt stateful stream",
        **kwargs,
        supervised_native_resume=_resume_authorization(tmp_path, "supervised-turn"),
    )
    assert outcome.value == {"ok": True}
    assert provider.calls == 2
    assert "Supervised native-session recovery" in provider.prompts[-1]
    assert "original paid request" not in provider.prompts[-1]
    attempt_record = json.loads(
        (
            (tmp_path / "artifacts")
            / outcome.call_record["attempts"][0]["diagnostic_path"]
        ).read_text()
    )
    assert attempt_record["checkpoint_binding"]["native_resume_authorization"] == {
        "control_address": str((tmp_path / "control-ledger.json").resolve()),
        "session_key": "chapter/translation",
        "logical_unit": "segment-1",
        "generation": 1,
        "idempotency_key": "supervised-turn",
    }

    replay = run_json_result(
        "rebuilt stateful stream",
        **kwargs,
        supervised_native_resume=_resume_authorization(tmp_path, "supervised-turn"),
    )
    assert replay.logical_receipt["replayed"] is True
    assert provider.calls == 2


@pytest.mark.parametrize(
    "authorization",
    [
        True,
        ("/tmp/control", "chapter/translation", "segment-1", 1),
    ],
)
@pytest.mark.parametrize("entrypoint", [run_json_result, runner.run_text_result])
def test_public_resume_refuses_boolean_or_partial_authorization(
    tmp_path, monkeypatch, authorization, entrypoint,
):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)

    kwargs = _kwargs(tmp_path, manager)
    if entrypoint is runner.run_text_result:
        kwargs.pop("schema")
    with pytest.raises(ValueError, match="complete five-field authorization"):
        entrypoint(
            "must not run", **kwargs, supervised_native_resume=authorization,
        )

    assert provider.calls == 0


def test_public_resume_refuses_full_tuple_that_differs_from_initial_authorization(
    tmp_path, monkeypatch,
):
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    kwargs = _kwargs(tmp_path, manager, "exact-turn")
    changed = list(kwargs["initial_native_authorization"])
    changed[2] = "different-logical-unit"

    with pytest.raises(ValueError, match="exact complete initial"):
        run_json_result(
            "must not run", **kwargs, supervised_native_resume=tuple(changed),
        )
    assert provider.calls == 0


def test_text_provider_receives_complete_initial_authorization_contract(
    tmp_path, monkeypatch,
):
    captured = []

    class TextProvider:
        def generate_text_result(self, prompt, **kwargs):
            captured.append(kwargs["initial_native_authorization"])
            return LLMProviderResponse("ok", native_session_id="native-text")

    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(
        runner, "select_provider", lambda *_args, **_kwargs: TextProvider(),
    )
    kwargs = _kwargs(tmp_path, manager, "text-turn")
    kwargs.pop("schema")
    outcome = runner.run_text_result("text", **kwargs)

    assert outcome.value == "ok"
    assert tuple(captured[0]) == (
        str((tmp_path / "control-ledger.json").resolve()),
        "chapter/translation", "segment-1", 1, "text-turn",
    )


def test_stateful_runner_creates_missing_session_at_authorized_generation(
    tmp_path, monkeypatch,
) -> None:
    seen_generations: list[int] = []

    class GenerationProvider(CountingProvider):
        def generate_json_result(self, prompt, **kwargs):
            seen_generations.append(kwargs["session"].generation)
            return super().generate_json_result(prompt, **kwargs)

    provider = GenerationProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    kwargs = _kwargs(tmp_path, manager, "generation-4")
    authorization = list(kwargs["initial_native_authorization"])
    authorization[3] = 4
    kwargs["initial_native_authorization"] = tuple(authorization)

    outcome = run_json_result("fresh generation", **kwargs)

    assert outcome.generation == 4
    assert seen_generations == [4]
    created = manager.get_existing("chapter/translation")
    assert created is not None
    assert created.generation == 4
    assert created.provider == "codex-cli"
    assert created.model == "m"


def test_stateful_runner_fails_closed_on_authorized_generation_mismatch(
    tmp_path, monkeypatch,
) -> None:
    provider = CountingProvider()
    manager = LLMSessionManager(tmp_path / "sessions")
    runtime_fp = runner._runtime_fp(
        provider_used="codex-cli", model="m", model_tier=None,
        env={"ARC_PAPER_ACCESS": "none"}, process_chain=[],
    )
    manager.get_or_create(
        key="chapter/translation", provider="codex-cli", model="m",
        runtime_fingerprint=runtime_fp,
    )
    manager.rotate("chapter/translation", reason="generation 2")
    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: provider)
    kwargs = _kwargs(tmp_path, manager, "requires-generation-3")
    authorization = list(kwargs["initial_native_authorization"])
    authorization[3] = 3
    kwargs["initial_native_authorization"] = tuple(authorization)

    with pytest.raises(runner.LLMTaskError) as exc_info:
        run_json_result("must not submit", **kwargs)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "expected 3, found 2" in str(exc_info.value.__cause__)
    assert provider.calls == 0
    assert manager.get_existing("chapter/translation").generation == 2
