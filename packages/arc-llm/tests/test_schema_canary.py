from __future__ import annotations

import json
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import validate

from arc_llm import runner
from arc_llm.proposers_reviewer import runner as proposer_runner
from arc_llm.proposers_reviewer.config import (
    CacheGuardOptions,
    OutputRecoveryOptions,
    PromptConfig,
    WorkerConfig,
)
from arc_llm.providers.base import (
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerError,
)
from arc_llm.schema_cache import schema_hash
from arc_llm.schema_canary import (
    SCHEMA_CANARY_RECEIPT_SCHEMA,
    SchemaCanaryBlocked,
    SchemaCanaryIdentity,
    run_schema_canary,
    schema_canary_receipt_path,
)
from arc_llm.sessions import LLMSessionManager
from arc_llm.usage import LLMProviderResponse, ResponseCandidateMaterial


def _identity(**changes: str | None) -> SchemaCanaryIdentity:
    values: dict[str, str | None] = {
        "provider_id": "codex-cli",
        "runtime_fingerprint": "runtime-a",
        "effective_model": "model-a",
        "effective_schema_sha256": "a" * 64,
        "transport_mode": "strict",
    }
    values.update(changes)
    return SchemaCanaryIdentity(**values)  # type: ignore[arg-type]


def _run_process_canary(
    root: str,
    identity_values: dict[str, str | None],
    counter,
    counter_lock,
    first_started,
    release_first,
    results,
) -> None:
    identity = SchemaCanaryIdentity(**identity_values)

    def invoke() -> int:
        with counter_lock:
            counter.value += 1
            sequence = counter.value
        if sequence == 1:
            first_started.set()
            release_first.wait(5)
        return sequence

    try:
        results.put(("ok", run_schema_canary(root=Path(root), identity=identity, invoke=invoke)))
    except BaseException as exc:  # pragma: no cover - reported to the parent process
        results.put(("error", type(exc).__name__))


def _run_process_blocked(root, identity_values, counter, results) -> None:
    identity = SchemaCanaryIdentity(**identity_values)

    def invoke() -> None:
        counter.value += 1

    try:
        run_schema_canary(root=Path(root), identity=identity, invoke=invoke)
    except BaseException as exc:  # pragma: no cover - reported to the parent process
        results.put(type(exc).__name__)


def _run_process_crash(root, identity_values, started) -> None:
    identity = SchemaCanaryIdentity(**identity_values)

    def invoke() -> None:
        started.set()
        os._exit(17)

    run_schema_canary(root=Path(root), identity=identity, invoke=invoke)


def test_successful_first_real_task_releases_24_followers(tmp_path):
    identity = _identity()
    counter = 0
    counter_lock = threading.Lock()
    first_started = threading.Event()
    release_first = threading.Event()
    followers_started = threading.Event()
    release_followers = threading.Event()
    follower_count = 0

    def invoke() -> int:
        nonlocal counter, follower_count
        with counter_lock:
            counter += 1
            sequence = counter
        if sequence == 1:
            first_started.set()
            assert release_first.wait(5)
        else:
            with counter_lock:
                follower_count += 1
                if follower_count == 23:
                    followers_started.set()
            assert release_followers.wait(5)
        return sequence

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(
                run_schema_canary,
                root=tmp_path,
                identity=identity,
                invoke=invoke,
            )
            for _ in range(24)
        ]
        assert first_started.wait(2)
        assert counter == 1
        release_first.set()
        assert followers_started.wait(2)
        assert follower_count == 23
        release_followers.set()
        assert sorted(future.result(timeout=5) for future in futures) == list(range(1, 25))

    assert counter == 24
    receipt = json.loads(schema_canary_receipt_path(tmp_path, identity).read_text())
    validate(receipt, SCHEMA_CANARY_RECEIPT_SCHEMA)
    assert receipt["status"] == "proven"
    assert receipt["identity"] == identity.to_json()


def test_deterministic_rejection_submits_once_and_blocks_23_followers(tmp_path):
    identity = _identity()
    counter = 0
    counter_lock = threading.Lock()
    first_started = threading.Event()
    release_first = threading.Event()

    def reject() -> None:
        nonlocal counter
        with counter_lock:
            counter += 1
        first_started.set()
        assert release_first.wait(5)
        raise LLMWorkerError(
            "provider rejected structured-output schema",
            retryable=False,
            category=LLMFailureCategory.SCHEMA,
            submission_state=LLMSubmissionState.SUBMITTED,
        )

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(
                run_schema_canary,
                root=tmp_path,
                identity=identity,
                invoke=reject,
            )
            for _ in range(24)
        ]
        assert first_started.wait(2)
        assert counter == 1
        release_first.set()
        errors = []
        for future in futures:
            with pytest.raises(LLMWorkerError) as caught:
                future.result(timeout=5)
            errors.append(caught.value)

    assert counter == 1
    assert sum(not isinstance(error, SchemaCanaryBlocked) for error in errors) == 1
    blocked = [error for error in errors if isinstance(error, SchemaCanaryBlocked)]
    assert len(blocked) == 23
    assert all(error.submission_state == LLMSubmissionState.NOT_SUBMITTED for error in blocked)
    receipt = json.loads(schema_canary_receipt_path(tmp_path, identity).read_text())
    validate(receipt, SCHEMA_CANARY_RECEIPT_SCHEMA)
    assert receipt["status"] == "rejected"
    assert receipt["failure"] == {
        "category": "schema",
        "submission_state": "submitted",
    }


def test_retryable_failure_does_not_prove_or_block_followers(tmp_path):
    identity = _identity()
    counter = 0
    counter_lock = threading.Lock()

    def transient_then_success() -> int:
        nonlocal counter
        with counter_lock:
            counter += 1
            sequence = counter
        if sequence == 1:
            raise LLMWorkerError(
                "temporary transport failure",
                retryable=True,
                category=LLMFailureCategory.TRANSPORT,
                submission_state=LLMSubmissionState.UNKNOWN,
            )
        return sequence

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(
                run_schema_canary,
                root=tmp_path,
                identity=identity,
                invoke=transient_then_success,
            )
            for _ in range(24)
        ]
        failures = 0
        values = []
        for future in futures:
            try:
                values.append(future.result(timeout=5))
            except LLMWorkerError as exc:
                assert exc.retryable is True
                failures += 1

    assert failures == 1
    assert len(values) == 23
    assert counter == 24
    assert json.loads(schema_canary_receipt_path(tmp_path, identity).read_text())["status"] == "proven"


def test_request_specific_invalid_request_does_not_reject_identity(tmp_path):
    identity = _identity()
    submissions = 0

    def reject_prompt() -> None:
        nonlocal submissions
        submissions += 1
        raise LLMWorkerError(
            "prompt exceeds provider context length",
            retryable=False,
            category=LLMFailureCategory.INVALID_REQUEST,
            submission_state=LLMSubmissionState.SUBMITTED,
        )

    for _ in range(2):
        with pytest.raises(LLMWorkerError):
            run_schema_canary(root=tmp_path, identity=identity, invoke=reject_prompt)

    assert submissions == 2
    assert schema_canary_receipt_path(tmp_path, identity).exists() is False


def test_cancelled_follower_is_not_submitted(tmp_path):
    identity = _identity()
    first_started = threading.Event()
    release_first = threading.Event()

    def first() -> str:
        first_started.set()
        assert release_first.wait(5)
        return "ok"

    with ThreadPoolExecutor(max_workers=1) as executor:
        owner = executor.submit(
            run_schema_canary,
            root=tmp_path,
            identity=identity,
            invoke=first,
        )
        assert first_started.wait(2)
        with pytest.raises(LLMWorkerError) as caught:
            run_schema_canary(
                root=tmp_path,
                identity=identity,
                invoke=lambda: pytest.fail("cancelled follower must not submit"),
                cancel_check=lambda: True,
            )
        release_first.set()
        assert owner.result(timeout=5) == "ok"

    assert caught.value.submission_state == LLMSubmissionState.NOT_SUBMITTED


def test_crash_before_commit_leaves_identity_unproven(monkeypatch, tmp_path):
    from arc_llm import schema_canary

    identity = _identity()
    submissions = 0
    original_write = schema_canary._write_receipt

    class SimulatedCrash(BaseException):
        pass

    def invoke() -> str:
        nonlocal submissions
        submissions += 1
        return "response"

    def crash_before_write(*args, **kwargs):
        raise SimulatedCrash()

    monkeypatch.setattr(schema_canary, "_write_receipt", crash_before_write)
    with pytest.raises(SimulatedCrash):
        run_schema_canary(root=tmp_path, identity=identity, invoke=invoke)
    assert schema_canary_receipt_path(tmp_path, identity).exists() is False

    monkeypatch.setattr(schema_canary, "_write_receipt", original_write)
    assert run_schema_canary(root=tmp_path, identity=identity, invoke=invoke) == "response"
    assert submissions == 2


def test_crash_after_commit_keeps_proof_for_followers(monkeypatch, tmp_path):
    from arc_llm import schema_canary

    identity = _identity()
    submissions = 0
    original_release = schema_canary._release_lock

    class SimulatedCrash(BaseException):
        pass

    def invoke() -> str:
        nonlocal submissions
        submissions += 1
        return "response"

    def release_then_crash(handle):
        original_release(handle)
        raise SimulatedCrash()

    monkeypatch.setattr(schema_canary, "_release_lock", release_then_crash)
    with pytest.raises(SimulatedCrash):
        run_schema_canary(root=tmp_path, identity=identity, invoke=invoke)
    assert json.loads(schema_canary_receipt_path(tmp_path, identity).read_text())["status"] == "proven"

    monkeypatch.setattr(schema_canary, "_release_lock", original_release)
    assert run_schema_canary(root=tmp_path, identity=identity, invoke=invoke) == "response"
    assert submissions == 2


@pytest.mark.skipif(multiprocessing.get_start_method() == "spawn", reason="requires inherited sync primitives")
def test_processes_racing_for_same_identity_launch_one_canary(tmp_path):
    context = multiprocessing.get_context("fork")
    identity = _identity()
    counter = context.Value("i", 0)
    counter_lock = context.Lock()
    first_started = context.Event()
    release_first = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_run_process_canary,
            args=(
                str(tmp_path),
                identity.to_json(),
                counter,
                counter_lock,
                first_started,
                release_first,
                results,
            ),
        )
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    assert first_started.wait(2)
    assert counter.value == 1
    release_first.set()
    for process in processes:
        process.join(5)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=1) for _ in processes]
    assert all(status == "ok" for status, _ in outcomes)
    assert counter.value == len(processes)


@pytest.mark.skipif(multiprocessing.get_start_method() == "spawn", reason="requires fork")
def test_rejected_receipt_blocks_after_process_restart(tmp_path):
    context = multiprocessing.get_context("fork")
    identity = _identity()

    def reject() -> None:
        raise LLMWorkerError(
            "schema rejected",
            retryable=False,
            category=LLMFailureCategory.SCHEMA,
            submission_state=LLMSubmissionState.SUBMITTED,
        )

    with pytest.raises(LLMWorkerError):
        run_schema_canary(root=tmp_path, identity=identity, invoke=reject)

    counter = context.Value("i", 0)
    results = context.Queue()
    process = context.Process(
        target=_run_process_blocked,
        args=(str(tmp_path), identity.to_json(), counter, results),
    )
    process.start()
    process.join(5)

    assert process.exitcode == 0
    assert results.get(timeout=1) == "SchemaCanaryBlocked"
    assert counter.value == 0


@pytest.mark.skipif(multiprocessing.get_start_method() == "spawn", reason="requires fork")
def test_hard_killed_owner_releases_lock_without_proof(tmp_path):
    context = multiprocessing.get_context("fork")
    identity = _identity()
    started = context.Event()
    process = context.Process(
        target=_run_process_crash,
        args=(str(tmp_path), identity.to_json(), started),
    )
    process.start()
    assert started.wait(2)
    process.join(5)

    assert process.exitcode == 17
    assert schema_canary_receipt_path(tmp_path, identity).exists() is False
    assert run_schema_canary(
        root=tmp_path,
        identity=identity,
        invoke=lambda: "recovered",
    ) == "recovered"


def test_identity_components_invalidate_proof_and_do_not_cross_block(tmp_path):
    base = _identity()
    identities = [
        base,
        _identity(provider_id="claude-cli"),
        _identity(runtime_fingerprint="runtime-b"),
        _identity(effective_model="model-b"),
        _identity(effective_schema_sha256="b" * 64),
        _identity(transport_mode="prompt"),
    ]
    entered = set()
    entered_lock = threading.Lock()
    release = threading.Event()

    def invoke(identity_sha256: str) -> str:
        with entered_lock:
            entered.add(identity_sha256)
        assert release.wait(5)
        return identity_sha256

    with ThreadPoolExecutor(max_workers=len(identities)) as executor:
        futures = [
            executor.submit(
                run_schema_canary,
                root=tmp_path,
                identity=identity,
                invoke=lambda current=identity: invoke(current.sha256),
            )
            for identity in identities
        ]
        for _ in range(100):
            with entered_lock:
                if len(entered) == len(identities):
                    break
            threading.Event().wait(0.01)
        assert len(entered) == len(identities)
        release.set()
        assert {future.result(timeout=5) for future in futures} == {
            identity.sha256 for identity in identities
        }
    assert len(list((tmp_path / "schema-canaries").glob("*.json"))) == len(identities)


def test_runner_passes_t03_transport_mode_to_shared_admission(monkeypatch, tmp_path):
    captured: list[SchemaCanaryIdentity] = []
    canary_root = tmp_path / "run-root"

    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            return LLMProviderResponse({"ok": True})

    def capture_canary(*, root, identity, invoke, cancel_check=None):
        assert root == canary_root
        captured.append(identity)
        return invoke()

    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: Provider())
    monkeypatch.setattr(runner, "run_schema_canary", capture_canary)
    closed_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }
    open_schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }

    for schema in (closed_schema, open_schema):
        runner.run_json(
            "prompt",
            schema=schema,
            provider="codex-cli",
            model="model-a",
            env={"ARC_HOME": str(tmp_path / "arc-home")},
            process_chain=[],
            schema_canary_root=canary_root,
        )

    assert [identity.transport_mode for identity in captured] == ["strict", "prompt"]
    assert captured[0].effective_schema_sha256 == schema_hash(closed_schema)
    assert captured[1].effective_schema_sha256 == schema_hash(open_schema)


def test_proposer_default_and_custom_runners_share_batch_canary_root(
    monkeypatch, tmp_path
):
    batch_root = tmp_path / "batch-run"
    cache_warnings_path = batch_root / "cache_warnings.jsonl"
    worker = WorkerConfig(
        id="proposer-1",
        prompt=PromptConfig(system="system", template="template"),
        output_schema={"type": "object"},
        provider="manual",
        model=None,
        model_tier=None,
        runtime={},
        evidence_enabled=False,
        worker_idle_timeout_seconds=None,
    )
    session_manager = LLMSessionManager(tmp_path / "sessions")
    prefix_limiter = proposer_runner.PrefixConcurrencyLimiter(24)
    cache_guard = CacheGuardOptions(
        enabled=False,
        mode="warn",
        warmup_calls=1,
        min_cached_input_ratio=0.0,
    )
    output_recovery = OutputRecoveryOptions(
        enabled=True,
        mode="warn",
        allow_natural_language=True,
        schema_violation_policy="peer_visible",
        schema_formatter_enabled=True,
    )
    roots: list[Path] = []

    def fake_builtin_run_json(prompt, **kwargs):
        roots.append(kwargs["schema_canary_root"])
        return {"ok": True}

    def custom_json_runner(prompt, *, schema_canary_root, **kwargs):
        roots.append(schema_canary_root)
        return {"ok": True}

    monkeypatch.setattr(proposer_runner, "run_json", fake_builtin_run_json)
    common = {
        "prompt": "prompt",
        "worker": worker,
        "base_env": {},
        "process_chain": [],
        "session_policy": "stateless",
        "session_manager": session_manager,
        "session_key": "loop/proposer-1",
        "artifact_dir": batch_root / "loops" / "loop" / "llm_calls" / "proposer-1",
        "prefix_limiter": prefix_limiter,
        "static_prefix": None,
        "cache_guard": cache_guard,
        "cache_warnings_path": cache_warnings_path,
        "output_recovery": output_recovery,
    }

    proposer_runner._call_json_runner(
        None,
        call_label="default-call",
        **common,
    )
    proposer_runner._call_json_runner(
        custom_json_runner,
        call_label="custom-call",
        **common,
    )

    assert roots == [batch_root, batch_root]


def test_direct_json_call_with_artifacts_does_not_enable_batch_canary(monkeypatch, tmp_path):
    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            return LLMProviderResponse({"ok": True})

    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: Provider())
    monkeypatch.setattr(
        runner,
        "run_schema_canary",
        lambda **_kwargs: pytest.fail("direct call must not use a batch canary"),
    )

    result = runner.run_json(
        "prompt",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        provider="codex-cli",
        model="model-a",
        env={"ARC_HOME": str(tmp_path / "arc-home")},
        process_chain=[],
        artifact_dir=tmp_path / "call-artifacts",
    )

    assert result["ok"] is True


def test_deferred_invalid_terminal_does_not_prove_schema_canary(monkeypatch, tmp_path):
    error = LLMWorkerError(
        "terminal strict parse error",
        category=LLMFailureCategory.OUTPUT_INVALID,
        submission_state=LLMSubmissionState.SUBMITTED,
    )

    class Provider:
        def generate_json_result(self, prompt, *, defer_output_errors=False, **kwargs):
            assert defer_output_errors is True
            return LLMProviderResponse(
                {},
                candidate_material=(
                    ResponseCandidateMaterial(
                        "generic.provider_value", 0, value={"wrong": True}
                    ),
                ),
                deferred_output_error=error,
            )

    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: Provider())
    root = tmp_path / "batch-root"
    with pytest.raises(runner.LLMTaskError, match="terminal strict parse error"):
        runner.run_json(
            "prompt",
            schema={
                "type": "object", "additionalProperties": False,
                "required": ["ok"], "properties": {"ok": {"type": "boolean"}},
            },
            provider="codex-cli", model="model-a",
            env={"ARC_HOME": str(tmp_path / "arc-home")}, process_chain=[],
            artifact_dir=tmp_path / "call", schema_canary_root=root,
        )
    receipts = [json.loads(path.read_text()) for path in (root / "schema-canaries").glob("*.json")]
    assert not any(receipt.get("status") == "proven" for receipt in receipts)


def test_deferred_terminal_with_valid_earlier_candidate_proves_canary(monkeypatch, tmp_path):
    class Provider:
        def generate_json_result(self, prompt, *, defer_output_errors=False, **kwargs):
            assert defer_output_errors is True
            return LLMProviderResponse(
                {},
                candidate_material=(
                    ResponseCandidateMaterial(
                        "codex.completed_message", 0, value={"ok": True}
                    ),
                ),
                deferred_output_error=LLMWorkerError(
                    "terminal strict parse error",
                    category=LLMFailureCategory.OUTPUT_INVALID,
                    submission_state=LLMSubmissionState.SUBMITTED,
                ),
            )

    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: Provider())
    root = tmp_path / "batch-root"
    result = runner.run_json(
        "prompt",
        schema={
            "type": "object", "additionalProperties": False,
            "required": ["ok"], "properties": {"ok": {"type": "boolean"}},
        },
        provider="codex-cli", model="model-a",
        env={"ARC_HOME": str(tmp_path / "arc-home")}, process_chain=[],
        artifact_dir=tmp_path / "call", schema_canary_root=root,
    )
    assert result["ok"] is True
    receipts = [json.loads(path.read_text()) for path in (root / "schema-canaries").glob("*.json")]
    assert any(receipt.get("status") == "proven" for receipt in receipts)


def test_schema_formatter_inherits_batch_canary_root(monkeypatch, tmp_path):
    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            return LLMProviderResponse(
                {"wrong": "A detailed but structurally invalid response for repair."},
                raw_output='{"wrong":"A detailed but structurally invalid response for repair."}',
            )

    nested = []
    outer_run_json = runner.run_json

    def nested_run_json(prompt, **kwargs):
        nested.append(kwargs)
        return {"decision": "format", "formatted": {"ok": True}, "reason": "repair"}

    def fake_formatter(**kwargs):
        kwargs["json_runner"](
            "format",
            schema={"type": "object"},
            provider=kwargs["provider"],
            model=kwargs["model"],
        )
        return SimpleNamespace(
            action="format",
            value={"ok": True},
            reason="repair",
            structured_output={"mode": "recovered", "recovery_strategy": "schema_formatter"},
        )

    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: Provider())
    monkeypatch.setattr(runner, "run_json", nested_run_json)
    monkeypatch.setattr(runner, "format_to_schema_or_retry", fake_formatter)
    shared_root = tmp_path / "batch-root"

    result = outer_run_json(
        "prompt",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        provider="codex-cli",
        model="model-a",
        env={"ARC_HOME": str(tmp_path / "arc-home")},
        process_chain=[],
        artifact_dir=tmp_path / "call-artifacts",
        schema_canary_root=shared_root,
        output_recovery="warn",
    )

    assert result["ok"] is True
    assert len(nested) == 1
    assert nested[0]["schema_canary_root"] == shared_root


def test_runner_admission_submits_one_of_24_deterministic_rejections(
    monkeypatch, tmp_path
):
    counter = 0
    counter_lock = threading.Lock()
    first_started = threading.Event()
    release_first = threading.Event()

    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            nonlocal counter
            with counter_lock:
                counter += 1
            first_started.set()
            assert release_first.wait(5)
            raise LLMWorkerError(
                "provider rejected structured-output schema",
                retryable=False,
                category=LLMFailureCategory.SCHEMA,
                submission_state=LLMSubmissionState.SUBMITTED,
            )

    monkeypatch.setattr(runner, "select_provider", lambda *_args, **_kwargs: Provider())
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }
    root = tmp_path / "batch-root"

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(
                runner.run_json,
                "prompt",
                schema=schema,
                provider="codex-cli",
                model="model-a",
                env={"ARC_HOME": str(tmp_path / "arc-home")},
                process_chain=[],
                artifact_dir=tmp_path / f"call-{index}",
                schema_canary_root=root,
            )
            for index in range(24)
        ]
        assert first_started.wait(2)
        assert counter == 1
        release_first.set()
        failures = []
        for future in futures:
            with pytest.raises(runner.LLMTaskError) as caught:
                future.result(timeout=5)
            failures.append(caught.value)

    assert len(failures) == 24
    assert counter == 1
    receipts = list((root / "schema-canaries").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "rejected"
