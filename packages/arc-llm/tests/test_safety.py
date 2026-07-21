from __future__ import annotations

from contextlib import contextmanager
import errno
import os
import sqlite3
import subprocess
import sys

import pytest

from arc_llm.providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerError,
    failure_disposition,
)
from arc_llm.safety import (
    GLOBAL_MAX_CONCURRENCY,
    LLMCircuitOpen,
    LLMSafetyConfigurationError,
    LLMSafetyController,
)
from arc_llm import safety as safety_module


def controller(tmp_path, *, env=None, clock=None):
    return LLMSafetyController(
        env={"ARC_HOME": str(tmp_path), **(env or {})},
        db_path=tmp_path / "safety.sqlite3",
        now=(lambda: clock[0]) if clock is not None else __import__("time").time,
        heartbeat_seconds=60,
        slot_lease_seconds=1,
    )


def test_typed_failure_defaults_nonretryable_and_preserves_legacy_abort_batch():
    error = LLMWorkerError(
        "quota exhausted",
        category=LLMFailureCategory.QUOTA,
        abort_scope=LLMAbortScope.PROVIDER,
        submission_state=LLMSubmissionState.SUBMITTED,
    )

    assert error.retryable is False
    assert error.abort_batch is True
    assert error.disposition.abort_scope == LLMAbortScope.PROVIDER


def test_failure_disposition_walks_wrapped_cause():
    inner = LLMWorkerError("denied", category="permission", abort_scope="provider")
    try:
        try:
            raise inner
        except LLMWorkerError as exc:
            raise RuntimeError("wrapper") from exc
    except RuntimeError as outer:
        disposition = failure_disposition(outer)

    assert disposition is not None
    assert disposition.category == LLMFailureCategory.PERMISSION
    assert disposition.abort_scope == LLMAbortScope.PROVIDER


def test_failure_disposition_does_not_let_generic_worker_wrapper_hide_provider_fatal():
    try:
        try:
            raise LLMWorkerError("quota", category="quota", abort_scope="provider")
        except LLMWorkerError as exc:
            raise LLMWorkerError("worker failed") from exc
    except LLMWorkerError as outer:
        disposition = failure_disposition(outer)

    assert disposition is not None
    assert disposition.category == LLMFailureCategory.QUOTA
    assert disposition.abort_scope == LLMAbortScope.PROVIDER


def test_global_concurrency_is_shared_between_controller_instances(tmp_path):
    first = controller(tmp_path, env={"ARC_LLM_MAX_CONCURRENCY": "2"})
    second = controller(tmp_path, env={"ARC_LLM_MAX_CONCURRENCY": "2"})
    slot_1 = first.acquire_slot("codex-cli")
    slot_2 = second.acquire_slot("claude-cli")
    try:
        with pytest.raises(LLMWorkerError, match="LLM capacity"):
            first.acquire_slot("kimi-code-cli", timeout_seconds=0.02)
        assert first.status()["active_slots"] == 2
    finally:
        slot_1.release()
        slot_2.release()


def test_global_concurrency_is_shared_across_processes_and_reclaims_crash(tmp_path):
    env = dict(os.environ)
    env.update({"ARC_HOME": str(tmp_path), "ARC_LLM_MAX_CONCURRENCY": "1"})
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from arc_llm.safety import LLMSafetyController; import sys,time; "
                "slot=LLMSafetyController(db_path=sys.argv[1]).acquire_slot('codex-cli'); "
                "print('ready', flush=True); time.sleep(60)"
            ),
            str(tmp_path / "safety.sqlite3"),
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        safety = controller(tmp_path, env={"ARC_LLM_MAX_CONCURRENCY": "1"})
        with pytest.raises(LLMWorkerError, match="LLM capacity"):
            safety.acquire_slot("claude-cli", timeout_seconds=0.02)

        child.kill()
        child.wait(timeout=5)
        # The dead owner's PID/start identity is reclaimed without waiting for
        # a later status command or a full lease interval.
        slot = safety.acquire_slot("claude-cli", timeout_seconds=1)
        slot.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("value", ["0", "25", "invalid"])
def test_concurrency_override_can_only_lower_hard_cap(tmp_path, value):
    safety = controller(tmp_path, env={"ARC_LLM_MAX_CONCURRENCY": value})
    with pytest.raises(LLMSafetyConfigurationError):
        safety.effective_max_concurrency()


def test_default_global_concurrency_is_24(tmp_path):
    assert controller(tmp_path).effective_max_concurrency() == GLOBAL_MAX_CONCURRENCY == 24


def test_advisory_lock_treats_eacces_as_contention(monkeypatch, tmp_path):
    class ContendedFcntl:
        LOCK_EX = 1
        LOCK_NB = 2

        @staticmethod
        def flock(_fd, _flags):
            raise OSError(errno.EACCES, "contended")

    monkeypatch.setattr(safety_module, "_fcntl", ContendedFcntl())
    with (tmp_path / "lock").open("a+b") as handle:
        assert safety_module._advisory_try_lock(handle) is False


def test_advisory_lock_windows_branch_uses_one_byte_region(monkeypatch, tmp_path):
    calls = []

    class FakeMsvcrt:
        LK_NBLCK = 10
        LK_UNLCK = 11

        @staticmethod
        def locking(fd, mode, length):
            calls.append((fd, mode, length))

    monkeypatch.setattr(safety_module, "_fcntl", None)
    monkeypatch.setattr(safety_module, "_msvcrt", FakeMsvcrt())
    with (tmp_path / "lock").open("a+b") as handle:
        assert safety_module._advisory_try_lock(handle) is True
        safety_module._advisory_unlock(handle)
        assert handle.tell() == 0
    assert [item[1:] for item in calls] == [(FakeMsvcrt.LK_NBLCK, 1), (FakeMsvcrt.LK_UNLCK, 1)]


def test_provider_limit_does_not_reduce_capacity_for_other_providers(tmp_path):
    safety = controller(
        tmp_path,
        env={"ARC_LLM_MAX_CONCURRENCY": "3", "ARC_KIMI_MAX_CONCURRENCY": "1"},
    )
    codex = safety.acquire_slot("codex-cli")
    kimi = safety.acquire_slot("kimi-code-cli")
    try:
        with pytest.raises(LLMWorkerError, match="LLM capacity"):
            safety.acquire_slot("kimi-code-cli", timeout_seconds=0.02)
        claude = safety.acquire_slot("claude-cli", timeout_seconds=0.2)
        claude.release()
    finally:
        codex.release()
        kimi.release()


def test_lower_limit_counts_slots_acquired_at_higher_limit(tmp_path):
    permissive = controller(tmp_path, env={"ARC_LLM_MAX_CONCURRENCY": "24"})
    held = [permissive.acquire_slot("codex-cli") for _ in range(3)]
    restrictive = controller(tmp_path, env={"ARC_LLM_MAX_CONCURRENCY": "2"})
    try:
        with pytest.raises(LLMWorkerError, match="LLM capacity"):
            restrictive.acquire_slot("claude-cli", timeout_seconds=0.02)
    finally:
        for slot in held:
            slot.release()


@pytest.mark.parametrize(
    "category",
    [LLMFailureCategory.QUOTA, LLMFailureCategory.AUTHENTICATION, LLMFailureCategory.PERMISSION],
)
def test_provider_circuit_survives_new_controller_only_until_ttl(tmp_path, category):
    clock = [100.0]
    first = controller(tmp_path, clock=clock)
    first.report_failure(
        "kimi-code-cli",
        LLMWorkerError("provider blocked", category=category, abort_scope="provider"),
    )

    second = controller(tmp_path, clock=clock)
    with pytest.raises(LLMCircuitOpen) as caught:
        second.check_circuit("kimi-code-cli")
    assert caught.value.category == category
    assert caught.value.retry_after_seconds == pytest.approx(900)

    clock[0] += 901
    probe = second.check_circuit("kimi-code-cli")
    assert probe.probe_token
    second.report_success("kimi-code-cli", permit=probe)


def test_rate_limit_uses_15_minimum_cooldown_and_one_half_open_probe(tmp_path):
    clock = [100.0]
    first = controller(tmp_path, clock=clock)
    first.report_failure(
        "codex-cli",
        LLMWorkerError(
            "HTTP 429",
            category="rate_limit",
            abort_scope="provider",
            retry_after_seconds=10,
        ),
    )

    with pytest.raises(LLMCircuitOpen) as caught:
        first.check_circuit("codex-cli")
    assert caught.value.retry_after_seconds == pytest.approx(900)

    clock[0] += 901
    probe = first.check_circuit("codex-cli")
    assert probe.probe_token
    with pytest.raises(LLMCircuitOpen, match="probe already active"):
        controller(tmp_path, clock=clock).check_circuit("codex-cli")

    # A slow but live one-hour provider call remains the unique probe. Wall
    # time alone must never admit a second paid request.
    clock[0] += 3599
    with pytest.raises(LLMCircuitOpen, match="probe already active"):
        controller(tmp_path, clock=clock).check_circuit("codex-cli")

    first.report_success("codex-cli", permit=probe)
    assert first.status()["circuits"] == []


@pytest.mark.parametrize(
    "error",
    [
        ValueError("local failure"),
        LLMWorkerError("invalid request", category="invalid_request"),
    ],
)
def test_non_circuit_failure_releases_half_open_probe(tmp_path, error):
    clock = [100.0]
    safety = controller(tmp_path, clock=clock)
    safety.report_failure(
        "codex-cli",
        LLMWorkerError("authentication", category="authentication"),
    )
    clock[0] += 901
    probe = safety.check_circuit("codex-cli")

    safety.report_failure("codex-cli", error, permit=probe)

    replacement = controller(tmp_path, clock=clock).check_circuit("codex-cli")
    assert replacement.probe_token
    safety.report_success("codex-cli", permit=replacement)


def test_circuit_record_error_releases_half_open_probe(tmp_path, monkeypatch):
    clock = [100.0]
    safety = controller(tmp_path, clock=clock)
    safety.report_failure(
        "codex-cli",
        LLMWorkerError("authentication", category="authentication"),
    )
    clock[0] += 901
    probe = safety.check_circuit("codex-cli")
    original_transaction = safety._transaction

    @contextmanager
    def failing_transaction():
        raise sqlite3.OperationalError("database unavailable")
        yield

    monkeypatch.setattr(safety, "_transaction", failing_transaction)
    with pytest.raises(sqlite3.OperationalError, match="database unavailable"):
        safety.report_failure(
            "codex-cli",
            LLMWorkerError("quota", category="quota"),
            permit=probe,
        )
    monkeypatch.setattr(safety, "_transaction", original_transaction)

    replacement = controller(tmp_path, clock=clock).check_circuit("codex-cli")
    assert replacement.probe_token
    safety.report_success("codex-cli", permit=replacement)


def test_endpoint_identity_strips_credentials_and_query_from_status(tmp_path):
    safety = controller(tmp_path)
    safety.report_failure(
        "external",
        LLMWorkerError("quota", category="quota", abort_scope="provider"),
        endpoint="https://user:password@example.test/v1/?api_key=secret",
    )
    rendered = repr(safety.status())
    assert "password" not in rendered
    assert "api_key" not in rendered
    assert "secret" not in rendered
    assert "https://example.test" in rendered


def test_call_permit_releases_slot_and_records_circuit_failure(tmp_path):
    safety = controller(tmp_path)
    with pytest.raises(LLMWorkerError):
        with safety.acquire_call("claude-cli"):
            raise LLMWorkerError("quota", category="quota", abort_scope="provider")

    status = safety.status()
    assert status["active_slots"] == 0
    assert status["circuits"][0]["category"] == "quota"


def test_inflight_success_cannot_clear_newer_rate_limit_circuit(tmp_path):
    safety = controller(tmp_path)
    permit = safety.acquire_call("codex-cli")
    safety.report_failure(
        "codex-cli",
        LLMWorkerError("HTTP 429", category="rate_limit", abort_scope="provider"),
    )

    permit.report_success()
    permit.release()

    with pytest.raises(LLMCircuitOpen):
        safety.check_circuit("codex-cli")
