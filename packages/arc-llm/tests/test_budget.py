from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path

import pytest

from arc_llm.budget import (
    BudgetCorrupt,
    BudgetExhausted,
    SharedBudget,
    current_shared_budget_binding,
    shared_budget_context,
)
from arc_llm import runner
from arc_llm import budget as budget_module
from arc_llm.runner import run_json
from arc_llm.usage import LLMProviderResponse, LLMUsage


def _budget(tmp_path: Path, *, calls: int = 2, tokens: int = 100) -> SharedBudget:
    return SharedBudget.create(
        tmp_path / "private" / "budget.sqlite3",
        budget_id="parent-job",
        max_calls=calls,
        max_tokens=tokens,
    )


def _process_reserve(reference: dict[str, str], index: int, queue) -> None:
    budget = SharedBudget.open(reference)
    try:
        budget.reserve(
            checkpoint_identity=f"process-checkpoint-{index}",
            provider_attempt=1,
            prompt_bytes=0,
            output_reserve_tokens=10,
        )
    except BudgetExhausted:
        queue.put("exhausted")
    else:
        queue.put("reserved")


def test_budget_reserves_and_settles_known_usage(tmp_path: Path) -> None:
    budget = _budget(tmp_path)
    reservation = budget.reserve(
        checkpoint_identity="checkpoint-a",
        provider_attempt=1,
        prompt_bytes=20,
        output_reserve_tokens=15,
    )
    assert budget.snapshot().outstanding_tokens == 20
    reservation.mark_submitted()
    settlement = reservation.settle_known(input_tokens=7, output_tokens=4)

    assert settlement.charged_tokens == 11
    assert budget.snapshot().to_json() == {
        "max_calls": 2,
        "max_tokens": 100,
        "charged_calls": 1,
        "charged_tokens": 11,
        "outstanding_calls": 0,
        "outstanding_tokens": 0,
        "remaining_calls": 1,
        "remaining_tokens": 89,
    }


def test_reservation_is_idempotent_but_identity_mismatch_is_corrupt(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path)
    first = budget.reserve(
        checkpoint_identity="checkpoint-a",
        provider_attempt=1,
        prompt_bytes=4,
        output_reserve_tokens=9,
    )
    replay = budget.reserve(
        checkpoint_identity="checkpoint-a",
        provider_attempt=1,
        prompt_bytes=4,
        output_reserve_tokens=9,
    )
    assert replay.reservation_id == first.reservation_id
    assert budget.snapshot().outstanding_calls == 1

    with pytest.raises(BudgetCorrupt, match="identity changed"):
        budget.reserve(
            checkpoint_identity="checkpoint-a",
            provider_attempt=1,
            prompt_bytes=8,
            output_reserve_tokens=9,
        )


def test_atomic_contention_admits_only_finite_parent_budget(tmp_path: Path) -> None:
    budget = _budget(tmp_path, calls=3, tokens=30)

    def reserve(index: int) -> str:
        try:
            return budget.reserve(
                checkpoint_identity=f"checkpoint-{index}",
                provider_attempt=1,
                prompt_bytes=0,
                output_reserve_tokens=10,
            ).reservation_id
        except BudgetExhausted:
            return "exhausted"

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(reserve, range(12)))

    assert sum(value != "exhausted" for value in outcomes) == 3
    assert budget.snapshot().remaining_calls == 0
    assert budget.snapshot().remaining_tokens == 0


def test_process_contention_admits_only_finite_parent_budget(tmp_path: Path) -> None:
    budget = _budget(tmp_path, calls=2, tokens=20)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_process_reserve,
            args=(budget.reference.to_json(), index, queue),
        )
        for index in range(6)
    ]
    for process in processes:
        process.start()
    outcomes = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert outcomes.count("reserved") == 2
    assert outcomes.count("exhausted") == 4
    assert budget.snapshot().remaining_calls == 0


def test_exhaustion_happens_before_any_submission(tmp_path: Path) -> None:
    budget = _budget(tmp_path, calls=0, tokens=100)
    with pytest.raises(BudgetExhausted):
        budget.reserve(
            checkpoint_identity="checkpoint-a",
            provider_attempt=1,
            prompt_bytes=1,
            output_reserve_tokens=1,
        )
    assert budget.snapshot().charged_calls == 0
    assert budget.snapshot().outstanding_calls == 0


def test_unknown_usage_and_ambiguous_failure_charge_full_reservation(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path)
    unknown = budget.reserve(
        checkpoint_identity="checkpoint-a",
        provider_attempt=1,
        prompt_bytes=5,
        output_reserve_tokens=8,
    )
    settlement = unknown.settle_unknown_usage()
    assert settlement.warning == "budget.usage_unknown_charged_reserved"
    assert settlement.charged_tokens == 10

    ambiguous = budget.reserve(
        checkpoint_identity="checkpoint-b",
        provider_attempt=1,
        prompt_bytes=4,
        output_reserve_tokens=5,
    )
    reconciled = budget.reconcile(
        ambiguous.reservation_id,
        checkpoint_submission_state="not_submitted",
        owner_alive=None,
    )
    assert reconciled.disposition == "reconciled_ambiguous"
    assert reconciled.charged_tokens == 6


def test_known_usage_overdraw_is_recorded_and_blocks_future_admission(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path, calls=2, tokens=20)
    reservation = budget.reserve(
        checkpoint_identity="checkpoint-a",
        provider_attempt=1,
        prompt_bytes=0,
        output_reserve_tokens=10,
    )
    settlement = reservation.settle_known(input_tokens=15, output_tokens=10)
    assert settlement.charged_tokens == 25
    assert budget.snapshot().charged_tokens == 25
    assert budget.snapshot().remaining_tokens == 0
    with pytest.raises(BudgetExhausted):
        budget.reserve(
            checkpoint_identity="checkpoint-b",
            provider_attempt=1,
            prompt_bytes=0,
            output_reserve_tokens=0,
        )


def test_reconcile_releases_only_proven_unsubmitted_dead_owner(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path)
    reservation = budget.reserve(
        checkpoint_identity="checkpoint-a",
        provider_attempt=1,
        prompt_bytes=20,
        output_reserve_tokens=5,
    )
    settlement = budget.reconcile(
        reservation.reservation_id,
        checkpoint_submission_state="not_submitted",
        owner_alive=False,
    )
    assert settlement.disposition == "proven_not_submitted"
    assert budget.snapshot().remaining_calls == 2


def test_binding_atomically_takes_over_dead_unsubmitted_reservation(
    tmp_path: Path, monkeypatch,
) -> None:
    budget = _budget(tmp_path, calls=1, tokens=100)
    stale = budget.reserve(
        checkpoint_identity="checkpoint-a",
        provider_attempt=1,
        prompt_bytes=8,
        output_reserve_tokens=10,
    )
    monkeypatch.setattr(
        budget_module, "_reservation_owner_alive", lambda row: False,
    )

    with shared_budget_context(budget, output_reserve_tokens=10):
        recovered = current_shared_budget_binding(required=True).reserve(
            checkpoint_identity="checkpoint-a",
            provider_attempt=1,
            prompt_bytes=8,
        )

    assert recovered.reservation_id == stale.reservation_id
    assert budget.snapshot().outstanding_calls == 1
    recovered.release_not_submitted()
    assert budget.snapshot().outstanding_calls == 0


def test_release_submitted_charges_instead_of_releasing(tmp_path: Path) -> None:
    budget = _budget(tmp_path)
    reservation = budget.reserve(
        checkpoint_identity="checkpoint-a",
        provider_attempt=1,
        prompt_bytes=0,
        output_reserve_tokens=9,
    )
    reservation.mark_submitted()
    settlement = reservation.release_not_submitted()
    assert settlement.disposition == "release_denied_submitted"
    assert settlement.charged_calls == 1
    assert settlement.charged_tokens == 9


def test_first_call_admission_transfers_without_second_call_charge(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path, calls=1, tokens=100)
    admission = budget.reserve(
        checkpoint_identity="broker-admission:job-a",
        provider_attempt=1,
        prompt_bytes=0,
        output_reserve_tokens=20,
    )
    with shared_budget_context(
        budget,
        output_reserve_tokens=20,
        admission_reservation_id=admission.reservation_id,
    ):
        binding = current_shared_budget_binding(required=True)
        assert binding is not None
        actual = binding.reserve(
            checkpoint_identity="provider-checkpoint-a",
            provider_attempt=1,
            prompt_bytes=12,
        )
        assert actual.reservation_id != admission.reservation_id
        assert budget.reservation(admission.reservation_id)["state"] == "released"
        assert budget.snapshot().outstanding_calls == 1
        actual.settle_known(input_tokens=4, output_tokens=3)
    replayed_admission = budget.reserve(
        checkpoint_identity="broker-admission:job-a",
        provider_attempt=1,
        prompt_bytes=0,
        output_reserve_tokens=20,
    )
    assert replayed_admission.reservation_id == admission.reservation_id
    assert budget.snapshot().charged_calls == 1
    assert budget.snapshot().charged_tokens == 7
    assert budget.snapshot().outstanding_calls == 0


def test_reference_permissions_and_context_are_explicit(tmp_path: Path) -> None:
    budget = _budget(tmp_path)
    assert oct(os.stat(budget.path.parent).st_mode & 0o777) == "0o700"
    assert oct(os.stat(budget.path).st_mode & 0o777) == "0o600"
    assert current_shared_budget_binding() is None
    with shared_budget_context(budget, output_reserve_tokens=17):
        binding = current_shared_budget_binding(required=True)
        assert binding is not None
        assert binding.budget is budget
        assert binding.output_reserve_tokens == 17
    assert current_shared_budget_binding() is None


def test_open_rejects_changed_reference_identity(tmp_path: Path) -> None:
    budget = _budget(tmp_path)
    reference = budget.reference.to_json()
    reference["identity_sha256"] = "0" * 64
    with pytest.raises(BudgetCorrupt, match="does not match"):
        SharedBudget.open(reference)


class _ResultProvider:
    name = "codex-cli"

    def __init__(self) -> None:
        self.calls = 0

    def generate_json_result(self, prompt, **kwargs):
        del prompt, kwargs
        self.calls += 1
        return LLMProviderResponse(
            {"ok": True},
            usage=LLMUsage(input_tokens=7, output_tokens=3),
        )


def test_schema_failure_settles_observed_usage_instead_of_full_reserve(
    tmp_path: Path, monkeypatch,
) -> None:
    class InvalidProvider:
        name = "codex-cli"

        def generate_json_result(self, prompt, **kwargs):
            del prompt, kwargs
            return LLMProviderResponse(
                {"ok": "not-a-boolean"},
                usage=LLMUsage(input_tokens=7, output_tokens=3),
            )

    monkeypatch.setattr(
        runner, "select_provider", lambda *args, **kwargs: InvalidProvider(),
    )
    budget = _budget(tmp_path, calls=1, tokens=2_000)
    with shared_budget_context(budget, output_reserve_tokens=500):
        with pytest.raises(Exception):
            run_json(
                "prompt",
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                provider="codex-cli",
                env={},
                process_chain=[],
                artifact_dir=tmp_path / "invalid-response",
                call_label="invalid-response",
                idempotency_key="invalid-response",
            )

    snapshot = budget.snapshot()
    assert snapshot.charged_calls == 1
    assert snapshot.charged_tokens == 10
    assert snapshot.outstanding_calls == 0


@pytest.mark.parametrize(
    ("provider_name", "usage"),
    [
        ("kimi-code-cli", LLMUsage(input_tokens=7, output_tokens=3)),
        ("codex-cli", LLMUsage()),
        ("codex-cli", LLMUsage(input_tokens=7)),
    ],
)
def test_runner_replays_persisted_conservative_budget_settlement(
    tmp_path: Path, monkeypatch, provider_name: str, usage: LLMUsage,
) -> None:
    provider = _ResultProvider()
    provider.name = provider_name
    provider.generate_json_result = lambda prompt, **kwargs: (
        setattr(provider, "calls", provider.calls + 1)
        or LLMProviderResponse({"ok": True}, usage=usage)
    )
    monkeypatch.setattr(runner, "select_provider", lambda *args, **kwargs: provider)
    budget = _budget(tmp_path, calls=1, tokens=2_000)
    kwargs = {
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "provider": provider_name,
        "env": {},
        "process_chain": [],
        "artifact_dir": tmp_path / "calls",
        "call_label": "section-a",
        "idempotency_key": "stable-section-a",
    }
    with shared_budget_context(budget, output_reserve_tokens=50):
        first = run_json("prompt", **kwargs)
        replay = run_json("prompt", **kwargs)

    assert provider.calls == 1
    assert first["arc_llm_call_record"]["budget_receipt"][
        "disposition"
    ] == "unknown_usage"
    assert replay["arc_llm_call_record"]["budget_receipt"][
        "disposition"
    ] == "unknown_usage"
    assert budget.snapshot().charged_calls == 1
    assert budget.snapshot().outstanding_calls == 0


def test_runner_reserves_after_replay_check_and_replay_costs_zero(
    tmp_path: Path, monkeypatch,
) -> None:
    provider = _ResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda *args, **kwargs: provider)
    budget = _budget(tmp_path, calls=2, tokens=2_000)
    kwargs = {
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "provider": "codex-cli",
        "env": {},
        "process_chain": [],
        "artifact_dir": tmp_path / "calls",
        "call_label": "section-a",
        "idempotency_key": "stable-section-a",
    }
    with shared_budget_context(budget, output_reserve_tokens=50):
        first_result = run_json("prompt", **kwargs)
        assert first_result["ok"] is True
        first = budget.snapshot()
        replay_result = run_json("prompt", **kwargs)
        assert replay_result["ok"] is True

    assert provider.calls == 1
    assert first_result["arc_llm_call_record"]["budget_receipt"][
        "disposition"
    ] == "known_usage"
    assert replay_result["arc_llm_call_record"]["budget_receipt"][
        "disposition"
    ] == "known_usage"
    assert first.charged_calls == 1
    assert first.charged_tokens == 10
    assert budget.snapshot() == first


def test_runner_exhaustion_happens_before_controlled_provider_call(
    tmp_path: Path, monkeypatch,
) -> None:
    provider = _ResultProvider()
    monkeypatch.setattr(runner, "select_provider", lambda *args, **kwargs: provider)
    budget = _budget(tmp_path, calls=0, tokens=2_000)
    with shared_budget_context(budget, output_reserve_tokens=50):
        with pytest.raises(BudgetExhausted):
            run_json(
                "prompt",
                provider="codex-cli",
                env={},
                process_chain=[],
                artifact_dir=tmp_path / "calls",
                call_label="section-a",
                idempotency_key="stable-section-a",
            )
    assert provider.calls == 0


def test_runner_reconciles_crash_reservation_before_supervision(
    tmp_path: Path, monkeypatch,
) -> None:
    class CrashProvider:
        name = "codex-cli"

        def generate_json_result(
            self, prompt, *, progress_callback=None, **kwargs,
        ):
            del prompt, kwargs
            assert progress_callback is not None
            progress_callback({"event": "submitted"})
            raise SystemExit(73)

    provider = CrashProvider()
    monkeypatch.setattr(runner, "select_provider", lambda *args, **kwargs: provider)
    real_settle_failure = runner._settle_descendant_failure
    monkeypatch.setattr(
        runner, "_settle_descendant_failure",
        lambda reservation, exc, **kwargs: None,
    )
    budget = _budget(tmp_path, calls=1, tokens=2_000)
    kwargs = {
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "provider": "codex-cli",
        "env": {},
        "process_chain": [],
        "artifact_dir": tmp_path / "calls",
        "call_label": "crash",
        "idempotency_key": "stable-crash",
    }
    with shared_budget_context(budget, output_reserve_tokens=50):
        with pytest.raises(SystemExit):
            run_json("prompt", **kwargs)
    assert budget.snapshot().outstanding_calls == 1

    monkeypatch.setattr(
        runner, "_settle_descendant_failure", real_settle_failure,
    )
    with shared_budget_context(budget, output_reserve_tokens=50):
        with pytest.raises(Exception, match="needs explicit supervision"):
            run_json("prompt", **kwargs)

    assert budget.snapshot().outstanding_calls == 0
    assert budget.snapshot().charged_calls == 1
