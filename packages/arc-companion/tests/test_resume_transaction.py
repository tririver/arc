from __future__ import annotations

import pytest

from arc_companion.resume_transaction import (
    AutomaticRegenerationExhausted,
    authorize_manual_restart,
    append_entries,
    begin_transaction,
    bind_transaction_checkpoint,
    claim_automatic_restart, ensure_auto_transaction,
    load_transaction,
    mark_entry,
    mark_replacement, plan_replacement, update_replacement,
)
from arc_companion.io import write_json


def _entry(tmp_path):
    return {
        "ledger_path": str(tmp_path / "ledger.json"),
        "session_key": "ch-0001:translation",
        "segment_id": "s2",
        "idempotency_key": "generation-1-key",
    }


def test_incomplete_native_transaction_upgrades_to_auto_without_reopening_resolved(tmp_path) -> None:
    begin_transaction(
        tmp_path, action="resume-native", recovery_options={}, entries=[_entry(tmp_path)],
    )
    mark_entry(tmp_path, 0, status="resolved", output_sha256="accepted")
    before = load_transaction(tmp_path)["entries"][0]
    mark_entry(tmp_path, 0, status="resolved", output_sha256="must-not-rewrite")

    upgraded = begin_transaction(
        tmp_path, action="auto", policy="auto", recovery_options={},
        entries=[{**_entry(tmp_path), "blocking_reason": "later discovery"}],
    )
    append_entries(tmp_path, [{**_entry(tmp_path), "blocking_reason": "must not mutate"}])
    final = load_transaction(tmp_path)

    assert upgraded["action"] == "auto"
    assert upgraded["policy"] == "auto"
    assert [item["action"] for item in upgraded["action_history"]] == [
        "resume-native", "auto",
    ]
    assert final["entries"][0] == before


def test_automatic_restart_budget_allows_three_group_attempts_and_replays(tmp_path) -> None:
    begin_transaction(
        tmp_path, action="auto", policy="auto",
        recovery_options={"max_auto_replacements": 3},
        entries=[_entry(tmp_path)],
    )
    kwargs = {
        "session_key": "ch-0001:translation", "segment_id": "s2",
        "ledger_path": tmp_path / "ledger.json", "source_generation": 1,
        "target_generation": 2, "suffix_segment_ids": ["s2", "s3"],
        "trigger_code": "invalid_paid_response", "trigger_reason": "schema mismatch",
        "abandoned_logical_key": "generation-1-key",
        "possible_duplicate_charge": True,
    }
    first = claim_automatic_restart(tmp_path, **kwargs)
    replay = claim_automatic_restart(
        tmp_path, **{**kwargs, "trigger_code": "native_session_missing"},
    )

    assert replay == first
    assert load_transaction(tmp_path)["restart_budgets"][0]["attempts_used"] == 1
    second = claim_automatic_restart(
        tmp_path, **{**kwargs, "source_generation": 2, "target_generation": 3},
    )
    third = claim_automatic_restart(
        tmp_path, **{**kwargs, "source_generation": 3, "target_generation": 4},
    )
    assert (second["attempt"], third["attempt"]) == (2, 3)
    assert second["group_id"] == third["group_id"]
    assert load_transaction(tmp_path)["restart_budgets"][0]["attempts_used"] == 3
    with pytest.raises(AutomaticRegenerationExhausted):
        claim_automatic_restart(
            tmp_path, **{**kwargs, "source_generation": 4, "target_generation": 5},
        )


def test_v3_checkpoint_binding_archives_mixed_checkpoint_journal_verbatim(tmp_path) -> None:
    first = tmp_path / "checkpoints" / "first"
    second = tmp_path / "checkpoints" / "second"
    path = tmp_path / ".arc-companion" / "resume-transaction.json"
    path.parent.mkdir(parents=True)
    original = (
        '{"schema_version":"arc.companion.resume-transaction.v2","entries":['
        f'{{"ledger_path":"{first}/a.json"}},'
        f'{{"ledger_path":"{second}/b.json"}}]}}'
    ).encode()
    path.write_bytes(original)

    result = bind_transaction_checkpoint(
        tmp_path, checkpoint_path=second, checkpoint_fingerprint="second",
    )

    assert result["archived"] is True
    assert not path.exists()
    assert __import__("pathlib").Path(result["archive_path"]).read_bytes() == original


def test_v3_checkpoint_binding_archives_changed_fingerprint_at_same_path(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoints" / "current"
    begin_transaction(
        tmp_path, action="auto", policy="auto", recovery_options={},
        entries=[_entry(checkpoint)], checkpoint_path=checkpoint,
        checkpoint_fingerprint="old-fingerprint",
    )

    result = bind_transaction_checkpoint(
        tmp_path, checkpoint_path=checkpoint,
        checkpoint_fingerprint="new-fingerprint",
    )

    assert result["archived"] is True
    assert not (tmp_path / ".arc-companion" / "resume-transaction.json").exists()


def test_replacement_status_is_monotonic_and_terminal(tmp_path) -> None:
    begin_transaction(
        tmp_path, action="auto", policy="auto", recovery_options={},
        entries=[_entry(tmp_path)],
    )
    claimed = claim_automatic_restart(
        tmp_path, session_key="ch-0001:translation", segment_id="s2",
        ledger_path=tmp_path / "ledger.json", source_generation=1,
        target_generation=2, suffix_segment_ids=["s2"], trigger_code="invalid",
        trigger_reason="invalid response",
    )
    mark_replacement(tmp_path, claimed["replacement_id"], status="rotated")
    accepted = mark_replacement(
        tmp_path, claimed["replacement_id"], status="accepted",
        accepted_logical_key="generation-2-key",
    )

    assert accepted["accepted_logical_key"] == "generation-2-key"
    with pytest.raises(ValueError, match="Terminal replacement"):
        mark_replacement(tmp_path, claimed["replacement_id"], status="failed")


def test_semantic_auto_recovery_api_upgrades_plans_and_updates(tmp_path) -> None:
    begin_transaction(
        tmp_path, action="resume-native", recovery_options={}, entries=[_entry(tmp_path)],
    )
    transaction = ensure_auto_transaction(tmp_path, reason="native reconciliation failed")
    planned = plan_replacement(
        tmp_path, session_key="ch-0001:translation", segment_id="s3",
        ledger_path=tmp_path / "ledger.json", source_generation=1,
        target_generation=2, suffix_start_segment_id="s2",
        suffix_segment_ids=["s2", "s3"], trigger_code="native_session_missing",
        trigger_reason="session is unavailable",
    )
    updated = update_replacement(
        tmp_path, planned["replacement_id"], phase="suffix_invalidated",
    )

    assert transaction["action"] == "auto"
    assert planned["suffix_start_segment_id"] == "s2"
    assert updated["status"] == "suffix_invalidated"


def test_manual_override_atomically_replans_every_unresolved_entry(tmp_path) -> None:
    first = tmp_path / "first-ledger.json"
    second = tmp_path / "second-ledger.json"
    write_json(first, {"generation": 2})
    write_json(second, {"generation": 4})
    begin_transaction(
        tmp_path, action="auto", policy="auto", recovery_options={},
        entries=[
            {**_entry(tmp_path), "ledger_path": str(first)},
            {
                **_entry(tmp_path), "ledger_path": str(second),
                "session_key": "ch-0002:translation", "segment_id": "s7",
                "idempotency_key": "generation-4-key",
            },
        ],
    )

    restarted = authorize_manual_restart(tmp_path)

    assert restarted["action"] == "restart-generation"
    assert restarted["policy"] == "manual"
    assert [
        (entry["initial_generation"], entry["target_generation"])
        for entry in restarted["entries"]
    ] == [(2, 3), (4, 5)]
    assert all(entry["manual_restart_after_auto"] for entry in restarted["entries"])
