from __future__ import annotations

import json
import threading
import time

import pytest

from arc_companion.chapter_guide import generate_chapter_guide
from arc_companion.stateful_pipeline import (
    ContextRolloverBudget,
    CorrectionBudget,
    LLMSubmissionLimiter,
    STATEFUL_TURN_VERSION,
    StatefulPromptStream,
    read_stream_state,
    pin_lane_runtime_profile,
    resolve_lane_runtime_profile,
    validate_lane_paper_runtime_profile,
    write_stream_state,
)


def _stream() -> StatefulPromptStream:
    return StatefulPromptStream(
        chapter_id="ch-0001", lane="translation", generation=1,
        fixed_rules={"immutable": True},
        static_context={
            "chapter": {"title": "One"},
            "chapter_guide": {"main_content": "Guide"},
            "navigation": [{"section": "One"}],
        },
    )


def test_generation_bootstrap_is_not_repeated_in_delta() -> None:
    stream = _stream()
    first = json.loads(stream.request(
        "first", cursor="s1", source_sha256="a",
        current_payload={"source_blocks": [{"text": "field"}], "segment_glossary": []},
    ))
    second = json.loads(stream.request(
        "REPEATED FIXED RULES\n\nSEGMENT:\nsecond", cursor="s2", source_sha256="b",
        current_payload={
            "source_blocks": [{"text": "fermion"}],
            "segment_glossary": [{"source": "fermion", "target": "费米子"}],
        },
    ))
    assert first["schema_version"] == STATEFUL_TURN_VERSION
    assert first["turn_kind"] == "generation_bootstrap"
    assert first["static_context"]["chapter_guide"] == {"main_content": "Guide"}
    assert second["turn_kind"] == "delta"
    assert second["cursor"] == "s2"
    assert "static_context" not in second
    assert "chapter_id" not in second
    assert second["current_payload"]["source_blocks"] == [{"text": "fermion"}]
    assert second["current_payload"]["request"].startswith("REPEATED FIXED RULES")


def test_delta_size_does_not_grow_with_prior_payloads() -> None:
    stream = _stream()
    stream.request(
        "first" + "x" * 50_000, cursor="s1", source_sha256="a",
        current_payload={"source_blocks": [{"text": "large" + "x" * 50_000}]},
    )
    delta = stream.request(
        "second", cursor="s2", source_sha256="b",
        current_payload={"source_blocks": [{"text": "small"}]},
    )
    assert len(delta) < 1_000
    assert "large" not in delta


def test_stream_restart_uses_receipts_not_crash_incomplete_turn_count(tmp_path) -> None:
    stream = _stream()
    stream.request(
        "submitted", cursor="s1", source_sha256="a", current_payload={},
    )
    # A second request was prepared in memory but crashed before provider submission.
    stream.request(
        "not submitted", cursor="s2", source_sha256="b", current_payload={},
    )
    budget = ContextRolloverBudget(context_window_tokens=100)
    budget.record({"total_input_tokens": 40, "output_tokens": 5})
    path = tmp_path / "stream.json"
    write_stream_state(path, stream=stream, budget=budget)

    restored = read_stream_state(path, receipt_turn_count=1)
    assert restored is not None
    recovered_stream, recovered_budget = restored
    assert recovered_stream.turn_count == 1
    assert recovered_budget.input_tokens == 40
    turn = json.loads(recovered_stream.request(
        "retry", cursor="s2", source_sha256="b", current_payload={},
    ))
    assert turn["turn_kind"] == "delta"


def test_rollover_capsule_and_usage_survive_restart(tmp_path) -> None:
    stream = StatefulPromptStream(
        chapter_id="ch-0001", lane="translation", generation=2,
        fixed_rules={"immutable": True}, static_context={"chapter": "one"},
        continuity_capsule={
            "accepted_chain_sha256": "chain", "last_accepted_segment_id": "s1",
            "last_input_sha256": "in", "last_output_sha256": "out",
        },
    )
    budget = ContextRolloverBudget(context_window_tokens=100)
    write_stream_state(tmp_path / "stream.json", stream=stream, budget=budget)

    restored = read_stream_state(tmp_path / "stream.json", receipt_turn_count=0)
    assert restored is not None
    recovered_stream, _ = restored
    turn = json.loads(recovered_stream.request(
        "continue", cursor="s2", source_sha256="next", current_payload={},
    ))
    assert turn["turn_kind"] == "generation_bootstrap"
    assert turn["continuity_capsule"]["accepted_chain_sha256"] == "chain"


@pytest.mark.parametrize("requested_allow_internet", [False, True])
def test_new_translation_generation_pins_primary_and_repairs_offline(
    tmp_path, requested_allow_internet: bool,
) -> None:
    path = tmp_path / "profile.json"
    primary = resolve_lane_runtime_profile(
        path, chapter_id="ch-1", lane="translation", generation=2,
        requested_allow_internet=requested_allow_internet,
        inherit_host_tools=False, existing_generation=False,
        recorded_runtime_fingerprint=None,
        provider="codex-cli", model="fixed-model", model_tier="medium",
    )
    # Coverage/token repairs resolve the same durable generation profile.
    repair = resolve_lane_runtime_profile(
        path, chapter_id="ch-1", lane="translation", generation=2,
        requested_allow_internet=not requested_allow_internet,
        inherit_host_tools=True, existing_generation=True,
        recorded_runtime_fingerprint="different-current-runtime",
        provider="claude-cli", model="drifted-model", model_tier="high",
    )

    assert primary == repair
    assert primary["allow_internet"] is False
    assert primary["provider"] == "codex-cli"
    assert primary["model"] == "fixed-model"


def test_existing_translation_and_commentary_generations_preserve_access_choice(tmp_path) -> None:
    translation = resolve_lane_runtime_profile(
        tmp_path / "translation.json", chapter_id="ch-1", lane="translation",
        generation=1, requested_allow_internet=True, inherit_host_tools=False,
        existing_generation=True, recorded_runtime_fingerprint="recorded",
    )
    commentary = resolve_lane_runtime_profile(
        tmp_path / "commentary.json", chapter_id="ch-1", lane="companion",
        generation=1, requested_allow_internet=True, inherit_host_tools=False,
        existing_generation=False, recorded_runtime_fingerprint=None,
    )

    assert translation["allow_internet"] is True
    assert translation["recorded_runtime_fingerprint"] == "recorded"
    assert commentary["allow_internet"] is True


def test_auto_profile_pins_actual_provider_and_model_across_restart(tmp_path) -> None:
    path = tmp_path / "profile.json"
    selected = resolve_lane_runtime_profile(
        path, chapter_id="ch-1", lane="translation", generation=1,
        requested_allow_internet=False, inherit_host_tools=False,
        existing_generation=False, recorded_runtime_fingerprint=None,
        provider="auto", model=None, model_tier="medium",
    )
    pinned = pin_lane_runtime_profile(
        path, selected, provider="codex-cli", model="actual-model",
        runtime_fingerprint="manifest-hash",
    )
    restarted = resolve_lane_runtime_profile(
        path, chapter_id="ch-1", lane="translation", generation=1,
        requested_allow_internet=True, inherit_host_tools=True,
        existing_generation=True, recorded_runtime_fingerprint="drifted",
        provider="claude-cli", model="other", model_tier="high",
    )

    assert restarted == pinned
    assert restarted["provider"] == "codex-cli"
    assert restarted["model"] == "actual-model"


def test_lane_runtime_profile_pins_broker_policy_and_direct_decision(tmp_path) -> None:
    path = tmp_path / "profile.json"
    requested = {
        "arc_paper_access": "full",
        "paper_policy_sha256": None,
        "paper_catalog_sha256": None,
        "paper_network_authorized": True,
        "arc_paper_direct_shell": False,
        "paper_direct_decision": "controller",
        "direct_shell_probe_id": "probe-not-requested",
    }
    selected = resolve_lane_runtime_profile(
        path, chapter_id="ch-1", lane="translation", generation=1,
        requested_allow_internet=False, inherit_host_tools=False,
        existing_generation=False, recorded_runtime_fingerprint=None,
        provider="codex-cli", model="fixed-model", model_tier="medium",
        paper_runtime_profile=requested,
    )
    resolved = {
        **requested,
        "paper_policy_sha256": "policy-hash",
        "paper_catalog_sha256": "catalog-hash",
        "direct_shell_probe_id": "probe-not-requested",
    }
    pinned = pin_lane_runtime_profile(
        path, selected, provider="codex-cli", model="fixed-model",
        runtime_fingerprint="runtime-hash", paper_runtime_profile=resolved,
    )

    assert pinned["arc_paper_access"] == "full"
    assert pinned["paper_policy_sha256"] == "policy-hash"
    assert pinned["paper_catalog_sha256"] == "catalog-hash"
    assert pinned["paper_network_authorized"] is True
    assert pinned["paper_direct_decision"] == "controller"
    assert pinned["direct_shell_probe_id"] == "probe-not-requested"


def test_legacy_started_lane_profile_does_not_gain_new_paper_capabilities(
    tmp_path,
) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps({
            "schema_version": "arc.companion.lane-runtime-profile.v1",
            "chapter_id": "ch-1", "lane": "translation", "generation": 1,
            "allow_internet": False, "inherit_host_tools": False,
            "provider": "codex-cli", "model": "fixed-model",
            "model_tier": "medium", "recorded_runtime_fingerprint": "old-runtime",
        }),
        encoding="utf-8",
    )
    existing = resolve_lane_runtime_profile(
        path, chapter_id="ch-1", lane="translation", generation=1,
        requested_allow_internet=True, inherit_host_tools=True,
        existing_generation=True, recorded_runtime_fingerprint="old-runtime",
        paper_runtime_profile={"arc_paper_access": "full"},
    )

    assert existing["schema_version"] == "arc.companion.lane-runtime-profile.v2"
    assert existing["migrated_from_schema_version"].endswith(".v1")
    assert existing["arc_paper_access"] == "none"
    assert existing["arc_paper_direct_shell"] is False
    assert existing["paper_direct_decision"] == "disabled"


@pytest.mark.parametrize(
    "changed",
    [
        {"arc_paper_access": "none", "arc_paper_direct_shell": False},
        {"arc_paper_access": "full", "arc_paper_direct_shell": True},
        {
            "paper_managed_job_route": True,
            "paper_child_llm_max_calls": 2,
            "paper_child_llm_max_tokens": 2_000,
            "paper_child_llm_output_reserve_tokens": 100,
        },
    ],
)
def test_started_v2_lane_rejects_access_or_direct_recipe_drift(
    tmp_path, changed,
) -> None:
    path = tmp_path / "profile.json"
    requested = {
        "arc_paper_access": "full",
        "paper_policy_sha256": None,
        "paper_catalog_sha256": None,
        "paper_network_authorized": True,
        "arc_paper_direct_shell": False,
        "paper_direct_decision": "controller",
        "direct_shell_probe_id": "probe-not-requested",
    }
    resolve_lane_runtime_profile(
        path, chapter_id="ch-1", lane="translation", generation=1,
        requested_allow_internet=False, inherit_host_tools=False,
        existing_generation=False, recorded_runtime_fingerprint=None,
        paper_runtime_profile=requested,
    )

    with pytest.raises(ValueError, match="lane ARC-paper"):
        resolve_lane_runtime_profile(
            path, chapter_id="ch-1", lane="translation", generation=1,
            requested_allow_internet=False, inherit_host_tools=False,
            existing_generation=True, recorded_runtime_fingerprint=None,
            paper_runtime_profile={**requested, **changed},
        )


def test_started_v2_lane_rejects_resolved_policy_catalog_and_probe_drift() -> None:
    pinned = {
        "arc_paper_access": "full",
        "paper_policy_sha256": "policy-a",
        "paper_catalog_sha256": "catalog-a",
        "paper_network_authorized": True,
        "arc_paper_direct_shell": True,
        "paper_direct_decision": "direct",
        "direct_shell_probe_id": "probe-a",
    }
    for key, value in (
        ("paper_policy_sha256", "policy-b"),
        ("paper_catalog_sha256", "catalog-b"),
        ("direct_shell_probe_id", "probe-b"),
    ):
        with pytest.raises(ValueError, match="lane ARC-paper"):
            validate_lane_paper_runtime_profile(
                pinned, {**pinned, key: value},
            )


def test_generation_bootstrap_retries_until_a_native_turn_is_accepted() -> None:
    stream = _stream()
    first = json.loads(stream.request(
        "first", cursor="s1", source_sha256="one", current_payload={},
    ))
    stream.reconcile_turn_count(0)
    retry = json.loads(stream.request(
        "retry", cursor="s1", source_sha256="one", current_payload={},
    ))
    stream.reconcile_turn_count(1)
    next_segment = json.loads(stream.request(
        "next", cursor="s2", source_sha256="two", current_payload={},
    ))

    assert first["turn_kind"] == retry["turn_kind"] == "generation_bootstrap"
    assert next_segment["turn_kind"] == "delta"


def test_provisional_rollover_profile_migrates_preceding_generation_fingerprint(
    tmp_path,
) -> None:
    """A paid first turn pins an auto profile polluted by legacy rollover state."""
    path = tmp_path / "profile.json"
    provisional = resolve_lane_runtime_profile(
        path, chapter_id="ch-1", lane="translation", generation=2,
        requested_allow_internet=True, inherit_host_tools=False,
        existing_generation=False,
        recorded_runtime_fingerprint="preceding-generation-manifest",
        provider="auto", model=None, model_tier="medium",
    )

    pinned = pin_lane_runtime_profile(
        path, provisional, provider="codex-cli", model="actual-model",
        runtime_fingerprint="generation-two-manifest",
    )

    assert pinned["provider"] == "codex-cli"
    assert pinned["model"] == "actual-model"
    assert pinned["recorded_runtime_fingerprint"] == "generation-two-manifest"
    assert (
        pinned["migrated_from_runtime_fingerprint"]
        == "preceding-generation-manifest"
    )


def test_pinned_lane_profile_still_rejects_runtime_fingerprint_drift(tmp_path) -> None:
    path = tmp_path / "profile.json"
    selected = resolve_lane_runtime_profile(
        path, chapter_id="ch-1", lane="translation", generation=2,
        requested_allow_internet=False, inherit_host_tools=False,
        existing_generation=False, recorded_runtime_fingerprint=None,
        provider="codex-cli", model="fixed-model", model_tier="medium",
    )
    pinned = pin_lane_runtime_profile(
        path, selected, provider="codex-cli", model="fixed-model",
        runtime_fingerprint="generation-two-manifest",
    )

    with pytest.raises(ValueError, match="fingerprint changed"):
        pin_lane_runtime_profile(
            path, pinned, provider="codex-cli", model="fixed-model",
            runtime_fingerprint="drifted-manifest",
        )


def test_context_rollover_uses_seventy_percent_and_prompt_estimate() -> None:
    budget = ContextRolloverBudget(context_window_tokens=100)
    budget.record({"total_input_tokens": 60, "output_tokens": 9})
    assert not budget.rollover_due()
    budget.record({}, prompt_bytes=4)
    assert budget.rollover_due()


def test_context_rollover_does_not_sum_outputs_already_in_later_input() -> None:
    budget = ContextRolloverBudget(context_window_tokens=200)
    budget.record({"total_input_tokens": 60, "output_tokens": 20})
    budget.record({"total_input_tokens": 90, "output_tokens": 10})
    assert budget.input_tokens == 90
    assert budget.output_tokens == 20
    assert not budget.rollover_due()


def test_correction_budget_allows_exactly_one_turn() -> None:
    budget = CorrectionBudget()
    budget.consume("s1")
    with pytest.raises(RuntimeError, match="already consumed"):
        budget.consume("s1")


def test_submission_limiter_caps_nested_actual_submissions() -> None:
    limiter = LLMSubmissionLimiter(2)
    lock = threading.Lock()
    active = maximum = 0

    def submit() -> None:
        nonlocal active, maximum
        with limiter.permit():
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with lock:
                active -= 1

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert maximum == 2


def test_long_chapter_guide_resumes_bounded_windows_before_final(tmp_path) -> None:
    blocks = [
        {"block_id": f"b{index}", "text": str(index) + "x" * 25_000}
        for index in range(5)
    ]
    calls = []

    def call(prompt, schema, artifact_dir, label):
        calls.append((prompt, label))
        if "-window-" in label:
            return {"window_received": int(label.rsplit("-", 1)[1])}
        return {
            "motivation": None, "main_content": "content", "section_logic": None,
            "prerequisites": None, "pedagogical_comparison": None,
            "historical_context": [], "supplementary_reading": [],
        }

    result = generate_chapter_guide(
        {"chapter_id": "ch-0001", "title": "One"}, blocks,
        language="zh-CN", evidence={}, checkpoint_dir=tmp_path, force=True,
        call_model=call, stateful=True,
    )
    assert result["main_content"] == "content"
    assert len(calls) > 2
    assert all("source_blocks" in prompt for prompt, _ in calls[:-1])
    assert calls[-1][1].endswith("-final")
    assert "prepared_source_windows" in calls[-1][0]
