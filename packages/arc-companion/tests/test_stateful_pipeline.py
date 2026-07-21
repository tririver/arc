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
            "book_position": None, "prerequisites": None, "supplementary_reading": [],
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
