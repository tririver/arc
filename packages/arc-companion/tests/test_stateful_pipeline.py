from __future__ import annotations

import json
import threading
import time

import pytest

from arc_companion.chapter_guide import generate_chapter_guide
from arc_companion.stateful_pipeline import (
    ContextRolloverBudget,
    CorrectionBudget,
    GLOSSARY_SETUP_MAX_BYTES,
    LLMSubmissionLimiter,
    StatefulPromptStream,
)


def _stream(mapping=None) -> StatefulPromptStream:
    return StatefulPromptStream(
        chapter_id="ch-0001", lane="translation", generation=1,
        fixed_rules={"immutable": True}, chapter={"title": "One"},
        guide={"main_content": "Guide"}, compact_glossary=mapping or [],
    )


def test_generation_bootstrap_is_not_repeated_in_delta() -> None:
    stream = _stream([{"source": "field", "target": "场"}])
    first = json.loads(stream.request("first", cursor="s1", source_sha256="a"))
    second = json.loads(stream.request(
        "REPEATED FIXED RULES\n\nSEGMENT:\nsecond", cursor="s2", source_sha256="b",
        block_glossary=[{"source": "fermion", "target": "费米子"}],
    ))
    assert first["turn_kind"] == "generation_bootstrap"
    assert first["chapter_guide"] == {"main_content": "Guide"}
    assert first["chapter_glossary_mapping"] == [{"source": "field", "target": "场"}]
    assert second["turn_kind"] == "delta"
    assert second["cursor"] == "s2"
    assert "chapter_guide" not in second
    assert "chapter_glossary_mapping" not in second
    assert second["glossary_mapping"] == [{"source": "field", "target": "场"}]
    assert second["current_request"] == "SEGMENT:\nsecond"


def test_large_glossary_setup_is_lossless_and_only_emitted_once() -> None:
    mapping = [{"source": f"term-{index}", "target": "译" * 200} for index in range(400)]
    assert len(json.dumps(mapping, ensure_ascii=False).encode()) > GLOSSARY_SETUP_MAX_BYTES
    stream = _stream(mapping)
    turns = [json.loads(value) for value in stream.setup_turns()]
    assert len(turns) > 1
    assert turns[0]["turn_kind"] == "generation_bootstrap"
    assert all(
        len(json.dumps(turn["entries"], ensure_ascii=False, separators=(",", ":")).encode())
        <= GLOSSARY_SETUP_MAX_BYTES for turn in turns
    )
    assert [item for turn in turns for item in turn["entries"]] == mapping
    assert stream.setup_turns() == []
    assert json.loads(stream.request("block", cursor="s1", source_sha256="a"))["turn_kind"] == "delta"


def test_context_rollover_uses_seventy_percent_and_prompt_estimate() -> None:
    budget = ContextRolloverBudget(context_window_tokens=100)
    budget.record({"total_input_tokens": 60, "output_tokens": 9})
    assert not budget.rollover_due()
    budget.record({}, prompt_bytes=4)
    assert budget.rollover_due()


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
