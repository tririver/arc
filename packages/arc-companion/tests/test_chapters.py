from __future__ import annotations

import json
import threading
import time

import pytest

from arc_companion.chapter_glossary import generate_index_glossary, project_chapter_glossary
from arc_companion.chapter_guide import generate_chapter_guide
from arc_companion.chapter_scheduler import run_chapter_pipeline
from arc_companion.chapters import ChapterStructureError, build_chapters
from arc_companion.ledger import (
    advance_block,
    initialize_lane_ledger,
    invalidate_suffix,
    mark_needs_supervision,
    next_pending,
)
from arc_companion.progress import CompanionProgress


def _document() -> dict:
    return {"blocks": [
        {"block_id": "b1", "type": "heading", "level": 1, "title": "First"},
        {"block_id": "b2", "type": "prose", "text": "A gauge field."},
        {"block_id": "b3", "type": "heading", "level": 1, "title": "Second"},
        {"block_id": "b4", "type": "prose", "text": "A fermion."},
        {"block_id": "b5", "type": "heading", "title": "Index", "source_role": "index"},
    ]}


def test_chapters_have_stable_ids_and_exact_substantive_coverage() -> None:
    result = build_chapters(_document())
    assert [item["chapter_id"] for item in result["chapters"]] == ["ch-0001", "ch-0002"]
    assert [value for item in result["chapters"] for value in item["block_ids"]] == ["b1", "b2", "b3", "b4"]
    assert result["excluded_block_ids"] == ["b5"]


def test_authoritative_chapter_gap_is_rejected() -> None:
    with pytest.raises(ChapterStructureError, match="contiguous|cover"):
        build_chapters(_document(), structure={"chapters": [
            {"title": "First", "block_ids": ["b1", "b2"]},
            {"title": "Second", "block_ids": ["b4"]},
        ]})


def test_chapter_glossary_uses_source_and_index_overlap_and_keeps_parent() -> None:
    chapter = {"chapter_id": "ch-0001", "block_ids": ["b1", "b2"], "page_start": 10, "page_end": 20}
    glossary = {"entries": [
        {"entry_id": "parent", "source": "field", "target": "场"},
        {"entry_id": "child", "parent_id": "parent", "source": "Ward identity", "target": "沃德恒等式"},
        {"entry_id": "outside", "source": "fermion", "target": "费米子"},
    ]}
    index = [{"term": "Ward identity", "page_ranges": [{"start": 12, "end": 12}]}]
    result = project_chapter_glossary(chapter, _document(), glossary, index_entries=index)
    assert [item["entry_id"] for item in result["entries"]] == ["parent", "child"]


def test_chapter_glossary_keeps_every_ancestor_in_global_order() -> None:
    chapter = {"chapter_id": "ch-0001", "block_ids": ["b1", "b2"]}
    glossary = {"entries": [
        {"entry_id": "grand", "source": "symmetry", "target": "对称性"},
        {"entry_id": "parent", "parent_id": "grand", "source": "current", "target": "流"},
        {"entry_id": "child", "parent_id": "parent", "source": "gauge field", "target": "规范场"},
        {"entry_id": "outside", "source": "fermion", "target": "费米子"},
    ]}
    result = project_chapter_glossary(chapter, _document(), glossary)
    assert [item["entry_id"] for item in result["entries"]] == ["grand", "parent", "child"]


def test_chapter_glossary_assigns_stable_ids_to_non_index_terms() -> None:
    result = project_chapter_glossary(
        {"chapter_id": "ch-0001", "block_ids": ["b1", "b2"]},
        _document(), {"entries": [{"source": "gauge field", "target": "规范场"}]},
    )
    assert result["entries"][0]["entry_id"] == "term-0001"


def test_lane_ledger_orders_states_and_preserves_accepted_prefix(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    ledger = initialize_lane_ledger(path, chapter_id="ch-0001", lane="translation", segment_ids=["s1", "s2"])
    assert next_pending(ledger) == "s1"
    for state in ("submitted", "schema_valid", "invariant_valid", "accepted"):
        ledger = advance_block(path, segment_id="s1", state=state, input_sha256="i", output_sha256="o")
    assert next_pending(ledger) == "s2"
    ledger = mark_needs_supervision(path, segment_id="s2", reason="timeout", recovery_context={"submission_state": "unknown"})
    assert next_pending(ledger) is None
    ledger = invalidate_suffix(path, from_segment_id="s2", generation=2)
    assert ledger["blocks"][0]["state"] == "accepted"
    assert ledger["blocks"][1] == {"segment_id": "s2", "state": "pending", "generation": 2}


def test_build_review_is_emitted_only_at_a_safe_boundary(tmp_path) -> None:
    ticks = iter([0.0, 100.0, 1900.0])
    progress = CompanionProgress(tmp_path / "progress.jsonl", clock=lambda: next(ticks))
    assert len(progress.safe_boundary("chapter_prepared", chapter_id="ch-0001")) == 1
    events = progress.safe_boundary("block_accepted", segment_id="s1")
    assert [item["event"] for item in events] == ["block_accepted", "review_due"]
    written = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert written[-1]["review_sequence"] == 1


def test_concurrent_safe_boundaries_emit_one_review_sequence(tmp_path) -> None:
    class Clock:
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()
        def __call__(self):
            with self.lock:
                self.calls += 1
                return 0.0 if self.calls == 1 else 2000.0
    progress = CompanionProgress(tmp_path / "progress.jsonl", clock=Clock())
    threads = [threading.Thread(target=progress.safe_boundary, args=("block_accepted",), kwargs={"segment_id": str(index)}) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    events = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [item["review_sequence"] for item in events if item["event"] == "review_due"] == [1]


def test_real_index_glossary_has_no_entry_cap_or_loss(tmp_path) -> None:
    index = [{"term": f"Term {number}", "pages": [number]} for number in range(205)]
    def call_model(prompt, schema, artifact_dir, label):
        payload = json.loads(prompt.split("\n", 1)[1])
        return {"entries": [{"entry_id": item["entry_id"], "target": "译", "explanation": "释"} for item in payload]}
    result = generate_index_glossary(index, language="zh-CN", checkpoint_dir=tmp_path, force=False, call_model=call_model)
    assert result["entry_limit"] is None
    assert len(result["entries"]) == 205
    assert [item["source"] for item in result["entries"]] == [f"Term {number}" for number in range(205)]


def test_chapter_scheduler_keeps_lane_order_and_global_budget() -> None:
    lock = threading.Lock()
    active = 0
    maximum = 0
    calls: list[tuple[str, str, str]] = []
    def enter(kind, chapter, segment=None):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.005)
        with lock:
            calls.append((kind, chapter["chapter_id"], str((segment or {}).get("segment_id") or "")))
            active -= 1
        return {"ok": True}
    chapters = [{"chapter_id": "ch-0001"}, {"chapter_id": "ch-0002"}]
    result = run_chapter_pipeline(
        chapters, workers=2,
        prepare_guide=lambda chapter: enter("guide", chapter),
        prepare_segments=lambda chapter: (enter("segment", chapter), [{"segment_id": f"{chapter['chapter_id']}.seg-0001"}, {"segment_id": f"{chapter['chapter_id']}.seg-0002"}])[1],
        run_translation=lambda prepared, segment: enter("translation", prepared.chapter, segment),
        run_companion=lambda prepared, segment: enter("companion", prepared.chapter, segment),
    )
    assert maximum <= 2
    for chapter_id in result:
        for lane in ("translation", "companion"):
            assert [item[2] for item in calls if item[:2] == (lane, chapter_id)] == [f"{chapter_id}.seg-0001", f"{chapter_id}.seg-0002"]


def test_interactive_scheduler_never_starts_second_chapter() -> None:
    seen: list[str] = []
    result = run_chapter_pipeline(
        [{"chapter_id": "ch-0001"}, {"chapter_id": "ch-0002"}], workers=4,
        prepare_guide=lambda chapter: seen.append(chapter["chapter_id"]) or {},
        prepare_segments=lambda chapter: [{"segment_id": f"{chapter['chapter_id']}.seg-0001"}],
        run_translation=lambda prepared, segment: {}, run_companion=lambda prepared, segment: {},
        stop_after_first_chapter=True,
    )
    assert seen == ["ch-0001"]
    assert list(result) == ["ch-0001"]


def test_chapter_scheduler_can_disable_translation_lane_completely() -> None:
    companion_calls: list[str] = []
    result = run_chapter_pipeline(
        [{"chapter_id": "ch-0001"}],
        workers=2,
        prepare_guide=lambda _chapter: {"main_content": "Guide"},
        prepare_segments=lambda _chapter: [
            {"segment_id": "ch-0001.seg-0001"},
            {"segment_id": "ch-0001.seg-0002"},
        ],
        run_translation=None,
        run_companion=lambda _prepared, segment: companion_calls.append(
            str(segment["segment_id"])
        ) or {"explanation": "Commentary"},
    )

    assert companion_calls == ["ch-0001.seg-0001", "ch-0001.seg-0002"]
    assert result["ch-0001"]["translation"] == {}
    assert set(result["ch-0001"]["companion"]) == {
        "ch-0001.seg-0001",
        "ch-0001.seg-0002",
    }


def test_scheduler_stops_queued_calls_after_a_lane_failure() -> None:
    paid: list[str] = []

    def fail(_prepared, segment):
        paid.append(str(segment["segment_id"]))
        raise TimeoutError("submission state unknown")

    with pytest.raises(TimeoutError, match="submission state unknown"):
        run_chapter_pipeline(
            [{"chapter_id": "ch-0001"}], workers=1,
            prepare_guide=lambda _chapter: {},
            prepare_segments=lambda _chapter: [
                {"segment_id": "s1"}, {"segment_id": "s2"},
            ],
            run_translation=fail,
            run_companion=lambda _prepared, segment: paid.append(
                f"companion:{segment['segment_id']}"
            ),
        )

    assert paid == ["s1"]


def test_chapter_guide_rejects_unverified_supplementary_reading(tmp_path) -> None:
    with pytest.raises(ValueError, match="unverified"):
        generate_chapter_guide(
            {"chapter_id": "ch-0001", "title": "One"}, [{"block_id": "b1", "text": "Source"}],
            language="zh-CN", evidence={"papers": [{"evidence_id": "ok", "title": "Known"}]},
            checkpoint_dir=tmp_path, force=True,
            call_model=lambda *args: {"motivation": None, "main_content": "内容", "section_logic": None, "book_position": None, "prerequisites": None, "supplementary_reading": [{"title": "Made up", "identifier": None, "reason": "x", "evidence_id": "bad"}]},
        )


def test_chapter_guide_excludes_original_bibliography_by_all_normalized_identities(tmp_path) -> None:
    captured: list[str] = []

    def model(prompt, *_args):
        captured.append(prompt)
        return {"motivation": None, "main_content": None, "section_logic": None,
                "book_position": None, "prerequisites": None, "supplementary_reading": []}

    evidence = {
        "bibliography": [
            {"doi": "https://doi.org/10.1000/ABC."},
            {"arxiv_id": "arXiv:2401.01234v2"},
            {"title": "A  Third—Paper!"},
        ],
        "related_papers": [
            {"evidence_id": "by-doi", "doi": "10.1000/abc", "title": "Different"},
            {"evidence_id": "by-arxiv", "arxiv_id": "2401.01234", "title": "Different 2"},
            {"evidence_id": "by-title", "title": "A third paper"},
            {"evidence_id": "new", "doi": "10.1000/new", "title": "New paper"},
        ],
    }
    generate_chapter_guide(
        {"chapter_id": "ch-0001"}, [{"block_id": "b1", "text": "Source"}],
        language="zh-CN", evidence=evidence, checkpoint_dir=tmp_path, force=True,
        call_model=model,
    )
    assert "by-doi" not in captured[0]
    assert "by-arxiv" not in captured[0]
    assert "by-title" not in captured[0]
    assert '"evidence_id": "new"' in captured[0]
