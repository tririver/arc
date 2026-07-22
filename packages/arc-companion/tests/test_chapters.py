from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import pytest

from arc_companion.chapter_glossary import generate_index_glossary, project_segment_glossary
from arc_companion.chapter_guide import (
    CHAPTER_GUIDE_SCHEMA,
    CHAPTER_GUIDE_VERSION,
    chapter_guide_artifact_valid,
    generate_chapter_guide,
)
from arc_companion.chapter_scheduler import run_chapter_pipeline
from arc_companion.chapters import ChapterStructureError, build_chapters
from arc_companion.ledger import (
    LaneLedgerError,
    accept_deferred_block,
    advance_block,
    initialize_lane_ledger,
    invalidate_suffix,
    mark_needs_supervision,
    next_pending,
)
from arc_companion.io import sha256_json
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
    assert result["schema_version"] == "arc.companion.chapters.v2"
    assert result["chapters"][0]["title_block_ids"] == ["b1"]
    assert result["chapters"][0]["structural_block_ids"] == ["b1"]
    assert result["chapters"][0]["content_block_ids"] == ["b2"]


def test_chapters_have_no_fixed_count_limit() -> None:
    blocks = []
    for index in range(1, 26):
        blocks.extend([
            {
                "block_id": f"h{index:04d}",
                "type": "heading",
                "level": 1,
                "title": f"Chapter {index}",
            },
            {
                "block_id": f"p{index:04d}",
                "type": "prose",
                "text": f"Body {index}.",
            },
        ])

    result = build_chapters({"blocks": blocks})

    assert len(result["chapters"]) == 25
    assert [item["chapter_id"] for item in result["chapters"]] == [
        f"ch-{index:04d}" for index in range(1, 26)
    ]
    assert [
        block_id
        for chapter in result["chapters"]
        for block_id in chapter["block_ids"]
    ] == [str(block["block_id"]) for block in blocks]


def test_authoritative_chapter_v2_partitions_all_heading_levels_without_guessing() -> None:
    kinds = ["part", "chapter", "heading", "section", "subsection", "subsubsection"]
    blocks = [
        {"block_id": f"h{index}", "type": kind, "title": f"Heading {index}"}
        for index, kind in enumerate(kinds, 1)
    ]
    blocks.extend([
        {"block_id": "p1", "type": "prose", "text": "Body."},
        {"block_id": "e1", "type": "equation", "text": "x=y"},
    ])
    result = build_chapters(
        {"blocks": blocks},
        structure={"chapters": [{
            "title": "Heading 2",
            "block_ids": [item["block_id"] for item in blocks],
        }]},
    )
    chapter = result["chapters"][0]
    assert chapter["title_block_ids"] == ["h2"]
    assert chapter["structural_block_ids"] == [f"h{index}" for index in range(1, 7)]
    assert chapter["content_block_ids"] == ["p1", "e1"]

    unmatched = build_chapters(
        {"blocks": blocks},
        structure={"chapters": [{
            "title": "I PART",
            "block_ids": [item["block_id"] for item in blocks],
        }]},
    )["chapters"][0]
    assert unmatched["title_block_ids"] == []


def test_authoritative_chapter_gap_is_rejected() -> None:
    with pytest.raises(ChapterStructureError, match="contiguous|cover"):
        build_chapters(_document(), structure={"chapters": [
            {"title": "First", "block_ids": ["b1", "b2"]},
            {"title": "Second", "block_ids": ["b4"]},
        ]})


def test_segment_glossary_projects_only_terms_in_source_and_keeps_lineage() -> None:
    glossary = {"entries": [
        {"entry_id": "parent", "source": "field", "target": "场"},
        {"entry_id": "child", "parent_id": "parent", "source": "Ward identity", "target": "沃德恒等式"},
        {"entry_id": "outside", "source": "fermion", "target": "费米子"},
    ]}
    result = project_segment_glossary([{"text": "The Ward identity is useful."}], glossary)
    assert [item["entry_id"] for item in result["entries"]] == ["child"]
    assert result["entries"][0]["lineage"] == [
        {"entry_id": "parent", "source": "field", "target": "场"}
    ]


def test_segment_glossary_keeps_every_ancestor_as_ordered_lineage() -> None:
    glossary = {"entries": [
        {"entry_id": "grand", "source": "symmetry", "target": "对称性"},
        {"entry_id": "parent", "parent_id": "grand", "source": "current", "target": "流"},
        {"entry_id": "child", "parent_id": "parent", "source": "gauge field", "target": "规范场"},
        {"entry_id": "outside", "source": "fermion", "target": "费米子"},
    ]}
    result = project_segment_glossary([{"text": "A gauge field."}], glossary)
    assert [item["entry_id"] for item in result["entries"]] == ["child"]
    assert [item["entry_id"] for item in result["entries"][0]["lineage"]] == ["grand", "parent"]


def test_segment_glossary_assigns_stable_ids_to_non_index_terms() -> None:
    result = project_segment_glossary(
        [{"text": "A gauge field."}],
        {"entries": [{"source": "gauge field", "target": "规范场"}]},
    )
    assert result["entries"][0]["entry_id"] == "term-0001"


def test_segment_glossary_normalizes_aliases_nfkc_case_and_boundaries() -> None:
    glossary = {"entries": [
        {"source_term": "Gauge Field", "target_term": "规范场", "source_aliases": ["ＧＦ"]},
        {"term": "fermion", "translation": "费米子"},
    ]}
    result = project_segment_glossary([{"text": "A gf, not fermionic matter."}], glossary)
    assert [item["source"] for item in result["entries"]] == ["Gauge Field"]
    assert result["entries"][0]["aliases"] == ["ＧＦ"]


def test_segment_glossary_uses_unicode_latin_and_decimal_boundaries() -> None:
    glossary = {"entries": [
        {"source": "résumé", "target": "简历"},
        {"source": "phase 2", "target": "第二阶段"},
    ]}
    plural = project_segment_glossary(
        [{"text": "Several résumés and phase 2٢ variants."}], glossary,
    )
    assert plural["entries"] == []
    exact = project_segment_glossary(
        [{"text": "The RÉSUMÉ ends; PHASE ２ begins."}], glossary,
    )
    assert [item["source"] for item in exact["entries"]] == ["résumé", "phase 2"]


def test_segment_glossary_uses_unicode_boundaries_beyond_latin_and_keeps_cjk_runs() -> None:
    glossary = {"entries": [
        {"source": "Ландау", "target": "Landau"},
        {"source": "量子", "target": "quantum"},
    ]}
    embedded = project_segment_glossary(
        [{"text": "сЛандау discusses 量子场论."}], glossary,
    )
    assert [item["source"] for item in embedded["entries"]] == ["量子"]

    exact = project_segment_glossary(
        [{"text": "Теория Ландау discusses 量子场论."}], glossary,
    )
    assert [item["source"] for item in exact["entries"]] == ["Ландау", "量子"]


def test_segment_glossary_preserves_empty_target_and_global_order() -> None:
    glossary = {"entries": [
        {"source": "second", "target": ""},
        {"source": "first", "target": "一"},
    ]}
    result = project_segment_glossary([{"text": "first and second"}], glossary)
    assert [item["entry_id"] for item in result["entries"]] == ["term-0001", "term-0002"]
    assert result["entries"][0]["target"] == ""
    assert result["counts"] == {"source_entries": 2, "matched_entries": 2}


def test_lane_ledger_orders_states_and_preserves_accepted_prefix(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    ledger = initialize_lane_ledger(path, chapter_id="ch-0001", lane="translation", segment_ids=["s1", "s2"])
    assert next_pending(ledger) == "s1"
    assert ledger["blocks"][0]["state"] == "prepared"
    assert ledger["blocks"][0]["submission_state"] == "not_submitted"
    for state in ("submitted", "response_received", "schema_valid", "invariant_valid", "accepted"):
        ledger = advance_block(path, segment_id="s1", state=state, input_sha256="i", output_sha256="o")
    assert next_pending(ledger) == "s2"
    ledger = mark_needs_supervision(path, segment_id="s2", reason="timeout", recovery_context={"submission_state": "unknown"})
    assert next_pending(ledger) is None
    ledger = invalidate_suffix(path, from_segment_id="s2", generation=2)
    assert ledger["blocks"][0]["state"] == "accepted"
    assert ledger["blocks"][1] == {
        "segment_id": "s2", "state": "prepared",
        "submission_state": "not_submitted", "generation": 2,
    }


def test_targeted_suffix_stage_survives_reentry_and_rebuilds_chain(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    staged_output = {
        "blocks": [{"block_id": "b3", "translation": "three"}],
    }
    staged_output_sha256 = sha256_json(staged_output)
    initialize_lane_ledger(
        path, chapter_id="ch-0001", lane="translation",
        segment_ids=["s1", "s2", "s3"],
    )
    for segment_id in ("s1", "s2", "s3"):
        for state in (
            "submitted", "response_received", "schema_valid",
            "invariant_valid", "accepted",
        ):
            advance_block(
                path, segment_id=segment_id, state=state,
                input_sha256=f"old-input-{segment_id}",
                output_sha256=(
                    staged_output_sha256
                    if segment_id == "s3" else f"old-output-{segment_id}"
                ),
                receipt={"call_id": f"old-call-{segment_id}"},
                validation_receipt={"old_validation": True},
            )

    staged = {
        "s3": {
            "output": staged_output,
            "output_sha256": staged_output_sha256,
            "logical_receipt": {
                "kind": "targeted_regeneration_suffix_stage",
                "provider_calls": 0,
                "source_logical_receipt": {"call_id": "old-call-s3"},
            },
            "validation_receipt": {"staged_before_targeted_invalidation": True},
        },
    }
    invalidate_suffix(
        path, from_segment_id="s2", generation=2, staged_outputs=staged,
    )

    # Simulate a crash immediately after invalidation. Re-entering the same
    # targeted request must retain the durable suffix candidate without also
    # turning the selected target into a reuse candidate.
    ledger = invalidate_suffix(path, from_segment_id="s2", generation=3)
    assert "deferred_output" not in ledger["blocks"][1]
    assert ledger["blocks"][2]["deferred_output"] == staged["s3"]["output"]

    for state in (
        "submitted", "response_received", "schema_valid",
        "invariant_valid", "accepted",
    ):
        ledger = advance_block(
            path, segment_id="s2", state=state,
            input_sha256="new-input-s2", output_sha256="new-output-s2",
        )
    s2_chain = ledger["blocks"][1]["accepted_chain_sha256"]
    with pytest.raises(LaneLedgerError, match="deferred output hash changed"):
        accept_deferred_block(
            path, segment_id="s3", input_sha256="new-input-s3",
            output_sha256="tampered-output",
            logical_receipt=ledger["blocks"][2]["deferred_logical_receipt"],
            validation_receipt=ledger["blocks"][2]["deferred_validation_receipt"],
        )
    ledger = accept_deferred_block(
        path, segment_id="s3", input_sha256="new-input-s3",
        output_sha256=staged_output_sha256,
        logical_receipt=ledger["blocks"][2]["deferred_logical_receipt"],
        validation_receipt=ledger["blocks"][2]["deferred_validation_receipt"],
    )
    s3 = ledger["blocks"][2]
    assert s3["state"] == "accepted"
    assert s3["generation"] == 3
    assert s3["predecessor_accepted_chain_sha256"] == s2_chain
    assert ledger["accepted_chain_sha256"] == s3["accepted_chain_sha256"]
    assert not any(key.startswith("deferred_") for key in s3)
    assert s3["logical_receipt"]["source_logical_receipt"] == {
        "call_id": "old-call-s3"
    }


def test_lane_ledger_v1_upgrade_preserves_uncertain_submission(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        "schema_version": "arc.companion.chapter-lane-ledger.v1",
        "chapter_id": "ch-0001",
        "lane": "translation",
        "generation": 1,
        "needs_supervision": None,
        "blocks": [
            {"segment_id": "s1", "state": "pending", "generation": 1},
            {"segment_id": "s2", "state": "submitted", "generation": 1},
        ],
        "accepted_chain_sha256": "chain",
    }), encoding="utf-8")

    ledger = initialize_lane_ledger(
        path,
        chapter_id="ch-0001",
        lane="translation",
        segment_ids=["s1", "s2"],
    )

    assert ledger["schema_version"].endswith(".v2")
    assert ledger["blocks"][0]["state"] == "prepared"
    assert ledger["blocks"][0]["submission_state"] == "not_submitted"
    assert ledger["blocks"][1]["state"] == "submitted"
    assert ledger["blocks"][1]["submission_state"] == "submitted"


def test_invalidate_suffix_archives_all_suffix_supervision_idempotently(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    initialize_lane_ledger(
        path, chapter_id="ch-0001", lane="translation",
        segment_ids=["s1", "s2", "s3"],
    )
    for state in ("submitted", "response_received", "schema_valid", "invariant_valid", "accepted"):
        advance_block(path, segment_id="s1", state=state)
    mark_needs_supervision(
        path, segment_id="s2", reason="native session missing",
        recovery_context={"submission_state": "submitted"},
    )
    mark_needs_supervision(
        path, segment_id="s3", reason="invalid paid response",
        recovery_context={"submission_state": "submitted"},
    )

    first = invalidate_suffix(path, from_segment_id="s2", generation=2)
    second = invalidate_suffix(path, from_segment_id="s2", generation=2)

    assert first["needs_supervision"] is None
    assert first["supervision_entries"] == []
    assert [item["segment_id"] for item in first["supervision_history"]] == ["s2", "s3"]
    assert all(item["source_generation"] == 1 for item in first["supervision_history"])
    assert all(item["target_generation"] == 2 for item in first["supervision_history"])
    assert all(item["suffix_start_segment_id"] == "s2" for item in first["supervision_history"])
    assert second["supervision_history"] == first["supervision_history"]


def test_scheduler_local_failure_drains_other_active_lane() -> None:
    both_active = threading.Barrier(2)
    release_companion = threading.Event()
    companion_completed = threading.Event()

    def translation(_prepared, _segment):
        both_active.wait(timeout=5)
        raise ValueError("ordinary local lane failure")

    def companion(_prepared, _segment):
        both_active.wait(timeout=5)
        assert release_companion.wait(timeout=5)
        companion_completed.set()
        return {"ok": True}

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            run_chapter_pipeline,
            [{"chapter_id": "ch-0001"}],
            workers=2,
            prepare_guide=lambda _chapter: {},
            prepare_segments=lambda _chapter: [{"segment_id": "s1"}],
            run_translation=translation,
            run_companion=companion,
        )
        release_companion.set()
        with pytest.raises(ValueError, match="ordinary local lane failure"):
            result.result(timeout=5)

    assert companion_completed.is_set()


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

    assert paid == ["s1", "companion:s1", "companion:s2"]


def test_chapter_guide_rejects_unverified_supplementary_reading(tmp_path) -> None:
    with pytest.raises(ValueError, match="unverified"):
        generate_chapter_guide(
            {"chapter_id": "ch-0001", "title": "One"}, [{"block_id": "b1", "text": "Source"}],
            language="zh-CN", evidence={"papers": [{"evidence_id": "ok", "title": "Known"}]},
            checkpoint_dir=tmp_path, force=True,
            call_model=lambda *args: {
                "motivation": None, "main_content": "内容", "section_logic": None,
                "prerequisites": None, "pedagogical_comparison": None,
                "historical_context": [], "supplementary_reading": [{
                    "title": "Made up", "identifier": None, "reason": "x",
                    "evidence_id": "bad",
                }],
            },
        )


def test_chapter_guide_excludes_original_bibliography_by_all_normalized_identities(tmp_path) -> None:
    captured: list[str] = []

    def model(prompt, *_args):
        captured.append(prompt)
        return {
            "motivation": None, "main_content": None, "section_logic": None,
            "prerequisites": None, "pedagogical_comparison": None,
            "historical_context": [], "supplementary_reading": [],
        }

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
    assert "pedagogical_comparison" in captured[0]
    assert "historical_context" in captured[0]
    assert "book_position" not in captured[0]
    assert "never describe or predict pagination in the generated companion document" in captured[0]


def test_chapter_guide_v3_schema_has_ordered_nullable_and_sourced_fields() -> None:
    assert list(CHAPTER_GUIDE_SCHEMA["properties"]) == [
        "motivation", "main_content", "section_logic", "prerequisites",
        "pedagogical_comparison", "historical_context", "supplementary_reading",
    ]
    assert "book_position" not in str(CHAPTER_GUIDE_SCHEMA)
    comparison = CHAPTER_GUIDE_SCHEMA["properties"]["pedagogical_comparison"]
    assert comparison["type"] == ["object", "null"]
    assert comparison["properties"]["sources"]["minItems"] == 1
    history = CHAPTER_GUIDE_SCHEMA["properties"]["historical_context"]
    assert history["maxItems"] == 3
    assert history["items"]["properties"]["sources"]["maxItems"] == 3


def test_chapter_guide_artifact_validator_rejects_v2_and_extra_fields() -> None:
    valid = {
        "schema_version": CHAPTER_GUIDE_VERSION,
        "source_sha256": "source",
        "chapter_id": "ch-0001",
        "motivation": None,
        "main_content": "Content",
        "section_logic": None,
        "prerequisites": None,
        "pedagogical_comparison": None,
        "historical_context": [],
        "supplementary_reading": [],
    }
    assert chapter_guide_artifact_valid(valid)
    assert not chapter_guide_artifact_valid({
        **valid,
        "schema_version": "arc.companion.chapter-guide.v2",
        "book_position": "First",
    })
    assert not chapter_guide_artifact_valid({**valid, "book_position": "First"})


def test_chapter_guide_artifact_validator_enforces_offline_evidence_policy() -> None:
    guide = {
        "schema_version": CHAPTER_GUIDE_VERSION,
        "source_sha256": "source",
        "chapter_id": "ch-0001",
        "motivation": None,
        "main_content": None,
        "section_logic": None,
        "prerequisites": None,
        "pedagogical_comparison": {
            "text": "The textbook reverses the order.",
            "sources": [{
                "title": "Textbook",
                "url": "https://example.test/textbook",
                "locator": "Chapter 2",
            }],
        },
        "historical_context": [],
        "supplementary_reading": [{
            "title": "Textbook", "identifier": None, "reason": "Comparison",
            "evidence_id": "known",
        }],
    }
    evidence = {"related_papers": [{
        "evidence_id": "known", "title": "Textbook", "doi": "10.1000/known",
        "url": "https://example.test/textbook",
    }]}

    assert chapter_guide_artifact_valid(
        guide, evidence=evidence, allow_internet=False,
    )
    assert not chapter_guide_artifact_valid(
        guide, evidence={}, allow_internet=False,
    )
    assert chapter_guide_artifact_valid(
        {**guide, "supplementary_reading": []},
        evidence={}, allow_internet=True,
    )


def test_chapter_guide_offline_sources_are_limited_to_prompt_evidence_urls(tmp_path) -> None:
    evidence = {"related_papers": [{
        "evidence_id": "known", "title": "Known textbook", "doi": "10.1000/known",
        "url": "https://example.test/known",
    }]}

    def result(url: str) -> dict:
        return {
            "motivation": None, "main_content": None, "section_logic": None,
            "prerequisites": None,
            "pedagogical_comparison": {
                "text": "The reference reverses the teaching order.",
                "sources": [{
                    "title": "Known textbook", "url": url, "locator": "Chapter 2",
                }],
            },
            "historical_context": [], "supplementary_reading": [],
        }

    accepted = generate_chapter_guide(
        {"chapter_id": "ch-0001"}, [{"block_id": "b1", "text": "Source"}],
        language="zh-CN", evidence=evidence, checkpoint_dir=tmp_path / "accepted",
        force=True, allow_internet=False,
        call_model=lambda *_args: result("https://example.test/known"),
    )
    assert accepted["pedagogical_comparison"]["sources"][0]["locator"] == "Chapter 2"

    with pytest.raises(ValueError, match="outside supplied local evidence"):
        generate_chapter_guide(
            {"chapter_id": "ch-0001"}, [{"block_id": "b1", "text": "Source"}],
            language="zh-CN", evidence=evidence, checkpoint_dir=tmp_path / "rejected",
            force=True, allow_internet=False,
            call_model=lambda *_args: result("https://outside.test/source"),
        )


def test_empty_chapter_guide_uses_silent_nulls_and_arrays(tmp_path) -> None:
    captured: list[str] = []

    def model(prompt, *_args):
        captured.append(prompt)
        return {
            "motivation": None, "main_content": None, "section_logic": None,
            "prerequisites": None, "pedagogical_comparison": None,
            "historical_context": [], "supplementary_reading": [],
        }

    guide = generate_chapter_guide(
        {"chapter_id": "ch-empty"}, [{"block_id": "b1", "text": "Source"}],
        language="zh-CN", evidence={}, checkpoint_dir=tmp_path, force=True,
        call_model=model,
    )

    assert guide["pedagogical_comparison"] is None
    assert guide["historical_context"] == []
    assert "never explain missing material" in captured[0]


def test_chapter_guide_cache_identity_includes_host_tool_policy(tmp_path) -> None:
    calls = 0

    def model(*_args):
        nonlocal calls
        calls += 1
        return {
            "motivation": None, "main_content": None, "section_logic": None,
            "prerequisites": None, "pedagogical_comparison": None,
            "historical_context": [], "supplementary_reading": [],
        }

    arguments = {
        "chapter": {"chapter_id": "ch-cache"},
        "source_blocks": [{"block_id": "b1", "text": "Source"}],
        "language": "zh-CN", "evidence": {}, "checkpoint_dir": tmp_path,
        "force": False, "call_model": model, "allow_internet": True,
    }
    first = generate_chapter_guide(**arguments, inherit_host_tools=False)
    cached = generate_chapter_guide(**arguments, inherit_host_tools=False)
    changed = generate_chapter_guide(**arguments, inherit_host_tools=True)

    assert first["source_sha256"] == cached["source_sha256"]
    assert changed["source_sha256"] != first["source_sha256"]
    assert calls == 2


def test_chapter_guide_cache_identity_includes_full_lane_recipe(tmp_path) -> None:
    calls = 0

    def model(*_args):
        nonlocal calls
        calls += 1
        return {
            "motivation": None, "main_content": None, "section_logic": None,
            "prerequisites": None, "pedagogical_comparison": None,
            "historical_context": [], "supplementary_reading": [],
        }

    arguments = {
        "chapter": {"chapter_id": "ch-recipe"},
        "source_blocks": [{"block_id": "b1", "text": "Source"}],
        "language": "zh-CN", "evidence": {}, "checkpoint_dir": tmp_path,
        "force": False, "call_model": model,
    }
    first = generate_chapter_guide(**arguments, recipe_identity="provider-a:model-a")
    cached = generate_chapter_guide(**arguments, recipe_identity="provider-a:model-a")
    changed = generate_chapter_guide(**arguments, recipe_identity="provider-b:model-a")

    assert first["source_sha256"] == cached["source_sha256"]
    assert changed["source_sha256"] != first["source_sha256"]
    assert calls == 2
