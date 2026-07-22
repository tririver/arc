from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import threading

import pytest

from arc_companion.prompts import CUT_SCHEMA
from arc_companion.projection import annotation_input_block
from arc_companion.pipeline import _segment_input_hash
from arc_companion.segmentation import (
    SegmentationError,
    build_block_inventory,
    build_segmentation_windows,
    construct_segments_from_cuts,
    segment_document,
    validate_exact_coverage,
)


def _document(kinds: list[str], *, section_size: int = 10, text_size: int = 24) -> dict:
    blocks = []
    equations = []
    figures = []
    tables = []
    bibliography = []
    for index, kind in enumerate(kinds, start=1):
        block_id = f"b{index:04d}"
        section = f"S{(index - 1) // section_size + 1}"
        block = {
            "block_id": block_id,
            "source_id": block_id,
            "order": index,
            "kind": kind,
            "section_id": section,
            "text": f"source-{index}-" + ("x" * text_size),
        }
        if kind == "heading":
            block["title"] = f"Section {index}"
        blocks.append(block)
        if kind == "equation":
            equations.append({
                "id": block_id,
                "tex": [f"x_{{{index}}}=y_{{{index}}}"],
                "printed_equation_numbers": [str(index)],
            })
        elif kind == "figure":
            figures.append({"id": block_id, "tag": f"Figure {index}", "caption": f"Caption {index}"})
        elif kind == "table":
            tables.append({"id": block_id, "tag": f"Table {index}", "caption": f"Rows {index}"})
        elif kind == "bibliography":
            bibliography.append({"id": block_id, "label": f"[{index}]", "text": f"Reference {index}"})
    return {
        "blocks": blocks,
        "equations": equations,
        "figures": figures,
        "tables": tables,
        "bibliography": bibliography,
    }


def _mixed_610_document() -> dict:
    counts = {
        "equation": 273,
        "prose": 261,
        "bibliography": 33,
        "heading": 23,
        "figure": 11,
        "footnote": 6,
        "table": 2,
        "list": 1,
    }
    kinds = [kind for kind, count in counts.items() for _ in range(count)]
    # Interleave unlike blocks to model the observed paper rather than giving
    # the inventory one artificial run per kind.
    kinds = [kinds[(index * 197) % len(kinds)] for index in range(len(kinds))]
    assert len(kinds) == 610
    return _document(kinds, section_size=27)


def _cut_model(cuts_by_window: dict[str, list[int]], calls: list[str] | None = None):
    def call_model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        if calls is not None:
            calls.append(call_label)
        for marker, cuts in cuts_by_window.items():
            if marker in call_label:
                return {"cut_after_ordinals": cuts}
        return {"cut_after_ordinals": []}

    return call_model


def test_inventory_uses_stable_ordinals_and_type_specific_compact_fields() -> None:
    document = _document(
        ["heading", "prose", "equation", "figure", "table", "bibliography"],
        section_size=3,
    )
    document["blocks"][2]["text"] = "NOISY PRESENTATION MATH THAT MUST NOT WIN"

    inventory = build_block_inventory(document)

    assert [item["ordinal"] for item in inventory] == list(range(1, 7))
    assert all("id" not in item and "block_id" not in item for item in inventory)
    assert inventory[0]["title"] == "Section 1"
    assert inventory[1]["text"].startswith("source-2-")
    assert inventory[2]["formula"] == "x_{3}=y_{3}"
    assert inventory[2]["numbers"] == ["3"]
    assert "NOISY" not in inventory[2]["formula"]
    assert inventory[3]["caption"] == "Caption 4"
    assert inventory[4]["caption"] == "Rows 5"
    assert inventory[5]["citation"] == "Reference 6"


def test_windows_follow_section_runs_and_caps_with_read_only_context() -> None:
    inventory = build_block_inventory(_document(["prose"] * 7, section_size=3))

    windows = build_segmentation_windows(inventory, max_blocks=2, max_projected_chars=30_000)

    assert [(item["start_ordinal"], item["end_ordinal"]) for item in windows] == [
        (1, 2),
        (3, 3),
        (4, 5),
        (6, 6),
        (7, 7),
    ]
    assert windows[1]["context_before"][0]["ordinal"] == 2
    assert windows[1]["context_after"][0]["ordinal"] == 4
    assert windows[-1]["context_after"] == []


def test_cut_only_construction_is_deterministic_and_exact() -> None:
    document = _document(["prose"] * 8)

    first = construct_segments_from_cuts([6, 2, 4], document)
    second = construct_segments_from_cuts([4, 6, 2], document)

    assert first == second
    assert [item["segment_id"] for item in first] == ["seg-0001", "seg-0002", "seg-0003", "seg-0004"]
    assert [item["block_ids"] for item in first] == [
        ["b0001", "b0002"],
        ["b0003", "b0004"],
        ["b0005", "b0006"],
        ["b0007", "b0008"],
    ]
    validate_exact_coverage(first, document["blocks"])


def test_segments_preserve_structural_navigation_but_project_augmentation_ids() -> None:
    document = _document(["heading", "prose", "heading", "equation"])

    segments = construct_segments_from_cuts([1, 3], document)

    assert segments[0]["block_ids"] == ["b0001"]
    assert segments[0]["augmentation_block_ids"] == []
    assert segments[0]["structural_only"] is True
    assert segments[1]["block_ids"] == ["b0002", "b0003"]
    assert segments[1]["augmentation_block_ids"] == ["b0002"]
    assert segments[1]["structural_only"] is False
    assert segments[2]["augmentation_block_ids"] == ["b0004"]
    assert segments[2]["structural_only"] is False
    validate_exact_coverage(segments, document["blocks"])


def test_augmentation_semantic_hash_ignores_heading_text_but_not_body_text() -> None:
    document = _document(["heading", "prose"])
    segment = construct_segments_from_cuts([], document)[0]
    by_id = {item["block_id"]: item for item in document["blocks"]}
    baseline = _segment_input_hash(segment, by_id)

    by_id["b0001"] = {**by_id["b0001"], "title": "Renamed heading", "text": "Renamed"}
    assert _segment_input_hash(segment, by_id) == baseline
    by_id["b0002"] = {**by_id["b0002"], "section_title": "Renamed heading"}
    assert _segment_input_hash(segment, by_id) == baseline

    by_id["b0002"] = {**by_id["b0002"], "text": "Changed body"}
    assert _segment_input_hash(segment, by_id) != baseline


def test_duplicate_cuts_are_rejected_explicitly() -> None:
    document = _document(["prose"] * 5)

    with pytest.raises(SegmentationError, match=r"cut ordinals must be unique; duplicates: \[1, 3\]"):
        construct_segments_from_cuts([3, 1, 3, 1], document)


def test_portable_cut_schema_leaves_duplicate_rejection_to_companion_validation(
    tmp_path: Path,
) -> None:
    cut_array_schema = CUT_SCHEMA["properties"]["cut_after_ordinals"]
    assert "uniqueItems" not in cut_array_schema
    calls: list[str] = []

    def model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        calls.append(call_label)
        assert "uniqueItems" not in schema["properties"]["cut_after_ordinals"]
        return {"cut_after_ordinals": [1, 1]}

    with pytest.raises(
        SegmentationError,
        match=r"failed semantic cut validation after 3 attempts: .*duplicate cut ordinals: \[1\]",
    ):
        segment_document(
            _document(["prose"] * 3),
            checkpoint_dir=tmp_path,
            workers=1,
            force=False,
            call_model=model,
        )

    assert calls == [
        "companion-segmentation-w-0001-attempt-1",
        "companion-segmentation-w-0001-attempt-2",
        "companion-segmentation-w-0001-attempt-3",
    ]
    attempts = sorted((tmp_path / "segmentation" / "attempts").rglob("attempt-*.json"))
    assert len(attempts) == 3
    assert all(json.loads(path.read_text(encoding="utf-8"))["accepted"] is False for path in attempts)
    assert not (tmp_path / "segmentation.json").exists()

    calls.clear()
    with pytest.raises(SegmentationError, match="exhausted its lifetime"):
        segment_document(
            _document(["prose"] * 3),
            checkpoint_dir=tmp_path,
            workers=1,
            force=False,
            call_model=model,
        )
    assert calls == []


def test_provider_failure_is_not_multiplied_by_semantic_validation_retries(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        calls.append(call_label)
        raise RuntimeError("provider quota exhausted")

    with pytest.raises(RuntimeError, match="provider quota exhausted"):
        segment_document(
            _document(["prose"] * 3),
            checkpoint_dir=tmp_path,
            workers=1,
            force=False,
            call_model=model,
        )

    assert calls == ["companion-segmentation-w-0001-attempt-1"]
    assert not list((tmp_path / "segmentation" / "attempts").rglob("attempt-*.json"))
    assert not (tmp_path / "segmentation.json").exists()


@pytest.mark.parametrize("cuts", [["2"], [True], [0], [-1], [5], [99]])
def test_malformed_or_out_of_range_cuts_are_rejected(cuts: list[object]) -> None:
    with pytest.raises(SegmentationError):
        construct_segments_from_cuts(cuts, _document(["prose"] * 5))  # type: ignore[arg-type]


def test_observed_seg_0028_and_seg_0077_are_contiguous_ranges_not_block_ordinals() -> None:
    document = _mixed_610_document()
    inventory = build_block_inventory(document)
    cuts = list(range(8, 610, 8))

    segments = construct_segments_from_cuts(cuts, document, inventory=inventory)

    assert len(inventory) == 610
    assert Counter(item["type"] for item in inventory) == Counter(
        item["kind"] for item in document["blocks"]
    )
    by_id = {segment["segment_id"]: segment for segment in segments}
    assert {"seg-0028", "seg-0077"} <= by_id.keys()
    positions = {
        block["block_id"]: ordinal
        for ordinal, block in enumerate(document["blocks"], start=1)
    }
    for previous, current in zip(segments, segments[1:]):
        assert positions[current["start_block_id"]] == positions[previous["end_block_id"]] + 1
    validate_exact_coverage(segments, document["blocks"])
    assert [value for segment in segments for value in segment["block_ids"]] == [
        item["block_id"] for item in document["blocks"]
    ]


def test_full_610_block_document_segments_through_cut_only_windows(tmp_path: Path) -> None:
    document = _mixed_610_document()
    inventory = build_block_inventory(document)
    windows = build_segmentation_windows(inventory)
    calls: list[tuple[str, list[int]]] = []

    def model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        window = json.loads(prompt.split("WINDOW:\n", maxsplit=1)[1])
        first = int(window["start_ordinal"])
        end = int(window["end_ordinal"])
        cuts = list(range(first + 3, end, 4))
        calls.append((call_label, cuts))
        return {"cut_after_ordinals": cuts}

    segments = segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=8,
        force=False,
        call_model=model,
    )

    expected_labels = {
        f"companion-segmentation-{window['window_id']}-attempt-1"
        for window in windows
    }
    assert len(calls) == len(windows) == 23
    assert {label for label, _ in calls} == expected_labels
    assert all(3 <= len(cuts) <= 12 for _, cuts in calls)
    assert all(3 <= len(segment["block_ids"]) <= 4 for segment in segments)
    validate_exact_coverage(segments, document["blocks"])
    assert sum(len(segment["block_ids"]) for segment in segments) == 610
    window_checkpoints = sorted((tmp_path / "segmentation" / "windows").glob("*.json"))
    attempts = sorted((tmp_path / "segmentation" / "attempts").rglob("attempt-1.json"))
    assert len(window_checkpoints) == len(attempts) == 23
    assert all(json.loads(path.read_text(encoding="utf-8"))["accepted"] for path in attempts)
    assert not list((tmp_path / "segmentation" / "refinements").rglob("*.json"))
    checkpoint = json.loads((tmp_path / "segmentation.json").read_text(encoding="utf-8"))
    assert checkpoint["segments"] == segments
    assert checkpoint["cuts"] == sorted(checkpoint["cuts"])


def test_validate_exact_coverage_rejects_gap_overlap_reordering_and_bad_edges() -> None:
    blocks = _document(["prose"] * 4)["blocks"]
    valid = construct_segments_from_cuts([2], {"blocks": blocks})
    mutations = []
    gap = deepcopy(valid)
    gap[1]["block_ids"].remove("b0003")
    gap[1]["start_block_id"] = "b0004"
    mutations.append(gap)
    overlap = deepcopy(valid)
    overlap[1]["block_ids"].insert(0, "b0002")
    overlap[1]["start_block_id"] = "b0002"
    mutations.append(overlap)
    reordered = deepcopy(valid)
    reordered.reverse()
    mutations.append(reordered)
    bad_edge = deepcopy(valid)
    bad_edge[0]["end_block_id"] = "b0001"
    mutations.append(bad_edge)

    for segments in mutations:
        with pytest.raises(SegmentationError):
            validate_exact_coverage(segments, blocks)


def test_oversized_interval_is_locally_refined(tmp_path: Path) -> None:
    document = _document(["prose"] * 30, section_size=100)
    calls = []

    def model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        calls.append(call_label)
        return {"cut_after_ordinals": [15] if "refine" in call_label else []}

    segments = segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=2,
        force=True,
        call_model=model,
    )

    assert [len(item["block_ids"]) for item in segments] == [15, 15]
    assert any("refine" in value for value in calls)
    validate_exact_coverage(segments, document["blocks"])


def test_preservation_html_does_not_inflate_semantic_refinement_size(tmp_path: Path) -> None:
    document = _document(["equation"] * 3, section_size=100)
    for index, block in enumerate(document["blocks"], start=1):
        block["html"] = f"<math data-index='{index}'>" + ("x" * 25_000) + "</math>"
    assert sum(len(block["html"]) for block in document["blocks"]) > 60_000
    original = deepcopy(document)
    calls: list[str] = []

    segments = segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=1,
        force=False,
        call_model=_cut_model({}, calls),
    )

    assert calls == ["companion-segmentation-w-0001-attempt-1"]
    assert len(segments) == 1
    assert segments[0]["block_ids"] == ["b0001", "b0002", "b0003"]
    assert not list((tmp_path / "segmentation" / "refinements").rglob("*.json"))
    assert (tmp_path / "segmentation.json").is_file()
    assert document == original


def test_rich_inline_math_and_equation_layout_do_not_inflate_prompt_projection(
    tmp_path: Path,
) -> None:
    document = _document(["prose", "equation", "prose"], section_size=100)
    for index, block in enumerate(document["blocks"], start=1):
        block["inline_runs"] = [{
            "kind": "math",
            "token_id": f"math-{index}",
            "content_hash": f"{index}" * 64,
            "content": "m" * 24_000,
            "mathml": "<math>" + ("x" * 24_000) + "</math>",
            "layout": {"cells": ["y" * 24_000]},
        }]
    document["equations"][0]["mathml"] = "<math>" + ("z" * 30_000) + "</math>"
    document["equations"][0]["layout"] = {"rows": ["w" * 30_000]}
    projected_equation = annotation_input_block(document["blocks"][1], document)["equation"]
    assert projected_equation["tex"] == ["x_{2}=y_{2}"]
    assert "mathml" not in projected_equation
    assert "layout" not in projected_equation
    original = deepcopy(document)
    calls: list[str] = []

    segments = segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=1,
        force=False,
        call_model=_cut_model({}, calls),
    )

    assert calls == ["companion-segmentation-w-0001-attempt-1"]
    assert len(segments) == 1
    assert not list((tmp_path / "segmentation" / "refinements").rglob("*.json"))
    assert document == original


def test_indivisible_display_equation_is_not_mechanically_refined_for_size(
    tmp_path: Path,
) -> None:
    document = _document(["equation"], section_size=100)
    document["equations"][0]["tex"] = ["x" * 70_000]
    calls: list[str] = []

    segments = segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=1,
        force=False,
        call_model=_cut_model({}, calls),
    )

    assert calls == []
    assert segments[0]["block_ids"] == ["b0001"]
    assert not list((tmp_path / "segmentation" / "refinements").rglob("*.json"))


def test_large_non_html_semantic_fields_still_trigger_refinement_failure(
    tmp_path: Path,
) -> None:
    document = _document(["equation"] * 3, section_size=100)
    for block in document["blocks"]:
        block["text"] = "semantic-source-" + ("y" * 25_000)
    assert sum(len(block["text"]) for block in document["blocks"]) > 60_000
    original = deepcopy(document)
    calls: list[str] = []

    def model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        calls.append(call_label)
        return {"cut_after_ordinals": []}

    with pytest.raises(
        SegmentationError,
        match=r"refine-seg-0001 failed semantic cut validation after 3 attempts",
    ) as caught:
        segment_document(
            document,
            checkpoint_dir=tmp_path,
            workers=1,
            force=False,
            call_model=model,
        )

    assert calls == [
        "companion-segmentation-w-0001-attempt-1",
        "companion-segmentation-refine-1-seg-0001-attempt-1",
        "companion-segmentation-refine-1-seg-0001-attempt-2",
        "companion-segmentation-refine-1-seg-0001-attempt-3",
    ]
    assert caught.value.diagnostic()["context"]["phase"] == "refinement"
    assert not (tmp_path / "segmentation.json").exists()
    assert document == original


def test_refinement_fails_only_after_three_cut_adding_rounds(tmp_path: Path) -> None:
    document = _document(["prose"] * 30, section_size=100)
    calls: list[str] = []

    def model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        calls.append(call_label)
        if "refine" not in call_label:
            return {"cut_after_ordinals": []}
        window = json.loads(prompt.split("WINDOW:\n", maxsplit=1)[1])
        return {"cut_after_ordinals": [int(window["start_ordinal"])]}

    with pytest.raises(
        SegmentationError,
        match=r"remains above the hard size limit after 3 refinement rounds",
    ) as caught:
        segment_document(
            document,
            checkpoint_dir=tmp_path,
            workers=1,
            force=False,
            call_model=model,
        )

    assert calls == [
        "companion-segmentation-w-0001-attempt-1",
        "companion-segmentation-refine-1-seg-0001-attempt-1",
        "companion-segmentation-refine-2-seg-0002-attempt-1",
        "companion-segmentation-refine-3-seg-0003-attempt-1",
    ]
    assert caught.value.context == {
        "phase": "refinement",
        "round": 3,
        "intervals": ["seg-0004[4..30]"],
    }
    refinement_paths = sorted((tmp_path / "segmentation" / "refinements").rglob("*.json"))
    assert len(refinement_paths) == 3
    assert {path.parent.name for path in refinement_paths} == {"round-1", "round-2", "round-3"}
    assert not (tmp_path / "segmentation.json").exists()


def test_invalid_refinement_responses_report_round_window_and_attempt(tmp_path: Path) -> None:
    document = _document(["prose"] * 30, section_size=100)
    calls: list[str] = []

    def model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        calls.append(call_label)
        if "refine" not in call_label:
            return {"cut_after_ordinals": []}
        return {"cut_after_ordinals": [99]}

    with pytest.raises(SegmentationError, match=r"failed semantic cut validation after 3 attempts") as caught:
        segment_document(
            document,
            checkpoint_dir=tmp_path,
            workers=1,
            force=False,
            call_model=model,
        )

    assert calls == [
        "companion-segmentation-w-0001-attempt-1",
        "companion-segmentation-refine-1-seg-0001-attempt-1",
        "companion-segmentation-refine-1-seg-0001-attempt-2",
        "companion-segmentation-refine-1-seg-0001-attempt-3",
    ]
    context = caught.value.diagnostic()["context"]
    assert context["phase"] == "refinement"
    assert context["round"] == 1
    assert context["window_id"] == "refine-seg-0001"
    assert context["start_ordinal"] == 1
    assert context["end_ordinal"] == 30
    assert context["attempt"] == 3
    assert context["refinement"] is True
    assert context["section_ordinals"] == [1]
    assert not (tmp_path / "segmentation.json").exists()


@pytest.mark.parametrize(
    "response",
    [
        pytest.param([], id="non-object"),
        pytest.param({}, id="missing-cuts"),
        pytest.param({"cut_after_ordinals": None}, id="null-cuts"),
        pytest.param({"cut_after_ordinals": ["not-an-integer"]}, id="non-integer-cut"),
    ],
)
def test_malformed_model_output_exhausts_three_attempts_without_publication(
    tmp_path: Path, response: object
) -> None:
    document = _document(["prose"] * 3)
    calls = []

    def bad_model(prompt: str, schema: dict, artifact_dir: Path, call_label: str):
        calls.append(call_label)
        return response

    with pytest.raises(SegmentationError, match="after 3 attempts"):
        segment_document(
            document,
            checkpoint_dir=tmp_path,
            workers=1,
            force=False,
            call_model=bad_model,
        )

    assert len(calls) == 3
    assert not (tmp_path / "segmentation.json").exists()
    attempts = sorted((tmp_path / "segmentation" / "attempts").rglob("attempt-*.json"))
    assert len(attempts) == 3
    assert all(json.loads(path.read_text())["accepted"] is False for path in attempts)


def test_retry_prompt_includes_prior_validation_error_and_same_inventory(tmp_path: Path) -> None:
    document = _document(["prose"] * 3, section_size=10)
    prompts: list[str] = []

    def model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        prompts.append(prompt)
        if len(prompts) == 1:
            return {"cut_after_ordinals": [99]}
        return {"cut_after_ordinals": []}

    segments = segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=1,
        force=False,
        call_model=model,
    )

    assert len(prompts) == 2
    assert "CORRECTION REQUIRED" not in prompts[0]
    base, feedback = prompts[1].split("\n\nCORRECTION REQUIRED:", maxsplit=1)
    assert base == prompts[0]
    assert "previous response was rejected" in feedback
    assert "got [99]" in feedback
    validate_exact_coverage(segments, document["blocks"])


def test_successful_refinement_checkpoint_is_reused_when_final_cache_is_absent(
    tmp_path: Path,
) -> None:
    document = _document(["prose"] * 30, section_size=100)
    first_calls: list[str] = []

    def first_model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        first_calls.append(call_label)
        return {"cut_after_ordinals": [15] if "refine" in call_label else []}

    first = segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=2,
        force=False,
        call_model=first_model,
    )
    refinement_paths = sorted((tmp_path / "segmentation" / "refinements").rglob("*.json"))
    assert refinement_paths
    assert any("refine" in label for label in first_calls)
    (tmp_path / "segmentation.json").unlink()
    repeated_calls: list[str] = []

    second = segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=2,
        force=False,
        call_model=_cut_model({}, repeated_calls),
    )

    assert second == first
    assert repeated_calls == []
    validate_exact_coverage(second, document["blocks"])


def test_windows_execute_in_parallel(tmp_path: Path) -> None:
    document = _document(["prose"] * 6, section_size=2)
    barrier = threading.Barrier(3)
    threads: set[str] = set()
    lock = threading.Lock()

    def model(prompt: str, schema: dict, artifact_dir: Path, call_label: str) -> dict:
        with lock:
            threads.add(threading.current_thread().name)
        barrier.wait(timeout=5)
        return {"cut_after_ordinals": []}

    segments = segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=3,
        force=False,
        call_model=model,
    )

    assert len(threads) == 3
    assert [len(item["block_ids"]) for item in segments] == [2, 2, 2]


def test_final_cache_invalidates_when_inventory_changes(tmp_path: Path) -> None:
    document = _document(["prose"] * 4, section_size=2)
    calls: list[str] = []
    model = _cut_model({}, calls)
    segment_document(document, checkpoint_dir=tmp_path, workers=2, force=False, call_model=model)
    first_count = len(calls)

    changed = deepcopy(document)
    changed["blocks"][0]["text"] = "changed semantic source"
    segment_document(changed, checkpoint_dir=tmp_path, workers=2, force=False, call_model=model)

    assert first_count == 2
    assert len(calls) == 4


@pytest.mark.parametrize("corruption", ["not-json", "{}"])
def test_corrupt_final_cache_is_ignored_and_rebuilt_without_model_calls(
    tmp_path: Path, corruption: str
) -> None:
    document = _document(["prose"] * 4, section_size=2)
    calls: list[str] = []
    model = _cut_model({}, calls)
    segment_document(document, checkpoint_dir=tmp_path, workers=2, force=False, call_model=model)
    assert len(calls) == 2
    (tmp_path / "segmentation.json").write_text(corruption)
    calls.clear()

    rebuilt = segment_document(document, checkpoint_dir=tmp_path, workers=2, force=False, call_model=model)

    assert calls == []
    validate_exact_coverage(rebuilt, document["blocks"])
    assert json.loads((tmp_path / "segmentation.json").read_text())["segments"] == rebuilt


def test_corrupt_window_cache_is_recomputed_without_losing_valid_windows(tmp_path: Path) -> None:
    document = _document(["prose"] * 4, section_size=2)
    calls: list[str] = []
    model = _cut_model({}, calls)
    segment_document(document, checkpoint_dir=tmp_path, workers=2, force=False, call_model=model)
    (tmp_path / "segmentation.json").unlink()
    window_paths = sorted((tmp_path / "segmentation" / "windows").glob("*.json"))
    assert len(window_paths) == 2
    window_paths[0].write_text("not-json")
    calls.clear()

    rebuilt = segment_document(document, checkpoint_dir=tmp_path, workers=2, force=False, call_model=model)

    # The accepted response is durable before the derived window cache. A
    # missing/corrupt derived cache must replay validation without another
    # paid model call.
    assert calls == []
    validate_exact_coverage(rebuilt, document["blocks"])


def test_segmentation_never_mutates_source_document(tmp_path: Path) -> None:
    document = _document(["heading", "prose", "equation", "figure"])
    original = deepcopy(document)

    segment_document(
        document,
        checkpoint_dir=tmp_path,
        workers=1,
        force=False,
        call_model=_cut_model({}),
    )

    assert document == original
