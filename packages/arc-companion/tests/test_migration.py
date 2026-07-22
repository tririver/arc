from __future__ import annotations

import json

import pytest

from arc_companion.migration import (
    LegacyMigrationError,
    migrate_legacy_cuts,
    migrate_legacy_glossary,
    migrate_legacy_translations,
    plan_legacy_migration,
    read_legacy_checkpoint,
)


def _blocks(count: int = 4) -> list[dict]:
    return [
        {"block_id": f"b{index}", "kind": "prose", "text": f"Source text {index}"}
        for index in range(1, count + 1)
    ]


def test_legacy_cuts_reuse_only_exact_in_chapter_ranges() -> None:
    chapters = [
        {"chapter_id": "ch-0001", "block_ids": ["b1", "b2"]},
        {"chapter_id": "ch-0002", "block_ids": ["b3", "b4"]},
    ]

    accepted = migrate_legacy_cuts(
        [1, 2, 3], blocks=_blocks(), chapters=chapters,
        max_segment_blocks=2, max_segment_source_chars=10_000,
    )
    crossed = migrate_legacy_cuts(
        [3], blocks=_blocks(), chapters=chapters,
        max_segment_blocks=4, max_segment_source_chars=10_000,
    )

    assert accepted["reused"] == {"ch-0001": [1], "ch-0002": [1]}
    assert all(item["accepted"] for item in accepted["receipts"])
    assert crossed["reused"] == {}
    assert {item["reason"] for item in crossed["receipts"]} == {"legacy_segment_crosses_chapter"}


def test_legacy_cuts_reject_segments_that_exceed_new_size_limit() -> None:
    result = migrate_legacy_cuts(
        [2],
        blocks=_blocks(3),
        chapters=[{"chapter_id": "ch-0001", "block_ids": ["b1", "b2"]}],
        max_segment_blocks=1,
        max_segment_source_chars=10_000,
    )
    assert result["receipts"][0]["reason"] == "legacy_segment_exceeds_limits"


def test_old_glossary_requires_all_hashes_and_no_real_index() -> None:
    metadata = {
        "source_hash": "source", "language": "Chinese",
        "prompt_hash": "prompt", "validator_hash": "validator",
    }
    glossary = {**metadata, "entries": [{"source": "mass", "target": "质量"}]}

    accepted = migrate_legacy_glossary(
        glossary, metadata={}, source_hash="source", language="Chinese",
        prompt_hash="prompt", validator_hash="validator", index_entries={"entries": []},
    )
    mismatch = migrate_legacy_glossary(
        glossary, metadata={}, source_hash="changed", language="Chinese",
        prompt_hash="prompt", validator_hash="validator", index_entries={"entries": []},
    )
    indexed = migrate_legacy_glossary(
        glossary, metadata={}, source_hash="source", language="Chinese",
        prompt_hash="prompt", validator_hash="validator",
        index_entries={"entries": [{"term": "mass"}]},
    )

    assert accepted["accepted"] is True
    assert mismatch["mismatched_fields"] == ["source_hash"]
    assert indexed["reason"] == "real_index_requires_complete_index_glossary"


def _translation_fixture(*, text: str = "Einstein 质量") -> tuple[list[dict], list[dict], dict]:
    blocks = [{"block_id": "b1", "kind": "prose", "text": "Einstein mass"}]
    segments = [{"chapter_id": "ch-0001", "segment_id": "ch-0001.seg-0001", "block_ids": ["b1"]}]
    candidate = {
        "old-segment": {
            "block_ids": ["b1"], "source_hash": "source", "language": "Chinese",
            "translation": {"blocks": [{"block_id": "b1", "text": text}]},
        }
    }
    return blocks, segments, candidate


def test_translation_is_revalidated_and_imported_as_accepted_ledger_prefix() -> None:
    blocks, segments, candidates = _translation_fixture()
    result = migrate_legacy_translations(
        candidates,
        metadata={},
        blocks=blocks,
        chapters=[{"chapter_id": "ch-0001", "block_ids": ["b1"]}],
        segments=segments,
        source_hash="source",
        language="Chinese",
        glossary={"entries": [{"source": "mass", "target": "质量"}]},
        protected_names=["Einstein"],
    )

    receipt = result["receipts"][0]
    block = result["ledgers"]["ch-0001"]["blocks"][0]
    assert receipt["accepted"] is True
    assert block["state"] == "accepted"
    assert block["logical_receipt"] == {"kind": "legacy_migration", "provider_calls": 0}
    assert block["validation_receipt"]["opaque_tokens"] is True
    assert block["translation"]["blocks"][0]["text"] == "Einstein 质量"


def test_translation_projection_drops_legacy_structural_heading_output() -> None:
    blocks = [
        {"block_id": "h1", "type": "section", "title": "Introduction"},
        {"block_id": "b1", "kind": "prose", "text": "Einstein mass"},
    ]
    segments = [{
        "chapter_id": "ch-0001", "segment_id": "ch-0001.seg-0001",
        "block_ids": ["h1", "b1"],
    }]
    candidates = {"old": {
        "block_ids": ["h1", "b1"], "source_hash": "source", "language": "Chinese",
        "translation": {"blocks": [
            {"block_id": "h1", "text": "引言"},
            {"block_id": "b1", "text": "Einstein 质量"},
        ]},
    }}

    result = migrate_legacy_translations(
        candidates, metadata={}, blocks=blocks,
        chapters=[{"chapter_id": "ch-0001", "block_ids": ["h1", "b1"]}],
        segments=segments, source_hash="source", language="Chinese",
        glossary={"entries": [{"source": "mass", "target": "质量"}]},
        protected_names=["Einstein"],
    )

    receipt = result["receipts"][0]
    assert receipt["accepted"] is True
    assert receipt["dropped_structural_block_ids"] == ["h1"]
    assert result["ledgers"]["ch-0001"]["blocks"][0]["translation"] == {
        "blocks": [{"block_id": "b1", "text": "Einstein 质量"}],
    }


def test_translation_reuses_terminology_mismatch_with_warning() -> None:
    blocks, segments, candidates = _translation_fixture(text="Einstein mass")
    result = migrate_legacy_translations(
        candidates, metadata={}, blocks=blocks,
        chapters=[{"chapter_id": "ch-0001", "block_ids": ["b1"]}], segments=segments,
        source_hash="source", language="Chinese",
        glossary={"entries": [{"source": "mass", "target": "质量"}]},
        protected_names=["Einstein"],
    )
    receipt = result["receipts"][0]
    block = result["ledgers"]["ch-0001"]["blocks"][0]
    assert receipt["accepted"] is True
    assert receipt["status"] == receipt["reason"] == "warning_reuse"
    assert receipt["terminology_warnings"] == ["mass"]
    assert block["state"] == "accepted"
    assert block["validation_receipt"]["terminology"] is False
    assert block["validation_receipt"]["terminology_warnings"] == ["mass"]
    assert block["validation_receipt"]["reuse_status"] == "warning_reuse"


def test_translation_still_rejects_protected_name_failure() -> None:
    blocks, segments, candidates = _translation_fixture(text="质量")
    result = migrate_legacy_translations(
        candidates, metadata={}, blocks=blocks,
        chapters=[{"chapter_id": "ch-0001", "block_ids": ["b1"]}], segments=segments,
        source_hash="source", language="Chinese",
        glossary={"entries": [{"source": "mass", "target": "质量"}]},
        protected_names=["Einstein"],
    )
    assert result["receipts"][0]["reason"] == "protected_name_mismatch"
    assert result["ledgers"]["ch-0001"]["blocks"][0]["state"] == "prepared"


def test_translation_blocks_compose_across_merged_and_split_segments() -> None:
    blocks = [
        {"block_id": f"b{index}", "kind": "prose", "text": f"Source {index}"}
        for index in range(1, 4)
    ]
    candidates = {
        "old-1": {
            "block_ids": ["b1", "b2"], "source_hash": "source", "language": "Chinese",
            "translation": {"blocks": [
                {"block_id": "b1", "text": "译文一"},
                {"block_id": "b2", "text": "译文二"},
            ]},
        },
        "old-2": {
            "block_ids": ["b3"], "source_hash": "source", "language": "Chinese",
            "translation": {"blocks": [{"block_id": "b3", "text": "译文三"}]},
        },
    }
    segments = [
        {"chapter_id": "ch-0001", "segment_id": "new-1", "block_ids": ["b1"]},
        {"chapter_id": "ch-0001", "segment_id": "new-2", "block_ids": ["b2", "b3"]},
    ]

    result = migrate_legacy_translations(
        candidates, metadata={}, blocks=blocks,
        chapters=[{"chapter_id": "ch-0001", "block_ids": ["b1", "b2", "b3"]}],
        segments=segments, source_hash="source", language="Chinese",
        glossary={"entries": []}, protected_names=[],
    )

    assert [item["status"] for item in result["receipts"]] == [
        "composed_hit", "composed_hit",
    ]
    ledger_blocks = result["ledgers"]["ch-0001"]["blocks"]
    assert [item["state"] for item in ledger_blocks] == ["accepted", "accepted"]
    assert ledger_blocks[0]["translation"]["blocks"] == [
        {"block_id": "b1", "text": "译文一"},
    ]
    assert [item["block_id"] for item in ledger_blocks[1]["translation"]["blocks"]] == [
        "b2", "b3",
    ]


def test_valid_suffix_translation_is_persisted_as_deferred_hit() -> None:
    blocks = [
        {"block_id": f"b{index}", "kind": "prose", "text": f"Source {index}"}
        for index in range(1, 4)
    ]
    candidates = {
        "first": {
            "block_ids": ["b1"], "source_hash": "source", "language": "Chinese",
            "translation": {"blocks": [{"block_id": "b1", "text": "译文一"}]},
        },
        "third": {
            "block_ids": ["b3"], "source_hash": "source", "language": "Chinese",
            "translation": {"blocks": [{"block_id": "b3", "text": "译文三"}]},
        },
    }
    segments = [
        {"chapter_id": "ch-0001", "segment_id": f"s{index}", "block_ids": [f"b{index}"]}
        for index in range(1, 4)
    ]

    result = migrate_legacy_translations(
        candidates, metadata={}, blocks=blocks,
        chapters=[{"chapter_id": "ch-0001", "block_ids": ["b1", "b2", "b3"]}],
        segments=segments, source_hash="source", language="Chinese",
        glossary={"entries": []}, protected_names=[],
    )

    assert result["receipts"][1]["reason"] == "translation_missing_or_ambiguous"
    deferred_receipt = result["receipts"][2]
    assert deferred_receipt["accepted"] is False
    assert deferred_receipt["status"] == deferred_receipt["reason"] == "deferred_hit"
    ledger = result["ledgers"]["ch-0001"]
    first, missing, deferred = ledger["blocks"]
    assert first["state"] == "accepted"
    assert missing == {
        "segment_id": "s2", "state": "prepared",
        "submission_state": "not_submitted", "generation": 1,
    }
    assert deferred["state"] == "prepared"
    assert deferred["deferred_translation"] == {
        "blocks": [{"block_id": "b3", "text": "译文三"}],
    }
    assert deferred["deferred_input_sha256"]
    assert deferred["deferred_output_sha256"]
    assert deferred["deferred_validation_receipt"]["reuse_status"] == "deferred_hit"
    assert ledger["accepted_chain_sha256"] == first["accepted_chain_sha256"]
    assert "accepted_chain_sha256" not in deferred


def test_translation_rejects_opaque_token_loss() -> None:
    digest = "a" * 64
    token = f"[[ARC_INLINE:eq-1:{digest}]]"
    blocks = [{
        "block_id": "b1", "kind": "prose", "text": "value x",
        "inline_runs": [
            {"kind": "text", "content": "value "},
            {"kind": "math", "content": "x", "token_id": "eq-1", "content_hash": digest},
        ],
    }]
    segments = [{"chapter_id": "ch-0001", "segment_id": "ch-0001.seg-0001", "block_ids": ["b1"]}]
    candidates = {"old": {
        "block_ids": ["b1"], "source_hash": "source", "language": "Chinese",
        "translation": {"blocks": [{"block_id": "b1", "text": "数值"}]},
    }}
    result = migrate_legacy_translations(
        candidates, metadata={}, blocks=blocks,
        chapters=[{"chapter_id": "ch-0001", "block_ids": ["b1"]}], segments=segments,
        source_hash="source", language="Chinese", glossary={"entries": []}, protected_names=[],
    )
    assert token not in candidates["old"]["translation"]["blocks"][0]["text"]
    assert result["receipts"][0]["reason"] == "opaque_token_mismatch"


def test_plan_is_read_only_and_never_migrates_generated_layers(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    legacy = {
        "source_hash": "source", "language": "Chinese",
        "prompt_hash": "prompt", "validator_hash": "validator",
        "cuts": [], "glossary": {
            "source_hash": "source", "language": "Chinese",
            "prompt_hash": "prompt", "validator_hash": "validator", "entries": [],
        },
        "translations": {}, "guides": {"old": True}, "annotations": {"old": True},
        "reviews": {"old": True}, "tex": "old.tex", "pdf": "old.pdf",
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    original = path.read_bytes()
    loaded = read_legacy_checkpoint(path)
    plan = plan_legacy_migration(
        loaded,
        document={"blocks": _blocks(1)},
        chapters={"chapters": [{"chapter_id": "ch-0001", "block_ids": ["b1"]}]},
        segments=[{"chapter_id": "ch-0001", "segment_id": "ch-0001.seg-0001", "block_ids": ["b1"]}],
        source_hash="source", language="Chinese", prompt_hash="prompt", validator_hash="validator",
        glossary={"entries": []}, index_entries={"entries": []},
    )

    assert path.read_bytes() == original
    assert plan["read_only_source"] is True
    assert plan["never_migrated"] == ["tex", "pdf"]
    assert not any(key in plan for key in ("guides", "annotations", "reviews", "tex", "pdf"))


def test_read_legacy_checkpoint_rejects_non_object(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(LegacyMigrationError):
        read_legacy_checkpoint(path)


def test_read_legacy_checkpoint_directory_loads_only_eligible_artifacts(tmp_path) -> None:
    (tmp_path / "segmentation.json").write_text(json.dumps({"cuts": [1]}), encoding="utf-8")
    (tmp_path / "glossary.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    (tmp_path / "document.json").write_text(json.dumps({"source_hash": "source"}), encoding="utf-8")
    (tmp_path / "migration-metadata.json").write_text(json.dumps({
        "language": "Chinese", "prompt_hash": "prompt", "validator_hash": "validator"
    }), encoding="utf-8")
    translations = tmp_path / "translations"
    translations.mkdir()
    (translations / "one.json").write_text(json.dumps({
        "segment_id": "old-1", "translation": {"blocks": [{"block_id": "b1", "text": "译文"}]}
    }), encoding="utf-8")
    (tmp_path / "review.v3.json").write_text(json.dumps({"must_not": "load"}), encoding="utf-8")

    loaded = read_legacy_checkpoint(tmp_path)

    assert loaded["source_hash"] == "source"
    assert loaded["cuts"] == [1]
    assert set(loaded["translations"]) == {"old-1"}
    assert "review" not in loaded
