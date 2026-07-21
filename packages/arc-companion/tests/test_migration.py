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


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Einstein mass", "terminology_mismatch"),
        ("质量", "protected_name_mismatch"),
    ],
)
def test_translation_rejects_terminology_and_protected_name_failures(text: str, reason: str) -> None:
    blocks, segments, candidates = _translation_fixture(text=text)
    result = migrate_legacy_translations(
        candidates, metadata={}, blocks=blocks,
        chapters=[{"chapter_id": "ch-0001", "block_ids": ["b1"]}], segments=segments,
        source_hash="source", language="Chinese",
        glossary={"entries": [{"source": "mass", "target": "质量"}]},
        protected_names=["Einstein"],
    )
    assert result["receipts"][0]["reason"] == reason
    assert result["ledgers"]["ch-0001"]["blocks"][0]["state"] == "pending"


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
    assert plan["never_migrated"] == ["guides", "annotations", "reviews", "tex", "pdf"]
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
