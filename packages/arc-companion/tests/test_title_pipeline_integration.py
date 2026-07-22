from __future__ import annotations

from pathlib import Path

import pytest
import arc_companion.pipeline as pipeline_module

from arc_companion.content import (
    ContentBundleError,
    reader_content_from_overrides,
    store_reader_content,
)
from arc_companion.pipeline import BuildOptions, _generate_title_translations
from arc_companion.source import SourceBundle


def _fixture() -> tuple[SourceBundle, list[dict]]:
    document = {
        "front_matter": {
            "title": "Original Paper",
            "block_ids": {"title": ["title"]},
        },
        "blocks": [
            {
                "block_id": "title", "type": "heading",
                "text": "Original Paper", "source_role": "front_matter_title",
            },
            {"block_id": "section", "type": "section", "text": "1 Results"},
            {"block_id": "body", "type": "text", "text": "Source body."},
            {
                "block_id": "caption", "type": "figure",
                "caption": "Figure title must stay source-only",
            },
        ],
        "integrity": {"status": "complete", "document_hash": "title-fixture"},
    }
    bundle = SourceBundle(
        paper_id="local:title-fixture",
        parsed={"paper_id": "local:title-fixture", "document": document},
        document=document,
        metadata={"title": "Metadata fallback"},
        references=[], citers=[],
    )
    chapters = [{
        "chapter_id": "ch-0001", "title": "1 Results",
        "block_ids": ["section", "body"], "title_block_ids": ["section"],
    }]
    return bundle, chapters


def test_title_lane_is_independent_reusable_and_never_requests_commentary(
    tmp_path: Path,
) -> None:
    bundle, chapters = _fixture()
    options = BuildOptions(
        paper_id=bundle.paper_id, project_dir=tmp_path,
        source_language="en", annotation_language="zh-CN",
    )
    calls: list[str] = []

    def model(prompt, _schema, _artifact_dir, label, **_recovery):
        calls.append(label)
        assert label.startswith("title-translation-")
        assert "This is title translation only" in prompt
        return {"titles": [
            {"title_id": "document:title", "text": "原始论文"},
            {"title_id": "block:section", "text": "1 结果"},
        ]}

    output = _generate_title_translations(
        options=options, bundle=bundle, document=bundle.document,
        chapters=chapters, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=tmp_path / "checkpoint", call_model=model,
    )
    assert len(calls) == 1
    assert calls[0].startswith("title-translation-")
    assert [item["title_id"] for item in output["titles"]] == [
        "document:title", "block:section",
    ]
    assert all(item.get("block_id") != "caption" for item in output["titles"])

    reused = _generate_title_translations(
        options=options, bundle=bundle, document=bundle.document,
        chapters=chapters, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=tmp_path / "checkpoint",
        call_model=lambda *_args, **_recovery: pytest.fail("accepted title artifact was not reused"),
    )
    assert reused == output


def test_skip_translation_makes_no_title_call(tmp_path: Path) -> None:
    bundle, chapters = _fixture()
    output = _generate_title_translations(
        options=BuildOptions(
            paper_id=bundle.paper_id, project_dir=tmp_path,
            source_language="en", annotation_language="en", skip_translation=True,
        ),
        bundle=bundle, document=bundle.document, chapters=chapters,
        glossary={}, protected_names=[], checkpoint_dir=tmp_path / "checkpoint",
        call_model=lambda *_args, **_recovery: pytest.fail("skip mode submitted a title call"),
    )
    assert output is None


def test_multi_chunk_title_crash_reuses_prefix_and_keeps_independent_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, chapters = _fixture()
    options = BuildOptions(
        paper_id=bundle.paper_id,
        project_dir=tmp_path,
        source_language="en",
        annotation_language="zh-CN",
    )
    original_chunks = pipeline_module.chunk_title_records

    def two_chunks(records):
        values = list(records)
        assert len(values) == 2
        return [[values[0]], [values[1]]]

    monkeypatch.setattr(pipeline_module, "chunk_title_records", two_chunks)
    calls: list[tuple[str, Path, dict]] = []

    def crashing_model(
        prompt, _schema, artifact_dir, label, *, recovery_descriptor=None,
    ):
        calls.append((label, artifact_dir, recovery_descriptor))
        if "block:section" in prompt:
            raise RuntimeError("middle title chunk crash")
        return {"titles": [{"title_id": "document:title", "text": "原始论文"}]}

    with pytest.raises(RuntimeError, match="middle title chunk crash"):
        _generate_title_translations(
            options=options,
            bundle=bundle,
            document=bundle.document,
            chapters=chapters,
            glossary={"entries": []},
            protected_names=[],
            checkpoint_dir=tmp_path / "checkpoint",
            call_model=crashing_model,
        )

    def resumed_model(
        prompt, _schema, artifact_dir, label, *, recovery_descriptor=None,
    ):
        calls.append((label, artifact_dir, recovery_descriptor))
        assert "document:title" not in prompt
        return {"titles": [{"title_id": "block:section", "text": "1 结果"}]}

    output = _generate_title_translations(
        options=options,
        bundle=bundle,
        document=bundle.document,
        chapters=chapters,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=tmp_path / "checkpoint",
        call_model=resumed_model,
    )

    assert [item["title_id"] for item in output["titles"]] == [
        "document:title", "block:section",
    ]
    assert len(calls) == 3
    first_label, first_root, first_descriptor = calls[0]
    second_label, second_root, second_descriptor = calls[1]
    assert first_label != second_label
    assert first_root != second_root
    assert first_root.name == first_label
    assert second_root.name == second_label
    assert first_descriptor["ordered_siblings"] == second_descriptor["ordered_siblings"]
    assert first_descriptor["suffix"] == first_descriptor["ordered_siblings"]
    assert second_descriptor["suffix"] == [second_label]
    monkeypatch.setattr(pipeline_module, "chunk_title_records", original_chunks)


def test_reader_content_requires_exact_title_coverage_when_declared(
    tmp_path: Path,
) -> None:
    bundle, chapters = _fixture()
    segment = {"segment_id": "seg-0001", "block_ids": ["body"]}
    base = {
        "document": bundle.document,
        "chapters": chapters,
        "segments": [segment],
        "chapter_guides": {"ch-0001": {}},
        "translations": {"seg-0001": {"blocks": [{"block_id": "body", "text": "正文"}]}},
        "annotations": {"seg-0001": {"commentary": "Commentary"}},
        "glossary": {}, "metadata": bundle.metadata,
        "language": "zh-CN", "source_language": "en",
        "translation_mode": "enabled",
        "title_translations": {"titles": [
            {"title_id": "document:title", "text": "原始论文"},
            {"title_id": "block:section", "text": "1 结果"},
        ]},
    }
    content = reader_content_from_overrides(
        base, reader_evidence_by_segment={"seg-0001": []},
    )
    assert store_reader_content(tmp_path, content=content)["content"] == content

    content["title_translations"]["titles"].pop()
    with pytest.raises(ContentBundleError, match="title translations"):
        store_reader_content(tmp_path, content=content)
