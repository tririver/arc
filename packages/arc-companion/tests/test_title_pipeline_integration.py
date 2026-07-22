from __future__ import annotations

from pathlib import Path

import pytest

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

    def model(prompt, _schema, _artifact_dir, label):
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
    assert calls == ["title-translation-0001"]
    assert [item["title_id"] for item in output["titles"]] == [
        "document:title", "block:section",
    ]
    assert all(item.get("block_id") != "caption" for item in output["titles"])

    reused = _generate_title_translations(
        options=options, bundle=bundle, document=bundle.document,
        chapters=chapters, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=tmp_path / "checkpoint",
        call_model=lambda *_args: pytest.fail("accepted title artifact was not reused"),
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
        call_model=lambda *_args: pytest.fail("skip mode submitted a title call"),
    )
    assert output is None


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
