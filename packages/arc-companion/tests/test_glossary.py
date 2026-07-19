from __future__ import annotations

from pathlib import Path

import pytest

from arc_companion.glossary import generate_glossary, glossary_entry_limit


def _document() -> dict:
    return {
        "blocks": [
            {"block_id": "b1", "type": "section", "title": "Feynman diagrams"},
            {"block_id": "b2", "type": "text", "text": "A Feynman diagram represents a perturbative term."},
        ]
    }


def test_glossary_is_consolidated_ordered_and_resumable(tmp_path: Path) -> None:
    calls: list[str] = []

    def call_model(prompt, schema, artifact_dir, call_label):
        calls.append(call_label)
        if "consolidation" in call_label:
            return {"entries": [{
                "source_term": "Feynman diagram",
                "target_term": "Feynman 图",
                "brief_explanation": "微扰项的图形表示",
                "aliases": ["diagram"],
                "protected_names": ["Feynman"],
                "first_block_id": "b1",
            }]}
        return {"entries": [{
            "source_term": "Feynman diagram",
            "target_term": "Feynman 图",
            "brief_explanation": "微扰项的图形表示",
            "aliases": [],
            "protected_names": ["Feynman"],
            "first_block_id": "b1",
        }]}

    first = generate_glossary(
        _document(), language="zh-CN", protected_names=["Feynman"],
        checkpoint_dir=tmp_path, workers=12, force=False, call_model=call_model,
    )
    assert first["entries"][0]["target_term"] == "Feynman 图"
    count = len(calls)
    assert generate_glossary(
        _document(), language="zh-CN", protected_names=["Feynman"],
        checkpoint_dir=tmp_path, workers=12, force=False, call_model=call_model,
    ) == first
    assert len(calls) == count


def test_glossary_rejects_translated_personal_name(tmp_path: Path) -> None:
    def call_model(prompt, schema, artifact_dir, call_label):
        return {"entries": [{
            "source_term": "Feynman diagram",
            "target_term": "费曼图",
            "brief_explanation": "微扰项的图形表示",
            "aliases": [],
            "protected_names": ["Feynman"],
            "first_block_id": "b1",
        }]}

    with pytest.raises(RuntimeError, match="protected personal names"):
        generate_glossary(
            _document(), language="zh-CN", protected_names=["Feynman"],
            checkpoint_dir=tmp_path, workers=12, force=False, call_model=call_model,
        )


def test_glossary_protected_name_matching_uses_complete_lexical_boundaries(tmp_path: Path) -> None:
    document = {
        "blocks": [{
            "block_id": "b1", "type": "text",
            "text": "We compute primordial Non-Gaussianities.",
        }]
    }

    def call_model(prompt, schema, artifact_dir, call_label):
        return {"entries": [{
            "source_term": "Non-Gaussianities",
            "target_term": "非高斯性",
            "brief_explanation": "偏离高斯统计的关联结构",
            "aliases": [],
            # A window model may suggest a protected name conservatively; the
            # controller must still require a full lexical match in the term.
            "protected_names": ["Tie"],
            "first_block_id": "b1",
        }]}

    glossary = generate_glossary(
        document, language="zh-CN", protected_names=["Tie"],
        checkpoint_dir=tmp_path, workers=12, force=False, call_model=call_model,
    )

    assert glossary["entries"][0]["target_term"] == "非高斯性"


def test_glossary_receives_terms_beyond_segmentation_preview_limit(tmp_path: Path) -> None:
    late_term = "late-time transfer vertex"
    document = {
        "blocks": [{
            "block_id": "b1",
            "type": "text",
            "text": ("ordinary setup " * 100) + late_term,
            "preservation_html": "<div>SHOULD_NOT_REACH_GLOSSARY</div>",
            "raw_html": "<span>ALSO_EXCLUDED</span>",
        }]
    }

    def call_model(prompt, schema, artifact_dir, call_label):
        if "window" in call_label:
            assert late_term in prompt
            assert "SHOULD_NOT_REACH_GLOSSARY" not in prompt
            assert "ALSO_EXCLUDED" not in prompt
        return {"entries": [{
            "source_term": late_term,
            "target_term": "晚时转移顶点",
            "brief_explanation": "发生于晚时演化中的转移相互作用",
            "aliases": [],
            "protected_names": [],
            "first_block_id": "b1",
        }]}

    glossary = generate_glossary(
        document, language="zh-CN", protected_names=[], checkpoint_dir=tmp_path,
        workers=12, force=False, call_model=call_model,
    )

    assert glossary["entries"][0]["source_term"] == late_term


@pytest.mark.parametrize(
    ("page_count", "expected"),
    [(50, 50), (51, 100), (100, 100), (101, 200), (None, 200)],
)
def test_glossary_page_count_boundaries_and_absolute_limit(page_count, expected) -> None:
    assert glossary_entry_limit(page_count) == expected


@pytest.mark.parametrize("page_count,expected", [(50, 50), (51, 100), (100, 100), (101, 200), (None, 200)])
def test_glossary_deterministically_caps_entries(tmp_path: Path, page_count, expected) -> None:
    entries = [{
        "source_term": f"specialized concept {index}",
        "target_term": f"专业概念 {index}",
        "brief_explanation": "领域读者仍需辨析的专门定义",
        "aliases": [],
        "protected_names": [],
        "first_block_id": "b2",
    } for index in range(250)]

    def call_model(prompt, schema, artifact_dir, call_label):
        assert "Do not fill a quota" in prompt
        assert "broad field" in prompt or "broad fields" in prompt
        return {"entries": entries}

    result = generate_glossary(
        _document(), language="zh-CN", protected_names=[], checkpoint_dir=tmp_path,
        workers=2, force=False, call_model=call_model, page_count=page_count,
    )

    assert len(result["entries"]) == expected
    assert result["entry_limit"] == expected


def test_glossary_does_not_pad_to_page_limit(tmp_path: Path) -> None:
    entry = {
        "source_term": "specialized concept", "target_term": "专业概念",
        "brief_explanation": "专门定义", "aliases": [], "protected_names": [],
        "first_block_id": "b2",
    }
    result = generate_glossary(
        _document(), language="zh-CN", protected_names=[], checkpoint_dir=tmp_path,
        workers=2, force=False, call_model=lambda *args: {"entries": [entry]}, page_count=100,
    )
    assert len(result["entries"]) == 1
