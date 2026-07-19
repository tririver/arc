from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc_companion.glossary import (
    GLOSSARY_VERSION,
    _validate_protected_names,
    generate_glossary,
    glossary_entry_limit,
)


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


def test_glossary_restores_translated_personal_name_and_keeps_strict_validation(tmp_path: Path) -> None:
    def call_model(prompt, schema, artifact_dir, call_label):
        return {"entries": [{
            "source_term": "Feynman diagram",
            "target_term": "费曼图",
            "brief_explanation": "微扰项的图形表示",
            "aliases": [],
            "protected_names": ["Feynman"],
            "first_block_id": "b1",
        }]}

    glossary = generate_glossary(
        _document(), language="zh-CN", protected_names=["Feynman"],
        checkpoint_dir=tmp_path, workers=12, force=False, call_model=call_model,
    )

    assert glossary["entries"][0]["target_term"] == "费曼图（Feynman）"
    assert glossary["entries"][0]["protected_names"] == ["Feynman"]
    with pytest.raises(RuntimeError, match="protected personal names"):
        _validate_protected_names([{
            "source_term": "Feynman diagram",
            "target_term": "费曼图",
            "protected_names": ["Feynman"],
        }], ["Feynman"])


def test_glossary_restores_poisson_without_replacing_standard_translation(tmp_path: Path) -> None:
    document = {"blocks": [{
        "block_id": "b1", "type": "text",
        "text": "The Poisson bracket structure controls the classical algebra.",
    }]}

    def call_model(prompt, schema, artifact_dir, call_label):
        return {"entries": [{
            "source_term": "Poisson bracket structure",
            "target_term": "泊松括号结构",
            "brief_explanation": "经典可观测量的代数结构",
            "aliases": [],
            "protected_names": [],
            "first_block_id": "b1",
        }]}

    glossary = generate_glossary(
        document, language="zh-CN", protected_names=["Poisson"],
        checkpoint_dir=tmp_path, workers=1, force=False, call_model=call_model,
    )

    entry = glossary["entries"][0]
    assert entry["target_term"] == "泊松括号结构（Poisson）"
    assert entry["protected_names"] == ["Poisson"]


def test_glossary_does_not_repeat_protected_name_already_in_target(tmp_path: Path) -> None:
    def call_model(prompt, schema, artifact_dir, call_label):
        return {"entries": [{
            "source_term": "Feynman diagram",
            "target_term": "Feynman 图",
            "brief_explanation": "微扰项的图形表示",
            "aliases": [],
            "protected_names": [],
            "first_block_id": "b1",
        }]}

    glossary = generate_glossary(
        _document(), language="zh-CN", protected_names=["Feynman"],
        checkpoint_dir=tmp_path, workers=1, force=False, call_model=call_model,
    )

    entry = glossary["entries"][0]
    assert entry["target_term"] == "Feynman 图"
    assert entry["protected_names"] == ["Feynman"]


def test_glossary_protected_name_repair_uses_complete_lexical_boundaries(tmp_path: Path) -> None:
    document = {"blocks": [{
        "block_id": "b1", "type": "text",
        "text": "A Poissonian point process is used.",
    }]}

    def call_model(prompt, schema, artifact_dir, call_label):
        return {"entries": [{
            "source_term": "Poissonian point process",
            "target_term": "泊松点过程",
            "brief_explanation": "独立计数过程",
            "aliases": [],
            "protected_names": ["Poisson"],
            "first_block_id": "b1",
        }]}

    glossary = generate_glossary(
        document, language="zh-CN", protected_names=["Poisson"],
        checkpoint_dir=tmp_path, workers=1, force=False, call_model=call_model,
    )

    entry = glossary["entries"][0]
    assert entry["target_term"] == "泊松点过程"
    assert entry["protected_names"] == []


def test_glossary_version_invalidates_pre_repair_final_cache(tmp_path: Path) -> None:
    stale = {
        "schema_version": "arc.companion.glossary.v3",
        "source_sha256": "irrelevant",
        "entries": [],
    }
    (tmp_path / "glossary.json").write_text(json.dumps(stale), encoding="utf-8")
    calls: list[str] = []

    def call_model(prompt, schema, artifact_dir, call_label):
        calls.append(call_label)
        return {"entries": []}

    result = generate_glossary(
        _document(), language="zh-CN", protected_names=[],
        checkpoint_dir=tmp_path, workers=1, force=False, call_model=call_model,
    )

    assert calls
    assert result["schema_version"] == GLOSSARY_VERSION


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
