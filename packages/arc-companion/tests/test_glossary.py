from __future__ import annotations

import json
from pathlib import Path

import pytest

import arc_companion.glossary as glossary_module
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


def _many_window_document(count: int = 155) -> dict:
    return {
        "blocks": [
            {
                "block_id": f"b{index:04d}",
                "type": "text",
                "section_id": f"section-{index:04d}",
                "text": f"Specialized concepts for window {index}.",
            }
            for index in range(1, count + 1)
        ]
    }


def _window_entries(index: int) -> list[dict]:
    return [
        {
            "source_term": f"specialized concept {index:04d}-{offset}",
            "target_term": f"专业概念 {index:04d}-{offset}",
            "brief_explanation": "领域读者仍需辨析的专门定义",
            "aliases": [],
            "protected_names": [],
            "first_block_id": f"b{index:04d}",
        }
        for offset in range(3)
    ]


def _prompt_candidates(prompt: str) -> list[dict]:
    return json.loads(prompt.split("CANDIDATES:\n", maxsplit=1)[1])


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


def test_direct_glossary_consolidation_cache_survives_missing_final_envelope(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def call_model(prompt, schema, artifact_dir, call_label):
        calls.append(call_label)
        return {"entries": [{
            "source_term": "Feynman diagram",
            "target_term": "Feynman 图",
            "brief_explanation": "图形表示",
            "protected_names": ["Feynman"],
            "first_block_id": "b1",
        }]}

    first = generate_glossary(
        _document(), language="zh-CN", protected_names=["Feynman"],
        checkpoint_dir=tmp_path, workers=1, force=False, call_model=call_model,
    )
    assert (tmp_path / "glossary-consolidation" / "direct.json").is_file()
    (tmp_path / "glossary.json").unlink()
    calls.clear()

    assert generate_glossary(
        _document(), language="zh-CN", protected_names=["Feynman"],
        checkpoint_dir=tmp_path, workers=1, force=False, call_model=call_model,
    ) == first
    assert calls == []


def test_glossary_refuses_no_progress_hierarchy_before_model_calls(tmp_path: Path) -> None:
    calls: list[str] = []
    huge = "x" * 25_000
    candidates = [{"entries": [{
        "source_term": f"term-{index}",
        "target_term": f"target-{index}",
        "brief_explanation": huge,
        "first_block_id": "b1",
    }]} for index in range(2)]

    with pytest.raises(RuntimeError, match="refusing to spend calls"):
        glossary_module._consolidate_candidates(
            candidates,
            blocks=_document()["blocks"],
            language="zh-CN",
            protected_names=[],
            entry_limit=1,
            checkpoint_dir=tmp_path,
            workers=2,
            force=False,
            call_model=lambda *args: calls.append("llm") or {"entries": []},
        )

    assert calls == []


def test_glossary_hierarchically_consolidates_155_windows_with_bounded_nodes(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int, int]] = []

    def call_model(prompt, schema, artifact_dir, call_label):
        if call_label.startswith("companion-glossary-window-"):
            index = int(call_label.rsplit("-", maxsplit=1)[1])
            return {"entries": _window_entries(index)}
        candidates = _prompt_candidates(prompt)
        entries = [entry for candidate in candidates for entry in candidate["entries"]]
        calls.append((call_label, len(entries), len(prompt.encode("utf-8"))))
        return {"entries": entries}

    result = generate_glossary(
        _many_window_document(), language="zh-CN", protected_names=[],
        checkpoint_dir=tmp_path, workers=12, force=False, call_model=call_model,
    )

    assert {label for label, _, _ in calls} == {
        "companion-glossary-consolidation-l0001-n0001",
        "companion-glossary-consolidation-l0001-n0002",
        "companion-glossary-consolidation-l0001-n0003",
        "companion-glossary-consolidation-l0001-n0004",
        "companion-glossary-consolidation-l0001-n0005",
        "companion-glossary-consolidation-l0002-n0001",
        "companion-glossary-consolidation-l0002-n0002",
        "companion-glossary-consolidation-l0002-n0003",
    }
    assert all(count <= 100 for _, count, _ in calls)
    assert all(
        prompt_bytes < glossary_module.CONSOLIDATION_PROMPT_MAX_BYTES
        for _, _, prompt_bytes in calls
    )
    assert "companion-glossary-consolidation" not in {label for label, _, _ in calls}
    assert 100 < len(result["entries"]) <= result["entry_limit"] == 200
    block_ids = [entry["first_block_id"] for entry in result["entries"]]
    assert block_ids == sorted(block_ids)
    assert block_ids[0] == "b0001"
    assert (
        tmp_path / "glossary-consolidation" / "level-0001" / "0001.json"
    ).is_file()
    assert (
        tmp_path / "glossary-consolidation" / "level-0002" / "0001.json"
    ).is_file()


def test_glossary_batches_long_cjk_entries_by_complete_prompt_utf8_bytes(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int]] = []

    def call_model(prompt, schema, artifact_dir, call_label):
        if call_label.startswith("companion-glossary-window-"):
            index = int(call_label.rsplit("-", maxsplit=1)[1])
            entries = _window_entries(index)
            for entry in entries:
                entry["brief_explanation"] = "量子场论中的专门定义" * 500
            return {"entries": entries}
        prompt_bytes = len(prompt.encode("utf-8"))
        calls.append((call_label, prompt_bytes))
        candidates = _prompt_candidates(prompt)
        return {"entries": [
            entry for candidate in candidates for entry in candidate["entries"]
        ]}

    result = generate_glossary(
        _many_window_document(70), language="zh-CN", protected_names=[],
        checkpoint_dir=tmp_path, workers=12, force=False, call_model=call_model,
    )

    level_one = [label for label, _ in calls if "-l0001-" in label]
    assert len(level_one) > 1
    assert all(
        prompt_bytes < glossary_module.CONSOLIDATION_PROMPT_MAX_BYTES
        for _, prompt_bytes in calls
    )
    assert max(prompt_bytes for _, prompt_bytes in calls) > 30_000
    assert 0 < len(result["entries"]) <= result["entry_limit"]


def test_glossary_rejects_single_essential_entry_above_prompt_hard_cap(
    tmp_path: Path,
) -> None:
    consolidation_calls: list[str] = []

    def call_model(prompt, schema, artifact_dir, call_label):
        if call_label.startswith("companion-glossary-window-"):
            index = int(call_label.rsplit("-", maxsplit=1)[1])
            if index == 1:
                entry = _window_entries(index)[0]
                entry["brief_explanation"] = "不可删减的必要内容" * 4_000
                return {"entries": [entry]}
            return {"entries": _window_entries(index)}
        consolidation_calls.append(call_label)
        return {"entries": []}

    with pytest.raises(
        RuntimeError,
        match=r"essential content exceeds the strict 61440-byte prompt limit",
    ):
        generate_glossary(
            _many_window_document(70), language="zh-CN", protected_names=[],
            checkpoint_dir=tmp_path, workers=12, force=False, call_model=call_model,
        )

    assert consolidation_calls == []
    consolidation_dir = tmp_path / "glossary-consolidation"
    assert not consolidation_dir.exists() or not list(consolidation_dir.rglob("*.json"))


def test_glossary_resumes_only_missing_hierarchical_consolidation_node(
    tmp_path: Path,
) -> None:
    first_calls: list[str] = []

    def interrupted_model(prompt, schema, artifact_dir, call_label):
        first_calls.append(call_label)
        if call_label.startswith("companion-glossary-window-"):
            index = int(call_label.rsplit("-", maxsplit=1)[1])
            return {"entries": _window_entries(index)}
        if call_label == "companion-glossary-consolidation-l0002-n0003":
            raise RuntimeError("interrupted hierarchical consolidation")
        candidates = _prompt_candidates(prompt)
        return {"entries": [
            entry for candidate in candidates for entry in candidate["entries"]
        ]}

    with pytest.raises(RuntimeError, match="interrupted hierarchical consolidation"):
        generate_glossary(
            _many_window_document(), language="zh-CN", protected_names=[],
            checkpoint_dir=tmp_path, workers=12, force=False,
            call_model=interrupted_model,
        )

    assert "companion-glossary-consolidation-l0001-n0001" in first_calls
    assert "companion-glossary-consolidation-l0001-n0002" in first_calls
    resume_calls: list[str] = []

    def resumed_model(prompt, schema, artifact_dir, call_label):
        resume_calls.append(call_label)
        candidates = _prompt_candidates(prompt)
        return {"entries": [
            entry for candidate in candidates for entry in candidate["entries"]
        ]}

    result = generate_glossary(
        _many_window_document(), language="zh-CN", protected_names=[],
        checkpoint_dir=tmp_path, workers=12, force=False, call_model=resumed_model,
    )

    assert resume_calls == ["companion-glossary-consolidation-l0002-n0003"]
    assert 100 < len(result["entries"]) <= result["entry_limit"]


def test_parallel_glossary_consolidation_fails_closed_without_caching_empty_node(
    tmp_path: Path,
) -> None:
    failed_label = "companion-glossary-consolidation-l0001-n0002"

    def call_model(prompt, schema, artifact_dir, call_label):
        if call_label.startswith("companion-glossary-window-"):
            index = int(call_label.rsplit("-", maxsplit=1)[1])
            return {"entries": _window_entries(index)}
        if call_label == failed_label:
            return {"entries": []}
        candidates = _prompt_candidates(prompt)
        return {"entries": [
            entry for candidate in candidates for entry in candidate["entries"]
        ]}

    with pytest.raises(
        RuntimeError,
        match=r"node .*n0002 returned no usable entries.*non-empty input",
    ):
        generate_glossary(
            _many_window_document(), language="zh-CN", protected_names=[],
            checkpoint_dir=tmp_path, workers=12, force=False,
            call_model=call_model,
        )

    level_dir = tmp_path / "glossary-consolidation" / "level-0001"
    assert not (level_dir / "0002.json").exists()
    assert any(
        path.name != "0002.json"
        for path in level_dir.glob("*.json")
    )


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
        "schema_version": "arc.companion.glossary.v5",
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
def test_glossary_deterministically_enforces_entry_cap(tmp_path: Path, page_count, expected) -> None:
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

    assert 0 < len(result["entries"]) <= expected
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


def test_non_english_glossary_requires_exact_source_terms_and_preserves_unicode_names(
    tmp_path: Path,
) -> None:
    document = {"blocks": [{
        "block_id": "b1",
        "type": "text",
        "text": "Le théorème de 南部 contrôle cette limite.",
    }]}

    def call_model(prompt, schema, artifact_dir, call_label):
        if "window" in call_label:
            assert "source-language term exactly as it appears" in prompt
        assert "preserve each name exactly in its source spelling" in prompt
        assert "English source term" not in prompt
        assert "Latin spelling" not in prompt
        return {"entries": [
            {
                "source_term": "théorème de 南部",
                "target_term": "定理",
                "brief_explanation": "控制该极限的专门结果",
                "aliases": [],
                "protected_names": ["南部"],
                "first_block_id": "b1",
            },
            {
                # Source spelling is deliberately wrong and must not survive.
                "source_term": "Théorème absent",
                "target_term": "不存在的定理",
                "brief_explanation": "模型虚构项",
                "aliases": [],
                "protected_names": [],
                "first_block_id": "b1",
            },
        ]}

    result = generate_glossary(
        document,
        language="zh-CN",
        source_language="fr",
        protected_names=["南部"],
        checkpoint_dir=tmp_path,
        workers=1,
        force=False,
        call_model=call_model,
    )

    assert [entry["source_term"] for entry in result["entries"]] == ["théorème de 南部"]
    assert result["entries"][0]["target_term"] == "定理（南部）"
    assert result["entries"][0]["protected_names"] == ["南部"]
    assert result["source_language"] == "fr"
