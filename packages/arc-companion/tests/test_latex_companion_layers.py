from __future__ import annotations

import hashlib
from pathlib import Path

from arc_companion.latex import (
    _png_needs_latex_flattening,
    _render_equation,
    escape_tex,
    render_companion_tex,
    validate_tex_fidelity,
)


def test_renderer_orders_layers_and_repeats_only_unnumbered_equations(tmp_path: Path) -> None:
    image = tmp_path / "source-image.png"
    image.write_bytes(b"fixture image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    document = {
        "blocks": [
            {"block_id": "p", "kind": "prose", "text": "An equation follows."},
            {"block_id": "eq", "kind": "equation", "equation_id": "eq"},
            {"block_id": "fig", "kind": "figure", "figure_id": "fig"},
            {"block_id": "tab", "kind": "table", "table_id": "tab"},
        ],
        "equations": [{"id": "eq", "tex": "x=y", "number": "(7)", "label": "eq:seven"}],
        "figures": [{"id": "fig", "asset_id": "asset", "caption": "Unique figure caption"}],
        "assets": [{"id": "asset", "cache_path": str(image), "sha256": digest}],
        "tables": [{
            "id": "tab",
            "column_count": 1,
            "rows": [[{"text": "Unique table cell", "row": 0, "column": 0}]],
        }],
        "integrity": {"status": "complete"},
    }
    segments = [{
        "segment_id": "s1",
        "title": "Result",
        "start_block_id": "p",
        "end_block_id": "tab",
        "block_ids": ["p", "eq", "fig", "tab"],
    }]
    translations = {"s1": {"blocks": [
        {"block_id": "p", "text": "随后是公式。", "translate": True},
        {"block_id": "eq", "text": "x=y", "translate": True},
        {"block_id": "fig", "text": "", "translate": False},
        {"block_id": "tab", "text": "", "translate": False},
    ]}}
    annotations = {"s1": {
        "explanation": "这解释了结果。",
        "prior_work": [{"text": "前人结果", "evidence_ids": ["ref-1"]}],
        "later_work": [{"text": "后续推广", "evidence_ids": ["cite-1"]}],
    }}
    glossary = {"entries": [{
        "source_term": "equation",
        "target_term": "方程",
        "brief_explanation": "表达量之间关系的数学式。",
        "first_block_id": "p",
        "protected_names": [],
    }]}

    tex, manifest = render_companion_tex(
        document,
        segments,
        annotations,
        translations=translations,
        glossary=glossary,
        output_dir=tmp_path,
        language="zh-CN",
    )

    assert tex.index(r"\textbf{原文}") < tex.index(r"\textbf{译文}") < tex.index(r"\textbf{伴读}")
    assert tex.count("x=y") == 2
    assert tex.count(r"\tag{7}") == 1
    assert tex.count(r"\label{eq:seven}") == 1
    assert tex.count(r"\includegraphics") == 1
    assert tex.count("Unique figure caption") == 1
    assert tex.count("Unique table cell") == 1
    assert "随后是公式" in tex
    assert "术语表" in tex and "equation" in tex and "方程" in tex
    assert "本版说明" not in tex
    assert "ArcSourceBackground" not in tex
    assert r"\newenvironment{arcsource}{\par\begingroup}{\par\endgroup}" in tex
    assert "ArcTranslationBackground" in tex
    assert "ArcCompanionBackground" in tex
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_translation_renders_opaque_math_tokens_from_source_without_identity(tmp_path: Path) -> None:
    math_hash = hashlib.sha256(b"math-token").hexdigest()
    token = f"[[ARC_INLINE:p.token-0002:{math_hash}]]"
    document = {
        "front_matter": {},
        "blocks": [{
            "block_id": "p",
            "kind": "prose",
            "text": r"The t_{NL} term.",
            "inline_runs": [
                {"kind": "text", "content": "The ", "token_id": "p.token-0001", "content_hash": hashlib.sha256(b"text").hexdigest()},
                {"kind": "math", "content": r"t_{NL}", "tex": r"t_{NL}\tag{9}\label{bad}", "token_id": "p.token-0002", "content_hash": math_hash},
                {"kind": "text", "content": " term.", "token_id": "p.token-0003", "content_hash": hashlib.sha256(b"tail").hexdigest()},
            ],
        }],
        "equations": [], "figures": [], "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    segments = [{"segment_id": "s", "block_ids": ["p"], "start_block_id": "p", "end_block_id": "p"}]
    translations = {"s": {"blocks": [{"block_id": "p", "text": f"术语 {token}。"}]}}
    annotations = {"s": {"explanation": "解释", "prior_work": "", "later_work": "", "commentary": "解释"}}

    tex, _ = render_companion_tex(
        document, segments, annotations, translations=translations, output_dir=tmp_path, language="zh-CN"
    )

    translation = tex.split(r"\begin{arctranslation}", 1)[1].split(r"\end{arctranslation}", 1)[0]
    assert r"\(t_{NL}\)" in translation
    assert token not in translation
    assert r"\tag" not in translation and r"\label" not in translation


def test_translation_math_delimiters_are_preserved_but_surrounding_tex_is_escaped(tmp_path: Path) -> None:
    document = {
        "blocks": [{"block_id": "p", "kind": "prose", "text": "Math."}],
        "integrity": {"status": "complete"},
    }
    tex, _ = render_companion_tex(
        document,
        [{"segment_id": "s", "start_block_id": "p", "end_block_id": "p", "block_ids": ["p"]}],
        {"s": {"commentary": "note"}},
        translations={"s": {"blocks": [{
            "block_id": "p",
            "translate": True,
            "text": (
                r"量 \(x_i\) 与 50% 有关，且 \[z=1\tag{9}\label{eq:nine}\]。"
                r"未分隔的 V_{\rm sr}^{\prime\prime\prime} 也应排为公式。"
            ),
        }]}},
        output_dir=tmp_path,
        language="zh-CN",
    )
    assert r"\(x_i\)" in tex
    assert r"50\%" in tex
    assert r"z=1" in tex
    assert r"\tag{9}" not in tex
    assert r"\label{eq:nine}" not in tex
    assert r"\(V_{\rm sr}^{\prime\prime\prime}\)" in tex
    assert r"V\_" not in tex


def test_unnumbered_equation_copy_strips_identity_embedded_in_source_tex() -> None:
    copied = _render_equation(
        {
            "tex": r"x=y\tag{\mathrm{A}_{1}}\label{eq:{nested}:seven}",
            "number": "(7)",
            "label": "eq:seven",
        },
        include_numbers=False,
        include_labels=False,
    )

    assert "x=y" in copied
    assert r"\tag" not in copied
    assert r"\label" not in copied


def test_fidelity_validation_audits_translation_boundaries_and_forbidden_content(tmp_path: Path) -> None:
    document = {
        "blocks": [
            {"block_id": "p", "kind": "prose", "text": "Text."},
            {"block_id": "eq", "kind": "equation", "equation_id": "eq"},
        ],
        "equations": [{"id": "eq", "tex": "x=y", "number": "(2)", "label": "eq:two"}],
        "integrity": {"status": "complete"},
    }
    tex, manifest = render_companion_tex(
        document,
        [{"segment_id": "s", "start_block_id": "p", "end_block_id": "eq", "block_ids": ["p", "eq"]}],
        {"s": {"commentary": "note"}},
        translations={"s": {"blocks": [
            {"block_id": "p", "translate": True, "text": "译文。"},
            {"block_id": "eq", "translate": True, "text": "x=y"},
        ]}},
        output_dir=tmp_path,
        language="zh-CN",
    )
    assert validate_tex_fidelity(tex, document, manifest) == []

    numbered = tex.replace("译文。", r"译文。\(z=1\tag{\mathrm{A}}\)", 1)
    errors = validate_tex_fidelity(numbered, document, manifest)
    assert any("equation number or label" in error for error in errors)

    cloned_figure = tex.replace("译文。", r"译文。\includegraphics{clone.png}", 1)
    errors = validate_tex_fidelity(cloned_figure, document, manifest)
    assert any("duplicates a figure or table" in error for error in errors)

    missing_boundary = tex.replace("% ARC-COMPANION-END", "% REMOVED-COMPANION-END", 1)
    errors = validate_tex_fidelity(missing_boundary, document, manifest)
    assert any("not delimited exactly once" in error for error in errors)


def test_bibliography_is_preservation_only_outside_mixed_semantic_unit(tmp_path: Path) -> None:
    document = {
        "blocks": [
            {"block_id": "p", "kind": "prose", "text": "Semantic source."},
            {"block_id": "bib1", "kind": "bibliography", "text": "First reference."},
            {"block_id": "bib2", "kind": "bibliography", "text": "Second reference."},
        ],
        "bibliography": [
            {"id": "bib1", "label": "[1]", "text": "First reference."},
            {"id": "bib2", "label": "[2]", "text": "Second reference."},
        ],
        "integrity": {"status": "complete"},
    }
    segments = [
        {"segment_id": "mixed", "start_block_id": "p", "end_block_id": "bib1", "block_ids": ["p", "bib1"]},
        {"segment_id": "references", "start_block_id": "bib2", "end_block_id": "bib2", "block_ids": ["bib2"]},
    ]
    tex, manifest = render_companion_tex(
        document,
        segments,
        {"mixed": {"commentary": "semantic note"}, "references": {"commentary": "must not render"}},
        translations={
            "mixed": {"blocks": [
                {"block_id": "p", "translate": True, "text": "语义译文。"},
                {"block_id": "bib1", "translate": False, "text": ""},
            ]},
            "references": {"blocks": [{"block_id": "bib2", "translate": False, "text": ""}]},
        },
        output_dir=tmp_path,
        language="zh-CN",
    )

    layers = manifest["companion_layers"]
    assert layers["semantic_segment_ids"] == ["mixed"]
    assert layers["preservation_only_segment_ids"] == ["references"]
    assert layers["rendered_translation_segment_ids"] == ["mixed"]
    assert layers["rendered_annotation_segment_ids"] == ["mixed"]
    assert "must not render" not in tex
    assert tex.count("First reference") == 1
    assert tex.count("Second reference") == 1
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_16_bit_alpha_png_is_detected_for_xelatex_flattening(tmp_path: Path) -> None:
    rgba16 = tmp_path / "rgba16.png"
    rgba16.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + b"\x00" * 8 + bytes([16, 6]))
    rgb16 = tmp_path / "rgb16.png"
    rgb16.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + b"\x00" * 8 + bytes([16, 2]))

    assert _png_needs_latex_flattening(rgba16)
    assert not _png_needs_latex_flattening(rgb16)


def test_unicode_math_glyphs_are_normalized_without_touching_equation_numbers() -> None:
    glyphs = "δνθσζτ αεϵβφπη ∼≪≫≲∝≡≈∑⟨⟩∙ Ḣ ℒℋ 𝐻𝐤𝜃𝒪𝑚𝜎𝑝𝛿𝑡𝑁𝐿𝐼𝜈𝜁𝑉𝑅𝒞𝑓𝑎𝑙𝑘𝑖𝑏𝑃 ₀₁₂₃ᵢ′\u200b"
    rendered = escape_tex(glyphs)

    for glyph in glyphs.replace(" ", "").replace("\u200b", ""):
        assert glyph not in rendered
    assert r"\delta" in rendered
    assert r"\mathcal{O}" in rendered
    assert r"\mathbf{k}" in rendered
    assert r"\textsubscript{2}" in rendered
    assert r"\textsuperscript{\(\prime\)}" in rendered

    equation = _render_equation({"tex": r"x=y", "number": "(A.12)", "label": "eq:a12"})
    assert r"\tag{A.12}" in equation
    assert r"\label{eq:a12}" in equation
