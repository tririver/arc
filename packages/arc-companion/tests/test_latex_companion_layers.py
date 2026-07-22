from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata

import pytest

from arc_companion.latex import (
    _equation_environment,
    _begin_guarded_box,
    _layer_region,
    _preamble,
    _png_needs_latex_flattening,
    _render_annotation,
    _render_equation,
    _render_glossary,
    _render_html_fragment,
    escape_tex,
    render_companion_tex,
    validate_tex_fidelity,
)
from arc_companion.reader_text import clean_reader_annotation, clean_reader_translation


def test_empty_optional_annotation_has_no_visible_companion_panel() -> None:
    rendered = _render_annotation(
        "s-clear",
        {"explanation": "", "commentary": "", "prior_work": "", "later_work": ""},
        language="zh-CN",
    )

    assert "ARC-COMPANION-BEGIN" in rendered
    assert "ARC-COMPANION-END" in rendered
    assert r"\begin{arccompanion}" not in rendered
    assert "伴读" not in rendered


def test_annotation_merges_distinct_explanation_and_commentary_without_repetition() -> None:
    rendered = _render_annotation(
        "s-merged",
        {
            "explanation": "EXPLANATION-UNIQUE",
            "commentary": "COMMENTARY-UNIQUE",
            "prior_work": "",
            "later_work": "",
        },
        language="zh-CN",
    )
    repeated = _render_annotation(
        "s-repeated",
        {
            "explanation": "共享解释。",
            "commentary": "共享解释。",
            "prior_work": "",
            "later_work": "",
        },
        language="zh-CN",
    )

    assert "EXPLANATION-UNIQUE" in rendered
    assert "COMMENTARY-UNIQUE" in rendered
    assert rendered.count(r"\textbf{解释}") == 1
    assert (
        "\\Needspace{4\\baselineskip}\n\\medskip\\noindent\\textbf{解释}"
        in rendered
    )
    assert repeated.count("共享解释") == 1
    assert r"\Needspace{6\baselineskip}" in rendered


def test_annotation_renders_direct_sources_as_linked_titles_with_locators() -> None:
    rendered = _render_annotation(
        "s-direct-sources",
        {
            "explanation": "A sourced explanation.",
            "commentary": "",
            "commentary_sources": [{
                "title": "Primary Source",
                "url": "https://example.test/paper_a#section-3",
                "locator": "Section 3 / p. 12",
            }],
            "prior_work": [{
                "text": "An earlier result.",
                "sources": [{
                    "title": "Earlier Paper",
                    "url": "https://example.test/earlier?view=full&lang=en",
                    "locator": "Abstract",
                }],
            }],
            "later_work": [],
        },
        language="en",
    )

    assert (
        r"\href{https://example.test/paper\_a\#section-3}{Primary Source}"
        in rendered
    )
    assert "Section 3 / p. 12" in rendered
    assert (
        "\\Needspace{4\\baselineskip}\n\\medskip\\noindent\\textbf{Prior work}"
        in rendered
    )
    assert (
        r"\href{https://example.test/earlier?view=full\&lang=en}{Earlier Paper}"
        in rendered
    )
    assert "Abstract" in rendered


@pytest.mark.skipif(
    shutil.which("xelatex") is None or shutil.which("pdftotext") is None,
    reason="XeLaTeX and pdftotext are required",
)
def test_companion_heading_stays_with_body_when_page_space_is_short(tmp_path: Path) -> None:
    document = {
        "front_matter": {},
        "blocks": [{"block_id": "p", "kind": "prose", "text": "SOURCE-TEXT"}],
        "equations": [], "figures": [], "tables": [], "bibliography": [],
        "assets": [], "links": [], "integrity": {"status": "complete"},
    }
    segments = [{
        "segment_id": "s-pagination",
        "block_ids": ["p"],
        "start_block_id": "p",
        "end_block_id": "p",
    }]
    tex, _ = render_companion_tex(
        document,
        segments,
        {"s-pagination": {
            "explanation": "EXPLANATION-FIRST-LINE",
            "commentary": "COMMENTARY-SECOND-LINE",
        }},
        output_dir=tmp_path,
        language="zh-CN",
    )
    tex = tex.replace(
        r"\Needspace{6\baselineskip}",
        "\\clearpage\n"
        r"\vspace*{\dimexpr\textheight-4\baselineskip\relax}" "\n"
        r"\Needspace{6\baselineskip}",
        1,
    )
    tex_path = tmp_path / "companion-pagination.tex"
    tex_path.write_text(tex, encoding="utf-8")

    result = subprocess.run(
        [
            shutil.which("xelatex") or "xelatex",
            "-halt-on-error",
            "-interaction=nonstopmode",
            f"-output-directory={tmp_path}",
            str(tex_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    log = (tmp_path / "companion-pagination.log").read_text(
        encoding="utf-8", errors="replace"
    )
    assert result.returncode == 0, result.stdout + result.stderr + log
    extracted = subprocess.run(
        ["pdftotext", str(tmp_path / "companion-pagination.pdf"), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\f")
    heading_page = next(
        index
        for index, page in enumerate(extracted)
        if any(line.strip() == "解释" for line in page.splitlines())
    )
    explanation_page = next(
        index for index, page in enumerate(extracted) if "EXPLANATION-FIRST-LINE" in page
    )
    commentary_page = next(
        index for index, page in enumerate(extracted) if "COMMENTARY-SECOND-LINE" in page
    )
    assert heading_page == explanation_page == commentary_page


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

    assert r"\textbf{译文}" not in tex
    assert r"\textbf{伴读}" not in tex
    assert tex.index("An equation follows") < tex.index("随后是公式") < tex.index("这解释了结果")
    assert r"\textbf{原文}" not in tex
    assert "伴读单元" not in tex
    assert tex.count("x=y") == 2
    assert tex.count(r"\tag{7}") == 1
    assert tex.count(r"\label{eq:seven}") == 1
    assert tex.count(r"\includegraphics") == 1
    assert tex.count("Unique figure caption") == 1
    assert tex.count("Unique table cell") == 1
    assert "随后是公式" in tex
    assert "术语表" in tex and "equation" in tex and "方程" in tex
    assert tex.index("这解释了结果") < tex.index(r"\clearpage") < tex.index("术语表")
    assert tex.index("术语表") < tex.index(r"\end{document}")
    assert "本版说明" not in tex
    assert "ArcSourceBackground" not in tex
    assert r"\newenvironment{arcsource}{\par\begingroup}{\par\endgroup}" in tex
    assert "ArcTranslationBackground" in tex
    assert "ArcCompanionBackground" in tex
    assert tex.count("colback=ArcCompanionBackground") == 1
    assert r"\newtcolorbox{arccompanion}[1][]{arccompanionsurface,#1}" in tex
    assert r"\newtcolorbox{arcchapterguide}[1][]{arccompanionsurface,#1}" in tex
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_glossary_is_back_matter_after_references(tmp_path: Path) -> None:
    document = {
        "front_matter": {},
        "blocks": [
            {"block_id": "body", "kind": "prose", "text": "SOURCE BODY"},
            {"block_id": "ref", "kind": "bibliography", "text": "REFERENCE TEXT"},
        ],
        "bibliography": [{"id": "ref", "label": "[1]", "text": "REFERENCE TEXT"}],
        "equations": [], "figures": [], "tables": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    tex, _ = render_companion_tex(
        document,
        [{
            "segment_id": "body", "block_ids": ["body"],
            "start_block_id": "body", "end_block_id": "body",
        }],
        {"body": {"explanation": "COMPANION NOTE"}},
        translations={"body": {"blocks": [
            {"block_id": "body", "translate": True, "text": "TRANSLATED BODY"},
        ]}},
        glossary={"entries": [{
            "source_term": "SOURCE TERM", "target_term": "TARGET TERM",
            "brief_explanation": "GLOSSARY NOTE",
        }]},
        output_dir=tmp_path,
        language="en",
    )

    assert (
        tex.index("SOURCE BODY")
        < tex.index("REFERENCE TEXT")
        < tex.index(r"\clearpage")
        < tex.index("GLOSSARY NOTE")
        < tex.index(r"\end{document}")
    )


def test_bibliography_reuses_translated_structural_heading_without_duplicate(
    tmp_path: Path,
) -> None:
    document = {
        "front_matter": {},
        "blocks": [
            {
                "block_id": "refs-title", "kind": "heading",
                "section_id": "references", "text": "Literatur",
            },
            {
                "block_id": "ref", "kind": "bibliography",
                "section_id": "references", "text": "Reference text.",
            },
        ],
        "bibliography": [{"id": "ref", "label": "[1]", "text": "Reference text."}],
        "equations": [], "figures": [], "tables": [], "assets": [],
        "integrity": {"status": "complete"},
    }

    tex, manifest = render_companion_tex(
        document,
        [],
        {},
        output_dir=tmp_path,
        source_language="de",
        language="zh-CN",
        title_translations={
            "titles": [{"title_id": "block:refs-title", "text": "参考文献"}],
        },
    )

    assert tex.count("Literatur") == 1
    assert tex.count(r"\section*{Literatur") == 1
    assert "参考文献" in tex
    assert r"\section*{References}" not in tex
    assert tex.count(r"\begin{thebibliography}{9999}") == 1
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_same_language_render_ignores_passed_glossary_and_preserves_source_index(
    tmp_path: Path,
) -> None:
    document = {
        "front_matter": {},
        "blocks": [
            {"block_id": "body", "kind": "prose", "text": "SOURCE BODY"},
            {
                "block_id": "index-heading", "kind": "heading", "level": 1,
                "title": "Index", "text": "Index", "source_role": "index",
            },
            {
                "block_id": "index-entry", "kind": "prose",
                "text": "Gauge field, 42", "source_role": "index",
            },
        ],
        "bibliography": [], "equations": [], "figures": [], "tables": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    tex, manifest = render_companion_tex(
        document,
        [{
            "segment_id": "body", "block_ids": ["body"],
            "start_block_id": "body", "end_block_id": "body",
        }],
        {"body": {"explanation": "COMPANION NOTE"}},
        translations=None,
        glossary={"entries": [{
            "source_term": "STALE SOURCE TERM", "target_term": "STALE TARGET TERM",
            "brief_explanation": "STALE GLOSSARY NOTE",
        }]},
        output_dir=tmp_path,
        language="en",
        augmentation_scope="substantive",
    )

    assert manifest["companion_layers"]["translation_mode"] is False
    assert "STALE SOURCE TERM" not in tex
    assert "STALE TARGET TERM" not in tex
    assert "STALE GLOSSARY NOTE" not in tex
    assert r"\section*{Glossary}" not in tex
    assert tex.count(r"\section*{Index}") == 1
    assert tex.count("Gauge field, 42") == 1
    assert tex.index("SOURCE BODY") < tex.index(r"\section*{Index}") < tex.index("Gauge field, 42")


def test_reader_layers_remove_html_wrappers_and_internal_evidence_labels(tmp_path: Path) -> None:
    source = "Reader source prose."
    document = {
        "front_matter": {},
        "blocks": [{"block_id": "p", "kind": "prose", "text": source}],
        "equations": [], "figures": [], "tables": [], "bibliography": [],
        "assets": [], "links": [], "integrity": {"status": "complete"},
    }
    segments = [{
        "segment_id": "seg-internal",
        "block_ids": ["p"],
        "start_block_id": "p",
        "end_block_id": "p",
    }]
    translations = {"seg-internal": {"blocks": [{
        "block_id": "p",
            "text": "<details><summary>natural_image</summary>有意义的译文。</details>",
    }]}}
    annotations = {"seg-internal": {
        "commentary": "解释【context-b65f8935dea5c99f8b42】。",
        "explanation": "解释（证据：context-b65f8935dea5c99f8b42）。",
        "prior_work": [{"text": "已有结果", "evidence_ids": ["prior-001"]}],
        "later_work": "",
        "evidence_ids": ["context-b65f8935dea5c99f8b42", "prior-001"],
    }}
    evidence_records = [{
        "evidence_id": "context-b65f8935dea5c99f8b42",
        "title": "Quantum Field Theory",
        "blocks": [
            {"block_id": "sec.heading", "text": "2 Scattering"},
            {"block_id": "sec.p1", "text": "Relevant discussion"},
        ],
        "selected_snippets": [{"block_id": "sec.p1", "text": "Relevant discussion"}],
    }, {
        "evidence_id": "prior-001",
        "title": "Earlier Work",
    }]

    tex, manifest = render_companion_tex(
        document,
        segments,
        annotations,
        translations=translations,
        evidence_by_segment={"seg-internal": evidence_records},
        output_dir=tmp_path,
        language="zh-CN",
    )

    assert "Reader source prose" in tex
    assert "有意义的译文" not in tex
    assert "已有结果" in tex
    assert "details" not in tex
    assert "summary" not in tex
    assert r"natural\_image" not in tex
    assert "context-b65f8935dea5c99f8b42" not in tex
    assert "prior-001" not in tex
    assert "证据：" not in tex
    assert "Quantum Field Theory" in tex
    assert "2 Scattering" in tex
    assert "Earlier Work" in tex
    assert "伴读单元" not in tex
    assert r"\textbf{原文}" not in tex
    assert "seg-internal" not in _layer_region(tex, "TRANSLATION", "seg-internal")
    assert "seg-internal" not in _layer_region(tex, "COMPANION", "seg-internal")
    assert "% ARC-TRANSLATION-BEGIN " in tex
    assert "% ARC-COMPANION-BEGIN " in tex
    assert manifest["companion_layers"]["rendered_translation_segment_ids"] == ["seg-internal"]
    assert manifest["companion_layers"]["rendered_annotation_segment_ids"] == ["seg-internal"]
    assert manifest["blocks"][0]["sha256"]
    assert document["blocks"][0]["text"] == source
    assert annotations["seg-internal"]["evidence_ids"] == [
        "context-b65f8935dea5c99f8b42", "prior-001"
    ]
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_review_normalization_drops_machine_detail_body_and_keeps_evidence() -> None:
    annotation = {
        "explanation": "<details><summary>natural_image</summary>解释正文。</details>【context-1】",
        "prior_work": [{"text": "前人结论 [prior-1]", "evidence_ids": ["prior-1"]}],
        "evidence_ids": ["context-1", "prior-1"],
    }
    translation = {"blocks": [{
        "block_id": "p",
        "text": (
            "&lt;details&gt;&lt;summary&gt;OCR metadata&lt;/summary&gt;"
            "译文正文。&lt;/details&gt;"
        ),
    }]}

    cleaned_annotation = clean_reader_annotation(
        annotation,
        evidence_records=[
            {"evidence_id": "context-1", "title": "Reference Book"},
            {"evidence_id": "prior-1", "title": "Prior Paper"},
        ],
        language="zh-CN",
    )
    cleaned_translation = clean_reader_translation(translation)

    assert cleaned_annotation["explanation"] == ""
    assert cleaned_annotation["prior_work"][0]["text"] == "前人结论（参考：《Prior Paper》）"
    assert cleaned_annotation["evidence_ids"] == ["context-1", "prior-1"]
    assert cleaned_annotation["prior_work"][0]["evidence_ids"] == ["prior-1"]
    assert cleaned_translation["blocks"][0]["text"] == ""
    assert annotation["explanation"].startswith("<details>")
    assert translation["blocks"][0]["text"].startswith("&lt;details&gt;")


def test_markdown_escaped_detail_markup_is_cleaned_in_rich_html_renderer() -> None:
    rendered = _render_html_fragment(
        "<p>Before &lt;details&gt;<br/>"
        "&lt;summary&gt;natural_image&lt;/summary&gt;<br/>"
        "Meaningful body.<br/>&lt;/details&gt; "
        "&lt;details&gt;&lt;summary&gt;Proof sketch&lt;/summary&gt;"
        "Argument body.&lt;/details&gt; After "
        "<math alttext=\"x+y\"><semantics><annotation "
        "encoding=\"application/x-tex\">x+y</annotation></semantics></math>.</p>",
        rendered_links=[],
    )

    assert "Meaningful body" not in rendered
    assert "Proof sketch" in rendered and "Argument body" in rendered
    assert "Before" in rendered and "After" in rendered
    assert r"\(x+y\)" in rendered
    assert "details" not in rendered
    assert "summary" not in rendered
    assert "natural" not in rendered


def test_cleaned_html_wrappers_do_not_leave_bare_line_break_commands() -> None:
    empty_wrapper = _render_html_fragment(
        '<p id="wrapper">&lt;details&gt;<br/>'
        '&lt;summary&gt;natural_image&lt;/summary&gt;</p>',
        rendered_links=[],
    )
    described_image = _render_html_fragment(
        "<p>Meaningful image description.<br/>&lt;/details&gt;</p>",
        rendered_links=[],
    )
    interior_break = _render_html_fragment(
        "<p>First reader-visible line.<br/>Second reader-visible line.</p>",
        rendered_links=[],
    )

    assert empty_wrapper.strip() == r"\phantomsection\label{wrapper}"
    assert described_image.strip() == "Meaningful image description."
    assert r"\\" not in empty_wrapper
    assert r"\\" not in described_image
    assert "First reader-visible line.\\\\\nSecond reader-visible line." in interior_break


@pytest.mark.skipif(shutil.which("xelatex") is None, reason="XeLaTeX is not installed")
def test_cleaned_empty_html_wrapper_compiles_outside_a_paragraph(tmp_path: Path) -> None:
    rendered = _render_html_fragment(
        '<p id="wrapper">&lt;details&gt;<br/>'
        '&lt;summary&gt;natural_image&lt;/summary&gt;</p>',
        rendered_links=[],
    )
    tex_path = tmp_path / "empty-reader-wrapper.tex"
    tex_path.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{hyperref}\n"
        "\\begin{document}\n"
        f"{rendered}\n"
        "Visible prose.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            shutil.which("xelatex") or "xelatex",
            "-halt-on-error",
            "-interaction=nonstopmode",
            f"-output-directory={tmp_path}",
            str(tex_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    log = (tmp_path / "empty-reader-wrapper.log").read_text(
        encoding="utf-8", errors="replace"
    )
    assert result.returncode == 0, result.stdout + result.stderr + log
    assert "There's no line here to end" not in log


def test_reader_cleanup_replaces_bare_unwrapped_and_soft_wrapped_registered_ids() -> None:
    annotation = {
        "explanation": "Bare context-abc. 证据：prior-001。软换行【context-\nabc】。",
        "commentary": "Unregistered context-other remains auditable prose.",
        "prior_work": "",
        "later_work": "",
        "evidence_ids": ["context-abc", "prior-001"],
    }

    cleaned = clean_reader_annotation(
        annotation,
        evidence_records=[
            {"evidence_id": "context-abc", "title": "Reference Book"},
            {"evidence_id": "prior-001", "title": "Prior Paper"},
        ],
        language="zh-CN",
    )

    assert "context-abc" not in cleaned["explanation"]
    assert "prior-001" not in cleaned["explanation"]
    assert "Reference Book" in cleaned["explanation"]
    assert "Prior Paper" in cleaned["explanation"]
    assert "context-other" in cleaned["commentary"]
    assert cleaned["evidence_ids"] == ["context-abc", "prior-001"]


def test_reader_cleanup_uses_nested_bindings_and_exact_reader_location() -> None:
    annotation = {
        "commentary": (
            "所给背景资料和所给 context 证据只支持这一点【context-\nabc】；"
            "另见证据：prior-001。"
        ),
        "explanation": "",
        "prior_work": [],
        "later_work": [],
        # Legacy annotations can lack the top-level union while retaining the
        # claim-level audit binding and exact source locator.
        "context_claims": [{
            "text": "Audited claim.",
            "evidence_ids": ["context-abc", "prior-001"],
            "source_locators": [
                {"evidence_id": "context-abc", "locator": "sec_0042"},
                {"evidence_id": "prior-001", "locator": "chapter-3"},
            ],
        }],
    }
    records = [{
        "evidence_id": "context-abc",
        "title": "Reference Book",
        "section_title": "Generic root section",
        "selected_snippets": [{
            "block_id": "sec_wrong",
            "section_title": "Unrelated section",
            "text": "Wrong passage.",
        }],
        "blocks": [{
            "block_id": "sec_0042",
            "section_title": "1.2 Why quantum field theory?",
            "text": "Exact cited passage.",
        }],
    }, {
        "evidence_id": "prior-001",
        "title": "Prior Paper",
        "blocks": [{
            "block_id": "chapter-3",
            "section_title": "Chapter 3: Scattering",
            "text": "Exact cited passage.",
        }],
    }]

    cleaned = clean_reader_annotation(
        annotation, evidence_records=records, language="zh-CN"
    )

    assert "context-abc" not in cleaned["commentary"]
    assert "prior-001" not in cleaned["commentary"]
    assert "context 证据" not in cleaned["commentary"]
    assert "所给背景资料" not in cleaned["commentary"]
    assert "所引参考资料" in cleaned["commentary"]
    assert "《Reference Book》，1.2 Why quantum field theory?" in cleaned["commentary"]
    assert "《Prior Paper》，Chapter 3: Scattering" in cleaned["commentary"]
    assert "Unrelated section" not in cleaned["commentary"]
    assert "Generic root section" not in cleaned["commentary"]
    assert cleaned["context_claims"] == annotation["context_claims"]


def test_reader_cleanup_uses_segment_records_when_legacy_annotation_lost_ids() -> None:
    annotation = {
        "commentary": "Legacy prose cites context-\nabc without a retained ID union.",
        "explanation": "",
        "prior_work": [],
        "later_work": [],
    }

    cleaned = clean_reader_annotation(
        annotation,
        evidence_records=[{
            "evidence_id": "context-abc",
            "title": "Reference Book",
            "selected_snippets": [{
                "block_id": "sec_0042",
                "section_title": "Section 4.2",
                "text": "Cited passage.",
            }],
        }],
        language="en",
    )

    assert cleaned["commentary"] == (
        "Legacy prose cites (Source: Reference Book, Section 4.2) "
        "without a retained ID union."
    )


def test_unmatched_exact_locator_falls_back_to_title_not_wrong_section() -> None:
    annotation = {
        "commentary": "说明【context-abc】。",
        "explanation": "",
        "prior_work": [],
        "later_work": [],
        "context_claims": [{
            "evidence_ids": ["context-abc"],
            "source_locators": [{
                "evidence_id": "context-abc",
                "locator": "missing-exact-block",
            }],
        }],
    }
    record = {
        "evidence_id": "context-abc",
        "title": "Reference Book",
        "section_title": "Wrong generic root section",
        "selected_snippets": [{
            "block_id": "old-selection",
            "section_title": "Wrong old selected section",
            "text": "Old passage.",
        }],
        "snippets": [{
            "block_id": "old-snippet",
            "section_title": "Wrong old snippet section",
            "text": "Old passage.",
        }],
        "blocks": [{
            "block_id": "available-block",
            "section_title": "Wrong available section",
            "text": "A different passage.",
        }],
    }

    cleaned = clean_reader_annotation(
        annotation, evidence_records=[record], language="zh-CN"
    )

    assert cleaned["commentary"] == "说明（参考：《Reference Book》）。"
    assert "Wrong" not in cleaned["commentary"]


def test_actual_html_summary_keeps_reader_heading_unless_it_is_machine_only() -> None:
    rendered = _render_html_fragment(
        "<details><summary>Proof sketch</summary><p>Argument.</p></details>"
        "<details><summary>OCR metadata</summary><p>Recognized text.</p></details>",
        rendered_links=[],
    )

    assert "Proof sketch" in rendered and "Argument" in rendered
    assert "OCR metadata" not in rendered
    assert "Recognized text" not in rendered


def test_escaped_machine_details_do_not_consume_neighboring_authored_details() -> None:
    rendered = _render_html_fragment(
        "Before &lt;details&gt;Authored body.&lt;/details&gt; Between "
        "&lt;details&gt;&lt;summary&gt;natural_image&lt;/summary&gt;"
        "Generated description.&lt;/details&gt; After",
        rendered_links=[],
    )

    assert "Before" in rendered and "Authored body" in rendered
    assert "Between" in rendered and "After" in rendered
    assert "Generated description" not in rendered


def test_legacy_evidence_snippet_recovers_only_an_explicit_heading_prefix() -> None:
    annotation = {
        "commentary": "说明【context-old】",
        "explanation": "说明",
        "prior_work": "",
        "later_work": "",
        "evidence_ids": ["context-old"],
    }
    cleaned = clean_reader_annotation(
        annotation,
        evidence_records=[{
            "evidence_id": "context-old",
            "title": "Modern Quantum Field Theory",
            "snippets": [{
                "block_id": "sec_0019",
                "text": "1.2 Why quantum field theory? Students often enter from quantum mechanics.",
            }],
        }],
        language="zh-CN",
    )

    assert cleaned["commentary"] == (
        "说明（参考：《Modern Quantum Field Theory》，1.2 Why quantum field theory?）"
    )


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


def _leading_bracket_translation_tex(tmp_path: Path) -> str:
    document = {
        "front_matter": {"title": "Leading bracket"},
        "blocks": [{
            "block_id": "p", "kind": "prose", "text": "Source footnote marker.",
        }],
        "equations": [], "figures": [], "tables": [], "bibliography": [],
        "assets": [], "links": [], "integrity": {"status": "complete"},
    }
    tex, _ = render_companion_tex(
        document,
        [{"segment_id": "s", "block_ids": ["p"]}],
        {"s": {"explanation": "Companion note."}},
        translations={"s": {"blocks": [{
            "block_id": "p", "text": "[^1] Leading translated footnote marker.",
        }]}},
        output_dir=tmp_path,
        language="en",
        source_language="en",
    )
    return tex


def test_renderer_guards_optional_box_arguments_from_leading_bracket(
    tmp_path: Path,
) -> None:
    tex = _leading_bracket_translation_tex(tmp_path)

    assert _begin_guarded_box("arctranslation") == (
        "\\begin{arctranslation}\\relax\n"
    )
    assert _begin_guarded_box("arccompanion") == "\\begin{arccompanion}\\relax\n"
    assert _begin_guarded_box("arcchapterguide") == (
        "\\begin{arcchapterguide}\\relax\n"
    )
    translation = tex.split(r"\begin{arctranslation}", 1)[1].split(
        r"\end{arctranslation}", 1
    )[0]
    assert translation.startswith(r"\relax" + "\n[")
    assert "Leading translated footnote marker." in translation


@pytest.mark.skipif(shutil.which("xelatex") is None, reason="XeLaTeX is not installed")
def test_leading_bracket_translation_box_compiles_with_xelatex(tmp_path: Path) -> None:
    tex_path = tmp_path / "leading-bracket-translation.tex"
    tex_path.write_text(_leading_bracket_translation_tex(tmp_path), encoding="utf-8")

    result = subprocess.run(
        [
            shutil.which("xelatex") or "xelatex",
            "-halt-on-error",
            "-interaction=nonstopmode",
            f"-output-directory={tmp_path}",
            str(tex_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    log = (tmp_path / "leading-bracket-translation.log").read_text(
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stdout + result.stderr + log
    assert "Missing \\endcsname" not in log


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


def test_equation_renderer_disambiguates_bracketed_array_row_starts() -> None:
    rendered = _render_equation({
        "tex": (
            r"\begin{array}{l} [c]=LT^{-1} \\ [\hbar]=L^2MT^{-1} "
            r"\\[2pt] [G]=L^3M^{-1}T^{-2} \end{array}"
        ),
    })

    assert r"\\{} [\hbar]" in rendered
    assert r"\\[2pt] [G]" in rendered


def test_equation_renderer_canonicalizes_embedded_tag_or_adds_cached_number(tmp_path: Path) -> None:
    document = {
        "blocks": [{"block_id": "eq", "kind": "equation", "equation_id": "eq"}],
        "equations": [{"id": "eq", "tex": r"x=y\tag {0.1}", "number": "(0.1)"}],
        "integrity": {"status": "complete"},
    }
    tex, manifest = render_companion_tex(
        document,
        [{"segment_id": "s", "block_ids": ["eq"]}],
        {"s": {"commentary": "note"}},
        output_dir=tmp_path,
        language="en",
    )

    assert len(re.findall(r"\\tag\*?\s*\{", tex)) == 1
    assert r"\tag{0.1}" in tex
    assert r"\tag {0.1}" not in tex
    assert validate_tex_fidelity(tex, document, manifest) == []

    cached_only = _render_equation({"tex": "z=1", "number": "(0.2)"})
    assert cached_only.count(r"\tag{0.2}") == 1

    embedded_only = _render_equation({"tex": r"w=2\tag{raw}"})
    assert embedded_only.count(r"\tag{raw}") == 1


def test_renderer_drops_unsupported_unicode_controls(tmp_path: Path) -> None:
    document = {
        "blocks": [{"block_id": "p", "kind": "prose", "text": "a\x03b\x0fc\x7fd\x85e"}],
        "integrity": {"status": "complete"},
    }
    tex, _ = render_companion_tex(
        document,
        [{"segment_id": "s", "block_ids": ["p"]}],
        {"s": {"commentary": "x\x03y\x0fz\\[\\partial\x7f\\phi\\]"}},
        output_dir=tmp_path,
        language="en",
    )

    assert "abcde" in tex and "xyz" in tex
    assert r"\[\partial\phi\]" in tex
    assert not any(
        unicodedata.category(char) == "Cc"
        for char in tex
        if char not in "\n\r\t"
    )


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


def test_substantive_scope_suppresses_front_routes_but_keeps_preface_layers(tmp_path: Path) -> None:
    preface = (
        "This preface explains the motivation, scope, assumptions, and conceptual "
        "background of the work in enough detail to constitute substantive prose. "
    ) * 2
    document = {
        "front_matter": {
            "title": "A Book",
            "authors": ["A. Author"],
            "block_ids": {"title": ["title"], "authors": ["author"]},
        },
        "blocks": [
            {"block_id": "title", "kind": "heading", "section_id": "fm1", "text": "A Book"},
            {"block_id": "author", "kind": "prose", "section_id": "fm2", "text": "A. Author"},
            {"block_id": "resources", "kind": "heading", "section_id": "route", "text": "Reading list"},
            {"block_id": "resource-1", "kind": "prose", "section_id": "route", "text": "• First source"},
            {"block_id": "resource-2", "kind": "prose", "section_id": "route", "text": "• Second source"},
            {"block_id": "foreword", "kind": "heading", "section_id": "foreword", "text": "Foreword"},
            {"block_id": "foreword-text", "kind": "prose", "section_id": "foreword", "text": preface},
            {"block_id": "toc", "kind": "heading", "section_id": "toc", "source_role": "table_of_contents", "text": "Contents"},
            {"block_id": "toc-preface", "kind": "heading", "section_id": "toc-p", "text": "Preface 3"},
            {"block_id": "toc-body", "kind": "heading", "section_id": "toc-b", "text": "1 Beginning 5"},
            {"block_id": "preface", "kind": "heading", "section_id": "preface", "text": "Preface"},
            {"block_id": "preface-text", "kind": "prose", "section_id": "preface", "text": preface},
            {"block_id": "body", "kind": "heading", "section_id": "body", "text": "1 Beginning"},
            {"block_id": "body-text", "kind": "prose", "section_id": "body", "text": "Substantive body text."},
        ],
        "integrity": {"status": "complete"},
    }
    segments = [
        {"segment_id": "meta", "block_ids": ["author"]},
        {"segment_id": "route", "block_ids": ["resources", "resource-1", "resource-2"]},
        {"segment_id": "foreword", "block_ids": ["foreword", "foreword-text"]},
        {"segment_id": "toc-entry", "block_ids": ["toc-preface", "toc-body"]},
        {"segment_id": "preface", "block_ids": ["preface", "preface-text"]},
        {"segment_id": "body", "block_ids": ["body", "body-text"]},
    ]
    annotations = {
        segment["segment_id"]: {"commentary": f"COMMENT-{segment['segment_id']}"}
        for segment in segments
    }
    translations = {
        segment["segment_id"]: {"blocks": [
            {"block_id": bid, "text": f"TRANSLATION-{segment['segment_id']}-{bid}"}
            for bid in segment["block_ids"]
        ]}
        for segment in segments
    }

    tex, manifest = render_companion_tex(
        document,
        segments,
        annotations,
        translations=translations,
        output_dir=tmp_path,
        language="en",
        augmentation_scope="substantive",
    )

    # Source remains visible, while cached generated layers for route material do not.
    assert "First source" in tex and "Preface 3" in tex
    for segment_id in ("meta", "route", "toc-entry"):
        assert f"COMMENT-{segment_id}" not in tex
        assert f"TRANSLATION-{segment_id}" not in tex
    for segment_id in ("foreword", "preface", "body"):
        assert f"COMMENT-{segment_id}" in tex
        assert f"TRANSLATION-{segment_id}" in tex
    layers = manifest["companion_layers"]
    assert layers["semantic_segment_ids"] == ["foreword", "preface", "body"]
    assert layers["preservation_only_segment_ids"] == ["meta", "route", "toc-entry"]
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_substantive_scope_keeps_preface_layers_in_mixed_toc_segment(tmp_path: Path) -> None:
    document = {
        "front_matter": {},
        "blocks": [
            {
                "block_id": "toc", "kind": "heading", "section_id": "toc",
                "source_role": "table_of_contents", "text": "Contents",
            },
            {
                "block_id": "preface", "kind": "heading", "section_id": "preface",
                "text": "Preface",
            },
            {
                "block_id": "preface-text", "kind": "prose", "section_id": "preface",
                "text": "Substantive motivation for the book.",
            },
        ],
        "integrity": {"status": "complete"},
    }
    segments = [{"segment_id": "mixed", "block_ids": ["toc", "preface", "preface-text"]}]
    annotations = {"mixed": {"commentary": "PREFACE-COMMENTARY"}}
    translations = {"mixed": {"blocks": [
        {"block_id": "toc", "text": "TOC-TRANSLATION"},
        {"block_id": "preface", "text": "PREFACE-TRANSLATION"},
        {"block_id": "preface-text", "text": "PREFACE-TEXT-TRANSLATION"},
    ]}}

    tex, manifest = render_companion_tex(
        document,
        segments,
        annotations,
        translations=translations,
        output_dir=tmp_path,
        language="en",
        augmentation_scope="substantive",
    )

    assert "Contents" in tex
    assert "TOC-TRANSLATION" not in tex
    assert "PREFACE-TRANSLATION" in tex
    assert "PREFACE-TEXT-TRANSLATION" in tex
    assert tex.count("PREFACE-COMMENTARY") == 1
    assert manifest["companion_layers"]["semantic_segment_ids"] == ["mixed"]
    translation = manifest["companion_layers"]["translations"][0]
    assert translation["translated_block_ids"] == ["preface", "preface-text"]
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_16_bit_alpha_png_is_detected_for_xelatex_flattening(tmp_path: Path) -> None:
    rgba16 = tmp_path / "rgba16.png"
    rgba16.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + b"\x00" * 8 + bytes([16, 6]))
    rgb16 = tmp_path / "rgb16.png"
    rgb16.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + b"\x00" * 8 + bytes([16, 2]))

    assert _png_needs_latex_flattening(rgba16)
    assert not _png_needs_latex_flattening(rgb16)


def test_unicode_math_glyphs_are_normalized_without_touching_equation_numbers() -> None:
    glyphs = "δνθσζτ αεϵβφπη ∼≪≫≲∝≡≈∑∫ℓ⟨⟩∙ Ḣ ℒℋℏ −√∂→⊥∥ᐟ 𝐻𝐤𝜃𝒪𝑚𝜎𝑝𝛿𝑡𝑁𝐿𝐼𝜈𝜁𝑉𝑅𝒞𝑓𝑎𝑙𝑘𝑖𝑏𝑃 ⁿ₀₁₂₃ᵢ′\u200b"
    rendered = escape_tex(glyphs)

    for glyph in glyphs.replace(" ", "").replace("\u200b", ""):
        if glyph in "δνθσζταεϵβφπη":
            continue
        assert glyph not in rendered
    assert "δνθσζτ αεϵβφπη" in rendered
    assert r"\int" in rendered
    assert r"\ell" in rendered
    assert r"{}^{n}" in rendered
    assert r"\mathcal{O}" in rendered
    assert r"\mathbf{k}" in rendered
    assert r"\surd" in rendered
    assert r"\partial" in rendered
    assert r"\to" in rendered
    assert r"\perp" in rendered
    assert r"\parallel" in rendered
    assert "ᐟ" not in rendered and "/" in rendered
    assert escape_tex("ℏ") == r"{\rmfamily\(\hbar\)}"
    assert r"\textsubscript{2}" in rendered
    assert r"\textsuperscript{{\rmfamily\(\prime\)}}" in rendered

    assert escape_tex("Η ειδική θεωρία της σχετικότητας") == "Η ειδική θεωρία της σχετικότητας"
    assert escape_tex("field ψ†γ⁰") == r"field {\rmfamily\(\psi\dagger\gamma{}^{0}\)}"

    glossary_math = _render_glossary({"entries": [{
        "source_term": "Dirac adjoint",
        "target_term": "狄拉克共轭",
        "brief_explanation": "ψ†γ⁰; φ⁴ theory",
    }]}, language="zh-CN")
    assert r"\(\psi\dagger\gamma{}^{0}\)" in glossary_math
    assert r"\(\phi{}^{4}\)" in glossary_math
    assert not any(glyph in glossary_math for glyph in "ψ†γ⁰φ⁴")

    equation = _render_equation({"tex": r"x=y", "number": "(A.12)", "label": "eq:a12"})
    assert r"\tag{A.12}" in equation
    assert r"\label{eq:a12}" in equation


def test_preamble_uses_sans_body_and_serif_math_with_cjk_fallbacks() -> None:
    tex = _preamble(title="T", authors="A", language="zh-CN")

    assert r"\setCJKsansfont{Noto Sans CJK SC}" in tex
    assert "Source Han Sans SC" in tex
    assert "Source Han Sans CN" in tex
    assert "FandolHei-Regular" in tex
    assert "\\begin{document}\n\\sffamily" in tex
    assert r"\setCJKmainfont{Noto Serif CJK SC}" in tex
    assert "阅读语言: zh-CN" in tex
    assert "伴读语言" not in tex
    english = _preamble(title="T", authors="A", language="en")
    assert "Reading language: en" in english
    assert "Companion language" not in english
    equation = _equation_environment(r"x=\text{mass}", number=None, label=None)
    assert r"\begingroup\rmfamily" in equation


def test_chapter_opening_reserves_space_for_title_and_guide_body(tmp_path: Path) -> None:
    document = {
        "front_matter": {},
        "blocks": [
            {"block_id": "chapter-title", "kind": "heading", "text": "Chapter One"},
            {"block_id": "body", "kind": "prose", "text": "Source body."},
        ],
        "equations": [], "figures": [], "tables": [], "bibliography": [],
        "assets": [], "links": [], "integrity": {"status": "complete"},
    }
    tex, _ = render_companion_tex(
        document,
        [{"segment_id": "s1", "block_ids": ["chapter-title", "body"]}],
        {"s1": {"explanation": "Reader explanation.", "commentary": ""}},
        output_dir=tmp_path,
        language="zh-CN",
        chapters=[{
            "chapter_id": "ch-1",
            "block_ids": ["chapter-title", "body"],
        }],
        chapter_guides={"ch-1": {
            "main_content": "Guide opening text.",
            "pedagogical_comparison": {
                "text": "Textbooks reverse this order.",
                "sources": [{"title": "Course notes", "url": "https://example.test/notes", "locator": "Chapter 2"}],
            },
            "historical_context": [{
                "text": "The result was found historically.",
                "sources": [{"title": "History", "url": "https://example.test/history", "locator": "p. 4"}],
            }],
        }},
    )

    opening_guard = tex.index(r"\Needspace{10\baselineskip}")
    assert opening_guard < tex.index("Chapter One") < tex.index("ARC-CHAPTER-GUIDE-BEGIN")
    guide_region = tex[
        tex.index("ARC-CHAPTER-GUIDE-BEGIN"):tex.index("ARC-CHAPTER-GUIDE-END")
    ]
    assert r"\begin{arcchapterguide}" in guide_region
    assert r"\end{arcchapterguide}" in guide_region
    assert r"\begin{quote}" not in guide_region
    assert r"\Needspace{4\baselineskip}" in guide_region
    assert tex.index("章导读") < tex.index("Guide opening text.")
    assert (
        "\\Needspace{4\\baselineskip}\n\\medskip\\noindent\\textbf{主要内容}"
        in tex
    )
    assert "教材顺序比较" in tex and "历史背景" in tex
    assert "https://example.test/notes" in tex
    assert "https://example.test/history" in tex
    assert "原文位置" not in tex and "补充阅读" not in tex


def test_empty_chapter_guide_and_structural_only_segment_render_without_empty_layers(
    tmp_path: Path,
) -> None:
    document = {
        "front_matter": {"title": "Unique Paper Title"},
        "blocks": [
            {"block_id": "paper-title", "kind": "heading", "text": "Unique Paper Title", "source_role": "front_matter_title"},
            {"block_id": "chapter-title", "kind": "chapter", "text": "Only Title"},
        ],
        "equations": [], "figures": [], "tables": [], "bibliography": [],
        "assets": [], "links": [], "integrity": {"status": "complete"},
    }

    tex, manifest = render_companion_tex(
        document,
        [{
            "segment_id": "s1", "block_ids": ["paper-title", "chapter-title"],
            "augmentation_block_ids": [], "structural_only": True,
        }],
        {},
        output_dir=tmp_path,
        language="zh-CN",
        chapters=[{"chapter_id": "ch-1", "block_ids": ["paper-title", "chapter-title"]}],
        chapter_guides={"ch-1": {
            "motivation": None, "main_content": None, "section_logic": None,
            "prerequisites": None, "pedagogical_comparison": None,
            "historical_context": [],
        }},
    )

    assert tex.count("Only Title") == 2
    assert tex.count("Unique Paper Title") == 1
    assert "ARC-CHAPTER-GUIDE-BEGIN" not in tex
    assert "ARC-TRANSLATION-BEGIN" not in tex
    assert "ARC-COMPANION-BEGIN" not in tex
    assert manifest["companion_layers"]["semantic_segment_ids"] == []
    assert manifest["companion_layers"]["preservation_only_segment_ids"] == ["s1"]
    assert manifest["companion_layers"]["expected_chapter_guide_ids"] == []
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_pdf_renders_bilingual_titles_with_translated_navigation_and_no_title_companion(
    tmp_path: Path,
) -> None:
    document = {
        "front_matter": {"title": "Die Relativitätstheorie"},
        "blocks": [
            {
                "block_id": "paper-title", "kind": "heading",
                "text": "Die Relativitätstheorie", "source_role": "front_matter_title",
            },
            {"block_id": "chapter-title", "kind": "chapter", "text": "Grundlagen"},
        ],
        "equations": [], "figures": [], "tables": [], "bibliography": [],
        "assets": [], "links": [], "integrity": {"status": "complete"},
    }
    title_translations = {
        "schema_version": "arc.companion.title-translations.v1",
        "source_language": "de",
        "target_language": "zh-CN",
        "source_sha256": "0" * 64,
        "titles": [
            {"title_id": "document:title", "role": "document_title", "block_id": "paper-title", "text": "相对论"},
            {"title_id": "block:chapter-title", "role": "chapter", "block_id": "chapter-title", "chapter_id": "ch-1", "text": "基础"},
        ],
    }

    tex, manifest = render_companion_tex(
        document,
        [{
            "segment_id": "s1", "block_ids": ["paper-title", "chapter-title"],
            "augmentation_block_ids": [], "structural_only": True,
        }],
        {},
        output_dir=tmp_path,
        language="zh-CN",
        source_language="de",
        title_translations=title_translations,
        chapters=[{
            "chapter_id": "ch-1", "block_ids": ["paper-title", "chapter-title"],
            "title_block_ids": ["chapter-title"],
        }],
        chapter_guides={},
    )

    assert "Die Relativitätstheorie" in tex and "相对论" in tex
    assert r"\section*{Grundlagen\protect\\{\small 基础}}" in tex
    assert r"\addcontentsline{toc}{section}{基础}" in tex
    assert "ARC-COMPANION-BEGIN" not in tex
    assert manifest["source_language"] == "de"
    assert manifest["target_language"] == "zh-CN"
    assert manifest["render_warnings"] == []
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_pdf_manifest_warns_for_rtl_layout_without_rejecting_unicode(tmp_path: Path) -> None:
    document = {
        "blocks": [{"block_id": "b1", "kind": "heading", "text": "النسبية"}],
        "equations": [], "figures": [], "tables": [], "bibliography": [],
        "assets": [], "links": [], "integrity": {"status": "complete"},
    }
    tex, manifest = render_companion_tex(
        document,
        [{"segment_id": "s1", "block_ids": ["b1"], "augmentation_block_ids": [], "structural_only": True}],
        {},
        output_dir=tmp_path,
        language="en",
        source_language="ar",
    )

    assert "النسبية" in tex
    assert manifest["render_warnings"] == [{
        "code": "rtl_pdf_layout_not_guaranteed",
        "languages": ["ar"],
        "message": "PDF bidirectional layout is best-effort; use the HTML reader for authoritative RTL direction.",
    }]


@pytest.mark.skipif(
    shutil.which("xelatex") is None or shutil.which("pdftotext") is None,
    reason="XeLaTeX and pdftotext are required",
)
def test_common_unicode_math_glyphs_compile_as_math_atoms(tmp_path: Path) -> None:
    rendered = escape_tex(
        "Planck constant ℏ; integral ∫; length ℓ; power 10ⁿ; "
        "minus −; root √; partial ∂; arrow →; perpendicular ⊥; "
        "parallel ∥; normalized slash A ᐟ B"
    )
    assert r"\(\hbar\)" in rendered
    assert r"\(\int\)" in rendered
    assert r"\(\ell\)" in rendered
    assert r"\({}^{n}\)" in rendered
    assert "ᐟ" not in rendered and "A / B" in rendered

    tex_path = tmp_path / "planck-constant.tex"
    tex_path.write_text(
        _preamble(
            title="Unicode math glyphs", translated_title="", authors="",
            language="en", source_language="en",
        )
        + f"{rendered}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            shutil.which("xelatex") or "xelatex",
            "-halt-on-error",
            "-interaction=nonstopmode",
            f"-output-directory={tmp_path}",
            str(tex_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    log = (tmp_path / "planck-constant.log").read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr + log
    assert "Missing character" not in log
    extracted = subprocess.run(
        ["pdftotext", str(tmp_path / "planck-constant.pdf"), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    for glyph in "−√∂→⊥∥":
        assert glyph in extracted
    assert "A / B" in extracted
