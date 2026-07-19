from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc_companion import cli
from arc_companion.latex import (
    LatexError,
    _render_html_fragment,
    _render_table,
    render_companion_tex,
    validate_tex_fidelity,
)
from arc_companion.package import package_project
from arc_companion.source import SourceError, load_source_bundle, validate_complete_document


ROOT = Path(__file__).resolve().parents[3]


def test_source_adapter_requests_complete_ar5iv_document() -> None:
    calls = {}

    def parse(**kwargs):
        calls.update(kwargs)
        return {"ok": True, "data": {"paper_id": "arXiv:1", "document": {
            "blocks": [{"block_id": "b1", "type": "text", "text": "x"}],
            "integrity": {"status": "complete"},
        }}}

    bundle = load_source_bundle(
        "arXiv:1",
        refresh=False,
        recache=True,
        parse=parse,
        metadata_getter=lambda *args, **kwargs: {"ok": True, "data": {"title": "T"}},
        references_getter=lambda *args, **kwargs: {"ok": True, "data": []},
        citers_getter=lambda *args, **kwargs: {"ok": True, "data": []},
    )
    assert bundle.paper_id == "arXiv:1"
    assert calls == {"source": "ar5iv", "paper_id": "arXiv:1", "include_document": True, "refresh": False, "recache": True}


@pytest.mark.parametrize(
    ("failed_getter", "message"),
    [
        ("metadata", "Unable to load seed metadata: metadata offline"),
        ("references", "Unable to load seed references: references offline"),
    ],
)
def test_source_adapter_surfaces_required_seed_evidence_failures(
    failed_getter: str, message: str
) -> None:
    def parse(**kwargs):
        return {"ok": True, "data": {"paper_id": "arXiv:1", "document": {
            "blocks": [{"block_id": "b1", "type": "text", "text": "x"}],
            "integrity": {"status": "complete"},
        }}}

    metadata = {"ok": True, "data": {"title": "T"}}
    references = {"ok": True, "data": []}
    if failed_getter == "metadata":
        metadata = {"ok": False, "error": {"message": "metadata offline"}}
    else:
        references = {"ok": False, "error": {"message": "references offline"}}

    with pytest.raises(SourceError, match=message):
        load_source_bundle(
            "arXiv:1",
            parse=parse,
            metadata_getter=lambda *args, **kwargs: metadata,
            references_getter=lambda *args, **kwargs: references,
            citers_getter=lambda *args, **kwargs: {"ok": True, "data": []},
        )


def test_source_adapter_records_optional_citer_failure_as_warning() -> None:
    def parse(**kwargs):
        return {"ok": True, "data": {"paper_id": "arXiv:1", "document": {
            "blocks": [{"block_id": "b1", "type": "text", "text": "x"}],
            "integrity": {"status": "complete"},
        }}}

    bundle = load_source_bundle(
        "arXiv:1",
        parse=parse,
        metadata_getter=lambda *args, **kwargs: {"ok": True, "data": {"title": "T"}},
        references_getter=lambda *args, **kwargs: {"ok": True, "data": []},
        citers_getter=lambda *args, **kwargs: {
            "ok": False,
            "error": {"message": "INSPIRE citer endpoint unavailable"},
        },
    )

    assert bundle.citers == []
    assert bundle.diagnostics == ({
        "severity": "warning",
        "code": "citer_context_unavailable",
        "source": "arc-paper",
        "message": "Unable to load optional seed citers: INSPIRE citer endpoint unavailable",
    },)


def test_source_adapter_caches_bounded_related_full_text_through_parse_api() -> None:
    calls: list[dict] = []

    def parse(**kwargs):
        paper_id = kwargs["paper_id"]
        calls.append(dict(kwargs))
        data = {"paper_id": paper_id, "source_hash": "d" * 64, "sections": [{
            "section_id": f"{paper_id}-s1", "title": "Result", "text": "field theory"
        }]}
        if kwargs.get("include_document"):
            data["document"] = {
            "blocks": [{"block_id": f"{paper_id}-b1", "type": "text", "text": "field theory"}],
            "integrity": {"status": "complete"},
            }
        return {"ok": True, "data": data}

    references = [
        {"arxiv_id": f"0801.{index:04d}", "title": f"Prior {index}", "citation_count": index}
        for index in range(9)
    ]
    citers = [
        {"arxiv_id": f"2501.{index:04d}", "title": f"Later {index}", "citation_count": index}
        for index in range(9)
    ]
    bundle = load_source_bundle(
        "arXiv:1",
        parse=parse,
        metadata_getter=lambda *args, **kwargs: {"ok": True, "data": {"title": "T"}},
        references_getter=lambda *args, **kwargs: {"ok": True, "data": references},
        citers_getter=lambda *args, **kwargs: {"ok": True, "data": citers},
    )

    assert calls[0]["paper_id"] == "arXiv:1"
    assert calls[0]["include_document"] is True
    assert len(calls) == 17
    assert all(call["include_document"] is False for call in calls[1:])
    assert len(bundle.related_evidence) == 16
    assert {item["relation"] for item in bundle.related_evidence} == {"prior", "later"}
    assert all(item["evidence_level"] == "full_text" for item in bundle.related_evidence)
    assert all(item["source_descriptor"]["provider"] == "arc-paper" for item in bundle.related_evidence)
    assert all(item["source_descriptor"]["content_sha256"] for item in bundle.related_evidence)
    assert all(item["blocks"][0]["sha256"] for item in bundle.related_evidence)
    assert all(item["blocks"][0]["block_id"].endswith("-s1") for item in bundle.related_evidence)


@pytest.mark.parametrize(
    "integrity",
    [
        {"status": "partial"},
        {"complete": False},
        {"status": "complete", "blocking_issues": ["missing image"]},
    ],
)
def test_incomplete_documents_are_rejected(integrity) -> None:
    with pytest.raises(SourceError):
        validate_complete_document({"blocks": [{"block_id": "b"}], "integrity": integrity})


def test_cli_prints_default_language_notice_without_pausing(tmp_path: Path, monkeypatch, capsys) -> None:
    captured = {}

    def fake_build(options):
        captured["options"] = options
        return {"ok": True, "data": {"status": "complete"}, "errors": [], "meta": {"notice": "n"}}

    monkeypatch.setattr(cli, "build_companion", fake_build)
    code = cli.main(["build", "arXiv:1", "--project-dir", str(tmp_path), "--json"])
    streams = capsys.readouterr()
    assert code == 0
    assert "默认使用中文" in streams.err
    assert captured["options"].annotation_language == "zh-CN"
    assert captured["options"].language_was_defaulted is True
    assert captured["options"].workers == 24
    assert json.loads(streams.out)["ok"] is True


def test_cli_explicit_language_has_no_notice(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "build_companion", lambda options: {"ok": True, "data": {"status": "complete"}, "meta": {}})
    assert cli.main(["build", "arXiv:1", "--project-dir", str(tmp_path), "--annotation-language", "en"]) == 0
    assert "默认使用中文" not in capsys.readouterr().err


def test_cli_prints_structured_evidence_warnings(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "build_companion",
        lambda options: {
            "ok": True,
            "data": {"status": "complete"},
            "meta": {"diagnostics": [{
                "severity": "warning",
                "code": "citer_context_unavailable",
                "message": "Optional citer context is unavailable",
            }]},
        },
    )

    assert cli.main([
        "build",
        "arXiv:1",
        "--project-dir",
        str(tmp_path),
        "--annotation-language",
        "en",
        "--json",
    ]) == 0
    streams = capsys.readouterr()
    assert "WARNING: Optional citer context is unavailable" in streams.err
    assert json.loads(streams.out)["meta"]["diagnostics"][0]["code"] == "citer_context_unavailable"


def test_companion_docs_describe_bounded_full_text_evidence_and_package_contents() -> None:
    manual = (ROOT / "plugins/arc/skills/arc/manuals/arc-companion.md").read_text(encoding="utf-8")
    workflow = (ROOT / "plugins/arc/skills/arc/workflows/companion.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "related-paper full texts cached through existing `arc-paper` APIs" in manual
    assert "concurrent translations and 24 concurrent companion commentaries" in manual
    assert "first_round_preview.pdf" in manual
    assert "first `min(workers, unit_count)`" in manual
    assert "before any remaining unit is submitted" in manual
    assert "Table-of-contents blocks, acknowledgment sections, and\nreference-list headings" in manual
    assert "lanes drain all\n  submitted units before reporting their aggregated failures" in manual
    assert "targeted reference and citer full text cached through `arc-paper`" in readme
    assert "default is 24 concurrent translations plus 24 concurrent commentaries" in workflow
    assert "preview before submitting any remaining unit" in workflow
    assert "source-only table-of-contents blocks" in workflow
    assert "fix the scheduler in `packages/arc-companion`" in workflow


def test_package_includes_only_validated_deliverables(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    tex = tmp_path / "paper.tex"
    pdf.write_bytes(b"%PDF fixture")
    tex.write_text("tex", encoding="utf-8")
    (tmp_path / "source-manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation.json").write_text('{"ok":true}', encoding="utf-8")
    (tmp_path / "state.json").write_text(json.dumps({
        "status": "complete",
        "paper_id": "arXiv:1",
        "fingerprint": "abc",
        "output_pdf": str(pdf),
        "output_tex": str(tex),
    }), encoding="utf-8")
    result = package_project(tmp_path)
    assert result["ok"]
    assert Path(result["data"]["archive_path"]).is_file()


def test_renderer_accepts_current_arc_paper_rich_contract(tmp_path: Path) -> None:
    from arc_paper.parse.source import parse_source_input

    parsed = parse_source_input(
        html_text="""
        <article class="ltx_document">
          <h1 class="ltx_title_document">Rich title</h1>
          <div class="ltx_role_affiliation">Physics Institute</div>
          <div class="ltx_abstract">Rich abstract</div>
          <section id="S1"><h2>Result</h2>
            <table id="S1.E1" class="ltx_equation"><tr><td><math alttext="x=y"></math></td>
              <td class="ltx_eqn_eqno"><span class="ltx_tag_equation">(11)</span></td></tr></table>
            <figure id="S1.T1" class="ltx_table"><figcaption><span class="ltx_tag_table">Table 3:</span> Values</figcaption>
              <table><tr><td rowspan="2">mass</td><td>one</td></tr><tr><td>two</td></tr></table></figure>
          </section>
          <ul class="ltx_bibliography"><li id="bib.one" class="ltx_bibitem"><span class="ltx_tag_bibitem">[1]</span> A reference.</li></ul>
        </article>
        """,
        source_id="rich",
    )
    document = parsed["document"]
    blocks = document["blocks"]
    segments = [{
        "segment_id": "all",
        "title": "All",
        "start_block_id": blocks[0]["block_id"],
        "end_block_id": blocks[-1]["block_id"],
        "block_ids": [item["block_id"] for item in blocks],
    }]
    tex, _ = render_companion_tex(
        document,
        segments,
        {"all": {"commentary": "Companion"}},
        output_dir=tmp_path,
        language="en",
    )
    assert "x=y" in tex
    assert "['x=y']" not in tex
    assert r"\tag{11}" in tex
    assert "Table 3: Values" in tex
    assert r"\multirow{2}{*}{mass} & one" in tex
    assert "\n & two" in tex
    assert "Physics Institute" in tex
    assert "Rich abstract" in tex


def test_table_renderer_preserves_sparse_positions_rowspan_and_colspan() -> None:
    entity = {
        "id": "table",
        "column_count": 3,
        "rows": [
            [
                {"text": "A", "row": 0, "column": 0, "rowspan": 2, "colspan": 1},
                {"text": "wide", "row": 0, "column": 1, "rowspan": 1, "colspan": 2},
            ],
            [
                {"text": "B", "row": 1, "column": 1, "rowspan": 1, "colspan": 1},
                {"text": "C", "row": 1, "column": 2, "rowspan": 1, "colspan": 1},
            ],
        ],
        "grid": [
            [
                {"text": "A", "source_row": 0, "source_column": 0},
                {"text": "wide", "source_row": 0, "source_column": 1},
                {"text": "wide", "source_row": 0, "source_column": 1},
            ],
            [
                {"text": "A", "source_row": 0, "source_column": 0},
                {"text": "B", "source_row": 1, "source_column": 1},
                {"text": "C", "source_row": 1, "source_column": 2},
            ],
        ],
    }
    tex = _render_table(entity)
    assert r"\begin{longtable}{lll}" in tex
    assert r"\multirow{2}{*}{A} & \multicolumn{2}{l}{wide}" in tex
    assert "\n & B & C" in tex

    with pytest.raises(LatexError, match="expected exactly 2"):
        _render_table({**entity, "column_count": 2})


def test_table_renderer_can_reconstruct_spans_from_canonical_grid_only() -> None:
    entity = {
        "grid": [
            [
                {"text": "A", "source_row": 0, "source_column": 0},
                {"text": "wide", "source_row": 0, "source_column": 1},
                {"text": "wide", "source_row": 0, "source_column": 1},
            ],
            [
                {"text": "A", "source_row": 0, "source_column": 0},
                {"text": "B", "source_row": 1, "source_column": 1},
                {"text": "C", "source_row": 1, "source_column": 2},
            ],
        ]
    }
    tex = _render_table(entity)
    assert r"\multirow{2}{*}{A}" in tex
    assert r"\multicolumn{2}{l}{wide}" in tex
    assert "\n & B & C" in tex


def test_heading_and_nested_list_fields_render_without_falling_back_to_prose(tmp_path: Path) -> None:
    document = {
        "blocks": [
            {"block_id": "h", "kind": "heading", "heading_level": 2, "title": "Detailed result"},
            {
                "block_id": "l",
                "kind": "prose",
                "list_kind": "ordered",
                "list_items": [
                    {"text": "First", "items": [{"text": "Nested"}]},
                    {"content": "Second"},
                ],
            },
        ],
        "integrity": {"status": "complete"},
    }
    segments = [{
        "segment_id": "all",
        "start_block_id": "h",
        "end_block_id": "l",
        "block_ids": ["h", "l"],
    }]
    tex, _ = render_companion_tex(
        document,
        segments,
        {"all": {"commentary": "note"}},
        output_dir=tmp_path,
        language="en",
    )
    assert r"\subsection*{Detailed result}" in tex
    assert r"\addcontentsline{toc}{subsection}{Detailed result}" in tex
    assert r"\begin{enumerate}" in tex
    assert r"\item First" in tex
    assert r"\begin{itemize}" in tex
    assert r"\item Nested" in tex
    assert r"\item Second" in tex


def test_html_ordered_list_with_years_keeps_automatic_numbering() -> None:
    tex = _render_html_fragment(
        "<ol><li>2020 result</li><li>2021 follow-up</li></ol>",
        rendered_links=[],
    )

    assert r"\begin{enumerate}" in tex
    assert r"\item 2020 result" in tex
    assert r"\item 2021 follow-up" in tex
    assert r"\begin{description}" not in tex


def test_html_renderer_preserves_inline_structure_without_front_or_reference_duplication(tmp_path: Path) -> None:
    document = {
        "front_matter": {"title": "One title", "authors": ["A. Author"]},
        "blocks": [
            {
                "block_id": "title",
                "kind": "heading",
                "text": "One title",
                "title": "One title",
                "html": '<h1 id="title">One title</h1>',
                "section_id": "",
            },
            {
                "block_id": "p1",
                "source_id": "p1",
                "section_id": "S1",
                "kind": "prose",
                "text": "An important result x_i cites [1] and site.",
                "html": (
                    '<p id="p1">An <em>important</em> result '
                    '<math alttext="x_i"></math> cites <a href="#bib1">[1]</a> '
                    'and <a href="https://example.test/a_b?x=1&amp;y=2">site</a>.</p>'
                ),
            },
            {
                "block_id": "bib1",
                "source_id": "bib1",
                "kind": "bibliography",
                "text": "[1] Reference text.",
                "html": '<li id="bib1"><span class="ltx_tag_bibitem">[1]</span> Reference text.</li>',
            },
        ],
        "bibliography": [{
            "id": "bib1",
            "label": "[1]",
            "text": "[1] Reference text.",
            "html": '<li id="bib1"><span class="ltx_tag_bibitem">[1]</span> Reference text.</li>',
        }],
        "links": [
            {"href": "#bib1", "target_id": "bib1", "text": "[1]"},
            {"href": "https://example.test/a_b?x=1&y=2", "target_id": "", "text": "site"},
        ],
        "integrity": {"status": "complete"},
    }
    segments = [{
        "segment_id": "all",
        "start_block_id": "title",
        "end_block_id": "bib1",
        "block_ids": ["title", "p1", "bib1"],
    }]
    tex, manifest = render_companion_tex(
        document,
        segments,
        {"all": {"commentary": "note"}},
        output_dir=tmp_path,
        language="en",
    )
    assert tex.count("One title") == 1
    assert tex.count("Reference text") == 1
    assert r"\emph{important}" in tex
    assert r"\(x_i\)" in tex
    assert r"\hyperref[bib1]{[1]}" in tex
    assert r"\href{https://example.test/a\_b?x=1\&y=2}{site}" in tex
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_structural_combined_creator_block_renders_author_and_affiliation_once(tmp_path: Path) -> None:
    document = {
        "front_matter": {
            "title": "Structured Title",
            "authors": ["An Author"],
            "affiliations": ["An Institute"],
            "block_ids": {
                "title": ["title"],
                "authors": ["creator"],
                "affiliations": ["creator"],
            },
        },
        "blocks": [
            {
                "block_id": "title", "kind": "heading", "text": "Structured Title",
                "source_role": "front_matter_title",
            },
            {
                "block_id": "creator", "kind": "prose", "text": "An Author An Institute",
                "source_role": "front_matter",
                "front_matter_roles": ["front_matter_authors", "front_matter_affiliations"],
            },
            {"block_id": "body", "kind": "prose", "text": "Body text."},
        ],
        "integrity": {"status": "complete"},
    }
    segments = [{"segment_id": "body", "block_ids": ["body"]}]

    tex, _ = render_companion_tex(
        document,
        segments,
        {"body": {"commentary": "Note."}},
        output_dir=tmp_path,
        language="en",
    )

    assert tex.count("Structured Title") == 1
    assert tex.count("An Author") == 1
    assert tex.count("An Institute") == 1
    assert "An Author An Institute" not in tex


def test_source_only_toc_acknowledgments_and_references_render_once_with_toc_structure(tmp_path: Path) -> None:
    document = {
        "front_matter": {},
        "blocks": [
            {
                "block_id": "toc-title", "kind": "heading", "level": 6,
                "text": "Contents", "title": "Contents", "source_role": "table_of_contents",
                "html": '<h6 class="ltx_title_contents">Contents</h6>',
            },
            {
                "block_id": "toc-list", "kind": "list", "source_role": "table_of_contents",
                "text": "1 Main 1.1 Detail 2 Other", "list_kind": "ordered", "items": [],
                "html": (
                    '<ol class="ltx_toclist"><li><a href="#S1">1 Main</a>'
                    '<ol><li><a href="#S1.SS1">1.1 Detail</a></li></ol>'
                    '</li><li>2 Other</li></ol>'
                ),
            },
            {
                "block_id": "S1", "kind": "heading", "level": 2, "section_id": "S1",
                "text": "1 Main", "title": "Main", "html": '<h2 id="S1">1 Main</h2>',
            },
            {
                "block_id": "body", "source_id": "S1.SS1", "kind": "prose",
                "section_id": "S1", "text": "Body text.",
            },
            {
                "block_id": "ack-title", "kind": "heading", "section_id": "Sx",
                "text": "Acknowledgments", "title": "Acknowledgments", "source_role": "acknowledgments",
            },
            {
                "block_id": "ack-body", "kind": "prose", "section_id": "Sx",
                "text": "We thank our colleagues.", "source_role": "acknowledgments",
            },
            {
                "block_id": "refs-title", "kind": "heading", "section_id": "bib",
                "text": "References", "title": "References", "source_role": "references",
            },
            {
                "block_id": "bib1", "kind": "bibliography", "section_id": "bib",
                "text": "[1] Reference work.", "source_role": "references",
            },
        ],
        "bibliography": [{"id": "bib1", "label": "[1]", "text": "[1] Reference work."}],
        "links": [
            {"href": "#S1", "target_id": "S1", "text": "1 Main"},
            {"href": "#S1.SS1", "target_id": "S1.SS1", "text": "1.1 Detail"},
        ],
        "integrity": {"status": "complete"},
    }
    segments = [{
        "segment_id": "body", "start_block_id": "S1", "end_block_id": "body",
        "block_ids": ["S1", "body"], "title": "Main",
    }]

    tex, manifest = render_companion_tex(
        document,
        segments,
        {"body": {"commentary": "Body note", "explanation": "Body note"}},
        translations={"body": {"blocks": [
            {"block_id": "S1", "text": "主节"}, {"block_id": "body", "text": "正文。"},
        ]}},
        output_dir=tmp_path,
        language="zh-CN",
    )

    assert tex.count(r"\paragraph*{Contents}") == 1
    assert tex.count("We thank our colleagues") == 1
    assert tex.count("Reference work") == 1
    assert tex.count(r"\begin{description}") == 2
    assert r"\begin{enumerate}" not in tex
    assert r"\item[] \hyperref[S1]{1 Main}" in tex
    assert r"\item[] \hyperref[S1.SS1]{1.1 Detail}" in tex
    assert r"\item[] 2 Other" in tex
    assert manifest["rendered_links"] == manifest["expected_links"]
    assert manifest["companion_layers"]["semantic_segment_ids"] == ["body"]
    assert validate_tex_fidelity(tex, document, manifest) == []

    manifest["rendered_links"].append(dict(manifest["rendered_links"][0]))
    assert "rendered 1 unregistered source link occurrence(s)" in validate_tex_fidelity(
        tex, document, manifest,
    )


def test_multirow_equations_preserve_each_number_and_label(tmp_path: Path) -> None:
    document = {
        "blocks": [{"block_id": "eq", "kind": "equation", "equation_id": "eq"}],
        "equations": [{
            "id": "eq",
            "tex": ["a=b", "c=d"],
            "printed_equation_numbers": ["(4a)", "(4b)"],
            "labels": ["eq:4a", "eq:4b"],
        }],
        "integrity": {"status": "complete"},
    }
    tex, manifest = render_companion_tex(
        document,
        [{"segment_id": "all", "start_block_id": "eq", "end_block_id": "eq", "block_ids": ["eq"]}],
        {"all": {"commentary": "note"}},
        output_dir=tmp_path,
        language="en",
    )
    assert r"\tag{4a}" in tex and r"\tag{4b}" in tex
    assert r"\label{eq:4a}" in tex and r"\label{eq:4b}" in tex
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_preamble_has_portable_deterministic_cjk_font_fallback(tmp_path: Path) -> None:
    document = {
        "blocks": [{"block_id": "p", "kind": "prose", "text": "中文"}],
        "integrity": {"status": "complete"},
    }
    tex, _ = render_companion_tex(
        document,
        [{"segment_id": "all", "start_block_id": "p", "end_block_id": "p", "block_ids": ["p"]}],
        {"all": {"commentary": "伴读"}},
        output_dir=tmp_path,
        language="zh-CN",
    )
    candidates = ["Noto Serif CJK SC", "Source Han Serif SC", "Source Han Serif CN", "FandolSong-Regular"]
    positions = [tex.index(value) for value in candidates]
    assert positions == sorted(positions)
    assert r"\PackageError{arc-companion}{No supported CJK serif font found}" in tex
