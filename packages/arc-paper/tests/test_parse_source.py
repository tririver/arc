import re

import arc_paper.parse.source as source
from arc_paper.parse.source import parse_source_input


def test_parse_local_html_returns_current_ar5iv_shape(tmp_path):
    html_path = tmp_path / "paper.html"
    html_path.write_text(
        """
        <html><body>
          <section id="S1"><h2>1 Intro</h2><p>Before.</p>
            <table class="ltx_equation" id="E1"><tr><td>E = mc^2</td></tr></table>
            <p>After.</p>
          </section>
        </body></html>
        """,
        encoding="utf-8",
    )

    parsed = parse_source_input(html_path=html_path, source_id="local-html")

    assert set(parsed) == {"paper_id", "parser_version", "source_hash", "toc", "sections", "equations"}
    assert parsed["paper_id"] == "local-html"
    assert parsed["toc"] == [{"id": "S1", "title": "1 Intro", "level": 2}]
    assert parsed["sections"][0]["section_id"] == "S1"
    assert parsed["sections"][0]["title"] == "1 Intro"
    assert parsed["equations"][0]["id"] == "E1"
    assert parsed["equations"][0]["equation"] == "E = mc^2"
    assert parsed["equations"][0]["before"] == "Before."
    assert parsed["equations"][0]["after"] == "After."


def test_parse_tex_pdf_returns_same_shape_with_optional_details(monkeypatch, tmp_path):
    tex_path = tmp_path / "lecture9.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Dynamics}",
                r"\label{sec:dynamics}",
                "Before text introduces Friedmann.",
                r"\begin{equation}",
                r"\label{eq:friedmann}",
                r"H^2 = \frac{8\pi G}{3}\rho",
                r"\end{equation}",
                "After text discusses expansion.",
            ]
        ),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(
        source,
        "extract_pdf_pages",
        lambda path: [
            "Preface.",
            "Before text introduces Friedmann.\nH^2 = 8 pi G / 3 rho (9.12)\nAfter text discusses expansion.",
        ],
    )

    parsed = parse_source_input(tex_path=tex_path, pdf_path=pdf_path, source_id="lecture-9")

    assert set(parsed) == {"paper_id", "parser_version", "source_hash", "toc", "sections", "equations"}
    assert parsed["paper_id"] == "lecture-9"
    assert parsed["toc"] == [{"id": "sec:dynamics", "title": "Dynamics", "level": 1}]
    assert parsed["sections"][0]["section_id"] == "sec:dynamics"
    assert parsed["sections"][0]["title"] == "Dynamics"
    assert parsed["sections"][0]["pdf_page_start"] == 2
    equation = parsed["equations"][0]
    assert equation["id"] == "eq_00001"
    assert equation["equation"] == r"H^2 = \frac{8\pi G}{3}\rho"
    assert equation["before"] == "Before text introduces Friedmann."
    assert equation["after"] == "After text discusses expansion."
    assert equation["section_id"] == "sec:dynamics"
    assert equation["tex_label"] == "eq:friedmann"
    assert equation["printed_equation_number"] == "9.12"
    assert equation["pdf_page"] == 2


def test_parse_pdf_only_returns_best_effort_shape(monkeypatch, tmp_path):
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(
        source,
        "extract_pdf_pages",
        lambda path: [
            "1 Dynamics\nBefore text.\nH^2 = 8 pi G / 3 rho (9.12)\nAfter text.",
            "2 Appendix\nNo checked equation here.",
        ],
    )

    parsed = parse_source_input(pdf_path=pdf_path, source_id="lecture-pdf")

    assert set(parsed) == {"paper_id", "parser_version", "source_hash", "toc", "sections", "equations"}
    assert parsed["paper_id"] == "lecture-pdf"
    assert parsed["toc"][0]["title"] == "1 Dynamics"
    assert parsed["sections"][0]["pdf_page_start"] == 1
    equation = parsed["equations"][0]
    assert equation["id"] == "eq_00001"
    assert equation["equation"] == "H^2 = 8 pi G / 3 rho"
    assert equation["printed_equation_number"] == "9.12"
    assert equation["pdf_page"] == 1
    assert equation["confidence"] in {"low", "medium"}
    assert "tex_label" not in equation or equation["tex_label"] == ""


def test_parse_without_id_generates_arc_id(tmp_path):
    html_path = tmp_path / "paper.html"
    html_path.write_text("<html><body><p>Text.</p></body></html>", encoding="utf-8")

    parsed = parse_source_input(html_path=html_path)

    assert re.fullmatch(r"arc-\d{8}", parsed["paper_id"])
