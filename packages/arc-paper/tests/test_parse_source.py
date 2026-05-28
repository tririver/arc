import re

import arc_paper.parse.source as source
from arc_paper.parse.source import parse_source_input, parse_source_input_with_warnings


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


def test_parse_tex_pdf_uses_nearest_printed_equation_number(monkeypatch, tmp_path):
    tex_path = tmp_path / "lecture9.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Inflation}",
                "于是，暴胀场背景的作用量为：",
                r"\begin{align}\label{eq:bg-action}",
                r"S_\phi = \int d^4 x ~ a^3(t) \left[ \frac{1}{2} \dot \phi_0^2 - V(\phi_0) \right]~.",
                r"\end{align}",
                "暴胀场背景的能量密度为",
                r"\begin{align}\label{eq:rho}",
                r"\rho = \frac{1}{2} \dot\phi_0^2 + V(\phi_0)~.",
                r"\end{align}",
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
            "\n".join(
                [
                    "前文引用 (9.28) 和 (9.30)。",
                    "于是，暴胀场背景的作用量为：",
                    "Sφ = d4x a3(t) 1/2 φ0^2 - V(φ0).        (9.29)",
                    "暴胀场背景的能量密度为",
                    "ρ = 1/2 φ0^2 + V(φ0).                    (9.30)",
                ]
            )
        ],
    )

    parsed = parse_source_input(tex_path=tex_path, pdf_path=pdf_path, source_id="lecture-9")
    by_label = {equation["tex_label"]: equation for equation in parsed["equations"]}

    assert by_label["eq:bg-action"]["printed_equation_number"] == "9.29"
    assert by_label["eq:rho"]["printed_equation_number"] == "9.30"


def test_parse_tex_pdf_uses_nearby_prose_to_choose_pdf_page(monkeypatch, tmp_path):
    tex_path = tmp_path / "lecture9.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Inflation}",
                r"由于拉氏量是动能减势能，不难猜出 \note{严格地，可由$T_{00}$得到} ，暴胀场背景的能量密度为",
                r"\begin{align}\label{eq:rho}",
                r"\rho = \frac{1}{2} \dot\phi_0^2 + V(\phi_0)~.",
                r"\end{align}",
                r"对作用量 \eqref{eq:bg-action} 变分，得暴胀场的运动方程：",
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
            "Unrelated page.\nρ = 1/2 φ0^2 + V(φ0).                    (1.12)",
            "\n".join(
                [
                    "由于拉氏量是动能减势能，不难猜出（严格地，可由 T00 得到），",
                    "暴胀场背景的能量密度为",
                    "ρ = 1/2 φ0^2 + V(φ0).                    (9.30)",
                    "对作用量 (9.29) 变分，得暴胀场的运动方程：",
                ]
            ),
        ],
    )

    parsed = parse_source_input(tex_path=tex_path, pdf_path=pdf_path, source_id="lecture-9")
    equation = parsed["equations"][0]

    assert equation["pdf_page"] == 2
    assert equation["printed_equation_number"] == "9.30"


def test_parse_tex_pdf_reports_warning_when_pdf_text_is_unavailable(monkeypatch, tmp_path):
    tex_path = tmp_path / "lecture9.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Dynamics}",
                r"\begin{equation}",
                r"x = y",
                r"\end{equation}",
            ]
        ),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF test")

    def missing_pdftotext(*args, **kwargs):
        raise FileNotFoundError("pdftotext")

    monkeypatch.setattr(source.subprocess, "run", missing_pdftotext)

    parsed, warnings = parse_source_input_with_warnings(tex_path=tex_path, pdf_path=pdf_path, source_id="lecture-9")

    assert set(parsed) == {"paper_id", "parser_version", "source_hash", "toc", "sections", "equations"}
    assert "pdf_page" not in parsed["equations"][0]
    assert warnings == [
        {
            "code": "pdf_not_used",
            "message": "PDF input was provided but pdftotext is not installed; PDF was not used.",
            "pdf_path": str(pdf_path),
        }
    ]


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
