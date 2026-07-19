import re

import arc_paper.parse.source as source
import pytest
from arc_paper.parse.document import DOCUMENT_SCHEMA_VERSION
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

    assert {"paper_id", "parser_version", "source_hash", "toc", "sections", "equations"} <= set(parsed)
    assert parsed["document"]["schema_version"] == DOCUMENT_SCHEMA_VERSION
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


def test_parse_tex_ignores_comment_environment_sections_and_equations(tmp_path):
    tex_path = tmp_path / "notes.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Active}",
                "Visible text.",
                r"\begin{comment}",
                r"\section{Hidden}",
                r"\begin{equation}",
                r"x = y",
                r"\end{equation}",
                r"\end{comment}",
                "Still visible.",
                r"\begin{equation}",
                r"z = w",
                r"\end{equation}",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_source_input(tex_path=tex_path, source_id="notes")

    assert [item["title"] for item in parsed["toc"]] == ["Active"]
    assert len(parsed["equations"]) == 1
    equation = parsed["equations"][0]
    assert equation["equation"] == "z = w"
    assert equation["before"] == "Still visible."
    assert equation["tex_line_start"] == 10


def test_parse_tex_ignores_percent_commented_sections_and_equations(tmp_path):
    tex_path = tmp_path / "notes.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Active}",
                "Visible text.",
                r"% \section{Hidden}",
                r"% \begin{equation}",
                r"% x = y",
                r"% \end{equation}",
                r"\begin{equation}",
                r"z = w",
                r"\end{equation}",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_source_input(tex_path=tex_path, source_id="notes")

    assert [item["title"] for item in parsed["toc"]] == ["Active"]
    assert len(parsed["equations"]) == 1
    equation = parsed["equations"][0]
    assert equation["equation"] == "z = w"
    assert equation["tex_line_start"] == 7


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


def test_parse_tex_pdf_brackets_number_between_before_and_after_text(monkeypatch, tmp_path):
    tex_path = tmp_path / "lecture8.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Waves}",
                r"考虑双星系统，其中每颗星质量都是$M$，运动速度都远小于光速，轨道半径为$R\ll r$。两颗星$A$, $B$的坐标可以写成",
                r"\begin{align}",
                r"\mathbf{y}_A & = (R \cos(\Omega t), R \sin(\Omega t), 0)~,",
                r"\nonumber\\",
                r"\mathbf{y}_B & = (-R \cos(\Omega t), -R \sin(\Omega t), 0)~.",
                r"\end{align}",
                r"所以，系统的能量密度为 \note{其中 $y_1, y_2, y_3$为$\mathbf{y}$坐标的三个分量}",
                r"\begin{align}",
                r"T_{00}(t,\mathbf{y}) = M \delta(y_3) [\delta(y_1-R\cos(\Omega t))\delta(y_2-R\sin(\Omega t))]~.",
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
                    "考虑双星系统，其中每颗星质量都是 M，运动速度都远小于光速，轨道",
                    "半径为 R < r . 两颗星 A, B 的坐标可以写成",
                    "yA=(Rcos(Ωt),Rsin(Ωt),0),",
                    "yB=(-Rcos(Ωt),-Rsin(Ωt),0).             (8.34)",
                    "所以，系统的能量密度为（其中 y1, y2, y3 为 y 坐标的三个分量）",
                    "T00(t, y) = Mδ(y3)[δ(y1 − R cos(Ωt))δ(y2 − R sin(Ωt))] . (8.35)",
                ]
            )
        ],
    )

    parsed = parse_source_input(tex_path=tex_path, pdf_path=pdf_path, source_id="lecture-8")

    assert parsed["equations"][0]["printed_equation_number"] == "8.34"
    assert parsed["equations"][1]["printed_equation_number"] == "8.35"


def test_parse_tex_pdf_ignores_inline_references_when_selecting_printed_number(monkeypatch, tmp_path):
    tex_path = tmp_path / "lecture9.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Cosmology}",
                r"为了得到$\ddot a$满足的方程，我们对弗里德曼方程 \eqref{eq:friedmann} 求时间导数：",
                r"\begin{align}",
                r"& 2H \dot H = \frac{8\pi G}{3} \dot\rho",
                r"\teq[$\dot\rho = -3H(\rho+p)$]{使用\eqref{eq:continuity}}",
                r"- 8\pi G H (\rho+p)~,",
                r"\\ \label{eq:friedmann-dotH}",
                r"& \mathrm{即} ~ \dot H = \frac{\ddot a}{a} - H^2 = -4\pi G (\rho + p)~.",
                r"\end{align}",
                r"再次使用弗里德曼方程消去$H^2$，得",
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
                    "为了得到 a 满足的方程，我们对弗里德曼方程 (9.15) 求时间导数：",
                    "使用 (9.8)",
                    "2H H = 8πG/3 ρ - 8πGH(ρ + p),                  (9.18)",
                    "即 H = a/a - H2 = -4πG(ρ + p).                  (9.19)",
                    "再次使用弗里德曼方程消去 H2，得",
                ]
            )
        ],
    )

    parsed = parse_source_input(tex_path=tex_path, pdf_path=pdf_path, source_id="lecture-9")

    assert parsed["equations"][0]["printed_equation_number"] == "9.19"
    assert parsed["equations"][0]["printed_equation_numbers"] == ["9.18", "9.19"]


def test_parse_tex_pdf_uses_isolated_printed_number_before_page_break(monkeypatch, tmp_path):
    tex_path = tmp_path / "lecture9.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Cosmology}",
                "后续内容中，我们只感兴趣平坦宇宙。其度规为",
                r"\begin{align}\label{eq:frw-flat}",
                r"ds^2 = -dt^2 + a^2(t) \left( dr^2 + r^2 d\Omega^2 \right)~.",
                r"\end{align}",
                "由均匀、各向同性，宇宙中物质的能动张量可以写成理想流体的形式：",
                r"\begin{align}",
                r"T^{\mu\nu} = (\rho + p) u^\mu u^\nu + g^{\mu\nu} p~.",
                r"\end{align}",
                r"其协变守恒 $\nabla_\mu T^{\mu \nu}=0$ 的$\nu=0$分量为：",
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
                    "后续内容中，我们只感兴趣平坦宇宙。其度规为",
                    "ds2 = -dt2 + a2(t)(dr2 + r2 dΩ2).",
                    "                                      (9.6)",
                    "这里，尺度因子 a(t) 的值本身没有物理意义。",
                    "由均匀、各向同性，宇宙中物质的能动张量可以写成理想流体的形式：",
                    "Tµν = (ρ + p)uµuν + gµνp",
                    "                                      (9.7)",
                ]
            ),
            r"其协变守恒 ∇µT µν = 0 的 ν=0 分量为：",
        ],
    )

    parsed = parse_source_input(tex_path=tex_path, pdf_path=pdf_path, source_id="lecture-9")

    assert parsed["equations"][0]["printed_equation_number"] == "9.6"
    assert parsed["equations"][1]["printed_equation_number"] == "9.7"


def test_parse_tex_pdf_records_all_printed_numbers_for_multirow_align(monkeypatch, tmp_path):
    tex_path = tmp_path / "lecture9.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Inflation}",
                r"回忆宇宙中只有宇宙常数的宇宙，其膨胀率为指数膨胀。",
                r"\begin{align}\label{eq:first-sr-param}",
                r"& \mathrm{第一慢滚参数：}",
                r"&&\epsilon \equiv - \frac{\dot H}{H^2}~, \\",
                r"& \mathrm{宇宙加速膨胀的条件：}",
                r"&&\ddot a >0",
                r"&&\Rightarrow",
                r"&&\epsilon <1~, \\",
                r"& \mathrm{接近指数膨胀的条件：}",
                r"&&H\simeq \mathrm{常数}",
                r"&&\Rightarrow",
                r"&&\epsilon \ll 1~.",
                r"\end{align}",
                r"为了解决视界等问题，我们需要暴胀持续足够长时间。",
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
                    "回忆宇宙中只有宇宙常数的宇宙，其膨胀率为指数膨胀。",
                    "第一慢滚参数： epsilon = -H/H2,                    (9.36)",
                    "宇宙加速膨胀的条件： a>0 => epsilon < 1,           (9.37)",
                    "接近指数膨胀的条件： H 常数 => epsilon",
                    "<< 1.                                                (9.38)",
                    "为了解决视界等问题，我们需要暴胀持续足够长时间。",
                ]
            )
        ],
    )

    parsed = parse_source_input(tex_path=tex_path, pdf_path=pdf_path, source_id="lecture-9")

    assert parsed["equations"][0]["printed_equation_number"] == "9.36"
    assert parsed["equations"][0]["printed_equation_numbers"] == ["9.36", "9.37", "9.38"]


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


def test_parse_markdown_records_sections_equations_and_line_anchors(tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text(
        "# Notes\n\nBefore the equation.\n$$\nE = m c^2\n$$\nAfter the equation.\n\n## Details\nText.\n",
        encoding="utf-8",
    )

    parsed = parse_source_input(source_path=markdown_path, source_id="notes")

    assert parsed["parser_version"] == 18
    assert parsed["toc"] == [
        {"id": "sec_0001", "title": "Notes", "level": 1},
        {"id": "sec_0002", "title": "Details", "level": 2},
    ]
    assert parsed["sections"][0]["markdown_line_start"] == 1
    assert parsed["sections"][0]["markdown_line_end"] == 8
    assert parsed["equations"][0]["equation"] == "E = m c^2"
    assert parsed["equations"][0]["markdown_line_start"] == 4
    assert parsed["equations"][0]["markdown_line_end"] == 6
    assert parsed["equations"][0]["before"] == "Before the equation."
    assert parsed["equations"][0]["after"] == "After the equation."


def test_parse_markdown_preserves_escaped_tilde_while_cleaning_strikethrough(tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text(
        "# Notes\n\nA \\~x label and ~~obsolete~~ text.\n",
        encoding="utf-8",
    )

    parsed = parse_source_input(source_path=markdown_path, source_id="notes")

    assert parsed["sections"][0]["text"] == "# Notes A ~x label and obsolete text."


def test_parse_markdown_pdf_uses_combined_hash_and_pdf_mapping(monkeypatch, tmp_path):
    markdown_path = tmp_path / "notes.markdown"
    markdown_path.write_text("# Dynamics\nBefore text.\n$$ H^2 = 8 pi G rho / 3 $$\nAfter text.\n", encoding="utf-8")
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(
        "arc_paper.parse.source.extract_pdf_pages",
        lambda path: ["Dynamics\nBefore text.\nH^2 = 8 pi G rho / 3 (1.2)\nAfter text."],
    )

    parsed = parse_source_input(source_path=markdown_path, pdf_path=pdf_path, source_id="notes")

    assert parsed["source_hash"] != parse_source_input(source_path=markdown_path, source_id="notes")["source_hash"]
    assert parsed["equations"][0]["pdf_page"] == 1
    assert parsed["equations"][0]["printed_equation_number"] == "1.2"
    assert parsed["sections"][0]["pdf_page_start"] == 1
    assert parsed["sections"][0]["pdf_page_end"] == 1


def test_source_tags_with_spacing_are_authoritative_for_markdown_and_tex(monkeypatch, tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text("# Notes\nBefore.\n$$ x = y \\tag {2.20} $$\nAfter.\n", encoding="utf-8")
    tex_path = tmp_path / "notes.tex"
    tex_path.write_text(
        "\\section{Notes}\nBefore.\n\\begin{equation}\nx = y \\tag {2.20}\n\\end{equation}\nAfter.\n",
        encoding="utf-8",
    )
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(source, "extract_pdf_pages", lambda path: ["Before.\nx = y (9.9)\nAfter."])

    for parsed in (
        parse_source_input(markdown_path=markdown_path, pdf_path=pdf_path, source_id="markdown"),
        parse_source_input(tex_path=tex_path, pdf_path=pdf_path, source_id="tex"),
    ):
        equation = parsed["equations"][0]
        assert equation["printed_equation_number"] == "2.20"
        assert equation["printed_equation_numbers"] == ["2.20"]


def test_source_number_directive_is_authoritative_and_removed_from_normalized_latex(
    monkeypatch, tmp_path
):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text(
        "# Notes\nBefore.\n$$\nx = y\n% arc:equation-number 2.20, 2.21\n$$\nAfter.\n",
        encoding="utf-8",
    )
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(source, "extract_pdf_pages", lambda path: ["Before.\nx = y (9.9)\nAfter."])

    equation = parse_source_input(markdown_path=markdown_path, pdf_path=pdf_path, source_id="notes")[
        "equations"
    ][0]

    assert equation["printed_equation_number"] == "2.20"
    assert equation["printed_equation_numbers"] == ["2.20", "2.21"]
    assert equation["equation"] == "x = y"
    assert "arc:" not in equation["normalized_latex"]


def test_unnumbered_directive_prevents_pdf_inference_and_wins_conflicts(monkeypatch, tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text(
        "# Notes\nBefore.\n$$\nx = y \\tag{2.20}\n% arc:equation-number 2.21\n% arc:unnumbered\n$$\nAfter.\n",
        encoding="utf-8",
    )
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(source, "extract_pdf_pages", lambda path: ["Before.\nx = y (9.9)\nAfter."])

    equation = parse_source_input(markdown_path=markdown_path, pdf_path=pdf_path, source_id="notes")[
        "equations"
    ][0]

    assert "printed_equation_number" not in equation
    assert "printed_equation_numbers" not in equation
    assert equation["equation"] == "x = y"
    assert equation["normalized_latex"] == "x = y"
    assert r"\tag" not in equation["equation"]
    assert "arc:" not in equation["normalized_latex"]
    assert {warning["code"] for warning in equation["parser_warnings"]} == {
        "conflicting_equation_number_directives"
    }


def test_markdown_ocr_trailing_number_is_normalized_and_removed_from_formula(tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text("# Notes\n$$\nx = y (2. 1 8)\n$$\n", encoding="utf-8")

    equation = parse_source_input(markdown_path=markdown_path, source_id="notes")["equations"][0]

    assert equation["equation"] == "x = y"
    assert equation["normalized_latex"] == "x = y"
    assert equation["printed_equation_number"] == "2.18"
    assert equation["printed_equation_numbers"] == ["2.18"]


def test_markdown_compact_parenthesized_value_is_not_treated_as_an_ocr_number(tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text("# Notes\n$$\nf(x) = (2.18)\n$$\n", encoding="utf-8")

    equation = parse_source_input(markdown_path=markdown_path, source_id="notes")["equations"][0]

    assert equation["equation"] == "f(x) = (2.18)"
    assert equation["normalized_latex"] == "f(x) = (2.18)"
    assert "printed_equation_number" not in equation
    assert "printed_equation_numbers" not in equation


def test_markdown_standalone_trailing_number_has_layout_evidence(tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text("# Notes\n$$\nx = y\n(2.18)\n$$\n", encoding="utf-8")

    equation = parse_source_input(markdown_path=markdown_path, source_id="notes")["equations"][0]

    assert equation["equation"] == "x = y"
    assert equation["printed_equation_number"] == "2.18"


def test_pdf_mapping_uses_source_number_anchors_as_monotonic_bounds(monkeypatch, tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Notes",
                "First context.",
                "$$ a = b \\tag{1.1} $$",
                "Middle context.",
                "$$ x = y $$",
                "Middle follows.",
                "Last context.",
                "$$ c = d \\tag{1.3} $$",
            ]
        ),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(
        source,
        "extract_pdf_pages",
        lambda path: [
            "First context.\na = b (1.1)",
            "Middle context.\nx = y (1.2)\nMiddle follows.",
            "Last context.\nc = d (1.3)",
            "Middle context.\nx = y\nMiddle follows.\nextra matching text",
        ],
    )

    equations = parse_source_input(markdown_path=markdown_path, pdf_path=pdf_path, source_id="notes")[
        "equations"
    ]

    assert [equation.get("pdf_page") for equation in equations] == [1, 2, 3]
    assert equations[1]["printed_equation_number"] == "1.2"


def test_source_number_does_not_anchor_to_wrong_page_without_formula_or_context(
    monkeypatch, tmp_path
):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text(
        "# Notes\nTarget context before.\n$$ H = p^2 + m^2 \\tag{1.1} $$\nTarget context after.\n",
        encoding="utf-8",
    )
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(
        source,
        "extract_pdf_pages",
        lambda path: [
            "Unrelated discussion.\nz = w (1.1)\nMore unrelated discussion.",
            "Target context before.\nH = p^2 + m^2\nTarget context after.",
        ],
    )

    equation = parse_source_input(
        markdown_path=markdown_path, pdf_path=pdf_path, source_id="notes"
    )["equations"][0]

    assert equation["pdf_page"] == 2
    assert equation["printed_equation_number"] == "1.1"


def test_pdf_number_binding_does_not_read_the_next_page(monkeypatch, tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text("# Notes\nBefore exact.\n$$ x = y $$\nAfter exact.\n", encoding="utf-8")
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(
        source,
        "extract_pdf_pages",
        lambda path: ["Before exact.\nx = y\nAfter exact.", "x = y (7.7)"],
    )

    equation = parse_source_input(markdown_path=markdown_path, pdf_path=pdf_path, source_id="notes")[
        "equations"
    ][0]

    assert equation["pdf_page"] == 1
    assert "printed_equation_number" not in equation


def test_pdf_number_binding_rejects_numbered_formula_above_unnumbered_formula(monkeypatch, tmp_path):
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text(
        "# Notes\nThe target expression starts here.\n$$ H = p^2 + m^2 $$\nThe target discussion continues here.\n",
        encoding="utf-8",
    )
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    monkeypatch.setattr(
        source,
        "extract_pdf_pages",
        lambda path: [
            "Earlier numbered result.\n"
            "H = p^2 + m^2 (5.8)\n"
            "The target expression starts here.\n"
            "H = p^2 + m^2\n"
            "The target discussion continues here."
        ],
    )

    equation = parse_source_input(markdown_path=markdown_path, pdf_path=pdf_path, source_id="notes")[
        "equations"
    ][0]

    assert equation["pdf_page"] == 1
    assert "printed_equation_number" not in equation


def test_pdf_text_extraction_preserves_internal_blank_pages(monkeypatch, tmp_path):
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"%PDF test")
    completed = source.subprocess.CompletedProcess(
        args=["pdftotext"], returncode=0, stdout="first\f   \fthird\f", stderr=""
    )
    monkeypatch.setattr(source.subprocess, "run", lambda *args, **kwargs: completed)

    assert source.extract_pdf_pages(pdf_path) == ["first", "", "third"]
