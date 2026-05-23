from pathlib import Path

from arc_paper.parse.ar5iv_html import get_section, parse_html


def sample_html():
    return (Path(__file__).parent / "fixtures" / "ar5iv_sample.html").read_text()


def test_toc_and_sections():
    parsed = parse_html(sample_html(), paper_id="arXiv:0000.0000")
    assert parsed["toc"] == [
        {"id": "S1", "title": "1 Introduction", "level": 2},
        {"id": "S2", "title": "2 Model", "level": 2},
    ]

    section = get_section(parsed, "S1")
    assert section["ok"] is True
    assert "Intro text." in section["data"]["text"]


def test_section_can_be_selected_by_title_fragment():
    parsed = parse_html(sample_html(), paper_id="arXiv:0000.0000")
    section = get_section(parsed, "model")
    assert section["ok"] is True
    assert section["data"]["section_id"] == "S2"


def test_missing_section_returns_toc():
    parsed = parse_html(sample_html(), paper_id="arXiv:0000.0000")
    missing = get_section(parsed, "S9")
    assert missing["ok"] is False
    assert missing["error"]["code"] == "section_not_found"
    assert len(missing["toc"]) == 2


def test_inline_paragraph_label_creates_section_until_next_label():
    html = """
    <html><body>
      <p><span class="ltx_text ltx_font_bold">Introduction and Summary</span>—
      Opening text.</p>
      <p>More introduction.</p>
      <p><span class="ltx_text ltx_font_bold">Discussion</span>— Main result.</p>
      <p>More discussion.</p>
      <p><span class="ltx_text ltx_font_italic">Acknowledgements</span>— Thanks.</p>
    </body></html>
    """

    parsed = parse_html(html, paper_id="arXiv:0000.0000")

    assert {"id": "inline-discussion", "title": "Discussion", "level": 2} in parsed["toc"]
    section = get_section(parsed, "discussion")
    assert section["ok"] is True
    assert section["data"]["section_id"] == "inline-discussion"
    assert "Main result." in section["data"]["text"]
    assert "More discussion." in section["data"]["text"]
    assert "Thanks." not in section["data"]["text"]


def test_inline_label_inside_multi_paragraph_block_starts_at_label_paragraph():
    html = """
    <html><body>
      <div class="ltx_para">
        <p>Previous result.</p>
        <p><span class="ltx_text ltx_font_bold">Outlook.</span>— Future work.</p>
        <p>More outlook.</p>
        <p><span class="ltx_text ltx_font_bold">Acknowledgements.</span>— Thanks.</p>
      </div>
    </body></html>
    """

    parsed = parse_html(html, paper_id="arXiv:0000.0000")
    section = get_section(parsed, "outlook")

    assert section["ok"] is True
    assert "Future work." in section["data"]["text"]
    assert "More outlook." in section["data"]["text"]
    assert "Previous result." not in section["data"]["text"]
    assert "Thanks." not in section["data"]["text"]


def test_inline_section_stops_at_embedded_end_label_after_breaks():
    html = """
    <html><body>
      <p><span class="ltx_text ltx_font_bold">Outlook.</span>— Future work.</p>
      <p>More outlook.<br/><br/>
      <span class="ltx_text ltx_font_bold">Acknowledgements.—</span> Thanks.</p>
    </body></html>
    """

    parsed = parse_html(html, paper_id="arXiv:0000.0000")
    section = get_section(parsed, "outlook")

    assert section["ok"] is True
    assert "Future work." in section["data"]["text"]
    assert "More outlook." in section["data"]["text"]
    assert "Acknowledgements" not in section["data"]["text"]
    assert "Thanks." not in section["data"]["text"]


def test_inline_section_stops_at_inline_references_label():
    html = """
    <html><body>
      <p><span class="ltx_text ltx_font_bold">Conclusion.</span>— Final remarks.</p>
      <p>Open problems remain.</p>
      <p><span class="ltx_text ltx_font_bold">References</span></p>
      <p>[1] A. Author, A paper.</p>
    </body></html>
    """

    parsed = parse_html(html, paper_id="arXiv:0000.0000")
    section = get_section(parsed, "conclusion")

    assert section["ok"] is True
    assert "Final remarks." in section["data"]["text"]
    assert "Open problems remain." in section["data"]["text"]
    assert "References" not in section["data"]["text"]
    assert "[1]" not in section["data"]["text"]


def test_inline_section_stops_at_reference_list_markers_without_references_label():
    html = """
    <html><body>
      <p><span class="ltx_text ltx_font_bold">Outlook.</span>— Future work.</p>
      <p>More outlook.</p>
      <p>[1] A. Author, A paper.</p>
      <p>[2] B. Author, Another paper.</p>
    </body></html>
    """

    parsed = parse_html(html, paper_id="arXiv:0000.0000")
    section = get_section(parsed, "outlook")

    assert section["ok"] is True
    assert "Future work." in section["data"]["text"]
    assert "More outlook." in section["data"]["text"]
    assert "[1]" not in section["data"]["text"]
    assert "[2]" not in section["data"]["text"]
