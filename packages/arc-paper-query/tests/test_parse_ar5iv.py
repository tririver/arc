from pathlib import Path

from arc_paper_query.parse.ar5iv_html import get_section, parse_html


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
