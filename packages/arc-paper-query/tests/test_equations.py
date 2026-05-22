from pathlib import Path

from arc_paper_query.parse.equations import find_equation_context


def test_equation_context():
    html = (Path(__file__).parent / "fixtures" / "ar5iv_sample.html").read_text()

    contexts = find_equation_context(html, "E = mc^2", window_paragraphs=1)

    assert len(contexts) == 1
    assert "Model text before equation." in contexts[0]["before"]
    assert "Model text after equation." in contexts[0]["after"]
