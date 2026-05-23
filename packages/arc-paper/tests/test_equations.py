from pathlib import Path

from arc_paper.parse.equations import extract_equation_contexts, find_equation_context


def test_equation_context():
    html = (Path(__file__).parent / "fixtures" / "ar5iv_sample.html").read_text()

    equations = extract_equation_contexts(html, window_paragraphs=1)
    contexts = find_equation_context(equations, "E = mc^2")

    assert len(contexts) == 1
    assert "Model text before equation." in contexts[0]["before"]
    assert "Model text after equation." in contexts[0]["after"]
    assert contexts[0]["section_id"] == "S2"
