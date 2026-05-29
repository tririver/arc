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


def test_find_equation_context_searches_labels_numbers_and_context():
    equations = [
        {
            "id": "eq_00001",
            "equation": r"H^2 = \frac{8\pi G}{3}\rho",
            "tex_label": "eq:friedmann",
            "printed_equation_number": "9.15",
            "before": "The Friedmann equation follows from the Hamiltonian constraint.",
            "after": "This controls expansion.",
            "section_title": "Cosmology",
        }
    ]

    assert find_equation_context(equations, "eq:friedmann")
    assert find_equation_context(equations, "9.15")
    assert find_equation_context(equations, "Hamiltonian constraint")
    assert find_equation_context(equations, "Cosmology")
