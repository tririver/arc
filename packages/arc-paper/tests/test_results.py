from arc_paper.results import err, ok


def test_ok_envelope():
    result = ok({"title": "A"}, provider="inspire", cache="hit")
    assert result["ok"] is True
    assert result["data"] == {"title": "A"}
    assert result["errors"] == []
    assert result["meta"]["provider"] == "inspire"
    assert result["meta"]["cache"] == "hit"


def test_error_envelope():
    result = err("section_not_found", "Section 9 not found", toc=[{"id": "1"}])
    assert result["ok"] is False
    assert result["data"] is None
    assert result["error"]["code"] == "section_not_found"
    assert result["toc"] == [{"id": "1"}]
