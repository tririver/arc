from arc_paper_query import service
from arc_paper_query.providers.base import ProviderError


class FakeInspire:
    def get_metadata(self, paper_id, *, refresh=False):
        return {
            "paper_id": "arXiv:0911.3380",
            "title": "A Test Paper",
            "abstract": "A useful abstract.",
            "authors": ["Alice", "Bob"],
            "citation_count": 5,
        }

    def get_references(self, paper_id, *, refresh=False):
        return [{"paper_id": "arXiv:0801.0001", "title": "Reference"}]

    def get_citers(self, paper_id, *, refresh=False):
        return [{"paper_id": "arXiv:2210.00001", "title": "Citer"}]

    def get_citer_count(self, paper_id, *, refresh=False):
        return 5


class FakeAr5iv:
    def get_html(self, paper_id, *, refresh=False):
        return """
        <html><body>
          <section id="S1"><h2>1 Introduction</h2><p>Intro.</p></section>
          <section id="S2"><h2>2 Model</h2><p>Before.</p>
            <table class="ltx_equation" id="E1"><tr><td>x = y</td></tr></table>
            <p>After.</p>
          </section>
        </body></html>
        """


class FailingAr5iv:
    def get_html(self, paper_id, *, refresh=False):
        raise ProviderError("ar5iv_not_found", "missing")


def test_get_title_single(monkeypatch):
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    result = service.get_title("0911.3380")
    assert result["ok"] is True
    assert result["data"] == "A Test Paper"


def test_list_input_returns_id_result_dict(monkeypatch):
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    result = service.get_authors(["0911.3380", "0911.3380"])
    assert list(result) == ["arXiv:0911.3380"]
    assert result["arXiv:0911.3380"]["data"] == ["Alice", "Bob"]


def test_references_citers_and_counts(monkeypatch):
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    assert service.get_references("0911.3380")["data"][0]["title"] == "Reference"
    assert service.get_citers("0911.3380")["data"][0]["title"] == "Citer"
    assert service.get_citer_count("0911.3380")["data"] == 5


def test_toc_section_and_equation_context(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    assert service.get_toc("0911.3380")["data"][0]["title"] == "1 Introduction"
    assert service.get_section("0911.3380", "model")["data"]["section_id"] == "S2"
    assert service.get_equation_context("0911.3380", "x = y")["data"][0]["after"] == "After."


def test_missing_section_keeps_error_and_toc(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    result = service.get_section("0911.3380", "missing")
    assert result["ok"] is False
    assert result["error"]["code"] == "section_not_found"
    assert len(result["toc"]) == 2


def test_section_and_equation_provider_errors_are_result_envelopes(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_ar5iv", FailingAr5iv())

    section = service.get_section("0911.3380", "intro")
    equation = service.get_equation_context("0911.3380", "x = y")

    assert section["ok"] is False
    assert section["error"]["code"] == "ar5iv_not_found"
    assert equation["ok"] is False
    assert equation["error"]["code"] == "ar5iv_not_found"
