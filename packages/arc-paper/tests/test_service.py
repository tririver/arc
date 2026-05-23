from arc_paper import reference_inference, service
from arc_paper.cache import CachePaths, read_json, text_query_cache_path, write_json, write_text
from arc_paper.providers.base import ProviderError


def test_extract_paper_ids_service():
    result = service.extract_paper_ids("See arXiv:0911.3380 and doi:10.1234/2512.06790.")

    assert result["ok"] is True
    assert result["data"] == ["arXiv:0911.3380", "doi:10.1234/2512.06790"]


def test_paper_ids_safe_dir_name_service():
    result = service.paper_ids_safe_dir_name(["0911.3380", "astro-ph/0610514"])

    assert result["ok"] is True
    assert result["data"] == "0911.3380_x_astro-ph_0610514"


def test_llm_infer_main_references_short_circuits_explicit_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(
        reference_inference,
        "run_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    result = service.llm_infer_main_references("Compare 0911.3380 and doi:10.1234/abc.")

    assert result["ok"] is True
    assert result["data"] == ["arXiv:0911.3380", "doi:10.1234/abc"]
    assert result["meta"]["llm_used"] is False
    assert result["meta"]["cache"] == "write"


def test_llm_infer_main_references_uses_cached_query(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))

    first = service.llm_infer_main_references("Compare 0911.3380.")
    assert first["meta"]["cache"] == "write"

    monkeypatch.setattr(
        reference_inference,
        "run_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )
    second = service.llm_infer_main_references("Compare 0911.3380.")

    assert second["ok"] is True
    assert second["data"] == ["arXiv:0911.3380"]
    assert second["meta"]["cache"] == "hit"
    assert second["meta"]["llm_used"] is False


def test_llm_infer_main_references_explicit_ids_override_stale_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    cache_path = text_query_cache_path("main-references", "Compare 0911.3380.")
    write_json(
        cache_path,
        {
            "schema_version": "arc.paper.main_reference_query.v1",
            "query_text": "Compare 0911.3380.",
            "paper_ids": ["arXiv:9999.99999"],
            "meta": {"provider": "codex-cli", "llm_used": True},
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )

    result = service.llm_infer_main_references("Compare 0911.3380.")

    assert result["ok"] is True
    assert result["data"] == ["arXiv:0911.3380"]
    assert result["meta"]["cache"] == "write"
    assert result["meta"]["llm_used"] is False


def test_llm_infer_main_references_enables_web_and_verifies_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))

    def run_json(prompt, *, schema=None, provider="auto", model=None, env=None, process_chain=None):
        assert "Use live web search" in prompt
        assert env["ARC_CODEX_ALLOW_INTERNET"] == "true"
        assert env["ARC_CLAUDE_ALLOW_INTERNET"] == "true"
        return {
            "focus_scope": "one_domain",
            "candidates": [
                {
                    "domain": "CMB non-Gaussianity",
                    "paper_id": "doi:10.1088/1475-7516/2010/04/027",
                    "title": "A Test Paper",
                    "evidence_urls": ["https://inspirehep.net/literature/837197"],
                    "reasoning": "Verified by search.",
                }
            ],
            "warnings": [],
        }

    monkeypatch.setattr(reference_inference, "run_json", run_json)
    monkeypatch.setenv("ARC_LLM_PROVIDER", "codex-cli")
    monkeypatch.setattr(service, "_inspire", FakeInspire())

    result = service.llm_infer_main_references("Find the key paper on CMB trispectrum.")

    assert result["ok"] is True
    assert result["data"] == ["arXiv:0911.3380"]
    assert result["meta"]["cache"] == "write"
    assert result["meta"]["llm_used"] is True
    assert result["meta"]["provider"] == "codex-cli"
    assert result["meta"]["focus_scope"] == "one_domain"
    assert result["meta"]["verified_references"][0]["input_paper_id"].startswith("doi:")


def test_llm_infer_main_references_rejects_unverified_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(
        reference_inference,
        "run_json",
        lambda *args, **kwargs: {
            "focus_scope": "one_domain",
            "candidates": [
                {
                    "domain": "Fake domain",
                    "paper_id": "arXiv:9999.99999",
                    "title": "Fake",
                    "evidence_urls": [],
                    "reasoning": "None.",
                }
            ],
            "warnings": [],
        },
    )

    class MissingInspire(FakeInspire):
        def get_metadata(self, paper_id, *, refresh=False):
            raise ProviderError("inspire_not_found", "missing")

    monkeypatch.setenv("ARC_LLM_PROVIDER", "codex-cli")
    monkeypatch.setattr(service, "_inspire", MissingInspire())

    result = service.llm_infer_main_references("Find the key paper.")

    assert result["ok"] is False
    assert result["error"]["code"] == "reference_inference_unverified"


class FakeInspire:
    def get_metadata(self, paper_id, *, refresh=False):
        return {
            "paper_id": "arXiv:0911.3380",
            "title": "A Test Paper",
            "abstract": "A useful abstract.",
            "authors": ["Alice", "Bob"],
            "arxiv_id": "0911.3380",
            "citation_count": 5,
        }

    def get_references(self, paper_id, *, refresh=False, enrich=False):
        reference = {"paper_id": "arXiv:0801.0001", "title": "Reference"}
        if enrich:
            reference["abstract"] = "Reference abstract."
        return [reference]

    def get_citers(self, paper_id, *, refresh=False, limit=1000, sort="mostrecent"):
        return [{"paper_id": "arXiv:2210.00001", "title": f"Citer {limit} {sort}"}]

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
    assert service.get_metadata("0911.3380")["data"]["title"] == "A Test Paper"
    assert service.get_references("0911.3380")["data"][0]["title"] == "Reference"
    assert service.get_references("0911.3380", enrich=True)["data"][0]["abstract"] == "Reference abstract."
    assert service.get_citers("0911.3380", limit=5, sort="mostcited")["data"][0]["title"] == "Citer 5 mostcited"
    assert service.get_citer_count("0911.3380")["data"] == 5


def test_toc_section_and_equation_context(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    assert service.get_toc("0911.3380")["data"][0]["title"] == "1 Introduction"
    assert service.get_section("0911.3380", "model")["data"]["section_id"] == "S2"
    assert service.get_equation_context("0911.3380", "x = y")["data"][0]["after"] == "After."


def test_stale_parsed_cache_is_reparsed_from_cached_html(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    paths = CachePaths.for_paper("arXiv:0911.3380")
    write_json(paths.ar5iv_parsed, {"paper_id": "arXiv:0911.3380", "toc": [], "sections": []})
    write_text(
        paths.ar5iv_html,
        """
        <html><body>
          <p><span class="ltx_text ltx_font_bold">Discussion</span>— Main result.</p>
        </body></html>
        """,
    )

    result = service.get_section("0911.3380", "discussion")

    assert result["ok"] is True
    assert result["data"]["section_id"] == "inline-discussion"
    assert read_json(paths.ar5iv_parsed).get("parser_version")


def test_full_text_resolves_doi_to_arxiv(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))

    class AssertingAr5iv(FakeAr5iv):
        def get_html(self, paper_id, *, refresh=False):
            assert paper_id == "arXiv:0911.3380"
            return super().get_html(paper_id, refresh=refresh)

    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", AssertingAr5iv())

    result = service.get_toc("doi:10.1088/1475-7516/2010/04/027")

    assert result["ok"] is True
    assert result["data"][0]["title"] == "1 Introduction"


def test_missing_section_keeps_error_and_toc(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    result = service.get_section("0911.3380", "missing")
    assert result["ok"] is False
    assert result["error"]["code"] == "section_not_found"
    assert len(result["toc"]) == 2


def test_section_and_equation_provider_errors_are_result_envelopes(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_ar5iv", FailingAr5iv())

    section = service.get_section("0911.3380", "intro")
    equation = service.get_equation_context("0911.3380", "x = y")

    assert section["ok"] is False
    assert section["error"]["code"] == "ar5iv_not_found"
    assert equation["ok"] is False
    assert equation["error"]["code"] == "ar5iv_not_found"
