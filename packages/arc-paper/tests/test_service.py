import os
import time

import arc_paper.parse.source as parse_source_module
from arc_paper import reference_inference, service
from arc_paper.cache import (
    CachePaths,
    parsed_source_annotations_cache_path,
    parsed_source_cache_path,
    read_json,
    text_query_cache_path,
    write_json,
)
from arc_paper.parse.source import PARSER_VERSION
from arc_paper.providers.base import ProviderError


def test_extract_paper_ids_service():
    result = service.extract_paper_ids("See arXiv:0911.3380 and doi:10.1234/2512.06790.")

    assert result["ok"] is True
    assert result["data"] == ["arXiv:0911.3380", "doi:10.1234/2512.06790"]


def test_paper_ids_safe_dir_name_service():
    result = service.paper_ids_safe_dir_name(["0911.3380", "astro-ph/0610514"])

    assert result["ok"] is True
    assert result["data"] == "0911.3380_x_astro-ph_0610514"


def test_paper_query_services_reject_missing_ids():
    result = service.get_title(None)  # type: ignore[arg-type]

    assert result["ok"] is False
    assert result["error"]["code"] == "paper_ids_required"


def test_cache_list_includes_paper_dirs_sources_and_filters_since(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    write_json(CachePaths.for_paper("0911.3380").inspire_metadata, {"title": "Cached Paper"})
    write_json(
        parsed_source_cache_path("lecture-9"),
        {"paper_id": "lecture-9", "parser_version": 7, "source_hash": "hash", "toc": [], "sections": [], "equations": []},
    )
    write_json(
        parsed_source_annotations_cache_path("lecture-9"),
        {"schema_version": "arc.parsed_source.annotations.v1", "source_id": "lecture-9", "annotations": []},
    )
    old_path = parsed_source_cache_path("old-note")
    write_json(
        old_path,
        {"paper_id": "old-note", "parser_version": 7, "source_hash": "old", "toc": [], "sections": [], "equations": []},
    )
    old_time = time.time() - 3 * 3600
    os.utime(old_path, (old_time, old_time))

    result = service.list_cached_papers()
    by_id = {item["paper_id"]: item for item in result["data"]["items"]}

    assert result["ok"] is True
    assert "arXiv:0911.3380" in by_id
    assert "paper_dir" in by_id["arXiv:0911.3380"]["kinds"]
    assert set(by_id["lecture-9"]["kinds"]) == {"source", "source_annotation"}

    recent = service.list_cached_papers(since="1h")
    recent_ids = {item["paper_id"] for item in recent["data"]["items"]}
    assert "lecture-9" in recent_ids
    assert "old-note" not in recent_ids

    selected = service.list_cached_papers(ids=["lecture-9"])
    assert [item["paper_id"] for item in selected["data"]["items"]] == ["lecture-9"]


def test_cache_remove_dry_run_and_delete_by_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    write_json(
        parsed_source_cache_path("lecture-9"),
        {"paper_id": "lecture-9", "parser_version": 7, "source_hash": "hash", "toc": [], "sections": [], "equations": []},
    )
    write_json(
        parsed_source_annotations_cache_path("lecture-9"),
        {"schema_version": "arc.parsed_source.annotations.v1", "source_id": "lecture-9", "annotations": []},
    )

    preview = service.remove_cached_papers(ids=["lecture-9"], dry_run=True)
    assert preview["ok"] is True
    assert preview["data"]["removed_count"] == 0
    assert parsed_source_cache_path("lecture-9").exists()

    removed = service.remove_cached_papers(ids=["lecture-9"], dry_run=False)
    assert removed["ok"] is True
    assert removed["data"]["removed_count"] == 2
    assert not parsed_source_cache_path("lecture-9").exists()
    assert not parsed_source_annotations_cache_path("lecture-9").exists()


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
    monkeypatch.setattr(service, "_inspire", FakeInspire())

    result = service.llm_infer_main_references("Find the key paper on CMB trispectrum.", provider="codex-cli")

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

    monkeypatch.setattr(service, "_inspire", MissingInspire())

    result = service.llm_infer_main_references("Find the key paper.", provider="codex-cli")

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


def test_parse_source_writes_sources_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    html_path = tmp_path / "paper.html"
    html_path.write_text(
        "<html><body><section id='S1'><h2>Intro</h2><p>Text.</p></section></body></html>",
        encoding="utf-8",
    )

    result = service.parse_source(html_path=html_path, source_id="lecture html")

    assert result["ok"] is True
    assert result["data"]["paper_id"] == "lecture html"
    assert result["meta"]["cache"] == "write"
    cache_path = tmp_path / "sources" / "lecture_html.json"
    assert cache_path.exists()
    assert read_json(cache_path)["paper_id"] == "lecture html"


def test_parse_source_ar5iv_writes_sources_cache_not_old_parsed_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())

    result = service.parse_source(paper_id="0911.3380", source="ar5iv")

    assert result["ok"] is True
    assert (tmp_path / "sources" / "0911.3380.json").exists()


def test_parse_source_rejects_explicit_source_mismatch(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    html_path = tmp_path / "paper.html"
    html_path.write_text("<html><body><p>Text.</p></body></html>", encoding="utf-8")

    result = service.parse_source(source="pdf", html_path=html_path, source_id="bad-source")

    assert result["ok"] is False
    assert result["error"]["code"] == "parse_source_invalid"


def test_parse_source_tex_pdf_requires_tex_and_pdf(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    tex_path = tmp_path / "note.tex"
    tex_path.write_text(r"\section{Only TeX}", encoding="utf-8")

    result = service.parse_source(source="tex-pdf", tex_path=tex_path, source_id="missing-pdf")

    assert result["ok"] is False
    assert result["error"]["code"] == "parse_source_invalid"


def test_parse_source_tex_rejects_pdf_companion(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    tex_path = tmp_path / "note.tex"
    tex_path.write_text(r"\section{Only TeX}", encoding="utf-8")
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF test")

    result = service.parse_source(source="tex", tex_path=tex_path, pdf_path=pdf_path, source_id="tex-only")

    assert result["ok"] is False
    assert result["error"]["code"] == "parse_source_invalid"
    assert "source=tex" in result["error"]["message"]


def test_parse_source_warns_when_given_pdf_is_not_used(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    tex_path = tmp_path / "note.tex"
    tex_path.write_text(r"\section{Only TeX}", encoding="utf-8")
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF test")

    def missing_pdftotext(*args, **kwargs):
        raise FileNotFoundError("pdftotext")

    monkeypatch.setattr(parse_source_module.subprocess, "run", missing_pdftotext)

    result = service.parse_source(tex_path=tex_path, pdf_path=pdf_path, source_id="lecture-9")

    assert result["ok"] is True
    assert result["meta"]["warnings"] == [
        {
            "code": "pdf_not_used",
            "message": "PDF input was provided but pdftotext is not installed; PDF was not used.",
            "pdf_path": str(pdf_path),
        }
    ]


def test_get_parsed_source_reads_sources_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    write_json(
        parsed_source_cache_path("lecture-9"),
        {"paper_id": "lecture-9", "parser_version": 7, "source_hash": "hash", "toc": [], "sections": [], "equations": []},
    )

    result = service.get_parsed_source("lecture-9")

    assert result["ok"] is True
    assert result["data"]["paper_id"] == "lecture-9"


def test_search_parsed_source_finds_equation(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    write_json(
        parsed_source_cache_path("lecture-9"),
        {
            "paper_id": "lecture-9",
            "parser_version": 7,
            "source_hash": "hash",
            "toc": [],
            "sections": [{"section_id": "S1", "title": "Dynamics", "level": 1, "text": "Friedmann setup"}],
            "equations": [
                {
                    "id": "eq_00001",
                    "equation": "H^2 = rho",
                    "before": "Before",
                    "after": "After",
                    "section_id": "S1",
                    "section_title": "Dynamics",
                }
            ],
        },
    )

    result = service.search_parsed_source("lecture-9", query="H^2")

    assert result["ok"] is True
    assert result["data"][0]["id"] == "eq_00001"


def test_equation_context_uses_cached_parsed_json_without_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    write_json(
        parsed_source_cache_path("arXiv:0911.3380"),
        parsed_cache(
            "arXiv:0911.3380",
            [{"section_id": "S2", "title": "2 Model", "level": 2, "text": "A section."}],
            equations=[
                {
                    "id": "E1",
                    "equation": "x = y",
                    "before": "Before.",
                    "after": "After.",
                    "section_id": "S2",
                    "section_title": "2 Model",
                }
            ],
        ),
    )

    class FailingAr5iv:
        def get_html(self, paper_id, *, refresh=False):
            raise AssertionError("cached equation context must not fetch HTML")

    monkeypatch.setattr(service, "_ar5iv", FailingAr5iv())

    result = service.get_equation_context("0911.3380", "x = y")

    assert result["ok"] is True
    assert result["data"][0]["id"] == "E1"
    assert result["data"][0]["section_id"] == "S2"


def parsed_cache(paper_id: str, sections: list[dict], equations: list[dict] | None = None):
    return {
        "paper_id": paper_id,
        "parser_version": PARSER_VERSION,
        "source_hash": "test-source",
        "toc": [{"id": section["section_id"], "title": section["title"], "level": section["level"]} for section in sections],
        "sections": sections,
        "equations": equations or [],
    }


def test_search_full_text_uses_cached_parsed_json_without_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    write_json(
        parsed_source_cache_path("arXiv:0911.3380"),
        parsed_cache(
            "arXiv:0911.3380",
            [
                {
                    "section_id": "S2",
                    "title": "2 Model",
                    "level": 2,
                    "text": "Inflationary loops are not relevant here.\n"
                    "The scalar trispectrum contains a collapsed-channel signal.\n"
                    "This following sentence should also help the reader.",
                }
            ],
        ),
    )

    class FailingAr5iv:
        def get_html(self, paper_id, *, refresh=False):
            raise AssertionError("cached parsed full-text search must not fetch")

    monkeypatch.setattr(service, "_ar5iv", FailingAr5iv())

    result = service.search_full_text("0911.3380", query="collapsed-channel")

    assert result["ok"] is True
    hit = result["data"][0]
    assert hit["paper_id"] == "arXiv:0911.3380"
    assert hit["section_id"] == "S2"
    assert hit["section_title"] == "2 Model"
    assert hit["matched_in"] == "section_text"
    assert "Inflationary loops are not relevant here." in hit["snippet"]
    assert "collapsed-channel signal" in hit["snippet"]
    assert "following sentence should also help" in hit["snippet"]
    assert hit["next_steps"] == {
        "read_section": {
            "mcp": 'get_section(paper_id="arXiv:0911.3380", section="S2")',
            "cli": "arc-paper get-section arXiv:0911.3380 --section S2 --json",
        },
        "get_metadata": {
            "mcp": 'get_metadata(paper_id="arXiv:0911.3380")',
            "cli": "arc-paper get-metadata arXiv:0911.3380 --json",
        },
    }
    assert "get_section_mcp" not in hit
    assert "get_section_cli" not in hit
    assert "get_metadata_mcp" not in hit
    assert "get_metadata_cli" not in hit
    assert "line_number" not in hit
    assert "context_before" not in hit
    assert "context_after" not in hit
    assert "cache_path" not in hit
    assert result["meta"]["provider"] == "local-cache"
    assert result["meta"]["searched_files"] == 1
    assert result["meta"]["context"] == 1


def test_search_full_text_includes_cached_title_and_abbreviated_authors(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    paths = CachePaths.for_paper("arXiv:0911.3380")
    write_json(
        parsed_source_cache_path("arXiv:0911.3380"),
        parsed_cache(
            "arXiv:0911.3380",
            [{"section_id": "S2", "title": "2 Model", "level": 2, "text": "collapsed-channel signal"}],
        ),
    )
    write_json(
        paths.inspire_metadata,
        {
            "metadata": {
                "titles": [{"title": "A Search Result Paper"}],
                "authors": [
                    {"full_name": "Alice A."},
                    {"full_name": "Bob B."},
                    {"full_name": "Carol C."},
                    {"full_name": "Diego D."},
                    {"full_name": "Eve E."},
                    {"full_name": "Frank F."},
                ],
            }
        },
    )

    result = service.search_full_text("0911.3380", query="collapsed-channel")

    assert result["ok"] is True
    hit = result["data"][0]
    assert hit["title"] == "A Search Result Paper"
    assert hit["authors"] == "Alice A. et al."


def test_search_full_text_can_search_all_cached_papers(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    write_json(
        parsed_source_cache_path("arXiv:0911.3380"),
        parsed_cache(
            "arXiv:0911.3380",
            [{"section_id": "S1", "title": "1 Intro", "level": 2, "text": "heavy scalar exchange"}],
        ),
    )
    write_json(
        parsed_source_cache_path("arXiv:astro-ph/0610514"),
        parsed_cache(
            "arXiv:astro-ph/0610514",
            [
                {
                    "section_id": "S1",
                    "title": "1 Intro",
                    "level": 2,
                    "text": "heavy scalar exchange in a second paper",
                }
            ],
        ),
    )

    result = service.search_full_text(None, query="heavy scalar", limit=1)

    assert result["ok"] is True
    assert len(result["data"]) == 1
    assert result["meta"]["truncated"] is True
    assert result["meta"]["searched_files"] == 2


def test_search_full_text_can_search_explicit_local_source_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    write_json(
        parsed_source_cache_path("lecture-9"),
        parsed_cache(
            "lecture-9",
            [{"section_id": "S1", "title": "1 Intro", "level": 2, "text": "local lecture heavy scalar"}],
        ),
    )

    class FailingInspire:
        def get_metadata(self, paper_id, *, refresh=False):
            raise AssertionError("local parsed source search must not resolve INSPIRE metadata")

    monkeypatch.setattr(service, "_inspire", FailingInspire())

    result = service.search_full_text("lecture-9", query="heavy scalar")

    assert result["ok"] is True
    assert result["data"][0]["paper_id"] == "lecture-9"


def test_search_full_text_deduplicates_nested_section_snippets(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    duplicated_text = "The squeezed limit contains a scalar exchange signal."
    write_json(
        parsed_source_cache_path("arXiv:0911.3380"),
        parsed_cache(
            "arXiv:0911.3380",
            [
                {"section_id": "S2", "title": "2 Model", "level": 2, "text": duplicated_text},
                {"section_id": "S2.SS1", "title": "2.1 Exchange", "level": 3, "text": duplicated_text},
            ],
        ),
    )

    result = service.search_full_text("0911.3380", query="scalar exchange", limit=10)

    assert result["ok"] is True
    assert len(result["data"]) == 1
    assert result["data"][0]["section_id"] == "S2.SS1"


def test_search_full_text_python_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    write_json(
        parsed_source_cache_path("arXiv:0911.3380"),
        parsed_cache(
            "arXiv:0911.3380",
            [{"section_id": "S1", "title": "1 Intro", "level": 2, "text": "Boostless contact terms."}],
        ),
    )
    monkeypatch.setattr("arc_paper.search.shutil.which", lambda name: None)

    result = service.search_full_text("0911.3380", query="boostless", case_sensitive=False)

    assert result["ok"] is True
    assert result["data"][0]["snippet"] == "Boostless contact terms."
    assert result["meta"]["search_backend"] == "python-parsed-json"


def test_stale_parsed_cache_is_reparsed_from_ar5iv_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    path = parsed_source_cache_path("arXiv:0911.3380")
    write_json(path, {"paper_id": "arXiv:0911.3380", "parser_version": 11, "toc": [], "sections": []})

    class ReparseAr5iv:
        def get_html(self, paper_id, *, refresh=False):
            return """
            <html><body>
              <p><span class="ltx_text ltx_font_bold">Discussion</span>— Main result.</p>
            </body></html>
            """

    monkeypatch.setattr(service, "_ar5iv", ReparseAr5iv())

    result = service.get_section("0911.3380", "discussion")

    assert result["ok"] is True
    assert result["data"]["section_id"] == "inline-discussion"
    assert read_json(path).get("parser_version") == PARSER_VERSION


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
