import hashlib
import json

from arc_paper import cli, service
from arc_paper.cache import (
    parsed_source_annotations_cache_path,
    parsed_source_cache_path,
    read_json,
    rich_document_cache_path,
    write_json,
)


def _write_tex(tmp_path):
    tex_path = tmp_path / "note.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\section{Dynamics}",
                "Intro text.",
                r"\begin{equation}",
                r"\label{eq:one}",
                r"x = y",
                r"\end{equation}",
            ]
        ),
        encoding="utf-8",
    )
    return tex_path


def test_service_parse_source_caches_and_lookup_apis(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    tex_path = _write_tex(tmp_path)

    parsed = service.parse_source(tex_path=tex_path, source_id="lecture-9")

    assert parsed["ok"] is True
    source_id = parsed["data"]["paper_id"]
    cache_path = parsed_source_cache_path(source_id)
    assert cache_path.exists()
    assert read_json(cache_path)["paper_id"] == source_id
    assert parsed["meta"]["cache"] == "write"

    parsed_source = service.get_parsed_source(source_id)
    toc = service.get_parsed_source_toc(source_id)
    section = service.get_parsed_source_section(source_id, "sec_0001")
    equations = service.get_parsed_source_equations(source_id)
    equation = service.get_parsed_source_equation(source_id, "eq_00001")
    hits = service.search_parsed_source(source_id, query="eq:one")

    assert parsed_source["data"]["paper_id"] == source_id
    assert toc["data"][0]["title"] == "Dynamics"
    assert section["data"]["title"] == "Dynamics"
    assert "Intro text" in section["data"]["text"]
    assert equations["data"][0]["id"] == "eq_00001"
    assert equation["data"]["tex_label"] == "eq:one"
    assert hits["data"][0]["id"] == "eq_00001"


def test_service_parse_source_reuses_cached_parse_when_source_hash_matches(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    tex_path = _write_tex(tmp_path)
    first = service.parse_source(tex_path=tex_path, source_id="lecture-9")

    def fail_parse(*args, **kwargs):
        raise AssertionError("source parser should not run for matching cached input")

    monkeypatch.setattr(service, "parse_source_input_with_warnings", fail_parse)

    second = service.parse_source(tex_path=tex_path, source_id="lecture-9")

    assert second["ok"] is True
    assert second["meta"]["cache"] == "hit"
    assert second["data"] == first["data"]


def test_service_parse_source_ignores_cached_parse_with_wrong_source_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    tex_path = _write_tex(tmp_path)
    first = service.parse_source(tex_path=tex_path, source_id="lecture-9")
    cache_path = parsed_source_cache_path("lecture-9")
    cached = dict(first["data"])
    cached["paper_id"] = "different-source"
    write_json(cache_path, cached)

    second = service.parse_source(tex_path=tex_path, source_id="lecture-9")

    assert second["ok"] is True
    assert second["meta"]["cache"] == "write"
    assert second["data"]["paper_id"] == "lecture-9"


def test_service_get_parsed_source_missing_returns_error(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))

    result = service.get_parsed_source("missing")

    assert result["ok"] is False
    assert result["error"]["code"] == "parsed_source_not_found"


def test_strict_cached_document_read_never_fetches_rebuilds_or_upgrades(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    html_path = tmp_path / "paper.html"
    html_path.write_text(
        """
        <article class="ltx_document">
          <h1 class="ltx_title_document">Cached</h1>
          <p class="ltx_creator_author"><span class="ltx_personname">Author</span></p>
          <p>Body.</p>
        </article>
        """,
        encoding="utf-8",
    )
    parsed = service.parse_source(
        html_path=html_path, source_id="cached-rich", include_document=True,
    )
    assert parsed["ok"] is True
    light = read_json(parsed_source_cache_path("cached-rich"))
    rich_path = rich_document_cache_path(
        "cached-rich", light["source_hash"], service.RICH_PARSER_VERSION,
    )

    hit = service.get_parsed_source(
        "cached-rich", include_document=True, strict_cache_only=True,
    )
    assert hit["ok"] is True
    assert hit["data"]["document"]["front_matter"]["authors"] == ["Author"]

    rich_path.unlink()
    monkeypatch.setattr(
        service, "_parsed",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network/upgrade path must not run")
        ),
    )
    monkeypatch.setattr(
        service, "_rebuild_local_rich_document_from_stale_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("local cache rebuild must not run")
        ),
    )

    miss = service.get_parsed_source(
        "cached-rich", include_document=True, strict_cache_only=True,
    )

    assert miss["ok"] is False
    assert miss["error"]["code"] == "parsed_source_document_not_cached"
    assert not rich_path.exists()


def test_strict_cached_document_read_rejects_malformed_current_cache(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    html_path = tmp_path / "paper.html"
    html_path.write_text("<article><h1>Cached</h1><p>Body.</p></article>", encoding="utf-8")
    service.parse_source(
        html_path=html_path, source_id="malformed-rich", include_document=True,
    )
    light = read_json(parsed_source_cache_path("malformed-rich"))
    rich_path = rich_document_cache_path(
        "malformed-rich", light["source_hash"], service.RICH_PARSER_VERSION,
    )
    write_json(rich_path, {"rich_parser_version": service.RICH_PARSER_VERSION})

    result = service.get_parsed_source(
        "malformed-rich", include_document=True, strict_cache_only=True,
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "parsed_source_document_not_cached"


def test_cached_source_author_evidence_is_minimal_strict_and_read_only(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    html_path = tmp_path / "person.html"
    html_path.write_text(
        """
        <article class="ltx_document">
          <h1 class="ltx_title_document">Profile</h1>
          <p class="ltx_creator_author">
            <span class="ltx_personname">Café Author</span>
          </p>
          <p>Private body must not leave the cache API.</p>
        </article>
        """,
        encoding="utf-8",
    )
    parsed = service.parse_source(
        html_path=html_path, source_id="cached-person", include_document=True,
    )
    assert parsed["ok"] is True
    monkeypatch.setattr(
        service, "_parsed",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network or rebuild path must not run")
        ),
    )

    hit = service.get_cached_source_author_evidence("cached-person")

    assert hit["ok"] is True
    assert hit["meta"]["provider"] == "local-cache"
    assert set(hit["data"]) == {
        "schema_version", "reference_identity", "source_hash",
        "document_sha256", "authors",
    }
    assert hit["data"]["authors"][0]["source_name"] == "Café Author"
    assert hit["data"]["authors"][0]["field_sha256"] == (
        hashlib.sha256("Café Author".encode("utf-8")).hexdigest()
    )
    assert "document" not in hit["data"]
    assert "Private body" not in json.dumps(hit, ensure_ascii=False)

    light = read_json(parsed_source_cache_path("cached-person"))
    rich_path = rich_document_cache_path(
        "cached-person", light["source_hash"], service.RICH_PARSER_VERSION,
    )
    before = parsed_source_cache_path("cached-person").read_bytes()
    rich_path.unlink()
    miss = service.get_cached_source_author_evidence("cached-person")
    assert miss["ok"] is False
    assert miss["error"]["code"] == "cached_source_author_evidence_rich_invalid"
    assert parsed_source_cache_path("cached-person").read_bytes() == before
    assert not rich_path.exists()


def test_cached_source_author_evidence_rejects_malformed_light_cache(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    write_json(parsed_source_cache_path("bad-person"), {
        "paper_id": "bad-person",
        "parser_version": -1,
        "source_hash": "bad",
        "structure": {"requested_document_kind": "auto"},
        "index_entries": {},
    })

    result = service.get_cached_source_author_evidence("bad-person")

    assert result["ok"] is False
    assert result["error"]["code"] == (
        "cached_source_author_evidence_light_invalid"
    )


def test_service_mark_parsed_equation_writes_sidecar_and_overlays_current_source_hash(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    tex_path = _write_tex(tmp_path)
    source_id = service.parse_source(tex_path=tex_path, source_id="lecture-9")["data"]["paper_id"]
    parsed_path = parsed_source_cache_path(source_id)
    original_parsed = read_json(parsed_path)

    result = service.mark_parsed_equation(source_id, "eq_00001", reason="Sign differs from reference.")

    assert result["ok"] is True
    assert result["data"]["status"] == "problematic"
    assert result["data"]["target_id"] == "eq_00001"
    assert result["data"]["reason"] == "Sign differs from reference."
    annotations_path = parsed_source_annotations_cache_path(source_id)
    annotations = read_json(annotations_path)
    assert annotations["schema_version"] == "arc.parsed_source.annotations.v1"
    assert annotations["source_id"] == source_id
    assert annotations["annotations"] == [result["data"]]
    assert read_json(parsed_path) == original_parsed

    equation = service.get_parsed_source_equation(source_id, "eq_00001")
    equations = service.get_parsed_source_equations(source_id)
    hits = service.search_parsed_source(source_id, query="eq:one")

    assert equation["data"]["annotations"] == [result["data"]]
    assert equations["data"][0]["annotations"] == [result["data"]]
    assert hits["data"][0]["annotations"] == [result["data"]]

    second_tex = tmp_path / "note2.tex"
    second_tex.write_text(tex_path.read_text(encoding="utf-8") + "\nChanged prose.\n", encoding="utf-8")
    service.parse_source(tex_path=second_tex, source_id=source_id)

    reparsed = service.get_parsed_source_equation(source_id, "eq_00001")
    assert "annotations" not in reparsed["data"]


def test_service_mark_parsed_equation_validates_source_and_equation(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    tex_path = _write_tex(tmp_path)
    service.parse_source(tex_path=tex_path, source_id="lecture-9")

    missing_source = service.mark_parsed_equation("missing", "eq_00001", reason="bad")
    missing_equation = service.mark_parsed_equation("lecture-9", "eq_missing", reason="bad")

    assert missing_source["ok"] is False
    assert missing_source["error"]["code"] == "parsed_source_not_found"
    assert missing_equation["ok"] is False
    assert missing_equation["error"]["code"] == "parsed_source_equation_not_found"


def test_cli_parse_and_get_parsed_commands(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    tex_path = _write_tex(tmp_path)

    assert cli.main(["parse", "--tex", str(tex_path), "--id", "lecture-9", "--json"]) == 0
    parsed_output = json.loads(capsys.readouterr().out)
    source_id = parsed_output["data"]["paper_id"]

    assert cli.main(["get-parsed", source_id, "--json"]) == 0
    parsed_source_output = json.loads(capsys.readouterr().out)
    assert parsed_source_output["data"]["paper_id"] == source_id

    assert cli.main(["get-parsed-section", source_id, "--section", "sec_0001", "--json"]) == 0
    section_output = json.loads(capsys.readouterr().out)
    assert section_output["data"]["title"] == "Dynamics"
    assert "Intro text" in section_output["data"]["text"]

    assert cli.main(["get-parsed-equation", source_id, "--equation-id", "eq_00001", "--json"]) == 0
    equation_output = json.loads(capsys.readouterr().out)
    assert equation_output["data"]["normalized_latex"] == "x = y"

    assert cli.main(["mark-parsed-equation", source_id, "--equation-id", "eq_00001", "--reason", "Bad sign", "--json"]) == 0
    mark_output = json.loads(capsys.readouterr().out)
    assert mark_output["data"]["status"] == "problematic"
    assert mark_output["data"]["reason"] == "Bad sign"


def test_cli_parsed_search_dispatches_to_service(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.service,
        "search_parsed_source",
        lambda source_id, *, query, limit=20, case_sensitive=False: {
            "ok": True,
            "data": [{"paper_id": source_id, "query": query}],
            "errors": [],
            "meta": {"limit": limit, "case_sensitive": case_sensitive},
        },
    )

    assert cli.main(["search-parsed", "lecture-9", "--query", "Friedmann", "--limit", "3", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == [{"paper_id": "lecture-9", "query": "Friedmann"}]
    assert output["meta"]["limit"] == 3
