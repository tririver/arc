import json

from arc_paper import cli, service
from arc_paper.cache import parsed_source_cache_path, read_json


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
    equations = service.get_parsed_source_equations(source_id)
    equation = service.get_parsed_source_equation(source_id, "eq_00001")
    hits = service.search_parsed_source(source_id, query="eq:one")

    assert parsed_source["data"]["paper_id"] == source_id
    assert toc["data"][0]["title"] == "Dynamics"
    assert equations["data"][0]["id"] == "eq_00001"
    assert equation["data"]["tex_label"] == "eq:one"
    assert hits["data"][0]["id"] == "eq_00001"


def test_service_get_parsed_source_missing_returns_error(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))

    result = service.get_parsed_source("missing")

    assert result["ok"] is False
    assert result["error"]["code"] == "parsed_source_not_found"


def test_cli_parse_and_get_parsed_commands(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    tex_path = _write_tex(tmp_path)

    assert cli.main(["parse", "--tex", str(tex_path), "--id", "lecture-9", "--json"]) == 0
    parsed_output = json.loads(capsys.readouterr().out)
    source_id = parsed_output["data"]["paper_id"]

    assert cli.main(["get-parsed", source_id, "--json"]) == 0
    parsed_source_output = json.loads(capsys.readouterr().out)
    assert parsed_source_output["data"]["paper_id"] == source_id

    assert cli.main(["get-parsed-equation", source_id, "--equation-id", "eq_00001", "--json"]) == 0
    equation_output = json.loads(capsys.readouterr().out)
    assert equation_output["data"]["normalized_latex"] == "x = y"


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
