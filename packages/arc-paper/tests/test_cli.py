import json

from arc_paper import cli


def test_cli_get_title(monkeypatch, capsys):
    monkeypatch.setattr(cli.service, "get_title", lambda ids, refresh=False: {"ok": True, "data": "Title"})

    assert cli.main(["get-title", "arXiv:0911.3380", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == "Title"


def test_cli_extract_paper_ids(capsys):
    assert cli.main(["extract-paper-ids", "See", "0911.3380", "and", "doi:10.1234/2512.06790", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == ["arXiv:0911.3380", "doi:10.1234/2512.06790"]


def test_cli_safe_dir_name(capsys):
    assert cli.main(["safe-dir-name", "0911.3380", "astro-ph/0610514", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == "0911.3380_x_astro-ph_0610514"


def test_cli_llm_infer_main_references(monkeypatch, capsys):
    def infer(text, provider="auto", model=None, refresh=False):
        return {
            "ok": True,
            "data": ["arXiv:0911.3380"],
            "errors": [],
            "meta": {"text": text, "provider": provider, "model": model, "refresh": refresh},
        }

    monkeypatch.setattr(cli.service, "llm_infer_main_references", infer)

    assert (
        cli.main(
            [
                "llm-infer-main-references",
                "CMB",
                "trispectrum",
                "--provider",
                "manual",
                "--model",
                "test-model",
                "--refresh",
                "--json",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == ["arXiv:0911.3380"]
    assert output["meta"]["text"] == "CMB trispectrum"
    assert output["meta"]["provider"] == "manual"
    assert output["meta"]["model"] == "test-model"
    assert output["meta"]["refresh"] is True


def test_cli_get_references_enrich(monkeypatch, capsys):
    def get_references(ids, refresh=False, enrich=False):
        return {"ok": True, "data": {"ids": ids, "refresh": refresh, "enrich": enrich}}

    monkeypatch.setattr(cli.service, "get_references", get_references)

    assert cli.main(["get-references", "0911.3380", "--enrich", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["ids"] == "0911.3380"
    assert output["data"]["enrich"] is True


def test_cli_get_citers_limit_sort(monkeypatch, capsys):
    def get_citers(ids, refresh=False, limit=1000, sort="mostrecent"):
        return {"ok": True, "data": {"ids": ids, "limit": limit, "sort": sort}}

    monkeypatch.setattr(cli.service, "get_citers", get_citers)

    assert cli.main(["get-citers", "0911.3380", "--limit", "7", "--sort", "mostcited", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["limit"] == 7
    assert output["data"]["sort"] == "mostcited"


def test_cli_get_metadata(monkeypatch, capsys):
    monkeypatch.setattr(cli.service, "get_metadata", lambda ids, refresh=False: {"ok": True, "data": {"title": ids}})

    assert cli.main(["get-metadata", "0911.3380", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["title"] == "0911.3380"


def test_cli_get_section(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.service,
        "get_section",
        lambda ids, section, refresh=False: {"ok": True, "data": {"section_id": section}},
    )

    assert cli.main(["get-section", "arXiv:0911.3380", "--section", "S2", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["section_id"] == "S2"


def test_cli_search_full_text(monkeypatch, capsys):
    def search_full_text(ids, *, query, refresh=False, limit=20, context=0, case_sensitive=False):
        return {
            "ok": True,
            "data": [{"paper_id": ids, "snippet": query}],
            "meta": {
                "refresh": refresh,
                "limit": limit,
                "context": context,
                "case_sensitive": case_sensitive,
            },
        }

    monkeypatch.setattr(cli.service, "search_full_text", search_full_text)

    assert (
        cli.main(
            [
                "search-full-text",
                "0911.3380",
                "--query",
                "scalar trispectrum",
                "--limit",
                "5",
                "--context",
                "2",
                "--case-sensitive",
                "--refresh",
                "--json",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["data"][0]["paper_id"] == "0911.3380"
    assert output["data"][0]["snippet"] == "scalar trispectrum"
    assert output["meta"]["limit"] == 5
    assert output["meta"]["context"] == 2
    assert output["meta"]["case_sensitive"] is True
    assert output["meta"]["refresh"] is True


def test_cli_doctor_host(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "detect_host",
        lambda: type("Detected", (), {"host": "codex", "confidence": 1.0, "signals": ["test"]})(),
    )

    assert cli.main(["doctor", "host", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["host"] == "codex"


def test_cli_doctor_provider(monkeypatch, capsys):
    host = type("Detected", (), {"host": "codex", "confidence": 1.0, "signals": ["test"]})()
    monkeypatch.setattr(
        cli,
        "select_llm_provider",
        lambda: type("Selected", (), {"provider": "codex-cli", "host": host, "signals": ["test"]})(),
    )

    assert cli.main(["doctor", "provider", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["provider"] == "codex-cli"


def test_cli_doctor_cache(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.service,
        "doctor_cache",
        lambda paper_id=None: {"ok": True, "data": {"paper": {"paper_id": paper_id}}},
    )

    assert cli.main(["doctor", "cache", "0911.3380", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["paper"]["paper_id"] == "0911.3380"
