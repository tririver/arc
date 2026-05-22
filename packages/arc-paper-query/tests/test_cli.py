import json

from arc_paper_query import cli


def test_cli_get_title(monkeypatch, capsys):
    monkeypatch.setattr(cli.service, "get_title", lambda ids, refresh=False: {"ok": True, "data": "Title"})

    assert cli.main(["get-title", "arXiv:0911.3380", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == "Title"


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
