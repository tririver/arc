import json

from arc_paper_query import cli


def test_cli_get_title(monkeypatch, capsys):
    monkeypatch.setattr(cli.service, "get_title", lambda ids, refresh=False: {"ok": True, "data": "Title"})

    assert cli.main(["get-title", "arXiv:0911.3380", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == "Title"


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
