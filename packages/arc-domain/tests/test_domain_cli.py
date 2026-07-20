import json

from arc_domain import cli


def test_cli_returns_nonzero_for_error_envelope(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda _args: {"ok": False, "data": None, "error": {"code": "failed", "message": "boom"}},
    )

    assert cli.main(["status", "0911.3380", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_treats_needs_llm_as_successful_handoff(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda _args: {"ok": False, "status": "needs_llm", "llm_task": {"prompt": "..."}},
    )

    assert cli.main(["status", "0911.3380", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "needs_llm"


def test_cli_json_wraps_dispatch_exception(monkeypatch, capsys) -> None:
    def fail(_args):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(cli, "_dispatch", fail)

    assert cli.main(["status", "0911.3380", "--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"] == {
        "code": "command_failed",
        "message": "service unavailable",
        "type": "RuntimeError",
    }


def test_cli_init(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))

    assert cli.main(["init", "0911.3380", "--intent", "test", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["data"]["config"]["seed_paper"] == "arXiv:0911.3380"


def test_cli_dispatches_incremental_commands(monkeypatch, capsys):
    calls = []

    def record(name):
        def handler(seed_paper=None, **kwargs):
            calls.append((name, seed_paper, kwargs))
            return {"ok": True, "data": {"command": name}}

        return handler

    monkeypatch.setattr(cli.service, "identify_foundation", record("identify-foundation"))
    monkeypatch.setattr(cli.service, "build_network", record("build-network"))
    monkeypatch.setattr(cli.service, "build_paper_json_pack", record("build-paper-json-pack"))
    monkeypatch.setattr(cli.service, "build_evidence_pack", record("build-evidence"))
    monkeypatch.setattr(cli.service, "summarize_domain", record("summarize"))
    monkeypatch.setattr(cli.service, "build_domain", record("build"))
    monkeypatch.setattr(cli.service, "status", record("status"))
    monkeypatch.setattr(cli.service, "get_domain_summary", record("get-summary"))
    monkeypatch.setattr(cli.service, "get_domain_graph", record("get-graph"))

    commands = [
        ["identify-foundation", "0911.3380", "--intent", "intent", "--provider", "manual", "--workers", "1"],
        ["build-network", "0911.3380", "--intent", "intent", "--provider", "manual", "--workers", "1"],
        ["build-paper-json-pack", "0911.3380", "--intent", "intent", "--workers", "1"],
        ["build-evidence", "0911.3380", "--intent", "intent", "--workers", "1"],
        ["summarize", "0911.3380", "--intent", "intent", "--provider", "manual"],
        ["build", "0911.3380", "--intent", "intent", "--provider", "manual", "--workers", "1"],
        ["status", "0911.3380", "--intent", "intent"],
        ["get-summary", "0911.3380", "--intent", "intent"],
        ["get-graph", "0911.3380", "--intent", "intent"],
    ]
    for args in commands:
        assert cli.main([*args, "--json"]) == 0

    capsys.readouterr()
    assert [item[0] for item in calls] == [
        "identify-foundation",
        "build-network",
        "build-paper-json-pack",
        "build-evidence",
        "summarize",
        "build",
        "status",
        "get-summary",
        "get-graph",
    ]
    assert all(item[1] == "0911.3380" for item in calls)


def test_cli_llm_commands_default_model_tier_to_medium(monkeypatch, capsys):
    calls = []

    def record(seed_paper=None, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "data": {}}

    monkeypatch.setattr(cli.service, "build_domain", record)

    assert cli.main(["llm-build", "0911.3380", "--intent", "intent", "--json"]) == 0

    capsys.readouterr()
    assert calls[0]["model_tier"] == "medium"


def test_cli_rejects_auto_model_tier(capsys):
    try:
        cli.main(["llm-build", "0911.3380", "--model-tier", "auto", "--json"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject --model-tier auto")

    err = capsys.readouterr().err
    assert "invalid choice: 'auto'" in err
