import json

from arc_domain_info import cli


def test_cli_init(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ARC_DOMAIN_INFO_CACHE", str(tmp_path / "domain-info"))

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
    monkeypatch.setattr(cli.service, "build_evidence_pack", record("build-evidence"))
    monkeypatch.setattr(cli.service, "summarize_domain", record("summarize"))
    monkeypatch.setattr(cli.service, "build_domain", record("build"))
    monkeypatch.setattr(cli.service, "status", record("status"))
    monkeypatch.setattr(cli.service, "get_domain_summary", record("get-summary"))
    monkeypatch.setattr(cli.service, "get_domain_graph", record("get-graph"))

    commands = [
        ["identify-foundation", "0911.3380", "--intent", "intent", "--provider", "manual", "--workers", "1"],
        ["build-network", "0911.3380", "--intent", "intent", "--provider", "manual", "--workers", "1"],
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
        "build-evidence",
        "summarize",
        "build",
        "status",
        "get-summary",
        "get-graph",
    ]
    assert all(item[1] == "0911.3380" for item in calls)
