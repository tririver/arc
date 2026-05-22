import json

from arc_paper import cli


def test_summary_batch_create_and_status(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    papers = tmp_path / "papers.txt"
    papers.write_text("0911.3380\n0911.3380\nhep-th/0601001\n", encoding="utf-8")

    assert cli.main(["summary-batch", "create", str(papers), "--name", "qft", "--json"]) == 0
    create_output = json.loads(capsys.readouterr().out)
    assert create_output["ok"] is True
    assert create_output["data"]["counts"] == {"queued": 2}

    assert cli.main(["summary-batch", "status", "qft", "--json"]) == 0
    status_output = json.loads(capsys.readouterr().out)
    assert status_output["data"]["counts"] == {"queued": 2}
