import json

from arc_paper.batch.db import BatchDB
from arc_paper.batch import runner


def test_prefetch_marks_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")

    monkeypatch.setattr(runner.service, "get_title", lambda paper_id, refresh=False: {"ok": True})
    monkeypatch.setattr(runner.service, "get_abstract", lambda paper_id, refresh=False: {"ok": True})
    monkeypatch.setattr(runner.service, "get_toc", lambda paper_id, refresh=False: {"ok": True})

    result = runner.prefetch_batch("qft", workers=1, db=db)

    assert result["counts"] == {"ready": 1}


def test_run_batch_marks_done_and_export(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")
    db.mark_status("qft", "arXiv:0911.3380", "ready")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"title": "Done"}), encoding="utf-8")

    monkeypatch.setattr(
        runner.service,
        "generate_llm_summary",
        lambda paper_id, provider="auto", model=None, refresh=False: {
            "ok": True,
            "data": {"title": "Done"},
            "meta": {"summary_path": str(summary_path)},
        },
    )

    run_result = runner.run_batch("qft", provider="auto", concurrency=1, db=db)
    assert run_result["counts"] == {"done": 1}

    output = tmp_path / "summaries.jsonl"
    export_result = runner.export_batch("qft", output=output, db=db)
    assert export_result["exported"] == 1
    assert json.loads(output.read_text().strip())["title"] == "Done"
