from arc_paper_query.batch.db import BatchDB


def test_create_batch_deduplicates_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    db = BatchDB.default()

    db.create_batch("qft", ["0911.3380", "arXiv:0911.3380", "hep-th/0601001"], "paper-summary-v1")

    assert db.status_counts("qft") == {"queued": 2}
    assert [item.paper_id for item in db.next_items("qft", status="queued", limit=10)] == [
        "arXiv:0911.3380",
        "arXiv:hep-th/0601001",
    ]


def test_mark_status_and_retry_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")

    db.mark_status("qft", "arXiv:0911.3380", "failed", last_error="boom", attempts=1)
    assert db.status_counts("qft") == {"failed": 1}

    db.retry_failed("qft")
    assert db.status_counts("qft") == {"queued": 1}
