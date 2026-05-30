from arc_paper.batch.db import BatchDB


def test_create_batch_deduplicates_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    db = BatchDB.default()

    db.create_batch("qft", ["0911.3380", "arXiv:0911.3380", "hep-th/0601001"], "paper-summary-v1")

    assert db.status_counts("qft") == {"queued": 2}
    assert [item.paper_id for item in db.next_items("qft", status="queued", limit=10)] == [
        "arXiv:0911.3380",
        "arXiv:hep-th/0601001",
    ]


def test_mark_status_and_retry_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")

    db.mark_status("qft", "arXiv:0911.3380", "failed", last_error="boom", attempts=1)
    assert db.status_counts("qft") == {"failed": 1}

    db.retry_failed("qft")
    assert db.status_counts("qft") == {"queued": 1}


def test_create_batch_replaces_existing_items(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")
    db.mark_status("qft", "0911.3380", "done", summary_path="/tmp/old.json")

    db.create_batch("qft", ["hep-th/0601001"], "paper-summary-v1")

    assert db.status_counts("qft") == {"queued": 1}
    assert [item.paper_id for item in db.next_items("qft", status="queued", limit=10)] == [
        "arXiv:hep-th/0601001"
    ]


def test_claim_ready_items_is_atomic_across_db_handles(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    db1 = BatchDB.default()
    db1.create_batch("qft", ["0911.3380", "hep-th/0601001"], "paper-summary-v1")
    db1.mark_status("qft", "0911.3380", "ready")
    db1.mark_status("qft", "hep-th/0601001", "ready")
    db2 = BatchDB.default()

    first = db1.claim_ready_items("qft", limit=1, worker_id="worker-1")
    second = db2.claim_ready_items("qft", limit=10, worker_id="worker-2")

    assert [item.paper_id for item in first] == ["arXiv:0911.3380"]
    assert [item.paper_id for item in second] == ["arXiv:hep-th/0601001"]
    assert db1.status_counts("qft") == {"running": 2}
    assert first[0].worker_id == "worker-1"
    assert first[0].lease_until
