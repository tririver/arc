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


def test_lease_heartbeat_and_completion_require_matching_token(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")
    db.mark_status("qft", "0911.3380", "ready")

    item = db.claim_ready_items("qft", limit=1, worker_id="worker-1")[0]

    assert item.lease_token
    assert db.heartbeat("qft", item.paper_id, lease_token="wrong") is False
    assert db.mark_status("qft", item.paper_id, "done", lease_token="wrong") is False
    assert db.status_counts("qft") == {"running": 1}
    assert db.heartbeat("qft", item.paper_id, lease_token=item.lease_token) is True
    assert db.mark_status("qft", item.paper_id, "done", lease_token=item.lease_token) is True
    assert db.status_counts("qft") == {"done": 1}


def test_legacy_database_is_migrated_additively(monkeypatch, tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE batches(name TEXT PRIMARY KEY, created_at TEXT NOT NULL, prompt_version TEXT NOT NULL);
            CREATE TABLE batch_items(
              batch_name TEXT NOT NULL, paper_id TEXT NOT NULL, status TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0, provider TEXT, model TEXT, source_hash TEXT,
              summary_path TEXT, last_error TEXT, worker_id TEXT, lease_until TEXT,
              updated_at TEXT NOT NULL, PRIMARY KEY(batch_name, paper_id)
            );
            """
        )

    BatchDB(path)

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(batch_items)")}
    assert {"lease_token", "owner_pid", "owner_started_at", "heartbeat_at"} <= columns


def test_expired_lease_is_not_reclaimed_from_live_owner_but_dead_owner_is(monkeypatch, tmp_path):
    import sqlite3

    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")
    db.mark_status("qft", "0911.3380", "ready")
    first = db.claim_ready_items("qft", limit=1, worker_id="worker-1")[0]
    with sqlite3.connect(db.path) as connection:
        connection.execute(
            "UPDATE batch_items SET lease_until = ? WHERE batch_name = ? AND paper_id = ?",
            ("2000-01-01T00:00:00+00:00", "qft", first.paper_id),
        )

    assert BatchDB(db.path).claim_ready_items("qft", limit=1, worker_id="worker-2") == []

    with sqlite3.connect(db.path) as connection:
        connection.execute(
            "UPDATE batch_items SET owner_pid = ?, owner_started_at = ? WHERE batch_name = ? AND paper_id = ?",
            (999999999, "dead", "qft", first.paper_id),
        )
    reclaimed = BatchDB(db.path).claim_ready_items("qft", limit=1, worker_id="worker-2")
    assert len(reclaimed) == 1
    assert reclaimed[0].lease_token != first.lease_token
