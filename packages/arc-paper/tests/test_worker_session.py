from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from arc_paper.cache import read_json
from arc_paper.worker_session import CachedFetchError, WorkerCacheSession, WorkerSessionError


def make_session(tmp_path, *, max_parallel_fetches=4):
    return WorkerCacheSession(
        base_root=tmp_path / "base",
        run_root=tmp_path / "run",
        session_id="run-1",
        max_parallel_fetches=max_parallel_fetches,
    )


def test_overlay_reads_fall_back_to_base_and_tombstone_hides_base(tmp_path):
    session = make_session(tmp_path)
    base_path = session.base_root / "papers" / "0911.3380" / "metadata.json"
    base_path.parent.mkdir(parents=True)
    base_path.write_text('{"title": "base"}', encoding="utf-8")

    overlay_path = session.overlay_path("papers/0911.3380/metadata.json")
    with session.activated():
        assert read_json(overlay_path) == {"title": "base"}

    session.stage_bytes(
        "papers/0911.3380/metadata.json",
        b'{"title": "overlay"}',
        source={"provider": "fake"},
    )
    with session.activated():
        assert read_json(overlay_path) == {"title": "overlay"}
    assert json.loads(base_path.read_text(encoding="utf-8")) == {"title": "base"}

    session.tombstone("papers/0911.3380/metadata.json", source={"operation": "remove"})
    with session.activated():
        assert read_json(overlay_path) is None
    assert base_path.exists()


def test_promotion_is_atomic_deduplicates_and_preserves_conflicts(tmp_path):
    session = make_session(tmp_path)
    first_payload = b'{"paper_id":"new","source_hash":"a","parser_version":1}'
    second_payload = b'{"paper_id":"new","source_hash":"b","parser_version":2}'
    session.stage_bytes("sources/new.json", first_payload, source={"provider": "fake"})
    first = session.promote()
    assert first.promoted == ("sources/new.json",)
    assert (session.base_root / "sources/new.json").read_bytes() == first_payload

    session.stage_bytes("sources/new.json", first_payload, source={"provider": "fake"})
    second = session.promote()
    assert second.deduplicated == ("sources/new.json",)

    session.stage_bytes("sources/new.json", second_payload, source={"provider": "fake"})
    third = session.promote()
    assert third.conflicted == ("sources/new.json",)
    assert (session.base_root / "sources/new.json").read_bytes() == first_payload
    conflicts = list((session.base_root / ".arc-paper-worker-conflicts").rglob("new.json.*"))
    assert any(path.read_bytes() == second_payload for path in conflicts if not path.name.endswith("record.json"))


def test_invalid_or_tampered_artifact_is_quarantined(tmp_path):
    session = make_session(tmp_path)
    staged = session.stage_bytes("sources/bad.json", b'{"ok": true}', source={"provider": "fake"})
    staged.write_bytes(b"not-json")

    result = session.promote()

    assert result.quarantined == ("sources/bad.json",)
    assert not (session.base_root / "sources/bad.json").exists()
    record = next(session.quarantine_root.rglob("*.record.json"))
    assert json.loads(record.read_text(encoding="utf-8"))["reason"] == "content_hash_mismatch"


def test_finish_call_promotes_after_failure_and_audit_redacts_secrets(tmp_path):
    session = make_session(tmp_path)
    with session.activated(), session.call_scope("call-3"):
        from arc_paper.cache import write_json

        write_json(session.overlay_path("queries/result.json"), {"schema_version": "query.v1"})

    result = session.finish_call(
        worker_id="worker-2",
        call_id="call-3",
        operation="get-metadata",
        status="failed",
        paper_ids=["arXiv:0911.3380"],
        parameters={"query": "safe", "api_token": "do-not-log"},
        source={"provider": "fake", "Authorization": "Bearer hidden"},
        result_hash="abc",
    )

    assert result.promoted == ("queries/result.json",)
    event = json.loads(session.audit_path.read_text(encoding="utf-8"))
    assert event["status"] == "failed"
    assert event["paper_ids"] == ["arXiv:0911.3380"]
    assert event["parameters"]["api_token"] == "[REDACTED]"
    assert event["source"]["Authorization"] == "[REDACTED]"
    assert "do-not-log" not in session.audit_path.read_text(encoding="utf-8")


def test_promoted_tombstone_moves_base_content_to_recoverable_trash(tmp_path):
    session = make_session(tmp_path)
    target = session.base_root / "sources/old.json"
    target.parent.mkdir(parents=True)
    target.write_text("old", encoding="utf-8")
    session.tombstone("sources/old.json", source={"operation": "remove"})

    result = session.promote()

    assert result.deleted == ("sources/old.json",)
    assert not target.exists()
    trash = session.base_root / ".arc-paper-worker-trash" / "run-1" / "sources/old.json"
    assert trash.read_text(encoding="utf-8") == "old"


def test_fetch_once_deduplicates_same_canonical_id_across_workers(tmp_path):
    session = make_session(tmp_path)
    calls = 0
    guard = threading.Lock()

    def fetch():
        nonlocal calls
        with guard:
            calls += 1
        time.sleep(0.03)
        return {"paper_id": "0911.3380"}

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: session.fetch_once("arXiv:0911.3380", fetch), range(6)))

    assert calls == 1
    assert results == [{"paper_id": "0911.3380"}] * 6


def test_fetch_concurrency_is_bounded_and_terminal_errors_are_not_retried(tmp_path):
    session = make_session(tmp_path, max_parallel_fetches=2)
    active = 0
    maximum = 0
    guard = threading.Lock()

    def fetch():
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        return "ok"

    with ThreadPoolExecutor(max_workers=5) as pool:
        assert list(pool.map(lambda value: session.fetch_once(str(value), fetch), range(5))) == ["ok"] * 5
    assert maximum == 2

    class RateLimited(RuntimeError):
        status_code = 429

    calls = 0

    def limited():
        nonlocal calls
        calls += 1
        raise RateLimited("slow down")

    with pytest.raises(RateLimited):
        session.fetch_once("2201.00001", limited)
    with pytest.raises(CachedFetchError):
        session.fetch_once("arXiv:2201.00001", limited)
    assert calls == 1


def test_same_canonical_id_serializes_operations_without_cross_operation_replay(tmp_path):
    session = make_session(tmp_path, max_parallel_fetches=4)
    active = 0
    maximum = 0
    guard = threading.Lock()

    def fetch(value):
        def run():
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with guard:
                active -= 1
            return value
        return run

    with ThreadPoolExecutor(max_workers=2) as pool:
        metadata = pool.submit(
            session.fetch_once, "0911.3380", fetch({"kind": "metadata"}), operation="metadata"
        )
        html = pool.submit(
            session.fetch_once, "arXiv:0911.3380", fetch("<html/>"), operation="html"
        )
        assert metadata.result() == {"kind": "metadata"}
        assert html.result() == "<html/>"
    assert maximum == 1

    refresh_calls = 0
    def refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        return refresh_calls
    assert session.fetch_once("0911.3380", refresh, operation="metadata", replay_success=False) == 1
    assert session.fetch_once("0911.3380", refresh, operation="metadata", replay_success=False) == 2


def test_terminal_rate_limit_blocks_refresh_and_other_operations_for_same_paper(tmp_path):
    session = make_session(tmp_path)

    class RateLimited(RuntimeError):
        status_code = 429

    with pytest.raises(RateLimited):
        session.fetch_once("0911.3380", lambda: (_ for _ in ()).throw(RateLimited()), operation="metadata")
    with pytest.raises(CachedFetchError):
        session.fetch_once(
            "arXiv:0911.3380", lambda: "must-not-run", operation="html", replay_success=False
        )


def test_missing_paper_provider_fetch_is_deduplicated_without_network_or_llm(monkeypatch, tmp_path):
    from arc_paper.providers.inspire import InspireProvider

    session = make_session(tmp_path)
    for key, value in session.environment().items():
        monkeypatch.setenv(key, value)
    calls = 0
    guard = threading.Lock()

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "metadata": {
                    "control_number": 123,
                    "arxiv_eprints": [{"value": "0911.3380"}],
                    "titles": [{"title": "Deduplicated paper"}],
                }
            }

    class FakeClient:
        def get(self, *_args, **_kwargs):
            nonlocal calls
            with guard:
                calls += 1
            time.sleep(0.03)
            return Response()

    provider = InspireProvider(client=FakeClient())
    with ThreadPoolExecutor(max_workers=6) as pool:
        titles = list(pool.map(lambda _: provider.get_metadata("0911.3380")["title"], range(6)))

    assert titles == ["Deduplicated paper"] * 6
    assert calls == 1


def test_cache_paths_reject_traversal_and_session_state(tmp_path):
    session = make_session(tmp_path)
    with pytest.raises(ValueError):
        session.stage_bytes("../escape", b"bad", source={"provider": "fake"})
    with pytest.raises(ValueError):
        session.stage_bytes(".arc-paper-worker/audit.jsonl", b"bad", source={"provider": "fake"})
    with pytest.raises(ValueError, match="allowed ARC paper namespaces"):
        session.stage_bytes("evil/payload.json", b"{}", source={"provider": "fake"})


@pytest.mark.parametrize("session_id", ["../escape", "a/b", "..", ".hidden", "x" * 129, "bad id"])
def test_session_id_is_safe_for_conflict_and_trash_paths(tmp_path, session_id):
    with pytest.raises(ValueError, match="safe"):
        WorkerCacheSession(base_root=tmp_path / "base", run_root=tmp_path / "run", session_id=session_id)


def test_unowned_artifact_is_quarantined_and_other_call_cannot_claim_it(tmp_path):
    session = make_session(tmp_path)
    malicious = session.overlay_path("queries/injected.json")
    malicious.parent.mkdir(parents=True)
    malicious.write_text('{"schema_version":"query.v1"}', encoding="utf-8")

    session.record_call(worker_id="w", call_id="call-other", operation="get-title", status="success")
    result = session.promote()

    assert result.quarantined == ("queries/injected.json",)
    assert not (session.base_root / "queries/injected.json").exists()


def test_call_record_ownership_is_not_reassigned_by_later_call(tmp_path):
    from arc_paper.cache import write_json

    session = make_session(tmp_path)
    with session.activated(), session.call_scope("call-one"):
        write_json(session.overlay_path("queries/owned.json"), {"schema_version": "query.v1"})
    session.record_call(
        worker_id="w1", call_id="call-one", operation="get-metadata", status="success",
        source={"provider": "fake"},
    )
    session.record_call(
        worker_id="w2", call_id="call-two", operation="cache remove", status="success",
        source={"provider": "other"},
    )

    record = json.loads(session._record_path(Path("queries/owned.json")).read_text(encoding="utf-8"))
    assert record["writer_call_id"] == "call-one"
    assert record["operation"] == "get-metadata"
    assert record["source"]["provider"] == "fake"


def test_promotion_rewrites_real_manifest_overlay_paths_to_base(tmp_path):
    session = make_session(tmp_path)
    overlay = str(session.overlay_root)
    arxiv_relative = "papers/arXiv%3A0911.3380/arxiv-source/v2/manifest.json"
    arxiv_manifest = {
        "schema_version": "arc.arxiv_source.v1",
        "paper_id": "arXiv:0911.3380",
        "archive_path": f"{overlay}/papers/arXiv%3A0911.3380/arxiv-source/v2/archives/x.bin",
        "files_root": f"{overlay}/papers/arXiv%3A0911.3380/arxiv-source/v2/files/x",
        "files": [{"path": f"{overlay}/papers/arXiv%3A0911.3380/arxiv-source/v2/files/x/main.tex"}],
    }
    asset_relative = "papers/arXiv%3A0911.3380/ar5iv/assets/manifest.json"
    asset_manifest = {
        "schema_version": "arc.ar5iv_assets.v1",
        "paper_id": "arXiv:0911.3380",
        "assets": [{"cache_path": f"{overlay}/papers/arXiv%3A0911.3380/ar5iv/assets/sha256/aa/a.png"}],
    }
    session.stage_bytes(arxiv_relative, json.dumps(arxiv_manifest).encode(), source={"provider": "arxiv"})
    session.stage_bytes(asset_relative, json.dumps(asset_manifest).encode(), source={"provider": "ar5iv"})

    result = session.promote()

    assert set(result.promoted) == {arxiv_relative, asset_relative}
    for relative in (arxiv_relative, asset_relative):
        text = (session.base_root / relative).read_text(encoding="utf-8")
        assert str(session.overlay_root) not in text
        assert str(session.base_root) in text


def test_provider_generated_arxiv_source_manifest_has_live_promoted_paths(monkeypatch, tmp_path):
    from arc_paper.providers.arxiv_source import ArxivSourceProvider

    session = make_session(tmp_path)
    for key, value in session.environment().items():
        monkeypatch.setenv(key, value)

    class Response:
        status_code = 200
        content = b"\\documentclass{article}\\begin{document}ok\\end{document}"
        headers = {"x-arxiv-license": "test-license"}
        @staticmethod
        def raise_for_status():
            return None

    class Client:
        @staticmethod
        def get(*_args, **_kwargs):
            return Response()

    with session.call_scope("source-call"):
        manifest = ArxivSourceProvider(client=Client()).cache_source("0911.3380", version=1)
    session.record_call(
        worker_id="w", call_id="source-call", operation="source-cache", status="success",
        source={"provider": "arxiv-source"},
    )
    result = session.promote()

    assert any(path.endswith("manifest.json") for path in result.promoted)
    relative_manifest = Path(manifest["archive_path"]).parent.parent / "manifest.json"
    promoted_manifest = Path(str(relative_manifest).replace(str(session.overlay_root), str(session.base_root)))
    stored = json.loads(promoted_manifest.read_text(encoding="utf-8"))
    assert Path(stored["archive_path"]).is_file()
    assert Path(stored["files_root"]).is_dir()
    assert str(session.overlay_root) not in promoted_manifest.read_text(encoding="utf-8")


def test_provider_generated_ar5iv_asset_manifest_has_live_promoted_path(monkeypatch, tmp_path):
    from arc_paper.providers.ar5iv import Ar5ivProvider

    session = make_session(tmp_path)
    for key, value in session.environment().items():
        monkeypatch.setenv(key, value)

    class Response:
        is_redirect = False
        headers = {"content-type": "image/png", "content-length": "3"}
        content = b"png"
        url = "https://ar5iv.labs.arxiv.org/assets/a.png"
        @staticmethod
        def raise_for_status():
            return None

    class Client:
        @staticmethod
        def get(*_args, **_kwargs):
            return Response()

    provider = Ar5ivProvider(client=Client())
    with session.call_scope("asset-call"):
        assets = provider.cache_assets(
            "0911.3380", '<html><body><img src="/assets/a.png"></body></html>'
        )
    session.record_call(
        worker_id="w", call_id="asset-call", operation="parse", status="success",
        source={"provider": "ar5iv"},
    )
    session.promote()

    promoted_path = Path(str(assets[0]["cache_path"]).replace(str(session.overlay_root), str(session.base_root)))
    manifest = session.base_root / "papers/arXiv%3A0911.3380/ar5iv/assets/manifest.json"
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert promoted_path.is_file()
    assert stored["assets"][0]["cache_path"] == str(promoted_path)


def test_path_lock_prevents_promotion_from_losing_concurrent_writer(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    relative = "queries/race.json"
    session.stage_bytes(relative, b'{"value":1}', source={"provider": "first"})
    entered = threading.Event()
    release = threading.Event()
    original = session._promotion_bytes

    def slow(path):
        entered.set()
        assert release.wait(1)
        return original(path)

    monkeypatch.setattr(session, "_promotion_bytes", slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        promoting = pool.submit(session.promote)
        assert entered.wait(1)
        writing = pool.submit(
            session.stage_bytes, relative, b'{"value":2}', source={"provider": "second"}
        )
        time.sleep(0.03)
        assert not writing.done()
        release.set()
        assert promoting.result().promoted == (relative,)
        writing.result()

    assert json.loads((session.base_root / relative).read_text()) == {"value": 1}
    assert json.loads(session.overlay_path(relative).read_text()) == {"value": 2}


def test_promotion_ignores_ephemeral_lock_and_temporary_files(tmp_path):
    session = make_session(tmp_path)
    lock = session.overlay_root / "locks/sources/paper.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    temporary = session.overlay_path("sources/paper.json.deadbeef.tmp")
    temporary.parent.mkdir(parents=True)
    temporary.write_text("partial", encoding="utf-8")

    result = session.finish_call(
        worker_id="w1", call_id="c1", operation="parse", status="failed", source={"provider": "fake"}
    )

    assert result.as_dict() == {
        "promoted": [], "deduplicated": [], "conflicted": [], "quarantined": [], "deleted": []
    }
    assert not (session.base_root / "locks/sources/paper.lock").exists()
    assert not (session.base_root / "sources/paper.json.deadbeef.tmp").exists()


def test_from_environment_reopens_controller_session(tmp_path):
    session = make_session(tmp_path)
    with session.activated():
        reopened = WorkerCacheSession.from_environment()
    assert reopened is not None
    assert reopened.base_root == session.base_root
    assert reopened.overlay_root == session.overlay_root
    assert reopened.session_id == session.session_id


def test_open_or_create_needs_only_controller_owned_paths(monkeypatch, tmp_path):
    run_root = tmp_path / "run-from-controller"
    monkeypatch.delenv("ARC_PAPER_CACHE", raising=False)
    monkeypatch.setenv("ARC_PAPER_WORKER_BASE_CACHE", str(tmp_path / "global-cache"))
    monkeypatch.setenv("ARC_PAPER_WORKER_SESSION_DIR", str(run_root))
    monkeypatch.setenv("ARC_PAPER_WORKER_SESSION_ID", "controller-run")

    session = WorkerCacheSession.open_or_create_from_environment()

    assert session is not None
    assert session.run_root == run_root.resolve()
    assert session.overlay_root == run_root.resolve() / "paper-cache-overlay"
    with session.activated():
        assert WorkerCacheSession.from_environment() is not None


def test_open_or_create_fails_closed_on_partial_or_changed_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_WORKER_BASE_CACHE", str(tmp_path / "base"))
    with pytest.raises(WorkerSessionError, match="incomplete worker session environment"):
        WorkerCacheSession.open_or_create_from_environment()

    session = make_session(tmp_path)
    env = session.environment()
    env["ARC_PAPER_WORKER_SESSION_ID"] = "different-run"
    monkeypatch.setenv("ARC_PAPER_WORKER_BASE_CACHE", env["ARC_PAPER_WORKER_BASE_CACHE"])
    monkeypatch.setenv("ARC_PAPER_WORKER_SESSION_DIR", env["ARC_PAPER_WORKER_SESSION_DIR"])
    monkeypatch.setenv("ARC_PAPER_WORKER_SESSION_ID", env["ARC_PAPER_WORKER_SESSION_ID"])
    monkeypatch.setenv("ARC_PAPER_CACHE", env["ARC_PAPER_CACHE"])
    with pytest.raises(WorkerSessionError, match="manifest does not match"):
        WorkerCacheSession.open_or_create_from_environment()


def test_cache_list_search_and_remove_see_union_base(monkeypatch, tmp_path):
    from arc_paper import service

    session = make_session(tmp_path)
    source = session.base_root / "sources" / "lecture-9.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "paper_id": "lecture-9",
                "parser_version": 1,
                "source_hash": "abc",
                "toc": [],
                "sections": [{"title": "Intro", "text": "worker union evidence"}],
                "equations": [],
                "structure": {"requested_document_kind": "auto"},
                "index_entries": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("arc_paper.search.shutil.which", lambda _name: None)

    with session.activated():
        listed = service.list_cached_papers(ids=["lecture-9"])
        searched = service.search_full_text(None, query="union evidence")
        removed = service.remove_cached_papers(ids=["lecture-9"], dry_run=False)
        hidden = service.list_cached_papers(ids=["lecture-9"])

    assert listed["data"]["count"] == 1
    assert searched["ok"] is True
    assert searched["data"][0]["paper_id"] == "lecture-9"
    assert removed["data"]["removed_count"] == 1
    assert hidden["data"]["count"] == 0
    assert source.exists(), "worker removal must not directly mutate the base"
