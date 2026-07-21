import hashlib
import json

from arc_paper import worker_controller
from arc_paper.worker_session import WorkerCacheSession


def _guard(monkeypatch, tmp_path, *, session, token="t" * 48):
    guard = tmp_path / "controller-guard.json"
    guard.write_text(
        json.dumps(
            {
                "schema_version": worker_controller.GUARD_SCHEMA_VERSION,
                "session_id": session.session_id,
                "run_root": str(session.run_root),
                "base_root": str(session.base_root),
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    guard.chmod(0o600)
    monkeypatch.setenv("ARC_PAPER_CONTROLLER_MODE", "trusted")
    monkeypatch.setenv("ARC_PAPER_CONTROLLER_GUARD", str(guard))
    monkeypatch.setenv("ARC_PAPER_CONTROLLER_TOKEN", token)


def _args(session):
    return [
        "finalize",
        "--run-root",
        str(session.run_root),
        "--base-root",
        str(session.base_root),
        "--session-id",
        session.session_id,
        "--worker-id",
        "w1",
        "--call-id",
        "c1",
        "--status",
        "failed",
    ]


def test_controller_finalize_requires_trusted_guard(monkeypatch, tmp_path, capsys):
    base = tmp_path / "base"
    base.mkdir()
    session = WorkerCacheSession(base_root=base, run_root=tmp_path / "run", session_id="s1")
    monkeypatch.delenv("ARC_PAPER_CONTROLLER_MODE", raising=False)

    assert worker_controller.main(_args(session)) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "worker_controller_forbidden"


def test_controller_finalize_promotes_and_audits_completion(monkeypatch, tmp_path, capsys):
    base = tmp_path / "base"
    base.mkdir()
    session = WorkerCacheSession(base_root=base, run_root=tmp_path / "run", session_id="s1")
    session.stage_bytes(
        "papers/example.json",
        b'{"schema_version":"test.v1"}',
        source={"operation": "test"},
    )
    _guard(monkeypatch, tmp_path, session=session)

    assert worker_controller.main(_args(session)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data"]["promotion"]["promoted"] == ["papers/example.json"]
    assert (base / "papers" / "example.json").is_file()
    event = json.loads(session.audit_path.read_text(encoding="utf-8"))
    assert event["operation"] == "controller finalize"
    assert event["status"] == "failed"
    assert event["promotion_status"] == "complete"
