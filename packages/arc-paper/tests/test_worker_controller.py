import json

from arc_paper import worker_controller
from arc_paper.worker_session import WorkerCacheSession


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


def test_controller_finalize_promotes_and_audits_completion(tmp_path, capsys):
    base = tmp_path / "base"
    base.mkdir()
    session = WorkerCacheSession(base_root=base, run_root=tmp_path / "run", session_id="s1")
    session.stage_bytes(
        "papers/example.json",
        b'{"schema_version":"test.v1"}',
        source={"operation": "test"},
    )
    assert worker_controller.main(_args(session)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data"]["promotion"]["promoted"] == ["papers/example.json"]
    assert (base / "papers" / "example.json").is_file()
    event = json.loads(session.audit_path.read_text(encoding="utf-8"))
    assert event["operation"] == "controller finalize"
    assert event["status"] == "failed"
    assert event["promotion_status"] == "complete"
