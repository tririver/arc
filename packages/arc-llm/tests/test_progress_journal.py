from __future__ import annotations

import json

from arc_llm.progress_journal import ProgressJournal


def test_progress_journal_persists_session_and_recovery_context(tmp_path) -> None:
    journal = ProgressJournal(
        artifact_dir=tmp_path,
        call_label="worker-1",
        provider="codex-cli",
        callback=None,
    )

    journal(
        {
            "event": "provider_progress",
            "activity_kind": "session",
            "summary": "provider session established",
            "substantive": False,
            "native_session_id": "session-1",
            "resumable": True,
        }
    )
    journal(
        {
            "event": "provider_progress",
            "activity_kind": "assistant",
            "summary": "completed the first derivation",
            "substantive": True,
        }
    )

    events = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert events[0]["substantive"] is False
    assert events[1]["native_session_id"] == "session-1"
    assert events[1]["resumable"] is True
    assert json.loads((tmp_path / "latest_progress.json").read_text()) == events[1]
