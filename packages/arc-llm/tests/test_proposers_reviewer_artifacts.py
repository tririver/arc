from __future__ import annotations

import json

import pytest

from arc_llm.proposers_reviewer.artifacts import (
    LockConflictError,
    RunPaths,
    atomic_write_json,
    atomic_write_text,
    acquire_lock,
)


def test_run_paths_use_configured_run_dir_as_direct_parent(tmp_path):
    paths = RunPaths(run_dir=tmp_path / "project" / "ideas", run_id="run_001")
    loop_paths = paths.loop("loop_001")
    round_paths = loop_paths.round(1)

    assert paths.run_root == tmp_path / "project" / "ideas" / "run_001"
    assert loop_paths.loop_root == paths.run_root / "loops" / "loop_001"
    assert round_paths.context_dir == loop_paths.loop_root / "rounds" / "round_001" / "context"
    assert round_paths.proposer_output("proposer_001") == (
        loop_paths.loop_root / "rounds" / "round_001" / "proposer_outputs" / "proposer_001.json"
    )
    assert round_paths.review("reviewer_001") == (
        loop_paths.loop_root / "rounds" / "round_001" / "reviews" / "reviewer_001.json"
    )


def test_atomic_write_json_and_text_create_complete_files(tmp_path):
    json_path = tmp_path / "nested" / "data.json"
    text_path = tmp_path / "nested" / "prompt.md"

    atomic_write_json(json_path, {"ok": True})
    atomic_write_text(text_path, "hello")

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    assert text_path.read_text(encoding="utf-8") == "hello"
    assert not list((tmp_path / "nested").glob("*.tmp"))


def test_acquiring_same_lock_twice_fails(tmp_path):
    lock_path = tmp_path / "run_001" / "loops" / "loop_001" / "lock.json"

    with acquire_lock(lock_path, run_id="run_001", loop_id="loop_001"):
        with pytest.raises(LockConflictError, match="lock already exists"):
            with acquire_lock(lock_path, run_id="run_001", loop_id="loop_001"):
                pass


def test_lock_file_records_run_and_loop_id(tmp_path):
    lock_path = tmp_path / "run_001" / "loops" / "loop_001" / "lock.json"

    with acquire_lock(lock_path, run_id="run_001", loop_id="loop_001"):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))

    assert payload["run_id"] == "run_001"
    assert payload["loop_id"] == "loop_001"
    assert "pid" in payload
    assert "thread_id" in payload
