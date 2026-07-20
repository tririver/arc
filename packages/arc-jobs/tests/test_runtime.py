from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from arc_jobs import runtime


def _env(tmp_path: Path) -> dict[str, str]:
    return {"ARC_HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "old")}


def test_runtime_paths_use_arc_home_and_overrides(tmp_path):
    paths = runtime.resolve_runtime_paths(_env(tmp_path))
    assert paths.paper_cache == tmp_path / "home" / "cache" / "arc-paper"
    assert paths.domain_cache == tmp_path / "home" / "cache" / "arc-domain"
    assert paths.llm_cache == tmp_path / "home" / "cache" / "arc-llm"
    assert paths.jobs == tmp_path / "home" / "jobs"
    env = {**_env(tmp_path), "ARC_PAPER_CACHE": str(tmp_path / "custom")}
    assert runtime.resolve_runtime_paths(env).paper_cache == tmp_path / "custom"


def test_migration_moves_deduplicates_and_preserves_conflicts(tmp_path):
    env = _env(tmp_path)
    old = tmp_path / "old" / "arc" / "arc-paper"
    old.mkdir(parents=True)
    for name, value in (("move", "move"), ("same", "same"), ("conflict", "old")):
        (old / name).write_text(value, encoding="utf-8")
    target = tmp_path / "home" / "cache" / "arc-paper"
    target.mkdir(parents=True)
    (target / "same").write_text("same", encoding="utf-8")
    (target / "conflict").write_text("new", encoding="utf-8")

    result = runtime.prepare_runtime(env)["migration"]

    assert result["status"] == "completed"
    assert (result["files_moved"], result["files_deduplicated"], result["files_conflicted"]) == (
        1,
        1,
        1,
    )
    assert (target / "move").read_text(encoding="utf-8") == "move"
    conflicts = list((tmp_path / "home" / "migration-conflicts").rglob("conflict.*"))
    assert len(conflicts) == 1 and conflicts[0].read_text(encoding="utf-8") == "old"
    assert runtime.prepare_runtime(env)["migration"] == result


def test_cross_device_migration_verifies_before_source_delete(tmp_path, monkeypatch):
    env = _env(tmp_path)
    source_dir = tmp_path / "old" / "arc" / "arc-domain"
    source_dir.mkdir(parents=True)
    source = source_dir / "entry"
    source.write_text("payload", encoding="utf-8")
    real_replace = os.replace

    def cross_device(old, new):
        if Path(old) in {source_dir, source}:
            raise OSError(errno.EXDEV, "cross-device")
        return real_replace(old, new)

    monkeypatch.setattr(runtime.os, "replace", cross_device)
    report = runtime.prepare_runtime(env)["migration"]
    target = tmp_path / "home" / "cache" / "arc-domain" / "entry"
    assert report["files_verified"] == 1
    assert target.read_text(encoding="utf-8") == "payload"
    assert not source.exists()


def test_migration_failure_records_manifest_and_stops(tmp_path):
    env = _env(tmp_path)
    source = tmp_path / "old" / "arc" / "arc-paper"
    source.mkdir(parents=True)
    (source / "unsafe").symlink_to(tmp_path / "outside")
    with pytest.raises(runtime.RuntimeMigrationError, match="split cache state"):
        runtime.prepare_runtime(env)
    manifest = json.loads(
        (tmp_path / "home" / "migrations" / "unified-arc-home-v1.json").read_text()
    )
    assert manifest["status"] == "failed"


def test_doctor_reports_migration_host_and_provider(tmp_path):
    env = {**_env(tmp_path), "ARC_AGENT_HOST": "codex"}
    runtime.prepare_runtime(env)
    report = runtime.doctor(env)
    assert report["migration"]["status"] == "completed"
    assert report["host"] == "codex"
    assert report["provider"] == "codex-cli"
