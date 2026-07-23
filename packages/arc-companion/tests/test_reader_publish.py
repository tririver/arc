from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from arc_companion.io import read_json, write_json
import arc_companion.pipeline as pipeline_module
from arc_companion.reader_publish import (
    PreparedReaderCandidate,
    ReaderObject,
    ReaderPublishCoordinator,
    READER_PUBLISH_STATE_VERSION,
    parse_reader_commit_utc,
)


class _Clock:
    def __init__(self) -> None:
        self.utc = datetime(2026, 7, 23, tzinfo=timezone.utc)
        self.mono = 100.0

    def advance(self, seconds: float) -> None:
        self.utc += timedelta(seconds=seconds)
        self.mono += seconds


def _candidate(
    text: str,
    *,
    status: str = "active",
    created_at: str = "first",
    checkpoint_path: str = "/one",
    builtin: bytes = b"app",
    sources: tuple[ReaderObject, ...] = (),
) -> PreparedReaderCandidate:
    return PreparedReaderCandidate(
        snapshot={
            "schema_version": "reader.v1",
            "status": status,
            "text": text,
            "created_at": created_at,
            "checkpoint_path": checkpoint_path,
        },
        web_render_version="web.v1",
        builtin_objects=(ReaderObject("assets/app.js", builtin, "builtin"),),
        source_objects=sources,
    )


def _coordinator(
    state: dict[str, object],
    clock: _Clock,
    published: list[str],
    *,
    fail: bool = False,
) -> ReaderPublishCoordinator:
    def merge(values: dict[str, object]) -> dict[str, object]:
        state.update(values)
        return dict(state)

    def publish(candidate: PreparedReaderCandidate) -> dict[str, object]:
        if fail:
            raise RuntimeError("publish failed")
        published.append(candidate.semantic_sha256)
        return {"output_html": "/reader/index.html"}

    return ReaderPublishCoordinator(
        state_loader=lambda: dict(state),
        state_merger=merge,
        preparer=lambda value: value,
        publisher=publish,
        utc_now=lambda: clock.utc,
        monotonic=lambda: clock.mono,
    )


def test_semantic_digest_ignores_only_operational_metadata_and_order() -> None:
    first_sources = (
        ReaderObject("reader/assets/z.png", b"z", "source"),
        ReaderObject("reader/assets/a.png", b"a", "source"),
    )
    second_sources = tuple(reversed(first_sources))
    first = _candidate(
        "same",
        created_at="one",
        checkpoint_path="/machine/one",
        sources=first_sources,
    )
    second = _candidate(
        "same",
        created_at="two",
        checkpoint_path="/machine/two",
        sources=second_sources,
    )
    assert first.semantic_sha256 == second.semantic_sha256

    assert _candidate("changed").semantic_sha256 != _candidate("same").semantic_sha256
    assert _candidate("same", status="complete").semantic_sha256 != second.semantic_sha256
    assert _candidate("same", builtin=b"changed").semantic_sha256 != second.semantic_sha256
    assert _candidate(
        "same",
        sources=(ReaderObject("reader/assets/a.png", b"different", "source"),),
    ).semantic_sha256 != second.semantic_sha256


def test_exact_interval_uses_monotonic_and_final_bypasses_timing() -> None:
    state: dict[str, object] = {}
    clock = _Clock()
    published: list[str] = []
    coordinator = _coordinator(state, clock, published)

    first = _candidate("first")
    assert coordinator.request(lambda: first).status == "published"
    second = _candidate("second")
    clock.advance(59.999)
    assert coordinator.request(lambda: second).status == "deferred"
    assert len(published) == 1
    clock.advance(0.001)
    assert coordinator.request(lambda: second).status == "published"

    third = _candidate("third")
    assert coordinator.request(lambda: third, final=True).status == "published"
    assert len(published) == 3


def test_restart_derives_one_deadline_from_utc_then_uses_monotonic() -> None:
    clock = _Clock()
    state: dict[str, object] = {
        "reader_publish_state_version": READER_PUBLISH_STATE_VERSION,
        "reader_dirty": True,
        "reader_committed_at": (clock.utc - timedelta(seconds=20)).isoformat(),
    }
    published: list[str] = []
    coordinator = _coordinator(state, clock, published)
    value = _candidate("next")

    clock.utc += timedelta(days=20)
    clock.mono += 39.999
    assert coordinator.request(lambda: value).status == "deferred"
    clock.mono += 0.001
    assert coordinator.request(lambda: value).status == "published"


def test_commit_timestamp_requires_canonical_utc_offset() -> None:
    assert parse_reader_commit_utc("2026-07-23T00:00:00+00:00") is not None
    assert parse_reader_commit_utc("2026-07-23T00:00:00Z") is None
    assert parse_reader_commit_utc("2026-07-23T01:00:00+01:00") is None
    assert parse_reader_commit_utc("2026-07-23T00:00:00") is None


@pytest.mark.parametrize("seconds", [1, 600])
def test_future_or_rollback_commit_timestamp_waits_full_interval(seconds: int) -> None:
    clock = _Clock()
    state: dict[str, object] = {
        "reader_committed_at": (clock.utc + timedelta(seconds=seconds)).isoformat(),
    }
    published: list[str] = []
    coordinator = _coordinator(state, clock, published)
    value = _candidate("next")

    clock.advance(59.999)
    assert coordinator.request(lambda: value).status == "deferred"
    clock.advance(0.001)
    assert coordinator.request(lambda: value).status == "published"


@pytest.mark.parametrize(
    "committed_at",
    ["malformed", "2026-07-23T01:00:00+01:00"],
)
def test_nonempty_invalid_commit_timestamp_waits_full_interval(
    committed_at: str,
) -> None:
    clock = _Clock()
    state: dict[str, object] = {"reader_committed_at": committed_at}
    coordinator = _coordinator(state, clock, [])
    value = _candidate("next")

    clock.advance(59.999)
    assert coordinator.request(lambda: value).status == "deferred"
    clock.advance(0.001)
    assert coordinator.request(lambda: value).status == "published"


def test_missing_commit_timestamp_is_immediately_eligible() -> None:
    clock = _Clock()
    state: dict[str, object] = {"reader_committed_at": ""}
    published: list[str] = []
    coordinator = _coordinator(state, clock, published)
    assert coordinator.request(lambda: _candidate("next")).status == "published"


def test_semantic_dedupe_writes_no_files_and_does_not_advance_time() -> None:
    clock = _Clock()
    value = _candidate("same")
    state: dict[str, object] = {
        "reader_publish_state_version": READER_PUBLISH_STATE_VERSION,
        "reader_dirty": True,
        "reader_committed_semantic_sha256": value.semantic_sha256,
        "reader_committed_at": clock.utc.isoformat(),
    }
    published: list[str] = []
    coordinator = _coordinator(state, clock, published)
    before = state["reader_committed_at"]

    result = coordinator.request(lambda: value, final=True)

    assert result.status == "deduplicated"
    assert published == []
    assert state["reader_committed_at"] == before
    assert state["reader_dirty"] is False


def test_failure_preserves_dirty_and_final_is_strict() -> None:
    state: dict[str, object] = {}
    clock = _Clock()
    coordinator = _coordinator(state, clock, [], fail=True)
    value = _candidate("next")

    assert coordinator.request(lambda: value).status == "failed"
    assert state["reader_dirty"] is True
    with pytest.raises(RuntimeError, match="publish failed"):
        coordinator.request(lambda: value, final=True)
    assert state["reader_dirty"] is True


def test_interrupt_is_never_downgraded_to_nonstrict_failure() -> None:
    state: dict[str, object] = {}
    clock = _Clock()

    def merge(values: dict[str, object]) -> dict[str, object]:
        state.update(values)
        return dict(state)

    coordinator = ReaderPublishCoordinator(
        state_loader=lambda: dict(state),
        state_merger=merge,
        preparer=lambda value: value,
        publisher=lambda _candidate: (_ for _ in ()).throw(
            KeyboardInterrupt()
        ),
        utc_now=lambda: clock.utc,
        monotonic=lambda: clock.mono,
    )

    with pytest.raises(KeyboardInterrupt):
        coordinator.request(lambda: _candidate("next"))
    assert state["reader_dirty"] is True


def test_content_accepted_during_publish_remains_dirty() -> None:
    state: dict[str, object] = {}
    clock = _Clock()
    current = [_candidate("first")]
    entered = threading.Event()
    release = threading.Event()
    published: list[str] = []

    def merge(values: dict[str, object]) -> dict[str, object]:
        state.update(values)
        return dict(state)

    def publish(candidate: PreparedReaderCandidate) -> dict[str, object]:
        published.append(candidate.semantic_sha256)
        entered.set()
        assert release.wait(timeout=2)
        return {"output_html": "/reader/index.html"}

    coordinator = ReaderPublishCoordinator(
        state_loader=lambda: dict(state),
        state_merger=merge,
        preparer=lambda value: value,
        publisher=publish,
        utc_now=lambda: clock.utc,
        monotonic=lambda: clock.mono,
    )
    result: list[object] = []
    thread = threading.Thread(
        target=lambda: result.append(coordinator.request(lambda: current[0]))
    )
    thread.start()
    assert entered.wait(timeout=2)
    current[0] = _candidate("newly accepted")
    release.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result[0].dirty is True
    assert state["reader_dirty"] is True
    assert len(published) == 1


def test_pipeline_first_coordinator_creation_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.RLock()
    created: list[object] = []
    sentinel = object()
    barrier = threading.Barrier(2)
    pipeline_module._READER_COORDINATORS.pop(lock, None)

    def create(*_args):
        created.append(object())
        return sentinel

    monkeypatch.setattr(
        pipeline_module, "_create_reader_coordinator", create,
    )
    results: list[object] = []

    def worker() -> None:
        barrier.wait(timeout=2)
        results.append(
            pipeline_module._reader_coordinator(
                tmp_path, tmp_path / "state.json", lock,
            )
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert results == [sentinel, sentinel]
    assert len(created) == 1
    pipeline_module._READER_COORDINATORS.pop(lock, None)


def test_late_acceptance_after_sample_cannot_be_overwritten_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state_path = project / "state.json"
    write_json(state_path, {"schema_version": "arc.companion.state.v3"})
    lock = threading.RLock()
    sampled = threading.Event()
    second_started = threading.Event()
    current = [{"content": "old"}]
    seen: list[object] = []

    class FakeCoordinator:
        calls = 0

        def request(self, latest, **_kwargs):
            self.calls += 1
            seen.append(latest())
            if self.calls == 1:
                sampled.set()
                assert second_started.wait(timeout=2)
                pipeline_module._state(state_path, reader_dirty=False)
                status = "published"
            else:
                status = "deferred"
            return SimpleNamespace(
                status=status,
                state=read_json(state_path),
            )

    fake = FakeCoordinator()
    monkeypatch.setattr(
        pipeline_module,
        "_reader_coordinator",
        lambda *_args: fake,
    )
    first = threading.Thread(
        target=lambda: pipeline_module._publish_reader_update(
            project, state_path, lock,
            final_overrides=lambda: dict(current[0]),
        )
    )
    first.start()
    assert sampled.wait(timeout=2)
    current[0] = {"content": "new"}

    def accept_late() -> None:
        second_started.set()
        pipeline_module._publish_reader_update(
            project, state_path, lock,
            final_overrides=lambda: dict(current[0]),
        )

    second = threading.Thread(target=accept_late)
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert seen == [{"content": "old"}, {"content": "new"}]
    assert read_json(state_path)["reader_dirty"] is True
