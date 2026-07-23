from __future__ import annotations

import json
import hashlib
import multiprocessing
import os
from pathlib import Path
import stat
import threading

import pytest

from arc_companion.artifact_ids import (
    ARTIFACT_ID_RECEIPT_NAME,
    ARTIFACT_ID_RECEIPT_VERSION,
    ArtifactIdError,
    allocate_artifact_dir,
    ensure_artifact_alias_receipt,
    read_artifact_alias_receipt,
    render_artifact_identity,
    resolve_artifact_dir,
)


def _identity(prefix: str, fill: str) -> str:
    return prefix + fill * (64 - len(prefix))


def _process_allocate(root: str, identity: str, queue: object) -> None:
    allocation = allocate_artifact_dir(
        Path(root), identity, kind="checkpoint",
    )
    queue.put((str(allocation.path), allocation.disposition))


def _process_allocate_stem(
    root: str, identity: str, stem: str, queue: object,
) -> None:
    allocation = allocate_artifact_dir(
        Path(root), identity, kind="render", stem=stem,
    )
    queue.put((str(allocation.path), allocation.prefix_length))


def test_allocates_shortest_prefix_and_records_full_identity(
    tmp_path: Path,
) -> None:
    identity = "a" * 64
    allocation = allocate_artifact_dir(
        tmp_path, identity, kind="checkpoint",
    )
    receipt = json.loads(
        (allocation.path / ARTIFACT_ID_RECEIPT_NAME).read_text()
    )
    assert allocation.path.name == identity[:12]
    assert allocation.prefix_length == 12
    assert receipt == {
        "schema_version": ARTIFACT_ID_RECEIPT_VERSION,
        "kind": "checkpoint",
        "full_identity": identity,
        "display_id": identity[:12],
        "prefix_length": 12,
        "stem": "",
    }
    resolved = resolve_artifact_dir(
        tmp_path, allocation.path, expected_identity=identity,
        kind="checkpoint",
    )
    assert resolved.path == allocation.path
    assert resolved.identity == allocation.identity
    assert resolved.disposition == "adopted"


def test_collision_extends_12_16_20_until_unique(tmp_path: Path) -> None:
    shared12 = "1234567890ab"
    first = _identity(shared12 + "aaaa", "1")
    second = _identity(shared12 + "bbbb", "2")
    third = _identity(shared12 + "bbbb" + "cccc", "3")
    assert allocate_artifact_dir(
        tmp_path, first, kind="render", stem="paper",
    ).prefix_length == 12
    assert allocate_artifact_dir(
        tmp_path, second, kind="render", stem="paper",
    ).prefix_length == 16
    allocation = allocate_artifact_dir(
        tmp_path, third, kind="render", stem="paper",
    )
    assert allocation.prefix_length == 20


def test_collision_prefix_namespace_is_shared_across_stems(
    tmp_path: Path,
) -> None:
    shared12 = "1234567890ab"
    first = _identity(shared12 + "aaaa", "1")
    second = _identity(shared12 + "bbbb", "2")
    assert allocate_artifact_dir(
        tmp_path, first, kind="render", stem="paper-one",
    ).prefix_length == 12
    allocation = allocate_artifact_dir(
        tmp_path, second, kind="render", stem="paper-two",
    )
    assert allocation.prefix_length == 16


def test_invalid_same_prefix_receipt_under_another_stem_fails_closed(
    tmp_path: Path,
) -> None:
    identity = _identity("1234567890ab" + "aaaa", "1")
    occupied = tmp_path / "other-1234567890ab"
    occupied.mkdir()
    (occupied / ARTIFACT_ID_RECEIPT_NAME).write_text(
        "{}", encoding="utf-8",
    )
    with pytest.raises(ArtifactIdError):
        allocate_artifact_dir(
            tmp_path, identity, kind="render", stem="paper",
        )


def test_same_identity_concurrent_allocation_is_idempotent(
    tmp_path: Path,
) -> None:
    identity = "b" * 64
    barrier = threading.Barrier(2)
    results = []

    def worker() -> None:
        barrier.wait(timeout=2)
        results.append(
            allocate_artifact_dir(
                tmp_path, identity, kind="checkpoint",
            )
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert len(results) == 2
    assert results[0].path == results[1].path
    assert results[0].prefix_length == 12


def test_legacy_full_identity_directory_is_read_without_rename(
    tmp_path: Path,
) -> None:
    identity = "c" * 64
    legacy = tmp_path / identity
    legacy.mkdir()
    allocation = allocate_artifact_dir(
        tmp_path, identity, kind="checkpoint",
    )
    assert allocation.path == legacy
    assert allocation.legacy is True
    assert not (legacy / ARTIFACT_ID_RECEIPT_NAME).exists()
    assert resolve_artifact_dir(
        tmp_path, legacy, expected_identity=identity, kind="checkpoint",
    ).legacy is True


def test_resolver_rejects_receipt_mismatch_and_escape(tmp_path: Path) -> None:
    allocation = allocate_artifact_dir(
        tmp_path, "d" * 64, kind="checkpoint",
    )
    receipt_path = allocation.path / ARTIFACT_ID_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["full_identity"] = "e" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ArtifactIdError, match="differ"):
        resolve_artifact_dir(tmp_path, allocation.path)
    outside = tmp_path.parent / "outside-artifact"
    outside.mkdir(exist_ok=True)
    with pytest.raises(ArtifactIdError, match="escapes"):
        resolve_artifact_dir(tmp_path, outside)


def test_render_identity_round_trip_binds_payload_nonce_and_stem(
    tmp_path: Path,
) -> None:
    payload = {
        "content_sha256": "a" * 64,
        "render_recipe_sha256": "b" * 64,
        "validator_version": "validator-v1",
        "stem": "paper",
    }
    nonce = "1" * 32
    identity = render_artifact_identity(
        kind="pdf-render", payload=payload, nonce=nonce,
    )
    created = allocate_artifact_dir(
        tmp_path,
        identity,
        kind="pdf-render",
        stem="paper",
        payload=payload,
        nonce=nonce,
        allow_legacy=False,
    )
    assert created.disposition == "created"
    adopted = allocate_artifact_dir(
        tmp_path,
        identity,
        kind="pdf-render",
        stem="paper",
        payload=payload,
        nonce=nonce,
        allow_legacy=False,
    )
    assert adopted.path == created.path
    assert adopted.disposition == "adopted"
    resolved = resolve_artifact_dir(
        tmp_path,
        created.path,
        expected_identity=identity,
        kind="pdf-render",
        stem="paper",
        payload=payload,
        nonce=nonce,
        allow_legacy=False,
    )
    assert resolved.identity == identity
    with pytest.raises(ArtifactIdError, match="payload differs"):
        resolve_artifact_dir(
            tmp_path,
            created.path,
            kind="pdf-render",
            payload={**payload, "stem": "other"},
            nonce=nonce,
        )
    with pytest.raises(ArtifactIdError, match="payload differs"):
        resolve_artifact_dir(
            tmp_path,
            created.path,
            kind="pdf-render",
            payload=payload,
            nonce="2" * 32,
        )


def test_empty_crash_residue_grows_prefix_without_adoption(
    tmp_path: Path,
) -> None:
    identity = "1" * 64
    (tmp_path / identity[:12]).mkdir()
    allocation = allocate_artifact_dir(
        tmp_path, identity, kind="checkpoint",
    )
    assert allocation.prefix_length == 16
    assert allocation.disposition == "created"


def test_invalid_existing_receipt_fails_instead_of_growing_prefix(
    tmp_path: Path,
) -> None:
    identity = "2" * 64
    candidate = tmp_path / identity[:12]
    candidate.mkdir()
    (candidate / ARTIFACT_ID_RECEIPT_NAME).write_text(
        "{not-json", encoding="utf-8",
    )
    with pytest.raises(ArtifactIdError, match="unreadable"):
        allocate_artifact_dir(tmp_path, identity, kind="checkpoint")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "wrong", "version"),
        ("kind", "other", "kind"),
        ("display_id", "0" * 12, "directory and receipt"),
        ("stem", "unexpected", "directory and receipt"),
    ],
)
def test_resolver_rejects_malformed_receipt_fields(
    tmp_path: Path, field: str, value: object, match: str,
) -> None:
    allocation = allocate_artifact_dir(
        tmp_path, "3" * 64, kind="checkpoint",
    )
    receipt_path = allocation.path / ARTIFACT_ID_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ArtifactIdError, match=match):
        resolve_artifact_dir(
            tmp_path,
            allocation.path,
            expected_identity="3" * 64,
            kind="checkpoint",
        )


def test_resolver_rejects_extra_keys_and_oversized_receipt(
    tmp_path: Path,
) -> None:
    allocation = allocate_artifact_dir(
        tmp_path, "4" * 64, kind="checkpoint",
    )
    receipt_path = allocation.path / ARTIFACT_ID_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["extra"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ArtifactIdError, match="shape"):
        resolve_artifact_dir(
            tmp_path, allocation.path, kind="checkpoint",
        )
    receipt_path.write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(ArtifactIdError, match="oversized"):
        resolve_artifact_dir(
            tmp_path, allocation.path, kind="checkpoint",
        )


def test_symlinked_root_candidate_and_receipt_are_rejected(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ArtifactIdError, match="symbolic link"):
        allocate_artifact_dir(
            linked_root, "5" * 64, kind="checkpoint",
        )
    candidate = real_root / ("6" * 12)
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactIdError, match="candidate"):
        allocate_artifact_dir(
            real_root, "6" * 64, kind="checkpoint",
        )
    allocation = allocate_artifact_dir(
        real_root, "7" * 64, kind="checkpoint",
    )
    receipt = allocation.path / ARTIFACT_ID_RECEIPT_NAME
    saved = tmp_path / "saved-receipt.json"
    receipt.replace(saved)
    receipt.symlink_to(saved)
    with pytest.raises(ArtifactIdError, match="unreadable"):
        resolve_artifact_dir(
            real_root, allocation.path, kind="checkpoint",
        )


def test_legacy_directory_is_checkpoint_only(tmp_path: Path) -> None:
    identity = "8" * 64
    legacy = tmp_path / identity
    legacy.mkdir()
    with pytest.raises(ArtifactIdError, match="missing"):
        resolve_artifact_dir(
            tmp_path,
            legacy,
            expected_identity=identity,
            kind="pdf-render",
        )
    with pytest.raises(ArtifactIdError, match="payload and nonce"):
        allocate_artifact_dir(
            tmp_path,
            identity,
            kind="pdf-render",
            allow_legacy=False,
        )


def test_full_prefix_exhaustion_is_bounded(tmp_path: Path) -> None:
    identity = "9" * 64
    for length in (*range(12, 64, 4), 64):
        (tmp_path / identity[:length]).mkdir()
    with pytest.raises(ArtifactIdError, match="exhausted"):
        allocate_artifact_dir(
            tmp_path,
            identity,
            kind="checkpoint",
            allow_legacy=False,
        )


def test_new_root_directory_receipt_and_lock_modes_ignore_umask(
    tmp_path: Path,
) -> None:
    root = tmp_path / "modes"
    previous = os.umask(0o777)
    try:
        allocation = allocate_artifact_dir(
            root, "a1" * 32, kind="checkpoint",
        )
    finally:
        os.umask(previous)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(allocation.path.stat().st_mode) == 0o700
    assert stat.S_IMODE(
        (allocation.path / ARTIFACT_ID_RECEIPT_NAME).stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        (root / ".artifact-ids.lock").stat().st_mode
    ) == 0o600


def test_process_concurrent_allocation_uses_one_directory(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    identity = "b1" * 32
    root = tmp_path / "fresh-concurrent-root"
    processes = [
        context.Process(
            target=_process_allocate,
            args=(str(root), identity, queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=5) for _ in processes]
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    assert len({path for path, _ in results}) == 1
    assert sorted(disposition for _, disposition in results) == [
        "adopted",
        "created",
    ]


@pytest.mark.parametrize(
    "stem", ["", ".", "..", "/absolute", "a/b", r"a\b"],
)
def test_stem_must_be_one_nonempty_basename(
    tmp_path: Path, stem: str,
) -> None:
    with pytest.raises(ArtifactIdError, match="stem"):
        allocate_artifact_dir(
            tmp_path, "c1" * 32, kind="render", stem=stem,
        )


def test_resolver_rejects_parent_traversal_after_resolution(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    with pytest.raises(ArtifactIdError, match="escapes"):
        resolve_artifact_dir(tmp_path, tmp_path / ".." / outside.name)


def test_pdf_render_payload_is_exact_and_required(tmp_path: Path) -> None:
    base = {
        "content_sha256": "d1" * 32,
        "render_recipe_sha256": "e1" * 32,
        "validator_version": "validator-v1",
        "stem": "paper",
    }
    with pytest.raises(ArtifactIdError, match="shape"):
        render_artifact_identity(
            kind="pdf-render",
            payload={**base, "extra": True},
            nonce="1" * 32,
        )
    with pytest.raises(ArtifactIdError, match="recipe"):
        render_artifact_identity(
            kind="pdf-render",
            payload={**base, "render_recipe_sha256": "bad"},
            nonce="1" * 32,
        )
    identity = render_artifact_identity(
        kind="pdf-render", payload=base, nonce="1" * 32,
    )
    with pytest.raises(ArtifactIdError, match="stem identity differs"):
        allocate_artifact_dir(
            tmp_path,
            identity,
            kind="pdf-render",
            stem="other",
            payload=base,
            nonce="1" * 32,
        )


def test_checkpoint_receipt_rejects_render_fields(tmp_path: Path) -> None:
    allocation = allocate_artifact_dir(
        tmp_path, "f1" * 32, kind="checkpoint",
    )
    path = allocation.path / ARTIFACT_ID_RECEIPT_NAME
    receipt = json.loads(path.read_text())
    receipt.update({"payload": {}, "nonce": "1" * 32})
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ArtifactIdError, match="shape"):
        resolve_artifact_dir(
            tmp_path, allocation.path, kind="checkpoint",
        )


def test_bad_full64_receipt_never_falls_back_to_legacy(
    tmp_path: Path,
) -> None:
    identity = "a2" * 32
    full = tmp_path / identity
    full.mkdir()
    (full / ARTIFACT_ID_RECEIPT_NAME).write_text(
        "{bad", encoding="utf-8",
    )
    with pytest.raises(ArtifactIdError, match="unreadable"):
        allocate_artifact_dir(
            tmp_path, identity, kind="checkpoint",
        )


def test_receipt_hash_is_from_the_validated_bytes(tmp_path: Path) -> None:
    allocation = allocate_artifact_dir(
        tmp_path, "b2" * 32, kind="checkpoint",
    )
    raw = allocation.receipt_path.read_bytes()
    assert allocation.receipt_sha256 == hashlib.sha256(raw).hexdigest()
    resolved = resolve_artifact_dir(
        tmp_path, allocation.path, kind="checkpoint",
    )
    assert resolved.receipt_sha256 == hashlib.sha256(raw).hexdigest()


def _alias_value(identity: str, legacy: Path) -> dict[str, object]:
    return {
        "schema_version": "arc.companion.checkpoint-alias.v1",
        "kind": "workers-to-total-concurrency-budget",
        "alias_identity": identity,
        "legacy_fingerprint": legacy.name,
        "content_fingerprint": identity,
        "legacy_checkpoint_dir": str(legacy),
        "legacy_workers_per_lane": 2049,
    }


def test_alias_receipt_create_adopt_conflict_and_read(
    tmp_path: Path,
) -> None:
    identity = "c2" * 32
    legacy = tmp_path / ("d2" * 32)
    legacy.mkdir()
    value = _alias_value(identity, legacy)
    created = ensure_artifact_alias_receipt(
        tmp_path, identity, value,
    )
    assert created.disposition == "created"
    assert "created_at" not in created.value
    adopted = ensure_artifact_alias_receipt(
        tmp_path, identity, value,
    )
    assert adopted.disposition == "adopted"
    assert adopted.sha256 == created.sha256
    assert read_artifact_alias_receipt(
        tmp_path, identity,
    ).sha256 == created.sha256
    with pytest.raises(ArtifactIdError, match="conflicts"):
        ensure_artifact_alias_receipt(
            tmp_path,
            identity,
            {**value, "legacy_workers_per_lane": 2050},
        )


def test_alias_symlink_and_oversize_are_rejected(tmp_path: Path) -> None:
    identity = "e2" * 32
    root = tmp_path / "root"
    root.mkdir()
    aliases = root / "aliases"
    aliases.mkdir()
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    path = aliases / f"{identity}.json"
    path.symlink_to(external)
    with pytest.raises(ArtifactIdError, match="unreadable"):
        read_artifact_alias_receipt(root, identity)
    path.unlink()
    path.write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(ArtifactIdError, match="oversized"):
        read_artifact_alias_receipt(root, identity)


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_invalid_allocation_lock_is_rejected(
    tmp_path: Path, kind: str,
) -> None:
    root = tmp_path / kind
    root.mkdir()
    lock = root / ".artifact-ids.lock"
    if kind == "symlink":
        external = tmp_path / "external-lock"
        external.write_text("", encoding="utf-8")
        lock.symlink_to(external)
    else:
        lock.mkdir()
    with pytest.raises(ArtifactIdError, match="lock"):
        allocate_artifact_dir(
            root, "f2" * 32, kind="checkpoint",
        )


def test_receipt_temp_is_cleaned_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arc_companion.artifact_ids as artifact_ids

    original = artifact_ids.os.replace

    def fail_replace(source: object, target: object) -> None:
        if str(target).endswith(ARTIFACT_ID_RECEIPT_NAME):
            raise OSError("injected replace failure")
        original(source, target)

    monkeypatch.setattr(artifact_ids.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        allocate_artifact_dir(
            tmp_path, "a3" * 32, kind="checkpoint",
        )
    assert not list(tmp_path.glob("a3a3a3a3a3a3/.directory-identity.json.*"))


def test_process_collision_grows_prefix_for_different_identities(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    shared = "1234567890ab"
    identities = [
        _identity(shared + "aaaa", "1"),
        _identity(shared + "bbbb", "2"),
    ]
    processes = [
        context.Process(
            target=_process_allocate,
            args=(str(tmp_path), identity, queue),
        )
        for identity in identities
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=5) for _ in processes]
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    assert sorted(len(Path(path).name) for path, _ in results) == [12, 16]


def test_process_collision_namespace_is_shared_across_stems(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    shared = "abcdef123456"
    requests = [
        (_identity(shared + "aaaa", "1"), "paper-one"),
        (_identity(shared + "bbbb", "2"), "paper-two"),
    ]
    processes = [
        context.Process(
            target=_process_allocate_stem,
            args=(str(tmp_path), identity, stem, queue),
        )
        for identity, stem in requests
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=5) for _ in processes]
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    assert sorted(prefix for _path, prefix in results) == [12, 16]


def test_existing_legacy_permissions_are_not_changed(tmp_path: Path) -> None:
    identity = "b3" * 32
    os.chmod(tmp_path, 0o755)
    legacy = tmp_path / identity
    legacy.mkdir(mode=0o755)
    allocate_artifact_dir(
        tmp_path, identity, kind="checkpoint",
    )
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o755
    assert stat.S_IMODE(legacy.stat().st_mode) == 0o755
