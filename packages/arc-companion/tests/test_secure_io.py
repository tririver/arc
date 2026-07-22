from __future__ import annotations

import os
from pathlib import Path

import pytest

from arc_companion.secure_io import SecureReadError, read_bounded_file
from arc_llm import secure_io as secure_io_module


def test_bounded_reader_rejects_unsafe_names_and_oversize(tmp_path: Path) -> None:
    (tmp_path / "ok.json").write_bytes(b"12345")
    for value in ("../ok.json", "./ok.json", "bad\\name.json", "bad\x00name.json"):
        with pytest.raises(SecureReadError):
            read_bounded_file(tmp_path, value, max_bytes=5, suffixes=(".json",))
    with pytest.raises(SecureReadError, match="suffix"):
        read_bounded_file(tmp_path, "ok.json", max_bytes=5, suffixes=(".txt",))
    with pytest.raises(SecureReadError, match="byte limit"):
        read_bounded_file(tmp_path, "ok.json", max_bytes=4, suffixes=(".json",))


def test_bounded_reader_rejects_symlink_hardlink_and_special_file(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}")
    os.symlink(target, tmp_path / "link.json")
    os.link(target, tmp_path / "hard.json")
    os.mkfifo(tmp_path / "pipe.json")
    for name in ("link.json", "hard.json", "pipe.json"):
        with pytest.raises(SecureReadError):
            read_bounded_file(tmp_path, name, max_bytes=32, suffixes=(".json",))


@pytest.mark.parametrize("cutpoint", ["leaf:after_open", "leaf:after_read"])
def test_bounded_reader_rejects_named_leaf_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cutpoint: str,
) -> None:
    leaf = tmp_path / "control.json"
    leaf.write_bytes(b'{"safe":true}')
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"attacker":true}')
    fired = False

    def swap(point: str) -> None:
        nonlocal fired
        if point == cutpoint and not fired:
            fired = True
            leaf.unlink()
            replacement.rename(leaf)

    monkeypatch.setattr(secure_io_module, "_secure_read_fault", swap)
    with pytest.raises(SecureReadError, match="identity|changed"):
        read_bounded_file(tmp_path, leaf.name, max_bytes=64, suffixes=(".json",))


def test_bounded_reader_rejects_parent_component_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "controls"
    parent.mkdir()
    (parent / "control.json").write_bytes(b'{"safe":true}')
    moved = tmp_path / "moved-controls"
    fired = False

    def swap(point: str) -> None:
        nonlocal fired
        if point == "leaf:after_open" and not fired:
            fired = True
            parent.rename(moved)
            parent.mkdir()
            (parent / "control.json").write_bytes(b'{"attacker":true}')

    monkeypatch.setattr(secure_io_module, "_secure_read_fault", swap)
    with pytest.raises(SecureReadError, match="directory named identity"):
        read_bounded_file(
            tmp_path, "controls/control.json", max_bytes=64,
            suffixes=(".json",),
        )
