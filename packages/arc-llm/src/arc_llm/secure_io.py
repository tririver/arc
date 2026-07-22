from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any


class SecureReadError(RuntimeError):
    """A control artifact could not be read without following mutable links."""


def _secure_read_fault(_cutpoint: str) -> None:
    """Test-only hook for deterministic address-swap checks."""


def safe_relative_path(value: Any, *, suffixes: tuple[str, ...] = ()) -> Path:
    text = str(value or "")
    if not text or "\x00" in text or "\\" in text:
        raise SecureReadError("control path is not a safe relative path")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        raise SecureReadError("control path is not a safe relative path")
    path = Path(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SecureReadError("control path is not a safe relative path")
    if suffixes and not any(path.name.endswith(suffix) for suffix in suffixes):
        raise SecureReadError("control path has an unexpected suffix")
    return path


def read_bounded_file(
    root: Path,
    relative: Path | str,
    *,
    max_bytes: int,
    suffixes: tuple[str, ...] = (),
) -> bytes:
    """Read one regular, singly-linked file beneath *root* without link traversal."""

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    rel = safe_relative_path(relative, suffixes=suffixes)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        file_flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    descriptors: list[int] = []
    directory_identities: list[tuple[int, int]] = []
    try:
        current = os.open(os.fspath(root), directory_flags)
        descriptors.append(current)
        root_stat = os.fstat(current)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise SecureReadError("control root is not a directory")
        directory_identities.append((root_stat.st_dev, root_stat.st_ino))
        for part in rel.parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
            current_stat = os.fstat(current)
            if not stat.S_ISDIR(current_stat.st_mode):
                raise SecureReadError("control path component is not a directory")
            directory_identities.append((current_stat.st_dev, current_stat.st_ino))
        leaf = os.open(rel.parts[-1], file_flags, dir_fd=current)
        descriptors.append(leaf)
        before = os.fstat(leaf)
        if not stat.S_ISREG(before.st_mode):
            raise SecureReadError("control path is not a regular file")
        if before.st_nlink != 1:
            raise SecureReadError("control file must have exactly one link")
        _secure_read_fault("leaf:after_open")
        _verify_directory_chain(
            root, rel.parts[:-1], directory_flags, directory_identities,
        )
        named_before = os.stat(
            rel.parts[-1], dir_fd=current, follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named_before.st_mode)
            or named_before.st_nlink != 1
            or (named_before.st_dev, named_before.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise SecureReadError("control file named identity changed")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(leaf, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise SecureReadError("control file exceeds its byte limit")
        after = os.fstat(leaf)
        _secure_read_fault("leaf:after_read")
        _verify_directory_chain(
            root, rel.parts[:-1], directory_flags, directory_identities,
        )
        named_after = os.stat(
            rel.parts[-1], dir_fd=current, follow_symlinks=False,
        )
        before_identity = (
            before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
            before.st_size, before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
            after.st_size, after.st_mtime_ns,
        )
        if (
            before_identity != after_identity
            or after.st_size != len(raw)
            or not stat.S_ISREG(named_after.st_mode)
            or named_after.st_nlink != 1
            or (named_after.st_dev, named_after.st_ino)
            != (after.st_dev, after.st_ino)
        ):
            raise SecureReadError("control file changed while reading")
        return raw
    except SecureReadError:
        raise
    except OSError as exc:
        raise SecureReadError("control path is unsafe or unavailable") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_directory_chain(
    root: Path,
    parts: tuple[str, ...],
    directory_flags: int,
    expected: list[tuple[int, int]],
) -> None:
    descriptors: list[int] = []
    try:
        current = os.open(os.fspath(root), directory_flags)
        descriptors.append(current)
        current_stat = os.fstat(current)
        if (current_stat.st_dev, current_stat.st_ino) != expected[0]:
            raise SecureReadError("control root named identity changed")
        for index, part in enumerate(parts, 1):
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
            current_stat = os.fstat(current)
            if (current_stat.st_dev, current_stat.st_ino) != expected[index]:
                raise SecureReadError("control directory named identity changed")
    except SecureReadError:
        raise
    except OSError as exc:
        raise SecureReadError("control directory named identity changed") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def read_bounded_json(
    root: Path,
    relative: Path | str,
    *,
    max_bytes: int,
    suffixes: tuple[str, ...] = (".json",),
) -> Any:
    raw = read_bounded_file(root, relative, max_bytes=max_bytes, suffixes=suffixes)
    try:
        return json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SecureReadError("control file is not valid UTF-8 JSON") from exc
