"""Compatibility re-export of ARC's shared no-follow control reader."""

from arc_llm.secure_io import (  # noqa: F401
    SecureReadError,
    read_bounded_file,
    read_bounded_json,
    safe_relative_path,
)

__all__ = [
    "SecureReadError", "read_bounded_file", "read_bounded_json",
    "safe_relative_path",
]
