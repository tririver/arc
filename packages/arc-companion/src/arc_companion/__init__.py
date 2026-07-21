"""Build source-faithful, chapter-aware companions for papers and books."""

from pathlib import Path
from typing import Any


def __getattr__(name: str) -> Any:
    """Keep lightweight render/reader imports independent of the LLM pipeline."""
    if name == "BuildOptions":
        from .pipeline import BuildOptions

        return BuildOptions
    if name == "build_companion":
        from .pipeline import build_companion

        return build_companion
    raise AttributeError(name)


def build_reader_snapshot(
    project_dir: Path,
    *,
    state: dict[str, Any] | None = None,
    final_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .web import build_reader_snapshot as implementation

    return implementation(
        project_dir, state=state, final_overrides=final_overrides
    )


def publish_reader(
    project_dir: Path,
    *,
    snapshot: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    final_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .web import publish_reader as implementation

    return implementation(
        project_dir,
        snapshot=snapshot,
        state=state,
        final_overrides=final_overrides,
    )


def validate_reader_project(
    project_dir: Path,
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .web import validate_reader_project as implementation

    return implementation(project_dir, state=state)


__all__ = [
    "BuildOptions",
    "build_companion",
    "build_reader_snapshot",
    "publish_reader",
    "validate_reader_project",
]
__version__ = "1.0.0"
