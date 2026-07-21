"""Build source-faithful, chapter-aware companions for papers and books."""

from .pipeline import BuildOptions, build_companion

__all__ = ["BuildOptions", "build_companion"]
__version__ = "1.0.0"
