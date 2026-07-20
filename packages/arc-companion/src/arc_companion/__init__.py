"""Generate source-faithful annotated companions for arXiv papers."""

from .pipeline import BuildOptions, build_companion

__all__ = ["BuildOptions", "build_companion"]
__version__ = "1.0.0"
