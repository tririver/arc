"""Protocol-neutral persistent jobs for ARC command-line tools."""

from .jobs import (
    ALLOWED_COMMANDS,
    TERMINAL_STATUSES,
    JobCancelled,
    JobManager,
    JobPaths,
)

__all__ = [
    "ALLOWED_COMMANDS",
    "TERMINAL_STATUSES",
    "JobCancelled",
    "JobManager",
    "JobPaths",
]
__version__ = "1.0.0"
