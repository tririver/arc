"""Protocol-neutral persistent jobs for ARC command-line tools."""

from .jobs import (
    ALLOWED_COMMANDS,
    TERMINAL_STATUSES,
    JobCancelled,
    JobManager,
    JobPaths,
    append_event,
    is_cancel_requested,
    read_job,
    read_json,
    record_progress,
    submission_lock,
    write_json,
)

__all__ = [
    "ALLOWED_COMMANDS",
    "TERMINAL_STATUSES",
    "JobCancelled",
    "JobManager",
    "JobPaths",
    "append_event",
    "is_cancel_requested",
    "read_job",
    "read_json",
    "record_progress",
    "submission_lock",
    "write_json",
]
__version__ = "1.0.0"
