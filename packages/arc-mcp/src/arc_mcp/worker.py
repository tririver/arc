"""Thin module entry point for the protocol-neutral ARC job worker."""

from arc_jobs.worker import main, run_job

__all__ = ["main", "run_job"]


if __name__ == "__main__":
    raise SystemExit(main())
