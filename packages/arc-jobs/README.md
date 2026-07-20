# arc-jobs

`arc-jobs` provides protocol-neutral persistent execution for ARC command-line
tools. It stores job state and output on disk and exposes `submit`, `list`,
`status`, `watch`, `result`, and `cancel` commands.

Only ARC console scripts installed in the same Python runtime are accepted.
Commands are passed as an argument vector and are never evaluated by a shell.

```bash
arc-jobs submit --json -- arc-paper get-title 0911.3380 --json
arc-jobs watch JOB_ID --json
```

Successful submit, status, cancel, and list operations return `ok: true`.
Failed/cancelled job status and result commands use a nonzero process exit code
so shell callers do not mistake a terminal command failure for success.

Set `ARC_JOBS_CACHE` to override the job-state directory and
`ARC_JOBS_WORKER_MODE=thread` only when embedding a cooperative in-process
runner. Normal CLI jobs use isolated worker processes.

Job directories are private (`0700`) and state, result, event, lock, log, and
SQLite files are private (`0600`). Recovery leases use both the worker PID and
the operating-system process start identity, rather than expiring a healthy
silent worker by heartbeat age. A process job is restarted only when its ARC
command was never launched; loss of a worker after command launch terminates
the orphaned process group and records a terminal failure.
