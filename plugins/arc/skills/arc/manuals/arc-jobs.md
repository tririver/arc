# ARC Jobs

`arc-jobs` is ARC's protocol-neutral persistent job runner. Use it for slow
CLI work, concurrency, status inspection, cancellation, and report export.
It does not require or start MCP.

## Submit And Watch

Submit only an ARC CLI argv. `arc-jobs` does not accept shell command strings
and resolves the executable from the same isolated ARC runtime.

```bash
arc-jobs submit --job-type <type> --cwd <project-dir> --json -- \
  arc-domain llm-build <seed-paper> --intent "<intent>" --json
```

The accepted response contains `job_id`, `status=job_running`, and
`ok=true` plus `next.cli_command`. Submit independent jobs before watching them so they can
run concurrently.

```bash
arc-jobs list --json
arc-jobs status <job-id> --json
arc-jobs watch <job-id> --progress-jsonl --json
arc-jobs watch <job-id> --until-review --after-review-sequence 0 --json
arc-jobs result <job-id> --json
arc-jobs cancel <job-id> --json
```

Terminal statuses include successful `done`, `completed`, `degraded`,
`stopped`, and `needs_llm`, plus unsuccessful `failed` and `cancelled`.
`degraded` preserves usable work, failure counts, and warnings; it is not
equivalent to a clean completion. A command is successful only when its process exit status is zero
and the returned JSON does not report `ok: false`. Do not cancel a job merely
because it is slow.
Status and cancellation calls use `ok=true` when the control operation itself
succeeds; `arc-jobs status` still exits nonzero for a failed or cancelled job,
and `result` carries the command's success or failure envelope.

`ARC_JOBS_DIR` overrides the persistent job root; legacy `ARC_JOBS_CACHE`
remains an earlier-layout override, and otherwise jobs use `ARC_HOME/jobs`.
Submission snapshots only the allowlisted ARC runtime, cache,
host, and idle-timeout context. It never persists tokens, API keys, or arbitrary
environment variables. This setting is independent of optional MCP
configuration.

Removed `ARC_LLM_TIMEOUT_SECONDS`, `ARC_CODEX_TIMEOUT_SECONDS`,
`ARC_CLAUDE_TIMEOUT_SECONDS`, and `ARC_KIMI_TIMEOUT_SECONDS` values make job
submission fail before persistence or worker launch. Replace them with the
corresponding `*_IDLE_TIMEOUT_SECONDS` setting; ARC never silently drops an old
total-timeout value from a detached job.

Status includes the latest phase, round, worker counts, job-level review
sequence, the
last substantive excerpt, artifact paths, and validated progress events when the
child CLI supplies them. `watch --progress-jsonl` streams those events without
changing the run. `watch --until-review --after-review-sequence N` returns
successfully after the next `review_due` sequence greater than `N`; returning
does not pause or cancel the job. Provider-local review numbers are retained as
`provider_review_sequence` in events, while `review_sequence` is ARC Jobs' strictly
increasing cursor across all provider calls in the job.

LLM calls have no absolute runtime deadline. They stop only after 1800 seconds
without substantive provider output. Configure `worker_idle_timeout_seconds`,
`--idle-timeout-seconds`, or a documented provider idle-timeout environment
variable when an override is required. Heartbeats, repeated text, and transport
noise do not reset the idle timer.

For a long-running job, set `cursor=0` and run
`arc-jobs watch <job-id> --until-review --after-review-sequence <cursor> --json`.
At each review checkpoint, inspect the latest excerpt and artifacts. When there
is a concrete result, new evidence, a completed step, a reusable artifact, or a
meaningfully narrowed problem, set `cursor` to the returned `review_sequence`
and run the command again. For repeated heartbeats or errors, off-task work, or
output with no substantive progress, run `arc-jobs cancel <job-id> --json`.
Never cancel solely because total runtime is long. A terminal result is returned
normally by watch and ends this loop.

`SIGINT`, `SIGTERM`, and `arc-jobs cancel` request cancellation and terminate
the full provider process group before the job reaches terminal `cancelled`.
An `idle_timeout` is terminal for the current call and does not automatically
start another paid call. Resume only through the owning workflow's explicit
checkpoint/session continuation path.

ARC stores job directories with user-only permissions. Worker recovery uses a
PID plus process-start identity lease, not a time-only heartbeat: a silent live
worker remains valid. ARC retries a lost worker only before its command starts;
after command launch it terminates the orphaned process group and reports a
terminal failure instead of risking duplicate work.

## Markdown Report Export

Start report conversion as a background CLI job:

```bash
arc-jobs submit --job-type md2pdf --cwd <project-dir> --json -- \
  arc-typeset md2pdf <project-dir>/<report>.md --json
```

The report-export gate is satisfied after the job is accepted. Do not wait for
PDF completion unless the owning workflow explicitly requires it. Record the
job id in the current host/run log or next mutable workflow artifact. If job
submission fails, print `WARNING:` with the exact error and continue according
to the owning workflow; do not debug Pandoc or TeX unless the user requested
that work.
