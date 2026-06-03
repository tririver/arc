# Arc MCP Package

`arc-mcp` exposes ARC paper/domain tools to MCP clients and owns persistent
background job management. Use it for MCP tool names, timeout behavior,
background jobs, job status, and CLI job watching.

## Tool Groups

Paper tools:

```text
extract_paper_ids
paper_ids_safe_dir_name
llm_infer_main_references
get_title
get_abstract
get_authors
get_metadata
get_references
get_citers
get_citer_count
get_toc
get_section
search_full_text
get_equation_context
llm_get_summary
llm_generate_summary
store_llm_summary
summary_batch_create
summary_batch_prefetch
llm_summary_batch_run
summary_batch_status
summary_batch_export
summary_batch_retry_failed
```

Domain tools:

```text
domain_status
domain_get_summary
domain_get_graph
llm_domain_build
llm_domain_get_summary
llm_domain_get_graph
```

Typeset tools:

```text
md2pdf
translate
batch_translate
```

Job and doctor tools:

```text
job_status
job_result
list_jobs
cancel_job
doctor_host
doctor_provider
doctor_cache
```

Tools that may invoke a host LLM provider use the `llm_` prefix.
`md2pdf`, `translate`, and `batch_translate` always start background jobs and
return immediately. Translation defaults to Chinese and low-tier LLMs unless
the user asks for another language, locale, model, or quality pass.

## Markdown Report Export

When a workflow writes or copies a user-facing Markdown report to
`<project-dir>/`, PDF export is a completion gate. The Markdown deliverable is
not complete until the agent has either called the ARC MCP `md2pdf` tool for
that project-level Markdown file, or recorded a `WARNING:` with the exact reason
PDF export could not be started.

Use the host's MCP tool call for `md2pdf`; do not type `md2pdf(...)` as a shell
command. The examples below show the argument shape. `md2pdf` starts a
background PDF job; record the returned job id if present and do not wait before
continuing unless the user explicitly asks.

If the MCP tool is unavailable, use `arc-mcp md2pdf <project-dir>/report.md
--json` to start the same ARC MCP background job through the CLI. This is only a
job-launch fallback, not a TeX/Pandoc debugging step. If MCP and CLI launch both
fail, do not silently skip export and do not run unrelated TeX build commands.
Record `WARNING: PDF export not started: <reason>` in the workflow log,
self-reflection entry, work-note journal, or final response as appropriate for
the workflow.
Do not debug or fix PDF generation unless the user explicitly asks; continue
the owning ARC workflow after reporting the warning.

## Background Jobs

Use `background=true` for slow tools, large launches, and anything likely to
exceed the MCP client timeout.

### Phase 1: Launch the MCP job.
Step 1: Call `md2pdf` directly, or call the relevant `llm_` tool with
`background=true`.
Step 2: Capture the returned `job_id`.

Examples:

```text
md2pdf(input="<project-dir>/work-note.md")
arc-mcp md2pdf <project-dir>/work-note.md --json
translate(input="<project-dir>/work-note.md")
batch_translate(project_dir="<project-dir>")
llm_infer_main_references(text="<user-intent>", background=true)
llm_generate_summary(paper_id="<seed-paper>", provider="auto", background=true)
llm_domain_build(seed_paper="<seed-paper>", intent="<user-intent>", background=true)
llm_summary_batch_run(name="<batch-name>", provider="auto", concurrency=2, background=true)
```

### Phase 2: Watch with CLI.
Step 1: Immediately run the returned `next.cli_command`, for example:

```bash
arc-mcp watch <job_id> --json
```

Plugin or Codex shells may not have `arc-mcp` on `PATH`; `next.cli_command`
may use an absolute runtime command. Use it exactly when present.

Step 2: Use the watcher output as the final result when it completes.

### Phase 3: Fallback if CLI is unavailable.
Step 1: Poll `job_status(job_id)`.
Step 2: When status is `done`, call `job_result(job_id)`.
Step 3: If status is `needs_llm`, follow the package-specific manual fallback.

Do not call `cancel_job` unless the user explicitly asks.

## Job CLI

```bash
arc-mcp root --json
arc-mcp list --json
arc-mcp status <job_id> --json
arc-mcp result <job_id> --json
arc-mcp watch <job_id> --json
arc-mcp watch <job_id> --progress-jsonl
arc-mcp cancel <job_id> --json
```

`watch --json` blocks until a terminal result. `watch --progress-jsonl` streams
progress events.

## Timeout Behavior

MCP jobs do not push completion notifications. Clients should poll or use the
CLI watcher.

Inline wait time is resolved in this order:

1. `ARC_MCP_INLINE_WAIT_SEC`
2. `ARC_MCP_TOOL_TIMEOUT_SEC - ARC_MCP_BACKGROUND_MARGIN_SEC`
3. Codex `tool_timeout_sec` from `~/.codex/config.toml`
4. 90 seconds

With `background=true`, the inline wait is zero.

## Job Cache

Discover the active MCP job cache path:

```text
doctor_cache
```

Job state is persisted under:

```text
cache/arc-mcp/jobs/<job_id>/
```

MCP tools and the `arc-mcp` job CLI read the same persisted job files. ETA appears
after enough matching jobs have completed.
