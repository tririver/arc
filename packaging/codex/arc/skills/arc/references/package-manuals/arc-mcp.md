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

## Background Jobs

Use `background=true` for slow tools, large launches, and anything likely to
exceed the MCP client timeout.

### Phase 1: Launch the MCP job.
Step 1: Call the relevant `llm_` tool with `background=true`.
Step 2: Capture the returned `job_id`.

Examples:

```text
llm_infer_main_references(text="<user-intent>", background=true)
llm_generate_summary(paper_id="<seed-paper>", provider="auto", background=true)
llm_domain_build(seed_paper="<seed-paper>", intent="<user-intent>", background=true)
llm_summary_batch_run(name="<batch-name>", provider="auto", concurrency=2, background=true)
```

### Phase 2: Watch with CLI.
Step 1: Immediately run:

```bash
arc-mcp jobs watch <job_id> --json
```

Step 2: Use the watcher output as the final result when it completes.

### Phase 3: Fallback if CLI is unavailable.
Step 1: Poll `job_status(job_id)`.
Step 2: When status is `done`, call `job_result(job_id)`.
Step 3: If status is `needs_llm`, follow the package-specific manual fallback.

Do not call `cancel_job` unless the user explicitly asks.

## Job CLI

```bash
arc-mcp jobs root --json
arc-mcp jobs list --json
arc-mcp jobs status <job_id> --json
arc-mcp jobs result <job_id> --json
arc-mcp jobs watch <job_id> --json
arc-mcp jobs watch <job_id> --progress-jsonl
arc-mcp jobs cancel <job_id> --json
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

Default checkout cache:

```text
/arc-dev/cache/arc-mcp/
```

Job state is persisted under:

```text
cache/arc-mcp/jobs/<job_id>/
```

MCP tools and `arc-mcp jobs ...` read the same persisted job files. ETA appears
after enough matching jobs have completed.
