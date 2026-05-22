# Arc Paper Reference

## Single Paper

Use the CLI with `--json`; outputs are result envelopes suitable for MCP:

```bash
arc-paper get-title arXiv:0911.3380 --json
arc-paper get-metadata arXiv:0911.3380 --json
arc-paper get-section arXiv:0911.3380 --section S2 --json
```

If `get-section` cannot find the requested section, it returns an error
envelope plus `toc`. Use the returned `toc` to choose a valid section.

For citing-paper lists:

```bash
arc-paper get-citers arXiv:0911.3380 --limit 1000 --sort mostrecent --json
arc-paper get-citers arXiv:0911.3380 --limit 1000 --sort mostcited --json
```

INSPIRE citer responses are cached for one month and include title, abstract,
authors, identifiers, year, and citation count when INSPIRE returns those
fields.

## LLM Summary

Get or build the summary:

```bash
arc-paper llm-summary arXiv:0911.3380 --json
```

This command first checks the cache. If the summary is missing and a host LLM
provider is available, it generates and caches the summary automatically.
ARC uses fast summary defaults unless overridden: `gpt-5.4-mini` for Codex and
`haiku` for Claude Code. Summary generation first creates short section
summaries sequentially, then synthesizes the paper-level summary from title,
abstract, TOC, and section summaries. The final JSON keeps `toc` as navigation
metadata and stores the canonical per-section content under
`section_summaries`. References are intentionally omitted from the summary input
pack.

When using MCP, use `llm_get_summary` or `llm_generate_summary`. These tools may
call the host LLM provider, so they wait only until the MCP deadline margin. If
the result is not ready, they return a `job_id`. Pass `background=true` to
schedule the job and return the `job_id` immediately.

Phase 1: Start or reuse the result.
Step 1: Call `llm_get_summary`.
Step 2: If it returns a result, use it.
Step 3: For massive or slow launches, call `llm_get_summary` or
`llm_generate_summary` with `background=true`.
Step 4: If it returns `status: "job_running"` with a `job_id`, run:

```bash
arc-mcp jobs watch <job_id> --json
```

Phase 2: Read the result.
Step 1: Use the CLI watcher output as the final result.
Step 2: If the CLI watcher is unavailable, poll `job_status` and call
`job_result` when status is `done`.
Step 3: If status is `needs_llm`, use the manual fallback below.
Step 4: Do not call `cancel_job` unless the user explicitly asks.

ARC stores MCP job state under `cache/arc-mcp/jobs/`. The CLI watcher and MCP
tools read the same persisted job files.

If the result has `status: "needs_llm"`, no runnable provider was available.
Use the manual fallback:

1. Use `llm_task.system_prompt`, `llm_task.user_prompt`, `llm_task.input_pack`,
   and `llm_task.output_schema`.
2. Generate JSON only, conforming to `output_schema`.
3. Call:

```bash
arc-paper store-llm-summary arXiv:0911.3380 --summary-json - --json
```

For explicit generation through the host CLI or provider override:

```bash
arc-paper llm-generate-summary arXiv:0911.3380 --provider auto --json
```

`--provider auto` uses `ARC_LLM_PROVIDER` first, then `ARC_AGENT_HOST`, then
parent-process detection. Plugin wrappers set these env vars automatically.

## Batch Summary

For more than 10 papers:

```bash
arc-paper summary-batch create papers.txt --name qft-ideas --json
arc-paper summary-batch prefetch qft-ideas --workers 8 --json
arc-paper summary-batch run qft-ideas --provider auto --concurrency 2 --max-items 10 --json
arc-paper summary-batch status qft-ideas --json
```

Review the first 10 summaries before running the full batch:

```bash
arc-paper summary-batch run qft-ideas --provider auto --concurrency 2 --json
arc-paper summary-batch export qft-ideas --format jsonl --output summaries.jsonl --json
```

Retry failures:

```bash
arc-paper summary-batch retry-failed qft-ideas --json
```

## Troubleshooting

Check host and provider detection:

```bash
arc-paper doctor host --json
arc-paper doctor provider --json
```

Expected plugin env:

```text
Codex: ARC_AGENT_HOST=codex, ARC_LLM_PROVIDER=codex-cli
Claude Code: ARC_AGENT_HOST=claude-code, ARC_LLM_PROVIDER=claude-cli
```
