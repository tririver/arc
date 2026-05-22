# Arc Paper Package

`arc-paper` is the single-paper information package. Use it for metadata,
INSPIRE references/citers, ar5iv full text, table of contents, section lookup,
equation context, LLM paper summaries, and paper-summary batches.

All single-paper operations in higher-level tools should go through
`arc-paper`.

## Deterministic CLI

Phase 1: Fetch or read cached paper data.
Step 1: Use `--json` for agent-readable result envelopes.
Step 2: Prefer non-refreshing reads unless the user asks to refetch.

```bash
arc-paper get-title 0911.3380 --json
arc-paper get-abstract 0911.3380 --json
arc-paper get-authors 0911.3380 --json
arc-paper get-metadata 0911.3380 --json
arc-paper get-references 0911.3380 --json
arc-paper get-citers 0911.3380 --limit 1000 --sort mostrecent --json
arc-paper get-citers 0911.3380 --limit 1000 --sort mostcited --json
arc-paper get-citer-count 0911.3380 --json
arc-paper get-toc 0911.3380 --json
arc-paper get-section 0911.3380 --section S2 --json
arc-paper get-equation-context 0911.3380 --query "dot theta" --json
```

Phase 2: Resolve missing sections.
Step 1: If `get-section` cannot find the requested section, read the returned
`toc`.
Step 2: Retry with a valid section id, number, or heading from that `toc`.

## LLM Summary CLI

Phase 1: Try the cached-or-generate command.
Step 1: Run:

```bash
arc-paper llm-summary 0911.3380 --json
```

Step 2: If it returns a summary, use it.
Step 3: If it returns `status: "needs_llm"`, use the manual fallback below.

Phase 2: Explicitly generate or refresh when needed.
Step 1: Use this when the user asks to regenerate, choose a provider, or bypass
an old cache:

```bash
arc-paper llm-generate-summary 0911.3380 --provider auto --json
arc-paper llm-generate-summary 0911.3380 --provider codex-cli --model gpt-5.4-mini --json
```

Step 2: Use `--refresh` only when the user wants fresh source data or a forced
new summary.

Summary generation first writes section summaries, then synthesizes the final
paper summary from title, abstract, TOC, and section summaries. References are
intentionally omitted from the summary input pack.

## Manual Summary Fallback

Use this only when no runnable host LLM provider is available and the command
returns `status: "needs_llm"`.

Phase 1: Generate schema-valid JSON.
Step 1: Use `llm_task.system_prompt`, `llm_task.user_prompt`,
`llm_task.input_pack`, and `llm_task.output_schema`.
Step 2: Return JSON only, conforming to `output_schema`.

Phase 2: Store the summary.
Step 1: Pipe the generated JSON into:

```bash
arc-paper store-llm-summary 0911.3380 --summary-json - --json
```

## Batch Summary CLI

Use summary batches for more than 10 papers. Do not run one interactive LLM
step per paper.

Phase 1: Create and prefetch.
Step 1: Put one paper id per line in a text file.
Step 2: Run:

```bash
arc-paper summary-batch create papers.txt --name qft-ideas --json
arc-paper summary-batch prefetch qft-ideas --workers 8 --json
arc-paper summary-batch status qft-ideas --json
```

Phase 2: Generate in controlled chunks.
Step 1: Review the first chunk before launching the full batch.

```bash
arc-paper summary-batch run qft-ideas --provider auto --concurrency 2 --max-items 10 --json
arc-paper summary-batch run qft-ideas --provider auto --concurrency 2 --json
```

Step 2: Export completed summaries.

```bash
arc-paper summary-batch export qft-ideas --format jsonl --output summaries.jsonl --json
```

Step 3: Retry failures only after checking the error cause.

```bash
arc-paper summary-batch retry-failed qft-ideas --json
```

## MCP Tools

Read `references/package-manuals/arc-mcp.md` before using MCP.

Paper MCP tools:

```text
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

For LLM paper summaries through MCP, use `background=true` for slow or massive
launches and then follow `references/package-manuals/arc-mcp.md`.

## Cache Notes

Default checkout cache:

```text
/arc-dev/cache/arc-paper/
```

INSPIRE citer lists are cached for one month. Cached citer records include
title, abstract, authors, identifiers, year, and citation count when INSPIRE
returns those fields.

Check cache/provider state:

```bash
arc-paper doctor cache 0911.3380 --json
arc-paper doctor host --json
arc-paper doctor provider --json
```
