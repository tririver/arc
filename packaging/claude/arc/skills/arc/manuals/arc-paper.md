# Arc Paper Package

`arc-paper` is the single-paper information package. Use it for metadata,
INSPIRE references/citers, parsed ar5iv JSON, cached full-text search, table of
contents, section lookup, equation context, LLM paper summaries, and
paper-summary batches.

All single-paper operations in higher-level tools should go through
`arc-paper`.

## Deterministic CLI

Use `extract-paper-ids` when the input is natural-language text that may
mention papers in mixed formats.

```bash
arc-paper extract-paper-ids "Compare <paper-a>, <paper-b>, and <doi-paper>" --json
```

It returns normalized identifiers such as `arXiv:<arxiv-id>`,
`inspire:<recid>`, and `doi:<doi-value>`. DOI identifiers are
usable for INSPIRE-backed metadata lookups. DOI spans are removed before bare
arXiv-like IDs are scanned, so DOI suffixes do not create false arXiv IDs.

Use `safe-dir-name` when a workflow needs a stable directory stem for one or
more paper ids. Dots in arXiv and DOI identifiers are preserved; unsafe
separators such as `/` and `:` are replaced with underscores.

```bash
arc-paper safe-dir-name <paper-a> <paper-b> --json
```

This returns a stable directory stem such as `<paper-a-safe>_x_<paper-b-safe>`.

Use `llm-infer-main-references` when the input has no explicit paper id and the
task is to infer the main reference paper from a natural-language research
description. It first runs `extract-paper-ids`; if any ids are found, it returns
them directly without calling an LLM. Otherwise it calls the host LLM with
internet search enabled, then verifies returned candidates through INSPIRE
before returning ids. Query text and returned ids are cached, so repeated calls
with the same stripped input string do not call the LLM again unless `--refresh`
is used.

```bash
arc-paper llm-infer-main-references "<user-intent>" --json
```

### Phase 1: Fetch or read cached paper data.
Step 1: Use `--json` for agent-readable result envelopes.
Step 2: Prefer non-refreshing reads unless the user asks to refetch.

```bash
arc-paper get-title <seed-paper> --json
arc-paper get-abstract <seed-paper> --json
arc-paper get-authors <seed-paper> --json
arc-paper get-metadata <seed-paper> --json
arc-paper get-references <seed-paper> --json
arc-paper get-citers <seed-paper> --limit 1000 --sort mostrecent --json
arc-paper get-citers <seed-paper> --limit 1000 --sort mostcited --json
arc-paper get-citer-count <seed-paper> --json
arc-paper get-toc <seed-paper> --json
arc-paper get-section <seed-paper> --section <section> --json
arc-paper search-full-text <seed-paper> --query "<word-or-phrase>" --context 1 --json
arc-paper get-equation-context <seed-paper> --query "<equation-query>" --json
```

Use `search-full-text` to search cached parsed JSON text. When paper ids are
omitted, it searches all cached parsed papers:

```bash
arc-paper search-full-text --query "<word-or-phrase>" --limit 20 --json
```

It uses `rg` when available and falls back to Python search. Returned hits
include `paper_id`, `snippet`, and `next_steps` with MCP and CLI commands for
retrieving the full section.

### Phase 2: Resolve missing sections.
Step 1: If `get-section` cannot find the requested section, read the returned
`toc`.
Step 2: Retry with a valid section id, number, or heading from that `toc`.

## LLM Summary CLI

### Phase 1: Try the cached-or-generate command.
Step 1: Run:

```bash
arc-paper llm-summary <seed-paper> --json
```

Step 2: If it returns a summary, use it.
Step 3: If it returns `status: "needs_llm"`, use the manual fallback below.

### Phase 2: Explicitly generate or refresh when needed.
Step 1: Use this when the user asks to regenerate, choose a provider, or bypass
an old cache:

```bash
arc-paper llm-generate-summary <seed-paper> --provider auto --json
arc-paper llm-generate-summary <seed-paper> --provider codex-cli --model <model> --json
```

Step 2: Use `--refresh` only when the user wants fresh source data or a forced
new summary.

Summary generation first writes section summaries, then synthesizes the final
paper summary from title, abstract, TOC, and section summaries. References are
intentionally omitted from the summary input pack.

## Manual Summary Fallback

Use this only when no runnable host LLM provider is available and the command
returns `status: "needs_llm"`.

### Phase 1: Generate schema-valid JSON.
Step 1: Use `llm_task.system_prompt`, `llm_task.user_prompt`,
`llm_task.input_pack`, and `llm_task.output_schema`.
Step 2: Return JSON only, conforming to `output_schema`.

### Phase 2: Store the summary.
Step 1: Pipe the generated JSON into:

```bash
arc-paper store-llm-summary <seed-paper> --summary-json - --json
```

## Batch Summary CLI

Use summary batches for more than 10 papers. Do not run one interactive LLM
step per paper.

### Phase 1: Create and prefetch.
Step 1: Put one paper id per line in a text file.
Step 2: Run:

```bash
arc-paper summary-batch create <papers-file> --name <batch-name> --json
arc-paper summary-batch prefetch <batch-name> --workers 8 --json
arc-paper summary-batch status <batch-name> --json
```

### Phase 2: Generate in controlled chunks.
Step 1: Review the first chunk before launching the full batch.

```bash
arc-paper summary-batch run <batch-name> --provider auto --concurrency 2 --max-items 10 --json
arc-paper summary-batch run <batch-name> --provider auto --concurrency 2 --json
```

Step 2: Export completed summaries.

```bash
arc-paper summary-batch export <batch-name> --format jsonl --output <summaries-file> --json
```

Step 3: Retry failures only after checking the error cause.

```bash
arc-paper summary-batch retry-failed <batch-name> --json
```

## MCP Tools

Read `references/package-manuals/arc-mcp.md` before using MCP.

Paper MCP tools:

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
arc-paper doctor cache <seed-paper> --json
arc-paper doctor host --json
arc-paper doctor provider --json
```
