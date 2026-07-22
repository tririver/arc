# Arc Paper Package

`arc-paper` is the single-paper information package. Use it for metadata,
INSPIRE references and citers, parsed ar5iv JSON, cached full-text search, table of
contents, section lookup, equation context, LLM paper summaries, and
paper-summary batches.

All single-paper operations in higher-level tools should go through
`arc-paper`.

## Worker CLI

Ordinary ARC LLM workers receive the deterministic paper interface as
`arc-paper-worker`. It uses the same non-LLM commands documented below, can
read the complete shared ARC paper cache, and may fetch missing paper data
through ARC's built-in providers. Summary generation, reference inference,
LLM batch execution, and every alias that can start a model are rejected with
`nested_llm_forbidden`.

Each run uses a writable cache overlay in front of the read-only shared cache.
Parsing, refreshes, imports, annotations, and removals first change the
overlay; removals are recoverable tombstones. At the end of a worker call ARC
validates overlay records and atomically promotes valid content. Invalid data
is quarantined, and conflicting content is preserved as a version with a
conflict record rather than silently overwriting the shared cache.

Worker results larger than 64 KiB are stored under the run artifacts. The
result envelope contains a hash and page handle; read it with:

```bash
arc-paper-worker artifact-read <handle> --offset <byte-offset> --limit <bytes> --json
```

There is no total paper-query limit. Calls are audited without credentials or
full host configuration. Files obtained with explicitly inherited host tools
must be imported with `arc-paper-worker parse` before they are ARC evidence.

## Controller-safe capability catalog

Controllers discover the structured ARC-paper surface with
`arc_paper.catalog_document()` and execute a selected operation with
`arc_paper.dispatch_operation()`. The versioned catalog is the authority for
stable operation IDs, closed JSON argument schemas, result serialization,
inline/job execution, network and cache effects, nested LLM use, and
supervision requirements. It covers the public service operations while
intentionally excluding CLI transport flags and Python-only callback objects.
Catalog dispatch returns the declared ARC result envelope directly. The
catalog's result-delivery block separately requires the Controller to apply
64-KiB inline paging through artifact handles and to persist an operation
receipt containing the operation identity, version, and arguments hash.

Catalog dispatch accepts an operation and structured parameters only; it does
not accept argv or shell text. Local inputs and export destinations are opaque
`{"handle_id": "..."}` objects. The Controller must supply an artifact resolver
that maps the issued handle to an authorized read or write path. A raw path or
a handle without that resolver is rejected before the service is called. The
ordinary CLI retains its trusted-local path interface for direct user and host
operation.

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

## Parsed Source CLI

For Markdown/TeX/PDF notes, `arc-paper parse` is the required
source-ingestion step.
Parsed ARC paper output is the source of truth for sections, equations, line
anchors, and PDF page anchors.

Parse accessible sources before checking claims:

```bash
arc-paper parse --markdown NOTE.md --pdf NOTE.pdf --id NOTE_ID --json
arc-paper parse --markdown NOTE.md --id NOTE_ID --json
arc-paper parse --tex NOTE.tex --pdf NOTE.pdf --id NOTE_ID --json
arc-paper parse --tex NOTE.tex --id NOTE_ID --json
arc-paper parse --pdf NOTE.pdf --id NOTE_ID --json
arc-paper parse --html NOTE.html --id NOTE_ID --json
arc-paper parse --paper-id 0911.3380 --source ar5iv --json
```

When Markdown or TeX was derived from a PDF, use the paired Markdown+PDF or
TeX+PDF parse command so ARC paper can locate equation numbers and pages from
nearby prose, equation tokens, and printed number candidates. If a PDF cannot
be used, rely on warnings returned by `arc-paper parse`.

Read parsed sources through ARC paper commands:

```bash
arc-paper get-parsed NOTE_ID --json
arc-paper get-parsed-toc NOTE_ID --json
arc-paper get-parsed-section NOTE_ID --section SECTION_ID --json
arc-paper get-parsed-equations NOTE_ID --json
arc-paper get-parsed-equation NOTE_ID --equation-id EQUATION_ID --json
```

If checking shows that a parsed equation is problematic, annotate it or reparse
with the same source id:

```bash
arc-paper mark-parsed-equation NOTE_ID --equation-id eq_00042 \
  --status problematic --reason "Short reason from the check"
```

For re-parse, update the parse input and rerun `arc-paper parse` with the same
`--id`.

## LLM Summary CLI

### Phase 1: Try the cached-or-generate command.
Step 1: Run:

```bash
arc-paper llm-summary <seed-paper> --json
```

Paper-summary generation defaults to `--model-tier low`; pass another tier
explicitly when the user requests a different quality/cost tradeoff.

Step 2: If it returns a summary, use it.
Step 3: If it returns `status: "needs_llm"`, use the manual fallback below.

### Phase 2: Explicitly generate or refresh when needed.
Step 1: Use this when the user asks to regenerate, choose a provider, or bypass
an old cache:

```bash
arc-paper llm-generate-summary <paper-id> [<paper-id> ...] --provider auto --json
arc-paper llm-generate-summary <paper-id> --provider codex-cli --model <model> --json
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

## Optional MCP Tools

Skip this section for normal Skill/CLI workflows. Read `manuals/arc-mcp.md`
only when the user explicitly installed or requested the MCP companion.

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

For LLM paper summaries through optional MCP, use `background=true` for slow
or massive launches and follow `manuals/arc-mcp.md`. Default workflows should
submit the equivalent CLI through `arc-jobs`.

## Cache Notes

Parsed paper access is lightweight by default. It stores the paper ID, source
hash, TOC, section text, and equation context without building a rich document
or downloading image assets. Use `arc-paper parse --include-document` only for
an explicit rich-document consumer such as `arc-companion`; rich sidecars are
keyed independently by source hash and rich parser version.

arXiv author source is also explicit-only and never participates in normal
parsing or rendering:

```bash
arc-paper source-cache 0911.3380 --version 3 --json
arc-paper source-probe 0911.3380 --version 3 --json
```

The cache records the fixed version, reported license, archive and file hashes,
sizes, file manifest, and static main-TeX candidates. Extraction rejects unsafe
paths, links, special files, and excessive expansion. ARC does not execute the
author TeX or treat a cached source as permission to publish derivatives.

Discover the active cache path:

```bash
arc-paper doctor-cache --json
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

List or remove cached paper entries:

```bash
arc-paper cache list --json
arc-paper cache list --id 0911.3380 --json
arc-paper cache list --since 1h --json
arc-paper cache list --past-day --json
arc-paper cache remove --id 0911.3380 --dry-run --json
arc-paper cache remove --since 1h --json
arc-paper cache remove --since 1h --yes --json
```

`cache remove` prints selected papers and asks for `y` confirmation unless
`--yes` or `--dry-run` is used. Use `--all` to remove all cached paper entries
explicitly.
