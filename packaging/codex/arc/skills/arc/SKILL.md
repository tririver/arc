---
name: arc
description: Use for ARC research workflows involving paper metadata, ar5iv full text, INSPIRE references/citers, paper section lookup, equation context, LLM paper summaries, and research-domain construction from seed papers.
---

# ARC

Use `arc-paper` or ARC MCP tools for paper data. Do not manually scrape
ar5iv or INSPIRE when the CLI/MCP tools are available.

## Paper

For single-paper deterministic data:

```bash
arc-paper get-title arXiv:0911.3380 --json
arc-paper get-abstract arXiv:0911.3380 --json
arc-paper get-authors arXiv:0911.3380 --json
arc-paper get-references arXiv:0911.3380 --json
arc-paper get-citers arXiv:0911.3380 --json
arc-paper get-toc arXiv:0911.3380 --json
arc-paper get-section arXiv:0911.3380 --section S2 --json
arc-paper get-equation-context arXiv:0911.3380 --query "E = mc^2" --json
```

For LLM summaries:

```bash
arc-paper llm-summary arXiv:0911.3380 --json
arc-paper llm-generate-summary arXiv:0911.3380 --provider auto --json
```

Phase 1: Try the cache.
Step 1: Use `arc-paper llm-summary ... --json`.
Step 2: If a cached result is returned, use it.

Phase 2: Generate only when needed.
Step 1: Use `arc-paper llm-generate-summary ... --provider auto --json` for an
explicit generation step or provider override.
Step 2: For MCP, pass `background=true` when launching many or slow summaries.
Step 3: If MCP returns `status: "job_running"` with a `job_id`, immediately run:

```bash
arc-mcp jobs watch <job_id> --json
```

Step 4: Use MCP `job_status` and `job_result` only when a CLI watcher is not
available.
Step 5: Do not call `cancel_job` unless the user explicitly asks.

If `llm-summary` returns `status: "needs_llm"`, use the returned
`llm_task` to generate schema-valid JSON manually, then store it with:

```bash
arc-paper store-llm-summary arXiv:0911.3380 --summary-json - --json
```

For more than 10 papers, use `summary-batch`; do not run one interactive LLM
step per paper. Read `references/arc-paper.md` for the batch workflow.

## Domain Info

For a research-field/domain package from a seed paper:

```bash
arc-domain llm-build arXiv:0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain status arXiv:0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain get-summary arXiv:0911.3380 --intent "quasi-single-field inflation observables" --json
```

Use `arc-domain` or ARC MCP domain tools instead of reimplementing domain
construction in the skill. Read `references/arc-domain.md` before starting a
domain build.

Phase 1: Check cached domain artifacts.
Step 1: Use `arc-domain status`, `arc-domain get-summary`, or MCP
`domain_get_summary`.
Step 2: If artifacts are missing, move to Phase 2.

Phase 2: Build the domain.
Step 1: Use `arc-domain llm-build` or MCP `llm_domain_build`.
Step 2: For MCP, pass `background=true` when the build should be scheduled
immediately instead of waiting inline.
Step 3: If MCP returns `status: "job_running"` with a `job_id`, immediately run:

```bash
arc-mcp jobs watch <job_id> --json
```

Step 4: If the CLI watcher is unavailable, poll `job_status` or `domain_status`
until complete.
Step 5: Read cached artifacts with `domain_get_summary` or `domain_get_graph`.
