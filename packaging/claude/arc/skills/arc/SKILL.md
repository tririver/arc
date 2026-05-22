---
name: arc
description: Use for ARC research workflows involving paper metadata, ar5iv full text, INSPIRE references/citers, paper section lookup, equation context, and LLM paper summaries.
---

# ARC

Use `arc-paper-query` or ARC MCP tools for paper data. Do not manually scrape
ar5iv or INSPIRE when the CLI/MCP tools are available.

## Paper Query

For single-paper deterministic data:

```bash
arc-paper-query get-title arXiv:0911.3380 --json
arc-paper-query get-abstract arXiv:0911.3380 --json
arc-paper-query get-authors arXiv:0911.3380 --json
arc-paper-query get-references arXiv:0911.3380 --json
arc-paper-query get-citers arXiv:0911.3380 --json
arc-paper-query get-toc arXiv:0911.3380 --json
arc-paper-query get-section arXiv:0911.3380 --section S2 --json
arc-paper-query get-equation-context arXiv:0911.3380 --query "E = mc^2" --json
```

For LLM summaries:

```bash
arc-paper-query get-llm-summary arXiv:0911.3380 --json
arc-paper-query generate-llm-summary arXiv:0911.3380 --provider auto --json
```

`get-llm-summary` reads the cache first, then automatically generates and caches
the summary when a host LLM provider is available. Use
`generate-llm-summary` only when an explicit generation step or provider
override is needed.

If `get-llm-summary` returns `status: "needs_llm"`, use the returned
`llm_task` to generate schema-valid JSON manually, then store it with:

```bash
arc-paper-query store-llm-summary arXiv:0911.3380 --summary-json - --json
```

For more than 10 papers, use `summary-batch`; do not run one interactive LLM
step per paper. Read `references/paper-query.md` for the batch workflow.
