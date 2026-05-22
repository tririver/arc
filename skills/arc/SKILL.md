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
arc-paper get-llm-summary arXiv:0911.3380 --json
arc-paper generate-llm-summary arXiv:0911.3380 --provider auto --json
```

`get-llm-summary` reads the cache first, then automatically generates and caches
the summary when a host LLM provider is available. Use
`generate-llm-summary` only when an explicit generation step or provider
override is needed.

If `get-llm-summary` returns `status: "needs_llm"`, use the returned
`llm_task` to generate schema-valid JSON manually, then store it with:

```bash
arc-paper store-llm-summary arXiv:0911.3380 --summary-json - --json
```

For more than 10 papers, use `summary-batch`; do not run one interactive LLM
step per paper. Read `references/arc-paper.md` for the batch workflow.

## Domain Info

For a research-field/domain package from a seed paper:

```bash
arc-domain build arXiv:0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain status arXiv:0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain get-summary arXiv:0911.3380 --intent "quasi-single-field inflation observables" --json
```

Use `arc-domain` or ARC MCP domain tools instead of reimplementing domain
construction in the skill. Read `references/arc-domain.md` before starting a
domain build.
