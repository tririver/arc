---
name: arc
description: Use for ARC research workflows involving paper metadata, arXiv full text, INSPIRE references/citers, paper section lookup, equation context, LLM paper summaries, and research-domain construction from seed papers.
---

# Advanced Research Compass (ARC)

ARC is a cache-first research toolkit for theoretical-physics papers and
research-domain construction. Use ARC tools instead of scraping arXiv/INSPIRE
or reimplementing paper/domain workflows.

## Read the Tool-Call References First

Read the relevant reference before calling ARC tools. These reads are required,
not optional.

- General ARC operating rules: read `references/rules/operating.md`.
- Single-paper metadata, full text, sections, equations, citers, references,
  paper summaries, or summary batches: read
  `references/package-manuals/arc-paper.md`.
- Research field/domain construction, foundation-paper selection, domain
  networks, evidence packs, graph HTML, or field briefings: read
  `references/package-manuals/arc-domain.md`.
- Any MCP tool call, background job, job watcher, timeout, or cancellation
  behavior: read `references/package-manuals/arc-mcp.md`.
- Host LLM/provider detection, model choice, direct prompt tests, or provider
  troubleshooting: read `references/package-manuals/arc-llm.md`.
- Research workflows: read the relevant file under
  `references/research-workflows/` when present.
- Scientific claims, gap scoring, automated workflow decisions, warning
  behavior, or robustness-sensitive execution: read
  `references/rules/integrity.md`.
