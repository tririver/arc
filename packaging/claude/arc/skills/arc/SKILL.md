---
name: arc
description: Use for ARC research workflows involving paper metadata, ar5iv full text, INSPIRE references/citers, paper section lookup, equation context, LLM paper summaries, and research-domain construction from seed papers.
---

# ARC

ARC is a cache-first research toolkit for theoretical-physics papers and
research-domain construction. Use ARC tools instead of scraping ar5iv/INSPIRE
or reimplementing paper/domain workflows.

This file is only the entrance. Concrete commands, MCP tool names, options,
and step-by-step procedures live in `references/`.

## Required References

Read the relevant reference before calling ARC tools. These reads are required,
not optional.

- Single-paper metadata, full text, sections, equations, citers, references,
  paper summaries, or summary batches: read `references/arc-paper.md`.
- Research field/domain construction, foundation-paper selection, domain
  networks, evidence packs, graph HTML, or field briefings: read
  `references/arc-domain.md`.
- Any MCP tool call, background job, job watcher, timeout, or cancellation
  behavior: read `references/arc-mcp.md`.
- Host LLM/provider detection, model choice, direct prompt tests, or provider
  troubleshooting: read `references/arc-llm.md`.
- Scientific claims, gap scoring, automated workflow decisions, warning
  behavior, or robustness-sensitive execution: read
  `references/arc-integrity.md`.

If a task uses MCP for paper or domain work, read both the package reference
and `references/arc-mcp.md`.

## Operating Rules

- Prefer cache reads first; generate or refresh only when needed.
- Use structured CLI output when available.
- Paper IDs may omit the `arXiv:` prefix, for example `0911.3380`.
- For slow or large MCP work, use the background-job procedure in
  `references/arc-mcp.md`.
- Do not cancel a job unless the user explicitly asks.
- Report cache paths or artifact paths when they help the user inspect results.
- The scientific integrity and robustness rules in
  `references/arc-integrity.md` apply to all ARC workflows.
