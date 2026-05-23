# ARC Operating Reference

This reference contains general operating rules for ARC workflows. Package
commands and MCP tool names live in the package-specific references.

## General Rules

- Prefer cache reads first; generate or refresh only when needed.
- Use structured CLI output when available.
- Paper IDs may omit the `arXiv:` prefix.
- For slow or large MCP work, use the background-job procedure in
  `references/package-manuals/arc-mcp.md`.
- For user choices and confirmations, use
  `references/rules/interaction.md`.
- Do not cancel a job unless the user explicitly asks.
- Report cache paths or artifact paths when they help the user inspect results.
- The scientific integrity and robustness rules in
  `references/rules/integrity.md` apply to all ARC workflows.

## Reference Selection

### Phase 1: Identify the package surface.
Step 1: For single-paper work, read `references/package-manuals/arc-paper.md`.
Step 2: For domain or research-field work, read
`references/package-manuals/arc-domain.md`.
Step 3: For MCP calls or background jobs, read
`references/package-manuals/arc-mcp.md`.
Step 4: For provider/model/runtime diagnosis, read
`references/package-manuals/arc-llm.md`.

### Phase 2: Execute through ARC.
Step 1: Use ARC package tools instead of scraping arXiv/INSPIRE directly.
Step 2: Keep generated or refreshed work explicit.
Step 3: Preserve warning and artifact contracts from
`references/rules/integrity.md`.
