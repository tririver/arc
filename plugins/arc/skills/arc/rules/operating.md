# ARC Operating Reference

This reference contains general operating rules for ARC workflows. Package
commands and MCP tool names live in the package-specific references.

## General Rules

- Prefer cache reads first; generate or refresh only when needed.
- Use structured CLI output when available.
- Paper IDs may omit the `arXiv:` prefix.
- For slow or large MCP work, use the background-job procedure in
  `manuals/arc-mcp.md`.
- If MCP is unavailable, check the relevant package manual and use the
  corresponding CLI command with structured output.
- For user choices and confirmations, use
  `rules/interaction.md`.
- Do not cancel a job unless the user explicitly asks.
- Report cache paths or artifact paths when they help the user inspect results.
- The scientific integrity and robustness rules in
  `rules/integrity.md` apply to all ARC workflows.

## Reference Selection

### Phase 1: Identify the package surface.
Step 1: For single-paper work, read `manuals/arc-paper.md`.
Step 2: For domain or research-field work, read
`manuals/arc-domain.md`.
Step 3: For MCP calls or background jobs, read
`manuals/arc-mcp.md`.
Step 4: For provider/model/runtime diagnosis, read
`manuals/arc-llm.md`.

### Phase 2: Execute through ARC.
Step 1: Use ARC package tools instead of scraping arXiv/INSPIRE directly.
Step 2: Keep generated or refreshed work explicit.
Step 3: Preserve warning and artifact contracts from
`rules/integrity.md`.
