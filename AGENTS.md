# ARC Development Guidance

This repository contains ARC research tooling rebuilt as Python packages plus
thin agent adapters. The reference snapshot in `0_ref/` is read-only context and
must not be modified.

## Research Tool Development

- Build ARC tools as general theoretical-physics research infrastructure, not
  as optimizations for one seed paper, one subfield, or one generated wiki.
- Avoid hard-coded paper IDs, author names, subfield labels, or field-specific
  keyword lists in discovery, paper-query, summary, batch, or workflow logic.
- Prefer configurable, documented heuristics that transfer across theoretical
  physics domains. When a heuristic is motivated by a concrete failure case,
  encode the general failure mode and add tests or diagnostics that would catch
  analogous cases in other fields.
- Treat example papers as regression cases only. They should validate general
  behavior, not define special-case behavior.

## Instruction Review

- Apply this review gate to development work that changes ARC instructions,
  workflows, prompts, schemas, scripts, tests, documentation, packaging
  metadata, MCP tools, or Python package behavior.
- Before implementing such changes, judge whether the requested instruction is
  sound, portable across supported agent hosts, compatible with ARC's
  general-purpose research goals, and consistent with existing workflow policy.
- If the instruction is acceptable, proceed and preserve the general design. If
  it is not acceptable, explain the specific conflict before changing code.

## Agent Host Portability

- Keep skills, prompts, scripts, MCP adapters, package behavior, and
  documentation compatible with multiple coding-agent hosts, including Claude
  Code, Cursor, GitHub Copilot, Codex, and similar agents.
- Do not assume Codex as the required host except in Codex-specific packaging or
  installation notes.
- Do not rely on one host's UI behavior, command syntax, environment variables,
  or tool names for core workflows. When host-specific behavior is useful, keep
  it optional and provide a portable fallback.
- Prefer generic terms such as "agent", "host", "skill directory", "MCP
  server", and "workflow" in reusable documentation.

## Package Boundaries

- `packages/arc-llm-worker` owns reusable host LLM execution: host detection,
  provider selection, model defaults, and Codex/Claude prompt calls.
- `packages/arc-paper-query` owns deterministic paper data access, ID
  normalization, caching, parsing, paper-summary contracts, paper-summary
  orchestration, and batch execution.
- `packages/arc-mcp` should stay a thin MCP adapter over `arc_paper_query`
  service functions and batch helpers.
- `skills/arc`, `prompts/`, `schemas/`, and `packaging/` should describe or
  wrap package behavior rather than reimplementing it.
- Keep `0_ref/` as reference-only material. Do not preserve old compatibility
  when it conflicts with the new package architecture.

## Testing

- For package changes, run the focused pytest tests for the touched package
  first, then the combined local suite when practical:

  ```bash
  packages/arc-paper-query/.venv/bin/python -m pytest packages/arc-paper-query/tests packages/arc-mcp/tests
  ```

- Unit tests must not require network access. Network integration tests should
  stay opt-in through `ARC_RUN_NET_TESTS=1`.
- Keep tests close to the module they cover. Use repository-level tests only for
  cross-package integration behavior.

## Language Policy

- User-facing discussion may be in the user's language.
- Skills, prompts, schemas, code comments, docstrings, package metadata, and
  durable documentation should be written in English unless there is a specific
  reason not to.
