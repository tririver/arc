# ARC Development Guidance

This repository contains ARC research tooling rebuilt as Python packages plus
thin agent adapters. The reference snapshot in `0_ref/` is read-only context and
must not be modified.

## Project Map

ARC is an agent-skill layer in `plugins/arc/skills/arc/` backed by reusable
Python packages in `packages/`; when the ARC skill is unavailable, agents
should read the relevant workflow file directly and call the package CLI or MCP
tools it names.

- `plugins/arc/skills/arc/workflows/domain.md`: Use this workflow to build
  project-local research-domain artifacts from seed papers, including domain
  summaries, domain HTML, graph data, and paper JSON packs.
- `plugins/arc/skills/arc/workflows/ideas.md`: Use this workflow to run
  concurrent proposer-reviewer idea loops, rank completed ideas, and choose a
  calculation candidate.
- `plugins/arc/skills/arc/workflows/plan.md`: Use this workflow when a task to
  be planned is available to gather evidence and write a source-aware,
  reviewable `<project-dir>/work-note.md` with foundations, accepted premises,
  ready calculation steps, rough later steps, and archived work-note versions.
- `plugins/arc/skills/arc/workflows/calculate.md`: Use this workflow after
  `plan.md` creates `<project-dir>/work-note.md` to check work-note premises,
  run blind reference checks, execute ready consensus calculation steps, and
  update calculation status in the work note.
- `plugins/arc/skills/arc/workflows/check.md`: Use this workflow when the user
  asks to check Markdown or PDF research notes by preparing a planning handoff
  for `plan.md` and reusing `calculate.md` for claim verification.
- `packages/arc-domain`: Owns research-domain construction from seed papers,
  including foundation/domain-paper selection, graph artifacts, evidence packs,
  HTML rendering, domain summaries, and paper JSON pack exports.
- `packages/arc-llm`: Owns reusable host LLM execution, provider/model
  selection, background jobs, proposer-reviewer batches, consensus execution,
  and related benchmarking helpers.
- `packages/arc-paper`: Owns deterministic paper access and caching, ID
  normalization, ar5iv/INSPIRE metadata, references, citers,
  full-text/equation search, paper summaries, and summary batches.
- `packages/arc-typeset`: Owns report typesetting utilities, including
  Markdown-to-PDF conversion, Markdown translation, and batch translation of
  project reports.

## Research Tool Development

- Build ARC tools as general theoretical-physics research infrastructure, not
  as optimizations for one seed paper, one subfield, or one generated wiki.
- Avoid hard-coded paper IDs, author names, subfield labels, or field-specific
  keyword lists in discovery, paper access, summary, batch, or workflow logic.
- Prefer configurable, documented heuristics that transfer across theoretical
  physics domains. When a heuristic is motivated by a concrete failure case,
  encode the general failure mode and add tests or diagnostics that would catch
  analogous cases in other fields.
- Treat example papers as regression cases only. They should validate general
  behavior, not define special-case behavior.

## Workflow Design Principles

- Design ARC workflows to support agent reasoning, not to replace it with
  exhaustive mechanical micro-steps. Avoid artifact shapes or instructions that
  force unnatural reasoning order, such as treating every extracted equation as
  an executable plan step.
- Handoff artifacts should provide the relevant source context, user intent,
  constraints, locations, and coverage requirements. The downstream workflow
  that owns planning should decide classification, granularity, dependency
  order, and execution strategy.
- Prefer context-rich, reviewable prompts and rolling plans over one-item-per-step
  expansion. Use structured inventories for traceability, but do not make
  inventory items imply plan steps.
- When a workflow must eventually cover many claims, equations, or source
  items, state the coverage requirement explicitly while preserving freedom for
  the planner to group checks by logical derivation flow, physical argument,
  dependencies, approximation regime, or risk.

## Instruction Review

- Apply this review gate to development work that changes ARC instructions,
  workflows, prompts, schemas, scripts, tests, documentation, packaging
  metadata, MCP tools, or Python package behavior.
- Before implementing such changes, judge whether the requested instruction is
  sound, portable across supported agent hosts, compatible with ARC's
  general-purpose research goals, and consistent with existing workflow policy.
- If the instruction is acceptable, proceed and preserve the general design. If
  it is not acceptable, explain the specific conflict before changing code.
- When rejecting or disagreeing with a requested instruction change, do not
  partially implement it, rename artifacts, update mirrors, or make compromise
  edits unless the user explicitly approves a revised instruction. Explain the
  disagreement and leave files unchanged.

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

## Skill Layer

- Keep skill files simple, concise, and easy for a human to scan: clear
  headings, short bullets, and obvious commands.
- Use the skill layer to explain when and how to use ARC tools. Do not encode
  complex control flow, long decision trees, or package internals in skills.
- When a skill needs a step-by-step workflow, label it with explicit phases and
  steps: `Phase 1`, `Step 1`, `Step 2`, then `Phase 2`, and so on.
- Put detailed examples, troubleshooting, and longer workflows in focused
  reference files. Keep `SKILL.md` as the readable entry point.
- Prefer commands, MCP tool names, and expected outputs over prose-heavy
  instructions. Avoid repeating implementation details already owned by
  `arc-paper`, `arc-domain`, `arc-llm`, or `arc-mcp`.

## Package Boundaries

- `packages/arc-llm` owns reusable host LLM execution: host detection,
  provider selection, model defaults, and Codex/Claude prompt calls.
- `packages/arc-paper` owns deterministic paper data access, ID
  normalization, caching, parsing, paper-summary contracts, paper-summary
  orchestration, and batch execution.
- `packages/arc-domain` owns research-domain construction from seed
  papers: foundation selection, domain paper selection, network artifacts,
  evidence packs, HTML rendering, and domain summaries. It should call
  `arc-paper` for single-paper operations and `arc-llm` for host
  LLM work.
- `packages/arc-mcp` should stay a thin MCP adapter over `arc_paper`,
  `arc_domain`, and batch service functions.
- `plugins/arc/skills/arc`, `prompts/`, `schemas/`, and plugin manifests
  should describe or wrap package behavior rather than reimplementing it.
- Keep `0_ref/` as reference-only material. Do not preserve old compatibility
  when it conflicts with the new package architecture.

## Long-Running Terminal Work

- When a terminal command or background job is still running, be patient and do
  not poll it more often than once every 10 minutes unless there is a clear
  reason, such as expected near-term completion, visible error output, or a user
  request for status.
- Do not print routine "still running" updates unless the command status
  changed, an error appeared, or at least 10 minutes passed.
- Prefer quiet logging for noisy long-running commands: write output to a log
  file, then inspect only the tail on completion, failure, or an explicit status
  request.
- Prefer blocking watcher commands with sensible timeout windows over repeated
  manual polling when the relevant package or host provides them.

## Testing

- For package changes, run the focused pytest tests for the touched package
  first, then the combined local suite when practical:

  ```bash
  packages/arc-paper/.venv/bin/python -m pytest \
    packages/arc-llm/tests \
    packages/arc-paper/tests \
    packages/arc-domain/tests \
    packages/arc-mcp/tests
  ```

- Unit tests must not require network access. Network integration tests should
  stay opt-in through `ARC_RUN_NET_TESTS=1`.
- Keep tests close to the module they cover. Use repository-level tests only for
  cross-package integration behavior.
- Keep ARC development assets, including package-local unit tests and fixtures,
  in their designated repository locations. For every non-development
  execution that uses or evaluates ARC—including workflow tests, benchmarks,
  integration or evaluation runs, and ordinary research use—create the project
  or run directory under the git-ignored `arc-tests/` tree and write all
  generated artifacts there. Do not create ad hoc project, run, benchmark, or
  output directories at the repository root, under `packages/` or `plugins/`,
  or elsewhere in source directories.
- Workflow runs that exercise planning should write
  `<project-dir>/work-note.md` and archived work-note versions under the run
  directory instead of separate foundation artifacts.

## Language Policy

- User-facing discussion may be in the user's language.
- Skills, prompts, schemas, code comments, docstrings, package metadata, and
  durable documentation should be written in English unless there is a specific
  reason not to.
