---
name: arc
description: Use for ARC research workflows involving paper metadata, arXiv full text, INSPIRE references/citers, paper section lookup, equation context, LLM paper summaries, and research-domain construction from seed papers.
---

# Advanced Research Compass (ARC)

ARC is a cache-first research toolkit for theoretical-physics papers and
research-domain construction. Use ARC tools instead of scraping arXiv/INSPIRE
or reimplementing paper/domain workflows.

## Required References

Read the relevant reference before calling ARC tools. These reads are required,
not optional.

- User choices, automation mode, and confirmation behavior: read
  `references/rules/interaction.md`. Any ARC user question must use the
  selection tool; do not wait for typed input.
- Scientific claims, gap scoring, automated workflow decisions, warning
  behavior, or robustness-sensitive execution: read
  `references/rules/integrity.md`.
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

## Workflow

Follow the workflow step by step. Do not skip any step, and do not kill or
cancel any job because it is slow or time consuming.

### Phase 1: Setup

Step 1: Decide the automation level.
Use an explicit user choice. If the user asks for automatic or non-interactive
work, use `auto`. If the user asks to review or confirm steps, use
`interactive`. Do not treat `continue`, `resume`, or a bare approval to proceed
as `auto`. 

Step 2: Extract `<user-intent>`.
Keep the research/scientific request. Remove operational instructions such as
automation mode, project directory, and output formatting.

Step 3: Resolve `<seed-paper-list>`.
Use explicit paper identifiers when present. Otherwise infer seed papers from
`<user-intent>` through ARC paper tools. If a slow MCP call returns a background
job id, immediately use the blocking CLI watcher described in
`references/package-manuals/arc-mcp.md`.

Step 4: Resolve `<project-dir>`.
Use a user-specified project directory when present. Otherwise derive a safe
directory name from `<seed-paper-list>` with ARC paper tools. If the directory
already exists, follow the automation policy in
`references/rules/interaction.md`.

Step 5: Write `<project-dir>/context.json`.
Include `automation_level`, `workflow`, `original_request`, `user_intent`,
`project_dir`, `seed_paper_list`, `provider`, `model`, `workers`, and
`refresh`.

### Phase 2: Route Selection

Resolve the user's intent and classify it into one of the three cases below.

Case 1: Build domain references only.
Read and execute `references/research-workflows/build-domain.md`.

Case 2: Suggest research ideas from a not-yet-explicit request.
First complete Case 1. Then read and execute
`references/research-workflows/suggest-ideas.md`.

Case 3: Calculate from an explicit idea.
If the idea is not explicit enough, first complete Case 1 and Case 2, then ask
the user to select one concrete idea.
If the idea is explicit enough:
Step 1: Read and execute `references/research-workflows/research-plan.md`.
Step 2: Read and execute `references/research-workflows/research-foundation.md`.
Step 3: Read and execute `references/research-workflows/research-execute.md`.

### Phase 3: Self-Reflection

Before marking any ARC workflow complete, append a self-reflection entry to
`<project-dir>/self-reflect.md`. Use this file name because the step records
after-action reflection, not only optional suggestions.

Include concrete, portable improvement suggestions when the run reveals a
workflow, prompt, package, documentation, cache, or test weakness. Make each
suggestion actionable: affected file or phase, evidence from the run, exact
command or edit to try, and an acceptance check.

If no concrete improvement was found, still append a dated entry saying that no
actionable ARC improvement was identified for this run. The workflow is not
complete until this append step is done.
