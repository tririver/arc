---
name: arc
description: Use for ARC research workflows involving paper metadata, arXiv full text, INSPIRE references/citers, paper section lookup, equation context, LLM paper summaries, research-domain construction from seed papers, and checking Markdown/PDF research notes.
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
- ARC workflow completion checks and improvement notes: read
  `references/rules/self-reflection.md`.
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
- User-facing Markdown report export: when a workflow writes a Markdown report
  to `<project-dir>/` for human readers, call MCP `md2pdf` on that project-level file.
  `md2pdf` starts a background PDF job; record the returned job id if present
  and do not wait before continuing unless the user explicitly asks.

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

Preserve scientific domain anchors in `<user-intent>`, including phrases such
as "in the field started by arXiv:..." or "in the literature around ...".
Those phrases are part of the scientific request, not workflow metadata. Keep
the same paper identifiers separately in `seed_paper_list` as structured
routing data.

If the request references or attaches accessible files such as `.md`, `.pdf`,
`.doc`, or `.jpg`, read or extract the relevant content and summarize it as
part of `<user-intent>`. Treat collaborator notes or images as source context
for routing, domain building, checking, or later workflows.

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

Resolve the user's intent and classify it into one of the four cases below.

Case 1: Build domain references only.
Read and execute `references/research-workflows/build-domain.md`.

Case 2: Suggest research ideas from a not-yet-explicit request.
First complete Case 1. Then read and execute
`references/research-workflows/research-ideas.md`.

Case 3: Check note files or collaborator notes.
Use when the request asks to check, verify, audit, or mark foundation items in
one or more accessible `.md` or `.pdf` notes. Read and execute
`references/research-workflows/check.md`.

Case 4: Calculate from an explicit idea.
If the idea is not explicit enough, first complete Case 1 and Case 2, then ask
the user to select one concrete idea.
If the idea is explicit enough:
Step 1: Read and execute `references/research-workflows/research-plan.md`.
Step 2: Read and execute `references/research-workflows/research-foundation.md`.
Step 3: Read and execute `references/research-workflows/research-execute.md`.

### Phase 3: Self-Reflection

Read and follow `references/rules/self-reflection.md`.
