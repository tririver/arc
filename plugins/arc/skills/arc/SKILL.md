---
name: arc
description: Use for ARC research workflows involving paper metadata, arXiv full text, INSPIRE references and citers, paper section lookup, equation context, LLM paper summaries, research-domain construction from seed papers, and checking Markdown/PDF research notes.
---

# Advanced Research Compass (ARC)

ARC is a cache-first research toolkit for theoretical-physics papers and
research-domain construction. Use ARC tools instead of scraping arXiv/INSPIRE
or reimplementing paper/domain workflows.

## Preflight Gate

Before any ARC workflow MCP or CLI call, decide whether the user requested a
workflow deliverable or a direct factual lookup.

Workflow deliverables include recommendations, research directions, idea generation,
domain construction, note checking, planning, calculations,
reports, rankings, or follow-up project artifacts. For these, read
`rules/interaction.md` and obtain an explicit automation mode before calling
ARC paper/domain/LLM tools. Do not perform preliminary calls such as
`get_metadata`, `get_citers`, `llm_get_summary`, `domain_get_summary`, seed
resolution, or project-directory derivation before the mode choice.
There is no "lightweight recommendation" exception.

Direct factual lookup is exempt only when the user asks for a bounded paper
fact such as title, authors, abstract, citation count, section text, or
equation context, and does not ask for recommendations, ideas, domain work,
checking, planning, calculation, reports, rankings, or project artifacts.

## Required References

Read the relevant reference before calling ARC tools. These reads are required,
not optional.

- User choices, automation mode, and confirmation behavior: read
  `rules/interaction.md`. Any ARC user question must use the
  selection tool; do not wait for typed input.
- Scientific claims, gap scoring, automated workflow decisions, warning
  behavior, or robustness-sensitive execution: read
  `rules/integrity.md`.
- General ARC operating rules: read `rules/operating.md`.
- User-facing Markdown math and TeX typesetting: read
  `rules/math_typeset.md`.
- Note checking, verification, or audit requests: read
  `workflows/check.md` before any parse, section read, or equation extraction call.
- When the user intent triggers a workflow-specific file
  (`workflows/check.md`, `workflows/domain.md`, `workflows/ideas.md`,
  `workflows/plan.md`, or `workflows/calculate.md`), read that workflow file
  and follow its steps. Reading the workflow file is a blocking requirement
  before any workflow MCP or CLI call.
- ARC workflow completion checks and improvement notes: read
  `rules/self-reflection.md`.
- Single-paper metadata, full text, sections, equations, citers, references,
  paper summaries, or summary batches: read
  `manuals/arc-paper.md`.
- Research field/domain construction, foundation-paper selection, domain
  networks, evidence packs, graph HTML, or field briefings: read
  `manuals/arc-domain.md`.
- Any MCP tool call, background job, job watcher, timeout, or cancellation
  behavior: read `manuals/arc-mcp.md`.
- Host LLM/provider detection, model choice, direct prompt tests, or provider
  troubleshooting: read `manuals/arc-llm.md`.
- User-facing Markdown report export: see `rules/math_typeset.md` and
  `manuals/arc-mcp.md`.

## Workflow

Follow the workflow step by step. Do not skip any step, and do not kill or
cancel any job because it is slow or time consuming.

### Phase 1: Setup

Step 1: Decide the automation level.
Use an explicit user choice. If the user asks for automatic or non-interactive
work, use `auto`. If the user asks to review or confirm steps, use
`interactive`. If the user did not specify `auto` or `interactive` explicitly,
do not infer the mode. Use the host's selection/menu tool, following
`rules/interaction.md`, with these options: `Run automatically (Recommended)`,
`Confirm major steps`, and `Discuss before running`. If no suitable
selection/menu tool is available, use the typed fallback from
`rules/interaction.md`. Do not treat `continue`, `resume`, or a bare approval
to proceed as `auto`.

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
`<user-intent>` through ARC paper tools. See `manuals/arc-paper.md` for paper
identifier inference and `manuals/arc-mcp.md` for background jobs.

Step 4: Resolve `<project-dir>`.
Capture `<arc-run-root>` by running `pwd -P` in the directory where the user
launched the agent command. Do not use host-internal project/cache locations.
If `<arc-run-root>` is under `.claude`, `.codex`, a plugin directory, or a
cache directory, print `WARNING:` and stop before writing artifacts.

Use a user-specified project directory when present. Otherwise derive
`<project_dir_name>` as a safe directory stem from `<seed-paper-list>` with ARC
paper tools, then resolve `<project-dir>` with:

```bash
python3 <skill-dir>/workflows/scripts/resolve-project-dir.py \
  --name <project_dir_name> \
  --run-root <arc-run-root> \
  --json
```

The generated `<project-dir>` must be the direct child
`<arc-run-root>/<project_dir_name>`. Do not create
`arc-output/<project_dir_name>`, do not wrap the safe name in another directory,
and do not write generated workflow artifacts under `.claude`, `.codex`,
plugin dirs, or cache dirs. If the directory already exists, follow the
automation policy in `rules/interaction.md`.

Step 5: Write `<project-dir>/context.json`.
Include `automation_level`, `workflow`, `original_request`, `user_intent`,
`arc_run_root`, `project_dir_name`, `project_dir`, `run_id`, `created_at`,
`skill_version`, `skill_dir`, `skill_workflow_json_dir`, `seed_paper_list`,
`provider`, `model_tier`, `workers`, and `refresh`.

Set `provider` to `auto` unless the user pins a provider. Set `model_tier` to
`medium` unless the user explicitly asks otherwise. See `manuals/arc-llm.md`
for model tiers.

Use a stable safe `run_id`: lowercase ASCII letters, digits, and underscores,
for example a short intent slug plus UTC timestamp. Set `skill_dir` to the ARC
skill directory and `skill_workflow_json_dir` to
`<skill-dir>/workflows/json`.

### Phase 2: Route Selection

Resolve the user's intent and classify it into one of the four cases below.

Case 1: Build domain references only.
Read and execute `workflows/domain.md`.

Case 2: Suggest ideas from a not-yet-explicit request.
First complete Case 1. Then read and execute
`workflows/ideas.md`.

Case 3: Check note files or collaborator notes.
Use when the request asks to check, verify, audit, or mark work-note premises
and claims in one or more accessible `.md` or `.pdf` notes.
`workflows/check.md` was already loaded in Required References.
Follow its 5-phase workflow: Parse -> Preflight -> Write Planning Handoff ->
Execute `plan.md` and `calculate.md` -> Record Note-Check Status.
Do not skip directly to parsing results; the preflight, planning handoff, and
owned-workflow execution steps are mandatory.

Before leaving Case 3 or sending a final response, read
`<project-dir>/work-note.md`. If any ready detailed step exists, execute
`workflows/calculate.md`. If no ready detailed step exists but rough or pending
coverage remains from the original note-check request, return to
`workflows/plan.md`. Adjudicate every item in `Rough Steps For Later Planning`:
promote and execute it, remove or mark it obsolete/not triggered, or record an
explicit stop condition in `Open Questions` or `Calculation Status`. Only stop
when requested coverage is complete and no triggered rough/pending item remains.

Case 4: Calculate from an explicit idea.
If the idea is not explicit enough, first complete Case 1 and Case 2, then ask
the user to select one concrete idea.
If the idea is explicit enough:
Step 1: Read and execute `workflows/plan.md`. It writes or updates
`<project-dir>/work-note.md` and an immutable version under
`<project-dir>/calculate/<run-id>/work-notes/`.
Step 2: Read and execute `workflows/calculate.md`. If `calculate.md` requests
macro expansion or blocked-step refinement, return to Step 1 for that region.

Before leaving Case 4 or sending a final response, read
`<project-dir>/work-note.md`. If any ready detailed step exists, execute
`workflows/calculate.md`. If no ready detailed step exists but rough or pending
coverage remains from the original calculation request, return to
`workflows/plan.md` to promote the next coherent chunk. Adjudicate every item
in `Rough Steps For Later Planning`: promote and execute it, remove or mark it
obsolete/not triggered, or record an explicit stop condition in `Open Questions`
or `Calculation Status`. Only stop when requested calculation coverage is
complete and no triggered rough/pending item remains.

### Phase 3: Self-Reflection

Read and follow `rules/self-reflection.md`.
