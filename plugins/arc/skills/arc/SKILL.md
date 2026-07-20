---
name: arc
description: Use for ARC research workflows involving paper metadata, arXiv full text, INSPIRE references and citers, paper section lookup, equation context, LLM paper summaries, research-domain construction from seed papers, and checking Markdown/PDF research notes.
---

# Agent Research Copilot  (ARC)

ARC is a cache-first research toolkit for theoretical-physics papers and
research-domain construction. Use ARC tools instead of scraping arXiv/INSPIRE
or reimplementing paper/domain workflows.

## Preflight Gate

Before any ARC CLI call, decide whether the request is a managed ARC
workflow run or a direct ARC tool task. Also determine whether the current
request came directly from a human who explicitly named ARC, rather than from
another agent or from a human request that did not name ARC. Use the message
provenance exposed by the host; do not classify quoted or forwarded text as a
direct human invocation. If provenance is unavailable or ambiguous, treat the
request as not mode-eligible and continue in `auto` without asking.

Managed workflow runs follow `workflows/domain.md`, `workflows/ideas.md`,
`workflows/check.md`, `workflows/plan.md`, `workflows/calculate.md`, or
`workflows/companion.md`, and
create project-local workflow artifacts such as domain references, ranked
ideas, work notes, note-check records, calculation records, reports, rankings,
recommendations, research directions, or follow-up project directories. For
these, read `rules/interaction.md`. Ask for an automation mode only when the
managed workflow was invoked directly by a human whose current prompt
explicitly names ARC. In that case, obtain the mode before calling ARC
paper/domain/LLM tools; do not perform preliminary calls such as `get_metadata`,
`get_citers`, `llm_get_summary`, `domain_get_summary`, seed resolution, or
project-directory derivation before the mode choice. There is no "lightweight
recommendation" exception for a mode-eligible managed workflow.

For every other managed invocation, do not ask for an automation mode. Use
`auto` as the execution mode and perform exactly the workflow scope requested
by the caller. Finish at that scope boundary: an automatic domain request does
not authorize idea generation, and automatic idea generation does not
authorize planning or calculation. Required prerequisites named by the owning
workflow may still run, but they do not expand the requested outcome.

Direct ARC tool tasks are exempt from the automation mode gate. These include
bounded paper facts such as title, authors, abstract, citation count, section
text, or equation context, plus user-directed tool orchestration such as
collecting citers or references, filtering papers by date, generating paper
summaries or summary batches, translating named reports, or combining those
steps into a non-evaluative paper-data output. Direct tasks must not produce
recommendations, research directions, scientific rankings, ARC reports, or
project-local workflow artifacts; route those through the managed workflow
gate. Run direct tasks automatically with safe defaults unless the user
explicitly asks to review or confirm steps. Example: `use arc to download
papers that cited 0911.3380 since 2024 and create a full summary of these
papers` is direct ARC tool orchestration, not a managed workflow mode prompt.

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
  `workflows/plan.md`, `workflows/calculate.md`, or `workflows/companion.md`),
  read that workflow file
  and follow its steps. Reading the workflow file is a blocking requirement
  before any workflow CLI call.
- ARC workflow completion checks and improvement notes: read
  `rules/self-reflection.md`.
- Single-paper metadata, full text, sections, equations, citers, references,
  paper summaries, or summary batches: read
  `manuals/arc-paper.md`.
- Research field/domain construction, foundation-paper selection, domain
  networks, evidence packs, graph HTML, or field briefings: read
  `manuals/arc-domain.md`.
- Any background job, job watcher, timeout, cancellation, or asynchronous
  report export: read `manuals/arc-jobs.md`.
- Optional MCP calls: read `manuals/arc-mcp.md`, but only when the user
  explicitly installed or requested the separate MCP companion.
- Host LLM/provider detection, model choice, direct prompt tests, or provider
  troubleshooting: read `manuals/arc-llm.md`.
- Companion-reading PDF generation: read `workflows/companion.md` and
  `manuals/arc-companion.md` before fetching a paper or starting LLM work.
- User-facing Markdown report export: see `rules/math_typeset.md` and
  `manuals/arc-jobs.md`.

## CLI Resolution

Use `arc-paper`, `arc-domain`, `arc-llm`, `arc-typeset`, `arc-companion`, and
`arc-jobs` directly when the host plugin exposes them on `PATH`. For a
standalone Skill install, or when a bare command is unavailable, invoke the
same command through:

```bash
<skill-dir>/scripts/arc-runtime <arc-command> [args...]
```

The first real CLI call lazily installs the immutable core runtime. Managed,
CI, or offline-preparation environments may prewarm it with
`<skill-dir>/scripts/arc-runtime setup --profile core`; diagnose it with
`<skill-dir>/scripts/arc-runtime doctor --profile core`. The base Skill never
installs or starts MCP.

## Workflow

Follow the workflow step by step. Do not skip any step, and do not kill or
cancel any job because it is slow or time consuming.

### Phase 1: Setup

Step 1: Decide the automation level and requested scope.
Use this step only for managed ARC workflow runs. For direct ARC tool tasks,
skip the Workflow section and use the relevant manuals and CLI tools
directly.

For managed workflows invoked directly by a human whose current prompt
explicitly names ARC, use an explicit user choice. If that human asks for
automatic or non-interactive work, use `auto`. If they ask to review or confirm
steps, use `interactive`. If they did not specify `auto` or `interactive`, do
not infer the mode. Use the host's selection/menu tool, following
`rules/interaction.md`, with these options: `Run automatically (Recommended)`,
`Confirm major steps`, and `Discuss before running`. If no suitable
selection/menu tool is available, use the typed fallback from
`rules/interaction.md`. Do not treat `continue`, `resume`, or a bare approval
to proceed as `auto`.

For agent-invoked managed workflows and human prompts that do not explicitly
name ARC, set `automation_level` to `auto` without asking. Preserve the
caller's requested workflow boundary; `auto` suppresses routine confirmation
questions but never opts the caller into downstream workflows.

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
identifier inference and `manuals/arc-jobs.md` for background jobs.

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
`medium` unless the user explicitly asks otherwise. Never select the `max`
model tier automatically; use it only when the user explicitly requests the
`max` model tier. See `manuals/arc-llm.md` for model tiers.

Use a stable safe `run_id`: lowercase ASCII letters, digits, and underscores,
for example a short intent slug plus UTC timestamp. Set `skill_dir` to the ARC
skill directory and `skill_workflow_json_dir` to
`<skill-dir>/workflows/json`.

### Phase 2: Route Selection

Resolve the user's intent and classify it into one of the five cases below.
Choose only the case needed for the requested outcome. Run another case only
when it is an explicit prerequisite below or the caller also requested that
outcome. Never interpret `auto` as permission to advance to a later case.

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

Case 5: Generate a companion-reading PDF.
Use only when the user explicitly requests a companion reading or asks for the
original paper to be split into semantic units with interleaved translation
and commentary.
Read and execute `workflows/companion.md`. The default deliverable is one PDF;
never generate a reproducibility package unless the user explicitly requests
it.

### Phase 3: Self-Reflection

Read and follow `rules/self-reflection.md`.
