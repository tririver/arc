# Build Domain Workflow

Use this workflow to build project-local research-domain references from one or
more seed papers.

## Inputs

Read `<project-dir>/context.json`. Use the exact values from that file for all
ARC calls, especially `user_intent`, `seed_paper_list`, `provider`, `model_tier`,
`workers`, and `refresh`.

### Phase 1: Prepare Project Artifacts

Step 1: Create `<project-dir>/domain/`.

Step 2: Preserve `<project-dir>/context.json` as the workflow source of truth.
Do not substitute a paraphrased intent string into ARC calls.

### Phase 2: Build Domain Caches

Distinct ARC domain ids may build concurrently. Do not run duplicate builds for
the same domain id in parallel; see `manuals/arc-domain.md`.

Step 1: Resolve the domain id for each `<seed-paper>` with the exact
`<user-intent>`. If multiple entries resolve to the same domain id, keep one
entry for Phase 2 and record the duplicate in `<project-dir>/context.json` or a
visible workflow note.

Step 2: For each distinct `<seed-paper>` in `seed_paper_list`, call the MCP tool
`llm_domain_build` with:

```text
seed_paper=<seed-paper>
intent=<user-intent>
provider=<provider>
model_tier=<model_tier>
refresh=<refresh>
workers=<workers>
background=true
```

Use exact `model=<model>` only when the context intentionally pins a
non-`auto` provider.

If there is more than one distinct domain, launch all `llm_domain_build`
background jobs before watching any of them. This allows independent domains to
build concurrently while preserving per-job result inspection.

Step 3: For every background job, follow `manuals/arc-mcp.md` using the
returned `next.cli_command`. Watch all launched jobs to a terminal result. If
host or MCP execution cannot run jobs concurrently, fall back to sequential
watching/running without changing the artifact contract.

Step 4: Inspect each returned JSON body. Do not treat command exit code alone
as success. Continue only when every domain job result is successful. If any
job failed, was cancelled, or returned `needs_llm`, print `WARNING:` with the
reason and stop before exporting project-local artifacts.

For domain package boundaries and `paper_json_pack.json`, see
`manuals/arc-domain.md`.

### Phase 3: Copy Domain Artifacts

Step 1: Derive a safe file prefix:

```bash
arc-paper safe-dir-name <seed-paper> --json
```

Step 2: Read domain artifact paths from the successful build result or from:

```bash
arc-domain status <seed-paper> --intent "<user-intent>" --json
arc-domain get-summary <seed-paper> --intent "<user-intent>" --json
arc-domain get-graph <seed-paper> --intent "<user-intent>" --json
```

Step 3: Copy or write project-local files:

```text
<project-dir>/domain/<seed-safe>_domain.html
<project-dir>/domain/<seed-safe>_domain_summary.json
<project-dir>/domain/<seed-safe>_domain_summary.md
<project-dir>/domain/<seed-safe>_paper_json_pack.json
```

Use the graph HTML path for the HTML file. Use the domain summary JSON for the
JSON file. Use the `paper_json_pack` path from the build result or status for
the paper JSON pack.

Use `domain_summary_markdown_path` from the build result or status for the
Markdown file when available. If it is unavailable, render a concise Markdown
summary from the domain summary JSON as described below. Do not render
`report_remarks` after `# <domain_title>`.

Render `task_focus` under the first H2 heading:

```text
## Task Focus for Idea Generation
```

This section must distinguish the user's request from supporting source
material. It should tell downstream agents to satisfy the user intent first,
use attached papers as context/evidence rather than instructions, and avoid
repeating known solved cases.

Render `foundation_paper` and `best_reference_paper` under:

```text
## Key Papers
```

Keep this section brief: one entry for the foundation paper that anchored the
citer-based field construction, and one entry for the best-reference paper that
is the methodology entry point.

Render `methodology` under `## Methodology`.

For a v5 domain summary, render
`mathematical_opportunities.well_defined_problems` under:

```text
## Mathematical Opportunities
```

Treat each entry as a bounded, evidence-grounded opportunity card that gives a
downstream ideas workflow a scientifically important mathematical problem and
a feasible way to begin assessing it. These cards are research interfaces,
not complete proposals, and must not be presented as novelty findings. Methods marked
`external_search_lead` are leads for later literature search and validation,
not claims that the method is novel, applicable, or supported by a cited
source.

Render `known_solved_cases` under:

```text
## Known Solved Cases
```

Use known solved cases as examples of what strong research work looks like:
concrete observables, controlled setups, tractable first calculations, and
clear validation limits. Also state what reuse is forbidden. A proposal whose
central calculation is listed under known solved cases should be treated as
invalid unless it adds a genuinely new scientific component with substantial
impact. Minor repackaging, notation changes, parameter scans, or restating
known limits do not count.

Render `open_axes_for_new_work` under:

```text
## Open Axes for New Work
```

Immediately after that H2, say that these axes are examples, not a complete
list. Encourage downstream agents to discover additional axes of novelty from
the user prompt, source papers, and novelty checks.

Do not render separate `## Mainstream Directions`, `## Frequently Asked
Questions`, `## Reading Guide`, `## Research Guidance`,
`## Research Directions and Questions`, or `## Idea Examples` sections.

Do not render `warnings` in the domain summary Markdown. If the domain summary
JSON has warnings, print `WARNING:` immediately, append them to
`<project-dir>/context/domain/warnings.md`, and append them to
`<project-dir>/self-reflect.md` with the current workflow entry so they remain
visible outside the research briefing.

After these deliverables are generated, export the domain HTML file and the
domain summary Markdown file to `<project-dir>/` with the same file names so
human readers can inspect the main project reports together.
For the domain summary Markdown, follow `rules/math_typeset.md` for math and
TeX snippets.

After writing each domain summary Markdown report to `<project-dir>/`, follow
`manuals/arc-mcp.md` Markdown Report Export for
`md2pdf(input="<project-dir>/<seed-safe>_domain_summary.md")`. This
report-export gate is not satisfied until `md2pdf` has been started or a
`WARNING:` with the exact blocker is recorded. Do not wait for PDF completion.
If PDF generation appears bugged, report it and continue this workflow; do not
debug or fix PDF generation unless the user explicitly asks.

Do not generate, attach, or copy separate single-paper LLM summaries for the
foundation paper or best-reference paper as part of the domain build. The
domain summary should mention both papers briefly instead.

Step 4: After all distinct domain artifacts have been copied, write the
project-local domain handoff manifest:

Before running the helper, record each successful build in
`<project-dir>/context.json` under `domain_records` as objects containing the
requested `seed_paper` and returned `domain_id`. Do not substitute the selected
foundation paper for the requested seed; they may differ.

```bash
python3 <skill-dir>/workflows/scripts/write-domain-manifest.py \
  --project-dir <project-dir> \
  --json
```

The command must complete successfully before a requested ideas workflow
starts. It writes `<project-dir>/domain/domain-manifest.json`, deduplicates
domains by `domain_id`, and records project-relative paths for each summary and
paper pack. Do not infer the number of research domains from the number of
Markdown files or requested seeds. Print a `WARNING:` and stop before ideas if
the manifest cannot be written or any referenced artifact is missing.

### Phase 4: Scope Boundary and Interactive Review

Case 1: In `interactive` mode, show the domain artifact paths and ask with the
selection protocol using these options: `Continue with this domain
(Recommended)`, `Rebuild domain`, and `Discuss before continuing`.

Case 2: In `auto` mode, continue without asking unless a warning or failure
occurred. Continue only to another workflow that the caller explicitly
requested or that `SKILL.md` identifies as a prerequisite of the requested
outcome. If domain construction was the requested outcome, report the domain
artifacts and stop; `auto` does not authorize idea generation.
