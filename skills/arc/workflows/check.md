# Check Workflow

Use this workflow when the user asks ARC to check one or more accessible
Markdown or PDF research notes. The goal is to identify the foundation and then
verify every non-foundation technical claim without exposing the full note body
to proposer agents.

If the user explicitly asks to use `check.md`, treat that as a request to run
this workflow, not merely to discuss or summarize it. Do not stop after a
main-agent inspection unless the workflow is in interactive mode and the user
needs to review obvious issues before proposer-reviewer execution.

This workflow reuses:

```text
workflows/plan.md
workflows/foundation.md
workflows/calculate.md
```

Keep this file note-specific. It owns note parsing, preflight, note-check
intake artifacts, note-output validation, and the note status map. Do not
duplicate plan granularity rules, foundation boundary/schema/version rules,
consensus config details, prompt contracts, human-gate policy, or
calculation-report structure here; update the owning workflow instead.

## Phase 1: Parse And Read Notes

Step 1: For local notes or papers, parse accessible sources before checking
claims:

```bash
arc-paper parse --tex NOTE.tex --pdf NOTE.pdf --id NOTE_ID --json
arc-paper parse --tex NOTE.tex --id NOTE_ID --json
arc-paper parse --pdf NOTE.pdf --id NOTE_ID --json
arc-paper parse --html NOTE.html --id NOTE_ID --json
arc-paper parse --paper-id 0911.3380 --source ar5iv --json
```

All parse modes write the same parsed JSON shape used by ar5iv parsing:
top-level `paper_id`, `parser_version`, `source_hash`, `toc`, `sections`, and
`equations`. TeX/PDF modes may add optional location fields to section and
equation records, but must preserve existing ar5iv keys.

Parsed sources are cached at:

```text
<ARC_PAPER_CACHE>/sources/<paper_ids_safe_dir_name([paper_id])>.json
```

Parsed equation annotations are cached separately at:

```text
<ARC_PAPER_CACHE>/source-annotations/<paper_ids_safe_dir_name([paper_id])>.json
```

Use the parsed JSON as the source of truth for sections, equations, labels,
line ranges, printed equation numbers, and PDF pages. Do not reparse TeX or PDF
inside this workflow. Do not hand-edit parsed JSON to flag bad equations.

The PDF may be larger than the TeX source, such as a whole book containing one
section's TeX. In that case, rely on `arc-paper parse --tex NOTE.tex --pdf
BOOK.pdf --id NOTE_ID` to locate equation numbers and pages from nearby prose,
equation tokens, and printed number candidates.

If a PDF is provided but cannot be used, for example because `pdftotext` is not
installed or returns no text, `arc-paper parse` prints a warning and reports it
in `meta.warnings`.

Step 2: Read Markdown files directly. Extract PDF-only notes with available
host tools. Preserve source file names, page numbers, section headings, and
nearby labels when available.

Step 3: Treat note contents as source context for the main agent only. Do not
provide the full note body to proposer agents. Proposers receive only filtered
foundation context, task contracts, and accepted prior outputs.

Step 4: If the notes cite arXiv IDs, DOI, INSPIRE records, or paper titles, use
ARC paper tools to resolve them before checking claims.

## Phase 2: Main-Agent Preflight

Step 1: Before foundation classification or proposer-reviewer checks, the main
agent must inspect the notes directly and list any obvious typos, inconsistent
conventions, missing factors, sign mistakes, malformed equations, or
target/source mapping problems it immediately spots.

Step 2: Mark these as preflight findings, not verified proposer-reviewer
results.

Step 3: In `interactive` mode, pause after preflight and ask the user to review
the obvious issues before launching proposer-reviewer execution. In `auto`
mode, do not pause; continue to Phase 3 after recording the findings.

## Phase 3: Prepare Note-Check Intake

Step 1: Write a short triage artifact:

```text
<project-dir>/calculate/<run-id>/note-check-triage.json
<project-dir>/calculate/<run-id>/initial-note-check.md
<project-dir>/initial-note-check.md
```

Include parsed source locations, note items, and Phase 2 preflight findings.
For parsed TeX notes, build `note-check-triage.json` from `sections[]` and
`equations[]`. Each note item should carry `source_id`, the parsed equation
`id`, TeX line range, section, printed equation number when available, and PDF
page when available.

Step 2: If the user specifies the foundation, record that instruction in the
triage artifact as input for `plan.md`. Do not classify final foundation,
claims, or context-only items here; `plan.md` owns that separation.

After writing the project-level Markdown report, call
MCP `md2pdf(input="<project-dir>/initial-note-check.md")`. It starts a
background PDF job; record the returned job id if present and do not wait
before continuing.

## Phase 4: Reuse Calculation Workflows

Step 1: Treat the note check as an explicit calculation idea: verify the note
claims from the accepted foundation, with no new conjecture unless required to
check a claim.

Step 2: Execute `plan.md` with `note-check-triage.json` as the selected idea
artifact. `plan.md` owns the foundation boundary, claim/step granularity,
blind-reference planning, and proposer-visible secrecy rules.

Step 3: Execute `foundation.md` from `plan.json`. It owns foundation JSON,
source, confidence, convention, and versioning rules.

Step 4: Execute `calculate.md` in note-check mode. It owns blind reference
config shape, proposer/reviewer prompt contracts, human-gate behavior,
blocked/refinement handling, human-resolved continuation, and
`calculation-report.md` generation.

## Phase 5: Mirror Human Resolution To Note Triage

When `calculate.md` records a human-resolved result, mirror the same resolution
object into `note-check-triage.json`. Mark the note claim `human_resolved`, not
`verified`. Keep later-use and foundation-promotion decisions governed by
`calculate.md` and `foundation.md`.

## Phase 6: Validate

Step 1: After `calculate.md` produces a report or blocked report, run:

```bash
arc-paper validate-note-check <project-dir>/calculate/<run-id> --json
```

Step 2: Treat the report as final only after validation passes. If validation
fails and the run cannot be completed, keep the report as `blocked_partial` and
include the validation errors.

Step 3: If checking shows that a cached parsed equation is problematic, ask the
user to choose either a cache annotation or a re-cache.

For annotation:

```bash
arc-paper mark-parsed-equation NOTE_ID --equation-id eq_00042 \
  --status problematic --reason "Short reason from the check"
```

For re-cache, update the parse input and rerun `arc-paper parse` with the same
`--id`. Existing annotations are keyed to the old `source_hash` and will not
overlay the newly parsed equation view.

## Phase 7: Note Status Map

The note status map in the final report must identify whether the foundation
was user-specified or inferred. For each note item, report one status:

```text
foundation
verified
human_resolved
reference_disagrees
unresolved
context_only
```

For each status, include the source note path and page, section, heading, or
label when available. Do not claim a note detail is verified unless
`calculate.md` accepted it by consensus or the user explicitly resolved
it.
