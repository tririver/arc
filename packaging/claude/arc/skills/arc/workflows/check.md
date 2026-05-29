# Check Workflow

Use this workflow when the user asks ARC to check one or more accessible
Markdown or PDF research notes. The goal is to turn note content into a
source-extracted calculation request that can be handled by the standard
plan/foundation/calculate workflow.

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

Keep this file note-specific. It owns note parsing, preflight, source-extracted
request artifact creation, and optional note-output validation. Do not duplicate
plan granularity rules, foundation boundary/schema/version rules, consensus
config details, prompt contracts, human-gate policy, human-resolution policy,
or calculation-report structure here; update the owning workflow instead.

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

The PDF may be larger than the TeX source, such as a whole book containing one
section's TeX. In that case, use the TeX+PDF parse command so ARC paper can
locate equation numbers and pages from nearby prose, equation tokens, and
printed number candidates. If a PDF is provided but cannot be used, rely on
warnings returned by `arc-paper parse`.

Step 2: Read the parsed source through ARC paper commands. Treat ARC paper
command output as the source of truth for parsed note structure. Do not read,
depend on, or edit package internals directly.

```bash
arc-paper get-parsed NOTE_ID --json
arc-paper get-parsed-toc NOTE_ID --json
arc-paper get-parsed-section NOTE_ID --section SECTION_ID --json
arc-paper get-parsed-equations NOTE_ID --json
arc-paper get-parsed-equation NOTE_ID --equation-id EQUATION_ID --json
```

For Markdown notes, read the Markdown file directly and preserve source file
names and headings. For PDF-only notes, use the parse command in Step 1 first,
then read the parsed source with the ARC paper commands above.

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

## Phase 3: Prepare Source-Extracted Request

Step 1: Write a short task-to-be-planned artifact:

```text
<project-dir>/calculate/<run-id>/task-to-be-planned.json
<project-dir>/calculate/<run-id>/initial-note-check.md
<project-dir>/initial-note-check.md
```

Build `task-to-be-planned.json` as a compact planning handoff, not as a plan
or a source copy. Include the user's check request, parsed source IDs, original
source paths when local, the ARC paper commands `plan.md` should use to read
the parsed source, the coverage scope, Phase 2 preflight findings, and any
user-specified foundation instruction.

Do not copy note prose, equation bodies, or derived source maps into this
artifact. Do not write one task item per equation. Do not classify final
foundation, claims, or context-only items here; `plan.md` owns that separation.

After writing the project-level Markdown report, call
MCP `md2pdf(input="<project-dir>/initial-note-check.md")`. It starts a
background PDF job; record the returned job id if present and do not wait
before continuing.

## Phase 4: Reuse Calculation Workflows

Step 1: Run `plan.md`. It reads
`<project-dir>/calculate/<run-id>/task-to-be-planned.json`. `plan.md` owns the
foundation boundary, claim/step granularity, blind-reference planning, and
proposer-visible secrecy rules.

Step 2: Execute `foundation.md` from `plan.json`. It owns foundation JSON,
source, confidence, convention, and versioning rules.

Step 3: Execute `calculate.md`. It owns blind reference config shape,
proposer/reviewer prompt contracts, human-gate behavior, blocked/refinement
handling, human-resolved continuation, and `calculation-report.md` generation.

## Phase 5: Validate

Step 1: After `calculate.md` produces a report or blocked report, write or
update `<project-dir>/calculate/<run-id>/note-check-triage.json` as the
validation status map from the final plan, source items, consensus results, and
human-resolved items. This is not input to `plan.md`; the planning input remains
`task-to-be-planned.json`.

Step 2: Run:

```bash
arc-paper validate-note-check <project-dir>/calculate/<run-id> --json
```

Step 3: Treat the report as final only after validation passes. If validation
fails and the run cannot be completed, keep the report as `blocked_partial` and
include the validation errors.

Step 4: If checking shows that a parsed equation is problematic, ask the user
to choose either an ARC paper annotation or a re-parse.

For annotation:

```bash
arc-paper mark-parsed-equation NOTE_ID --equation-id eq_00042 \
  --status problematic --reason "Short reason from the check"
```

For re-parse, update the parse input and rerun `arc-paper parse` with the same
`--id`.
