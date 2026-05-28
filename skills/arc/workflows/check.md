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

## Phase 2: Mark The Foundation

Step 1: If the user specifies the foundation, use that as the candidate
foundation and record that it was user-specified.

Step 2: Otherwise infer the candidate foundation from note items that are
definitions, conventions, axioms, standard starting principles, or equations
explicitly allowed as assumptions.

Step 3: Split note items into:

```text
foundation
claims_to_check
context_only
```

If an item could be either foundation or a derived claim, put it in
`claims_to_check`. Do not make note-derived equations foundation merely because
they appear early, are boxed, or are used later in the note.

Step 4: Write a short triage artifact:

```text
<project-dir>/calculate/<run-id>/note-check-triage.json
<project-dir>/calculate/<run-id>/initial-note-check.md
<project-dir>/initial-note-check.md
```

For parsed TeX notes, build `note-check-triage.json` from `sections[]` and
`equations[]`. Each `claims_to_check` item should carry `source_id`, the parsed
equation `id`, TeX line range, section, printed equation number when available,
and PDF page when available.

After writing the project-level Markdown report, call
MCP `md2pdf(input="<project-dir>/initial-note-check.md")`. It starts a
background PDF job; record the returned job id if present and do not wait
before continuing.

### Phase 2a: Main-Agent Preflight

Step 1: Before running proposer-reviewer checks, the main agent must inspect
the notes directly and list any obvious typos, inconsistent conventions,
missing factors, sign mistakes, malformed equations, or target/source mapping
problems it immediately spots.

Step 2: Record these preflight findings in `initial-note-check.md` and
`note-check-triage.json`. Mark them as preflight findings, not verified
proposer-reviewer results.

Step 3: In `interactive` mode, pause after preflight and ask the user to review
the obvious issues before launching proposer-reviewer execution. In `auto`
mode, do not pause; continue to Phase 3 after recording the findings.

## Phase 3: Reuse Calculation Workflows

Step 1: Treat the note check as an explicit calculation idea: verify the note
claims from the accepted foundation, with no new conjecture unless required to
check a claim.

Step 2: Execute `plan.md`. For each item in `claims_to_check`, create a blind
reference check. Prefer one equation per step, or a tightly coupled equation
pair when the derivation cannot be separated. The proposer prompt must name the
quantity to derive, its dependencies, and allowed checked inputs, but must not
disclose the note's target formula or result.

For parsed TeX notes, identify the target by stable location:

```text
Target equation: eq_00042
Location: NOTE.tex lines 360-362, PDF page 14, printed equation (9.12)
Task: derive the named quantity from the allowed inputs.
Allowed inputs: accepted foundation items only.
Do not use the note formula.
```

Step 3: Execute `foundation.md`. The foundation contains only accepted
definitions, axioms, conventions, and truly foundational equations.

Step 4: Execute `calculate.md`. Put each note claim into
`reviewer_reference_claim` only. The reviewer may compare proposer derivations
against the note claim; proposers must not see the note claim unless the user
explicitly requests non-blind checking.

For parsed TeX notes, reviewer-only reference claims should include
the parsed equation `id`, raw TeX, normalized LaTeX, printed equation number, PDF/line
location, section, and nearby text.

Step 5: Directly trigger proposer-reviewer execution through
`arc-llm proposers-reviewer-consensus --config <config> --json` or the
equivalent host/MCP wrapper. Do not write the final `calculation-report.md`
until proposer-reviewer execution has run, unless the report is explicitly a
blocked or partial-status report that says consensus did not complete.

## Phase 4: Validate

Step 1: Before writing the final report, run:

```bash
arc-paper validate-note-check <project-dir>/calculate/<run-id> --json
```

Step 2: Do not write final `calculation-report.md` unless validation passes.
If validation fails but the run cannot be completed, write the report only with
status `blocked_partial` and include the validation errors.

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

## Phase 5: Report

The final report must identify whether the foundation was user-specified or
inferred. For each note item, report one status:

```text
foundation
verified
reference_disagrees
unresolved
context_only
```

For each status, include the source note path and page, section, heading, or
label when available. Do not claim a note detail is verified unless
`calculate.md` accepted it by consensus or the user explicitly resolved
it.
