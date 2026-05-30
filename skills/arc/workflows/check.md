# Check Workflow

Use this workflow when the user asks ARC to check accessible Markdown or PDF research notes.
`check.md owns note parsing`, preflight, the concise
`planning-request.md` planning handoff, and note source audit handoff.

In broad requests such as "check this content", "check the paper", or
"verify the note", default requested coverage is the whole accessible note.
Do not silently narrow broad note-check coverage to a few targeted checks.
If the run intentionally samples or prioritizes only part of the note, record
that as partial coverage in `planning-request.md` and keep remaining coverage
pending unless the user explicitly approves a narrower scope.

This workflow reuses:

```text
workflows/plan.md
workflows/calculate.md
```

Keep this file note-specific. It does not define work-note structure,
ready-step boundaries, reviewer-only target placement, runtime settings, or
execution status rules. `plan.md` owns work-note planning and structure.
`calculate.md` owns consensus execution and result recording. When another
phase needs behavior outside note parsing and handoff, refer to the owning workflow.

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

If a PDF is larger than the TeX source, such as a book containing one TeX
section, use the TeX+PDF parse command so ARC paper can locate equation numbers
and pages from nearby prose, equation tokens, and printed number candidates. If
a PDF cannot be used, rely on warnings returned by `arc-paper parse`.

Step 2: Read parsed sources through ARC paper commands. Treat command output as
the source of truth for parsed note structure.

```bash
arc-paper get-parsed NOTE_ID --json
arc-paper get-parsed-toc NOTE_ID --json
arc-paper get-parsed-section NOTE_ID --section SECTION_ID --json
arc-paper get-parsed-equations NOTE_ID --json
arc-paper get-parsed-equation NOTE_ID --equation-id EQUATION_ID --json
```

For Markdown notes, read the Markdown file directly and preserve source file
names and headings. For PDF-only notes, parse first, then read parsed sections
and equations with the commands above.

## Phase 2: Main-Agent Preflight

Step 1: Inspect the notes directly before proposer-reviewer execution. List
obvious typos, inconsistent conventions, missing factors, sign mistakes,
malformed equations, or target/source mapping problems.

Step 2: Record these as preflight findings, not verified results.

Step 3: In `interactive` mode, pause after preflight and ask the user to review
obvious issues. In `auto` mode, continue after recording findings.

## Phase 3: Write Planning Handoff

Write:

```text
<project-dir>/calculate/<run-id>/planning-request.md
<project-dir>/calculate/<run-id>/initial-note-check.md
<project-dir>/initial-note-check.md
```

Build `planning-request.md` as a compact planning handoff, not a plan and not a
source copy. Include:

- user request
- parsed source IDs
- original source paths
- ARC paper commands for parsed sections and equations
- coverage requirements and claims to check
- preflight findings
- user-specified premise instructions, preserved as input and not inferred
- note source audit handoff: which paths, source IDs, sections, equations, and
  commands should appear later in the work-note Source Audit Trail

Do not copy full note prose or all equation bodies into the planning request.
Do not write one task item per equation. Do not classify final premises,
derived claims, or context-only items here; `plan.md` owns that separation.

Do not pass the full note body to proposer agents. `plan.md` writes
proposer-facing work-note context and reviewer-only target IDs. The work note
must be at least as clear as the original note prefix while hiding the target
equation and later text from proposers. Source claims that need blind reference
checking should be represented for reviewers as `reviewer_reference_claim`.

After writing `<project-dir>/initial-note-check.md`, call
`md2pdf(input="<project-dir>/initial-note-check.md")` in the background. Do not wait for PDF completion.
Record any returned job id in host/run logs or later
work-note journal entries.

## Phase 4: Execute Owning Workflows

Step 1: Run `plan.md`. It reads
`<project-dir>/calculate/<run-id>/planning-request.md` and writes
`<project-dir>/work-note.md` plus the first immutable work-note version.
`plan.md` owns work-note structure, foundation boundary, ready-step planning,
blind reference check placement, and proposer-visible secrecy rules.

Step 2: Run `calculate.md`. It executes ready steps from the current work note
and writes the next work-note version. `calculate.md` owns runtime execution,
reviewer judgment, accepted/blocked current-step status, and result recording.

Step 3: Repeat Steps 1 and 2 until the requested note-check coverage is
complete or a workflow stop condition applies. Do not stop only because one
ready step was accepted; if rough or pending coverage remains, return to
`plan.md` to promote the next coherent chunk before the next `calculate.md`
run.

## Phase 5: Note-Check Status

Do not create separate note-check triage JSON. Note-check status is recorded in
the work note by the owning workflows. When `calculate.md` or `plan.md` records
note status, it should do so inline near the checked or human-resolved item in
the work note. Human-resolved accepted content must follow the owning
workflow's marker-only color rule for `human-resolved`.

If checking shows that a parsed equation is problematic, ask the user to choose
either an ARC paper annotation or a re-parse.

For annotation:

```bash
arc-paper mark-parsed-equation NOTE_ID --equation-id eq_00042 \
  --status problematic --reason "Short reason from the check"
```

For re-parse, update the parse input and rerun `arc-paper parse` with the same
`--id`.
