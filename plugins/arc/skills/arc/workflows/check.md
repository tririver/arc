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

Heavy Workload Rule: This workflow can be long. Heavy workload and many
claims/equations are expected runtime facts; workload size is not a stop condition.
The agent must not skip mandatory phases or shorten requested coverage because work is heavy.
Continue until requested coverage is complete, a concrete workflow stop condition applies, or the user explicitly stops the workflow.

`calculate.md` uses high reasoning effort by default for mathematical derivations; lower it only for cheap exploratory runs.
Follow `rules/math_typeset.md` for math and TeX snippets in ARC-generated
Markdown reports and planning handoffs.

## Phase 1: Parse And Read Notes

Step 1: For local notes or papers, parse accessible sources before checking
claims. See `manuals/arc-paper.md` for parse commands and parsed-source reads.

This phase is a parsing workflow, not a TeX build workflow. Do not run
`pdflatex`, `latexmk`, `chktex`, or other TeX compilers/linters as part of
content checking unless the user explicitly asks for build/typesetting QA.
For TeX/PDF notes, ARC parsed paper output is the source of truth for sections,
equations, line anchors, and PDF page anchors.

Step 2: Read parsed sources through ARC paper commands. Treat command output as
the source of truth for parsed note structure.

For Markdown notes, read the Markdown file directly and preserve source file
names and headings. For PDF-only notes, parse first, then read parsed sections
and equations with commands from `manuals/arc-paper.md`.

## Phase 2: Main-Agent Preflight

Step 1: Inspect the notes directly before proposer-reviewer execution. List
obvious typos, inconsistent conventions, missing factors, sign mistakes,
malformed equations, or target/source mapping problems.

Step 2: Record these as preflight findings, not verified results.

Step 3: In `interactive` mode, pause after preflight, before writing the
planning handoff, and ask the user to review obvious issues. In `auto` mode,
continue after recording findings.

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
- parsed equation inventory and an instruction that `plan.md` must write an
  `Equation Coverage Ledger`
- coverage requirements and claims to check
- preflight findings
- user-specified premise instructions, preserved as input and not inferred
- note source audit handoff: which paths, source IDs, sections, equations, and
  commands should appear later in the work-note Source Audit Trail

Do not copy full note prose or all equation bodies into the planning request.
Do not write one task item per equation. Do require the handoff to preserve
traceability: every parsed equation id must later map to a ready step, rough
step, or skipped-with-reason entry in the Equation Coverage Ledger. A ready step
may cover multiple equations, but each covered equation id or equation-id range
must be explicit. Do not classify final premises, derived claims, or
context-only items here; `plan.md` owns that separation.

Do not pass the full note body to proposer agents. `plan.md` writes
proposer-facing work-note context and reviewer-only target IDs. The work note
must be at least as clear as the original note prefix while hiding the target
equation and later text from proposers. Source claims that need blind reference
checking should be represented for reviewers as `reviewer_reference_claim`.
When source tools are disabled for a calculation step, `source_anchor` alone is
not enough: the ready-step packet should include `source_excerpt`, exact
displayed formulas, or accepted prior derivations sufficient for proposers to
perform the check without reading the original source.

After writing `<project-dir>/initial-note-check.md`, follow
`manuals/arc-jobs.md` Markdown Report Export for
`<project-dir>/initial-note-check.md`. This report-export gate
is not satisfied until `md2pdf` has been started or a `WARNING:` with the exact
blocker is recorded. Do not wait for PDF completion.
If PDF generation appears bugged, report it and continue this workflow; do not
debug or fix PDF generation unless the user explicitly asks.
This `md2pdf` call applies only to ARC-generated Markdown reports, not to the
original TeX/PDF note being checked.
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
run. Before final response, adjudicate every rough-step item: promote and run
triggered items, remove or mark false-trigger items as obsolete/not triggered,
or record a concrete stop condition in `Open Questions` or
`Calculation Status`.

## Phase 5: Note-Check Status

Do not create separate note-check triage JSON. Note-check status is recorded in
the work note by the owning workflows. When `calculate.md` or `plan.md` records
note status, it should do so inline near the checked or human-resolved item in
the work note. Human-resolved accepted content must follow the owning
workflow's marker-only color rule for `human-resolved`.

If checking shows that a parsed equation is problematic, ask the user to choose
either an ARC paper annotation or a re-parse. See `manuals/arc-paper.md`.
