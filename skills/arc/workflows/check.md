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

## Phase 1: Read Notes

Step 1: Read Markdown files directly. Extract PDF text with available host
tools. Preserve source file names, page numbers, section headings, and nearby
labels when available.

Step 2: Treat note contents as source context for the main agent only. Do not
provide the full note body to proposer agents. Proposers receive only filtered
foundation context, task contracts, and accepted prior outputs.

Step 3: If the notes cite arXiv IDs, DOI, INSPIRE records, or paper titles, use
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

Step 2: Execute `plan.md`. For each item in `claims_to_check`, create
a blind reference check. The proposer prompt must name the quantity to derive,
its dependencies, and allowed checked inputs, but must not disclose the note's
target formula or result.

Step 3: Execute `foundation.md`. The foundation contains only accepted
definitions, axioms, conventions, and truly foundational equations.

Step 4: Execute `calculate.md`. Put each note claim into
`reviewer_reference_claim` only. The reviewer may compare proposer derivations
against the note claim; proposers must not see the note claim unless the user
explicitly requests non-blind checking.

Step 5: Directly trigger proposer-reviewer execution through
`arc-llm proposers-reviewer-consensus --config <config> --json` or the
equivalent host/MCP wrapper. Do not write the final `calculation-report.md`
until proposer-reviewer execution has run, unless the report is explicitly a
blocked or partial-status report that says consensus did not complete.

## Phase 4: Report

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
