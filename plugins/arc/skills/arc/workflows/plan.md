# Plan Workflow

Use this workflow when a research task, planning request, or note-check handoff needs a human-readable calculation work note. `plan.md owns work-note structure`, `initial foundations`, `accepted-premise promotion`, `ready-step boundaries`, `rough-step planning`, and reviewer-only target placement.
It does not own consensus execution; `calculate.md` owns current-step status/result recording. For Markdown or PDF note parsing, use `check.md`; for execution and result capture, use `calculate.md`. When another phase needs behavior outside these boundaries, refer to the owning workflow.

No JSON file is the source of truth for planning. Runtime JSON, consensus config, and execution records belong to the workflows and packages that own runtime execution.

Read optional planning requests from:

```text
<project-dir>/calculate/<run-id>/planning-request.md
```

when present. Do not create or overwrite `planning-request.md` from this
workflow.

Write artifacts under:

```text
<project-dir>/work-note.md
<project-dir>/calculate/<run-id>/work-notes/work-note-v001.md
```

Write immutable version first, then mirror newest version to root
`<project-dir>/work-note.md`. Never edit old `work-note-vNNN.md` files.

Use this Work Note template:

```md
# Work Note

## Task
## Physics Background And Logic Flow
## Notation And Conventions
## Axioms And Starting Points
## Accepted Derived Results
## Validation-Only References
## Detailed Steps Ready To Calculate
## Rough Steps For Later Planning
## Equation Coverage Ledger
## Reviewer-Only Targets
## Calculation Status
## Open Questions
## Revision History
## Journal
## Source Audit Trail
```

Each equation-heavy section must include enough prose for a physicist to follow the argument. The work note is not only equations and must be at least as clear as the original note/source context. Main text explains physics; the Journal keeps verbatim or compact execution history. Use logic-flow sentences such as: `Use F1 and F2 to derive S3`.

## Inputs

Read `<project-dir>/context.json` only as routing metadata: project directory,
run id, automation mode, source locations, and host hints. Do not infer the
scientific task from `context.json` alone.

If `<project-dir>/calculate/<run-id>/planning-request.md` exists, use it as the
planning request. Otherwise use the user's intent. If the task came from
`check.md`, preserve note-check secrecy rules. If the task came from
`calculate.md`, treat proposed reusable results or blocked-step notes as a
request for planning judgment, not as automatic edits.

## Phase 1: Establish Foundation Boundary

Step 1: Identify the task, target quantity or claim family, source context, and
coverage requirement. Write this in `## Task` and `## Physics Background And
Logic Flow`.

For note-check tasks with parsed equations, create an `## Equation Coverage
Ledger`. Map every parsed equation id from the source inventory to a ready step,
rough step, or skipped-with-reason entry. Steps may cover multiple equations,
but the ledger must name the exact equation ids or equation-id ranges covered by
that step. A broad section label, source span, or source_anchor alone is not
enough coverage accounting.

Step 2: Add initial foundations to `## Axioms And Starting Points`. Foundations
may be definitions, conventions, axioms, variational principles, symmetry
assumptions, approximation regimes, boundary conditions, or accepted starting
equations. Do not accept a source-derived equation as foundation merely because
it appears early, is boxed, or is used later.

Step 3: Put checked reusable derivations in `## Accepted Derived Results`.
`calculate.md` may propose a candidate reusable result through a planning
request, but `plan.md` decides promotion into Accepted Derived Results, allowed
premises, and future dependencies. This is accepted-premise promotion.

Step 4: Put cross-check formulas, benchmark cases, source claims being checked,
and hidden answer targets in validation/reviewer sections. Validation-only
references can test results but are not allowed premises.

## Phase 2: Draft Calculation Structure

Step 1: Write the physics prose first. The note must explain what the derivation
is trying to show, why each block follows, what assumptions are active, and how
dependencies flow. Avoid turning extracted equations into a mechanical step
list.

Step 2: Choose ready-step boundaries for `## Detailed Steps Ready To Calculate`.
Use the largest coherent chunks current agents can calculate and reviewers can
check reliably. Split only when context, algebra, ambiguity, or target secrecy
requires it. Do not split by raw equation count.

Ordering rule: arrange derivation blocks by dependency/topological order. When multiple blocks have the same dependency priority, put the block with the earliest source anchor first: source line number if known, otherwise first target equation, page, or stable block id. Use this order for accepted results, detailed steps, and rough steps. Keep Journal and Revision History chronological.

Step 3: For each detailed step, include this contract:

```text
id
status: ready | accepted | blocked | pending
target quantity
allowed premises
forbidden inputs
proposer-visible context
reviewer-only target ids
expected form without target formula
acceptance standard
source-discrepancy handling
why follows
```

At the end of every step, state calculate which quantity, in terms of which quantity,
and what is forbidden as input. Do not disclose the exact expected expression
or expected final formula; ask proposers to derive the target quantity in terms of named dependencies.

For note-check steps that may contradict the source, add
`source-discrepancy handling`: define `confirmed_source_error` versus
`likely_source_error` or `ambiguous_convention`. Automatic acceptance as
`confirmed_source_error` requires blind proposer, reviewer, and main-agent
agreement; accepted premises only; no convention dependence; and reviewer
judgment that no human convention choice is needed. `calculate.md` owns any
auto-mode high-confidence agent resolution for other discrepancy classes.

Step 4: Write deferred work in `## Rough Steps For Later Planning`. Rough-step
planning records dependency order, likely inputs, risk, and expansion triggers.
Rough steps are not executable consensus steps. Future planning may refine them
after accepted results, blocked reasons, reviewer reports, or observed agent
ability are known.

Rough steps are only for deferred/pending work. When `plan.md` promotes a rough step into `## Detailed Steps Ready To Calculate`, remove that step from `## Rough Steps For Later Planning` in the same work-note version. Accepted, ready, or blocked detailed steps must not remain in the rough-step list.
Before the workflow finishes, every rough-step item must be adjudicated: promoted to a ready detailed step, removed/marked obsolete because its trigger did not fire, or recorded as an explicit stop condition in `Open Questions` or `Calculation Status`.

Step 5: Update `## Equation Coverage Ledger` whenever ready or rough steps
change. Keep equation ids traceable even when one coherent derivation step
covers several equations. If a ready step will run with source tools disabled,
its proposer-visible context must include `source_excerpt`, exact displayed
formulas, or accepted derivations sufficient to perform the check. Do not mark a
source-span-only ready step executable when proposers cannot access the source.

Step 6: Put hidden source answers in `## Reviewer-Only Targets`. For note-check
tasks, write clean proposer-facing explanation using only context up to the
target. Do not include the target equation or later text in proposer-visible
context. The target appears only in Reviewer-Only Targets, keyed by target id.
This creates a blind reference check with reviewer-only reference claims.

## Phase 3: Status And Revision Recording

Step 1: Use `## Calculation Status` only to summarize current step state:
ready, accepted, blocked, or pending. `calculate.md` records consensus execution
details and current-step result-status; plan.md updates structure only when a
planning decision is needed.

Step 2: Use `## Revision History` for version-level changes: added foundation,
promoted accepted result, changed ready-step boundaries, expanded rough step,
or moved a target into reviewer-only storage.

Step 3: Use `## Journal` for compact chronology. Preserve useful verbatim
execution notes, planning-request excerpts, PDF job ids, and reviewer decisions.
Keep it factual and short.

Step 4: Use `## Source Audit Trail` for every source that shaped the work note:
paper ids, note paths, sections, equations, commands, MCP tools, URLs, and why
each source matters.

Step 5: Any accepted or promoted work-note content whose acceptance depends on a specific human expert answer resolving an unresolved scientific acceptance question is human-resolved content. This does not include ordinary user task instructions, source excerpts, or constraints.
Add the literal marker `[human-resolved]` beside the content, and only color that marker's background dark blue with white text. Do not color the surrounding prose. Do not color the surrounding equations. If color is stripped or unavailable, the marker remains authoritative. For PDF-oriented Markdown, use the `calculate.md` marker-background template and write raw LaTeX directly in prose, for example: \colorbox{archumanresolved}{\textcolor{white}{[human-resolved]}}. Do not use Markdown code spans.

Step 6: For note-check content classified as `confirmed_source_error`, mark the source issue with the literal marker `[confirmed source issue]`, and only color that marker's background dark red with white text. Do not color the surrounding prose. Do not color the surrounding equations. For `likely_source_error` or `ambiguous_convention`, use red only for the `calculate.md` auto-mode `[agent-resolved decision]` marker; otherwise ask a human expert question before acceptance. For PDF-oriented Markdown, use the `calculate.md` marker-background template and write raw LaTeX directly in prose, for example: \colorbox{arcsourceissue}{\textcolor{white}{[confirmed source issue]}}. Do not use Markdown code spans.

## Phase 4: Version And Export
Step 1: Find the highest existing immutable version under
`<project-dir>/calculate/<run-id>/work-notes/work-note-vNNN.md`.
Step 2: Build final content before writing files and note planned PDF export in
`## Journal`.
Step 3: Write the next immutable version, starting with `work-note-v001.md`;
never overwrite an old version.
Step 4: Mirror it to `<project-dir>/work-note.md`.
Step 5: Call `md2pdf(input="<project-dir>/work-note.md")` in the background; do not wait.
Record any job id in host logs or the next work-note version, not by editing an
old immutable version.

## Phase 5: Review
Step 1: Review the plan before execution. If the host and workflow permissions
allow delegation, use an independent reviewer. Otherwise the main agent must
perform the same review.

Step 2: Check that foundations are separated from derived results, accepted derived results were actually accepted, validation-only references are not premises, ready steps have complete contracts, rough steps are not executable, target secrecy is preserved, no accepted/ready/blocked step is duplicated in `## Rough Steps For Later Planning`, all rough-step triggers are adjudicated, every parsed equation id is represented in the Equation Coverage Ledger, ready steps with disabled source tools have enough proposer-visible source excerpt or exact formula context, PDF color markers are not inside code spans, and source coverage is enough for the task.

Step 3: If review finds gaps, build final content with the planned PDF export
noted in the Journal, write a new immutable work-note version, mirror it to
root, and call background `md2pdf`. Record any returned job id in host/run logs
or the next work-note version, not by editing the immutable version just written.
Then hand off ready steps to `calculate.md` for consensus execution.
