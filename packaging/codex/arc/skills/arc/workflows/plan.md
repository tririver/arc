# Plan Workflow

Use this workflow when a task to be planned is available. It writes a rolling
plan with executable `detailed_steps` and deferred `macro_plan` blocks.
`calculate.md` executes only detailed steps; later blocks return here for
expansion before execution.

At any step, if more information would improve the plan, gather it. Use ARC
paper/domain tools for paper and domain evidence, and use internet search when
it helps resolve missing, recent, or ambiguous context. Record every useful
source in `literature_checks` with the tool, command or URL, paper id or
section when available, and why it matters.

Write artifacts under:

```text
<project-dir>/calculate/<run-id>/plan.json
<project-dir>/calculate/<run-id>/initial-plan.md
<project-dir>/calculate/<run-id>/latest-plan.md
<project-dir>/initial-plan.md
<project-dir>/latest-plan.md
```

`plan.json` must use `schema_version: "arc.plan.v1"`.

## Inputs

Read `<project-dir>/context.json` as project routing metadata, not as the
task itself. First check
`<project-dir>/calculate/<run-id>/task-to-be-planned.json`. If it does not
exist, use the user's intent passed to this workflow. Do not infer the
scientific task from `context.json` alone.

This workflow must produce a plan ready for `foundation.md` and `calculate.md`;
use that goal to decide what evidence to gather.

Before planning, make sure the task is clear enough to identify the quantity,
claim, source items, or research goal being planned. If automation is not
`auto`, ask the user about any ambiguity before writing `plan.json`. In `auto`
mode, use the safest explicit interpretation, record assumptions in `plan.json`,
and preserve warnings in `latest-plan.md`.

## Phase 1: Separate What Can Be Trusted

Step 1: Identify first principles. These may be axioms, definitions, symmetry
requirements, standard variational principles, conserved quantities, or other
starting points that do not depend on the target paper's derivation.

Step 2: Identify useful results from papers. Treat them as claims to check
later unless they are accepted first principles. Do not accept a target paper's
derived equation just because it is published.

Step 3: Mark validation-only results separately. These are results useful for
cross-checks, limits, benchmark cases, or sanity tests, but not allowed as
inputs to the new derivation.

Step 4: When the task-to-be-planned artifact contains source-extracted items,
split those items into `foundation`, `claims_to_check`, and `context_only`. If
an item could be either foundation or a derived claim, put it in
`claims_to_check`. Do not accept a source-derived equation as foundation merely
because it appears early, is boxed, or is used later in the source.

## Phase 2: Build The Rolling Calculation Plan

Step 1: Choose the planning horizon. Write near-term executable
`detailed_steps` and rough later `macro_plan` blocks:

```text
detailed_steps: executable semantic chunks for the next coherent batch
macro_plan: later dependency-ordered blocks, not executable until expanded
```

The first detailed step starts at the first nontrivial derivation after the
accepted foundation setup. Make the current executable batch detailed enough to
run with high confidence. Leave later work in `macro_plan` so future planning
can use accepted results, observed failure modes, and measured agent ability.

Use target budgets as soft guidance for the total expanded plan, not hard caps or a demand for at least 20 steps: exam-style 3-5; graduate take-home 5-10; simple research 10-20; challenging research 20-50.

Step 2: Choose step granularity by meaning and difficulty. Use the largest
coherent chunks current agents can check reliably, then split only when the
chunk is too hard or ambiguous. Do not split by raw equation count. A detailed
step may contain several checkpoint equations when they are one derivation
chain, physical argument, quantity family, approximation regime, or
source-context unit. Each checkpoint must have its own target quantity and
verification target.

For every checkpoint, set `canonical_output` and `comparison_rule` so two
correct proposers calculate the same object and the reviewer can form `A-B`
after declared rewrites. Include variables, coordinates/basis, index positions,
gauge, normalization, approximation order, dropped constants, and final form
when they affect equality. Do not ask for "the metric" or "the power spectrum"
without comparison form; for a metric, require `ds^2` in a specified basis and
symbols such as `dt`, `dr`, `dΩ^2`, `a(t)`, and `k`, or `g_{\mu\nu}` components.
If this would reveal the answer, split or add prerequisites.

Step 3: For every `detailed_steps[]` entry, specify:

```text
step_id
kind: foundation_check | new_calculation
goal
quantity_to_calculate
quantity_dependencies
allowed_inputs
source_context_policy
proposer_reference_packet
checkpoints
substeps
expected_output
verification
expands_macro_block_id
```

`source_context_policy` must say what context is visible to proposers. For
source-checking, provide full context up to the checkpoint being checked, but
not the checkpoint equation itself and not later source text. If a step has
multiple checkpoints, prepare separate redacted context slices for each
checkpoint. If a reference paper is relevant, attach a `proposer_reference_packet`
with the allowed paper sections, equations, or snippets the proposers may use.
Validation-only target formulas remain reviewer-only.

`checkpoints[]` may contain one or many items:

```text
checkpoint_id
source_item_ids
quantity_to_calculate
dependencies
canonical_output
comparison_rule
proposer_context_slice
reviewer_reference_claim_ids
verification
```

At the end of every step, make the quantity contract explicit:
calculate which quantity, in terms of which quantity, and what is not allowed
as an input. Do not disclose the exact expected expression or expected final
formula. Instead, say to derive the target quantity in terms of named
dependencies and in the checkpoint's canonical output form. If this cannot be
stated clearly enough for `A-B` comparison, split the step again.

For equations quoted from a reference or collaborator note that need checking,
do not disclose the target reference equation in `prompt`, `allowed_inputs`, or
`expected_output`. Make each checkpoint a blind reference check: proposers
derive the quantity from named dependencies, and the execute workflow supplies
targets only as reviewer-only reference claims.

Step 4: For every `macro_plan[]` entry, specify:

```text
macro_block_id
goal
source_range
dependency_boundary
likely_checkpoints
expected_inputs_from_detailed_steps
difficulty_estimate: easy | moderate | hard | unknown
suggested_step_budget
expansion_trigger
notes_for_later_planning
```

Macro blocks are not consensus steps; they are promises to replan later with more evidence.

Step 5: When `task-to-be-planned.json` requests expansion or refinement of an
existing plan, preserve accepted earlier detailed steps and expand only the
requested macro block or blocked region. Use current evidence: accepted outputs,
reviewer reports, retry counts, blocked reasons, useful context packets, and
observed agent ability. If agents handled broad chunks well, expand larger
chunks. If agents struggled, split more finely. Record why granularity changed.

Step 6: Write `plan.json` and render the current rolling plan. On the first
planning pass, write the initial snapshot directly to both
`<project-dir>/calculate/<run-id>/initial-plan.md` and
`<project-dir>/initial-plan.md`. On replanning passes, do not rewrite
`initial-plan.md`; preserve it as the first snapshot. Always write the current
complete plan view to both `<project-dir>/calculate/<run-id>/latest-plan.md`
and `<project-dir>/latest-plan.md`. The Markdown is not a status stub: it must
show the evidence summary, foundation boundary, validation-only results,
`detailed_steps`, `macro_plan`, each detailed step's quantity contracts,
context policy, checkpoints, canonical output contracts, comparison rules,
allowed inputs, substeps, expected output, verification method, and plan
revision history when present. The JSON is the source of truth for later
workflow phases, but the Markdown must be complete enough for a human to review
the current detailed work and macro-plan without opening the JSON.

After writing the project-level Markdown reports, call MCP
`md2pdf(input="<project-dir>/latest-plan.md")`. On the first planning pass,
also call `md2pdf(input="<project-dir>/initial-plan.md")`. Each starts a
background PDF job; record returned job ids if present and do not wait before
continuing.

## Phase 3: Review The Plan

Step 1: Review the plan before building the foundation. If the host and
workflow permissions allow delegation, use an independent reviewer agent or
subagent. Otherwise the main agent must perform the same review.

Step 2: The review must check:

```text
first-principles are separated from derived results
useful results are treated skeptically
source coverage is sufficient
detailed steps fit observed or expected agent ability
macro-plan blocks are ordered by real dependencies
checkpoint grouping preserves context and reviewer precision
checkpoint canonical outputs make A-B comparison well-defined
the first calculation step is clear
difficult steps have enough substeps
```

Step 3: If the review finds gaps, revise `plan.json` and render the current
plan to both latest-plan Markdown paths, then call
`md2pdf(input="<project-dir>/latest-plan.md")` in the background. Do not rewrite
`initial-plan.md` after the first snapshot. Proceed only when the review is
recorded in `plan.json` and the latest-plan Markdown artifacts.
