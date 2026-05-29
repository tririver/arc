# Plan Workflow

Use this workflow when a task to be planned is available.
The output is a careful, reviewable rolling plan. Do not start deriving
equations here.

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

Read `<project-dir>/context.json`, the task-to-be-planned artifact, domain
Markdown/JSON, domain summaries, and available domain graph files. The
task-to-be-planned artifact may be a user-written task, generated idea,
source-extracted request artifact with source items, preflight findings, and
locations, or a plan-expansion request written by `calculate.md`. Keep the
user's exact scientific intent visible in the plan.

## Phase 1: Gather Evidence

Step 1: Use ARC paper and domain tools before internet search. Use the CLI and
MCP surfaces documented in `manuals/arc-paper.md` and `manuals/arc-domain.md`;
read `manuals/arc-mcp.md` before calling MCP tools.

Step 2: Use internet search only after ARC checks when literature may be
uncached, recent, outside arXiv/INSPIRE, or ambiguous.

Step 3: Record every useful source in `literature_checks` with the tool,
command, paper id, section, and reason it matters.

## Phase 2: Separate What Can Be Trusted

Step 1: Identify first principles. These may be axioms, definitions, symmetry
requirements, standard variational principles, conserved quantities, or other
starting points that do not depend on the target paper's derivation.

Step 2: Identify useful results from papers. Treat them as claims to check
later unless they are accepted first principles. Do not accept a target paper's
derived equation just because it is published.

Step 3: Mark validation-only results separately. These are results useful for
cross-checks, limits, benchmark cases, or sanity tests, but not allowed as
inputs to the new derivation.

Step 4: When the task-to-be-planned artifact contains source-extracted items, split
those items into `foundation`, `claims_to_check`, and `context_only`. If an
item could be either foundation or a derived claim, put it in
`claims_to_check`. Do not accept a source-derived equation as foundation merely
because it appears early, is boxed, or is used later in the source.

## Phase 3: Build The Rolling Calculation Plan

Step 1: Make the first calculation step the first nontrivial derivation after
the accepted foundation setup.

Step 2: Plan adaptively. Use the largest coherent semantic chunks that current
agents can check reliably, then split only when the chunk is too hard or too
ambiguous. Do not make one equation equal one step by default. Stronger agents
should naturally receive larger chunks, and weaker agents should receive smaller
chunks.

Use these target budgets as planning priors, not hard caps:

```text
exam-style exercise: 3-5 detailed steps
graduate take-home problem: 5-10 detailed steps
simple research task: 10-20 detailed steps
challenging research task: 20-50 detailed steps
```

For source-checking tasks with many equations, group equations by meaning and
dependency, not by raw equation count. A step may contain several checkpoint
equations when they are one derivation chain, one physical argument, one
quantity family, one approximation regime, or one source-context unit. Each
checkpoint must have its own target quantity and verification target. If one
step contains multiple checkpoint equations, state why they belong together and
how the reviewer should map checkpoint names to reviewer-only reference claims.

Step 3: Split the plan into near-term detailed work and later macro work:

```text
detailed_steps: executable steps for the next coherent batch
macro_plan: rough later blocks, not executable until expanded
```

Make only the first few steps detailed enough to run with high confidence. For
later work, keep a macro-plan with the dependency order, source ranges, likely
checkpoints, expected inputs, difficulty estimate, and expansion trigger. This
lets later planning use accepted results, observed failure modes, and measured
agent ability instead of guessing the entire project upfront.

Step 4: For every `detailed_steps[]` entry, specify:

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
proposer_context_slice
reviewer_reference_claim_ids
verification
```

Step 5: At the end of every detailed step, make the quantity contract explicit:
calculate which quantity, in terms of which quantity, and what is not allowed
as an input. Do not disclose the exact expected expression or expected final
formula. Instead, say to derive the target quantity in terms of named
dependencies. If this cannot be stated clearly, split the step again.

For equations quoted from a reference or collaborator note that need checking,
do not disclose the target reference equation in `prompt`, `allowed_inputs`, or
`expected_output`. Make each checkpoint a blind reference check: proposers
derive the quantity from named dependencies, and the execute workflow supplies
targets only as reviewer-only reference claims.

Step 6: For every `macro_plan[]` entry, specify:

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

Macro blocks are not consensus steps. They are promises to replan later with
more evidence.

Step 7: When this workflow is called with a plan-expansion request, preserve
accepted earlier detailed steps and expand only the requested macro block or
blocked region. Use current evidence: accepted outputs, reviewer reports,
retry counts, blocked reasons, useful context packets, and observed agent
ability. If agents handled broad chunks well, expand larger chunks. If agents
struggled, split more finely. Record why granularity changed.

Step 8: Write `plan.json` and render the current rolling plan. On the first
planning pass, write the initial snapshot directly to both
`<project-dir>/calculate/<run-id>/initial-plan.md` and
`<project-dir>/initial-plan.md`. On plan-expansion passes, do not rewrite
`initial-plan.md`; preserve it as the first snapshot. Always write the current
complete plan view to both `<project-dir>/calculate/<run-id>/latest-plan.md`
and `<project-dir>/latest-plan.md`. The Markdown is not a status stub: it must
show the evidence summary, foundation boundary, validation-only results,
`detailed_steps`, `macro_plan`, each detailed step's quantity contracts,
context policy, checkpoints, allowed inputs, substeps, expected output,
verification method, and plan-expansion history when present. The JSON is the
source of truth for later workflow phases, but the Markdown must be complete
enough for a human to review the current detailed work and macro-plan without
opening the JSON.

After writing the project-level Markdown reports, call MCP
`md2pdf(input="<project-dir>/latest-plan.md")`. On the first planning pass,
also call `md2pdf(input="<project-dir>/initial-plan.md")`. Each starts a
background PDF job; record returned job ids if present and do not wait before
continuing.

## Phase 4: Review The Plan

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
the first calculation step is clear
difficult steps have enough substeps
```

Step 3: If the review finds gaps, revise `plan.json` and render the current
plan to both latest-plan Markdown paths, then call
`md2pdf(input="<project-dir>/latest-plan.md")` in the background. Do not rewrite
`initial-plan.md` after the first snapshot. Proceed only when the review is
recorded in `plan.json` and the latest-plan Markdown artifacts.
