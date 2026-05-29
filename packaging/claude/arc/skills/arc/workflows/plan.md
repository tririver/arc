# Plan Workflow

Use this workflow when a task to be planned is available.
The output is a careful, reviewable plan. Do not start deriving equations here.

Write artifacts under:

```text
<project-dir>/calculate/<run-id>/plan.json
<project-dir>/calculate/<run-id>/initial-plan.md
<project-dir>/initial-plan.md
```

`plan.json` must use `schema_version: "arc.plan.v1"`.

## Inputs

Read `<project-dir>/context.json`, the task-to-be-planned artifact, domain
Markdown/JSON, domain summaries, and available domain graph files. Keep the
user's exact scientific intent visible in the plan.

## Phase 1: Gather Evidence

Step 1: Use ARC paper and domain tools before internet search.

Useful MCP tools:

```text
get_metadata
get_references
get_citers
get_citer_count
get_toc
get_section
search_full_text
get_equation_context
domain_get_summary
domain_get_graph
```

Useful CLI commands:

```bash
arc-paper get-metadata <paper-id> --json
arc-paper get-references <paper-id> --enrich --json
arc-paper get-citers <paper-id> --limit 1000 --sort mostcited --json
arc-paper get-citer-count <paper-id> --json
arc-paper get-toc <paper-id> --json
arc-paper get-section <paper-id> --section <section> --json
arc-paper search-full-text <paper-id> --query "<phrase>" --context 1 --json
arc-paper get-equation-context <paper-id> --query "<symbol-or-label>" --json
arc-domain get-summary --seed-paper <paper-id> --json
arc-domain get-graph --seed-paper <paper-id> --json
```

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

## Phase 3: Build The Calculation Plan

Step 1: Make the first calculation step the first nontrivial derivation after
the accepted foundation setup.

Step 2: Break difficult derivations into small steps that a not-strong agent
can complete. As soft guidance, a typical research project should have at least 20 steps.
Use fewer only when the calculation is genuinely smaller. If a
step needs multiple identities, limits, field redefinitions, or approximations,
split it into substeps.

Step 3: For every step, specify:

```text
step_id
kind: foundation_check | new_calculation
goal
quantity_to_calculate
quantity_dependencies
allowed_inputs
substeps
expected_output
verification
```

Step 4: At the end of every step, make the quantity contract explicit:
calculate which quantity, in terms of which quantity, and what is not allowed
as an input. Do not disclose the exact expected expression or expected final formula.
Instead, say to derive the target quantity in terms of named dependencies.
If this cannot be stated clearly, split the step again.

For equations quoted from a reference or collaborator note that need checking,
do not disclose the target reference equation in `prompt`, `allowed_inputs`, or
`expected_output`. Make the step a blind reference check: proposers derive the
quantity from named dependencies, and the execute workflow supplies the target
only as a reviewer-only reference claim.

Step 5: Write `plan.json` and write the initial human-readable report directly
to both `<project-dir>/calculate/<run-id>/initial-plan.md` and
`<project-dir>/initial-plan.md`. The JSON is the source of truth for
later workflow phases.

After writing the project-level Markdown report, call
MCP `md2pdf(input="<project-dir>/initial-plan.md")`. It starts a background
PDF job; record the returned job id if present and do not wait before
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
steps are small enough for weaker agents
the first calculation step is clear
difficult steps have enough substeps
```

Step 3: If the review finds gaps, revise `plan.json` and
`initial-plan.md`, then review the plan again. Proceed only when the
review is recorded in both artifacts.
