# Research Execute Workflow

Use this workflow after `initial-research-foundation.md`. It checks non-axiom
foundation equations and then performs the new calculation steps through
`arc-llm` consensus execution.

Write artifacts under:

```text
<project-dir>/calculate/<run-id>/execute/consensus.config.json
<project-dir>/calculate/<run-id>/execute/<consensus-run-id>/
<project-dir>/calculate/<run-id>/calculation-report.md
<project-dir>/calculation-report.md
```

Execution reports must use `schema_version: "arc.research_execute.v1"`.

## Phase 1: Build The Consensus Config

Step 1: Read `plan.json` and `foundation/latest.json`.

Step 2: Create a consensus config:

```json
{
  "schema_version": "arc.llm.proposers_reviewer_consensus.config.v1",
  "run_id": "<consensus-run-id>",
  "run_dir": "<project-dir>/calculate/<run-id>/execute",
  "proposer_count": 3,
  "max_recalculations": 2,
  "defaults": {
    "integrity_reference_path": "skills/arc/references/rules/integrity.md"
  },
  "artifact_options": {
    "save_prompts": true
  },
  "steps": []
}
```

Step 3: Keep `proposer_count` and `max_recalculations` configurable. Defaults
are 3 proposers and 2 recalculations, giving 3 total attempts:
1 initial attempt + 2 recalculations.

## Phase 2: Add Foundation Checks

Step 1: Skip equations with `axiom_status: "axiom"`.

Foundation checks use the same 3-proposer reviewer consensus as new
calculation steps, with the same acceptance standard and no single-proposer acceptance.

Step 2: For every other foundation equation, add one step before new
calculations:

```json
{
  "step_id": "check_eq_001",
  "kind": "foundation_check",
  "prompt": "Check equation eq_001. Do not assume it is true. Use only the filtered foundation context, accepted axioms, checked equations, and explicit algebra."
}
```

Step 3: Put `foundation_file` and `target_equation_id` in `allowed_context`.
The consensus runner filters the context so proposers see only the target
equation plus axiom and checked foundation items. Unchecked equations must be
omitted.

Step 4: To inspect the same filtered context manually, run
`python3 scripts/filter-foundation-context.py foundation/latest.json --target-equation-id eq_001`
from this workflow directory.

## Phase 3: Add New Calculation Steps

Step 1: Append every `plan.json.steps[]` entry with
`kind: "new_calculation"` after all foundation checks.

Step 2: Each prompt must include the exact allowed inputs, accepted prior step
outputs, expected output, and verification target. Do not allow later steps to
use unaccepted or blocked outputs.

Step 3: At the end of each step prompt, state the required calculation contract:
calculate which quantity, in terms of which quantity, and what equality or
limit the reviewer must check.

## Phase 4: Enforce Calculation Rules

Step 1: Enclose `integrity.md` for both proposer and reviewer prompts.

Step 2: Proposer prompts must require a very clear step-by-step derivation and
must say never skip a step.

Step 2a: Proposer outputs that may appear in human reports must write
mathematical expressions in Markdown/LaTeX form. Use `$...$` for inline math
and `$$...$$` for display equations in prose fields. In JSON expression fields,
use LaTeX strings such as `\\rho_2`, `\\eta'`, and `T_{ab}`, not report-facing
ASCII placeholders such as `rho_2`, `eta_prime`, or `T_ab`. Proposer outputs
must include `validity_scope` for assumptions, conventions, limits, and
unresolved dependencies. Do not request or render date-like `reliable_until`
fields for derivation validity.

Step 3: Proposers may use ARC paper MCP tools to read the main reference and
cited source sections named by `plan.json` or
`foundation/latest.json`. Internet search is allowed only for source discovery
or uncached paper access. Proposers must cite any paper tool or internet source
they use. Other paper tools are not allowed. Proposers must not use
validation-only final formulas as derivation inputs.
They may also use SymPy, local algebra, and Wolfram only for algebraic checks.

Step 4: Proposers must strictly derive from the foundation context and accepted
prior outputs. External sources may inspire methods, but proposers do not directly use any result
from papers or the internet unless that result is in
the foundation file or has already been accepted. If they need an external
identity or intermediate result, they must derive it inside the current
calculation. Warn that external sources may use different conventions; map any
notation back to the foundation conventions before using it.

Step 5: Reviewers may use SymPy. For analytic checks, use `expand` first, then
`simplify`, then substitutions from checked equations in the foundation file.
Do not modify original equations.

Step 6: Before `all_agree`, at least two of `A-B=0`, `B-C=0`, and `A-C=0`
must be true. Never accept agreement by visual inspection, string equality,
spacing, or formatting. If SymPy is unavailable, either write explicit
algebraic differences for `A-B`, `B-C`, and `A-C`, or use the numerical
fallback.

Step 7: If a reviewer still suggests `all_agree` but its report is below this
standard, the main agent must not stop immediately. First run an independent
SymPy check of `A-B`, `B-C`, and `A-C` from proposer final results. If SymPy proves agreement,
accept and record the fallback check. If the main agent cannot
prove agreement, pause for human review.

Step 8: If analytic checking is not possible, use at least 10 randomly selected
data points. The minimum numerical fallback is 10 randomly selected data points.
Record `check_method: "numerical"`, the relative error, the sample count, and
the check history.

Step 9: For every `all_agree` review, the reviewer must also select one
proposer output as the full-detail source for the human report. Record this in
`review_payload.consensus.best_written_proposer_id` with
`best_written_selection_reason`. Choose from `agreed_proposer_ids` or accepted
locked outputs from earlier attempts using clearest logic, most complete
details, and best readability. This choice does not affect correctness
acceptance; it only chooses which agreeing derivation is rendered in full. If
the review is not `all_agree`, set
`best_written_proposer_id` to `null` and explain why.

Step 10: For an accepted foundation check, write a new foundation version that
marks the target equation checked. Keep the original equation unchanged and add
the reviewer check history, method, relative error when numerical, and accepted
consensus artifact path. Do not rewrite `initial-research-foundation.md`;
human-facing foundation changes belong in the final report appendix.

Step 11: When a new calculation result is accepted and will be useful as a later
input, write a new foundation version with a concise derived quantity record.
Record the statement, explanation, source step id, dependency ids, check status,
and consensus artifact. Keep paper-sourced equations and derived quantities
visibly separate in `latest.json`.

## Phase 5: Run Consensus And Refine Blocks

Step 1: Run:

```bash
arc-llm proposers-reviewer-consensus \
  --config <project-dir>/calculate/<run-id>/execute/consensus.config.json \
  --json
```

Step 2: Inspect the returned JSON. If a step returns `blocked_for_user` after
3 total attempts, enter `blocked_refinement`.

Step 3: In `blocked_refinement`, review plan.json, reviewer reports, and, if
needed, proposer calculations. Treat the block as evidence that the step is too
difficult unless the step is already atomic.

Step 4: If the blocked step can be split, revise the plan into smaller steps.
Each replacement step must have one clear quantity, inputs, output, and check.
The first replacement step should stop at the last calculation all proposers can
agree on.

If the full expression is still hard to split into complete sub-results, first
choose controlled limits or projections that are simpler but relevant, such as
one branch, one contour choice, one contraction, leading power only,
equal-mass/equal-scale limit, or coefficient-stripped form. Ask proposers to
agree on those limited results before returning to the full expression. If
proposers already disagree in a controlled limit, continue refining around that
limit instead of asking for the full expression.

Then rerun the 3-proposer reviewer consensus on the refined step. Append each blocked_refinement event
to the plan revision history inside
`# Appendix 2: Calculation Status`; do not create a separate plan-revision
report.

Step 5: If the block is caused by inadequate foundation or a wrong plan, do not
silently continue. Classify it as `foundation_inadequate` when definitions or
premises are missing but expandable without inconsistency, `foundation_conflict`
when existing foundation items appear inconsistent, or `plan_wrong` when the
step is not viable but a revised plan remains physically interesting.

For `foundation_inadequate` or `plan_wrong`, request two independent proposers
to propose the expansion or revision. Continue only if at least two proposers
agree, the reviewer agrees, and the main agent agrees after inspection. For
`foundation_conflict`, stop and ask the human expert unless the same agreement
process resolves the conflict. In interactive mode, ask human approval before
applying the revision; in auto mode, apply it and continue. Report the revision
in `calculation-report.md` with a `**Caution**` paragraph explaining what
changed, why, who agreed, approval mode, and which later results depend on it.

Step 6: If the blocked step is already atomic and cannot be split further, stop
as blocked. Do not choose a proposer yourself.

Step 7: Write `calculation-report.md` even when blocked, directly to both
`<project-dir>/calculate/<run-id>/calculation-report.md` and
`<project-dir>/calculation-report.md`. Include accepted outputs, blocked step,
disagreement map, reviewer-report summary, proposer positions, artifact paths,
the exact expert question, `# Appendix 1: Latest Research Foundation`, and
`# Appendix 2: Calculation Status`, and `# Appendix 3: Full Calculation
Details`. The first appendix must render the latest foundation updates from
`foundation/latest.json`, including checked equations, derived quantities,
version notes, and consensus artifacts. The second appendix must summarize
original steps, each blocked_refinement, plan revision history, what changed,
the replacement config, and whether that refined step was accepted or blocked.
The third appendix must render one full calculation copy per accepted planned
calculation step, using the reviewer-selected `best_written_proposer_id`.
Ask which proposer or result is correct, or what instruction should continue
the calculation.

Appendix 3 must use this structure:

```text
# Appendix 3: Full Calculation Details

## Planned Step 1

### Plan

<render the latest active plan step or refined replacement step: goal, quantity
contract, allowed inputs, dependencies, substeps, expected output, verification
target, and revision note when it replaced an original step>

### Calculation Details

<render the selected proposer's full derivation/calculation, assumptions,
validity_scope, source/tool citations, and final_result>

## Planned Step 2

### Plan

...

### Calculation Details

...
```

For each accepted planned step, render the latest active plan, not the stale
original plan. If blocked refinement replaced a step, use the replacement step
that was actually accepted, include the original `step_id`, the replacement
`step_id`, and the refinement reason. If `plan.json` was revised, read the
latest `plan.json`; if the refinement is represented only in a replacement
consensus config, render that replacement config's step prompt and append the
plan revision history entry from Appendix 2. Then read
`reviewer_consensus.best_written_proposer_id` from the accepted consensus step
and find that proposer's JSON in the accepted attempt's `proposer_output_paths`.
If the selected proposer output was locked from an earlier attempt, locate the
most recent matching proposer path in earlier `attempts[]`. Render only that
selected copy in full; do not duplicate the other agreeing derivations. Render
math from the selected proposer output as Markdown/LaTeX display math where
possible, not as raw ASCII prose. Do not render date-like `reliable_until`
fields as scientific validity. Prefer `validity_scope`; if older artifacts only
contain `reliable_until`, translate it to a short validity-scope note or omit
it. Still include artifact paths for the selected proposer output, the reviewer
report, and the consensus run state. If there is no accepted consensus or no
selected best-written proposer, say why no single full-detail copy is rendered
and point to the available artifact paths.

Step 7: If the human expert decides that one proposer or result is correct,
continue from that premise and mark it as human-resolved in the next report.
If all steps are accepted without human intervention, write the same report with
accepted outputs, reviewer consensus summaries, unresolved risks, and artifact
paths. Do not claim a result that is not present in accepted consensus output or
human-resolved input.

After writing the project-level Markdown report, call
MCP `md2pdf(input="<project-dir>/calculation-report.md")`. It starts a
background PDF job; record the returned job id if present and do not wait
before continuing.
