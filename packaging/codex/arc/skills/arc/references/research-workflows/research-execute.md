# Research Execute Workflow

Use this workflow after `research-foundation.md`. It checks non-axiom
foundation equations and then performs the new calculation steps through
`arc-llm` consensus execution.

Write artifacts under:

```text
<project-dir>/calculate/<run-id>/execute/consensus.config.json
<project-dir>/calculate/<run-id>/execute/<consensus-run-id>/
<project-dir>/calculate/<run-id>/report.md
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
  "max_recalculations": 3,
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
are 3 proposers and 3 recalculations.

## Phase 2: Add Foundation Checks

Step 1: Skip equations with `axiom_status: "axiom"`.

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

Step 3: Proposers must not use the internet or paper tools. They may use only
the provided foundation context, accepted prior outputs, SymPy or local algebra,
and a Wolfram MCP only when the host can expose that algebra tool without
internet or paper access. If tool allowlisting is unavailable, keep MCP disabled.

Step 4: Reviewers may use SymPy. For analytic checks, use `expand` first, then
`simplify`, then substitutions from checked equations in the foundation file.
Do not modify original equations.

Step 5: Before `all_agree`, at least two of `A-B=0`, `B-C=0`, and `A-C=0`
must be true. Never accept agreement by visual inspection, string equality,
spacing, or formatting. If SymPy is unavailable, either write explicit
algebraic differences for `A-B`, `B-C`, and `A-C`, or use the numerical
fallback.

Step 6: If analytic checking is not possible, use at least 10 randomly selected
data points. The minimum numerical fallback is 10 randomly selected data points.
Record `check_method: "numerical"`, the relative error, the sample count, and
the check history.

Step 7: For an accepted foundation check, write a new foundation version that
marks the target equation checked. Keep the original equation unchanged and add
the reviewer check history, method, relative error when numerical, and accepted
consensus artifact path.

## Phase 5: Run Consensus

Step 1: Run:

```bash
arc-llm proposers-reviewer-consensus \
  --config <project-dir>/calculate/<run-id>/execute/consensus.config.json \
  --json
```

Step 2: Inspect the returned JSON. If any step returns `blocked_for_user`
because the recalculation limit was reached, stop immediately and ask the user
for instructions. Do not continue in auto mode.

Step 3: If all steps are accepted, write `report.md` with accepted outputs,
reviewer consensus summaries, unresolved risks, and artifact paths. Do not
claim a result that is not present in accepted consensus output.

After `report.md` is generated, copy it to `<project-dir>/report.md` so human
readers can inspect the main project reports together.

After copying the Markdown report, call
MCP `md2pdf(input="<project-dir>/report.md")`. It starts a background PDF job;
record the returned job id if present and do not wait before continuing.
