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
  "prompt": "Check equation eq_001 from the foundation file. Do not assume it is true. Use the cited sources and accepted first principles only."
}
```

Step 3: The prompt must include the equation id, latex, confidence labels,
source commands, and allowed inputs from the plan.

## Phase 3: Add New Calculation Steps

Step 1: Append every `plan.json.steps[]` entry with
`kind: "new_calculation"` after all foundation checks.

Step 2: Each prompt must include the exact allowed inputs, accepted prior step
outputs, expected output, and verification target. Do not allow later steps to
use unaccepted or blocked outputs.

## Phase 4: Run Consensus

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
