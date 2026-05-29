# Calculate Workflow
Use this workflow after `initial-foundation.md`. It checks non-axiom
foundation equations, runs blind reference checks, then performs new calculation
steps through `arc-llm` consensus execution.
Write artifacts under:
```text
<project-dir>/calculate/<run-id>/execute/consensus.config.json
<project-dir>/calculate/<run-id>/execute/<consensus-run-id>/
<project-dir>/calculate/<run-id>/plan-expansion-requests/
<project-dir>/calculate/<run-id>/calculation-report.md
<project-dir>/calculation-report.md
```
Execution reports must use `schema_version: "arc.calculate.v1"`.

## Phase 1: Build Consensus Config
Read `plan.json` and `foundation/latest.json`. Execute only the current
`detailed_steps` (or legacy `steps` when older plans do not yet use rolling
plan fields). Do not execute `macro_plan` entries until `plan.md` expands them
into detailed steps. Create:
```json
{
  "schema_version": "arc.llm.proposers_reviewer_consensus.config.v1",
  "run_id": "<consensus-run-id>",
  "run_dir": "<project-dir>/calculate/<run-id>/execute",
  "proposer_count": 3,
  "max_recalculations": 2,
  "human_gate": {
    "enabled": false
  },
  "defaults": {
    "integrity_reference_path": "skills/arc/rules/integrity.md"
  },
  "artifact_options": {"save_prompts": true},
  "steps": []
}
```
Keep `proposer_count` and `max_recalculations` configurable. Defaults are 3
proposers and 2 recalculations: 3 total attempts, meaning 1 initial attempt + 2 recalculations.
Do not increase attempts unless the user asks.

For runs that should pause on failed or non-agreeing steps, set:

```json
"human_gate": {
  "enabled": true,
  "pause_on_statuses": [
    "reference_disagrees",
    "two_agree",
    "all_disagree",
    "unresolved",
    "failed"
  ]
}
```

This gate stops at the first failed or non-agreeing step. It returns
`blocked_for_user` when an expert decision is needed, or
`blocked_for_revision` when all proposer assessments, the reviewer, and the
main agent can agree on a foundation or plan revision without asking the user.

## Phase 2: Add Foundation Checks
Skip equations with `axiom_status: "axiom"`. Foundation checks use the same 3-proposer reviewer consensus as new calculation steps, with the same acceptance standard and no single-proposer acceptance.
For every non-axiom foundation equation, add one step before new calculations:
```json
{
  "step_id": "check_eq_001",
  "kind": "foundation_check",
  "prompt": "Check equation eq_001. Do not assume it is true. Use only the filtered foundation context, accepted axioms, checked equations, and explicit algebra.",
  "allowed_context": {
    "foundation_file": "<project-dir>/calculate/<run-id>/foundation/latest.json",
    "target_equation_id": "eq_001"
  }
}
```
The runner filters context so proposers see only the target equation plus axiom and checked foundation items. Unchecked equations are omitted. To inspect the
same filtered context manually, run `python3 scripts/filter-foundation-context.py
foundation/latest.json --target-equation-id eq_001` from this workflow
directory.

## Phase 2a: Add Blind Reference Checks
For source, reference, or collaborator equations that need checking, prefer a
blind reference check over `foundation_check`. Do not put the target equation in
`foundation/latest.json`, `prompt`, or `allowed_context`.
Add a `new_calculation` step with two proposers and reviewer-only C:
```json
{
  "step_id": "blind_ref_eq_001",
  "kind": "new_calculation",
  "prompt": "Derive the named target quantity from supplied definitions and checked foundation items. Do not use papers, internet search, or target formulas.",
  "allowed_context": {
    "quantity_to_calculate": "target quantity name",
    "quantity_dependencies": ["dependency names"],
    "allowed_inputs": ["checked foundation ids only"]
  },
  "proposer_runtime": {"allow_internet": false, "allow_mcp": false},
  "reviewer_reference_claim": {
    "id": "ref_eq_001",
    "latex": "...",
    "source": {"paper_id": "arXiv:...", "section": "..."}
  }
}
```

For a blind reference check, proposers default to no paper tools and no internet
search unless the user explicitly requests source access. The reviewer compares
A and B from blind proposers with C from `reviewer_reference_claim`: `A=B=C`
verifies the reference. In standard calculation mode, `A=B!=C` accepts the
blind derivation and marks `reference_disagrees`. With
`human_gate.enabled=true`, `A=B!=C` stops immediately for expert resolution
unless a shared plan/foundation revision fully explains the mismatch.
`A!=B` means recalculate, split the step, or stop for human review depending on
the active gate.

## Phase 3: Add New Calculation Steps

Append every executable detailed step with `kind: "new_calculation"` after all
checks. Prefer `plan.json.detailed_steps[]`; fall back to `plan.json.steps[]`
only for legacy plans. Each prompt must include exact allowed inputs, accepted
prior step outputs, expected output, verification target, source context
policy, and checkpoint list. Later steps must not use unaccepted or blocked
outputs.

If a detailed step has multiple checkpoints, build one proposer prompt that
names the checkpoint contracts without revealing the target formulas. Put all
checkpoint target formulas in reviewer-only `reviewer_reference_claim` data,
with stable checkpoint ids so the reviewer can report which checkpoints are
verified, disagree, or unresolved. Use the plan's redacted context slices:
proposers may see source context up to each checkpoint, but not the checkpoint
equation itself or later source text.

For each post-check new calculation that is not checking a reference formula,
turn source access on by default unless the user requested otherwise:

```json
"proposer_runtime": {"allow_internet": true, "allow_mcp": true}
```

End each step prompt with the quantity contract: calculate which quantity, in
terms of which quantity, and what equality or limit the reviewer must check.

## Phase 4: Enforce Calculation Rules

Enclose `integrity.md` for proposer and reviewer prompts. Proposers must give a
clear step-by-step derivation and never skip a step. Report-facing math must be
valid Markdown/LaTeX; use `validity_scope` for assumptions, conventions, limits,
and unresolved dependencies.

Proposers may use ARC paper MCP tools only when `proposer_runtime` allows MCP.
For post-check new calculation steps this is enabled by default, so proposers
may read the main reference and cited sections named by `plan.json` or
`foundation/latest.json`. Internet search is allowed only when
`proposer_runtime` allows it and only for source discovery or uncached paper
access. Proposers must cite any paper tool or internet source they use. Other
paper tools are not allowed. Proposers must not use validation-only final formulas as derivation inputs. Wolfram may be used only for algebraic
verification.

Proposers must strictly derive from the foundation context and accepted prior
outputs. External sources may inspire methods, but do not directly use any result from papers or the internet unless that result is in the foundation file
or already accepted. If an external identity or intermediate result is needed,
derive it inside the current calculation. External sources may use different conventions; map notation back to foundation conventions before using it.

Every proposer output must include `plan_foundation_assessment`: whether the
foundation or plan needs revision, the issue type, proposed revision if any,
rationale, and whether the step can continue without that revision.

Every reviewer output must include `workflow_action`: `continue`,
`pause_for_human`, `revise_foundation`, `revise_plan`, `split_step`, or
`retry`; whether a human is required; the issue type; any proposed revision;
and the expert question when pausing.

Reviewers may use SymPy. For analytic checks, use `expand`, then `simplify`,
then substitutions from checked equations in the foundation file. Do not modify
original equations. Before `all_agree`, at least two of `A-B=0`, `B-C=0`, and
`A-C=0` must be true. Never accept by visual inspection, string equality,
spacing, or formatting. If SymPy is unavailable, write explicit algebraic
differences or use the numerical fallback.

If a reviewer still suggests `all_agree` but its report is below this standard,
the main agent must run an independent SymPy check of `A-B`, `B-C`, and `A-C`.
If SymPy proves agreement, accept and record the fallback check. If the main
agent cannot prove agreement, pause for human review. If analytic checking is
not possible, use at least 10 randomly selected data points and record
`check_method: "numerical"`, relative error, sample count, and check history.

For every `all_agree` review, record
`review_payload.consensus.best_written_proposer_id` and
`best_written_selection_reason`. Pick from agreeing or locked proposer outputs
using clearest logic and most complete details. This chooses report prose only;
it does not affect correctness.

For an accepted foundation check, write a new foundation version marking the
target equation checked. Keep the original equation unchanged and add reviewer
check history, method, relative error when numerical, and consensus artifact
path. Do not rewrite `initial-foundation.md`; instead render the updated
`latest.json` to both
`<project-dir>/calculate/<run-id>/foundation/latest-foundation.md` and
`<project-dir>/latest-foundation.md`, then call
`md2pdf(input="<project-dir>/latest-foundation.md")` in the background. When a
new calculation result is accepted and useful later, write a new foundation
version with a concise derived quantity record, keeping paper-sourced equations
and derived quantities visibly separate in `latest.json`, and refresh the
latest-foundation Markdown/PDF artifacts the same way.

When all current detailed steps are accepted and `plan.json.macro_plan` still
has unresolved blocks, create a plan-expansion request instead of inventing new
steps inside this workflow. The request must include:

```json
{
  "schema_version": "arc.plan_expansion_request.v1",
  "request_type": "expand_macro_block",
  "target_macro_block_id": "<macro_block_id>",
  "current_plan_path": "<project-dir>/calculate/<run-id>/plan.json",
  "foundation_path": "<project-dir>/calculate/<run-id>/foundation/latest.json",
  "accepted_outputs": ["accepted step ids and artifact paths"],
  "reviewer_reports": ["review artifact paths"],
  "observed_agent_ability": {
    "accepted_step_count": 0,
    "retry_counts": {},
    "blocked_or_failed_steps": [],
    "useful_context_packets": [],
    "failure_modes": []
  },
  "request": "Expand this macro block into detailed steps using current evidence."
}
```

Write it under
`<project-dir>/calculate/<run-id>/plan-expansion-requests/`, then run
`plan.md` again with that request as the task-to-be-planned artifact. The
planning workflow owns new step boundaries, checkpoint grouping, context
packets, and revised `latest-plan.md`. After `plan.md` updates the plan,
create a new consensus config for the next detailed batch.

## Phase 5: Run Consensus And Refine Blocks

Run:

```bash
arc-llm proposers-reviewer-consensus \
  --config <project-dir>/calculate/<run-id>/execute/consensus.config.json \
  --json
```

Do not mark the workflow blocked merely because the consensus config has many
steps, the run is expected to require many LLM calls, or execution is likely to
be slow, expensive, or serial. Large workload is an execution/runtime property,
not a scientific or workflow block. In `auto` mode, start or continue the
configured consensus run and use the available watcher, background-job
procedure, or package status command rather than stopping for size alone. Only
mark blocked for an actual consensus status, failed execution, missing required
input, instruction conflict, unavailable runtime, or a human decision required
by the workflow.

Inspect the returned JSON. If a step returns `blocked_for_user`, ask the human
expert the reported `expert_question` before continuing. If a step returns
`blocked_for_revision`, inspect `workflow_action.proposed_revision`; apply it
only when the proposer assessments, reviewer, and main agent agree. Otherwise
ask the human expert.

If a standard calculation step returns `blocked_for_user` after the configured
attempt limit, enter `blocked_refinement`: review plan.json, reviewer reports,
and, if needed, proposer calculations. Treat the block as evidence the step is
too difficult unless already atomic. Do not write replacement plan logic inside
`calculate.md`.

For any split, refinement, or broader replanning need, write a
`plan-expansion-request` artifact and call `plan.md` recursively. Use
`request_type: "refine_blocked_step"` for blocked steps and include the blocked
step, last agreed checkpoint, proposer reports, reviewer analysis, retry count,
and suspected failure mode. Ask `plan.md` to produce a better detailed plan for
that blocked region using current evidence. The replacement plan may split the
step, add controlled limits or projections such as one branch, one contour
choice, one contraction, leading power only, equal-mass/equal-scale limit, or
coefficient-stripped form, or keep the step atomic and require human input.

After `plan.md` updates `plan.json`, confirm that `latest-plan.md` was
refreshed by the planning workflow. Do not rewrite `initial-plan.md` after the
first snapshot. Append each blocked_refinement event to the plan revision
history inside `# Appendix 2: Calculation Status`.

If the block is caused by missing or wrong premises, classify it as
`foundation_inadequate`, `foundation_conflict`, or `plan_wrong`. For
`foundation_inadequate` or `plan_wrong`, request two independent proposers to
propose the expansion or revision; continue only if two proposers agree, the
reviewer agrees, and the main agent agrees after inspection. For
`foundation_conflict`, stop for the human expert unless that same agreement
process resolves the conflict. In interactive mode ask approval before applying
the revision. In auto mode apply it only for `blocked_for_revision`, where the
returned `workflow_action.requires_human` is false and the main agent agrees
after inspection; otherwise ask the human expert. Report any revision in
`calculation-report.md` with a `**Caution**` paragraph explaining what changed,
why, who agreed, approval mode, and dependent later results. If the step is
already atomic and cannot be split, stop as blocked; do not choose a proposer
yourself.

## Phase 6: Write The Report

Write `calculation-report.md` even when blocked, directly to both
`<project-dir>/calculate/<run-id>/calculation-report.md` and
`<project-dir>/calculation-report.md`. Include accepted outputs, blocked step,
disagreement map, reviewer-report summary, proposer positions, artifact paths,
the exact expert question, `# Appendix 1: Latest Foundation`,
`# Appendix 2: Calculation Status`, and `# Appendix 3: Full Calculation
Details`. Ask which proposer or result is correct, or what instruction should
continue the calculation.

Appendix 1 renders latest foundation updates from `foundation/latest.json`.
Appendix 2 summarizes original steps, each blocked_refinement, plan revision
history, replacement config, and refined-step status. Appendix 3 renders one
full calculation copy per accepted planned calculation step using
`best_written_proposer_id`, with plan.json step, selected derivation,
assumptions, source/tool citations, validity_scope, final_result, and artifact
paths. If no selected proposer exists, say why and point to artifacts. If a
human expert resolves a result, continue from that premise and mark it
human-resolved. Do not claim a result absent from accepted consensus output or
human-resolved input.

If `plan.json` includes source items or reference-only targets, include a
status map for those items in the report. Use `foundation`, `verified`,
`human_resolved`, `reference_disagrees`, `unresolved`, or `context_only`, and
include source path, page, section, heading, or label when available.

When a human-resolved result is used to continue, write the resolution into the
run artifacts before launching the next consensus config:

```json
{
  "status": "human_resolved",
  "resolution": {
    "resolved_by": "user",
    "resolved_at": "<ISO-8601 timestamp>",
    "type": "corrected_formula | accepted_formula | premise | instruction",
    "corrected_latex": "<LaTeX formula when applicable>",
    "accepted_result": "<plain result when not LaTeX>",
    "rationale": "<why this resolves the block>",
    "use_as_later_premise": true
  }
}
```

Then create a continuation config starting from the next unresolved step. Add
human-resolved premises to the continuation config as accepted prior results,
with `status: "human_resolved"` and `source: "human_expert_resolution"`.
Record the continuation config path and run id in `calculation-report.md`.

After writing the project-level Markdown report, call
MCP `md2pdf(input="<project-dir>/calculation-report.md")`. It starts a
background PDF job; record the returned job id if present and do not wait before
continuing.
