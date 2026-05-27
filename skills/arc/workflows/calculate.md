# Calculate Workflow
Use this workflow after `initial-foundation.md`. It checks non-axiom
foundation equations, runs blind reference checks, then performs new calculation
steps through `arc-llm` consensus execution.
Write artifacts under:
```text
<project-dir>/calculate/<run-id>/execute/consensus.config.json
<project-dir>/calculate/<run-id>/execute/<consensus-run-id>/
<project-dir>/calculate/<run-id>/calculation-report.md
<project-dir>/calculation-report.md
```
Execution reports must use `schema_version: "arc.calculate.v1"`.

## Phase 1: Build Consensus Config
Read `plan.json` and `foundation/latest.json`. Create:
```json
{
  "schema_version": "arc.llm.proposers_reviewer_consensus.config.v1",
  "run_id": "<consensus-run-id>",
  "run_dir": "<project-dir>/calculate/<run-id>/execute",
  "proposer_count": 3,
  "max_recalculations": 2,
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
For paper or note equations that need checking, prefer a blind reference check
over `foundation_check`. Do not put the target equation in
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
verifies the reference; `A=B!=C` accepts the blind derivation and marks
`reference_disagrees`; `A!=B` means recalculate or split the step.

## Phase 3: Add New Calculation Steps

Append every `plan.json.steps[]` entry with `kind: "new_calculation"` after all
checks. Each prompt must include exact allowed inputs, accepted prior step
outputs, expected output, and verification target. Later steps must not use
unaccepted or blocked outputs.

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
path. Do not rewrite `initial-foundation.md`; human-facing foundation
changes belong in the final report appendix. When a new calculation result is
accepted and useful later, write a new foundation version with a concise derived
quantity record, keeping paper-sourced equations and derived quantities visibly
separate in `latest.json`.

## Phase 5: Run Consensus And Refine Blocks

Run:

```bash
arc-llm proposers-reviewer-consensus \
  --config <project-dir>/calculate/<run-id>/execute/consensus.config.json \
  --json
```

Inspect the returned JSON. If a step returns `blocked_for_user` after 3 total
attempts, enter `blocked_refinement`: review plan.json, reviewer reports, and,
if needed, proposer calculations. Treat the block as evidence the step is too
difficult unless already atomic.

If the blocked step can be split, revise the plan into smaller steps. Each
replacement step must have one clear quantity, inputs, output, and check. The
first replacement step should stop at the last calculation all proposers can
agree on. If full expression splitting is hard, first use controlled limits or projections such as one branch, one contour choice, one contraction, leading
power only, equal-mass/equal-scale limit, or coefficient-stripped form before returning to the full expression. Append each blocked_refinement event to the
plan revision history inside `# Appendix 2: Calculation Status`.

If the block is caused by missing or wrong premises, classify it as
`foundation_inadequate`, `foundation_conflict`, or `plan_wrong`. For
`foundation_inadequate` or `plan_wrong`, request two independent proposers to
propose the expansion or revision; continue only if two proposers agree, the reviewer agrees, and the main agent agrees after inspection. For
`foundation_conflict`, stop for the human expert unless that same agreement
process resolves the conflict. In interactive mode ask approval before applying
the revision; in auto mode apply it and continue. Report any revision in
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

After writing the project-level Markdown report, call
MCP `md2pdf(input="<project-dir>/calculation-report.md")`. It starts a
background PDF job; record the returned job id if present and do not wait before
continuing.
