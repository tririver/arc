# Calculate Workflow

Use this workflow after `plan.md` writes `<project-dir>/work-note.md`.
Execute only steps marked ready in `Detailed Steps Ready To Calculate`.
Do not write a separate calculation report; the updated work note is the
human-readable result.

`calculate.md owns consensus execution` and the `current-step result-status`.
It does not change ready-step boundaries, does not change rough steps, and
does not change future plan structure. Calculate does not own note parsing. When
a different workflow owns the needed change, refer to the owning workflow.

## Phase 1: Prepare Runtime

Runtime artifacts:

```text
<project-dir>/calculate/<run-id>/execute/calculate.config.json
<project-dir>/calculate/<run-id>/execute/<calculate-run-id>/
<project-dir>/calculate/<run-id>/execute/<calculate-run-id>/attempt_batches/
```

Copy `workflows/json/calculate.config.template.json` to:

```text
<project-dir>/calculate/<run-id>/execute/calculate.config.json
```

Replace `<calculate-run-id>`, `<project-dir>`, `<run-id>`, and
`<skill-workflow-json-dir>`. Use `skill_dir` from context as `<skill-dir>` in
commands below. Keep `"proposer_count": 2`,
`"max_recalculations": 2`, and `artifact_options.save_prompts` enabled unless
the user asks otherwise.

The runner reads worker prompt/schema templates from:

- `workflows/json/calculate-proposer.template.json`
- `workflows/json/calculate-reviewer.template.json`
- `workflows/json/calculate-reviewer-output.schema.json`

`"max_recalculations": 2` means 3 total attempts: 1 initial attempt + 2 recalculations.
Do not increase attempts unless the user asks.
For retryable proposer disagreement statuses, use the recalculation budget
before pausing for human input.
Also retry `reference_disagrees` while budget remains when reviewer feedback can
tell proposers what to recheck without revealing reviewer-only target formulas.

Remove foundation_check mechanics. Starting points are checked by ordinary ready
steps when they are marked not accepted in the work note.

## Phase 2: Build Step Packets

For each current ready step, add one config step with:

- current step prompt and quantity contract
- relevant work-note sections: notation, axioms, accepted results, and the
  current ready step
- clean proposer-facing source context in `allowed_context`

Do not expose reviewer-only targets, target equations, or later note text to
proposers.

For a blind reference check, include `reviewer_reference_claim` only in the
step object and disable source tools:

```json
"proposer_runtime": {"allow_internet": false, "allow_mcp": false}
```

If blind proposers agree with each other but not with the reviewer reference,
record `reference_disagrees`; use remaining recalculation budget with
non-revealing reviewer feedback before pausing for a human decision.

For a post-check new calculation, enable source access by default unless the
user requested otherwise:

```json
"proposer_runtime": {"allow_internet": true, "allow_mcp": true}
```

External sources may guide methods, but any used identity or intermediate
result must be derived or already accepted in the work note. Map all notation
back to work-note conventions.

## Phase 3: Run Consensus

Run:

```bash
python3 <skill-dir>/workflows/scripts/calculate_runner.py \
  --config <project-dir>/calculate/<run-id>/execute/calculate.config.json \
  --json
```

Inspect the returned JSON and saved artifacts. Large or slow runs are runtime
facts, not workflow blocks. Use package status or watcher commands instead of
manual polling when available.

## Phase 4: Review Acceptance

Acceptance depends on reviewer judgment. SymPy, Wolfram, explicit algebra, and
numerical checks are optional tools, not mandatory gates. Accept only if the
target quantity agrees in the declared regime and approximation order.

The reviewer must explain the comparison, conventions, rewrites, and identities
used to relate expressions. Special limits are sanity checks, not proof of full agreement unless the target itself is a limit, asymptotic result, or
leading-order claim.

The main agent audits the reviewer report before updating the work note. Reject
weak evidence such as formatting agreement, visual similarity, or agreement in
an undeclared special limit. Depending on the failure, retry, split, pause for
the expert question, or write a planning request.

If an accepted derivation contradicts a source or reviewer-only reference claim,
classify the discrepancy before updating the work note:

- `confirmed_source_error`: blind proposers agree, the reviewer agrees, the
  main agent agrees with the reviewer, the derivation uses only accepted
  premises, the mismatch is not convention-dependent, and the reviewer says no
  human convention choice is needed.
- `likely_source_error`: the derivation probably identifies a source problem,
  but one of the confirmation requirements is missing or weak.
- `ambiguous_convention`: the mismatch may be due to convention, normalization,
  notation, source mapping, or interpretation.

Only `confirmed_source_error` may continue without human interaction in
`interactive` mode. For `likely_source_error` or `ambiguous_convention`,
pause and ask a `Human expert question:` before accepting the result as a
premise or updating the affected source claim as resolved.

In `auto` mode, first judge whether a `likely_source_error` or
`ambiguous_convention` can be resolved with high confidence. High confidence
requires accepted premises, declared conventions, reviewer reasoning, and the
main-agent audit to support one option clearly; the choice must be local to the
current result; no plausible alternative may significantly change later plan
structure, future premises, source coverage, or physical interpretation. If
that standard is met, continue without pausing and record the decision beside
the affected work-note content with `[agent-resolved decision]`, coloring only
that literal marker red. Record the evidence, rejected alternative, and
downstream risk in `Journal` or `Calculation Status`. For PDF-oriented
Markdown, `\textcolor{red}{[agent-resolved decision]}` is acceptable.

If the high-confidence standard is not met, or the decision is important,
foundation-like, convention-setting, or likely to affect later plan structure,
future accepted premises, source coverage, or physical interpretation, pause
and ask a `Human expert question:` before accepting the result as a premise or
updating the affected source claim as resolved.

When pausing for a human expert, do not merely say that the workflow paused.
Write and ask one concrete question under the literal label
`Human expert question:`. The question must name the step, the unresolved equation or claim, the competing options, and what answer is needed to proceed.
Record the same question in `Open Questions` or `Calculation Status`.

## Phase 5: Update Work Note

For an accepted step, update only the current ready-step slot:

- mark the current ready step accepted
- record the selected derivation, current result, and status
- use main prose for the physics argument
- use `Journal` for execution facts, consensus paths, attempts, and reviewer
  judgment

For `confirmed_source_error`, put the literal marker `[confirmed source issue]`
beside the source-disagreement statement, but only color that literal marker
red. Do not color the surrounding prose. Do not color the surrounding equations.
For PDF-oriented Markdown, `\textcolor{red}{[confirmed source issue]}` is
acceptable. Do not use the red marker for likely or convention-dependent source
mismatches; those must remain blocked until the human expert question is
answered.

If the step is blocked, mark the current ready step blocked and record the
disagreement, proposer positions, reviewer judgment, expert question, and
proposed next action. If the block needs human input, ask the exact same
`Human expert question:` in the user-facing response before ending the turn.
Limits diagnose; they are not proof.

Any accepted work-note content whose acceptance depends on a specific human
expert answer resolving an unresolved scientific acceptance question is
human-resolved content. This includes a blocked step that the expert resolves,
a source convention the expert chooses, or a main-agent verification that is
accepted only because the expert allowed that acceptance standard. This does
not include ordinary user task instructions, source excerpts, or constraints.
Add marker `[human-resolved]` beside the accepted content, but only color the
string `human-resolved` blue. Do not color the surrounding prose. Do not color
the surrounding equations. If color is stripped or unavailable, the marker
remains authoritative. For PDF-oriented Markdown,
`[\textcolor{blue}{human-resolved}]` is acceptable.

A human expert later resolves a block and thereby unblocks the workflow; this
is not by itself a completion condition. After recording the human-resolved
result, continue unless the user explicitly asks to pause or stop, a new
`Human expert question:` is outstanding, a tool/runtime blocker prevents
progress, or `interactive` mode requires confirmation:

- If another ready detailed step exists, continue with the next ready detailed step
  using this workflow.
- If no ready detailed step exists but rough/pending coverage remains from the
  original request, write a planning request and return to `plan.md` to promote
  the next coherent chunk.
- End the turn only when requested coverage is complete or one of the stop
  conditions above applies.

When a result may help later steps, record it only as a candidate reusable
result. Promotion to an accepted premise belongs to `plan.md`.

Write an immutable next work-note version at
`<project-dir>/calculate/<run-id>/work-notes/work-note-vNNN.md`, then mirror it
to `<project-dir>/work-note.md`. After writing the root work note, start
`md2pdf(input="<project-dir>/work-note.md")` in the background. Do not wait, and
do not require any separate report.

## Phase 6: Planning Handoff

If proposers, reviewer, or the main agent agree that plan content should
change, or that a candidate reusable result should become a future premise, do
not edit ready-step boundaries, rough steps, or future plan structure.

Write `<project-dir>/calculate/<run-id>/planning-request.md` with:

- current step id and status
- consensus artifact paths
- evidence for the requested change
- proposer positions and reviewer judgment
- requested action for `plan.md`

Then return to `plan.md`. Use the same handoff when blocked refinement needs
splitting, limits, projections, different source context, or changed future
premises. When the issue came from note parsing or claim extraction, refer to
the owning workflow instead of changing it here.
