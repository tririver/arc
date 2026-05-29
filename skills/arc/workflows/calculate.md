# Calculate Workflow

Use this workflow after `plan.md` writes `<project-dir>/work-note.md`.
Execute only steps marked ready in `Detailed Steps Ready To Calculate`.
Do not write a separate calculation report; the updated work note is the
human-readable result.

`calculate.md owns consensus execution` and the `current-step result-status`.
It does not change ready-step boundaries, does not change rough steps, and
does not change future plan structure. Calculate does not own note parsing. When a
different workflow owns the needed change, refer to the owning workflow.

## Phase 1: Prepare Runtime

Consensus runtime artifacts:

```text
<project-dir>/calculate/<run-id>/execute/consensus.config.json
<project-dir>/calculate/<run-id>/execute/<consensus-run-id>/
```

Use this default consensus config, adjusting only documented run paths and step
packets:

```json
{
  "schema_version": "arc.llm.proposers_reviewer_consensus.config.v1",
  "run_id": "<consensus-run-id>",
  "run_dir": "<project-dir>/calculate/<run-id>/execute",
  "proposer_count": 2,
  "max_recalculations": 2,
  "human_gate": {"enabled": false},
  "artifact_options": {"save_prompts": true},
  "steps": []
}
```

`"max_recalculations": 2` means 3 total attempts: 1 initial attempt + 2
recalculations. Do not increase attempts unless the user asks.

Remove foundation_check mechanics. Starting points are checked by ordinary ready
steps when they are marked not accepted in the work note.

## Phase 2: Build Step Packets

For each current ready step, build one proposer packet from:

- current step prompt and quantity contract
- relevant work-note sections: notation, axioms, accepted results, and the
  current ready step
- clean proposer-facing source context

Do not expose reviewer-only targets, target equations, or later note text to
proposers.

For a blind reference check, include `reviewer_reference_claim` only for the
reviewer and disable source tools:

```json
"proposer_runtime": {"allow_internet": false, "allow_mcp": false}
```

If blind proposers agree with each other but not with the reviewer reference,
record `reference_disagrees` and pause unless the mismatch only asks for a
planning request.

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
arc-llm proposers-reviewer-consensus \
  --config <project-dir>/calculate/<run-id>/execute/consensus.config.json \
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

## Phase 5: Update Work Note

For an accepted step, update only the current ready-step slot:

- mark the current ready step accepted
- record the selected derivation, current result, and status
- use main prose for the physics argument
- use `Journal` for execution facts, consensus paths, attempts, and reviewer
  judgment

If the step is blocked, mark the current ready step blocked and record the
disagreement, proposer positions, reviewer judgment, expert question, and
proposed next action. Limits diagnose; they are not proof.

When a result may help later steps, record it only as a candidate reusable
result. Promotion to an accepted premise belongs to `plan.md`.

Write an immutable next work-note version at
`<project-dir>/calculate/<run-id>/work-notes/work-note-vNNN.md`, then mirror it
to `<project-dir>/work-note.md`. After writing the root work note, start
`md2pdf(input="<project-dir>/work-note.md")` in the background. Do not wait, and
do not require any separate report.

## Phase 6: Planning Handoff

If proposers, reviewer, or the main agent agree that plan content should change,
or that a candidate reusable result should become a future premise, do not edit
ready-step boundaries, rough steps, or future plan structure.

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
