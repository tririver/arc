# Suggest Ideas No-Info Workflow

Use this workflow to run a comparison idea-generation loop where proposers get
only the user's exact intent and internet access. This is an ablation of
`suggest-ideas.md`: do not feed proposers ARC domain Markdown files, ARC paper
tool guidance, local paper-cache context, or MCP access. Keep the reviewer
configuration the same as the normal idea workflow.

## Inputs

Read `<project-dir>/context.json`. Use the exact `user_intent`. Do not
substitute a paraphrased research goal into idea prompts.

### Phase 1: Prepare Comparison Run

Step 1: Create `<project-dir>/suggest-ideas/`.

Step 2: Choose a filesystem-safe `<run-id>` that makes the ablation clear, for
example `no_info_<timestamp>`. The run directory must be:

```text
<project-dir>/suggest-ideas/<run-id>/
```

Step 3: Do not require or attach `<project-dir>/domain/**/*.md` files for this
workflow. The point of this comparison is to withhold ARC-built domain context
from proposers.

### Phase 2: Prepare Idea Loop Config

Step 1: Write an `arc-llm` proposers-reviewer-loop config JSON to:

```text
<project-dir>/suggest-ideas/<run-id>.config.json
```

Start from
`references/research-workflows/suggest-ideas-batch.template.json` and replace
`<run-id>` and `<project-dir>`.

Step 2: Add the requested number of independent loops. Run `5` loops and keep `max_concurrent_loops=10`.
Keep `artifact_options.save_prompts=true` for debugging unless the project
context explicitly disables prompt artifacts.

Do not attach Markdown files under `<project-dir>/domain/` to
`caller_context`. Do not attach ARC-generated reports, paper summaries, paper
tool notes, local paper-cache guidance, logs, transcripts, or unrelated project
notes.

For each loop, start from
`references/research-workflows/suggest-ideas-no-info-loop.template.json` and
replace `<loop-id>` and `<user_intent>`.

Step 3: Configure one proposer per idea loop for the no-info comparison
workflow. The proposer must receive only the user's exact intent, full prior
correspondence, internet permission, and no MCP permission. Use
`references/research-workflows/suggest-ideas-no-info-proposer.template.json`.

Step 4: Configure exactly one reviewer per loop. Set `output_schema` to the
JSON object in
`references/research-workflows/suggest-ideas-reviewer-output.schema.json`. The
written config must contain the schema object, not the placeholder string in
`references/research-workflows/suggest-ideas-reviewer.template.json`. If the
proposer id changes, update the schema's `proposer_messages` keys to match. The
reviewer has ARC-only MCP and internet permission.

The reviewer output must use the `arc.llm.review_envelope.v1` envelope. The
machine-readable marks belong in `review_payload.marks`. The reviewer may
request early stop in the envelope, but this workflow sets
`early_stop.enabled=false`, so all five rounds are recorded.

### Phase 3: Run Idea Loops

Step 1: Run:

```bash
arc-llm proposers-reviewer-loop \
  --config <project-dir>/suggest-ideas/<run-id>.config.json \
  --json
```

Step 2: Inspect the returned JSON. Continue only if every loop status is
`completed` or intentionally `stopped`. If any loop is `failed`, print
`WARNING:` with the failing loop id and stop before writing downstream idea
reports.

### Phase 4: Report Artifacts

Step 1: Report the idea loop artifact root:

```text
<project-dir>/suggest-ideas/<run-id>/
```

Step 2: Report each loop's `state.json`, `transcript.jsonl`, per-round
proposer outputs, per-round reviews, per-round prompt artifacts, and any
per-round worker error artifacts.

Step 3: Do not invent final idea rankings, novelty claims, or gap scores from
memory. Any later idea selection or report must read the recorded loop
artifacts and cite the relevant rounds and reviewer marks.

Step 4: Write `<project-dir>/suggest-ideas/<run-id>/suggested-ideas.md` for
this phase. Collect each loop's recorded proposer outputs and reviewer reviews
from the artifact root. In the main text, outline the generated ideas and their
stated motivation, calculation target, required checks, and main reviewer
concerns. Do not hide weak or worse later rounds.

The appendix must start with a rounds-and-marks summary table. Include one row
per loop and round, with columns for `loop_id`, `round`, idea title or short
label, `evidence_of_novelty`, `feasibility`, `scientific_value`,
`user_intent_fit`, `first_calculation_clarity`, and `total_score`. After the
table, append the detailed correspondence history grouped by loop and round:
proposer output, reviewer message to the controller, reviewer message to the
proposer, and full `review_payload`. Full rendered prompts are debug artifacts
under each round's `prompts/` directory; mention their paths when useful, but do
not paste them into the correspondence appendix by default.

After `suggested-ideas.md` is generated, copy it to
`<project-dir>/suggested-ideas.md` so human readers can inspect the main
project reports together.
