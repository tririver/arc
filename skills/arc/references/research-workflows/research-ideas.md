# Research Ideas Workflow

Use this workflow for Case 2 idea generation. It runs every enabled idea
variant as concurrent proposer-reviewer loops. Each loop has exactly one
proposer and exactly one reviewer; the reviewer serves only that proposer and
sends five reviewer reports per loop by default.

## Inputs

Read `<project-dir>/context.json`. Use the exact `user_intent`.

### Phase 1: Prepare Config

Step 1: Create `<project-dir>/research-ideas/`.

Step 2: Copy
`references/research-workflows/research-ideas.config.template.json` to:

```text
<project-dir>/research-ideas/<run-id>.config.json
```

Step 3: Replace `<run-id>`, `<project-dir>`, `<user_intent>`, and
`<skill-workflow-dir>`.

Step 4: Keep `variant_glob` as `suggest-ideas-*.variant.json`. To disable a
variant, rename it so it no longer matches, for example
`suggest-ideas-no-info.variant_inactivated.json`.

Step 5: Keep `loops_per_variant` at `5` unless the run should use a different
number of concurrent instances for each setup.

### Phase 2: Check Planned Calls

Step 1: Run:

```bash
python3 references/research-workflows/research_ideas_runner.py \
  --config <project-dir>/research-ideas/<run-id>.config.json \
  --dry-run \
  --json
```

Step 2: Print any returned `WARNING:` messages. Unlimited loop concurrency is
intentional for this workflow. The dry run reports the generated loop plan but
does not create run artifacts.

### Phase 3: Run Ideas

Step 1: Run:

```bash
python3 references/research-workflows/research_ideas_runner.py \
  --config <project-dir>/research-ideas/<run-id>.config.json \
  --json
```

Step 2: Continue only if the returned status is `completed`. If status is
`failed`, print `WARNING:` with the error and artifact root.

The workflow runner writes only the generated batch config before launch:

```text
<project-dir>/research-ideas/<run-id>/research_ideas_batch_config.json
```

All concurrent proposer-reviewer artifacts are owned by `arc-llm` under the
batch run root. The workflow runner does not copy selected rounds or write a
project-level latest report while loops are running.

The runner result includes `round_score_table`, a Markdown and structured
per-loop table of reviewer total scores by round, built from the loop artifacts
available at completion time.

### Phase 4: Inspect Artifacts

Report these paths:

```text
<project-dir>/research-ideas/<run-id>/
<project-dir>/research-ideas/<run-id>/research_ideas_batch_config.json
<project-dir>/research-ideas/<run-id>/idea_loops/
<project-dir>/research-ideas/<run-id>/idea_loops/loops/
```

Step 1: After the run completes, use the read-only ranking helper to write the
deterministic ranked ideas report:

```bash
python3 references/research-workflows/scripts/rank-suggested-ideas.py \
  <project-dir>/research-ideas/<run-id>/idea_loops \
  --format markdown \
  > <project-dir>/research-ideas/<run-id>/ranked-ideas.md
```

Step 2: Copy the same report to `<project-dir>/ranked-ideas.md` so human
readers can inspect the main project reports together.

The report must start with `Suggested ideas:` followed by one blank line, then
one idea per paragraph in the form `<Title> (Mark: <Mark>)`, separated by blank
lines. Do not use a table or dense numbered list for the summary. The report
must then include an appendix with one subsection
per ranked idea. Each subsection lists all referee marks from every round in
that idea loop and quotes only the selected handoff text: title, idea summary,
and calculation plan. Use PDF-friendly wrapping for long titles and proposer
text; avoid wide code blocks or wide tables with long prose.

Step 3: After copying the Markdown report, call
MCP `md2pdf(input="<project-dir>/ranked-ideas.md")`. It starts a background
PDF job; record the returned job id if present and do not wait before
continuing.

Do not invent rankings or novelty claims. Use the recorded proposer outputs and
per-round reviewer reports from the `arc-llm` loop artifacts.
