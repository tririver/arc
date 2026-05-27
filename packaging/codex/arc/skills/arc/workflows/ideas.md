# Ideas Workflow

Use this workflow for Case 2 idea generation. It runs every enabled idea
variant as concurrent proposer-reviewer loops. Each loop has exactly one
proposer and exactly one reviewer; the reviewer serves only that proposer and
sends five reviewer reports per loop by default.

## Inputs

Read `<project-dir>/context.json`. Use the exact `user_intent`.

### Phase 1: Prepare Config

Step 1: Create `<project-dir>/ideas/`.

Step 2: Copy
`workflows/json/ideas.config.template.json` to:

```text
<project-dir>/ideas/<run-id>.config.json
```

Step 3: Replace `<run-id>`, `<project-dir>`, `<user_intent>`, and
`<skill-workflow-json-dir>`.

Step 4: Keep `variant_glob` as `ideas-*.variant.json`. To disable a
variant, rename it so it no longer matches, for example
`ideas-no-info.variant_inactivated.json`.

Step 5: Keep `loops_per_variant` at `5` unless the run should use a different
number of concurrent instances for each setup.

### Phase 2: Check Planned Calls

Step 1: Run:

```bash
python3 workflows/scripts/ideas_runner.py \
  --config <project-dir>/ideas/<run-id>.config.json \
  --dry-run \
  --json
```

Step 2: Print any returned `WARNING:` messages. Unlimited loop concurrency is
intentional for this workflow. The dry run reports the generated loop plan but
does not create run artifacts.

### Phase 3: Run Ideas

Step 1: Run:

```bash
python3 workflows/scripts/ideas_runner.py \
  --config <project-dir>/ideas/<run-id>.config.json \
  --json
```

Step 2: Continue only if the returned status is `completed`. If status is
`failed`, print `WARNING:` with the error and artifact root.

The workflow runner writes only the generated batch config before launch:

```text
<project-dir>/ideas/<run-id>/ideas_batch_config.json
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
<project-dir>/ideas/<run-id>/
<project-dir>/ideas/<run-id>/ideas_batch_config.json
<project-dir>/ideas/<run-id>/idea_loops/
<project-dir>/ideas/<run-id>/idea_loops/loops/
```

Step 1: After the run completes, use the read-only ranking helper to write the
deterministic ranked ideas report directly to both readable destinations:

```bash
python3 workflows/scripts/rank-ideas.py \
  <project-dir>/ideas/<run-id>/idea_loops \
  --format markdown \
  > <project-dir>/ideas/<run-id>/ranked-ideas.md

python3 workflows/scripts/rank-ideas.py \
  <project-dir>/ideas/<run-id>/idea_loops \
  --format markdown \
  > <project-dir>/ranked-ideas.md
```

The report must start with `# Ideas`, then `Abbreviations:`, then a
blank-line-separated abbreviation line in the form `IR=intent relevance,
N=novelty, CN=confidence of novelty, SV=scientific value, PL=planning,
WD=well-definedness, T=total.` List each ranked idea in the same form used by
`round_marks_by_idea.md`: a loop-id heading, the selected title, and the
compact round marks table with columns `Round`, `IR`, `N`, `CN`, `SV`, `PL`,
`WD`, and `T`. The report must then include `# Appendix: Idea Details` with one subsection per
ranked idea. Each subsection lists all referee marks from
every round in that idea loop and quotes only the selected handoff text: title,
idea summary, and calculation plan. Render that handoff text as normal
Markdown paragraphs, not a fenced code block. Use PDF-friendly wrapping for
long titles and proposer text; avoid wide tables with long prose.

Step 2: After writing the project-level Markdown report, call
MCP `md2pdf(input="<project-dir>/ranked-ideas.md")`. It starts a background
PDF job; record the returned job id if present and do not wait before
continuing.

Do not invent rankings or novelty claims. Use the recorded proposer outputs and
per-round reviewer reports from the `arc-llm` loop artifacts.

### Phase 5: Select Next Action

Step 1: Print the top three ranked ideas on screen.

Step 2: If the workflow is running in auto mode, use the host's discrete
selection tool, following `rules/interaction.md`, to ask whether to
proceed to calculation. Use these option labels exactly:

- `1` (default): proceed with ranked idea #1.
- `2`: proceed with ranked idea #2.
- `3`: proceed with ranked idea #3.
- `other`: enter another ranked idea number.
- `Let's discuss`: stop automated progression and discuss.

Do not render numbered-list prefixes inside option labels; for example, use
label `1`, not `1. 1`, and label `2`, not `2:`.
The option labels must be the raw labels listed above.

If no discrete selection tool is available, ask only for the idea number or
`other`. Do not print `quit` or `Let's discuss` in the typed fallback because
not selecting an idea already leaves the workflow in discussion mode.

If the workflow is running in interactive mode, stop after printing the top
three ideas.
