# Research Ideas Workflow

Use this workflow for Case 2 idea generation. It runs every enabled idea
variant as iterative proposer-reviewer loops, selects each loop's highest-scored
round, then asks one global reviewer to score the selected ideas on one common
scale.

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

### Phase 2: Check Planned Calls

Step 1: Run:

```bash
python3 references/research-workflows/research_ideas_runner.py \
  --config <project-dir>/research-ideas/<run-id>.config.json \
  --dry-run \
  --json
```

Step 2: Print any returned `WARNING:` messages. Unlimited loop concurrency is
intentional for this workflow.

### Phase 3: Run Ideas

Step 1: Run:

```bash
python3 references/research-workflows/research_ideas_runner.py \
  --config <project-dir>/research-ideas/<run-id>.config.json \
  --json
```

Step 2: Continue only if the returned status is `completed`. If status is
`failed`, print `WARNING:` with the error and artifact root.

### Phase 4: Report Artifacts

Report these paths:

```text
<project-dir>/research-ideas/<run-id>/
<project-dir>/research-ideas/<run-id>/loop_batch/idea_loops/loops/
<project-dir>/research-ideas/<run-id>/research-ideas.md
<project-dir>/research-ideas/<run-id>/global_review/review.json
<project-dir>/research-ideas.md
```

Do not invent rankings or novelty claims. Use the recorded proposer outputs and
per-round reviews, selected-round records, and global review JSON.
