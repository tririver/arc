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
<project-dir>/research-ideas.md
```

Do not invent rankings or novelty claims. Use the recorded proposer outputs and
per-round reviewer reports.
