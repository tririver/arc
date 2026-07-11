# Ideas Workflow

Use this workflow for Case 2 idea generation. It runs every enabled idea
variant as concurrent proposer-reviewer loops. Each loop has exactly one
proposer and exactly one reviewer; the reviewer serves only that proposer and
sends five reviewer reports per loop by default.

## Inputs

Read `<project-dir>/context.json`. Use the exact `user_intent`.
Use `skill_dir` from context as `<skill-dir>` in commands below.
If `<project-dir>/context.json` is missing, or if it does not contain an
explicit `automation_level`, return to `SKILL.md` Phase 1 Step 1 before doing
any idea-generation work. Agent-invoked or implicit ARC requests receive
`automation_level: auto` there without a mode question.
Do not synthesize ideas manually.

### Phase 1: Prepare Config

Step 1: Create `<project-dir>/ideas/`.

Step 2: Copy
`workflows/json/ideas.config.template.json` to:

```text
<project-dir>/ideas/<run-id>.config.json
```

Step 3: Replace `<run-id>`, `<project-dir>`, `<user_intent>`, and
`<skill-workflow-json-dir>`.

Step 4: Keep `variant_glob` as `ideas-*.variant.json`. The release package
runs only enabled variants; the normal idea-generation workflow uses the
domain variant.

Step 5: Keep the shipped proposer and reviewer `model_tier` values at `high`
for normal idea generation unless the user requests another quality/cost tier.

Step 6: Keep `loops_per_variant` at `5` unless the run should use a different
number of concurrent instances for each setup.

### Phase 2: Run Ideas

Step 1: Run:

```bash
python3 <skill-dir>/workflows/scripts/ideas_runner.py \
  --config <project-dir>/ideas/<run-id>.config.json \
  --json
```

Step 2: Print any returned `WARNING:` messages. For loop concurrency, see
`manuals/arc-llm.md`.

Step 3: Continue only if the returned status is `completed`. If status is
`failed`, print `WARNING:` with the error and artifact root.

The workflow runner writes this generated batch config before launch:

```text
<project-dir>/ideas/<run-id>/ideas_batch_config.json
```

For proposer-reviewer artifact ownership and runner result shape, see
`manuals/arc-llm.md`.
Final ranked ideas must come from `ideas_runner.py` artifacts and the
read-only ranking helper, not ad-hoc agent judgment.

### Phase 3: Inspect Artifacts

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
python3 <skill-dir>/workflows/scripts/rank-ideas.py \
  <project-dir>/ideas/<run-id>/idea_loops \
  --format markdown \
  > <project-dir>/ideas/<run-id>/ranked-ideas.md

python3 <skill-dir>/workflows/scripts/rank-ideas.py \
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
Markdown paragraphs, not a fenced code block. Follow `rules/math_typeset.md`
for math and TeX snippets. Use PDF-friendly wrapping for long titles and
proposer text; avoid wide tables with long prose.

Step 2: After writing the project-level Markdown report, follow
`manuals/arc-mcp.md` Markdown Report Export for
`md2pdf(input="<project-dir>/ranked-ideas.md")`. This report-export gate is not
satisfied until `md2pdf` has been started or a `WARNING:` with the exact blocker
is recorded. Do not wait for PDF completion.
If PDF generation appears bugged, report it and continue this workflow; do not
debug or fix PDF generation unless the user explicitly asks.

Do not invent rankings or novelty claims. Use the recorded proposer outputs and
per-round reviewer reports from the `arc-llm` loop artifacts.

### Phase 4: Select Next Action

Step 1: Print the top three ranked ideas on screen.

Step 2: Stop after printing the top three ideas unless the caller explicitly
requested planning or calculation as part of the original request. In
particular, `auto` does not authorize either a selection question or a move to
calculation.

If the caller explicitly requested calculation after idea generation, proceed
with ranked idea #1 in `auto` mode without asking. In `interactive` mode, use
the host's selection/menu tool, following `rules/interaction.md`, with these
option labels exactly:

- `Proceed with ranked idea #1 (Recommended)`
- `Proceed with ranked idea #2`
- `Proceed with ranked idea #3`

If no selection/menu tool is available, use the typed fallback from
`rules/interaction.md` with the same three options.

If the workflow is running in interactive mode and calculation was not
explicitly requested, stop after printing the top three ideas.
