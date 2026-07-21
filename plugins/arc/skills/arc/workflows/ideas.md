# Ideas Workflow

Use this workflow for Case 2 idea generation. It selects the single-domain or
cross-domain variant from the project domain manifest, then runs concurrent
proposer-reviewer loops. Each loop has exactly one proposer and exactly one
reviewer; the reviewer serves only that proposer and sends three reviewer reports per loop by default.

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

Set `domain_manifest_path` to
`<project-dir>/domain/domain-manifest.json`. Manifest v2 routes by
`field_count`: one field, including multiple seed-specific packages, uses the
single-domain prompts; two or more fields use cross-domain prompts, directed
transfer profiles, reviewer assessment, and qualification gates. Cross-domain
cards and source/target roles use `field_id`. A v1, missing, or invalid
manifest must be regenerated before cross-domain work.

Step 4: Keep `variant_glob` as `ideas-*.variant.json`. The release package
runs only enabled variants, then selects only the enabled variant applicable
to the manifest; it must not run both the single-domain and cross-domain
variants for the same request.

Step 5: Keep the shipped proposer and reviewer `model_tier` values at `high`
for normal idea generation unless the user requests another quality/cost tier.

Step 6: Keep `loops_per_variant` at `5` unless the run should use a different
number of concurrent instances for each setup. Cross-domain runs ship with
five distinct exploration profiles. If a different loop count is required,
set top-level `exploration_profiles` to the same number of profile objects so
the runner never creates duplicate loops that differ only by ID.

### Phase 2: Run Ideas

Step 1: Run:

```bash
python3 <skill-dir>/workflows/scripts/ideas_runner.py \
  --config <project-dir>/ideas/<run-id>.config.json \
  --json
```

LLM calls have no absolute runtime limit and stop after 30 minutes with no
substantive provider output. The foreground runner streams progress JSONL to
stderr. Keep its terminal session active and inspect the latest excerpt and
artifacts at each 30-minute `review_due`. Concrete results, new evidence,
reusable artifacts, or meaningful narrowing mean continue waiting in the same
session. Repetitive, erroneous, or off-task output means send `SIGINT` or
`SIGTERM`; never interrupt merely because a run is long. A terminal result
ends the foreground review loop normally. When an external
controller launches the runner with an ARC job side channel, use the job-level
`review_sequence` cursor loop documented in `manuals/arc-jobs.md` instead.
Override the idle bound through `worker_idle_timeout_seconds`,
`--idle-timeout-seconds`, or an applicable `ARC_*_IDLE_TIMEOUT_SECONDS`
variable. `SIGINT`, `SIGTERM`, and background cancellation remain available.

Step 2: Print any returned `WARNING:` messages. For loop concurrency, see
`manuals/arc-llm.md`.

Proposers and reviewers use `arc-paper-worker` directly when their host offers
reliable shell execution. They may query it repeatedly in one turn; writes go
to the run overlay and validated records are promoted automatically. On hosts
without direct CLI support, the workflow controller resolves equivalent
structured evidence requests through the deterministic `arc-paper` service
between rounds and returns provenance in the next worker context. Other ARC
CLIs, nested LLM entry points, and MCP tools remain unavailable. See
`manuals/arc-llm.md`.

Step 3: Continue when status is `completed` or `degraded`. For `degraded`,
print a prominent `WARNING:` with failed/degraded loop counts and the artifact
root, then rank only usable loops with valid proposer and reviewer results. If
status is `failed` or `cancelled`, print `WARNING:` and stop before ranking.

The workflow runner writes this generated batch config before launch:

```text
<project-dir>/ideas/<run-id>/ideas_batch_config.json
```

For proposer-reviewer artifact ownership and runner result shape, see
`manuals/arc-llm.md`.
Final ranked ideas must come from `ideas_runner.py` artifacts and the read-only
ranking helper, not ad-hoc agent judgment. This includes usable loops from a
degraded batch. In cross-domain mode,
only candidates marked as genuine transfers with a substantial target-domain
contribution and a feasible first calculation are eligible for the formal
ranking. The source domain may contribute a mature method or mechanism without
itself receiving a new result.

In single-domain mode, prioritize an important target-domain problem that is
mathematically well-defined and has an executable systematic route. A mature
method from another field may be imported when its structure, required
adaptation, applicability conditions, validation checks, and kill criterion are
made concrete; only the target domain needs a substantive result. Feasibility
is a qualification gate, while problem importance is scored strongly rather
than used as a binary gate. Do not promote a convenient but low-value exercise,
or an important problem without ready inputs and a bounded first calculation.

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

For cross-domain runs, use the abbreviations and score columns declared by the
selected cross-domain marking scheme. List qualified candidates first in
formal ranking order, never fill the top three with an unqualified candidate,
and add an unqualified appendix with explicit reasons. The helper also writes
`<project-dir>/ideas/<run-id>/cross-domain-diagnostics.json`; report that path
and print any insufficient-qualified-candidate `WARNING:` messages.

For new single-domain runs, formal ranking likewise contains only candidates
that pass the mathematical-definition and feasibility gate. Do not pad the top
three with infeasible candidates. Add explicit failures to the unqualified
appendix, write `<project-dir>/ideas/<run-id>/single-domain-diagnostics.json`,
and print any insufficient-qualified-candidate `WARNING:` messages. Historical
artifacts without `idea_assessment` remain readable under the visibly reported
`legacy_no_feasibility_gate` policy.

Step 2: After writing the project-level Markdown report, follow
`manuals/arc-jobs.md` Markdown Report Export for
`<project-dir>/ranked-ideas.md`. This report-export gate is not
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
