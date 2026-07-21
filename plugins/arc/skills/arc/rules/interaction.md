# ARC Interaction Reference

Use this reference whenever ARC needs any user question, choice, confirmation,
or runtime automation decision.

## Automation Policy

- Default to `automation_level: auto` and do not ask an execution-mode question
  at startup. Continue with safe defaults, preserve visible warnings, and
  record decisions in `<project-dir>/context.json` when a managed run has one.
- Set `automation_level: interactive` only when the user explicitly requests
  manual, step-by-step, staged review, or confirmation at key steps. Pause only
  at the major milestones defined by the owning workflow, not after every tool
  call.
- A request to discuss before running creates a pause before workflow tool calls. It is
  not an automation-level option; after the discussion, apply the user's latest
  explicit instruction or retain the prior level.

The latest explicit steer between `auto` and `interactive` overrides the
current level and persists until the managed run ends or another explicit
steer changes it. A bounded instruction such as "stop after the first chapter"
adds one checkpoint only and does not permanently switch the level. Bare
`continue`, `resume`, or approval at one checkpoint passes that checkpoint but
does not change the automation level.

For managed runs, write the current `automation_level` when creating or
resuming `<project-dir>/context.json`, and update that field in place after a
runtime switch. Direct ARC tool tasks do not create an extra state file; use
the latest explicit instruction in the current session.

An `auto` to `interactive` switch takes effect at the next safe, controllable
boundary. Do not cancel a provider call, CLI command, or background job already
submitted. An `interactive` to `auto` switch while awaiting ordinary milestone
confirmation also approves that checkpoint and continues.

Automation steering never bypasses authorization requirements, destructive
action review, duplicate-charge risk, unresolved scientific ambiguity, a
`Human expert question:`, error recovery, or another mandatory safety gate.
It also does not authorize a downstream workflow. Perform exactly the scope
requested by the caller.

In `interactive`, use these major milestones: Domain pauses after domain
artifacts and the manifest are complete and before an explicitly requested
downstream workflow; Ideas pauses after the top three and before candidate
selection or requested planning/calculation; Check pauses after main-agent
preflight and before the planning handoff; Plan pauses after the work note
passes internal review and before requested calculation; Calculate pauses after
each accepted step or coherent chunk and before the next; Companion uses
`--stop-after-first-chapter` and pauses only after the complete first chapter is
rendered and validated. Direct tool orchestration pauses only between major
user-requested stages, such as after collection/filtering and before batch
summarization or export, never after every underlying call.
For example, stop after domain construction when domain construction was the
requested outcome, and stop after ranked ideas when ideas were the requested
outcome. Run prerequisites required by the requested workflow, but do not turn
them into additional outcomes.

Direct ARC tool tasks default to automatic execution
with safe defaults unless the user explicitly asks to review or confirm steps.
Direct tasks may include several ARC calls, such as collecting citers,
filtering papers by date, generating summaries, using summary batches, looking
up sections or equations, translating named reports, or exporting requested
non-evaluative paper-data outputs. Direct tasks must not produce
recommendations, research directions, scientific rankings, ARC reports, or
project-local workflow artifacts.

Examples:

- `use ARC, in field arXiv:0911.3380, recommend research directions`: use
  `auto` without asking, perform the requested workflows, and stop at scope.
- `use ARC, suggest ideas step by step for massive scalar exchange`: use
  `interactive` and pause after the top three before candidate selection or a
  requested planning/calculation workflow.
- `build the research domain around arXiv:0911.3380`: use `auto`, build the
  domain, and stop.
- `use arc, what is the title and abstract of arXiv:0911.3380?`: direct paper lookup allowed;
  no automation mode question is needed.
- `use arc to download papers that cited 0911.3380 since 2024 and create a full
  summary of these papers`: direct tool orchestration allowed; no automation
  question is needed. If staged review was requested, pause only between the
  major collection/filtering and batch-summary/export stages.

## Selection Protocol

Use this protocol for real business choices such as seeds, project directories,
or candidate ideas. Do not use it to choose an automation level.

Ask user questions through the host's selection/menu tool when one is
available. Use a typed fallback only when no suitable selection/menu tool is
available or a tool call is rejected.

The default/recommended option must be first, so pressing Enter chooses it.
The user can use arrow keys to choose another option.

Every choice prompt must:

- Use two or three real, bounded options.
- Put the recommended/default continuation first and end that label with
  `(Recommended)`.
- Use short, informative option labels. Do not include list numbering inside
  option labels, such as `1. Run`, `2:`, or `3:`.
- Avoid open-ended prose questions when a bounded choice can express the
  decision.

If no selection/menu tool is available and a question is required, use a
portable typed fallback with the same bounded options. Present the options as a
short numbered list, mark the first option as the default, and ask the user to
enter the exact option label or number. Pressing Enter selects the default.

## Existing Project Directory

When `<project-dir>` already exists:

- In `interactive` mode, ask with these options: `Reuse existing directory
  (Recommended)`, `Archive existing directory and create fresh`, and
  `Choose another directory`.
- In `auto` mode, rename the existing directory to
  `<project-dir>_yy-mm-dd-hh-mm-ss`, then create a fresh `<project-dir>`.

## Multiple Seeds

When seed resolution returns multiple papers:

- In `interactive` mode, ask with these options: `Use all seeds
  (Recommended)`, `Choose seed subset`, and `Use first seed only`.
- In `auto` mode, keep all returned seed papers.
