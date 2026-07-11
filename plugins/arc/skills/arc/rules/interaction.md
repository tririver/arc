# ARC Interaction Reference

Use this reference whenever ARC needs any user question, choice,
confirmation, or mode decision.

## Automation Mode Gate

- `Run automatically (Recommended)` / `auto`: continue without asking
  additional questions. Use safe defaults,
  preserve warnings, and write decisions to `<project-dir>/context.json` when a
  managed workflow creates one.
- `Confirm major steps` / `interactive`: ask for confirmation after each major
  workflow step and before destructive or ambiguous actions.
- `Discuss before running`: stop before ARC workflow tool calls and ask what the
  user wants to change.

Ask for an automation mode only when both conditions are true:

1. The current request came directly from a human whose prompt explicitly
   names ARC. Quoted, forwarded, or delegated text does not count as a direct
   human invocation. If the host does not expose reliable provenance, this
   condition is not satisfied.
2. The task is a managed ARC workflow run: domain construction, idea
   generation, note checking, planning, calculation, or project-local workflow
   artifacts owned by those workflows, including recommendations, research
   directions, scientific rankings, or ARC reports.

When both conditions hold, ask once before resolving seeds, creating project
directories, building domains, suggesting ideas, checking notes, planning, or
calculating. Do not gather "just context" with ARC paper/domain/LLM tools before
the mode choice.

For an agent-invoked managed workflow, or a human prompt that does not
explicitly name ARC, do not ask for a mode; set the execution mode to `auto`
and perform exactly the scope requested by the caller. Automatic execution
removes routine confirmations; it does not authorize a downstream workflow.
For example, stop after domain construction when domain construction was the
requested outcome, and stop after ranked ideas when ideas were the requested
outcome. Run prerequisites required by the requested workflow, but do not turn
them into additional outcomes.

Direct ARC tool tasks do not need an automation mode. Run them automatically
with safe defaults unless the user explicitly asks to review or confirm steps.
Direct tasks may include several ARC calls, such as collecting citers,
filtering papers by date, generating summaries, using summary batches, looking
up sections or equations, translating named reports, or exporting requested
non-evaluative paper-data outputs. Direct tasks must not produce
recommendations, research directions, scientific rankings, ARC reports, or
project-local workflow artifacts.

Examples:

- Direct human prompt, `use ARC, in field arXiv:0911.3380, recommend research directions`: ask
  for `Run automatically (Recommended)`, `Confirm major steps`, or
  `Discuss before running` before any ARC paper/domain tool call.
- Direct human prompt, `use ARC, suggest ideas for massive scalar exchange around arXiv:0911.3380`:
  ask for mode first; after the choice, route through Case 1 and Case 2.
- Agent-delegated request, `build the research domain around arXiv:0911.3380`:
  do not ask for mode; use `auto`, build the domain, and stop.
- Direct human prompt, `build the research domain around arXiv:0911.3380`,
  without naming ARC: do not ask for mode; use `auto`, build the domain, and
  stop.
- `use arc, what is the title and abstract of arXiv:0911.3380?`: direct paper lookup allowed;
  no automation mode question is needed.
- `use arc to download papers that cited 0911.3380 since 2024 and create a full
  summary of these papers`: direct tool orchestration allowed; no automation
  mode question is needed.

If a mode-eligible managed workflow run lacks an explicit mode, stop before
tool calls and ask for mode. Otherwise default to `auto` without asking. Do not
infer `auto` from `continue`, `resume`, or a bare approval in a mode-eligible
human invocation.

## Selection Protocol

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
