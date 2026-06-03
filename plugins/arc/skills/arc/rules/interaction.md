# ARC Interaction Reference

Use this reference whenever ARC needs any user question, choice,
confirmation, or mode decision.

## Automation Mode Gate

- `Run automatically (Recommended)` / `auto`: continue without asking
  additional questions. Use safe defaults,
  preserve warnings, and write decisions to `<project-dir>/context.json`.
- `Confirm major steps` / `interactive`: ask for confirmation after each major
  workflow step and before destructive or ambiguous actions.
- `Discuss before running`: stop before ARC workflow tool calls and ask what the
  user wants to change.

If the mode is not clear from the user's request and the task is a workflow
deliverable, ask once before resolving seeds, creating project directories,
building domains, suggesting ideas, checking notes, planning, or calculating.
Do not gather "just context" with ARC paper/domain/LLM tools before the mode
choice.

Examples:

- `use arc, in field arXiv:0911.3380, recommend research directions`: ask
  for `Run automatically (Recommended)`, `Confirm major steps`, or
  `Discuss before running` before any ARC paper/domain tool call.
- `use arc, suggest ideas for massive scalar exchange around arXiv:0911.3380`:
  ask for mode first; after the choice, route through Case 1 and Case 2.
- `use arc, what is the title and abstract of arXiv:0911.3380?`: direct paper lookup allowed;
  no automation mode question is needed.

If a workflow deliverable lacks an explicit mode, stop before tool calls and
ask for mode. Do not infer `auto` from `continue`, `resume`, or a bare approval.

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
