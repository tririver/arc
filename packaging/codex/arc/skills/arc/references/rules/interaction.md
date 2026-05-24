# ARC Interaction Reference

Use this reference whenever ARC needs any user question, choice,
confirmation, or mode decision.

## Automation Modes

- `auto`: continue without asking additional questions. Use safe defaults,
  preserve warnings, and write decisions to `<project-dir>/context.json`.
- `interactive`: ask for confirmation after each major workflow step and before
  destructive or ambiguous actions.

If the mode is not clear from the user's request, ask once before resolving
seeds, creating project directories, building domains, suggesting ideas, or
planning calculations.

## Discrete Selection Protocol

Ask user questions through the host's discrete selection tool when one is
available. Do not use open-ended prose questions when a bounded choice can
express the decision.

The default/recommended option must be first, so pressing Enter chooses it.
The user can use arrow keys to choose another option.

Every choice prompt must:

- Use two or more bounded options.
- Put the recommended/default continuation first.
- Make the final option exactly `Let's discuss`.
- Avoid open-ended prose questions when a bounded choice can express the
  decision.

If no discrete selection tool is available and a question is required, use a
portable typed fallback with the same bounded options. Present the options as a
short numbered list, mark the first option as the default, and ask the user to
enter the exact option label or number. Pressing Enter selects the default.
Keep `Let's discuss` as the final option.

## Existing Project Directory

When `<project-dir>` already exists:

- In `interactive` mode, ask whether to reuse it, rename the existing
  directory, choose another directory, or `Let's discuss`.
- In `auto` mode, rename the existing directory to
  `<project-dir>_yy-mm-dd-hh-mm-ss`, then create a fresh `<project-dir>`.

## Multiple Seeds

When seed resolution returns multiple papers:

- In `interactive` mode, ask whether to use all seeds, choose a subset, use the
  first seed only, or `Let's discuss`.
- In `auto` mode, keep all returned seed papers.
