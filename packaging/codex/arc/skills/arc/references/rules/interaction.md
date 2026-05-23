# ARC Interaction Reference

Use this reference whenever ARC needs a user choice, confirmation, or mode
decision.

## Automation Modes

- `auto`: continue without asking additional questions. Use safe defaults,
  preserve warnings, and write decisions to `<project-dir>/context.json`.
- `interactive`: ask for confirmation after each major workflow step and before
  destructive or ambiguous actions.

If the mode is not clear from the user's request, ask once before resolving
seeds, creating project directories, building domains, suggesting ideas, or
planning calculations.

## Discrete Selection Protocol

Always ask choices through the host's discrete selection tool when it is
available. The default/recommended option must be first, so pressing Enter
chooses it.

Every choice prompt must:

- Use two or more bounded options.
- Put the recommended/default continuation first.
- Make the final option exactly `Let's discuss`.
- Avoid open-ended prose questions when a bounded choice can express the
  decision.

If the host genuinely has no discrete selection UI/tool, show the same options
as a numbered list with `Let's discuss` last.

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
