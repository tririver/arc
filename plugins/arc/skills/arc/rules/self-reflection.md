# Self-Reflection

Before marking any ARC workflow complete, check the requested skill outcome
against the artifacts and user-facing result. Confirm that the workflow
delivered what the skill requested. If it did not, identify the concrete gap
and likely reason, such as missing inputs, failed checks, incomplete artifacts,
tool/runtime limits, or an instruction conflict.

Append the outcome check and self-reflection entry to
`<project-dir>/self-reflect.md`.

Start each suggestion with available provenance:

```text
Git: <commit-hash>
Run: <run_id>
```

If Git metadata is unavailable, use:

```text
Git: unavailable
Archive: <checksum or extracted-dir name>
Run: <run_id>
```

Include concrete, portable improvement suggestions when the run reveals a
workflow, prompt, package, documentation, cache, or test weakness.

If the requested outcome was not delivered, append the reason and at least one
actionable follow-up suggestion unless the blocker is entirely outside ARC's
control.

Make the suggestion actionable: affected file or phase, evidence from the run,
exact command or edit to try, and an acceptance check.

If no concrete improvement was found, still append a dated entry saying that no
actionable ARC improvement was identified for this run. The workflow is not
complete until this append step is done.
