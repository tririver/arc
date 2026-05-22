# ARC Integrity Reference

This reference applies to all ARC research workflows.

## Scientific Integrity Notice

- This skill is part of serious, high-quality scientific research.
- Be completely honest at every step.
- Never fabricate gap scores, coverage data, pairwise support counts, citation
  evidence, paper relevance, calculation results, or workflow status for any
  reason.
- If a gap, score, citation relationship, or claim cannot be confidently
  supported, flag it explicitly.
- Never cheat to make an output look more complete or impressive.

## Robustness Notice

- Always follow the workflow steps as you work.
- Do not silently skip or compress important decisions.
- Do not skip steps or kill scripts/jobs because they are slow or temporarily
  quiet.
- Whenever you detect a potential error, anomalous result, inconsistency, or
  suspicious pattern, print a visible `WARNING:` message immediately and explain
  the concern before proceeding.
- Automation never silences warnings.
- If a downstream workflow is specified by this skill, do not bypass it by
  writing the final user-facing artifact directly.
- If host delegation or parallel workers are unavailable, fall back to
  single-agent sequential execution while preserving the same artifact contract.
