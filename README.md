# ARC Dev

ARC research tooling is organized as Python packages plus thin agent adapters.

## Packages

Install the current development packages:

```bash
python3 -m venv packages/arc-paper-query/.venv
. packages/arc-paper-query/.venv/bin/activate
python -m pip install -e packages/arc-paper-query[test]
python -m pip install -e packages/arc-mcp
```

## Paper Query

Deterministic paper data:

```bash
arc-paper-query get-title arXiv:0911.3380 --json
arc-paper-query get-references arXiv:0911.3380 --json
arc-paper-query get-toc arXiv:0911.3380 --json
arc-paper-query get-section arXiv:0911.3380 --section S2 --json
```

LLM summaries:

```bash
arc-paper-query get-llm-summary arXiv:0911.3380 --json
arc-paper-query generate-llm-summary arXiv:0911.3380 --provider auto --json
```

Batch workflow:

```bash
arc-paper-query summary-batch create papers.txt --name qft-ideas --json
arc-paper-query summary-batch prefetch qft-ideas --workers 8 --json
arc-paper-query summary-batch run qft-ideas --provider auto --concurrency 2 --max-items 10 --json
arc-paper-query summary-batch status qft-ideas --json
arc-paper-query summary-batch export qft-ideas --format jsonl --output summaries.jsonl --json
```

## Host Detection

Plugins should set:

```text
ARC_AGENT_HOST=codex
ARC_LLM_PROVIDER=codex-cli
```

or:

```text
ARC_AGENT_HOST=claude-code
ARC_LLM_PROVIDER=claude-cli
```

Without plugin env, `arc-paper-query` falls back to parent-process detection.

Debug:

```bash
arc-paper-query doctor host --json
arc-paper-query doctor provider --json
```

## Reference Code

`0_ref/` is read-only reference material. New code must not modify it or preserve
old compatibility assumptions.
