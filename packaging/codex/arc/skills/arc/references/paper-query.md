# Paper Query Reference

## Single Paper

Use the CLI with `--json`; outputs are result envelopes suitable for MCP:

```bash
arc-paper-query get-title arXiv:0911.3380 --json
arc-paper-query get-section arXiv:0911.3380 --section S2 --json
```

If `get-section` cannot find the requested section, it returns an error
envelope plus `toc`. Use the returned `toc` to choose a valid section.

## LLM Summary

First check cache:

```bash
arc-paper-query get-llm-summary arXiv:0911.3380 --json
```

If the result is a cache hit, use it directly.

If the result has `status: "needs_llm"`:

1. Use `llm_task.system_prompt`, `llm_task.user_prompt`, `llm_task.input_pack`,
   and `llm_task.output_schema`.
2. Generate JSON only, conforming to `output_schema`.
3. Call:

```bash
arc-paper-query store-llm-summary arXiv:0911.3380 --summary-json - --json
```

For automatic generation through the host CLI:

```bash
arc-paper-query generate-llm-summary arXiv:0911.3380 --provider auto --json
```

`--provider auto` uses `ARC_LLM_PROVIDER` first, then `ARC_AGENT_HOST`, then
parent-process detection. Plugin wrappers set these env vars automatically.

## Batch Summary

For more than 10 papers:

```bash
arc-paper-query summary-batch create papers.txt --name qft-ideas --json
arc-paper-query summary-batch prefetch qft-ideas --workers 8 --json
arc-paper-query summary-batch run qft-ideas --provider auto --concurrency 2 --max-items 10 --json
arc-paper-query summary-batch status qft-ideas --json
```

Review the first 10 summaries before running the full batch:

```bash
arc-paper-query summary-batch run qft-ideas --provider auto --concurrency 2 --json
arc-paper-query summary-batch export qft-ideas --format jsonl --output summaries.jsonl --json
```

Retry failures:

```bash
arc-paper-query summary-batch retry-failed qft-ideas --json
```

## Troubleshooting

Check host and provider detection:

```bash
arc-paper-query doctor host --json
arc-paper-query doctor provider --json
```

Expected plugin env:

```text
Codex: ARC_AGENT_HOST=codex, ARC_LLM_PROVIDER=codex-cli
Claude Code: ARC_AGENT_HOST=claude-code, ARC_LLM_PROVIDER=claude-cli
```
