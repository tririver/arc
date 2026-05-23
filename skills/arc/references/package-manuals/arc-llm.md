# Arc LLM Package

`arc-llm` is the reusable host LLM worker used by ARC packages. Most workflows
should call `arc-paper`, `arc-domain`, or ARC MCP tools instead of calling
`arc-llm` directly. Use this reference for provider diagnosis, direct prompt
tests, and advanced LLM runtime options.

## Provider Diagnosis

### Phase 1: Check host detection.
Step 1: Run:

```bash
arc-llm doctor host
arc-llm doctor provider
arc-llm doctor config
```

### Phase 2: Check package-level provider detection if paper summaries fail.
Step 1: Run:

```bash
arc-paper doctor host --json
arc-paper doctor provider --json
```

Expected plugin environment:

```text
Codex: ARC_AGENT_HOST=codex, ARC_LLM_PROVIDER=codex-cli
Claude Code: ARC_AGENT_HOST=claude-code, ARC_LLM_PROVIDER=claude-cli
```

## Direct Prompt Tests

Use direct `arc-llm` calls only for debugging or standalone LLM tasks.

Text output:

```bash
arc-llm run-text --prompt "Say hello" --provider auto
```

JSON output:

```bash
arc-llm run-json --prompt "Return {\"ok\": true}" --schema schema.json --provider auto --json
```

## Proposers-Reviewer Loops

Use the package loop for reusable LLM workflows where one or more proposers
produce outputs, one reviewer responds, and the exchange repeats for a
configured number of rounds.

Run from a JSON config:

```bash
arc-llm proposers-reviewer-loop --config loop-config.json --json
```

Validate a config without LLM calls:

```bash
arc-llm proposers-reviewer-loop --config loop-config.json --dry-run --json
```

The config must set `run_dir` directly. ARC writes artifacts under:

```text
<run_dir>/<run_id>/
```

For example, the idea workflow uses:

```json
{
  "run_dir": "<project-dir>/suggest-ideas",
  "run_id": "<run-id>",
  "artifact_options": {
    "save_prompts": true
  }
}
```

The loop runner owns all artifact writes. Worker prompts and outputs are stored
under per-loop and per-round directories, so distinct loops can run
concurrently without sharing mutable context.

`artifact_options.save_prompts` defaults to `true`. When enabled, full rendered
worker prompts are stored under each round's `prompts/` directory for
debugging. These prompt artifacts are not included in later worker context or
`transcript.jsonl`; worker context receives only proposer outputs, reviewer
reviews, controller messages, and reviewer-to-proposer messages. Worker-call
errors are written under each round's `errors/` directory.

Optional true-LLM integration tests are skipped by default. To run them
explicitly:

```bash
ARC_RUN_LLM_TESTS=1 ARC_RUN_NET_TESTS=1 \
  packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests/test_proposers_reviewer_llm_integration.py -q
```

Set `ARC_LLM_TEST_PROVIDER` or `ARC_LLM_TEST_MODEL` to override the provider or
model for that opt-in run.

## Model Tiers

Prefer `model_tier` for reusable workflows and package configs:

```text
low
medium
high
```

`arc-llm` maps these tiers to provider-specific model and reasoning defaults.
Exact model names are advanced overrides for project contexts that intentionally
pin a provider model.

## Runtime Options

By default ARC keeps provider calls lightweight. Enable extra capability only
when the task requires it.

Common options:

```text
--provider auto
--model-tier high
--model <model>
--allow-internet
--allow-mcp
--mcp-mode arc-only
--arc-mcp-command arc-mcp
--codex-reasoning-effort low
--codex-sandbox read-only
--codex-work-dir <project-dir>
--codex-add-dir <extra-dir>
--claude-effort low
```

Use `--allow-mcp` for LLM tasks that need ARC tools or other configured MCP
servers. Use `--allow-internet` only when fresh web access is required.

For proposers-reviewer JSON configs, prefer ARC-only MCP access when workers
need ARC paper/domain tools:

```json
{
  "runtime": {
    "allow_mcp": true,
    "mcp_mode": "arc-only",
    "codex_sandbox": "read-only"
  }
}
```

With Codex, `mcp_mode: "arc-only"` keeps the user host config ignored, injects
only the ARC MCP server, and approves that server's tools for the noninteractive
worker. If a worker also needs bounded filesystem access, use
`codex_sandbox: "workspace-write"` with `codex_work_dir` and `codex_add_dirs`;
do not use `danger-full-access` for normal research workflows.
