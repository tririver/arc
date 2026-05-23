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

Configured OpenAI-compatible providers:

```bash
arc-llm providers list
arc-llm providers doctor
```

Provider config defaults are checked in this order: `./llm-providers.json`,
then `~/.config/arc/llm-providers.json`. From this checkout, the project-local
default is `/arc-dev/llm-providers.json`. Override it with
`ARC_LLM_PROVIDER_CONFIG` or `--provider-config`. The file is only for URL-based
providers such as DeepSeek, Ollama, LM Studio, vLLM, and OpenRouter. Built-in
providers `codex-cli`, `claude-cli`, and `manual` are not configured there.

Provider config may store a raw `api_key` in a local ignored config file. Do
not commit real provider configs. The repository includes a redacted example at
`examples/llm-providers.example.json`; rename it to `llm-providers.json` and
put it in one of the default locations. Local files named `llm-providers.json`
are ignored by git. Store provider API keys in the local config file as
`api_key`; use `api_key_optional` only for local endpoints that do not require a
key. With `--provider auto`, ARC first uses configured providers with available
API keys, then optional local configured providers, then host providers.

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

Run reusable proposer/reviewer workflows from JSON:

```bash
arc-llm proposers-reviewer-loop --config loop-config.json --json
```

Validate without LLM calls:

```bash
arc-llm proposers-reviewer-loop --config loop-config.json --dry-run --json
```

## Proposers-Reviewer Benchmarks

Run many independent loop samples, ask an LLM to inspect artifact paths and
suggest prompt edits, then rerun candidates in an improve-and-measure loop:

```bash
arc-llm proposers-reviewer-bench --config bench-config.json --json
```

The input is the normal proposers-reviewer batch JSON plus an optional `bench`
object. Defaults include `samples: 25`, `max_rounds: 5`,
`max_iterations: 10`, `patience: 3`, `max_concurrent_loops: 100`, and
`default_provider: "deepseek"`.

Bench materialization asks each worker to add a top-level
`suggested_improvement` object in its output JSON. The prompt optimizer is told
to judge those worker suggestions alongside scores, transcripts, reviews, tool
traces, and the current prompt. It must not directly follow every suggestion.
For DeepSeek-style providers, `bench.improver_context_mode: "auto"` includes
expanded artifact excerpts by default, bounded by
`bench.improver_context_max_chars`. Use `"paths"` to send only file paths or
`"expanded"` to force inline artifact excerpts.

## Runtime Options

By default ARC keeps provider calls lightweight. Enable extra capability only
when the task requires it.

Common options:

```text
--provider auto
--provider-config <path>
--model <model>
--allow-internet
--allow-mcp
--codex-reasoning-effort low
--codex-sandbox read-only
--claude-effort low
```

Use `--allow-mcp` for LLM tasks that need ARC tools or other configured MCP
servers. Use `--allow-internet` only when fresh web access is required.
