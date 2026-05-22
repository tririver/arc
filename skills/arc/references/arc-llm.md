# Arc LLM Package

`arc-llm` is the reusable host LLM worker used by ARC packages. Most workflows
should call `arc-paper`, `arc-domain`, or ARC MCP tools instead of calling
`arc-llm` directly. Use this reference for provider diagnosis, direct prompt
tests, and advanced LLM runtime options.

## Provider Diagnosis

Phase 1: Check host detection.
Step 1: Run:

```bash
arc-llm doctor host
arc-llm doctor provider
arc-llm doctor config
```

Phase 2: Check package-level provider detection if paper summaries fail.
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

## Runtime Options

By default ARC keeps provider calls lightweight. Enable extra capability only
when the task requires it.

Common options:

```text
--provider auto
--model <model>
--allow-internet
--allow-mcp
--codex-reasoning-effort low
--codex-sandbox read-only
--claude-effort low
```

Use `--allow-mcp` for LLM tasks that need ARC tools or other configured MCP
servers. Use `--allow-internet` only when fresh web access is required.
