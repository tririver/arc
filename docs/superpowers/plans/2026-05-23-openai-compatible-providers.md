# OpenAI-Compatible Providers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable OpenAI-compatible HTTP providers, including DeepSeek and local Ollama-style endpoints, without changing the existing `codex-cli` and `claude-cli` provider API.

**Architecture:** Keep `codex-cli`, `claude-cli`, and `manual` as built-in providers. Add a provider-config loader for URL-based providers and a single `OpenAICompatibleProvider` that uses `base_url` plus an API key stored in the local ignored provider config file. `auto` first selects a usable configured provider, then falls back to the current host CLI behavior.

**Tech Stack:** Python 3.11, `openai` Python SDK, existing `arc_llm` provider protocol, `argparse`, `pytest`.

---

### Task 1: Provider Config And Selection

**Files:**
- Create: `packages/arc-llm/src/arc_llm/providers/config.py`
- Modify: `packages/arc-llm/src/arc_llm/host.py`
- Modify: `packages/arc-llm/src/arc_llm/model.py`
- Modify: `packages/arc-llm/src/arc_llm/providers/select.py`
- Test: `packages/arc-llm/tests/test_openai_compatible_config.py`

- [ ] Write failing tests for config path resolution, usable provider filtering, explicit configured provider selection, and `auto` selecting configured providers before host CLI fallback.
- [ ] Implement `ConfiguredProvider` dataclass and `ProviderConfigError`.
- [ ] Load JSON from `ARC_LLM_PROVIDER_CONFIG`, project-local `./llm-providers.json`, or `~/.config/arc/llm-providers.json`.
- [ ] Allow local ignored provider configs to store raw `api_key` values, with `api_key_optional` support for local endpoints that do not require a key.
- [ ] Update `select_llm_provider()` so explicit provider and `ARC_LLM_PROVIDER` still win, while `auto` checks usable configured providers before host CLI fallback.
- [ ] Update `resolve_model()` so configured providers resolve explicit model, provider `models.default`, tier model, and then `ARC_LLM_MODEL`.

### Task 2: OpenAI-Compatible Provider

**Files:**
- Create: `packages/arc-llm/src/arc_llm/providers/openai_compatible.py`
- Modify: `packages/arc-llm/src/arc_llm/providers/select.py`
- Modify: `packages/arc-llm/pyproject.toml`
- Test: `packages/arc-llm/tests/test_openai_compatible_provider.py`

- [ ] Write failing tests using a fake OpenAI client factory.
- [ ] Add `openai>=1.0` as an `arc-llm` dependency.
- [ ] Implement text calls through `client.chat.completions.create(...)`.
- [ ] Implement JSON calls with `response_format={"type": "json_schema", ...}` by default and parse JSON from the first choice.
- [ ] Support `json_mode` values `json_schema`, `json_object`, and `none`.
- [ ] Raise `LLMWorkerError` with sanitized messages and never include API key values.

### Task 3: CLI And Summary Integration

**Files:**
- Modify: `packages/arc-llm/src/arc_llm/cli.py`
- Modify: `packages/arc-paper/src/arc_paper/summary/providers/select.py`
- Create: `packages/arc-paper/src/arc_paper/summary/providers/prompt.py`
- Test: `packages/arc-llm/tests/test_cli.py`
- Test: `packages/arc-paper/tests/test_summary_provider_selection.py`

- [ ] Write failing tests for `arc-llm providers list`, `providers doctor`, and runtime `--provider-config`.
- [ ] Add CLI provider-config env plumbing.
- [ ] Add paper-summary adapter that wraps any `arc_llm` prompt provider so configured providers can generate paper summaries.
- [ ] Update tests to prove `select_summary_provider("deepseek")` returns the generic adapter when `deepseek` exists in provider config.

### Task 4: Docs, Ignore Rules, Verification

**Files:**
- Modify: `.gitignore`
- Modify: `skills/arc/references/package-manuals/arc-llm.md`
- Modify: `packaging/codex/arc/skills/arc/references/package-manuals/arc-llm.md`
- Modify: `packaging/claude/arc/skills/arc/references/package-manuals/arc-llm.md`
- Modify: `README.md`

- [ ] Ignore local env files and local provider config files.
- [ ] Document project-local `./llm-providers.json`, `~/.config/arc/llm-providers.json`, `ARC_LLM_PROVIDER_CONFIG`, raw `api_key`, and examples for DeepSeek and Ollama.
- [ ] Run `packages/arc-paper/.venv/bin/python -m pytest packages/arc-llm/tests -q`.
- [ ] Run `packages/arc-paper/.venv/bin/python -m pytest packages/arc-paper/tests/test_summary_provider_selection.py packages/arc-paper/tests/test_host_cli_providers.py -q`.
- [ ] Run the combined local suite when practical.
