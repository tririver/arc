# Arc LLM Package

`arc-llm` is the reusable host LLM worker used by ARC packages. Most workflows
should call `arc-paper`, `arc-domain`, or ARC MCP tools instead of calling
`arc-llm` directly. Use this reference for provider diagnosis, direct prompt
tests, and advanced LLM runtime options.

ARC tools are provided through the ARC MCP/plugin launcher and its bundled
runtime. Do not diagnose `arc-llm` by running `pip show arc-llm` in the system
Python. `arc_llm` is an internal Python module under ARC's bundled source/runtime.
If a workflow script cannot import `arc_llm`, it is using the wrong Python path/runtime;
use the ARC plugin launcher/runtime or source-tree `PYTHONPATH`/`ARC_MCP_REPO_ROOT`,
not `pip install arc-llm` from PyPI.

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
Codex: ARC_AGENT_HOST=codex
Claude Code: ARC_AGENT_HOST=claude-code
```

With `--provider auto`, ARC uses only host-native providers: Codex selects
`codex-cli`, Claude Code selects `claude-cli`, and unknown hosts select
`manual`. `arc-llm` does not read provider config files, API-key files, or
URL-based provider definitions.

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

`run-json` appends an `arc_llm_call_record` object to the returned JSON. This
records the requested provider/model tier, actual provider/model used,
fallback index, successful attempt number, host signal, and all failed/successful
attempts for that call. New records also include session policy, session key,
native session id when available, prompt/schema hashes, and provider usage
telemetry. Treat this as runtime audit data, not model-generated scientific
content.

ARC normalizes provider-facing JSON schemas to strict object schemas so they
stay compatible with Codex structured output. Do not ask workers to generate
`arc_llm_call_record`; ARC attaches that audit field after provider output.

Direct calls are stateless by default. For a debugging session that should
reuse host conversation state, pass all session fields explicitly:

```bash
arc-llm run-json \
  --prompt prompt.txt \
  --schema schema.json \
  --provider auto \
  --session-policy stateful \
  --session-root .arc-llm/sessions \
  --session-key debug/session_001 \
  --json
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
  "run_dir": "<project-dir>/ideas",
  "run_id": "<run-id>",
  "artifact_options": {
    "save_prompts": true
  }
}
```

The loop runner owns all artifact writes. Worker prompts and outputs are stored
under per-loop and per-round directories, so distinct loops can run
concurrently without sharing mutable context.

Idea workflow loop concurrency is bounded by `ARC_IDEAS_MAX_CONCURRENT_LOOPS`
and defaults to `12`.

The idea workflow runner writes only the generated batch config before launch:

```text
<project-dir>/ideas/<run-id>/ideas_batch_config.json
```

All concurrent proposer-reviewer artifacts are owned by `arc-llm` under the
batch run root. The workflow runner does not copy selected rounds or write a
project-level latest report while loops are running. Completed runner results
include `round_score_table`, a Markdown and structured per-loop table of
reviewer total scores by round, built from loop artifacts available at
completion time.

Proposers-reviewer configs default to stateful delta sessions. First worker
turns send the static task context and worker instructions; later turns send
only current deltas while reusing the same provider session. If a custom
`json_runner` does not accept session kwargs, the runner falls back to stateless
full prompts.
Custom `json_runner` wrappers must explicitly declare `session_policy`,
`session_manager`, `session_key`, `artifact_dir`, `call_label`, and
`static_prefix` to receive stateful session reuse. A bare `**kwargs` wrapper is
treated as legacy/stateless by design.

`artifact_options.save_prompts` defaults to `true`. When enabled, full rendered
worker prompts, or initial/delta prompt artifacts for stateful runs, are stored
under each round's `prompts/` directory for debugging. These prompt artifacts
are not copied into later worker context or `transcript.jsonl`; worker context
receives only proposer outputs, reviewer reviews, controller messages, and
reviewer-to-proposer messages. Worker-call errors are written under each
round's `errors/` directory.

Session config lives under the top-level `session` object. Use
`reuse_across_batch_calls: true` with a stable `scope_id` and `root` only when
separate batch run directories must reuse the same logical worker sessions, as
in calculation retries.

Audit prompt-cache behavior after a run:

```bash
arc-llm cache-audit <run-root>
```

Optional true-LLM integration tests are skipped by default. To run them
explicitly:

```bash
ARC_RUN_LLM_TESTS=1 ARC_RUN_NET_TESTS=1 \
  packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests/test_proposers_reviewer_llm_integration.py -q
```

Set `ARC_LLM_TEST_PROVIDER` to override the provider for that opt-in run.
`ARC_LLM_TEST_MODEL` is an exact-model override and requires an explicit
non-`auto` `ARC_LLM_TEST_PROVIDER`.

## Proposers-Reviewer Benchmarks

Use the benchmark wrapper to run many independent loop samples, ask an LLM to
inspect artifact paths and suggest prompt edits, then rerun candidates in an
improve-and-measure loop:

```bash
arc-llm proposers-reviewer-bench --config bench-config.json --json
```

The input is the normal proposers-reviewer batch JSON plus an optional `bench`
object. Defaults are `samples: 10`, `max_rounds: 5`, `max_iterations: 10`,
`patience: 3`, `max_concurrent_loops: 100`, and `default_provider: "auto"`.
The wrapper materializes sample loop IDs such as `idea_001` through `idea_010`
from the first configured loop template.
Benchmark sample workers default to `bench.sample_model_tier: "medium"` so
large batches use the provider's faster/cheaper test model when available
(`medium` tier). The prompt improver defaults to
`bench.improver_model_tier: "high"` so result analysis and prompt improvement
use the stronger provider model.

The improver is given score summaries and artifact file paths such as
`transcript.jsonl`; it should read detailed histories from disk instead of
receiving every correspondence inline. Automated edits are applied only to
explicit prompt-template targets, and reviewer prompt edits are disabled unless
`bench.allow_reviewer_prompt_edits` is true.

Bench materialization also asks each worker to add a top-level
`suggested_improvement` object in its output JSON. The prompt optimizer is told
to judge those worker suggestions alongside scores, transcripts, reviews, tool
traces, and the current prompt. It must not directly follow every suggestion.
Reusable prompt edits should transfer across theoretical-physics domains;
domain-specific technical advice belongs in reviewer-to-proposer feedback, not
global prompt templates.
`bench.improver_context_mode: "auto"` sends artifact paths only. Use
`"expanded"` to force inline artifact excerpts bounded by
`bench.improver_context_max_chars`.

## Model Tiers

Prefer `model_tier` for reusable workflows and package configs. Valid values:

```text
low
medium
high
```

`arc-llm` maps these tiers to provider-specific model and reasoning defaults.
Python API calls with no exact model or tier resolve to `medium`. Workflow
`context.json` files should write the explicit string `"medium"` so CLI and MCP
calls never pass an invalid `"auto"` tier.
`auto` is valid for `provider`, not for `model_tier`.
Exact model names are advanced overrides for project contexts that intentionally
pin a provider model. Exact `model` requires explicit `provider`; with
`provider: auto`, use `model_tier`.

## Runtime Options

By default ARC keeps provider calls lightweight. Enable extra capability only
when the task requires it.

Common auto-provider options:

```text
--provider auto
--model-tier high
```

Exact-model options:

```text
--provider <provider-id>
--model <model>
```

Runtime capability options:

```text
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
