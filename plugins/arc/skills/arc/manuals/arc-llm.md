# Arc LLM Package

`arc-llm` is the reusable host LLM worker used by ARC packages. Most workflows
should call `arc-paper` or `arc-domain` instead of calling
`arc-llm` directly. Use this reference for provider diagnosis, direct prompt
tests, and advanced LLM runtime options.

ARC tools are provided through the ARC Skill/CLI launcher and its isolated
runtime. Do not diagnose `arc-llm` by running `pip show arc-llm` in the system
Python. `arc_llm` is an internal Python module under ARC's bundled source/runtime.
If a workflow script cannot import `arc_llm`, it is using the wrong Python path/runtime;
use the ARC plugin launcher/runtime or source-tree `PYTHONPATH`/`ARC_INSTALL_REPO_ROOT`,
not `pip install arc-llm` from PyPI.

For a source-sensitive development run, set
`ARC_REQUIRE_REPO_ROOT=<checkout-root>`. ARC workflow scripts will then refuse
installed, marketplace-cached, or other-checkout ARC modules instead of using a
fallback runtime. Record the exact source state before the run with:

```bash
<skill-dir>/scripts/arc-runtime setup --profile core
python3 <skill-dir>/workflows/scripts/verify-source-runtime.py \
  --repo-root <checkout-root> \
  --output <project-dir>/source-provenance.json
```

The first command is the exact core-runtime setup required by the verifier.
When the verifier starts under system Python, it re-executes itself with that
installed runtime's Python before loading checkout sources and dependencies.

## Provider Diagnosis

### Phase 1: Check host detection.
Step 1: Run:

```bash
arc-llm doctor host --json
arc-llm doctor provider --json
arc-llm doctor config --json
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
Kimi Code: ARC_AGENT_HOST=kimi-code
```

With `--provider auto`, ARC uses only host-native providers: Codex selects
`codex-cli`, Claude Code selects `claude-cli`, Kimi Code selects the
experimental `kimi-code-cli`, and unknown hosts select `manual`. Kimi Code is
also detected from the `@moonshot-ai/kimi-code` package name or a reliable
`kimi` parent-process signal. An explicit `--provider kimi-code-cli` works from
another host. `arc-llm` does not add URL-based provider definitions or inspect
Kimi credential values.

The Kimi provider requires the Node.js/TypeScript Kimi Code CLI `>=0.28.0` and
an existing authentication created with `kimi login`. It uses `kimi acp` over
stdin/stdout JSON-RPC. ARC does not use the argv-based `kimi -p` transport and
does not add an OpenAI-compatible API provider.

`kimi-code-cli` is experimental. Before the first Kimi call in a process ARC
prints this warning:

> `kimi-code-cli is experimental and inherits Kimi Code configuration, instructions, skills, hooks, plugins, MCP, tool permissions, and persistent sessions; it may access the network, run commands, and modify files.`

ARC denies ACP permission and filesystem reverse requests, but this is not a
sandbox. Kimi's own automatic approvals, instructions, skills, hooks, plugins,
MCP servers, and local tools may still access the network, execute commands, or
modify files. `arc-llm doctor provider` and `arc-llm doctor config` report the
experimental status, provider-side persistence, missing native usage/schema
support, and detected risk categories and paths without printing configuration
values or credentials.

Codex and Claude internal workers disable inherited MCP, skills, plugins,
rules, and other host configuration by default. An ordinary worker receives
direct `arc-paper-worker` instructions only when the runtime reports
`nested_sandboxed_shell=true`; otherwise deterministic reads go through the
Controller evidence protocol. Claude and Kimi are not treated as proven
sandboxed shells. Generic `arc-llm` callers may opt in
to all host tools with `--inherit-host-tools`; this is a high-risk per-run
choice, is fingerprinted for audit/resume checks, and is not an ARC workflow
default. ARC records paths and risk categories, not credentials or complete
configuration. Kimi remains the documented exception because its host cannot
reliably suppress all inherited configuration.

## Direct Prompt Tests

Use direct `arc-llm` calls only for debugging or standalone LLM tasks.

Text output:

```bash
arc-llm run-text --prompt-text "Say hello" --provider auto
```

New ordinary calls enable paper CLI access by default. Use
`--no-arc-paper-cli` for an isolated call, or `--arc-paper-cli` to state the
default explicitly. Schema formatting, output recovery, provider preflight,
and blind-reference workers always disable it regardless of ordinary runtime
configuration. Checkpoints created before the access field existed resume as
`none` instead of silently gaining access.

For Codex `read-only` and `workspace-write`, ARC caches a short, harmless
`codex sandbox` probe with network disabled. A namespace denial disables only
the inner worker shell: it does not fail the model/provider call, switch
providers, enable `danger-full-access`, or affect the outer ARC CLI. Doctor
output reports only the stable capability classification and bounded receipt
metadata, never the probe command, environment, or raw output.

JSON output:

```bash
arc-llm run-json --prompt-text "Return {\"ok\": true}" --schema schema.json --provider auto --json
```

Use `--prompt-text` for literal text and `--prompt-file` for a UTF-8 file
(`--prompt-file -` reads stdin). The legacy `--prompt` option remains a
file/stdin alias for existing callers; it does not interpret its value as
inline text.

`run-json` appends an `arc_llm_call_record` v5 object to the returned JSON. This
records the requested provider/model tier, actual provider/model used,
fallback index, successful attempt number, host signal, and all failed/successful
attempts for that call. New records also include session policy, session key,
native session id when available, prompt/schema hashes, and provider usage
telemetry. Every record has a `warnings` array. Kimi records stable warnings
for its experimental status, inherited configuration, provider-side
persistence, and any model tier that was not actually mapped. Kimi ACP does not
provide token usage, so all fields in its usage object are null. Treat this as
runtime audit data, not model-generated scientific content.

ARC validates JSON Schemas locally before creating call artifacts. Closed
object schemas use a provider's native strict transport when supported. If a
schema contains an open object whose arbitrary-key semantics would be changed
by strict normalization, ARC instead appends the canonical original schema to
the prompt, applies relaxed JSON parsing, validates locally, and records
`structured_output.open_object_prompt_fallback` in the call warnings. Kimi has
no native schema mode and always uses its prompt contract without that fallback
warning. Do not ask workers to generate `arc_llm_call_record`; ARC attaches
that audit field after provider output.

For structured calls, ARC retains provider-ordered complete response material
until the call is validated. It schema-validates every complete JSON object,
distinguishes populated required result fields from valid empty placeholders,
and selects the last substantive equivalent result. A later empty
output-last-message therefore cannot erase an earlier completed result.
Malformed or truncated fragments remain in the existing relaxed-recovery path;
they are not treated as competing completed answers. If non-equivalent
substantive answers conflict, ARC stops for supervision unless the provider
protocol explicitly marks the later answer as superseding the earlier one.

Candidate decisions are written atomically beside the call checkpoint as
`*.candidate-selection.json`. The receipt contains only schema/material hashes,
candidate hashes and protocol positions, the decision, and selected or
conflicting hashes—not candidate bodies. The checkpoint retains the paid raw
response material needed for deterministic replay. Replaying either a selected
answer or a conflict makes no provider call, and a changed receipt fails
closed. The logical receipt links the selection receipt by relative path and
SHA-256.

When a caller supplies an artifact directory, each actual provider attempt gets
a unique directory below `attempts/`. Its finalize-once record retains bounded
sanitized stdout, provider events, stderr, preliminary parsed-response
candidates, native session metadata, failure classification, and the process
lifecycle timeline (submission, cancellation, stdin closure, TERM/KILL, and
process-group outcome). Large streams are gzip-compressed when compression is
beneficial; over-limit streams retain useful head and tail data plus explicit
truncation metadata. Provider
retries never overwrite an earlier attempt. The call record contains only the
relative attempt-record path and SHA-256, not the potentially large streams.
Credential-shaped fields, secret/config environment values, authorization
headers, and provider keys are redacted before persistence. These artifacts are
diagnostic evidence, not a credential store; ARC never serializes provider
configuration values intentionally.

A checkpoint replay creates an audit-only attempt record with outcome/status
`replayed` and submission state `not_submitted`; it does not claim that ARC
invoked or charged the provider again. If diagnostic finalization fails after a
provider has already returned, ARC preserves the provider result and records a
stable warning instead of retrying it.

Concurrent structured batches use the first real task as a schema canary for
each provider/runtime/model/schema/transport identity. Calls sharing an
unproven identity wait before provider admission. Success writes an atomic
proof below the caller-owned run root and restores normal concurrency; a
deterministic schema-contract rejection records one failure and blocks the
remaining calls without submitting them. Request-specific invalid requests
and retryable transport failures do not prove or reject the identity. Receipts
are batch-local rather
than stored beside provider configuration or credentials.

Direct calls are stateless by default. For a debugging session that should
reuse host conversation state, pass all session fields explicitly:

```bash
arc-llm run-json \
  --prompt-file prompt.txt \
  --schema schema.json \
  --provider auto \
  --session-policy stateful \
  --session-root .arc-llm/sessions \
  --session-key debug/session_001 \
  --json
```

For Kimi, ARC `stateless` means ARC does not reuse a native session ID. Kimi
Code still creates and persists its own session in its data directory. ARC
does not copy, migrate, or delete the user's Kimi configuration, credentials,
or sessions. A new session uses `ARC_KIMI_WORK_DIR`, or the current working
directory when unset; a stateful resume keeps the cwd stored in the native Kimi
session.

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

Workers without reliable direct shell access may return the optional top-level
`arc_evidence_requests` array. Each
request contains a loop-round-unique `request_id`, an `operation`, JSON
`arguments`, and a `reason`. The controller resolves requests outside the
worker process, records response data and provenance in `transcript.jsonl`,
and injects only the addressed worker's exchanges into its next turn. Empty
arrays are no-ops, malformed requests fail the loop, and resolver calls stop
after three evidence rounds. A request in the final configured worker round is
recorded with `no_followup_round` instead of being resolved, because no later
turn could consume it. Workers with a proven nested shell may call only the
bounded paper worker; other workers never call ARC CLI, shell, or MCP tools.

The ideas workflow installs an `arc-paper` service resolver by default. Its
portable operations are `paper.metadata`, `paper.section`,
`paper.full_text_search`, `paper.references`, `paper.citers`, and
`paper.search`. Python hosts may instead pass an `evidence_controller`
callback to `run_proposers_reviewer_batch()` or `run_ideas()`; the callback
receives typed requests and must return one typed response per request with
matching IDs and provenance. This controller route has the same operation
semantics as `arc-paper-worker`; it is the portable evidence fallback rather
than a weaker evidence class.

Evidence capability is explicit and hierarchical. Batch `evidence.enabled`
is the master switch; a loop or worker may further disable it with
`"evidence": {"enabled": false}` but cannot re-enable a disabled parent.
Disabled workers do not receive the evidence schema or prompt protocol, and
their output can never reach the controller resolver. The ideas workflow uses
this boundary to keep no-information variants isolated from ARC paper caches.

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
  packages/arc-llm/tests/test_cli_smoke_integration.py \
  packages/arc-llm/tests/test_proposers_reviewer_llm_integration.py -q
```

The CLI smoke covers stateful structured output and a live Codex evidence
schema response with `arc_evidence_requests: []`; a successful run confirms
the strict schema is accepted instead of returning HTTP 400.

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
max
```

`arc-llm` maps these tiers to provider-specific model and reasoning defaults.
Never select the `max` model tier automatically. Use it only when the user
explicitly requests the `max` model tier; no workflow default or automatic task
mapping may select it.
Python API calls with no exact model or tier resolve to `medium`. Workflow
`context.json` files should write the explicit string `"medium"` so CLI and MCP
calls never pass an invalid `"auto"` tier.
`auto` is valid for `provider`, not for `model_tier`.
Exact model names are advanced overrides for project contexts that intentionally
pin a provider model. Exact `model` requires explicit `provider`; with
`provider: auto`, use `model_tier`.

For Kimi, set `ARC_LLM_KIMI_LOW_MODEL`,
`ARC_LLM_KIMI_MEDIUM_MODEL`, `ARC_LLM_KIMI_HIGH_MODEL`, or
`ARC_LLM_KIMI_MAX_MODEL` to map a tier to a Kimi model alias. Without a mapping,
Kimi uses its `default_model` alias and the call record warns that the requested
tier was not actually realized. ARC does not treat a `highspeed` model as a
higher-quality tier and does not inject `KIMI_MODEL_*` API credentials.

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
--idle-timeout-seconds <positive-seconds>
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

`--idle-timeout-seconds` limits only continuous time without substantive
provider activity. ARC defaults to 1800 seconds and imposes no absolute total
runtime limit. New visible assistant content, a changed tool state, or a
confirmed artifact write resets the idle timer. Handshakes, hidden reasoning,
stderr noise, empty/repeated content, and `still alive` heartbeats do not.

Provider idle-timeout environment variables:

```text
ARC_CODEX_IDLE_TIMEOUT_SECONDS       Codex idle timeout
ARC_CLAUDE_IDLE_TIMEOUT_SECONDS      Claude idle timeout
ARC_KIMI_IDLE_TIMEOUT_SECONDS        Kimi idle timeout
ARC_LLM_IDLE_TIMEOUT_SECONDS         General fallback for every provider
```

An explicit CLI or worker idle timeout takes precedence, followed by the
provider-specific variable, `ARC_LLM_IDLE_TIMEOUT_SECONDS`, and finally 1800
seconds. Removed `--timeout-seconds`, `worker_call_timeout_seconds`, and
`ARC_LLM/CODEX/CLAUDE/KIMI_TIMEOUT_SECONDS` settings fail before a model call
with a migration message instead of silently retaining total-deadline behavior.
ARC also validates non-numeric and non-positive idle-timeout values before
creating a call checkpoint or acquiring a provider concurrency slot. These are
typed `not_submitted` configuration failures, so they cannot quarantine a
checkpoint or be mistaken for a paid request.

Every public ARC-LLM entry point adds the versioned runtime progress
contract once. Long, multi-stage workers must report concrete completed work,
evidence or results, reusable artifact paths, the next step, and blockers while
they work. They must not emit private chain-of-thought, secrets, full tool
arguments, repeated plans, or content-free `still alive` messages. Progress is
sent out of band as `arc.llm.progress.v1`, so it never contaminates structured
JSON. Short single-stage calls need not manufacture progress.

ARC persists the progress journal and available session/checkpoint context.
After idle timeout or explicit cancellation, the owning workflow may resume
from verified artifacts or a native provider session rather than repeating
completed work. Idle timeout never automatically starts a second paid call.

Kimi runtime environment variables:

```text
ARC_KIMI_BIN                         Kimi executable; default: kimi
ARC_KIMI_WORK_DIR                    Working directory for new sessions; default: current directory
ARC_KIMI_IDLE_TIMEOUT_SECONDS        Idle timeout; falls back to ARC_LLM_IDLE_TIMEOUT_SECONDS
ARC_KIMI_ALLOW_INTERNAL_RETRIES      Explicitly accept unsafe provider-internal retries
ARC_LLM_KIMI_LOW_MODEL               Low-tier model alias
ARC_LLM_KIMI_MEDIUM_MODEL            Medium-tier model alias
ARC_LLM_KIMI_HIGH_MODEL              High-tier model alias
ARC_LLM_KIMI_MAX_MODEL               Max-tier model alias
```

ARC starts the Kimi subprocess with `KIMI_CODE_NO_AUTO_UPDATE=1`,
`KIMI_DISABLE_TELEMETRY=1`, and `KIMI_DISABLE_CRON=1`. It preserves the user's
existing `KIMI_CODE_HOME` and login state.

Before starting Kimi, ARC requires either a supported per-process retry
override or `[loop_control] max_retries_per_step = 1` in the effective Kimi
configuration. Missing, unreadable, or larger values fail before an agent
session starts. `ARC_KIMI_ALLOW_INTERNAL_RETRIES=1` is an explicit cost-risk
escape hatch and is always reported by doctor and call diagnostics.

ARC_HOME-wide provider concurrency is capped at 24 real model calls. Local
worker counts may lower but never multiply this cap. Quota and authentication
failures open a persistent provider circuit; rate limits cool down for at least
15 minutes and then admit one half-open probe. Inspect or reset these states
with `arc-llm circuit status` and `arc-llm circuit reset`.

`--allow-mcp` is an explicit advanced opt-in for standalone LLM tasks using
caller-configured servers. ARC workflow workers must leave it disabled and use
controller-supplied evidence. Use `--allow-internet` only when fresh web access
is required.

For proposers-reviewer JSON configs, keep MCP disabled and place resolved ARC
paper/domain evidence in caller context:

```json
{
  "runtime": {
    "allow_mcp": false,
    "codex_sandbox": "read-only"
  }
}
```

For Codex and Claude, disabling MCP scrubs inherited user/profile MCP
configuration for the noninteractive worker. Kimi may still inherit its own
MCP, hooks, plugins, and tool configuration; ACP reverse-request denial is not
a filesystem or process sandbox. If a Codex worker also needs bounded
filesystem access, use `codex_sandbox: "workspace-write"` with
`codex_work_dir` and `codex_add_dirs`; do not use `danger-full-access` for
normal research workflows.
