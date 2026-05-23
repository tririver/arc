# Proposers-Reviewer Loop PRD

Date: 2026-05-23
Status: Draft for review
Owner: ARC development

## Summary

ARC needs a reusable LLM orchestration primitive for workflows where one or
more proposers produce independent outputs, one reviewer inspects those outputs,
and the process repeats for a configurable number of rounds. The first consumer
will be `skills/arc/references/research-workflows/suggest-ideas.md`, where ARC
will run multiple independent idea-generation loops concurrently. Future
consumers include calculation workflows where two proposers calculate the same
quantity independently and a reviewer checks agreement.

The implementation belongs in `packages/arc-llm` because the loop is generic
host-LLM orchestration. Scientific meaning, idea quality criteria, calculation
semantics, scoring rubrics, and final selection are caller-owned and must be
supplied through JSON configuration, prompts, and schemas.

## Problem

The reference ARC idea workflow in `0_ref/skills/arc` mixes several concerns in
one skill-layer procedure:

- generator and reviewer prompts;
- lane planning and cohort policy;
- revision policy;
- artifact validation;
- host delegation details;
- batch aggregation and ledgers;
- scientific acceptance/rejection semantics.

This makes the loop hard to test, hard to reuse, and fragile under concurrent
runs. The new ARC package architecture needs a clean package-level primitive
that can run many loops safely at the same time without cross-contaminating
prompt context, runtime permissions, or output files.

## Goals

- Provide a reusable `arc-llm` proposers-reviewer loop for idea generation,
  calculation comparison, and later ARC workflows.
- Accept a JSON configuration file rather than complex CLI strings.
- Let callers configure the run directory in JSON so different ARC workflows
  can keep independent shallow run roots.
- Support multiple proposers per loop.
- Support exactly one reviewer per loop in v1 while preserving a config shape
  that can later support multiple reviewers.
- Make the maximum number of rounds configurable.
- Allow reviewer-requested early stop when enabled by config.
- Preserve every round, prompt, proposer output, review, controller message,
  and proposer-addressed reviewer message.
- Include all past correspondences in proposer context by default.
- Run many independent loops concurrently in one invocation.
- Be thread-safe and process-safe for distinct loops in the same run.
- Keep `arc-llm` unaware of caller-specific semantics such as idea quality,
  novelty, best-round selection, calculation correctness, or acceptance.
- Keep the skill layer thin: `suggest-ideas.md` configures and invokes the
  package behavior, not reimplement the loop.

## Non-Goals

- Do not implement scientific idea quality logic in `arc-llm`.
- Do not implement calculation comparison logic in `arc-llm`.
- Do not make `arc-llm` choose the best idea, best round, accepted output, or
  final report content.
- Do not implement multiple-reviewer aggregation in v1.
- Do not require subagents or a specific coding-agent host.
- Do not require network access or MCP access for unit tests.
- Do not encode field-specific keywords, paper IDs, author names, or subfield
  labels in the generic loop.

## Users And Use Cases

### Idea Generation

The ARC skill builds domain context, then launches multiple independent
idea-review loops. Each loop runs a proposer and reviewer for five rounds by
default. The reviewer gives marks and improvement comments but does not accept
or reject. The loop preserves all correspondences so later ARC reporting can
choose a strong idea or compare rounds.

### Calculation Comparison

A later calculation workflow launches two proposers that independently compute
the same object. The reviewer compares their outputs. If the calculations agree
and the reviewer is satisfied, the reviewer can request early stop. Otherwise,
the reviewer sends independent messages to each proposer for another round.

### Prompt Experiments

ARC development can launch many loops with different information boundaries:
internet-only, domain-summary-assisted, ARC-paper-tool-assisted, or other
caller-defined prompt variants. The package must preserve artifacts so batch
experiments can compare outcomes after the run.

## Package Boundary

`packages/arc-llm` owns:

- JSON config parsing and validation;
- host/provider/model selection through existing `arc_llm.runner` facilities;
- concurrent loop execution;
- concurrent proposer execution within a loop;
- reviewer execution after proposer outputs exist;
- prompt materialization;
- immutable artifact writes;
- lock acquisition and release;
- loop state and transcript recording;
- CLI entry point.

Callers own:

- prompt text;
- prompt templates and variables;
- output schemas for proposers and reviewers;
- review criteria and marks;
- meaning of `review_payload`;
- whether early stop is allowed;
- how to choose the best round or final output;
- any final reporting or ledgers;
- any domain, paper, idea, or calculation-specific instructions.

## Public API

### Python API

The primary API is a package function:

```python
from arc_llm.proposers_reviewer import run_proposers_reviewer_batch

result = run_proposers_reviewer_batch(config)
```

The function returns a JSON-serializable result describing run status, loop
statuses, artifact paths, and failures. It must not return large prompt or
output bodies when those are already written as artifacts.

### CLI

Add a subcommand:

```bash
arc-llm proposers-reviewer-loop --config <config.json>
```

Optional CLI flags:

```text
--json
--dry-run
--max-concurrent-loops <n>
```

The config file remains authoritative. CLI options may override operational
settings only when they do not change prompt semantics.

## Configuration Model

The config is one run containing one or more independent loops.

```json
{
  "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
  "run_id": "idea-test-2026-05-23",
  "run_dir": "project/suggest-ideas",
  "max_concurrent_loops": 2,
  "defaults": {
    "provider": "auto",
    "model": "gpt-5.5",
    "runtime": {
      "allow_internet": false,
      "allow_mcp": false,
      "codex_reasoning_effort": "xhigh"
    }
  },
  "loops": [
    {
      "loop_id": "idea_001",
      "max_rounds": 5,
      "early_stop": {
        "enabled": false
      },
      "proposers": [
        {
          "id": "proposer_001",
          "prompt": {
            "system": "You are a theoretical physicist...",
            "template": "Use the full correspondence below..."
          },
          "output_schema": {
            "type": "object"
          },
          "runtime": {
            "allow_internet": true,
            "allow_mcp": false
          }
        }
      ],
      "reviewers": [
        {
          "id": "reviewer_001",
          "prompt": {
            "system": "You are a skeptical reviewer...",
            "template": "Review the proposer outputs..."
          },
          "output_schema": {
            "type": "object"
          },
          "runtime": {
            "allow_internet": true,
            "allow_mcp": true
          }
        }
      ],
      "caller_context": {
        "user_intent": "use scattering amplitude technique to calculate correlation functions in cosmological collider physics"
      }
    }
  ]
}
```

### Validation Rules

- `schema_version` must be recognized.
- `run_dir` is required and points to the configured workflow run directory
  that contains `<run_id>` subdirectories.
- `run_id` must be filesystem-safe after normalization.
- `loop_id` values must be unique within the run.
- v1 requires exactly one reviewer in `reviewers`.
- `max_rounds` must be a positive integer.
- `max_concurrent_loops` must be a positive integer.
- proposer IDs must be unique within a loop.
- reviewer IDs must be unique within a loop.
- prompt definitions must be explicit; the package must not invent scientific
  prompts.
- output schemas must be JSON objects when provided.
- runtime permissions are per worker after default merging.

## Reviewer Envelope

The reviewer output must be a structured object with a controller-facing part,
proposer-addressed messages, and caller-defined payload.

```json
{
  "schema_version": "arc.llm.review_envelope.v1",
  "controller": {
    "message": "Private message to the controller.",
    "stop_requested": false,
    "stop_reason": ""
  },
  "proposer_messages": {
    "proposer_001": {
      "message": "Message addressed to proposer_001."
    },
    "proposer_002": {
      "message": "Message addressed to proposer_002."
    }
  },
  "review_payload": {
    "caller_defined": true
  }
}
```

`arc-llm` validates only the envelope fields needed for orchestration. It does
not validate the scientific content of `review_payload` unless the caller also
provides a JSON schema for the full reviewer output.

If `controller.stop_requested=true` and `early_stop.enabled=true`, the loop
stops after writing the review artifact for that round. If early stop is not
enabled, the stop request is recorded but ignored for control flow.

## Prompt Context

For every proposer round, the default context must include the full prior
correspondence for that same loop:

- all previous proposer prompts;
- all previous proposer outputs;
- all previous reviewer prompts;
- all previous reviewer outputs;
- all controller messages;
- all proposer-addressed reviewer messages;
- current loop metadata;
- current round number;
- caller context for the loop.

This context never includes artifacts from other loops unless the caller
explicitly copied that information into the loop's `caller_context`. Concurrent
idea loops are independent by default.

The correspondence must be serialized deterministically and materialized into
the prompt artifact before the LLM call. The package will expose template
variables, but the caller controls the final prompt text.

The current round's proposer prompts must not include other proposers'
current-round outputs, because those outputs do not exist yet and because
current-round proposer independence is important for calculation comparison.
The reviewer prompt includes all current-round proposer outputs.

## Directory Structure

Use one caller-configured run directory with isolated run and loop
subdirectories. The `run_dir` field points directly at the configured workflow
directory that contains `<run_id>` directories; `arc-llm` must not append an
intermediate hard-coded `runs/` component.

```text
<run_dir>/
  <run_id>/
    config.json
    manifest.json
    state.json
    run.lock
    loops/
      <loop_id>/
        lock.json
        loop_config.json
        state.json
        transcript.jsonl
        rounds/
          round_001/
            context/
              proposer_001.json
              proposer_002.json
              reviewer_001.json
            prompts/
              proposer_001.md
              proposer_002.md
              reviewer_001.md
            proposer_outputs/
              proposer_001.json
              proposer_002.json
            reviews/
              reviewer_001.json
          round_002/
            ...
```

Example workflow-specific run directories:

```json
{ "run_dir": "project/suggest-ideas" }
```

```json
{ "run_dir": "project/calculate" }
```

`manifest.json` records the run-level plan and loop paths. `state.json` records
status only. Large correspondence content belongs in per-round artifacts and
`transcript.jsonl`.

## Thread And Process Safety Requirements

Concurrency safety is mandatory because ARC will run multiple independent
loops at the same time.

### Isolation

- Every loop writes only under its own `loops/<loop_id>/` directory.
- Every round writes only under its own `rounds/round_NNN/` directory.
- Every worker prompt, context, and output path includes worker ID and round ID.
- Worker runtime environment is built as a new dict per call.
- Config is deep-copied or frozen before dispatch to loop workers.
- No loop may mutate shared config, global state, `os.environ`, or current
  working directory.
- A loop may read run-level config, but it must not write run-level state while
  other loops are active.

### Locking

- The runner acquires run-level ownership for manifest/config/state creation by
  creating `run.lock` with exclusive creation semantics.
- `run.lock` protects only run-level metadata. It is not held while long LLM
  calls are running.
- The runner acquires loop ownership by creating `lock.json` with exclusive
  creation semantics.
- If `lock.json` or the loop directory already exists for a loop ID that this
  invocation wants to run, the runner fails that loop clearly instead of
  interleaving writes.
- The lock contains `run_id`, `loop_id`, process ID, thread ID, host name when
  available, and creation time.
- Locks are released or marked complete only by the owning controller.

### Writes

- All JSON and prompt writes use atomic replace in the target directory:
  write to a unique temporary sibling file, fsync when practical, then replace.
- `transcript.jsonl` append operations are performed only by the loop-owning
  controller thread, not by proposer or reviewer worker threads.
- Run-level state is updated only by the top-level coordinator while holding
  `run.lock`, or is derived by scanning loop state files.
- Workers never write output files directly. They return data to the
  controller, which writes the artifact.

### Executors

- The top-level coordinator may use a thread pool for independent loops.
- Each loop may use a proposer executor for current-round proposers.
- The reviewer for a round starts only after all current-round proposer futures
  complete successfully.
- A failed proposer marks that round and loop as failed unless future config
  explicitly introduces retries.
- Nested concurrency must not share mutable prompt-building buffers.

## Error Handling

- Config validation errors stop before any LLM calls.
- Duplicate loop IDs stop before any LLM calls.
- Existing active lock for a loop marks that loop as `blocked` or `failed` with
  a clear lock-conflict reason.
- LLM provider failures are recorded in the loop state and run result.
- Invalid JSON output is recorded as a worker failure.
- Reviewer output missing the required envelope is recorded as a review failure.
- If one loop fails, other independent loops may continue unless
  `fail_fast=true` is added and enabled in config.
- The final CLI result must identify succeeded, failed, stopped, and skipped
  loops.

## Resume Policy

V1 must not silently resume partial loops. A run invocation with an existing
`run_id` and loop artifact directory must require explicit config policy:

```json
{
  "existing_run_policy": "fail"
}
```

Allowed v1 values:

- `fail`: fail if the run directory already exists.
- `append_new_loops`: allow only loop IDs that do not already exist.

With `append_new_loops`, the runner must acquire `run.lock` before updating
run-level metadata and still acquire each loop's exclusive `lock.json` before
executing that loop.

Round-level resume can be added later after the artifact contract is stable.

## Suggest-Ideas Integration

`skills/arc/references/research-workflows/suggest-ideas.md` becomes a
thin orchestration workflow:

1. Read `<project-dir>/context.json`.
2. Ensure `build-domain.md` has produced domain artifacts.
3. Build an `arc-llm` loop config with
   `run_dir=<project-dir>/suggest-ideas`.
4. Launch two independent idea loops for the initial test.
5. Set `max_concurrent_loops=2`.
6. Set `max_rounds=5`.
7. Set `early_stop.enabled=false`.
8. Give the idea proposer the user intent and internet permission.
9. Optionally give proposers domain summaries and ARC paper-tool access notes.
10. Give the reviewer full available permissions.
11. Use strong configured models for proposer and reviewer.
12. Preserve all loop artifacts.
13. Report artifact paths and state that final idea selection is based on the
    recorded loop artifacts, not memory.

The workflow may later vary loop configs to test whether domain summaries,
paper tools, internet access, or other context improves idea quality.

## Security And Permission Model

Runtime permissions are explicit per worker and are merged from defaults.

Recommended runtime fields:

```json
{
  "allow_internet": true,
  "allow_mcp": false,
  "provider": "auto",
  "model": "gpt-5.5",
  "codex_reasoning_effort": "xhigh",
  "codex_sandbox": "read-only",
  "claude_effort": "high"
}
```

The package maps these runtime fields to provider environment variables already
used by `arc_llm.cli` and `arc_llm.runner`. Provider-specific options remain
optional and must not be required for non-Codex hosts.

## Testing Requirements

Unit tests must not call real LLM providers or require network access.

Required test areas:

- config validation accepts valid v1 configs;
- config validation rejects multiple reviewers in v1;
- config validation rejects duplicate loop IDs and duplicate proposer IDs;
- prompt context for round 2 includes all round 1 correspondence;
- current-round proposer prompts do not include other current-round proposer
  outputs;
- reviewer prompt includes all current-round proposer outputs;
- early stop is honored only when enabled;
- early stop request is recorded but ignored when disabled;
- two loops run concurrently without writing outside their loop directories;
- duplicate loop locks fail clearly;
- atomic writes leave complete JSON files;
- per-worker runtime envs do not mutate `os.environ`;
- failed loop does not corrupt successful loop artifacts;
- CLI reads config and returns structured JSON.

Use fake LLM providers in tests to produce deterministic proposer and reviewer
outputs. Add stress-style unit tests with multiple threads writing many small
loop artifacts through the package writer.

## Acceptance Criteria

- `arc-llm proposers-reviewer-loop --config <config.json> --json` runs a fake
  provider test config with two loops concurrently and produces isolated loop
  directories.
- The runner writes under the configured `run_dir/<run_id>/` path without
  appending an extra implicit `runs/` component.
- Each loop records all prompts, contexts, outputs, reviews, state, and
  transcript entries for every completed round.
- `max_rounds` controls the maximum number of rounds.
- Reviewer early stop works only when enabled.
- v1 validates exactly one reviewer.
- Proposer context includes full past correspondence by default.
- No shared mutable prompt context can leak between loops.
- Concurrent loops with distinct IDs complete without lock conflicts.
- Concurrent attempts to run the same loop ID fail cleanly.
- Unit tests pass for `packages/arc-llm`.
- The combined ARC suite remains practical to run after package changes.

## Open Extension Points

- Multiple reviewers with aggregation.
- Round-level resume.
- Retry policy for failed LLM calls.
- Rich template language beyond simple structured context injection.
- Optional transcript compression for long runs.
- Caller-owned final-selection helpers built outside `arc-llm`.
