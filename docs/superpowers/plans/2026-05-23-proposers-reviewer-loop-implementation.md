# Proposers Reviewer Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement the PRD-defined reusable concurrent proposers-reviewer loop in `arc-llm` and wire `suggest-ideas.md` to call it.

**Architecture:** Add a small `arc_llm.proposers_reviewer` package split into config validation, artifact/lock helpers, runtime environment merging, prompt/context rendering, and orchestration. The runner owns all file writes, uses `run_dir/<run_id>/loops/<loop_id>/...`, runs loops concurrently, runs same-round proposers concurrently, and calls the existing provider selection path through `arc_llm.runner.run_json`.

**Tech Stack:** Python 3.11 standard library, existing `arc_llm.runner` provider abstraction, `argparse`, `pytest`, JSON artifacts with atomic replace.

---

## File Structure

- Create `packages/arc-llm/src/arc_llm/proposers_reviewer/__init__.py`: public API exports.
- Create `packages/arc-llm/src/arc_llm/proposers_reviewer/config.py`: config dataclasses, validation, ID safety, default/runtime merging.
- Create `packages/arc-llm/src/arc_llm/proposers_reviewer/artifacts.py`: atomic JSON/text writes, exclusive locks, path builder.
- Create `packages/arc-llm/src/arc_llm/proposers_reviewer/prompts.py`: deterministic correspondence/context building and simple prompt rendering.
- Create `packages/arc-llm/src/arc_llm/proposers_reviewer/runner.py`: batch/loop/round orchestration.
- Modify `packages/arc-llm/src/arc_llm/cli.py`: add `proposers-reviewer-loop --config --json --dry-run --max-concurrent-loops`.
- Modify `packages/arc-llm/src/arc_llm/__init__.py`: export batch runner.
- Add `packages/arc-llm/tests/test_proposers_reviewer_config.py`.
- Add `packages/arc-llm/tests/test_proposers_reviewer_artifacts.py`.
- Add `packages/arc-llm/tests/test_proposers_reviewer_runner.py`.
- Add `packages/arc-llm/tests/test_proposers_reviewer_cli.py`.
- Modify `skills/arc/references/research-workflows/suggest-ideas.md`.

## Task 1: Config And Runtime Model

- [x] Write failing tests in `packages/arc-llm/tests/test_proposers_reviewer_config.py`:
  - valid config parses with `run_dir`, `run_id`, one loop, one reviewer, one proposer.
  - duplicate loop IDs fail.
  - multiple reviewers fail in v1.
  - duplicate proposer IDs fail.
  - runtime defaults merge into worker runtime without mutating input.
  - runtime maps to provider env keys for internet, MCP, model, and Codex/Claude effort fields.
- [x] Run:
  `packages/arc-paper/.venv/bin/python -m pytest packages/arc-llm/tests/test_proposers_reviewer_config.py -q`
  Expected: import/module failures before implementation.
- [x] Implement `config.py` with dataclasses and `load_batch_config`.
- [x] Run the same test and confirm it passes.

## Task 2: Artifact Paths, Atomic Writes, And Locks

- [x] Write failing tests in `packages/arc-llm/tests/test_proposers_reviewer_artifacts.py`:
  - `LoopPaths` creates `run_dir/run_id/loops/loop_id/rounds/round_001/...`.
  - JSON/text writes are complete files at the target path.
  - acquiring the same loop lock twice raises a lock conflict.
  - lock file records run ID and loop ID.
- [x] Run:
  `packages/arc-paper/.venv/bin/python -m pytest packages/arc-llm/tests/test_proposers_reviewer_artifacts.py -q`
  Expected: import/module failures before implementation.
- [x] Implement `artifacts.py`.
- [x] Run the same test and confirm it passes.

## Task 3: Prompt Context And Rendering

- [x] Write failing tests in `packages/arc-llm/tests/test_proposers_reviewer_runner.py` for prompt context:
  - round 2 proposer prompt includes round 1 proposer output and reviewer message.
  - current-round proposer prompt does not include another proposer's current-round output.
  - reviewer prompt includes all current-round proposer outputs.
- [x] Run the test file and confirm the prompt tests fail before implementation.
- [x] Implement `prompts.py` and the context-building portions of `runner.py`.
- [x] Run the test file and confirm the prompt tests pass.

## Task 4: Batch And Loop Runner

- [x] Extend `test_proposers_reviewer_runner.py` with fake LLM calls:
  - two loops run under one `run_dir/run_id` and produce isolated artifacts.
  - `max_rounds` controls round count.
  - early stop is honored when enabled.
  - early stop request is recorded but ignored when disabled.
  - a failed loop does not corrupt another loop's artifacts.
  - worker envs are separate dicts and `os.environ` is unchanged.
- [x] Run the test file and confirm failures before implementation.
- [x] Implement `runner.py` using `ThreadPoolExecutor`.
- [x] Run the test file and confirm it passes.

## Task 5: CLI

- [x] Write failing tests in `packages/arc-llm/tests/test_proposers_reviewer_cli.py`:
  - CLI accepts `proposers-reviewer-loop --config config.json --json`.
  - CLI dry-run validates config without LLM calls.
  - CLI `--max-concurrent-loops` overrides only operational concurrency.
- [x] Run:
  `packages/arc-paper/.venv/bin/python -m pytest packages/arc-llm/tests/test_proposers_reviewer_cli.py -q`
  Expected: command missing before implementation.
- [x] Modify `cli.py` and `__init__.py`.
- [x] Run the CLI test and confirm it passes.

## Task 6: Suggest Ideas Workflow Documentation

- [x] Update `skills/arc/references/research-workflows/suggest-ideas.md`:
  - remove placeholder stop-only language;
  - require domain artifacts first;
  - create `run_dir=<project-dir>/suggest-ideas`;
  - launch two loops with `max_concurrent_loops=2`, `max_rounds=5`, and early stop disabled;
  - specify proposer/reviewer permissions and artifact reporting;
  - keep final idea selection caller-owned and artifact-backed.
- [x] Run `git diff --check`.

## Task 7: Verification

- [x] Run focused tests:
  `packages/arc-paper/.venv/bin/python -m pytest packages/arc-llm/tests -q`
- [x] Run combined suite:
  `packages/arc-paper/.venv/bin/python -m pytest packages/arc-llm/tests packages/arc-paper/tests packages/arc-domain/tests packages/arc-mcp/tests -q`
- [x] Run `git status --short` and review changed files.
