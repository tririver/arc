# Simplify Research Ideas Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `research_ideas_runner.py` a thin wrapper around the standard `arc-llm` proposers-reviewer batch runner with no global reviewer.

**Architecture:** The workflow runner discovers enabled idea variants, materializes `loops_per_variant` independent one-proposer/one-reviewer loops per variant, attaches project context and the centralized marking scheme, then calls `run_proposers_reviewer_batch` once. Reporting summarizes loop artifacts and selected scored rounds from per-loop reviewer outputs only.

**Tech Stack:** Python, `arc_llm.proposers_reviewer`, pytest, JSON workflow templates.

---

### Task 1: Capture New Behavior In Tests

**Files:**
- Modify: `tests/test_research_ideas_runner.py`
- Modify: `tests/test_arc_research_workflow_docs.py`

- [ ] **Step 1: Replace global-review runner test**

Use a fake batch runner that writes five scored rounds per loop. Assert that `run_research_ideas()` never calls a JSON/global reviewer runner, returns `reviewer_call_count == 0`, sets `loop_reviewer_call_count == 10` when `loops_per_variant == 1`, and writes selected per-loop review paths.

- [ ] **Step 2: Update workflow-doc tests**

Change workflow documentation tests so they require no global reviewer references and keep validating the loop reviewer schema is generated from `suggest-ideas-marking-scheme.json`.

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest tests/test_research_ideas_runner.py tests/test_arc_research_workflow_docs.py -q
```

Expected before implementation: failures mentioning global reviewer assumptions or missing no-global result fields.

### Task 2: Simplify Runner And Config

**Files:**
- Modify: `skills/arc/references/research-workflows/research_ideas_runner.py`
- Modify: `skills/arc/references/research-workflows/research_ideas_config.py`
- Modify: `skills/arc/references/research-workflows/research-ideas.config.template.json`

- [ ] **Step 1: Remove global-review code paths**

Delete global reviewer dataclass/config parsing, global review prompt/schema/validation, and direct `run_json` usage from the research-ideas wrapper.

- [ ] **Step 2: Keep batch materialization only**

Keep variant discovery, context attachment, reviewer output-schema injection from `suggest-ideas-marking-scheme.json`, `run_proposers_reviewer_batch`, selected-round extraction, report writing, and dry-run summaries.

- [ ] **Step 3: Update report output**

Write a markdown report from per-loop selected review marks only. Do not rank across variants with a global reviewer.

### Task 3: Update Workflow Docs And Packaged Copies

**Files:**
- Modify: `skills/arc/references/research-workflows/research-ideas.md`
- Modify matching files under `packaging/codex/arc/skills/arc/references/research-workflows/`
- Modify matching files under `packaging/claude/arc/skills/arc/references/research-workflows/`

- [ ] **Step 1: Remove global-review instructions**

Describe the workflow as concurrent proposer-reviewer loops, default five loops per variant and five reviewer reports per loop.

- [ ] **Step 2: Sync packaged workflow files**

Copy changed source workflow files to both package trees.

### Task 4: Verify

- [ ] **Step 1: Run focused tests**

```bash
packages/arc-paper/.venv/bin/python -m pytest tests/test_research_ideas_runner.py tests/test_arc_research_workflow_docs.py -q
```

- [ ] **Step 2: Run combined local suite**

```bash
packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests \
  packages/arc-paper/tests \
  packages/arc-domain/tests \
  packages/arc-mcp/tests \
  tests/test_arc_research_workflow_docs.py \
  tests/test_research_ideas_runner.py \
  -q
```
