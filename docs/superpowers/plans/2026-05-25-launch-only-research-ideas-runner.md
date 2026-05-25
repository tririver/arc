# Launch-Only Research Ideas Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `research_ideas_runner.py` to a concurrency-safe launcher over the standard `arc-llm` proposers-reviewer batch runner.

**Architecture:** The runner builds the batch payload for enabled idea variants, writes only the generated batch config before launch, delegates all concurrent execution and artifacts to `arc_llm.proposers_reviewer`, and returns the standard batch result. Ranking and report generation stay in the existing read-only ranking script.

**Tech Stack:** Python, `arc_llm.proposers_reviewer`, pytest, JSON workflow templates.

---

### Task 1: Encode Launch-Only Behavior

**Files:**
- Modify: `tests/test_research_ideas_runner.py`
- Modify: `tests/test_arc_research_workflow_docs.py`

- [ ] **Step 1: Update runner test**

Change the fake batch runner so it does not create proposer/reviewer round artifacts. Assert the research-ideas runner returns the batch result, generated batch config path, and loop plan summaries, but does not return `ideas`, `report`, `selected_round`, `selected_review_path`, or copied summary artifacts.

- [ ] **Step 2: Update workflow docs tests**

Require the workflow docs to point users at the standard `arc-llm` loop artifacts and the read-only ranking script, not `<project-dir>/research-ideas.md` or copied selected-round records.

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest tests/test_research_ideas_runner.py tests/test_arc_research_workflow_docs.py -q
```

Expected before implementation: failures because the current runner still scans/copies selected rounds and writes a report.

### Task 2: Rewrite Runner As Thin Launcher

**Files:**
- Modify: `skills/arc/references/research-workflows/research_ideas_runner.py`

- [ ] **Step 1: Keep materialization helpers**

Keep `IdeaPlan`, `_materialize_ideas`, `_caller_context`, `_loop_batch_config`, `_idea_loop_payload`, proposer/reviewer payload helpers, domain markdown loading, placeholder replacement, and CLI parsing.

- [ ] **Step 2: Delete postprocessing and wrapper state**

Remove selected-round scanning, summary artifact copying, markdown report writing, wrapper state writing, wrapper atomic write helpers, and copied result/error helpers.

- [ ] **Step 3: Reuse arc-llm atomic writer**

Import `atomic_write_json` from `arc_llm.proposers_reviewer.artifacts` for the generated batch config. Do not write any files while loops are executing except through `run_proposers_reviewer_batch`.

- [ ] **Step 4: Return launch result**

Return `schema_version`, `status`, `run_id`, `run_root`, `batch_config_path`, `proposal_count`, `reviewer_call_count: 0`, `loop_reviewer_call_count`, `max_concurrent_loops`, `max_concurrent_proposal_calls`, `loops`, and `batch_result`.

### Task 3: Update Workflow Docs And Packaged Copies

**Files:**
- Modify: `skills/arc/references/research-workflows/research-ideas.md`
- Modify matching files under `packaging/codex/arc/skills/arc/references/research-workflows/`
- Modify matching files under `packaging/claude/arc/skills/arc/references/research-workflows/`

- [ ] **Step 1: Document concurrency-safe ownership**

State that `arc-llm` owns concurrent writes under the batch run root and that ranking/reporting is a separate read-only step after completion.

- [ ] **Step 2: Sync packaged copies**

Copy changed workflow files to both package trees.

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
