# Centralized Idea Marking Scheme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize ARC research-ideas reviewer scoring in one JSON marking-scheme file and use it for per-loop and global reviewer schemas, prompts, ranking, and reports.

**Architecture:** Add a workflow-local JSON scheme plus a small Python helper that loads it, renders reviewer guidance, builds strict JSON-schema fragments, and exposes mark ordering. Keep the workflow layer responsible for research-ideas scoring; do not change the generic `arc-llm` proposers-reviewer runner.

**Tech Stack:** Python standard library, JSON schema fragments, Markdown workflow references, pytest.

---

### Task 1: Capture Desired Behavior In Tests

**Files:**
- Modify: `tests/test_arc_research_workflow_docs.py`
- Modify: `tests/test_research_ideas_runner.py`

- [ ] Add tests that require `suggest-ideas-marking-scheme.json` to contain the six non-total fields: `user_intent_relevance`, `novelty`, `confidence_of_novelty`, `scientific_value`, `feasibility`, and `first_calculation_clarity`, with maxima 25, 15, 15, 15, 15, and 15.
- [ ] Add tests that require reviewer output schemas to be built from the central scheme rather than hard-coded old `evidence_of_novelty` ranges.
- [ ] Add tests that require per-loop and global reviewer prompt contexts to include `marking_scheme`.
- [ ] Run: `packages/arc-paper/.venv/bin/python -m pytest tests/test_arc_research_workflow_docs.py tests/test_research_ideas_runner.py -q`
- [ ] Expected before implementation: failures mentioning missing `suggest-ideas-marking-scheme.json` or missing new mark fields.

### Task 2: Add Central Marking Scheme And Helper

**Files:**
- Create: `skills/arc/references/research-workflows/suggest-ideas-marking-scheme.json`
- Create: `skills/arc/references/research-workflows/research_ideas_marking.py`

- [ ] Add the JSON scheme with `schema_version`, ordered `marks`, `total_score`, and `tie_break_order`.
- [ ] Add helper functions to load the scheme, return mark fields, build a strict marks schema, render compact reviewer instructions, normalize mark dictionaries, calculate rank keys, and build report headers/rows.
- [ ] Run the focused tests and make this task green before changing packaged copies.

### Task 3: Wire The Workflow To The Scheme

**Files:**
- Modify: `skills/arc/references/research-workflows/research_ideas_config.py`
- Modify: `skills/arc/references/research-workflows/research_ideas_runner.py`
- Modify: `skills/arc/references/research-workflows/suggest-ideas-reviewer-output.schema.json`
- Modify: `skills/arc/references/research-workflows/suggest-ideas-reviewer.template.json`
- Modify: `skills/arc/references/research-workflows/scripts/rank-suggested-ideas.py`

- [ ] Update the global reviewer default prompt to refer to attached `marking_scheme`.
- [ ] Replace per-loop reviewer schema `marks` using the helper-built marks schema.
- [ ] Attach the central marking scheme to loop and global reviewer context.
- [ ] Build global review item schemas from the helper-built marks schema.
- [ ] Replace hard-coded normalization, tie-break order, and report columns with helper functions.
- [ ] Update the ranking script to load the same JSON scheme by path.

### Task 4: Sync Packaged Host Copies And Verify

**Files:**
- Modify/create matching files under `packaging/codex/arc/skills/arc/references/research-workflows/`
- Modify/create matching files under `packaging/claude/arc/skills/arc/references/research-workflows/`

- [ ] Copy the source workflow files into the Codex and Claude packaged skill directories.
- [ ] Run: `packages/arc-paper/.venv/bin/python -m pytest tests/test_arc_research_workflow_docs.py tests/test_research_ideas_runner.py -q`
- [ ] Run: `packages/arc-paper/.venv/bin/python -m pytest packages/arc-llm/tests packages/arc-paper/tests packages/arc-domain/tests packages/arc-mcp/tests tests/test_arc_research_workflow_docs.py tests/test_research_ideas_runner.py -q`
- [ ] Expected final result: all focused and combined local tests pass.
