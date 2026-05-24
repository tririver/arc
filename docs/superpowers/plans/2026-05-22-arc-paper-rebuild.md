# ARC Paper Query Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `arc-paper` as a cache-first Python package with CLI, LLM summary generation, resumable batch processing, MCP tools, and Codex/Claude plugin wrappers.

**Architecture:** `arc-paper` owns all deterministic paper data, caching, parsing, LLM summary contracts, provider selection, and batch execution. MCP, skills, and plugins are thin adapters over the CLI/package; plugins inject `ARC_AGENT_HOST` and `ARC_LLM_PROVIDER`, with parent-process detection only as fallback. The first implementation targets ar5iv full text and INSPIRE metadata/references/citers only.

**Tech Stack:** Python 3.11+, stdlib `argparse`/`sqlite3`, `httpx`, `beautifulsoup4`, `lxml`, `jsonschema`, `pytest`, optional `mcp` Python SDK for `arc-mcp`, host CLI providers via `codex exec` and `claude -p`.

---

## Scope And Constraints

- Do not modify `0_ref/`; it is reference-only and ignored by git.
- Use the legacy `0_ref/skills/arc/utils/paper-query/` reference implementation for provider endpoints, old cache behavior, parser heuristics, and CLI affordances. Read it before implementing each related component, but do not preserve compatibility and do not copy the old OpenAlex/Semantic Scholar/network-building design.
- No backwards compatibility with the old reference utility.
- No Semantic Scholar, OpenAlex, paper graph construction, or network-building logic.
- Use arXiv-style IDs as primary IDs: `arXiv:0911.3380`, `arXiv:2512.06790`, `arXiv:hep-th/0601001`.
- Cache full text, references, title, abstract, and authors permanently; cache citers for 30 days.
- Unit tests must not require network. Network integration tests must be opt-in with `ARC_RUN_NET_TESTS=1`.
- Current workspace is not a git repository, so commit steps are omitted until the GitHub repository is initialized.

## Target File Map

### `packages/arc-paper`

- `pyproject.toml`: package metadata, dependencies, console script.
- `src/arc_paper/ids.py`: paper ID parsing and normalization.
- `src/arc_paper/results.py`: stable JSON result envelope.
- `src/arc_paper/cache.py`: cache paths, TTL rules, JSON/HTML read-write helpers.
- `src/arc_paper/host.py`: env-first host/provider detection, parent-process fallback on Linux/macOS/Windows.
- `src/arc_paper/providers/base.py`: provider protocols and errors.
- `src/arc_paper/providers/ar5iv.py`: ar5iv HTML download/cache.
- `src/arc_paper/providers/inspire.py`: INSPIRE metadata/references/citers download/cache.
- `src/arc_paper/parse/ar5iv_html.py`: HTML-to-text, TOC, section extraction.
- `src/arc_paper/parse/equations.py`: equation context extraction.
- `src/arc_paper/service.py`: public Python API for single/list paper queries.
- `src/arc_paper/cli.py`: CLI parser and JSON output.
- `src/arc_paper/summary/schema.py`: summary JSON schema loader and validator.
- `src/arc_paper/summary/input_pack.py`: deterministic input pack builder for LLM summary.
- `src/arc_paper/summary/store.py`: summary cache read/write and source-hash validation.
- `src/arc_paper/summary/providers/*.py`: LLM provider abstraction and implementations.
- `src/arc_paper/batch/db.py`: SQLite queue and status tracking.
- `src/arc_paper/batch/runner.py`: resumable concurrent batch runner.
- `tests/`: unit tests and opt-in network tests.

### Shared Assets

- `packages/arc-paper/src/arc_paper/summary/schemas/paper-summary-v1.schema.json`: stable output schema used by CLI, MCP, and host CLI providers.
- `packages/arc-paper/src/arc_paper/summary/prompts/paper-summary-v1.md`: prompt template for high-quality paper summaries.
- `examples/arc-paper/papers.txt`: sample IDs for batch testing.

### Adapters

- `packages/arc-mcp/src/arc_mcp/server.py`: MCP tool server over `arc_paper.service`.
- `skills/arc/SKILL.md`: short agent workflow for ARC research and arc-paper use.
- `skills/arc/references/arc-paper.md`: detailed `needs_llm`, batch, and troubleshooting workflow.
- `packaging/codex/arc/.codex-plugin/plugin.json`: Codex plugin manifest.
- `packaging/codex/arc/.mcp.json`: Codex plugin MCP server config.
- `packaging/codex/arc/scripts/arc-mcp-codex`: wrapper that exports `ARC_AGENT_HOST=codex`.
- `packaging/claude/arc/.mcp.json`: Claude plugin MCP server config.
- `packaging/claude/arc/scripts/arc-mcp-claude`: wrapper that exports `ARC_AGENT_HOST=claude-code`.

---

## Milestone 1: Deterministic `arc-paper`

### Task 1: Package Skeleton And Test Harness

**Files:**
- Create: `packages/arc-paper/pyproject.toml`
- Create: `packages/arc-paper/src/arc_paper/__init__.py`
- Create: `packages/arc-paper/tests/test_import.py`

- [ ] **Step 1: Create package metadata**

Use this package shape:

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "arc-paper"
version = "0.1.0"
description = "Cache-first ar5iv and INSPIRE paper query tools for ARC."
requires-python = ">=3.11"
dependencies = [
  "beautifulsoup4>=4.12",
  "httpx>=0.27",
  "jsonschema>=4.22",
  "lxml>=5.0"
]

[project.optional-dependencies]
test = ["pytest>=8.0"]

[project.scripts]
arc-paper = "arc_paper.cli:main"
```

- [ ] **Step 2: Add import smoke test**

```python
def test_package_imports():
    import arc_paper

    assert arc_paper.__version__ == "0.1.0"
```

- [ ] **Step 3: Run smoke test**

Run:

```bash
cd packages/arc-paper
python3 -m pip install -e ".[test]"
python3 -m pytest tests/test_import.py -v
```

Expected: one passing test.

### Task 2: IDs, Result Envelope, Cache, And Host Detection

**Files:**
- Create: `packages/arc-paper/src/arc_paper/ids.py`
- Create: `packages/arc-paper/src/arc_paper/results.py`
- Create: `packages/arc-paper/src/arc_paper/cache.py`
- Create: `packages/arc-paper/src/arc_paper/host.py`
- Create: `packages/arc-paper/tests/test_ids.py`
- Create: `packages/arc-paper/tests/test_results.py`
- Create: `packages/arc-paper/tests/test_host.py`

- [ ] **Step 1: Implement paper ID normalization tests**

Test these cases:

```python
from arc_paper.ids import normalize_paper_id, arxiv_path_id


def test_normalize_new_arxiv_id():
    assert normalize_paper_id("0911.3380") == "arXiv:0911.3380"
    assert normalize_paper_id("arxiv:0911.3380") == "arXiv:0911.3380"


def test_normalize_old_arxiv_id():
    assert normalize_paper_id("hep-th/0601001") == "arXiv:hep-th/0601001"
    assert arxiv_path_id("arXiv:hep-th/0601001") == "hep-th/0601001"
```

- [ ] **Step 2: Implement stable result envelope tests**

```python
from arc_paper.results import ok, err


def test_ok_envelope():
    result = ok({"title": "A"}, provider="inspire", cache="hit")
    assert result["ok"] is True
    assert result["data"] == {"title": "A"}
    assert result["errors"] == []
    assert result["meta"]["provider"] == "inspire"


def test_error_envelope():
    result = err("section_not_found", "Section 9 not found", toc=[{"id": "1"}])
    assert result["ok"] is False
    assert result["error"]["code"] == "section_not_found"
    assert result["toc"] == [{"id": "1"}]
```

- [ ] **Step 3: Implement cache contract**

Cache root resolution:

```text
ARC_PAPER_CACHE if set
else XDG_CACHE_HOME/arc/arc-paper
else ~/.cache/arc/arc-paper
```

Cache layout:

```text
<cache-root>/papers/<safe-paper-id>/
  ar5iv/fulltext.html
  ar5iv/parsed.json
  inspire/metadata.json
  inspire/references.json
  inspire/citers.json
  summaries/<prompt-version>/<source-hash>.json
```

Use `safe-paper-id = quote(normalized_id, safe="")`.

- [ ] **Step 4: Implement host detection**

Algorithm:

```text
if ARC_AGENT_HOST is set:
    return host with confidence 1.0 and signal env:ARC_AGENT_HOST
else walk parent processes:
    codex match: "codex" or "@openai/codex"
    claude match: "claude" or "@anthropic-ai/claude-code" or ".claude/shell-snapshots"
else unknown
```

Provider selection:

```text
if ARC_LLM_PROVIDER set: use it
elif host == codex: codex-cli
elif host == claude-code: claude-cli
else manual
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
cd packages/arc-paper
python3 -m pytest tests/test_ids.py tests/test_results.py tests/test_host.py -v
```

Expected: all pass.

### Task 3: ar5iv And INSPIRE Providers

**Files:**
- Create: `packages/arc-paper/src/arc_paper/providers/__init__.py`
- Create: `packages/arc-paper/src/arc_paper/providers/base.py`
- Create: `packages/arc-paper/src/arc_paper/providers/ar5iv.py`
- Create: `packages/arc-paper/src/arc_paper/providers/inspire.py`
- Create: `packages/arc-paper/tests/test_providers_ar5iv.py`
- Create: `packages/arc-paper/tests/test_providers_inspire.py`

- [ ] **Step 1: Define provider interfaces**

Use small, testable interfaces:

```python
class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class FullTextProvider:
    def get_html(self, paper_id: str, *, refresh: bool = False) -> str: ...


class MetadataProvider:
    def get_metadata(self, paper_id: str, *, refresh: bool = False) -> dict: ...
    def get_references(self, paper_id: str, *, refresh: bool = False) -> list[dict]: ...
    def get_citers(self, paper_id: str, *, refresh: bool = False) -> list[dict]: ...
```

- [ ] **Step 2: Implement ar5iv URL behavior**

Expected URLs:

```text
arXiv:0911.3380 -> https://ar5iv.labs.arxiv.org/html/0911.3380
arXiv:hep-th/0601001 -> https://ar5iv.labs.arxiv.org/html/hep-th/0601001
```

Provider behavior:

```text
if cached fulltext.html exists and refresh is false: return cache
else GET URL with timeout
if HTTP 200: write cache and return text
else raise ProviderError("ar5iv_fetch_failed", ...)
```

- [ ] **Step 3: Implement INSPIRE URL behavior**

Use:

```text
metadata: https://inspirehep.net/api/arxiv/<arxiv-path-id>
citers: https://inspirehep.net/api/literature?q=refersto:recid:<recid>&size=1000&page=N
```

References should first use metadata JSON fields when available; if INSPIRE requires a literature endpoint follow-up for full reference records, isolate that in one method so provider replacement remains easy.

- [ ] **Step 4: Test cache hit avoids network**

Use a fake transport/client or monkeypatch provider HTTP call. Assert second call reads cache and does not call network.

- [ ] **Step 5: Run provider tests**

```bash
cd packages/arc-paper
python3 -m pytest tests/test_providers_ar5iv.py tests/test_providers_inspire.py -v
```

Expected: all pass without network.

### Task 4: ar5iv HTML Parser, TOC, Sections, And Equation Context

**Files:**
- Create: `packages/arc-paper/src/arc_paper/parse/__init__.py`
- Create: `packages/arc-paper/src/arc_paper/parse/ar5iv_html.py`
- Create: `packages/arc-paper/src/arc_paper/parse/equations.py`
- Create: `packages/arc-paper/tests/fixtures/ar5iv_sample.html`
- Create: `packages/arc-paper/tests/test_parse_ar5iv.py`
- Create: `packages/arc-paper/tests/test_equations.py`

- [ ] **Step 1: Add a small fixture**

Fixture must include:

```html
<html>
  <body>
    <h1 class="ltx_title">Sample Paper</h1>
    <section id="S1"><h2>1 Introduction</h2><p>Intro text.</p></section>
    <section id="S2"><h2>2 Model</h2><p>Model text before equation.</p>
      <table class="ltx_equation" id="E1"><tr><td>E = mc^2</td></tr></table>
      <p>Model text after equation.</p>
    </section>
  </body>
</html>
```

- [ ] **Step 2: Parser tests**

```python
from arc_paper.parse.ar5iv_html import parse_html, get_section


def test_toc_and_sections(sample_html):
    parsed = parse_html(sample_html)
    assert parsed["toc"] == [
        {"id": "S1", "title": "1 Introduction"},
        {"id": "S2", "title": "2 Model"},
    ]
    assert "Intro text." in get_section(parsed, "S1")["text"]


def test_missing_section_returns_toc(sample_html):
    parsed = parse_html(sample_html)
    missing = get_section(parsed, "S9")
    assert missing["ok"] is False
    assert missing["error"]["code"] == "section_not_found"
    assert len(missing["toc"]) == 2
```

- [ ] **Step 3: Equation context tests**

```python
from arc_paper.parse.equations import find_equation_context


def test_equation_context(sample_html):
    contexts = find_equation_context(sample_html, "E = mc^2", window_paragraphs=1)
    assert len(contexts) == 1
    assert "Model text before equation." in contexts[0]["before"]
    assert "Model text after equation." in contexts[0]["after"]
```

- [ ] **Step 4: Run parser tests**

```bash
cd packages/arc-paper
python3 -m pytest tests/test_parse_ar5iv.py tests/test_equations.py -v
```

Expected: all pass.

### Task 5: Public Service API And CLI Query Commands

**Files:**
- Create: `packages/arc-paper/src/arc_paper/service.py`
- Create: `packages/arc-paper/src/arc_paper/cli.py`
- Create: `packages/arc-paper/tests/test_service.py`
- Create: `packages/arc-paper/tests/test_cli.py`

- [ ] **Step 1: Implement service API**

Expose these functions:

```python
get_title(ids)
get_abstract(ids)
get_authors(ids)
get_citers(ids)
get_citer_count(ids)
get_references(ids)
get_toc(ids)
get_section(ids, section)
get_equation_context(ids, query)
get_llm_summary(ids)
```

Input behavior:

```text
single paper ID -> single result envelope
list of paper IDs -> dict[normalized_id, result envelope]
```

- [ ] **Step 2: Implement CLI JSON commands**

Command shape:

```bash
arc-paper get-title arXiv:0911.3380 --json
arc-paper get-section arXiv:0911.3380 --section S2 --json
arc-paper get-references arXiv:0911.3380 arXiv:hep-th/0601001 --json
arc-paper doctor host --json
arc-paper doctor provider --json
```

All commands emit a result envelope and exit nonzero only for CLI usage errors, not for paper-level missing sections or provider errors.

- [ ] **Step 3: Run service and CLI tests**

```bash
cd packages/arc-paper
python3 -m pytest tests/test_service.py tests/test_cli.py -v
```

Expected: all pass.

---

## Milestone 2: LLM Summary For One Paper

### Task 6: Summary Schema, Prompt, Input Pack, And Store

**Files:**
- Create: `packages/arc-paper/src/arc_paper/summary/schemas/paper-summary-v1.schema.json`
- Create: `packages/arc-paper/src/arc_paper/summary/prompts/paper-summary-v1.md`
- Create: `packages/arc-paper/src/arc_paper/summary/__init__.py`
- Create: `packages/arc-paper/src/arc_paper/summary/schema.py`
- Create: `packages/arc-paper/src/arc_paper/summary/input_pack.py`
- Create: `packages/arc-paper/src/arc_paper/summary/store.py`
- Create: `packages/arc-paper/tests/test_summary_schema.py`
- Create: `packages/arc-paper/tests/test_summary_store.py`

- [ ] **Step 1: Add schema**

Required top-level fields:

```json
{
  "schema_version": "arc.paper_llm_summary.v1",
  "paper_id": "arXiv:0911.3380",
  "title": "string",
  "authors_short": "string",
  "high_value_summary": ["string"],
  "toc": [{"section_id": "string", "title": "string", "one_sentence_summary": "string"}],
  "reading_guide": [{"purpose": "string", "sections": ["string"], "reason": "string"}],
  "warnings": ["string"],
  "provenance": {
    "created_at": "ISO-8601 string",
    "method": "manual|codex-cli|claude-cli|openai|anthropic",
    "model": "string",
    "prompt_version": "paper-summary-v1",
    "source_hash": "sha256 hex string"
  }
}
```

- [ ] **Step 2: Add prompt template**

Prompt must require:

```text
- concise title and authors_short
- high-value summary more useful than abstract
- section-level one-sentence summaries
- reading guide for idea generation and targeted follow-up reading
- JSON only, conforming to schema
- no unsupported claims beyond input pack
```

- [ ] **Step 3: Implement input pack**

Input pack includes:

```json
{
  "paper_id": "...",
  "metadata": {"title": "...", "authors": [...], "abstract": "..."},
  "toc": [...],
  "sections": [{"section_id": "...", "title": "...", "text": "..."}],
  "references": [...],
  "source_hash": "..."
}
```

Limit section text deterministically if needed by max character budget; preserve section titles and start/end snippets before truncating middle text.

- [ ] **Step 4: Implement `needs_llm` response**

When no cached summary exists:

```json
{
  "ok": false,
  "status": "needs_llm",
  "paper_id": "arXiv:0911.3380",
  "llm_task": {
    "task_type": "paper_summary",
    "prompt_version": "paper-summary-v1",
    "system_prompt": "...",
    "user_prompt": "...",
    "input_pack": {},
    "output_schema": {}
  },
  "next": {
    "store_command": "arc-paper store-llm-summary arXiv:0911.3380 --summary-json -"
  }
}
```

- [ ] **Step 5: Implement store command**

Command:

```bash
arc-paper store-llm-summary arXiv:0911.3380 --summary-json summary.json --json
```

Behavior:

```text
validate JSON schema
verify paper_id matches
verify source_hash matches current input pack unless --allow-stale
write summary cache
return ok envelope with summary path and summary data
```

- [ ] **Step 6: Run summary tests**

```bash
cd packages/arc-paper
python3 -m pytest tests/test_summary_schema.py tests/test_summary_store.py -v
```

Expected: all pass.

### Task 7: LLM Provider Interface And Single-Paper Generation

**Files:**
- Create: `packages/arc-paper/src/arc_paper/summary/providers/__init__.py`
- Create: `packages/arc-paper/src/arc_paper/summary/providers/base.py`
- Create: `packages/arc-paper/src/arc_paper/summary/providers/manual.py`
- Create: `packages/arc-paper/src/arc_paper/summary/providers/codex_cli.py`
- Create: `packages/arc-paper/src/arc_paper/summary/providers/claude_cli.py`
- Create: `packages/arc-paper/src/arc_paper/summary/providers/select.py`
- Create: `packages/arc-paper/tests/test_provider_selection.py`
- Create: `packages/arc-paper/tests/test_host_cli_providers.py`

- [ ] **Step 1: Define provider protocol**

```python
class LLMProvider:
    name: str

    def generate_summary(self, task: dict, *, model: str | None = None) -> dict:
        """Return schema-conforming summary JSON or raise LLMProviderError."""
```

- [ ] **Step 2: Implement provider selection**

Selection order:

```text
explicit --provider argument
ARC_LLM_PROVIDER
ARC_AGENT_HOST -> codex-cli / claude-cli
parent process detection -> codex-cli / claude-cli
manual
```

- [ ] **Step 3: Implement `codex-cli` provider**

Use:

```bash
codex exec \
  --skip-git-repo-check \
  --ephemeral \
  --sandbox read-only \
  --output-schema <schema-file> \
  --output-last-message <output-file> \
  <prompt-file>
```

Implementation writes prompt, schema, and output to a temporary directory, parses output JSON, validates schema, and returns summary.

- [ ] **Step 4: Implement `claude-cli` provider**

Use:

```bash
claude -p \
  --bare \
  --tools "" \
  --no-session-persistence \
  --output-format json \
  --json-schema '<schema-json>' \
  '<prompt>'
```

Implementation extracts JSON content from Claude's output, validates schema, and returns summary.

- [ ] **Step 5: Add single-paper generation command**

```bash
arc-paper generate-llm-summary arXiv:0911.3380 --provider auto --json
```

Behavior:

```text
if cached summary current: return ok cache hit
else build task
select provider
if provider manual: return needs_llm
else call provider, validate, store, return ok
```

- [ ] **Step 6: Run provider tests**

Tests monkeypatch `subprocess.run` and assert exact commands include `codex exec` or `claude -p`.

```bash
cd packages/arc-paper
python3 -m pytest tests/test_provider_selection.py tests/test_host_cli_providers.py -v
```

Expected: all pass.

---

## Milestone 3: 500-Paper Batch Runner

### Task 8: Batch Database And Queue

**Files:**
- Create: `packages/arc-paper/src/arc_paper/batch/__init__.py`
- Create: `packages/arc-paper/src/arc_paper/batch/db.py`
- Create: `packages/arc-paper/tests/test_batch_db.py`
- Create: `examples/arc-paper/papers.txt`

- [ ] **Step 1: Define SQLite schema**

Tables:

```sql
CREATE TABLE batches (
  name TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  prompt_version TEXT NOT NULL
);

CREATE TABLE batch_items (
  batch_name TEXT NOT NULL,
  paper_id TEXT NOT NULL,
  status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  provider TEXT,
  model TEXT,
  source_hash TEXT,
  summary_path TEXT,
  last_error TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (batch_name, paper_id)
);
```

Valid statuses:

```text
queued, prefetching, ready, running, done, failed, skipped
```

- [ ] **Step 2: Implement create/list/status operations**

Core methods:

```python
create_batch(name: str, paper_ids: list[str], prompt_version: str) -> None
next_items(name: str, status: str, limit: int) -> list[BatchItem]
mark_status(name: str, paper_id: str, status: str, **fields) -> None
status_counts(name: str) -> dict[str, int]
```

- [ ] **Step 3: Add CLI commands**

```bash
arc-paper summary-batch create papers.txt --name qft-ideas --json
arc-paper summary-batch status qft-ideas --json
```

- [ ] **Step 4: Run DB tests**

```bash
cd packages/arc-paper
python3 -m pytest tests/test_batch_db.py -v
```

Expected: all pass.

### Task 9: Prefetch And Run Batch

**Files:**
- Create: `packages/arc-paper/src/arc_paper/batch/runner.py`
- Create: `packages/arc-paper/tests/test_batch_runner.py`

- [ ] **Step 1: Implement prefetch**

Command:

```bash
arc-paper summary-batch prefetch qft-ideas --workers 8 --json
```

Behavior:

```text
for queued/failed items:
  download/cache metadata, references, citers, ar5iv HTML
  build parsed HTML cache
  mark ready or failed
```

- [ ] **Step 2: Implement run**

Command:

```bash
arc-paper summary-batch run qft-ideas --provider auto --concurrency 2 --resume --json
```

Behavior:

```text
select ready/failed items unless summary cache is current
mark running
call generate-llm-summary
mark done with summary_path or failed with last_error
respect --max-items for calibration runs
```

- [ ] **Step 3: Add calibration workflow**

Command:

```bash
arc-paper summary-batch run qft-ideas --provider auto --max-items 10 --concurrency 1 --json
```

This supports reviewing 10 summaries before spending time on 500.

- [ ] **Step 4: Add export and retry**

Commands:

```bash
arc-paper summary-batch export qft-ideas --format jsonl --output summaries.jsonl
arc-paper summary-batch retry-failed qft-ideas --json
```

- [ ] **Step 5: Run runner tests**

```bash
cd packages/arc-paper
python3 -m pytest tests/test_batch_runner.py -v
```

Expected: all pass with fake providers and no network.

---

## Milestone 4: MCP, Skill, And Plugin Wrappers

### Task 10: MCP Server

**Files:**
- Create: `packages/arc-mcp/pyproject.toml`
- Create: `packages/arc-mcp/src/arc_mcp/__init__.py`
- Create: `packages/arc-mcp/src/arc_mcp/server.py`
- Create: `packages/arc-mcp/tests/test_server_tools.py`

- [ ] **Step 1: Add MCP package metadata**

Console script:

```toml
[project.scripts]
arc-mcp = "arc_mcp.server:main"
```

Runtime dependencies:

```text
arc-paper
mcp
```

- [ ] **Step 2: Expose MCP tools**

Tools:

```text
get_title
get_abstract
get_authors
get_citers
get_citer_count
get_references
get_toc
get_section
get_equation_context
get_LLM_summary
generate_LLM_summary
store_LLM_summary
summary_batch_create
summary_batch_prefetch
summary_batch_run
summary_batch_status
summary_batch_export
summary_batch_retry_failed
doctor_host
doctor_provider
```

- [ ] **Step 3: Capture MCP clientInfo when available**

If MCP initialize `clientInfo.name` says Codex or Claude Code, set process-local host context for provider selection. Plugin env still has priority.

- [ ] **Step 4: Test MCP tool wrappers**

Tests call tool handlers directly with fake service functions.

```bash
cd packages/arc-mcp
python3 -m pytest tests/test_server_tools.py -v
```

Expected: all pass.

### Task 11: ARC Skill

**Files:**
- Create: `skills/arc/SKILL.md`
- Create: `skills/arc/references/arc-paper.md`

- [ ] **Step 1: Write compact `SKILL.md`**

Rules:

```text
Use `arc-paper` or MCP tools for paper data.
For one-paper summaries, call `generate-llm-summary`.
If result status is `needs_llm`, generate JSON using returned task and call `store-llm-summary`.
For more than 10 papers, use summary-batch commands.
Never manually scrape ar5iv/INSPIRE when CLI/MCP is available.
```

- [ ] **Step 2: Write detailed reference**

Include:

```text
single-paper workflow
needs_llm fallback workflow
batch calibration workflow
500-paper batch workflow
doctor host/provider troubleshooting
cache refresh/offline guidance
```

- [ ] **Step 3: Validate skill**

Run Codex skill validation if available:

```bash
python3 /home/user/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/arc
```

Expected: validation passes.

### Task 12: Codex And Claude Plugin Packaging

**Files:**
- Create: `packaging/codex/arc/.codex-plugin/plugin.json`
- Create: `packaging/codex/arc/.mcp.json`
- Create: `packaging/codex/arc/scripts/arc-mcp-codex`
- Create: `packaging/codex/arc/skills/arc/SKILL.md`
- Create: `packaging/claude/arc/.mcp.json`
- Create: `packaging/claude/arc/scripts/arc-mcp-claude`
- Create: `packaging/claude/arc/skills/arc/SKILL.md`

- [ ] **Step 1: Codex wrapper**

```bash
#!/usr/bin/env bash
set -euo pipefail
export ARC_AGENT_HOST=codex
export ARC_LLM_PROVIDER="${ARC_LLM_PROVIDER:-codex-cli}"
exec arc-mcp "$@"
```

- [ ] **Step 2: Claude wrapper**

```bash
#!/usr/bin/env bash
set -euo pipefail
export ARC_AGENT_HOST=claude-code
export ARC_LLM_PROVIDER="${ARC_LLM_PROVIDER:-claude-cli}"
exec arc-mcp "$@"
```

- [ ] **Step 3: Plugin-specific skill command defaults**

Codex plugin skill should prefer:

```bash
ARC_AGENT_HOST=codex ARC_LLM_PROVIDER=codex-cli arc-paper ...
```

Claude plugin skill should prefer:

```bash
ARC_AGENT_HOST=claude-code ARC_LLM_PROVIDER=claude-cli arc-paper ...
```

This keeps direct CLI use zero-config even outside MCP.

- [ ] **Step 4: Validate plugin manifests**

For Codex plugin, use the local validator from `plugin-creator` when ready:

```bash
python3 /home/user/.codex/skills/.system/plugin-creator/scripts/validate_plugin.py packaging/codex/arc
```

Expected: manifest validation passes.

### Task 13: Final Integration And Documentation

**Files:**
- Create: `README.md`
- Create: `tests/integration/test_paper_query_network.py`
- Create: `tests/integration/test_cli_smoke.py`

- [ ] **Step 1: Add README install paths**

Document:

```text
pip install -e packages/arc-paper
pip install -e packages/arc-mcp
Codex plugin install path
Claude plugin install path
batch example for 500 papers
```

- [ ] **Step 2: Add opt-in network test**

Test:

```text
arXiv:0911.3380 metadata from INSPIRE
arXiv:0911.3380 HTML from ar5iv
arXiv:hep-th/0601001 old-style arXiv ID
```

Skip unless:

```bash
ARC_RUN_NET_TESTS=1
```

- [ ] **Step 3: Add end-to-end smoke commands**

Run:

```bash
arc-paper doctor host --json
arc-paper get-title arXiv:0911.3380 --json
arc-paper get-toc arXiv:0911.3380 --json
arc-paper get-llm-summary arXiv:0911.3380 --json
```

Expected:

```text
doctor host returns codex/claude-code/unknown
title returns ok or provider error envelope
toc returns ok after ar5iv fetch or provider error envelope
get-llm-summary returns cache hit or needs_llm
```

---

## Recommended Execution Order

1. Before each implementation task, inspect the relevant files under `0_ref/skills/arc/utils/paper-query/` for practical endpoint details and failure handling.
2. Implement Milestone 1 completely before touching LLM features.
3. Run a live manual check for one modern arXiv ID and one old-style ID.
4. Implement Milestone 2 and generate 3-5 summaries with `manual`, `codex-cli`, and `claude-cli` providers.
5. Tune `packages/arc-paper/src/arc_paper/summary/prompts/paper-summary-v1.md` on 10 hand-picked papers before batch work.
6. Implement Milestone 3 and run `--max-items 10` before any 500-paper batch.
7. Implement Milestone 4 adapters after the package CLI is stable.

## Self-Review

- Spec coverage: ar5iv, INSPIRE, permanent/TTL cache, single/list APIs, section fallback, equation context, LLM summary, batch runner, MCP, skill, plugin env injection, and `0_ref` as read-only reference are covered.
- Placeholder scan: no task uses open-ended "TODO" work; each task defines concrete files, commands, and expected behavior.
- Type consistency: paper IDs normalize through `normalize_paper_id`; all public outputs use result envelopes; summary outputs conform to `paper-summary-v1`.
- Risk: exact Claude plugin manifest details may need adjustment when implementing because Claude plugin packaging is less standardized than Codex plugin packaging. The stable part is the wrapper command exporting `ARC_AGENT_HOST` and `ARC_LLM_PROVIDER`.
