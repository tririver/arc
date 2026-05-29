# ARC

Advanced Research Compass (ARC) is a cache-first research toolkit for
theoretical-physics papers and paper-centered research workflows. It gives an
agent, or a human using the command line, structured access to arXiv full text,
INSPIRE metadata, references, citers, paper summaries, research-domain graphs,
and multi-agent idea/calculation workflows.

ARC is built around five Python command line tools; `arc-mcp` also runs the
optional MCP server:

- `arc-paper`: paper metadata, references, citers, ar5iv sections, equation
  context, full-text search, LLM paper summaries, and paper-summary batches.
- `arc-domain`: builds a cached research-domain package from a seed paper and
  optional scientific intent.
- `arc-llm`: reusable host LLM execution, provider selection, and
  proposers-reviewer workflows.
- `arc-typeset`: deterministic typesetting utilities, including Markdown to PDF
  conversion through Pandoc and XeLaTeX.
- `arc-mcp`: exposes ARC tools to MCP clients and manages background jobs.
- `skills/arc`: agent-facing workflow instructions for domain building, idea
  generation, and research calculations.

## Who This Is For

Use ARC when you want to:

- Look up reliable paper metadata, references, citers, sections, or equations.
- Summarize a paper from cached ar5iv/INSPIRE data.
- Build a research-domain overview from a seed paper.
- Generate ideas using domain context and reviewer scoring.
- Plan and execute a careful symbolic or numerical research calculation with
  explicit provenance and checks.

Deterministic paper queries do not need an LLM. Paper summaries, domain
briefings, idea loops, and calculation workflow runners need a host LLM
provider.

## Install

Requirements:

- Python 3.11 or newer.
- Network access for first-time INSPIRE/ar5iv fetches.
- Codex or Claude Code for host LLM work; unknown hosts fall back to manual
  prompt handoff.
- Optional for `arc-typeset md2pdf`: `pandoc`, `xelatex`, and a CJK-capable
  font such as `Noto Sans CJK SC`.

### Agent Plugin Setup

ARC can be used as a host plugin. The repository contains packaged plugin
directories for Codex and Claude:

```text
packaging/codex/arc
packaging/claude/arc
```

Those plugin packages include the ARC skill, MCP configuration, and wrapper
scripts. The wrapper scripts do not currently build or install the Python
packages automatically. They run `arc-mcp` from `PATH`, or the command pointed
to by `ARC_MCP_COMMAND`.

Use the plugin directly when `arc-mcp` is already installed in the environment
seen by the host. Otherwise, install the Python packages once with the source
install below, then point the plugin at that command:

```bash
export ARC_MCP_COMMAND=/path/to/arc/packages/arc-paper/.venv/bin/arc-mcp
```

Automatic plugin-only bootstrapping would need an additional packaged wheel or
bootstrap script. The current package is intentionally explicit about the
Python install step.

### Source Install

For development, local testing, or a plugin host that does not already have
`arc-mcp` on `PATH`, create one shared virtual environment and install every
package in editable mode:

```bash
git clone <repo-url> arc
cd arc

python3 -m venv packages/arc-paper/.venv
. packages/arc-paper/.venv/bin/activate
python -m pip install --upgrade pip

python -m pip install -e packages/arc-llm[test]
python -m pip install -e packages/arc-paper[test]
python -m pip install -e packages/arc-domain[test]
python -m pip install -e packages/arc-typeset[test]
python -m pip install -e packages/arc-mcp[test]
```

Check the installed commands:

```bash
arc-paper --help
arc-domain --help
arc-llm --help
arc-typeset --help
arc-mcp --help
```

Run a deterministic smoke test:

```bash
arc-paper extract-paper-ids "Compare arXiv:0911.3380 and hep-th/0601001." --json
arc-paper get-title arXiv:0911.3380 --json
```

Convert a Markdown report to PDF:

```bash
arc-typeset md2pdf <report>.md --json
```

Translate a Markdown report to Chinese and automatically convert the
translation to PDF:

```bash
arc-typeset translate <report>.md --json
```

Batch translate project reports when `<name>.md` and `<name>.pdf` appear in
the same folder and `<name>.zh_CN.pdf` is missing:

```bash
arc-typeset batch-translate <project-dir> --json
```

The same converter is available from MCP as `md2pdf`.
The MCP `md2pdf`, `translate`, and `batch_translate` tools always start
background jobs and return a `job_id` immediately; use `job_status`/`job_result`
or `arc-mcp watch <job_id> --json` to inspect completion.

## Configure LLM Providers

ARC uses built-in host providers.

Built-in host providers:

- Codex: `codex-cli`
- Claude Code: `claude-cli`
- Manual fallback: `manual`

Check what ARC detects:

```bash
arc-llm doctor host
arc-llm doctor provider
arc-llm doctor config
arc-paper doctor host --json
arc-paper doctor provider --json
```

With `--provider auto`, ARC uses only host-native providers: Codex selects
`codex-cli`, Claude Code selects `claude-cli`, and unknown hosts select
`manual`. `arc-llm` does not read provider config files, API-key files, or
URL-based provider definitions. Change run model through the run config/CLI:
`provider` plus `model_tier`, or exact `model` with an explicit built-in
provider.

## Use ARC Through An Agent

For an MCP-capable host, configure an MCP server named `arc` that runs
`arc-mcp`. If the host does not inherit your activated virtual environment, use
the full path to the installed command:

```json
{
  "mcpServers": {
    "arc": {
      "command": "/path/to/arc/packages/arc-paper/.venv/bin/arc-mcp",
      "args": []
    }
  }
}
```

Packaged host adapters are included:

- Codex package: `packaging/codex/arc`
- Claude package: `packaging/claude/arc`

Their `.mcp.json` files call host-specific wrapper scripts:

```text
packaging/codex/arc/scripts/arc-mcp-codex
packaging/claude/arc/scripts/arc-mcp-claude
```

The wrappers set `ARC_AGENT_HOST` before executing `arc-mcp`.

When using the ARC skill, ask the agent in research terms. Examples:

```text
Use ARC to summarize arXiv:0911.3380.
Use ARC to build a domain for arXiv:0911.3380 focused on quasi-single-field inflation observables.
Use ARC to develop ideas about cosmological collider scalar exchange.
Use ARC to plan and execute the task to be planned.
```

ARC workflows use two automation modes:

- `auto`: continue with safe defaults, while preserving visible warnings.
- `interactive`: ask for confirmation after major workflow steps.

If you do not specify a mode and the choice matters, the skill asks once.

## Use ARC From The CLI

The CLI is useful for direct paper checks, scripting, debugging, and working
without an MCP host.

### Paper Metadata And Full Text

```bash
arc-paper get-metadata arXiv:0911.3380 --json
arc-paper get-references arXiv:0911.3380 --enrich --json
arc-paper get-citers arXiv:0911.3380 --limit 1000 --sort mostrecent --json
arc-paper get-citers arXiv:0911.3380 --limit 1000 --sort mostcited --json
arc-paper get-citer-count arXiv:0911.3380 --json
arc-paper get-toc arXiv:0911.3380 --json
arc-paper get-section arXiv:0911.3380 --section S2 --json
arc-paper search-full-text arXiv:0911.3380 --query "bispectrum" --context 1 --json
arc-paper get-equation-context arXiv:0911.3380 --query "f_NL" --json
```

Paper IDs can be written as new arXiv IDs, old arXiv IDs, INSPIRE record IDs,
or DOI IDs:

```text
0911.3380
arXiv:0911.3380
hep-th/0601001
inspire:837197
doi:10.1088/1475-7516/2010/04/027
```

### Paper Summaries

Use `llm-summary` to read a cached summary or generate one when an LLM provider
is available:

```bash
arc-paper llm-summary arXiv:0911.3380 --provider auto --json
```

Use `llm-generate-summary` when you explicitly want to regenerate or choose a
provider/model:

```bash
arc-paper llm-generate-summary arXiv:0911.3380 --provider auto --json
```

If no runnable LLM provider is available, ARC returns a `needs_llm` task with
the prompt, input pack, and schema. Generate schema-valid JSON separately and
store it:

```bash
arc-paper store-llm-summary arXiv:0911.3380 --summary-json summary.json --json
```

### Summary Batches

For many papers, put one paper ID per line in a text file:

```bash
arc-paper summary-batch create papers.txt --name qft-summaries --json
arc-paper summary-batch prefetch qft-summaries --workers 8 --json
arc-paper summary-batch run qft-summaries --provider auto --concurrency 2 --max-items 10 --json
arc-paper summary-batch status qft-summaries --json
arc-paper summary-batch run qft-summaries --provider auto --concurrency 2 --json
arc-paper summary-batch export qft-summaries --format jsonl --output summaries.jsonl --json
```

Review a small chunk before launching a large batch.

### Research Domains

A domain is a cached package built from a seed paper plus optional intent. It
contains foundation selection, selected papers, citation graph data, an HTML
network, an evidence pack, and a compact field briefing.

```bash
arc-domain llm-build arXiv:0911.3380 \
  --intent "quasi-single-field inflation observables" \
  --provider auto \
  --json

arc-domain status arXiv:0911.3380 \
  --intent "quasi-single-field inflation observables" \
  --json

arc-domain get-summary arXiv:0911.3380 \
  --intent "quasi-single-field inflation observables" \
  --json

arc-domain get-graph arXiv:0911.3380 \
  --intent "quasi-single-field inflation observables" \
  --json
```

Use the exact same intent string when reading the cache. Different intent
strings produce different domain IDs.

### Direct LLM Checks

Most users should call `arc-paper`, `arc-domain`, or MCP tools instead of
calling `arc-llm` directly. Direct LLM calls are useful for diagnosis:

```bash
arc-llm run-text --prompt "Say hello." --provider auto
arc-llm run-json --prompt "Return {\"ok\": true}" --provider auto --json
```

## MCP Tools And Background Jobs

ARC MCP exposes paper tools, domain tools, job tools, and doctor tools. Tools
that may invoke a host LLM use the `llm_` prefix.

Paper tools:

```text
extract_paper_ids
paper_ids_safe_dir_name
llm_infer_main_references
get_title
get_abstract
get_authors
get_metadata
get_references
get_citers
get_citer_count
get_toc
get_section
search_full_text
get_equation_context
llm_get_summary
llm_generate_summary
store_llm_summary
summary_batch_create
summary_batch_prefetch
llm_summary_batch_run
summary_batch_status
summary_batch_export
summary_batch_retry_failed
```

Domain tools:

```text
llm_domain_build
llm_domain_get_summary
llm_domain_get_graph
domain_status
domain_get_summary
domain_get_graph
```

Job and doctor tools:

```text
job_status
job_result
list_jobs
cancel_job
doctor_host
doctor_provider
doctor_cache
```

Long-running MCP calls can return a `job_id`. Use the CLI watcher to block
until a terminal result:

```bash
arc-mcp watch <job_id> --json
arc-mcp watch <job_id> --progress-jsonl
arc-mcp root --json
arc-mcp status <job_id> --json
arc-mcp result <job_id> --json
arc-mcp list --json
arc-mcp cancel <job_id> --json
```

For slow tools or large launches, pass `background=true` from MCP so the tool
returns immediately with a job ID. Do not cancel jobs unless you explicitly no
longer want the result.

## End-To-End Research Workflows

The `skills/arc` layer turns the package commands into user-facing research
workflows. It writes a project directory with `context.json` and durable
artifacts so results can be inspected and resumed.

### 1. Build Domain References

Input: a seed paper and optional intent.

Output includes:

```text
<project-dir>/context.json
<project-dir>/domain/<seed-safe>_domain.html
<project-dir>/domain/<seed-safe>_domain_summary.json
<project-dir>/domain/<seed-safe>_domain_summary.md
<project-dir>/domain/foundation_<foundation-safe>.md
```

Use this when you need a reliable overview of a local research area before
asking for ideas or calculations.

### 2. Ideas

Input: a not-yet-explicit research request plus built domain context.

The normal workflow feeds ARC-built domain Markdown to proposers. The no-info
workflow intentionally withholds ARC domain context for comparison. Both use
reviewer marks and write a ranked task-to-be-planned candidate report:

```text
<project-dir>/ideas/<run-id>/
<project-dir>/ideas/<run-id>/ranked-ideas.md
<project-dir>/ranked-ideas.md
```

The report starts with a compact marked summary for each candidate, then
appends one detail section per idea with all round-by-round referee marks and
selected handoff text: title, idea summary, and calculation plan. It should not
invent novelty claims or hide failed idea history.

### 3. Plan And Execute A Calculation

Input: one task to be planned, such as an explicit calculation idea or a
source-extracted request.

The calculation workflow starts with two phases, then may loop back from
`calculate` to `plan` when a deferred macro block or blocked step needs
expansion:

1. `plan`: gather evidence, write or update `work-note.md`, promote accepted
   premises, define ready-step boundaries, and maintain rough later steps.
2. `calculate`: record current-step result/status, write planning requests
   when plan or foundation material must change, and execute current detailed
   steps through the calculate workflow runner and proposer-reviewer loops.

Primary outputs:

```text
<project-dir>/work-note.md
<project-dir>/calculate/<run-id>/work-notes/work-note-v001.md
<project-dir>/calculate/<run-id>/work-notes/work-note-v002.md
<project-dir>/calculate/<run-id>/execute/calculate.config.json
<project-dir>/calculate/<run-id>/execute/<calculate-run-id>/
```

`work-note.md` is the human and agent source of truth. It contains notation,
axioms, accepted derived results, ready detailed steps, rough later steps,
calculation status, open questions, revision history, journal, and source audit
trail. Main text explains physics and equation logic; journal records execution
events and human resolutions. Runtime JSON is generated only to drive CLI
execution.

The workflow is deliberately conservative: it requires source evidence,
explicit quantity contracts, independent agreement checks, and recorded
validation history before accepting results.

## Caches And Refreshing

ARC is cache-first. Repeated calls usually read local JSON/HTML artifacts
instead of refetching data or rerunning LLM work.

Default checkout cache paths:

```text
cache/arc-paper/
cache/arc-domain/
cache/arc-mcp/
```

Outside this checkout, ARC uses:

```text
~/.cache/arc/arc-paper/
~/.cache/arc/arc-domain/
~/.cache/arc/arc-mcp/
```

Set these environment variables to override cache locations:

```bash
export ARC_PAPER_CACHE=/path/to/arc-paper-cache
export ARC_DOMAIN_CACHE=/path/to/arc-domain-cache
export ARC_MCP_CACHE=/path/to/arc-mcp-cache
```

Use `--refresh` only when you intentionally want fresh source data or a forced
rebuild:

```bash
arc-paper get-metadata arXiv:0911.3380 --refresh --json
arc-domain llm-build arXiv:0911.3380 --intent "..." --refresh --json
```

Diagnose cache state:

```bash
arc-paper doctor cache arXiv:0911.3380 --json
arc-mcp root --json
```

Useful environment variables:

```text
ARC_AGENT_HOST                    Force host detection, for example codex or claude-code.
ARC_PAPER_CACHE                   Override the arc-paper cache root.
ARC_DOMAIN_CACHE                  Override the arc-domain cache root.
ARC_MCP_CACHE                     Override the arc-mcp job/cache root.
XDG_CACHE_HOME                    Base cache directory when ARC-specific cache vars are unset.
ARC_MCP_INLINE_WAIT_SEC           Inline MCP wait before returning a background job.
ARC_MCP_TOOL_TIMEOUT_SEC          Host MCP tool timeout used to derive inline wait.
ARC_MCP_BACKGROUND_MARGIN_SEC     Safety margin subtracted from the MCP tool timeout.
```

## Troubleshooting

If a paper query fails:

```bash
arc-paper extract-paper-ids "<your input>" --json
arc-paper doctor cache <paper-id> --json
arc-paper get-metadata <paper-id> --refresh --json
```

If LLM generation is unavailable:

```bash
arc-llm doctor host
arc-llm doctor provider
```

If an MCP call returns a job ID:

```bash
arc-mcp watch <job_id> --json
```

If a domain summary or graph is missing:

```bash
arc-domain status <seed-paper> --intent "<same-intent>" --json
arc-domain llm-build <seed-paper> --intent "<same-intent>" --json
```

Network integration tests are opt-in because they call external services:

```bash
ARC_RUN_NET_TESTS=1 packages/arc-paper/.venv/bin/python -m pytest tests/integration -q
```

True LLM integration tests are also opt-in:

```bash
ARC_RUN_LLM_TESTS=1 ARC_RUN_NET_TESTS=1 \
  packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests/test_proposers_reviewer_llm_integration.py -q
```

## Developer Notes

This repository is organized as Python packages plus thin agent adapters.
`0_ref/` is read-only reference material and must not be modified.

Package boundaries:

- `packages/arc-llm` owns reusable host LLM execution: host detection,
  provider selection, model defaults, direct prompt calls, and
  proposers-reviewer runners.
- `packages/arc-paper` owns deterministic paper data access, ID normalization,
  cache layout, ar5iv parsing, INSPIRE access, paper-summary contracts,
  paper-summary orchestration, full-text search, and summary batches.
- `packages/arc-domain` owns research-domain construction from seed papers:
  foundation selection, domain paper selection, graph artifacts, evidence
  packs, HTML rendering, and domain summaries. It calls `arc-paper` for
  single-paper work and `arc-llm` for LLM work.
- `packages/arc-mcp` stays a thin MCP adapter over package service functions
  and background-job management.
- `skills/arc`, `packaging/`, prompts, and schemas describe or wrap package
  behavior; they should not reimplement package internals.

Development rules:

- Keep ARC general-purpose across theoretical-physics domains. Do not hard-code
  seed papers, author names, subfield labels, or field-specific keyword lists.
- Apply the instruction review gate before changing ARC instructions,
  workflows, prompts, schemas, tests, package behavior, MCP tools, packaging
  metadata, or durable documentation. Changes should be portable across
  supported hosts and compatible with ARC's general-purpose research goals.
- Keep agent instructions portable across Codex, Claude Code, Cursor, GitHub
  Copilot, and similar hosts. Use generic terms such as agent, host, skill
  directory, MCP server, and workflow unless a file is host-specific.
- Keep skills concise. Put detailed workflows and troubleshooting in reference
  files.
- Unit tests must not require network access. Use `ARC_RUN_NET_TESTS=1` only
  for explicit network integration runs.
- Keep `0_ref/` read-only. Treat `cache/` and `arc-tests/` as generated,
  reference, or local-test material rather than normal source edits.
- Durable docs, skills, prompts, schemas, comments, package metadata, and
  workflow files should be written in English unless there is a specific reason
  to do otherwise.

Focused test command:

```bash
packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests \
  packages/arc-paper/tests \
  packages/arc-domain/tests \
  packages/arc-mcp/tests
```

Full local suite used by this checkout:

```bash
packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests \
  packages/arc-paper/tests \
  packages/arc-domain/tests \
  packages/arc-mcp/tests \
  tests -q
```

When changing packaged skills or workflows, keep source and packaged copies in
sync. The repository tests check that key files under `skills/arc` match the
Codex and Claude packaged copies.

Useful docs/packaging check:

```bash
packages/arc-paper/.venv/bin/python -m pytest tests/test_arc_research_workflow_docs.py -q
```
