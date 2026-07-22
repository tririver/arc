# ARC

Agent Research Copilot (ARC) is an angentic research toolkit for theoretical physics knowledge domain construction, idea generation and calculation workflows. It works as a plugin of coding agents such as Codex / Claude Code, with the strength of bringing coding agents into a research context, and generating publication-level ideas in theoretical research.

ARC is CLI-first. Its workflow Skill uses six core Python commands; a seventh,
separately installed adapter exposes the same services through optional MCP:

- `arc-paper`: paper metadata, references, citers, ar5iv sections, equation
  context, full-text search, LLM paper summaries, and paper-summary batches.
- `arc-domain`: builds a cached research-domain package from a seed paper and
  optional scientific intent.
- `arc-llm`: reusable host LLM execution, provider selection, and
  proposers-reviewer workflows.
- `arc-typeset`: deterministic typesetting utilities, including Markdown to PDF
  conversion through Pandoc and XeLaTeX.
- `arc-companion`: builds source-faithful, chapter-aware PDF and static-web
  original/translation/commentary readers for papers, lecture notes, and books
  from a paired rich source and PDF.
- `arc-jobs`: protocol-neutral persistent background execution for ARC CLIs.
- `arc-mcp` (optional): exposes ARC services to MCP clients; it delegates
  background work to `arc-jobs`.
- `plugins/arc/skills/arc`: agent-facing workflow instructions for domain
  building, idea generation, and research calculations.

## Who This Is For

Use ARC when you want to:

- Look up reliable paper metadata, references, citers, sections, or equations.
- Summarize a paper from cached ar5iv/INSPIRE data.
- Generate Chinese-by-default companion-reading PDF and static-web readers with
  chapter guides, a unified glossary, and an original/translation/commentary
  sequence while retaining source equations, figures, tables, links, and
  bibliography.
- Build a research-domain overview from a seed paper.
- Generate ideas using domain context and reviewer scoring.
- Plan and execute a careful symbolic or numerical research calculation with
  explicit provenance and checks.

Deterministic paper queries do not need an LLM. Paper summaries, domain
briefings, idea loops, and calculation workflow runners need a host LLM
provider.

## Install

### Remarks:

- Permission: the same as many heavy skills/plugins, ARC will need permissions to run Python scripts. Accepting permissions could be annoying. We recommend installing ARC within docker or a virtual machine, and allow all permissions in that virtual environment. As always for working with AI agents, be aware of risk to your data and system. 

- Token usage. As measured using Claude + DeepSeek, a typical run of domain build + idea generation consumes about 1M uncached input tokens, and 0.5M output tokens, in about an hour's running time. The token usage may vary depending on the specific tasks and LLM used. Be aware of token usage and costs. 

- If ARC has played a role in your research, please consider citing the ARC manual.

### Citation

Yanjiao Ma, Yi Wang, and Xingkai Zhang. _ARC: An LLM-Native Agent
Workflow for Theoretical Physics Research_. ChinaXiv:202606.00234, 2026.
https://chinaxiv.org/abs/202606.00234

```bibtex
@misc{ma2026arc,
  title         = {{ARC}: An {LLM}-Native Agent Workflow for Theoretical Physics Research},
  author        = {Ma, Yanjiao and Wang, Yi and Zhang, Xingkai},
  year          = {2026},
  month         = jun,
  publisher     = {ChinaXiv},
  eprint        = {202606.00234},
  archivePrefix = {ChinaXiv},
  url           = {https://chinaxiv.org/abs/202606.00234},
  note          = {Version 1}
}
```

### Requirements:

- Python 3.11 or newer.
- `uv` for fast first-time CLI runtime setup; Python `venv` + `pip` is the
  fallback.
- Network access for first-time INSPIRE/ar5iv fetches.
- Codex or Claude Code for host LLM work; unknown hosts fall back to manual
  prompt handoff.
- Optional for `arc-typeset md2pdf`: `pandoc`, `xelatex`, and a CJK-capable
  font such as `Noto Sans CJK SC`.
- For `arc-companion build`: `latexmk`, `xelatex`, Poppler command-line tools,
  and fonts covering the source and annotation languages.

### Agent Plugin Setup

ARC can be installed as a host plugin from this repository. `plugins/arc/` is
the plugin root for both Codex and Claude Code, and
`plugins/arc/skills/arc/` is the single canonical Skill source. The base plugin
contains no MCP manifest or MCP dependency.

Install for Codex (run in shell, or in Codex with `!` prefix):

```bash
codex plugin marketplace add tririver/arc --ref stable
codex plugin add arc@arc
```

Install for Claude Code (run in Claude Code):

```bash
/plugin marketplace add tririver/arc@stable
/plugin install arc
```

The plugin exposes `arc-paper`, `arc-domain`, `arc-llm`, `arc-typeset`,
`arc-companion`, `arc-jobs`, and `arc-runtime`. On first CLI use,
`arc-runtime` installs the immutable ARC release into an isolated core profile
under `~/.codex/arc/runtimes`. It never performs a global `pip install`.

Prewarm or diagnose the core runtime when needed:

```bash
plugins/arc/bin/arc-runtime setup --profile core
plugins/arc/bin/arc-runtime doctor --profile core
```

If that install fails, later starts fail fast with the saved log path and a
short log tail instead of repeatedly retrying a broken partial install. After
fixing the cause, run `arc-runtime setup --profile core --retry`. Marketplace
installs prefer the host-recorded full commit SHA and otherwise use the bundled
immutable `vX.Y.Z` tag; mutable refs such as `main` and `stable` are rejected.
Source checkouts use local `packages/` automatically. `ARC_INSTALL_REPO_ROOT`
and `ARC_INSTALL_SOURCE=local` select another development checkout.

### Standalone Skill Setup

Hosts without plugin support may install or copy `plugins/arc/skills/arc/` as
an Agent Skill. The Skill carries the same launcher and pinned constraints. If
the ARC commands are not on `PATH`, use:

```bash
<skill-dir>/scripts/arc-runtime arc-paper --help
<skill-dir>/scripts/arc-runtime setup --profile core
```

There is intentionally no install-time hook: the first real CLI call performs
the audited, isolated setup.

### Optional MCP Companion

Install MCP only when it is explicitly needed. The separate `arc-mcp` plugin is
self-contained and is not a dependency of the base plugin:

```bash
codex plugin add arc-mcp@arc
```

```text
/plugin install arc-mcp
```

It owns the MCP manifest and an isolated `mcp` runtime profile. MCP startup or
configuration failures therefore cannot affect the default Skill/CLI install.
Prewarm it with `plugins/arc-mcp/bin/arc-runtime setup --profile mcp`.

Development benchmarks that must not fall back to an installed or cached ARC
copy can set `ARC_REQUIRE_REPO_ROOT` to the checkout root. Workflow scripts then
prepend that checkout's package sources, verify module origins, and fail before
LLM work if any ARC module or workflow file comes from another installation.
First run `<skill-dir>/scripts/arc-runtime setup --profile core`, then use
`python3 <skill-dir>/workflows/scripts/verify-source-runtime.py --repo-root
<checkout> --output <record.json>` to capture module, Git working-tree, and
workflow-file provenance. A verifier launched by system Python re-executes
with the installed core runtime Python before loading checkout sources.

Check the launcher directly from a source checkout:

```bash
plugins/arc/bin/arc-paper --help
plugins/arc/bin/arc-jobs --help
plugins/arc/bin/arc-runtime doctor --profile core
```

Use the source install below only for development or local package testing.

### Release Process

ARC releases use explicit versions in Python package metadata and plugin
manifests. GitHub tags or releases do not update those files automatically.
Run the release helper from a clean checkout on the release branch:

```bash
scripts/release-arc.sh 0.2.0
```

The helper checks that the branch is not behind its upstream, that committed
changes exist since the latest `v*` release tag, and that the target tag does
not already exist. It then pauses for Enter before each mutating step, bumps
ARC package/plugin versions, commits the bump, creates `vX.Y.Z` and
performs push dry-runs, pushes the branch and tag, and
moves `stable` to the release commit.

If you abort after changing version files, after the version bump commit, or
after creating the local release tag, rerun the same command. The helper allows
dirty version-file-only resumes, skips the bump commit when the committed files
already match the requested version, and reuses a local `vX.Y.Z` tag that
already points at `HEAD`.

After the script succeeds, create the human-facing GitHub Release from the
`vX.Y.Z` tag. Marketplace users who should track stable releases should add ARC
with the stable ref:

```bash
codex plugin marketplace add tririver/arc --ref stable
claude plugin marketplace add tririver/arc@stable
```

### Source Install

For development and local testing, create one shared virtual environment and
install every package in editable mode:

```bash
git clone <repo-url> arc
cd arc

python3 -m venv "$HOME/.virtualenvs/arc-dev"
. "$HOME/.virtualenvs/arc-dev/bin/activate"
python -m pip install --upgrade pip

python -m pip install -e packages/arc-jobs[test]
python -m pip install -e packages/arc-llm[test]
python -m pip install -e packages/arc-paper[test]
python -m pip install -e packages/arc-domain[test]
python -m pip install -e packages/arc-typeset[test]
python -m pip install -e packages/arc-companion[test]
# Optional MCP development only:
python -m pip install -e packages/arc-mcp[test]
```

Check the installed commands:

```bash
arc-paper --help
arc-domain --help
arc-llm --help
arc-typeset --help
arc-companion --help
arc-jobs --help
# Optional:
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

Build or resume companion-reading PDF and static-web readers for a paper,
lecture note, or book. A
formal build pairs a rich Markdown/TeX/HTML source with its PDF: the rich source
provides faithful content and the PDF is authoritative for hierarchy, printed
pages, and chapter-boundary reconciliation. If `--annotation-language` is
omitted, ARC prints a language-switch notice and continues in Chinese:

```bash
arc-companion build arXiv:0911.3380 \
  --project-dir ./arc-tests/companion/0911.3380 --json
arc-companion render-web \
  --project-dir ./arc-tests/companion/0911.3380 --json
arc-companion validate \
  --project-dir ./arc-tests/companion/0911.3380 --json
```

Use `--document-kind auto|article|book` to select the structure policy,
`--idle-timeout-seconds` to override provider inactivity timeout, and
`--regenerate-commentary` to rebuild commentary while retaining reusable
translations. `--stop-after-first-chapter` provides the first-chapter
validation checkpoint for `interactive` runs or a one-time review gate; it
does not permanently change the managed run's automation level.

For a managed companion workflow, the executing agent inspects substantive
source body text near the beginning, middle, and end before building. It
compares normalized base languages (`EN_US`, `EN_UK`, and `en-GB` are `en`;
simplified and traditional Chinese are `zh`) and records the source judgment,
target language, translation mode, and reason in `context.json`. If all samples
clearly match the target base language, the agent passes `--skip-translation`;
mixed or uncertain text keeps translation. The CLI itself does not perform
language detection, so the workflow also passes the sampled canonical tag with
`--source-language`.

ARC reconciles PDF structure with rich-source blocks and validates exact
chapter and segment coverage. Chapters may prepare concurrently under one
global `workers` budget. Within each chapter, semantic segmentation and a
stateful guide prepare independent, source-ordered translation and commentary
sessions. A bootstrap turn carries fixed chapter context; later turns carry
only the current segment, cursor, source hash, terms, and bounded sources. Stable
idempotency keys and accepted-block ledgers make routine resume automatic
without repeating accepted provider calls.

During generation, each accepted segment atomically refreshes the static reader
snapshot and its hashed HTML/JavaScript asset bundle. The last complete bundle
remains readable if a later refresh fails. `arc-companion render-web` rebuilds
the reader manually from durable project checkpoints without repeating LLM
work. The reader has no server dependency and vendors KaTeX, fonts, style,
script, and media assets locally, so opening the completed HTML does not require
network access.

A translated reader places its glossary once at the end of the same
`index.html`, reachable from the sidebar and `#glossary`; large Index-based
glossaries mount only when opened or approached. Matching source terms,
aliases, and translations in original, translation, and commentary text use a
subtle blue-gray accessible tooltip. Math and links remain untouched. The PDF
likewise places a translated glossary after the references.

With `--skip-translation`, ARC runs guide, segmentation, commentary, review,
rendering, and validation without creating or reusing any translation or
glossary session, call, ledger, checkpoint, projection, prompt context,
artifact, or reader layer. Existing glossary cache files remain available for
a later translated build but are invisible to the current output. A source
book's own Index remains source content rather than a bilingual glossary. The
default remains the two-lane translation-and-commentary build.

A real Index becomes the complete global glossary when translation is enabled,
including nested entries,
page ranges, `see`, and `see also`; it is never truncated. Documents without an
Index use the page-scaled 50/100/200 terminology limits. Each segment receives
only its deterministic source-term projection. Commentary agents may search,
inspect, and directly cite external sources in the same generation turn; ARC
validates the returned title, HTTP(S) URL, and reader-understandable locator
without a separate evidence-controller rewrite pass. Companion workers do not
depend on MCP or project-file reading.

`--recovery-policy auto|manual` controls blocked-call recovery and defaults to
`auto`. Automatic recovery replays durable responses, attempts native session
reconciliation, and may start one replacement generation for an eligible
translation or commentary lane suffix, up to three times by default. Set
`--max-auto-replacements N` to change the persisted recovery budget without
changing content fingerprints. Use repeatable
`--regenerate-segment LANE:SEGMENT_ID` for precise translation/commentary
regeneration. Bare `resume` selects automatic
recovery; choose an explicit action for strict/manual behavior:

```bash
arc-companion resume --project-dir ./arc-tests/companion/0911.3380 --json
arc-companion resume --project-dir ./arc-tests/companion/0911.3380 \
  --action resume-native --json
arc-companion resume --project-dir ./arc-tests/companion/0911.3380 \
  --action restart-generation --confirm-possible-duplicate-charge --json
```

`--stop-after-first-chapter` schedules no later chapter and returns
`first_chapter_ready` only after the first substantive chapter passes guide,
all enabled lanes, review, PDF rendering, static-web publication, and
validation. Long background
builds emit a build-level `review_due` at the next safe boundary after each 30
minutes of cumulative runtime; `arc-jobs watch <job-id> --until-review --json`
returns for inspection without pausing the job.

Each chapter guide appears once after its title. Every segment renders original,
translation, then commentary without visible controller layer labels. Text uses
sans-serif fonts while mathematics remains LaTeX serif. Personal names retain
their exact source spelling in any script. Document and structural headings,
including References and Index, render as source title plus translation but do
not receive commentary; navigation prefers the translated title. Figure/table
captions remain source-only. The deliverables are the validated full-document
PDF and its static-web reader. A successful full build maintains a byte-identical
run-root delivery PDF directly in the resolved `--project-dir`, never its
parent; the immutable internal `output_pdf` remains authoritative. Ordinary
non-JSON CLI output prints the run-root delivery PDF path first, while JSON
records it as `output_run_pdf` and `output_run_pdf_sha256`.
`arc-companion validate` verifies both forms, while a reproducibility ZIP that
contains both plus every manifest-declared local web asset is generated only by
an explicit `arc-companion package` request.

Run slow conversion through `arc-jobs` when the caller should not block. The
optional MCP adapter exposes the same `md2pdf`, `translate`, and
`batch_translate` operations and delegates their background work to
`arc-jobs`.

## Configure LLM Providers

ARC uses built-in host providers.

Built-in host providers:

- Codex: `codex-cli`
- Claude Code: `claude-cli`
- Kimi Code: `kimi-code-cli` (experimental; Kimi Code CLI `>=0.28.0`)
- Manual fallback: `manual`

The Kimi provider requires the Node.js/TypeScript `@moonshot-ai/kimi-code`
CLI and an existing login created with `kimi login`. ARC talks to `kimi acp`
over stdin/stdout; it does not use `kimi -p`, add an OpenAI-compatible API
provider, or read and manage Kimi credentials itself.

Check what ARC detects:

```bash
arc-llm doctor host --json
arc-llm doctor provider --json
arc-llm doctor config --json
arc-paper doctor host --json
arc-paper doctor provider --json
```

With `--provider auto`, ARC uses only host-native providers: Codex selects
`codex-cli`, Claude Code selects `claude-cli`, Kimi Code selects
`kimi-code-cli`, and unknown hosts select `manual`. Kimi detection uses
`ARC_AGENT_HOST=kimi-code`, the `@moonshot-ai/kimi-code` package name, or a
reliable `kimi` parent-process signal. An explicit `--provider kimi-code-cli`
works under other hosts. ARC does not read URL-based provider definitions or
Kimi credential values, but the Kimi subprocess inherits the user's Kimi Code
home, authentication, configuration, and persistent sessions. Change the run
model through the run config/CLI: `provider` plus `model_tier`, or exact
`model` with an explicit built-in provider.

`kimi-code-cli` is experimental. Before its first call ARC warns:

> `kimi-code-cli is experimental and inherits Kimi Code configuration, instructions, skills, hooks, plugins, MCP, tool permissions, and persistent sessions; it may access the network, run commands, and modify files.`

ARC denies ACP permission and filesystem reverse requests, but that is not a
sandbox: Kimi automation, hooks, plugins, MCP servers, and local tools may act
outside those reverse requests. Review `arc-llm doctor provider` and
`arc-llm doctor config` before use. Kimi does not report token usage through
this ACP integration, so usage fields remain null.

## Use ARC Through An Agent

Codex and Claude Code can install the CLI-first repository plugin directly with
the marketplace commands in the install section. It loads the ARC Skill and
invokes CLI commands without registering an MCP server. Install the separate
`arc-mcp` marketplace entry only for hosts or workflows that explicitly need
MCP discovery.

When using the ARC skill, ask the agent in research terms. Examples:

```text
Use ARC to summarize arXiv:0911.3380.
Use ARC to build a domain for arXiv:0911.3380 focused on quasi-single-field inflation observables.
Use ARC to develop ideas about cosmological collider scalar exchange.
Use ARC to plan and execute the task to be planned.
```

Managed ARC workflows use two automation modes:

- `auto`: continue with safe defaults, while preserving visible warnings.
- `interactive`: pause at workflow-defined major milestones.

The skill defaults to `auto` without presenting a startup menu. Ask explicitly
for manual, step-by-step, staged review, or key-step confirmation to use
`interactive`; asking to discuss first creates a pre-run pause. You can steer a
managed run in either direction while it is active, with the change taking
effect at the next safe boundary. Automatic execution stays within the exact
requested scope: a domain-only request stops after the domain, and an ideas-only
request stops after ranked ideas. Direct ARC tool tasks also default to auto.
Neither mode bypasses authorization, safety, duplicate-charge, scientific, or
error-recovery gates.

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

Most users should call `arc-paper` or `arc-domain` instead of
calling `arc-llm` directly. Direct LLM calls are useful for diagnosis:

```bash
arc-llm run-text --prompt-text "Say hello." --provider auto
arc-llm run-json --prompt-text "Return {\"ok\": true}" --provider auto --json
```

Use `--prompt-text` for literal text and `--prompt-file` for a UTF-8 prompt
file (`--prompt-file -` reads stdin). The legacy `--prompt` flag remains a
file/stdin alias for compatibility.

Direct `arc-llm run-*` calls are stateless unless `--session-policy stateful`
is paired with a session root and session key. For Kimi, stateless means ARC
does not reuse the native session ID; Kimi Code still uses provider-side
persistence for its own session. ARC does not copy, migrate, or delete Kimi sessions.
Proposers-reviewer workflows use stateful delta sessions by default and write
cache/session audit data under the run artifacts.

Custom `json_runner` wrappers must explicitly declare `session_policy`,
`session_manager`, `session_key`, `artifact_dir`, `call_label`, and
`static_prefix` to receive stateful session reuse. A bare `**kwargs` wrapper is
treated as legacy/stateless by design.

## Background Jobs

Use `arc-jobs` for long-running CLI commands. It persists state and output,
executes an allowlisted ARC argv without a shell, and works whether or not MCP
is installed:

```bash
arc-jobs submit --job-type domain_build --cwd <project-dir> --json -- \
  arc-domain llm-build <seed-paper> --intent "<intent>" --json
arc-jobs watch <job-id> --json
arc-jobs result <job-id> --json
```

## Optional MCP Tools

The separately installed ARC MCP companion exposes paper tools, domain tools,
job tools, and doctor tools. Tools
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

Long-running MCP calls can return a `job_id`. Use `arc-jobs` to block
until a terminal result. Prefer the returned `next.cli_command`:

```bash
arc-jobs watch <job_id> --json
arc-jobs watch <job_id> --progress-jsonl
arc-jobs status <job_id> --json
arc-jobs result <job_id> --json
arc-jobs list --json
arc-jobs cancel <job_id> --json
```

For slow tools or large launches, pass `background=true` from MCP so the tool
returns immediately with a job ID. Do not cancel jobs unless you explicitly no
longer want the result.

## End-To-End Research Workflows

The `plugins/arc/skills/arc` layer turns the package commands into
user-facing research workflows. It writes a project directory with
`context.json` and durable artifacts so results can be inspected and resumed.

Generated workflow project directories are a direct child of the directory where
the agent command was launched: `<launch-cwd>/<safe-dir-name>/context.json`.
They are not under host-internal directories such as `.claude/projects` and are
not wrapped in `arc-output/`.

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

The release idea workflow feeds ARC-built domain Markdown to proposers. It uses
reviewer marks and writes a ranked task-to-be-planned candidate report:

```text
<project-dir>/ideas/<run-id>/
<project-dir>/ideas/<run-id>/ranked-ideas.md
<project-dir>/ranked-ideas.md
```

The report starts with a compact marked summary for each candidate, then
appends one detail section per idea with all round-by-round referee marks and
selected handoff text: title, idea summary, and calculation plan. It should not
invent novelty claims or hide failed idea history.

The no-info variant is disabled by default and kept as an opt-in test fixture
for workflow development.

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

Inside a source checkout, ARC writes generated cache files under:

```text
cache/arc-paper/
cache/arc-domain/
cache/arc-jobs/
```

Outside a source checkout, ARC uses the user cache directory:

```text
~/.cache/arc/arc-paper/
~/.cache/arc/arc-domain/
~/.cache/arc/arc-jobs/
```

Set these environment variables to override cache locations:

```bash
export ARC_PAPER_CACHE=/path/to/arc-paper-cache
export ARC_DOMAIN_CACHE=/path/to/arc-domain-cache
export ARC_JOBS_CACHE=/path/to/arc-jobs-cache
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
arc-jobs list --json
arc-runtime doctor --profile core
```

Useful environment variables:

```text
ARC_AGENT_HOST                    Force host detection, for example codex, claude-code, or kimi-code.
ARC_LLM_IDLE_TIMEOUT_SECONDS      General no-substantive-output timeout (default 1800 seconds).
ARC_CODEX_IDLE_TIMEOUT_SECONDS    Codex idle timeout; overrides the general idle timeout.
ARC_CLAUDE_IDLE_TIMEOUT_SECONDS   Claude idle timeout; overrides the general idle timeout.
ARC_KIMI_BIN                      Kimi Code CLI executable (default kimi).
ARC_KIMI_WORK_DIR                 Working directory for new Kimi ACP sessions (default current directory).
ARC_KIMI_IDLE_TIMEOUT_SECONDS     Kimi idle timeout; overrides the general idle timeout.
ARC_LLM_KIMI_LOW_MODEL            Kimi model alias for the low tier.
ARC_LLM_KIMI_MEDIUM_MODEL         Kimi model alias for the medium tier.
ARC_LLM_KIMI_HIGH_MODEL           Kimi model alias for the high tier.
ARC_LLM_KIMI_MAX_MODEL            Kimi model alias for the max tier.
ARC_PAPER_CACHE                   Override the arc-paper cache root.
ARC_DOMAIN_CACHE                  Override the arc-domain cache root.
ARC_JOBS_CACHE                    Override the protocol-neutral job/cache root.
ARC_RUNTIME_HOME                  Override private ARC runtime storage (default ~/.codex/arc/runtimes).
ARC_INSTALL_REF                   Override with a full commit SHA or immutable vX.Y.Z tag.
ARC_INSTALL_REPO_ROOT             Select a local development checkout.
ARC_INSTALL_SOURCE                Select auto, local, or git package installation.
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
arc-llm doctor host --json
arc-llm doctor provider --json
```

If an MCP call returns a job ID:

```bash
arc-jobs watch <job_id> --json
```

When using optional MCP, prefer its returned `next.cli_command`; job ownership
and persistence remain protocol-neutral.

If a domain summary or graph is missing:

```bash
arc-domain status <seed-paper> --intent "<same-intent>" --json
arc-domain llm-build <seed-paper> --intent "<same-intent>" --json
```

Network integration tests are opt-in because they call external services:

```bash
ARC_RUN_NET_TESTS=1 python -m pytest tests/integration -q
```

True LLM integration tests are also opt-in:

```bash
ARC_RUN_LLM_TESTS=1 ARC_RUN_NET_TESTS=1 \
  python -m pytest \
  packages/arc-llm/tests/test_cli_smoke_integration.py \
  packages/arc-llm/tests/test_proposers_reviewer_llm_integration.py -q
```

## Developer Notes

This repository is organized as Python packages plus thin agent adapters.

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
- `packages/arc-companion` owns paired-source/PDF chapter orchestration,
  Index-aware glossary projection, ordered stateful translation and commentary
  lanes, bounded source selection, supervised resume, review checkpoints,
  deterministic LaTeX/PDF and static-web rendering, and validation. It consumes
  document and asset caches from `arc-paper` and LLM calls from `arc-llm`.
- `packages/arc-jobs` owns protocol-neutral persistent CLI execution, status,
  cancellation, output capture, and ETA. It has no core package dependency.
- `packages/arc-mcp` stays a thin MCP adapter over package service functions
  and `arc-jobs`.
- `plugins/arc/skills/arc`, prompts, schemas, and plugin manifests describe or
  wrap package behavior; they should not reimplement package internals.

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
- Durable docs, skills, prompts, schemas, comments, package metadata, and
  workflow files should be written in English unless there is a specific reason
  to do otherwise.

Focused test command:

```bash
python -m pytest \
  packages/arc-jobs/tests \
  packages/arc-llm/tests \
  packages/arc-paper/tests \
  packages/arc-domain/tests \
  packages/arc-companion/tests \
  packages/arc-mcp/tests
```

Full local suite used by this checkout:

```bash
python -m pytest \
  packages/arc-jobs/tests \
  packages/arc-llm/tests \
  packages/arc-paper/tests \
  packages/arc-domain/tests \
  packages/arc-companion/tests \
  packages/arc-mcp/tests \
  tests -q
```

When changing packaged skills or workflows, edit
`plugins/arc/skills/arc` only. Codex and Claude load the same plugin skill tree;
there are no packaged skill copies to synchronize.

Useful docs/packaging check:

```bash
python -m pytest tests/test_arc_research_workflow_docs.py -q
```
