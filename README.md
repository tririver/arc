# ARC Dev

ARC research tooling is organized as Python packages plus thin agent adapters.

## Packages

Install the current development packages:

```bash
python3 -m venv packages/arc-paper-query/.venv
. packages/arc-paper-query/.venv/bin/activate
python -m pip install -e packages/arc-llm-worker[test]
python -m pip install -e packages/arc-paper-query[test]
python -m pip install -e packages/arc-domain-info[test]
python -m pip install -e packages/arc-mcp
```

## LLM Worker

Reusable host LLM execution:

```bash
arc-llm-worker doctor config
echo '{"task":"say ok"}' | arc-llm-worker run-json --provider auto
echo 'Say ok.' | arc-llm-worker run-text --provider auto
```

`arc-llm-worker` owns host detection, provider selection, model defaults, and
Codex/Claude CLI prompt execution. Other packages should use it instead of
shelling out to host LLMs directly.

## Paper Query

Deterministic paper data:

```bash
arc-paper-query get-title arXiv:0911.3380 --json
arc-paper-query get-references arXiv:0911.3380 --json
arc-paper-query get-toc arXiv:0911.3380 --json
arc-paper-query get-section arXiv:0911.3380 --section S2 --json
```

LLM summaries:

```bash
arc-paper-query get-llm-summary arXiv:0911.3380 --json
arc-paper-query generate-llm-summary arXiv:0911.3380 --provider auto --json
```

`get-llm-summary` first reads the local summary cache. If the summary is
missing and a host LLM provider is available, it generates and caches the
summary automatically. `generate-llm-summary` remains available when you want an
explicit generation command or provider override.

Summary generation uses fast host defaults unless overridden: Codex uses
`gpt-5.4-mini`, and Claude Code uses `haiku`. The LLM pipeline summarizes paper
sections sequentially first, then synthesizes the final paper summary from the
title, abstract, table of contents, and compact section summaries. The final
JSON keeps `toc` as navigation metadata and stores the richer per-section text
under `section_summaries`; it does not rewrite those into one-sentence TOC
entries. References are intentionally not included in the summary input pack.

When called through MCP, `get_LLM_summary` and `generate_LLM_summary` avoid long
tool calls: if no cached summary is available, they start a background job and
return a `job_id`. Poll `get_LLM_summary_status` with that `job_id` for
completion and progress fields such as `phase`, `sections_completed`,
`sections_total`, `current_section`, and recent `events`. Completed section
summaries are cached as they finish, so a failed or interrupted job can resume
without paying again for sections that already completed. MCP background jobs do
not stream or push a completion notification; clients should poll the status
tool.

Batch workflow:

```bash
arc-paper-query summary-batch create papers.txt --name qft-ideas --json
arc-paper-query summary-batch prefetch qft-ideas --workers 8 --json
arc-paper-query summary-batch run qft-ideas --provider auto --concurrency 2 --max-items 10 --json
arc-paper-query summary-batch status qft-ideas --json
arc-paper-query summary-batch export qft-ideas --format jsonl --output summaries.jsonl --json
```

## Host Detection

Plugins should set:

```text
ARC_AGENT_HOST=codex
ARC_LLM_PROVIDER=codex-cli
```

or:

```text
ARC_AGENT_HOST=claude-code
ARC_LLM_PROVIDER=claude-cli
```

Without plugin env, `arc-paper-query` falls back to parent-process detection.

Debug:

```bash
arc-paper-query doctor host --json
arc-paper-query doctor provider --json
arc-paper-query doctor cache 0911.3380 --json
```

When ARC is run from this checkout without `ARC_PAPER_QUERY_CACHE` or
`XDG_CACHE_HOME`, paper-query data is cached under:

```text
/arc-dev/cache/paper-query/
```

## Domain Info

Build a cached research-domain package from one seed paper plus an optional
intent:

```bash
arc-domain-info build 0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain-info status 0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain-info get-summary 0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain-info get-graph 0911.3380 --intent "quasi-single-field inflation observables" --json
```

`arc-domain-info` uses `arc-paper-query` for all single-paper operations. It
identifies a likely foundation paper, builds a citation-domain graph of up to
about 60 nodes, renders `network.html`, builds an evidence pack from titles,
abstracts, and conclusion/outlook sections, then asks `arc-llm-worker` for a
compact field briefing. If the host LLM is unavailable, deterministic fallback
artifacts are still written so the cache is inspectable.

When ARC is run from this checkout without `ARC_DOMAIN_INFO_CACHE` or
`XDG_CACHE_HOME`, domain data is cached under:

```text
/arc-dev/cache/domain-info/
```

## MCP

Install the packages above, then configure the MCP server command as
`arc-mcp`. The ARC MCP server exposes paper tools such as `get_metadata`,
`get_references`, `get_citers`, `get_section`, `get_LLM_summary`, and domain
tools:

```text
domain_build(seed_paper, intent="", provider="auto")
domain_status(job_id) or domain_status(seed_paper, intent="")
domain_get_summary(seed_paper, intent="")
domain_get_graph(seed_paper, intent="")
```

`domain_build` returns immediately with a `job_id`; poll `domain_status` until
the job is `done`, then read the cached summary or graph. `domain_get_summary`
and `domain_get_graph` are also cache-first: if the artifact is missing and a
`seed_paper` is supplied, they start the same background build and return a
`job_id`.

## Reference Code

`0_ref/` is read-only reference material. New code must not modify it or preserve
old compatibility assumptions.
