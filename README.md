# ARC Dev

ARC research tooling is organized as Python packages plus thin agent adapters.

## Packages

Install the current development packages:

```bash
python3 -m venv packages/arc-paper/.venv
. packages/arc-paper/.venv/bin/activate
python -m pip install -e packages/arc-llm[test]
python -m pip install -e packages/arc-paper[test]
python -m pip install -e packages/arc-domain[test]
python -m pip install -e packages/arc-mcp
```

## LLM Worker

Reusable host LLM execution:

```bash
arc-llm doctor config
echo '{"task":"say ok"}' | arc-llm run-json --provider auto
echo 'Say ok.' | arc-llm run-text --provider auto
```

`arc-llm` owns host detection, provider selection, model defaults, and
Codex/Claude CLI prompt execution. Other packages should use it instead of
shelling out to host LLMs directly.

## Paper

Deterministic paper data:

```bash
arc-paper get-title arXiv:0911.3380 --json
arc-paper get-references arXiv:0911.3380 --json
arc-paper get-toc arXiv:0911.3380 --json
arc-paper get-section arXiv:0911.3380 --section S2 --json
```

LLM summaries:

```bash
arc-paper llm-summary arXiv:0911.3380 --json
arc-paper llm-generate-summary arXiv:0911.3380 --provider auto --json
```

`llm-summary` first reads the local summary cache. If the summary is missing
and a host LLM provider is available, it generates and caches the summary
automatically. `llm-generate-summary` remains available when you want an
explicit generation command or provider override. Legacy aliases
`get-llm-summary` and `generate-llm-summary` still work.

Summary generation uses fast host defaults unless overridden: Codex uses
`gpt-5.4-mini`, and Claude Code uses `haiku`. The LLM pipeline summarizes paper
sections sequentially first, then synthesizes the final paper summary from the
title, abstract, table of contents, and compact section summaries. The final
JSON keeps `toc` as navigation metadata and stores the richer per-section text
under `section_summaries`; it does not rewrite those into one-sentence TOC
entries. References are intentionally not included in the summary input pack.

When called through MCP, tools that may invoke the host LLM are prefixed
`llm_`. They start the work, wait briefly, and return before the MCP client
timeout. If the result is not ready, they return a `job_id`. Poll
`job_status` and read with `job_result`, or use the blocking CLI watcher:

```bash
arc-mcp jobs watch <job_id> --json
```

MCP jobs are persisted under `cache/arc-mcp/jobs/`, so MCP tools and CLI tools
read the same job state. Completed section summaries are cached as they finish,
so a failed or interrupted paper-summary job can resume without paying again for
sections that already completed. MCP background jobs do not stream or push a
completion notification; clients should poll or use the CLI watcher. Do not call
`cancel_job` unless the user explicitly asks.

MCP LLM tools use this deadline rule: `ARC_MCP_INLINE_WAIT_SEC` if set;
otherwise `ARC_MCP_TOOL_TIMEOUT_SEC - ARC_MCP_BACKGROUND_MARGIN_SEC`; otherwise
a best-effort Codex `tool_timeout_sec` read from `~/.codex/config.toml`;
otherwise 90 seconds.

Job status includes ETA information after enough matching jobs have completed.
ARC stores runtime history in `cache/arc-mcp/stats/jobs.sqlite`; before three
similar samples exist, ETA is marked as unavailable.

Batch workflow:

```bash
arc-paper summary-batch create papers.txt --name qft-ideas --json
arc-paper summary-batch prefetch qft-ideas --workers 8 --json
arc-paper summary-batch run qft-ideas --provider auto --concurrency 2 --max-items 10 --json
arc-paper summary-batch status qft-ideas --json
arc-paper summary-batch export qft-ideas --format jsonl --output summaries.jsonl --json
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

Without plugin env, `arc-paper` falls back to parent-process detection.

Debug:

```bash
arc-paper doctor host --json
arc-paper doctor provider --json
arc-paper doctor cache 0911.3380 --json
```

When ARC is run from this checkout without `ARC_PAPER_CACHE` or
`XDG_CACHE_HOME`, arc-paper data is cached under:

```text
/arc-dev/cache/arc-paper/
```

## Domain Info

Build a cached research-domain package from one seed paper plus an optional
intent:

```bash
arc-domain llm-build 0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain status 0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain get-summary 0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain get-graph 0911.3380 --intent "quasi-single-field inflation observables" --json
```

`arc-domain` uses `arc-paper` for all single-paper operations. It
identifies a likely foundation paper, builds a citation-domain graph of up to
about 60 nodes, renders `network.html`, builds an evidence pack from titles,
abstracts, and conclusion/outlook sections, then asks `arc-llm` for a
compact field briefing. If the host LLM is unavailable, deterministic fallback
artifacts are still written so the cache is inspectable. Legacy aliases such as
`arc-domain build` still work.

When ARC is run from this checkout without `ARC_DOMAIN_CACHE` or
`XDG_CACHE_HOME`, domain data is cached under:

```text
/arc-dev/cache/arc-domain/
```

ARC MCP job state is cached under:

```text
/arc-dev/cache/arc-mcp/
```

## MCP

Install the packages above, then configure the MCP server command as
`arc-mcp`. The ARC MCP server exposes paper tools such as `get_metadata`,
`get_references`, `get_citers`, `get_section`, and cache-only domain tools.
Anything that can invoke the host LLM has an `llm_` prefix:

```text
llm_get_summary(paper_id, provider="auto")
llm_generate_summary(paper_id, provider="auto")
llm_domain_build(seed_paper, intent="", provider="auto")
domain_status(job_id) or domain_status(seed_paper, intent="")
domain_get_summary(seed_paper, intent="")
domain_get_graph(seed_paper, intent="")
job_status(job_id)
job_result(job_id)
cancel_job(job_id)
```

`llm_domain_build` may return a completed result if it finishes before the MCP
deadline margin, otherwise it returns a `job_id`. In skill workflows, prefer:

```bash
arc-mcp jobs watch <job_id> --json
```

Then read the cached summary or graph. `domain_get_summary` and
`domain_get_graph` are cache-only. Use `llm_domain_get_summary` or
`llm_domain_get_graph` when a missing artifact should trigger a domain build.

## Reference Code

`0_ref/` is read-only reference material. New code must not modify it or preserve
old compatibility assumptions.
