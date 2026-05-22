# Domain Info Reference

`arc-domain` builds cached research-domain artifacts from one seed paper
and an optional user intent. Use it when the user asks for a research field,
subfield map, foundation paper, domain network, or field briefing.

## CLI

Full build:

```bash
arc-domain llm-build 0911.3380 --intent "quasi-single-field inflation observables" --json
```

Incremental steps:

```bash
arc-domain llm-identify-foundation 0911.3380 --intent "..." --json
arc-domain llm-build-network 0911.3380 --intent "..." --json
arc-domain build-evidence 0911.3380 --intent "..." --json
arc-domain llm-summarize 0911.3380 --intent "..." --json
```

Read cached artifacts:

```bash
arc-domain status 0911.3380 --intent "..." --json
arc-domain get-summary 0911.3380 --intent "..." --json
arc-domain get-graph 0911.3380 --intent "..." --json
```

## MCP

Use `llm_domain_build` for long builds. It may call the host LLM provider, so
it waits only until the MCP deadline margin. If the result is not ready, it
returns a `job_id`.

Phase 1: Check cached artifacts.
Step 1: Call `domain_get_summary` or `domain_get_graph`.
Step 2: These tools are cache-only and do not call an LLM.

Phase 2: Build missing artifacts.
Step 1: Call `llm_domain_build`, `llm_domain_get_summary`, or
`llm_domain_get_graph`.
Step 2: If MCP returns `status: "job_running"` with a `job_id`, run:

```bash
arc-mcp jobs watch <job_id> --json
```

Step 3: If the CLI watcher is unavailable, poll `job_status` or `domain_status`
until status is `done`.
Step 4: Do not call `cancel_job` unless the user explicitly asks.

ARC stores MCP job state under `cache/arc-mcp/jobs/`. The CLI watcher and MCP
tools read the same persisted job files.

After completion:

```text
domain_get_summary(seed_paper="0911.3380", intent="...")
domain_get_graph(seed_paper="0911.3380", intent="...")
```

## Artifacts

The cache contains:

- `foundation_pool.json`: seed, newest citers, seed references, and witness set.
- `foundation_candidates.json`: top candidate foundation papers.
- `foundation_selection.json`: LLM or deterministic foundation choice.
- `citer_pool.json`: merged most-recent and most-cited foundation citers.
- `selected_papers.json`: up to 50 papers selected for domain construction.
- `reference_overlap.json`: common references added to the network.
- `domain_graph.json`: node/edge graph.
- `network.html`: static visualization.
- `evidence_pack.json`: titles, abstracts, and conclusion/outlook text.
- `domain_summary.json`: compact field briefing.

Default checkout cache:

```text
/arc-dev/cache/arc-domain/
```

## Notes

All single-paper data access should go through `arc-paper`. Domain Info
should not fetch INSPIRE or ar5iv directly except through that package.
