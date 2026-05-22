# Domain Info Reference

`arc-domain` builds cached research-domain artifacts from one seed paper
and an optional user intent. Use it when the user asks for a research field,
subfield map, foundation paper, domain network, or field briefing.

## CLI

Full build:

```bash
arc-domain build 0911.3380 --intent "quasi-single-field inflation observables" --json
```

Incremental steps:

```bash
arc-domain identify-foundation 0911.3380 --intent "..." --json
arc-domain build-network 0911.3380 --intent "..." --json
arc-domain build-evidence 0911.3380 --intent "..." --json
arc-domain summarize 0911.3380 --intent "..." --json
```

Read cached artifacts:

```bash
arc-domain status 0911.3380 --intent "..." --json
arc-domain get-summary 0911.3380 --intent "..." --json
arc-domain get-graph 0911.3380 --intent "..." --json
```

## MCP

Use `domain_build` for long builds. It returns immediately with a `job_id`;
poll `domain_status` until the status is `done`.

`domain_get_summary` and `domain_get_graph` are cache-first. If the requested
artifact is missing and `seed_paper` is supplied, they start a background
domain build and return a `job_id`.

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
