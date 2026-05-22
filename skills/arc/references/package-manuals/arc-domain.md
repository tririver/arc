# Arc Domain Package

`arc-domain` builds cached research-domain artifacts from a seed paper and an
optional user intent. Use it for foundation-paper identification, domain paper
selection, citation-network graphs, evidence packs, and compact field
briefings.

Intent affects the domain cache. A different intent string should be treated as
a different domain selection.

## Full Build CLI

Phase 1: Start from a seed paper.
Step 1: If the user gives an intent, pass it exactly after trimming outer
whitespace.
Step 2: Run:

```bash
arc-domain llm-build 0911.3380 --intent "quasi-single-field inflation observables" --json
```

Phase 2: Inspect cached artifacts.
Step 1: Check status and read outputs.

```bash
arc-domain status 0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain get-summary 0911.3380 --intent "quasi-single-field inflation observables" --json
arc-domain get-graph 0911.3380 --intent "quasi-single-field inflation observables" --json
```

Step 2: Report `network.html` when the user wants the graph visualization.

## Incremental CLI

Use incremental commands when debugging or when the user asks for an
intermediate artifact.

Phase 1: Identify the foundation paper.
Step 1: Run:

```bash
arc-domain llm-identify-foundation 0911.3380 --intent "..." --json
```

Phase 2: Build the network and evidence.
Step 1: Run:

```bash
arc-domain llm-build-network 0911.3380 --intent "..." --json
arc-domain build-evidence 0911.3380 --intent "..." --json
```

Phase 3: Summarize the domain.
Step 1: Run:

```bash
arc-domain llm-summarize 0911.3380 --intent "..." --json
```

## MCP Tools

Read `references/package-manuals/arc-mcp.md` before using MCP.

Domain MCP tools:

```text
domain_status
domain_get_summary
domain_get_graph
llm_domain_build
llm_domain_get_summary
llm_domain_get_graph
```

Phase 1: Check cache-only artifacts.
Step 1: Call `domain_get_summary` or `domain_get_graph`.
Step 2: If artifacts are missing and the user wants them built, continue.

Phase 2: Build missing artifacts.
Step 1: Call `llm_domain_build`, `llm_domain_get_summary`, or
`llm_domain_get_graph`.
Step 2: Use `background=true` for slow builds or large job launches.
Step 3: Follow the background-job procedure in
`references/package-manuals/arc-mcp.md`.

## Artifacts

The domain cache contains:

- `foundation_pool.json`: seed, newest citers, seed references, and witness set.
- `foundation_candidates.json`: top candidate foundation papers.
- `foundation_selection.json`: LLM or deterministic foundation choice.
- `citer_pool.json`: merged most-recent and most-cited foundation citers.
- `selected_papers.json`: papers selected for domain construction.
- `reference_overlap.json`: common references added to the network.
- `domain_graph.json`: node/edge graph.
- `network.html`: static visualization.
- `evidence_pack.json`: titles, abstracts, and conclusion/outlook text.
- `domain_summary.json`: compact field briefing.

Default checkout cache:

```text
/arc-dev/cache/arc-domain/
```

## Package Boundary

All single-paper metadata, references, citers, full text, and sections should
come through `arc-paper`. Do not fetch INSPIRE or ar5iv directly inside domain
workflows.
