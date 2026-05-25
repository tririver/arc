# Research Foundation Workflow

Use this workflow after `research-plan.md`. The output is versioned foundation
JSON. Do not verify equations here; checking belongs to `research-execute.md`.

Write artifacts under:

```text
<project-dir>/calculate/<run-id>/foundation/foundation.v001.json
<project-dir>/calculate/<run-id>/foundation/latest.json
<project-dir>/calculate/<run-id>/foundation/research-foundation.md
```

Each foundation file must use `schema_version: "arc.research_foundation.v1"`.

## Phase 1: Prepare Versioned Foundation JSON

Step 1: Read `<project-dir>/calculate/<run-id>/plan.json`.

Step 2: Create `foundation.v001.json`. If the foundation later changes, create
`foundation.v002.json`, update `latest.json`, and record why the version
changed. Do not edit older versions in place.

Step 3: Use only ARC paper/domain tools and recorded literature checks from
the plan to populate sources. Do not read hidden cache files directly.

## Phase 2: Align Conventions

Step 1: When building the foundation from multiple papers, choose one
consistent convention for signs, metric, Fourier transforms, units, field
normalizations, and symbol names.

Step 2: If a source uses a different convention, translate it into the chosen
foundation convention when the translation is clear. Record the source
convention and the chosen convention in `conventions`.

Step 3: If the convention is not consistent or the translation is uncertain,
make it as convenient as possible for the calculation, mark the item
`convention_check`, and add it as a non-axiom item for the check loop in
`research-execute.md`.

## Phase 3: Record Equations And Confidence Labels

Each equation must include:

```json
{
  "id": "eq_001",
  "label": "short human label",
  "latex": "...",
  "role": "first_principle | useful_result | convention | validation_only",
  "convention_ids": ["conv_001"],
  "axiom_status": "axiom | not_axiom",
  "publication_status": "published_high | published_low | not_in_publications",
  "citation_count": 0,
  "check_status": "not_checked",
  "judgment": "reasonable | doubt",
  "sources": [
    {
      "paper_id": "arXiv:...",
      "section": "S2",
      "mcp": "get_section(paper_id=\"arXiv:...\", section=\"S2\")",
      "cli": "arc-paper get-section arXiv:... --section S2 --json"
    }
  ]
}
```

Confidence fields mean:

```text
axiom_status: axiom | not_axiom
publication_status: published_high | published_low | not_in_publications
citation_count: INSPIRE citation count for the source paper when available
check_status: not_checked
judgment: reasonable | doubt
```

Use `published_high` only when the cited source has more than 50 citations.
Use `published_low` for cited sources with 50 or fewer citations. Use
`not_in_publications` only for equations introduced by this workflow.

## Phase 4: Preserve Sources For Later Checks

Step 1: For every source, include both an MCP reminder and a CLI command that
future agents can run to inspect the section or equation context.

Step 2: Prefer exact sections. If no exact section is known, include a TOC or
full-text search command that can recover the relevant location.

Step 3: Do not mark a non-axiom equation as checked. The execute workflow must
create one checking step for every non-axiom equation.

Step 4: Render `latest.json` into `research-foundation.md` with the chosen
conventions, equations, confidence labels, source locations, and any version
change notes. After `research-foundation.md` is generated, copy it to
`<project-dir>/research-foundation.md` so human readers can inspect the main
project reports together.

After copying the Markdown report, call
MCP `md2pdf(input="<project-dir>/research-foundation.md")`. It starts a
background PDF job; record the returned job id if present and do not wait
before continuing.
