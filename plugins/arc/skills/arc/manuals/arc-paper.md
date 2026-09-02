# ARC Paper Quick Start

`arc-paper` is ARC's deterministic paper-reading layer. Use it to identify a
paper, inspect its structure, read sections, search prose or equations, and
follow citations. Commands return
a typed JSON envelope. Always check `status`, top-level `warnings`, `error`, and
the relevant fields under `data`; an empty result is not enough to diagnose why
nothing was found.

Provider-neutral local import, deterministic parsing, frozen document reads,
keyword inventories, rich-document export, and document-cache administration
belong to `ac-document`; see `ac-document.md`. `arc-paper` may expose
compatibility commands for these operations, but new neutral workflows should
call their owning package.

The paper cache defaults to `.arc/cache/arc-paper` below the current directory.
Use the same working directory or set `ARC_PAPER_CACHE` consistently across
related commands. Inside an ARC source checkout, use an ignored location such
as `local/cache/arc-paper`.

## Run ARC Paper

Examples below assume `arc-paper` is on `PATH`. Check once with
`arc-paper --help`. If the command is unavailable, use the portable Skill
launcher; inside an ARC source checkout, the package virtual environment is a
direct development fallback:

```bash
<skill-dir>/scripts/arc-runtime arc-paper --help
packages/arc-paper/.venv/bin/arc-paper --help
```

Use the selected launcher in place of `arc-paper` in later examples. Do not
search package internals for another executable.

## Start Here: Read One Paper

These commands cover the usual reading loop for arXiv:0911.3380:

```bash
# Identify the paper.
arc-paper get-metadata arXiv:0911.3380

# See its structure.
arc-paper get-table-of-contents \
  --reference 0911.3380 --source-format html

# Find a term and its surrounding prose.
arc-paper search-full-text \
  --reference 0911.3380 --source-format html \
  --term "isocurvaton" --context-lines 1

# Read the complete conclusion, not only a search snippet.
arc-paper get-section \
  --reference 0911.3380 --source-format html "Conclusion"

# Read printed equation 2.30 with nearby PDF layout text.
arc-paper search-equations \
  --reference 0911.3380 --source-format pdf \
  --term "2.30" --context-lines 8
```

Use these result paths first:

| Task | Result to inspect |
| --- | --- |
| Metadata | `data.title`, `data.authors`, `data.abstract`, `data.identifiers` |
| Table of contents | `data.entries[]` (`section_id`, `title`, `ordinal`) |
| Full-text search | `data.occurrences[]` (`context`, `title`, `matched_terms`) |
| Complete section | `data.title`, `data.section_id`, and `data.text` |
| Equation search | `data.matches[]` (`source_label`, `normalized_tex`, `source_excerpt`) |

Every structural read also returns the exact parsed source at
`data.source.document`. Searches return it at
`data.documents[].source.document`. Keep that object when later work must read
the identical cached representation.

## Choose a Target and Representation

For ordinary reading, use `--reference`. It accepts one exact arXiv ID, DOI,
INSPIRE ID, URL, or exact cached title. Resolution is cache-first and may
acquire a missing source. For a frozen, provider-free read, pass a previously
returned document object as JSON:

```bash
arc-paper get-section \
  --document-ref '<data.source.document JSON>' \
  --cache-root <same-cache-root> "<section id or title>"
```

`--document-ref` never discovers another paper or contacts a provider. It
verifies the source digest, parser contract, and parsed-document digest. A
damaged derived parse can be rebuilt from verified source bytes; missing or
damaged source bytes produce a typed failure. Physical cache paths are private,
unstable implementation details and are not provenance.

For references, ARC otherwise uses the first parseable representation already
recorded for that identity. Choose explicitly when the task depends on format:

- Use `--source-format html`, `--source-format markdown`, or
  `--source-format tex` for headings and prose.
- Use `--source-format pdf` for printed equation numbers and page/layout
  evidence.
- Do not assume different representations have identical headings or equation
  labels. Confirm the returned format and digests.

`--refresh` applies only to reference targets. Use it only when fresh upstream
data is required; an exact document target is immutable.

## Search the Right Surface

| Goal | Command | Search surface |
| --- | --- | --- |
| Discover papers | `search-metadata` | Titles, abstracts, authors, identifiers |
| Find text in paper bodies | `search-full-text` | Selected documents, or cached corpus without targets |
| Find equations | `search-equations` | Labels, normalized math, nearby equation text |
| Filter direct citers | `search-citers` | Titles and abstracts of papers citing one paper |

### Search Full Text

Search one or more explicit targets when the intended papers are known:

```bash
arc-paper search-full-text \
  --reference 0911.3380 \
  --reference "doi:<doi>" \
  --term "specific multiword phrase" \
  --term "alternate phrase" \
  --limit 100 --context-lines 1
```

Repeated `--term` values form literal OR: a match for any term is returned.
They are not regular expressions. Matching ignores case unless
`--case-sensitive` is set. Prefer specific multiword alternatives, then inspect
`matched_terms`, `context`, location, section or page title, and source digest.

Mixed `--reference` and `--document-ref` targets are allowed and retain their
command-line order. Duplicate parsed content is searched once. If at least one
target resolves, failures for other targets appear in `data.failures`; if none
resolve, the operation fails.

Omit targets to search every currently materialized cached document:

```bash
arc-paper search-full-text \
  --term "specific multiword phrase" \
  --term "alternate phrase" \
  --limit 100 --context-lines 0
```

A broad corpus query may return `mode: "refinement_required"`, exact counts,
and up to 50 paper titles rather than occurrences. Refine the terms or select
explicit targets. `rg_unavailable` means the required `rg` executable is
missing; do not replace this operation with physical-cache inspection.

## Read Sections and Equations

`get-table-of-contents` returns section IDs, titles, levels, zero-based
ordinals, and page bounds when the selected representation provides them.
`get-section` accepts an unambiguous title or section ID. If titles are
ambiguous, use the ordinal reported by the table of contents:

```bash
arc-paper get-section \
  --reference 0911.3380 --source-format html --ordinal 10
```

`search-equations` uses the same literal OR and ranks an exact source
label before label substrings, normalized math, and nearby context. For labeled
PDF matches, inspect `page_candidates` and the layout-preserving
`source_excerpt`. PDF extraction is approximate, so retain its warning and
source digest with any derived claim.

When no printed label is known, first find PDF equation candidates containing
an equals sign. Choose a result with a nonempty numeric `source_label`, then
rerun that exact label for useful context:

```bash
arc-paper search-equations \
  --reference <paper-id> --source-format pdf \
  --term "=" --limit 100 --context-lines 0

arc-paper search-equations \
  --reference <paper-id> --source-format pdf \
  --term "<source_label>" --context-lines 8
```

This discovers candidates containing `=`; it is not an exhaustive list of all
mathematical expressions. Do not transfer an HTML converter's label to PDF
without confirming that label in the PDF results.

If a printed equation label is missing from results, compare representations
before diagnosing ARC:

```bash
arc-paper search-equations \
  --reference <paper-id> --source-format html --term "<label>"

arc-paper search-equations \
  --reference <paper-id> --source-format pdf \
  --term "<label>" --context-lines 8
```

HTML converters can omit labels, merge equation rows, or number them
differently. A PDF-only match therefore indicates a representation difference,
not by itself an ARC parser bug. Report an `arc-paper` parser bug only when the
selected raw representation visibly contains the label or math that its parsed
result omitted; include format, source digest, search term, and the smallest
reproducing command.

## Find Papers and Follow Citations

Use metadata search when the identifier is not yet known, then read normalized
metadata by exact ID:

```bash
arc-paper search-metadata "quasi-single field inflation" --limit 20
arc-paper get-metadata arXiv:0911.3380
```

Inspect a paper's citation neighborhood with exact metadata operations:

```bash
arc-paper get-citer-count arXiv:0911.3380
arc-paper get-references arXiv:0911.3380 --enrich
arc-paper get-citers arXiv:0911.3380 --limit 100 --sort mostrecent
```

Use `get-citer-count` before a bounded scan when the neighborhood size affects
the search strategy. Use `get-references` for papers cited by the target and
`get-citers` for papers that cite it. The count is returned at `data.result`.

### Search a Citation Neighborhood

Use specific alternatives to shortlist direct citers:

```bash
arc-paper search-citers arXiv:0911.3380 \
  --term "specific phrase" --term "synonym" \
  --scan-limit 1000 --limit 50
```

Matching is literal OR after normalizing case, punctuation, and hyphens. Large
neighborhoods split the scan between most recent and most cited records and
report `scan_complete: false`. Inspect matching fields and control samples for
keyword blind spots. No match is evidence about the scanned neighborhood, not
proof of novelty.

## Reuse an External Reference

These commands handle a DOI, URL, or already downloaded source without relying
on an arXiv identity:

```bash
# Exact cache-only lookup; no download.
arc-paper lookup-reference --doi "<doi>"

# Acquire one authorized remote resource into verified cache.
arc-paper acquire-reference --url "<url>"

# Admit a file already available to the agent.
arc-paper admit-reference <downloaded-file> --url "<source-url>"

# Copy verified cached bytes to an explicit workspace path.
arc-paper materialize-reference \
  --resource-ref '<data.resources[] item JSON>' \
  --output <workspace-file>
```

`lookup-reference` returning no material is not an acquisition attempt. After
acquisition or admission, keep the returned identity and resource objects;
`materialize-reference` requires one complete `CachedResourceRef` object.

## Acquire Remote HTML Dependencies

Remote arXiv HTML may reference authored SVG and image files. Use the explicit
bundle command before local parsing when figure fidelity matters:

```bash
arc-paper export-arxiv-html-bundle <arxiv-id> \
  --output-dir <new-or-empty-source-directory> \
  --cache-root <paper-cache-root>

ac-document export-rich-document \
  <new-or-empty-source-directory>/source.html \
  --output-dir <rich-document-workspace> \
  --cache-root <document-cache-root>
```

The legacy ARC bundle preserves both `img[src]` and `object[data]` authored
files. Explicit ACF acquisition is available through ARC's
`export-arxiv-html-acquisition` handoff, which writes a strict
`ac.document.html_source_export.v1` manifest. Local parse/import/export remain
network-free; ARC still owns paper identity and provider routing.

Inspect `data.manifest`, `data.resources[]`, and `data.warnings[]`. The source
bundle preserves the exact primary HTML and writes verified resources only at
safe authored relative paths. A dependency warning means partial availability;
it does not invalidate a valid primary source. SVG is stored as an external
file and is never executed or inlined by acquisition.

The bundle follows same-origin HTTPS only, validates every redirect, rejects
credentials/fragments/unsafe schemes, and enforces bounded counts and bytes.
It supports `object[data]`, `img[src]`, and `source[src]`; `srcset` remains an
explicit unsupported warning. `--refresh` refreshes the primary and bundle.
Without it, verified cache replay performs no HTTP request. A first bundle
request may refetch an older main-only cache entry to obtain missing final-URL
provenance. Replay also requires the exact same dependency count, per-resource
byte, document-byte, and redirect limits; older schema v1 sidecars lack that
policy identity and are reacquired.

`fetch-arxiv-auto`, `acquire-reference`, and ordinary local HTML parsing remain
single-file compatible. Local input never triggers dependency downloads. Use
`fetch-arxiv-html-bundle` when the cached manifest is needed without writing an
export directory. Cache archive selection includes reachable bundle resources;
cache removal retains shared content blobs rather than deleting possibly reused
bytes.

Treat an input ending in `vN` as an exact official-bundle identity. ARC keeps
that suffix in the official request, source/dependency cache keys, final URL,
and origin metadata. The final URL and strict authored revision signals (an
exact versioned HTML base root or paired official abstract/PDF header links)
must agree; absence or conflict fails closed. An exact-version 404 never falls
back to unversioned ar5iv. Inputs without `vN` retain latest/canonical fallback
behavior, and the legacy single-file commands remain unchanged.

Do not edit an unversioned schema v2 manifest or cache entry to add `vN`. Its
document URL, origin, key, policy, and digest bind the recorded provenance.
Schema v1 bundles lack policy identity and are not replayed by the v2 codec;
perform a new acquisition when either exact version or v2 policy provenance is
required.

## Parse a Local Source

Use one HTML, Markdown, flattened TeX, or PDF as the authoritative primary. A
validator checks fidelity and page evidence without replacing primary text:

```bash
arc-paper parse-local <chapter.tex> --validator <book.pdf>
arc-paper parse-local <note.md> --validator <note.pdf>
arc-paper parse-local <paper.pdf>
```

Keep reconciliation and page-mapping warnings visible. A validator may be the
same document or a whole-book PDF paired with one chapter source. Use
`arc-paper parse-local --help` for explicit formats and validation policies.

For an exact cached UTF-8 text source, `read-cached-source-range` reads one-based
inclusive lines without fetching. It is unavailable for PDF sources:

```bash
arc-paper read-cached-source-range \
  --document-ref '<document JSON>' \
  --cache-root <same-cache-root> <start-line> <end-line>
```

Advanced Markdown/PDF hierarchy reconciliation is available through
`reconstruct-cached-structure --help`.

## Build an Approximate Keyword Inventory

```bash
arc-paper extract-keywords <source> \
  --project-dir <project-dir>/keywords \
  --host-authority <host-authority> --approx-count 50
```

Use `unrestricted` only when explicitly granted; otherwise use `unknown`, and
reuse the same value on resume. Restricted or unknown requests follow
`manuals/ac-llm.md`. The target count is approximate: selected terms are
deduplicated and labeled with machine-counted occurrence frequency, without
padding. Explicit indexes receive model review. `matched_sentences` are
grounding hits, never definitions.

Keyword extraction is durable. Resume with the returned descriptor and the
same project/run identity; use `ac-jobs` for generic lifecycle operations.

## Help, Cache, and Recovery

```bash
arc-paper --help
arc-paper <command> --help
arc-paper cache --help
```

Use command help for reference acquisition, cache administration, exact flags,
and advanced operations. Remote failures do not invalidate verified cache
entries. Typed parse or search errors should be investigated at their stated
source or contract boundary; do not bypass verification by reading or editing
physical cache paths.
