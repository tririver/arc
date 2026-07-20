# ARC Companion Manual

`arc-companion` builds a companion-reading PDF for an INSPIRE/arXiv paper. It
uses `arc-paper` for all metadata, ar5iv full text, parsed source structures,
references, citers, and cached binary assets. It uses `arc-llm` for structured
segmentation, glossary construction, parallel translation and commentary, and
final review. It does not call providers directly and does not use
`arc-typeset`.

Literature context combines INSPIRE records with a bounded set of targeted
related-paper full texts cached through existing `arc-paper` APIs. Selection is
based on citation position and semantic relevance to the source units. A
verified INSPIRE abstract may replace an unavailable related full text, with
the weaker evidence level recorded explicitly. The package never treats
metadata alone as support for a technical claim.

## Commands

Build or resume a run:

```bash
arc-companion build <paper-id> --project-dir <dir> \
  --annotation-language zh-CN --provider auto --workers 24 --json
```

Useful build flags:

- `--recache`: reparse cached ar5iv HTML and retry missing source assets.
- `--refresh`: refetch INSPIRE/ar5iv data before parsing; mutually exclusive
  with `--recache`.
- `--force`: invalidate companion checkpoints, but not the `arc-paper` cache.
- `--workers`: bounds each independent parallel wave. The default permits 24
  concurrent translations and 24 concurrent companion commentaries; it also
  bounds segmentation windows and local refinements.
- `--domain-id <id>` / `--domain-manifest <path>`: mutually exclusive,
  read-only reuse of an explicitly named existing domain. Companion never
  discovers or builds a domain automatically.

Inspect or validate an existing project:

```bash
arc-companion status --project-dir <dir> --json
arc-companion validate --project-dir <dir> --json
```

Create an optional package only after an explicit request:

```bash
arc-companion package --project-dir <dir> --json
```

This is a reproducibility package, not a related-literature corpus. It contains
the validated PDF and TeX, primary-paper assets used by TeX, source and package
manifests, validation report, and build state. It does not contain downloaded
reference or citer papers.

## Language and Models

The default translation and companion language is `zh-CN`. When omitted, the
command prints a Chinese language-switch notice and continues; JSON mode also
returns it in `meta.notice`. The original paper is never rewritten.

Default model routing is:

- medium for segmentation and local boundary refinement;
- medium for the comprehensive terminology glossary;
- medium for per-unit translation;
- medium for the single failure-only opaque-token translation correction;
- high for per-unit companion commentary; and
- medium for section and whole-document review.

Segmentation and glossary construction begin concurrently. After both pass
validation, translation and commentary run as two independent stateless waves.
Each wave is bounded by `workers`, so the default peak is 24 translation calls
plus 24 commentary calls. Both calls receive the frozen full-paper glossary and
the same current source unit. Long documents use medium-tier section reviews
followed by a medium-tier consolidation review, with complete unit coverage and a
bounded source anchor for every unit in the consolidation payload. Review
patches may change translations and commentaries only.

Every new primary medium-tier translation prompt ends with an explicit controller
checklist for exact block coverage/order, byte-exact opaque-token coverage/order,
cross-block token isolation, and protected-name spelling. This reinforces the
existing deterministic validator without changing its contract. Previously
validated translation checkpoints already satisfy that validator and remain
reusable; the instruction-only checklist does not force wholesale regeneration.

If a generated translation changes, drops, or reorders opaque inline formula,
citation, or link tokens, the controller collects all mismatched blocks in that
segment into one shared medium-tier placement correction; ordinary translation
remains medium-tier and every valid block stays unchanged.
The correction runs offline and receives the prior invalid block text, its
natural-language residue after opaque markers are removed, source-run context,
and exactly `N+1` stable slots for `N` source-owned tokens. It may only partition
that residue among the slots: their concatenation must remain byte-for-byte
identical. The sole exception is exact insertion of specifically missing
protected names; removing those inserted names must recover the prior residue.
Well-formed and bounded malformed `[[ARC_INLINE:...]]` candidates are removed
from the prior text before slot comparison, so a damaged marker cannot make
repair impossible or survive as visible prose. The controller interleaves the original immutable tokens and reruns the same
strict whole-block validator. Every other translated block remains byte-for-byte
unchanged. Only a validated result is checkpointed, and there is never a second
correction attempt. The failure-only prompt version and medium tier are recorded
separately, so this exceptional path can change without invalidating valid
content checkpoints through the global prompt fingerprint.

Protected-name checks inspect natural text runs only. Names visible solely
inside controller-owned formula, citation, or link runs are already preserved
by token interleaving and never trigger a duplicate name insertion. Text runs
remain separated during this check so adjacent runs cannot accidentally form a
name, and generated Latin spelling is case-sensitive. A name-only repair is
accepted only when removing one exact, boundary-delimited insertion of each
requested name recovers the prior residue.

Before any slot repair, one shared segment preflight validates complete ordered
block coverage and object shape. It requires non-empty translated residue only
when the source block has non-empty natural text runs, falling back to block
text when inline runs are absent. A pure controller-owned link/citation block
may therefore be token-only, while a later natural-text block that is empty or
token-only blocks repair even when an earlier block has a token mismatch.

The controller first submits exactly the first `min(workers, unit_count)`
source-order units to both lanes. As soon as translation and commentary have
both finished for that first wave, and before any remaining unit is submitted,
the controller renders and fully validates a persistent prefix
`<paper-safe>_companion_<language>_first_round_preview.pdf`. Its TeX, source
manifest, validation report, paths, and hashes remain available through
`arc-companion status`. This diagnostic preview uses the unreviewed first-round
translation and commentary for that first wave and is not the final
deliverable. Only after it passes does the controller submit the remaining
units. A source-fidelity, LaTeX, or PDF-validation failure stops the run at this
boundary so layout and source-structure problems are found before further LLM
work.

For an explicit visual-QA pause, add `--stop-after-preview`. A successful build
then returns with status `preview_ready` immediately after those validations,
without submitting remaining units, resolving evidence requests, or starting
review. Inspect the recorded `preview_pdf`, correct ARC if the diagnostic PDF
does not meet the contract, then rerun the same command without
`--stop-after-preview`; valid checkpoints are reused and generation continues
from this boundary.

## Generation Access and Portability

Each per-unit translation and commentary call receives bounded `FULL-PAPER
NAVIGATION CONTEXT`: the source paper's table of contents or section map plus
compact neighboring and global anchors. This lets a stateless call resolve the
unit's place in the argument without embedding the complete parsed document in
every prompt.

When the selected host supports the capability, these calls run with ARC-only
MCP access to the ARC paper cache and with internet lookup enabled. This is an
internal generation default, not a public CLI option. It does not expose other
user-configured MCP servers. Segmentation, glossary extraction and
consolidation, and review remain tool-disabled and depend only on their frozen
inputs.

External access has different authority in the two generation tracks:

- Translation may consult ARC or the web only to establish standard
  terminology or disambiguate source context. The supplied source blocks remain
  authoritative; lookup results cannot add, replace, correct, or omit their
  content.
- Commentary may use ARC and web sources for explanation and related-work
  context, but every external factual claim must retain captured provenance.
  Claims without captured, verifiable provenance are removed before
  publication rather than attributed from model memory.

Commentary is selected by reader value, not by a requirement to explain every
segment. An evident passage may have empty explanation and commentary fields;
generation and review must not fill them with paraphrase or generic teaching
text. A commentary that merely restates the source, rewrites the same meaning,
or repeats the same reasoning in different words is prohibited; every
non-empty commentary must add concrete information, a new reasoning step, or a
useful reader connection. When explanation is useful, choose the most relevant
emphasis:

- why the material is needed and how it advances the argument, giving this
  priority at section and chapter openings;
- a genuinely different presentation supported by the supplied references,
  where the logical starting point, viewpoint, or organization differs; an
  equivalent restatement does not qualify;
- a cautiously stated, substantive incompatibility with another source, while
  excluding differences that reduce to convention, notation, normalization,
  or an equivalent formulation; or
- intermediate mathematics omitted from a derivation that is not evident to
  the intended reader; or
- a concrete connection to another useful concept, course, or discipline when
  the relationship is substantive rather than a loose analogy; or
- a reliable, relevant historical story or interesting fact, supported by a
  verifiable source and not folklore or uncertain attribution; or
- a materially useful current understanding or development considered
  proactively, but included only when a registered, verifiable source
  identifies what specifically changed and the commentary explains exactly how it
  materially changes the interpretation of this passage.

These are priorities, not mandatory fields. Do not force all of them into one
segment, repeat the source, chase novelty, present an ordinary reformulation as
progress, or invent a distinction solely to create commentary. Never present a
purported update, historical anecdote, or external connection from model memory
or from an unregistered search result.

MCP and internet access are optional. A host without either capability uses the
same portable prompt with the current segment, frozen glossary, bounded full-
paper navigation context, and prepared evidence pack. Generation therefore
remains portable across supported agent hosts; diagnostics record unavailable
capabilities, and prior/later-work fields stay empty when the fallback context
does not support them.

## Terminology and Name Contract

The glossary is constructed from the complete paper and presents three reader-
facing fields: English term, standard target-language term, and a short target-
language explanation. Windowed candidates are consolidated and deduplicated by
a medium-tier call before unit generation. Translation and commentary must use
the frozen adopted forms consistently.

Keep only specialist concepts, methods, non-standard parameters, key
approximations, and translation-ambiguous terms useful to a field reader; do
not pad the result. The caps are 50 entries through 50 pages, 100 through 100
pages, and 200 above 100 pages. If page count is unavailable, the absolute cap
is 200.

All personal names stay in their Latin-script source form. This includes name
roots in eponymous technical terms, such as `Feynman diagram` becoming
`Feynman` plus the target-language form of `diagram`. Seed-paper author records
and glossary-recognized names form a protected-name inventory. Global protected-name
generation deliberately ignores bibliography, reference, and citer authors;
related-work authors remain available only in the per-unit evidence supplied
to commentary. Deterministic validation rejects a translation or review patch
that removes or rewrites a protected name.

## Related-Work Evidence

`arc-companion` obtains references, citers, abstracts, and targeted related-
paper full text through existing `arc-paper` service interfaces. It does not
change the `arc-paper` public contract or place companion orchestration there.
Selection is bounded and relevance-based rather than tied to a paper, author,
subfield, or hard-coded keyword list. The default global cap is eight selected
references and eight selected citers. A source unit receives at most two of
each, with at most 4,000 relevant source characters from any one related paper.
Only the primary paper requests a rich document. Related papers use lightweight
parsed sections with stable section locators and content hashes. Explicit
domain context is a preferred relevance signal, not a closed corpus, and never
disables ARC, INSPIRE, references/citers, or web lookup.

Each commentary separates passage explanation from supported prior and
subsequent work. Stable evidence identifiers remain in structured evidence
fields and manifests and are never printed as reader-facing hashes or
controller labels. The rendered citation uses the source title and a section
or other human-readable location when available, falling back to the source
title.
When full text is unavailable, the evidence ledger may supply a verified
abstract marked `abstract_only`; when neither full text nor an abstract
supports a statement, that statement must be omitted. Companion-only citations
are rendered separately and never alter the source paper's bibliography.

Review normalization removes reader-visible HTML/Markdown container syntax
from source and translation presentation copies while preserving meaningful
body text and discarding machine-only container summaries. The renderer
repeats this cleanup defensively and converts legacy inline evidence-ID markers
into the same human-readable source citations. Immutable source blocks and
structured evidence bindings remain unchanged for audit.

The controller stores the global registry in `evidence.json` and each unit's
selection in `segment-evidence/<segment>.json`. Every registered item keeps its
stable `evidence_id` (`prior-NNN` or `later-NNN`); newly discovered web items
receive a controller-derived `web-<digest>` ID and are never named by the
model. Its immutable `source_descriptor` uses schema
`arc.companion.source-descriptor.v1` and records:

- `source_type` (`arc_cache` or `web`), provider, canonical paper ID or HTTP(S)
  URL, title, authors, year, and retrieval time;
- a SHA-256 digest of the captured content; and
- selected snippets with text, their own SHA-256 digest, and a stable ARC block
  ID or URL-fragment locator.

An `abstract_only` fallback records the abstract field locator and digest.
Model-returned URLs or source descriptors are discovery hints, not evidence;
the controller must capture and register the material before its ID can appear
in commentary. Unavailable or unrecorded internet material cannot support a
claim.

A first-round high-tier annotation may return at most two structured
`evidence_requests` (`relation`, `needed_claim`, queries, candidate paper IDs,
candidate URLs, and reason). The dependent related-work claim must remain out
of the draft. The controller batches all requests and runs ARC/full-text,
INSPIRE, and web-discovery lanes independently; a hit or failure in one lane
does not cancel or short-circuit the others. Raw web snippets remain discovery
hints and cannot be registered as claim evidence.

After canonical-source deduplication and provenance validation, only segments
with newly registered evidence receive one high-tier annotation rerun. The
rerun sees its first draft and registered evidence, never triggers a third
round, and does not rerun translation, segmentation, glossary, or unaffected
annotations. If reliable evidence remains unavailable, the corresponding
prior/later-work text is empty. `annotations.first-round.v1.json` and
`evidence-resolution.v1.json` retain the request, lane, acceptance/rejection,
content-hash, rerun, and final claim-to-evidence audit.

## Segmentation Contract

The controller first partitions the immutable `arc-paper` block sequence into
non-overlapping, section-aligned windows. Windows may run concurrently, but
their results are merged only in source order, and concurrency never exceeds
`workers`.

For each window, the medium stateless call sees inventory entries identified by
stable global 1-based ordinals. Its structured output may select only unique
internal cut-after ordinals inside that window; the controller sorts accepted
cuts into source order. It cannot
return source text, block IDs, start ranges, reordered blocks, replacement
content, or the window endpoint. The controller adds every window endpoint and
the final paper endpoint, then converts accepted cuts into canonical
`start_block_id`, `end_block_id`, and `block_ids` ranges from the original
order; no LLM-provided source range is trusted.

Every resulting unit must satisfy the implementation's fine-grain hard limits:
at most 24 atomic blocks or 60,000 characters in the largest actual source-block
projection supplied to either the translation or commentary prompt. The
projection retains the text, opaque inline tokens, compact TeX, captions,
labels, and structure those agents receive; it does not count rich-document
preservation data such as raw HTML, MathML, formula-layout records, or assets
that stay controller-owned and never enter those prompt fields. This accounting
rule does not delete, rewrite, or unpin any content in the pinned `arc-paper`
document and does not narrow the source-fidelity contract. When a splittable
unit is too large, only that unit is sent for another medium stateless cut-only
refinement. Refinement is limited to three local rounds. An individual atomic
block, including an indivisible display-equation block, is never mechanically
split to satisfy a prompt-size guard. An unresolved oversized splittable unit
is a blocking segmentation failure; the controller does not silently accept it,
mechanically split source structures, or fall back to LLM-authored source data.

Before translation and commentary start, validation must prove that the merged ranges cover
every source block exactly once, in order, with no gap, overlap, duplicate,
unknown ordinal, or out-of-window cut. This validation changes boundaries only;
it never changes an `arc-paper` block or source entity.

## Source and Cache Contract

The package accepts only `arc-paper` rich parsed documents with schema
`arc.paper.document.v2` and `integrity.status=complete`. The companion run pins
the document and asset-manifest hashes. Missing ar5iv content, LaTeXML errors,
unresolved required source structures, missing assets, unsupported tables, or
formulae without a reliable TeX/MathML representation are blocking errors.

The source fidelity claim is relative to the cached ar5iv/LaTeXML rendition,
not pixel identity with an author TeX tree or publisher PDF. The LLM never
supplies original-source fields.

Internal checkpoints live below `<project-dir>/.arc-companion/`. Each accepted
segmentation window and local refinement has its own source-window and
prompt/schema fingerprint. A rejected result is retained only as a diagnostic
attempt and is never reused as a valid cut cache. The final canonical
segmentation cache is written only after all windows, refinements, and exact
coverage checks pass. Its fingerprint also includes source, asset, evidence,
language, prompt, schema, and workflow versions. Rerunning the same command
reuses valid per-window and final results and retries only failed or stale work;
`--force` still invalidates companion checkpoints.

Segmentation failures return visible diagnostics with the window identity,
available section context, refinement round, rejected ordinals, and reason,
such as a non-integer, duplicate, out-of-window, oversized, or
incomplete-coverage result. Failure stops before annotation and cannot publish
a successful TeX, PDF, manifest, or validation result.

These rules do not change the public CLI or accepted paper identifiers. The
rich `arc-paper` source contract marks structurally source-only blocks with a
durable `source_role`. Table-of-contents blocks, acknowledgment sections, and
reference-list headings and entries are excluded before segmentation, glossary,
translation, commentary, evidence selection, and review. They remain pinned
source blocks and render exactly once in the original track; nested TOC levels
and links come from the preserved block HTML. Title, author, and affiliation
blocks remain excluded from generation under the existing front-matter rule.
These classifications do not weaken source fidelity:
segmentation chooses only presentation boundaries, while original-track LaTeX,
equation numbers, tables, figures, bibliography, links, and asset hashes still
come exclusively from the pinned `arc-paper` document.

## LaTeX and Validation

Rendering uses XeLaTeX through `latexmk`. Source equations receive their
visible original tags explicitly; figures use hash-verified cached bytes;
tables preserve their logical cell grid and spans; bibliography entries retain
their visible labels, order, and text. No BibTeX reordering is allowed.

The front matter uses a minimal title page followed by the three-column
glossary; it does not include a "version note" page. Each semantic unit then
uses the fixed order original, translation, companion commentary. The original
track stays on the plain page without a background or left rule. Translation
and companion tracks retain distinct, subtle light backgrounds, with
typography, spacing, and rules following the reference companion design and
remaining suitable for printing.

Visual layer styling replaces controller-facing unit headers. Do not print a
unit label or its segment ID, or an `Original` field, before each source
passage.

Paper headings use the number already present in the pinned source and render
with unnumbered LaTeX section commands. Register each heading explicitly at
its source hierarchy in the table of contents so LaTeX does not prepend a
second number while TOC entries, PDF bookmarks, and source anchors remain.

The renderer copies displayed formulas from the pinned source into the
translation so it can be read locally, but does not copy their equation
numbers. It does not repeat figures, tables, or other floating objects. Those
source objects appear exactly once in the original track. Translated captions
may be provided as prose without cloning the float. The original source
renderer remains isolated from all LLM output.

Runtime tools:

- Required for PDF generation: `latexmk`, `xelatex`, and fonts covering the
  source and annotation languages.
- Required for final validation: `pdfinfo`, `pdftotext`, and `pdffonts`.
- Required only for corresponding source assets: a supported deterministic
  SVG/EPS conversion utility.

`validate` checks ordered source coverage, entity hashes and labels, internal
links, PDF metadata, searchable text, fonts, glossary and protected-name
consistency, complete translation and annotation-record coverage (while
permitting intentionally empty commentary prose), formula-number omission
in translation, and absence of cloned floats. A failed check leaves the run
artifacts for diagnosis but never publishes a successful deliverable.

## Troubleshooting

- `parsed_document_needs_recache`: rerun `build ... --recache`.
- `document_integrity_incomplete`: inspect the returned warnings; use
  `--refresh` only if a current ar5iv conversion may fix the source.
- Missing LaTeX or Poppler command: install the named runtime dependency and
  rerun the same build; completed LLM checkpoints are retained.
- Failed segmentation window or local refinement: inspect the returned window,
  cut, and refinement diagnostics, then rerun the same build. Valid window
  checkpoints are retained; no partial final segmentation is accepted.
- Failed glossary, translation, or commentary unit: rerun the build. A
  translation with an opaque-token mismatch already receives one bounded
  controller-guided correction attempt during that build. Successful independent
  work is not repeated. Translation and commentary lanes drain all
  submitted units before reporting their aggregated failures, so later
  successes remain checkpointed and a new build schedules only missing or stale
  units.
- If a traceback suggests that one early unit cancelled unrelated work, compare
  the traceback with checkpoint coverage and reproduce the scheduling behavior
  in a minimal package test. Fix controller-owned cancellation in
  `packages/arc-companion`; never patch only one run directory.
- Changed paper version, language, model route, glossary, prompt, or evidence:
  the affected fingerprints invalidate automatically.
