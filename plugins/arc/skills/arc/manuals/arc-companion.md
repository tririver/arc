# ARC Companion Manual

`arc-companion` builds a source-faithful reading companion for a paper, lecture
note, or book. It consumes a rich source plus its paired PDF through
`arc-paper`, runs chapter-scoped stateful generation through `arc-llm`, and
renders and validates the PDF deterministically. The core workflow is CLI-only
and portable across supported agent hosts.

## Phase 1: Source and Structure

### Step 1: Supply a paired source

Formal output requires both:

- a rich Markdown, TeX, or HTML source for text, formulas, links, tables, and
  assets; and
- its paired PDF, authoritative for hierarchy, printed pages, and chapter
  boundaries.

The parsed cache records a PDF hash, structure origin, chapters, page ranges,
block-to-page alignment, and reconciliation proof. A legacy paired cache
without that proof must be recached. Boundary conflicts are resolved in favor
of the PDF; ambiguous alignment is a blocking error.

Use `--document-kind auto|article|book`. Automatic classification stops before
LLM work when it is not unique. An unsectioned article becomes one chapter; a
sectioned article uses substantive top-level sections. A book uses a real
repeated level, excluding front matter, references, and Index, and never
invents chapter titles merely to reach a target count.

### Step 2: Preserve exact coverage

Stable IDs use `ch-0001` and `ch-0001.seg-0001`. Validation proves that
eligible source blocks belong to exactly one chapter and that every chapter's
segments cover it once, in order, without gaps or overlaps. Original source
objects always come from the pinned rich document, never from an LLM.

## Phase 2: Commands

### Step 1: Build or resume normal progress

```bash
arc-companion build <source-id> --project-dir <dir> \
  --annotation-language zh-CN --document-kind auto \
  --provider auto --workers 24 --json
```

Useful flags:

- `--recache`: rebuild parsing and PDF reconciliation from cached inputs.
- `--refresh`: refresh remote inputs; mutually exclusive with `--recache`.
- `--idle-timeout-seconds <seconds>`: override provider inactivity timeout.
- `--regenerate-commentary`: keep reusable translations and rebuild commentary.
- `--skip-translation`: omit translation only after the workflow agent has
  confirmed from beginning, middle, and end body samples that the source and
  target have the same base language.
- `--stop-after-first-chapter`: schedule only the first substantive chapter.
- `--domain-id` or `--domain-manifest`: reuse one explicitly named domain;
  companion does not discover or build one.

`workers` is one global LLM-call concurrency budget shared by chapter
preparation, translation, commentary, evidence, and review. Changing it does
not invalidate content checkpoints.

The CLI deliberately performs no automatic language detection. The agent
running `workflows/companion.md` inspects substantive source body text near the
beginning, middle, and end, then compares normalized base languages. Language
tags are case-insensitive and `_` is equivalent to `-`: `EN_US`, `EN_UK`, and
`en-GB` are all `en`, while simplified and traditional Chinese are both `zh`.
Mixed or uncertain samples retain translation. The agent records
`source_language`, `source_base_language`, `target_language`,
`target_base_language`, `translation_mode`, and `translation_reason` in
`context.json`, and passes `--skip-translation` only for a clear same-language
decision.

Inspect and validate without changing generation state:

```bash
arc-companion status --project-dir <dir> --json
arc-companion validate --project-dir <dir> --json
```

### Step 2: Resume a supervised call

Routine accepted blocks automatically advance. A timeout, cancellation,
unknown submitted state, provider failure, or native session loss instead
returns `needs_supervision`; ARC never automatically repeats an uncertain paid
call.

```bash
arc-companion resume --project-dir <dir> --action resume-native --json
arc-companion resume --project-dir <dir> --action restart-generation \
  --confirm-possible-duplicate-charge --json
```

Prefer `resume-native` when the recovery context says the provider session is
resumable. `restart-generation` creates a new generation and requires explicit
confirmation because an uncertain submitted call may be billed twice.

### Step 3: Use background review checkpoints

Companion forwards provider progress and chapter/block lifecycle events to
`ARC_JOB_PROGRESS_FILE`. A long build emits a build-level `review_due` at the
next accepted boundary after every 30 minutes of cumulative runtime, even when
no individual call lasts that long.

```bash
arc-jobs watch <job-id> --until-review --json
```

The watch returns for inspection; it does not pause or cancel the build.

## Phase 3: Glossary and Chapter Preparation

### Step 1: Build the global glossary

If a real Index exists, preserve its complete hierarchy: every main entry,
subentry, page or range, `see`, and `see also`. There is no entry cap. Add
standard target-language terms and short explanations in bounded batches
without deleting, merging, or deduplicating source entries. Index pages render
once as the global glossary and never enter translation or commentary lanes.

Without an Index, use the concise terminology policy: at most 50 entries for
documents through 50 pages, 100 through 100 pages, and 200 for longer documents
or unknown page counts. Personal names remain in Latin script.

### Step 2: Project terms to chapters

A deterministic scanner uses only the chapter's source blocks plus Index page
intersection. It does not scan evidence or generated prose. A matching subentry
retains its parent. `chapter-glossary.json` stores every match. Prompts always
carry a compact source-to-target map; detailed explanations are block-local.
Mappings over 60 KiB are split into stateful setup turns without dropping terms.

### Step 3: Prepare guides and segments

Chapters may prepare concurrently under the global worker budget. In each
chapter, medium-tier segmentation and a high-tier stateful guide run in
parallel. A long chapter guide advances over bounded source windows before its
final synthesis. Guide fields are optional and selected for reader value:
motivation, contents, section logic, document position, prerequisites, and
supplementary reading.

Supplementary sources are deduplicated against the bibliography by DOI, arXiv
ID, and normalized title. Only controller-verified additions may appear, under
an explicit supplementary-reading label.

## Phase 4: Stateful Chapter Lanes

### Step 1: Advance in source order

After guide and segmentation validation, translation uses a medium-tier session
and commentary uses an independent high-tier session. The lanes may run in
parallel, while each advances strictly by segment order. With
`--skip-translation`, the translation lane is disabled completely: no
translation session, provider call, ledger, checkpoint, review overlay, or
migrated translation artifact is created. Guide, segmentation, glossary,
commentary, evidence resolution, and companion review are unchanged.
Review uses its commentary-only contract and rejects any proposed translation
patch. Legacy migration records a receipt saying that prior translations were
not migrated. Both chaptered builds and the legacy non-chaptered path support
this mode. Omitting the flag preserves the normal two-lane behavior.

The generation bootstrap carries fixed rules, guide, structure, chapter
glossary, and first source segment. Delta turns carry only the current segment,
block terms and evidence, cursor, source hash, and short instruction. Models do
not read project files. The static prefix is an audit hash, not a substitute for
the explicit bootstrap.

Every turn has a stable idempotency key. A repeated accepted turn replays its
recorded response without another provider call. The lane ledger records logical
call ID, input/output hashes, accepted-chain predecessor, session, generation,
native ID, usage, and validation receipt.

### Step 2: Validate before advancement

Each segment moves through `pending`, `submitted`, `schema_valid`,
`invariant_valid`, and `accepted`. ARC performs local JSON repair first, then
may send one aggregated correction turn in the same session. Validation covers
source order, block coverage, opaque tokens, language, protected names, and
registered evidence. Acceptance atomically updates the ledger and submits the
next block without placing the invoking agent in the critical path.

Commentary may request a bounded evidence round for its current segment. Only
controller-captured, auditable sources support external claims. Unsupported
claims are omitted, and evidence work cannot move behind the accepted cursor.

### Step 3: Roll over safely

At an accepted boundary, start a new generation when a session reaches 70% of
its known context window; use a conservative 128k-token estimate if unknown.
Send fixed context and a bounded continuity capsule, not the accepted source
prefix. If an earlier accepted block changes, retain the unchanged prefix,
invalidate the suffix, and rotate generation.

## Phase 5: Interactive Review and Rendering

### Step 1: Stop after the first chapter

With `--stop-after-first-chapter`, ARC must not schedule chapter two. It returns
`first_chapter_ready` only after the first substantive chapter's guide,
all enabled lanes, evidence, review, typesetting, and PDF validation complete.
Rerun without the flag after approval. The accepted first chapter is frozen and
cannot be silently rewritten by final consistency review. Its freeze record
includes the translation mode and uses a null translation hash in skip mode.
The first chapter artifact is not the full-document deliverable.

### Step 2: Render and validate

Each chapter contains exactly one guide after its title and before its first
segment. Normally each segment renders original, translation, then commentary.
In skip mode, typesetting receives `translations=None` and renders original
followed by commentary, with no translation layer. Styling, not visible
controller labels, identifies layers: reader text must not contain `译文`,
`伴读`, or `本段解释`; use `解释` when needed.

Source, guides, translations, commentary, and glossary use sans-serif text.
CJK fallbacks are Noto Sans CJK SC, Source Han Sans SC/CN, then
FandolHei-Regular. Mathematics and formula `\\text{}` remain LaTeX serif.
Invisible manifest or TeX markers validate hierarchy.

`arc-companion validate` checks exact coverage, glossary completeness, guide
placement, formulas, figures, tables, links, names, accepted ledgers, searchable
text, fonts, clipping, and overlap. In skip mode it also rejects translation IDs
in the manifest and translation markers in TeX. Deliver only the validated full
PDF.

Create a reproducibility package only when explicitly requested:

```bash
arc-companion package --project-dir <dir> --json
```
