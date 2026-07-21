# Companion-Reading Workflow

Use this workflow only for an explicit companion-reading request. It builds a
source-faithful PDF for a paper, lecture note, or book from a paired rich source
and PDF. Read `manuals/arc-companion.md` before running it. Use the CLI only;
MCP is not part of this workflow.

## Phase 1: Prepare the Run

### Step 1: Record intent

Follow `rules/interaction.md`. Set `workflow` to `companion` and record the
source identifier or path, annotation language, provider, workers, document
kind, and cache policy in `<project-dir>/context.json`.

If no annotation language is given, print the following notice and continue
with `zh-CN`:

```text
默认使用中文生成伴读；如需切换伴读语言，请直接指定目标语言。
```

The language applies to translation and commentary. Keep source text in its
original language. Default to `provider=auto` and `workers=24`. Reuse a domain
only when the user explicitly supplies `--domain-id` or `--domain-manifest`.

### Step 2: Establish the authoritative source

Require a rich Markdown/TeX/HTML source paired with its PDF. The rich source
provides faithful text, mathematics, links, tables, and assets; the PDF is
authoritative for document structure, printed page ranges, and boundary
reconciliation. Confirm that reproducing the source is authorized.

Run parsing with `--document-kind auto|article|book`. If automatic
classification is ambiguous, stop before any LLM call and ask for an explicit
kind. A paired-PDF cache without a PDF hash and reconciliation proof is stale
and must be recached. Never silently substitute rich-source headings when PDF
boundaries conflict or alignment is ambiguous.

## Phase 2: Build by Chapter

### Step 1: Start or resume the build

```bash
arc-companion build <source-id> \
  --project-dir <project-dir> \
  --annotation-language <language> \
  --document-kind auto \
  --provider <provider> \
  --workers <workers> \
  --json
```

Use `--recache` to rebuild cached parsing and PDF reconciliation, `--refresh`
only when fresh remote data was requested, and never both. Useful controls are
`--idle-timeout-seconds <seconds>` and `--regenerate-commentary`.

The controller must derive real chapters from the reconciled structure. An
article without substantive top-level sections is one chapter; a sectioned
article uses substantive top-level sections. For a book, exclude front matter,
bibliography, and Index, then choose a real repeated structural level without
inventing titles. Validate that chapters cover eligible source blocks exactly
once and that each chapter's semantic segments cover that chapter exactly once.

### Step 2: Build the glossary and chapter context

When the PDF contains a real Index, preserve every main entry, subentry, page
range, `see`, and `see also` relation as the global glossary. Do not cap,
deduplicate, merge, or place Index pages in ordinary generation lanes. Add a
standard target-language term and short explanation in bounded batches.

Without an Index, build a concise whole-document glossary capped at 50 entries
through 50 pages, 100 through 100 pages, and 200 above 100 pages or when page
count is unknown. Keep personal names in their Latin-script source form.

For each chapter, deterministically project only terms found in its source
blocks or whose Index page range intersects the chapter. Preserve a parent when
its subentry matches. Store the complete result in `chapter-glossary.json`.
Every block prompt receives the compact source-to-target mapping; detailed
explanations are limited to terms needed by that block. If the mapping exceeds
60 KiB, send bounded stateful setup turns instead of dropping terms.

### Step 3: Prepare each chapter

Chapters may run concurrently, subject to the single global `workers` budget.
Within each chapter, run medium-tier semantic segmentation and a high-tier
stateful guide session in parallel. Long guides advance through bounded source
windows, then produce an optional-field guide covering motivation, contents,
section logic, place in the document, prerequisites, and verified supplementary
reading when useful.

Deduplicate supplementary reading against the bibliography by DOI, arXiv ID,
and normalized title. Include only controller-verified sources and label them
as supplementary reading. Once guide and segmentation validate, write stable
chapter and segment IDs such as `ch-0001` and `ch-0001.seg-0001`.

### Step 4: Run ordered stateful lanes

Start independent translation and commentary sessions for the chapter.
Translation uses the medium tier; commentary uses the high tier. The two lanes
may run concurrently, but each lane advances strictly in segment order.

The bootstrap turn contains fixed rules, chapter guide, chapter structure,
chapter glossary, and the first segment. Later turns contain only the current
segment, its terms and evidence, cursor, source hash, and a short instruction.
The model does not read project files. Every turn uses a stable idempotency key.

Each segment advances through `pending`, `submitted`, `schema_valid`,
`invariant_valid`, and `accepted`. After acceptance, atomically update the lane
ledger and automatically submit the next segment. Validate coverage, order,
opaque tokens, protected names, language, and evidence locally. Attempt local
JSON repair first; aggregate remaining errors into at most one correction turn
in the same session. Do not use a paid formatter for routine blocks.

Resolve a commentary segment's bounded evidence requests before accepting that
segment. External claims require controller-captured provenance; omit unsupported
claims. Never let evidence resolution reorder or revisit an accepted cursor.

At a safe accepted boundary, roll over a session at 70% of a known context
window, or a conservative 128k-token estimate when unknown. The new generation
receives fixed context and a bounded continuity capsule, not the accepted
prefix. A changed earlier segment preserves the valid prefix, invalidates the
suffix, and starts a new generation.

### Step 5: Supervise exceptional states

Timeout, cancellation, unknown submission state, provider error, or native
session loss must stop automatic advancement and write `needs_supervision`.
Never automatically resubmit an uncertain paid call. Inspect without mutation:

```bash
arc-companion status --project-dir <project-dir> --json
```

Then explicitly choose native recovery or a new generation:

```bash
arc-companion resume --project-dir <project-dir> --action resume-native --json
arc-companion resume --project-dir <project-dir> --action restart-generation \
  --confirm-possible-duplicate-charge --json
```

Use restart only after accepting that an uncertain submitted call may be billed
twice. The ledger must retain call IDs, hashes, accepted-chain predecessor,
session and generation, native ID, usage, and validation receipt.

For background builds, forward provider progress plus `chapter_prepared`,
`block_accepted`, `chapter_complete`, and `needs_supervision` to
`ARC_JOB_PROGRESS_FILE`. Long builds emit build-level `review_due` at the next
safe boundary after each 30 minutes of cumulative runtime. This command returns
for inspection without pausing or cancelling the job:

```bash
arc-jobs watch <job-id> --until-review --json
```

### Step 6: Use the first-chapter review boundary

For interactive review, add `--stop-after-first-chapter`. Schedule only the
first substantive chapter. Return `first_chapter_ready` only after its guide,
both ordered lanes, evidence, chapter review, typesetting, and PDF validation
finish. Do not start chapter two. After approval, rerun without the flag; the
accepted first chapter remains frozen. Never present the first-chapter PDF as
the completed document.

## Phase 3: Render and Validate

### Step 1: Render the reader document

Place one chapter guide immediately after each chapter title and before its
first source segment. Render each segment as original, translation, then
commentary. Distinguish layers by styling without visible `译文`, `伴读`, or
controller labels; use `解释` where an explanation heading is needed.

Use sans-serif text for source, guide, translation, commentary, and glossary,
with CJK fallbacks `Noto Sans CJK SC`, `Source Han Sans SC/CN`, then
`FandolHei-Regular`. Keep mathematics and formula `\\text{}` in LaTeX serif.
Copy original text and source objects only from the pinned rich source.

### Step 2: Validate independently

```bash
arc-companion validate --project-dir <project-dir> --json
```

Require exact chapter and segment coverage, one guide per chapter in the right
position, faithful source objects and links, glossary completeness, accepted
lane ledgers, protected names, searchable text, valid fonts, and no clipping or
overlap. Use invisible manifest or TeX markers for hierarchy checks rather than
reader-visible labels.

### Step 3: Deliver

Return only the validated full-document PDF. Do not list internal JSON, TeX,
logs, ledgers, or evidence unless requested. Create a reproducibility package
only on explicit request:

```bash
arc-companion package --project-dir <project-dir> --json
```

## Phase 4: Self-Reflection

Read `rules/self-reflection.md`. Record coverage, page count, warnings,
supervision events, unresolved evidence, and improvement notes in run artifacts,
not in reader commentary.
