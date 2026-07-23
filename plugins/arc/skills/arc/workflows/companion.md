# Companion-Reading Workflow

Use this workflow only for an explicit companion-reading request. It builds a
source-faithful PDF and static-web reader for a paper, lecture note, or book
from a paired rich source and PDF. Read `manuals/arc-companion.md` before
running it. Use the CLI only; MCP is not part of this workflow.

## Phase 0: Classify the Request

### Step 1: Choose the least expensive valid operation

- Source text, target language, or relevant evidence changed: run `build` and
  inspect its reuse plan before any provider call.
- Prompt, model, tier, or another quality recipe changed: keep accepted
  content as `recipe_stale`; show the reuse plan and regenerate only an
  explicitly requested lane.
- Style, fonts, margins, LaTeX/CSS, renderer, validator, or web presentation
  changed: run `render`. If last-complete content exists, do not wait for a
  generation or research worker.
- The user requested checks only: run `validate`.
- The user requested delivery packaging only: run `package`.

Never turn a render, validation, or packaging request into a generation build.
Never spend tokens merely because a generation recipe changed.

## Phase 1: Prepare the Run

### Step 1: Select a safe run directory

Resolve `<project-dir>` before creating files. For every non-development ARC
run launched from inside a Git worktree, require the resolved run directory to
be ignored:

```bash
git check-ignore -q --no-index <resolved-project-dir>
```

Run this check before `arc-companion` or writing `context.json`. If it fails,
stop and select an ignored run directory; ARC repository runs belong under
`arc-tests/`. Do not create output in a repository root, `packages/`,
`plugins/`, or a sibling/parent of the selected run. Outside a Git worktree,
there is no Git-ignore preflight, but all output must still remain inside the
exact resolved `<project-dir>`.

### Step 2: Record intent

Follow `rules/interaction.md`. Set `workflow` to `companion` and record the
source identifier or path, annotation language, provider, workers, document
kind, cache policy, and the user's exact `user_intent` in
`<project-dir>/context.json`. Freeze that intent for recovery and resume.

If no annotation language is given, print the following notice and continue
with `zh-CN`:

```text
默认使用中文生成伴读；如需切换伴读语言，请直接指定目标语言。
```

The language applies to generated reader material. Keep source text in its
original language. Default to `provider=auto` and `workers=24`. Reuse a domain
only when the user explicitly supplies `--domain-id` or `--domain-manifest`.

### Step 3: Establish the authoritative source

Require a rich Markdown/TeX/HTML source paired with its PDF. The rich source
provides faithful text, mathematics, links, tables, and assets; the PDF is
authoritative for document structure, printed page ranges, and boundary
reconciliation. Confirm that reproducing the source is authorized.

Run parsing with `--document-kind auto|article|book`. If automatic
classification is ambiguous, stop before any LLM call and ask for an explicit
kind. A paired-PDF cache without a PDF hash and reconciliation proof is stale
and must be recached. Never silently substitute rich-source headings when PDF
boundaries conflict or alignment is ambiguous.

### Step 4: Decide whether translation is needed

After parsing, the agent running this workflow must inspect substantive body
text near the beginning, middle, and end of the rich source. Do not decide from
the title, abstract alone, metadata, table of contents, Index, bibliography, or
filename. The program does not detect the source language.

Compare source and target by base language. Normalize tags case-insensitively,
convert underscores to hyphens, and compare their primary language subtags:
`EN_US`, `EN_UK`, and `en-GB` are `en`; simplified and traditional Chinese
tags are both `zh`. Use the analogous primary-subtag rule for other languages.
Only choose `translation_mode=skip` when all three samples clearly have the
same base language as the target. Mixed-language or uncertain source text must
use `translation_mode=translate`.

Record `source_language`, `source_base_language`, `target_language`,
`target_base_language`, `translation_mode`, and a short `translation_reason`
in `<project-dir>/context.json`. The reason should identify the beginning,
middle, and end sampling result without copying long source passages.

## Phase 2: Build by Chapter

### Step 1: Start or resume the build

```bash
arc-companion build <source-id> \
  --project-dir <project-dir> \
  --source-language <source-language-tag> \
  --annotation-language <language> \
  --document-kind auto \
  --provider <provider> \
  --workers <workers> \
  --recovery-policy auto \
  --user-intent '<exact context.json user_intent>' \
  --json
```

When Phase 1 selected `translation_mode=skip`, add `--skip-translation` to the
build command. Otherwise omit it. Never pass the flag merely because source
metadata, a title, or a short excerpt appears to match the target language.
The CLI does not add a second language detector; pass the canonical BCP-47 tag
from the sampled agent decision as `--source-language`. This decision remains
the authoritative mode signal.

Use `--recache` to rebuild cached parsing and PDF reconciliation, `--refresh`
only when fresh remote data was requested, and never both. Useful controls are
`--idle-timeout-seconds <seconds>`, `--regenerate-commentary`, and
`--no-internet`. The last flag disables host search for commentary turns.

The controller must derive real chapters from the reconciled structure. An
article without substantive top-level sections is one chapter; a sectioned
article uses substantive top-level sections. For a book, exclude front matter,
bibliography, and Index, then choose a real repeated structural level without
inventing titles. Validate that chapters cover eligible source blocks exactly
once and that each chapter's semantic segments cover that chapter exactly once.

### Step 2: Build the whole-document glossary and segment projections

When `--skip-translation` is active, skip this step: do not generate, reuse,
migrate, register, project, inject, or render bilingual glossary data. Preserve
old cache files without making them visible. Record an explicit glossary
regeneration request as `status=skipped`, reason
`glossary_disabled_for_same_language_source`, with zero estimated calls. Keep
the source document's Index as source-only content.

When the PDF contains a real Index, preserve every main entry, subentry, page
range, `see`, and `see also` relation as the global glossary. Do not cap,
deduplicate, merge, or place Index pages in ordinary generation lanes. Add a
standard target-language term and short explanation in bounded batches.

Without an Index, build a concise whole-document glossary capped at 50 entries
through 50 pages, 100 through 100 pages, and 200 above 100 pages or when page
count is unknown. Keep personal names in their Latin-script source form.

Before each translation or commentary turn, deterministically scan only that
segment's original source blocks and project matching entries from the
whole-document glossary. Do not scan generated prose or external sources. Use
NFKC, case folding, and Latin-letter/digit word boundaries. Preserve glossary
order and stable IDs; when an Index subentry matches, retain its parent lineage
as metadata. Both lanes receive the same segment projection, containing only
the canonical source and target terms plus necessary aliases, explanation, and
protected names. The only two levels are the whole-document glossary and the
current segment projection; no intermediate artifact or preparatory call is
created.

### Step 3: Prepare each chapter

Chapters may run concurrently, subject to the single global `workers` budget.
Within each chapter, run medium-tier semantic segmentation and a high-tier
stateful guide session in parallel. Long guides advance through bounded source
windows, then produce an optional-field guide covering motivation, contents,
section logic, place in the document, prerequisites, and verified supplementary
reading when useful.

Deduplicate supplementary reading against the bibliography by DOI, arXiv ID,
and normalized title. Include only sources whose primary page was actually
read and label them
as supplementary reading. Once guide and segmentation validate, write stable
chapter and segment IDs such as `ch-0001` and `ch-0001.seg-0001`.

### Step 4: Run ordered stateful lanes

Normally, start independent translation and commentary sessions for the
chapter. Translation uses the medium tier; commentary uses the high tier. The
two lanes may run concurrently, but each lane advances strictly in segment
order. With `--skip-translation`, do not start or resume a translation session;
do not pass glossary data to commentary or review; the commentary lane and all
non-glossary chapter preparation continue normally.

In skip mode, review must use the commentary-only branch and reject any
translation patch. Do not migrate an old translation; record that decision in
the migration receipt. These requirements apply to both chaptered builds and
the legacy non-chaptered path. Omitting the flag preserves the two-lane default.

The bootstrap turn contains fixed rules, a compact chapter descriptor, chapter
guide, static whole-document navigation, capability instructions, and the
first segment. When `user_intent` is nonempty, ARC first generates one shared
global intent-guidance artifact from the intent plus authorized cached
reference metadata and compact TOCs. Every content worker receives that same
guidance in its bootstrap; segmentation, repairs, rendering, and validation do
not. Workers with `nested_sandboxed_shell=true` may read only
guidance-selected exact chapter locators through `arc-paper-worker
get-parsed-toc`, `get-parsed-section`, and read-only artifact pagination. Other
runtimes use the Controller evidence-request fallback without changing
provider or sandbox mode. Reference text may guide terminology, idiom, and style, but the
original source remains authoritative for facts, coverage, and structure.

Later turns contain only the current segment, its glossary
projection when translation is enabled, neighboring source anchors, bounded
sources already available for that segment, cursor, source hash, and a short
instruction. The model does not read project files. Every turn uses a stable
idempotency key.

Each segment advances through `prepared/not_submitted`, `submitted`,
`response_received`, `schema_valid`, `invariant_valid`, and `accepted`.
Only the provider transport barrier may mark `submitted`; local schema,
configuration, and capacity failures remain safely retryable. After acceptance, atomically update the lane
ledger and automatically submit the next segment. Validate coverage, order,
opaque tokens, protected names, language, and citation structure locally. Attempt local
JSON repair first; aggregate remaining errors into at most one correction turn
in the same session. Do not use a paid formatter for routine blocks.

An internet-enabled commentary agent searches, reads, writes, and returns
direct citations in the same generation turn. Prefer papers, publisher pages,
and official primary pages; never cite a search-results page or an aggregator
snippet as the final URL. Each source requires `title`, an HTTP(S) `url`, and a
reader-understandable `locator`; a claim may cite at most three distinct
sources. ARC validates only this structure and does not register claims or run
an extra citation-resolution or rewrite pass. With `--no-internet`, cite only sources
already supplied in the prompt or present in the local ARC cache with usable
URLs, and omit unsupported external claims.

The commentary prompt asks the model to use the chapter history already in the
native session and favor new value for the current segment. ARC does not send
old commentary in delta turns and does not generate, compare, or persist
summaries or covered-point lists.

At a safe accepted boundary, roll over a session at 70% of a known context
window, or a conservative 128k-token estimate when unknown. The new generation
receives fixed context and a hash-based continuity capsule, not accepted source
or commentary prose. Some repetition after rollover or restart is acceptable.
A changed earlier segment preserves the valid prefix, invalidates the
suffix, and starts a new generation.

### Step 5: Recover eligible blocked lanes

The default `--recovery-policy auto` first replays durable response candidates
through normal candidate selection, JSON normalization, and the exact call-site
business validator, accepting only candidates that pass. The owning handler
must durably accept its exact control before returning; a final sweep cannot
substitute for that handoff. A reconstructed failed raw response remains
`pending business validation and application`: schema validity does not accept
the block and does not invoke a repair or alternate handler. Before any
transition, validate the receipt against the current registered ledger and use
the returned ledger digest as the compare-and-swap precondition; re-read the
digest after each transition. A callback or receipt from an older generation
must never advance the current generation. ARC correlates an
indexed receipt, call, and native-resume authorization only when canonical
ledger path, session key, logical unit, generation, and idempotency key all
match; any durable native-session ID must match too. Partial tuples remain
supervised. After these gates, a durably typed `idle_timeout` with no complete response
suppresses all old-session reconciliation, preserves the accepted prefix, and
submits the original first-unaccepted task in one fresh generation.
Other eligible failures retain native-first recovery. ARC starts at most three
replacement generations by default and only for structurally owned,
side-effect-free recovery units.
`--max-auto-replacements N` changes that recovery budget without changing
content fingerprints. Inspect recovery state without mutation:

```bash
arc-companion status --project-dir <project-dir> --json
```

Bare `resume` selects automatic recovery. Alternatively, explicitly request
strict native recovery or a confirmed new generation:

```bash
arc-companion resume --project-dir <project-dir> --json
arc-companion resume --project-dir <project-dir> --action resume-native --json
arc-companion resume --project-dir <project-dir> --action restart-generation \
  --confirm-possible-duplicate-charge --json
```

Explicit native recovery requires the same complete identity tuple and must not
upgrade automatically. Use explicit restart
only after accepting that an uncertain submitted call may be billed twice.
`resume-native` keeps strict old-session behavior even after an idle timeout;
auto does not claim exactly-once execution for an ambiguous submitted call.
Cancellation, authentication, quota, rate-limit, missing-source, local-I/O, and
invalid-configuration failures remain supervised rather than being masked by
replacement. Possible duplicate charging remains an audit warning under auto;
strict native identity checks apply only to `resume-native`. The ledger must
retain call IDs, hashes, accepted-chain predecessor,
session and generation, native ID, usage, and validation receipt.

Status joins calls, controls, and transaction entries only through that exact
five-field identity, exposes separate bounded control/logical projections, and
redacts/bounds action-history reason, error, and message fields. Never include
prompt text, response bodies, credentials, or arbitrary configuration values in
observability output.

For background builds, forward provider progress plus `chapter_prepared`,
`block_accepted`, `chapter_complete`, and `needs_supervision` to
`ARC_JOB_PROGRESS_FILE`. Long builds emit build-level `review_due` at the next
safe boundary after each 30 minutes of cumulative runtime. This command returns
for inspection without pausing or cancelling the job:

```bash
arc-jobs watch <job-id> --until-review --json
```

### Step 6: Use the first-chapter review boundary

In `interactive` mode, or for a one-shot request to stop after chapter one, add
`--stop-after-first-chapter`. Schedule only the first substantive chapter.
Return `first_chapter_ready` only after its guide,
all enabled ordered lanes, chapter review, PDF typesetting, static-web
publication, and validation finish. Do not start chapter two. After approval, rerun without the
flag after approval; a bare approval passes this checkpoint but does not change
`automation_level`. The accepted first chapter remains frozen. Never present the first-chapter
PDF or web reader as the completed document.

## Phase 3: Render and Validate

### Step 1: Render the reader document

Place one chapter guide immediately after each chapter title and before its
first source segment. Normally render each segment as original, translation,
then commentary. With `--skip-translation`, render original followed by
commentary and do not create a translation layer. Distinguish layers by styling
without visible `译文`, `伴读`, or controller labels; use `解释` where an
explanation heading is needed.

When translation is enabled, translate the document title, Part and Chapter
titles, every section heading level, and source-only structural headings such
as References and Index. Display the source title followed by its translation,
and use the translation as the primary navigation label. Title translation is
a separate document-level lane: never send titles to commentary or chapter
guide generation. Keep figure/table captions and cited-paper titles unchanged.
With `--skip-translation`, render each source title once and make no title
translation call.

For translated output, put the PDF glossary after references and before the
document end. Keep one web entry point: append a standalone `#glossary` section
to `index.html`, link it after the chapters in the sidebar, and lazy-mount large
glossaries near the viewport or on link activation. Hash navigation, refresh,
and restored reading position must remain valid.

In original, translation, and commentary prose, mark deterministic matches for
source terms, source aliases, and target terms with subtle blue-gray text and a
plain-text `source ↔ target` tooltip available on hover and keyboard focus.
Never split math, URLs, citation/link text, or KaTeX-rendered DOM.

Use sans-serif text for source, guide, translation, commentary, and glossary.
Select offline font fallbacks from both source and target language: common LTR
scripts use Noto/DejaVu, and CJK uses the appropriate SC/TC/JP/KR family. Keep
mathematics and formula `\\text{}` in LaTeX serif. HTML must mark each layer's
`lang` and `dir`; RTL PDF layout is best-effort and must emit a warning rather
than claiming full bidi support.
Copy original text and source objects only from the pinned rich source.
Render every direct citation with its title linked to the source URL and its
locator visible. Preserve the per-segment citation objects unchanged in the
source manifest.

After each segment is accepted and checkpointed, atomically refresh the static
reader snapshot and hashed bundle. Never expose a partially written bundle; on
refresh failure, preserve the previous valid reader. The reader must reference
only local JavaScript, CSS, KaTeX, fonts, and copied media, with no CDN or
runtime network dependency. Manually rebuild it from durable checkpoints
without repeating LLM work when requested:

```bash
arc-companion render-web --project-dir <project-dir> --json
```

### Step 2: Validate independently

```bash
arc-companion validate --project-dir <project-dir> --json
```

Require exact chapter and segment coverage, one guide per chapter in the right
position, faithful source objects and links, glossary completeness, accepted
enabled-lane ledgers, protected names, searchable text, valid fonts, and no
clipping or overlap. In skip mode, require that the manifest contains no
translation IDs and TeX contains no translation markers. Use invisible
manifest or TeX markers for hierarchy checks rather than reader-visible labels.
Also require web-manifest path containment and hashes, complete local asset
closure, snapshot coverage, and the same source/translation/commentary order as
the PDF.

### Step 3: Deliver

After a successful full-document build or PDF rerender, ARC keeps the
validated immutable render revision and atomically copies the same PDF to the
resolved `<project-dir>` itself, never its parent. This run-root delivery PDF is
the convenient user-facing delivery; a first-chapter preview must not create or
replace it. Return that copy and the static reader, and lead with the run-root
delivery PDF path when it is available. The immutable internal `output_pdf`
remains authoritative for validation and packaging. Do not list internal JSON,
TeX, logs, ledgers, or citation data unless requested. Create a reproducibility
package only on explicit request:

```bash
arc-companion package --project-dir <project-dir> --json
```

The package must include the PDF validation set plus every manifest-declared
web file: HTML, snapshot, JavaScript, CSS, local KaTeX/fonts, and media. Reject
paths outside the project and any hash mismatch.

## Phase 4: Self-Reflection

Read `rules/self-reflection.md`. Record coverage, page count, warnings,
supervision events, citation warnings, and improvement notes in run artifacts,
not in reader commentary.
