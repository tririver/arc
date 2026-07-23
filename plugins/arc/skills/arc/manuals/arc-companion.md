# ARC Companion Manual

`arc-companion` builds a source-faithful reading companion for a paper, lecture
note, or book. It consumes a rich source plus its paired PDF through
`arc-paper`, runs chapter-scoped stateful generation through `arc-llm`, and
renders and validates both a PDF and a static-web reader deterministically. The
core workflow is CLI-only and portable across supported agent hosts.

Chapter workers use addressed Controller Broker requests for ARC-paper reads by
default. This route is independent of nested-shell support and generic web
access: `--no-internet` disables WebSearch/WebFetch and provider internet, but
`--arc-paper-access full` may still let the Controller fetch missing paper data
through ARC-paper's declared providers. `--arc-paper-access none` removes the
request schema, catalog, controls, direct wrapper, and paper-network route.

Trusted direct access is a separate explicit opt-in with
`--arc-paper-direct-shell`. It fails before a provider call unless the runtime
proves a nested sandboxed shell, and exposes only policy-authorized catalog
operations declared as `network=none`; possibly networked reads remain on the
Controller route. It never switches provider or enables unsafe sandbox access.
They do not inherit the user's MCP,
skills, plugins, rules, or extra CLIs unless the run explicitly enables the
high-risk `inherit_host_tools` option. When internet access is enabled, the
commentary agent may use host search, while direct `arc-paper-worker` use still
requires the explicit runtime capability.

## Phase 1: Source and Structure

Repository workflows must resolve `--project-dir` and run
`git check-ignore -q --no-index <resolved-project-dir>` before creating any run
file. A failed check is blocking; choose a path under the repository's ignored
`arc-tests/` tree. Outside a Git worktree this preflight does not apply. The
generic CLI never invents a parent or sibling output directory.

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
- `--regenerate LANE`: repeat to rebuild selected lanes. `--regenerate review`
  marks every current Review segment for explicit regeneration and bypasses the
  whole-Review fast path, response checkpoints, legacy imports, and segment
  acceptances.
- `--regenerate-commentary`: deprecated alias for `--regenerate commentary`;
  reusable translations remain eligible.
- `--no-internet`: disable host search; commentary may cite only prompt-supplied
  sources or local ARC cache records that have usable URLs.
- `--arc-paper-access full|none`: enable the default Controller Broker or remove
  ARC-paper access completely. The default is `full`.
- `--arc-paper-direct-shell`: opt in to the trusted cache-only direct subset;
  unavailable capability is a preflight error, not a Controller fallback.
- `--arc-paper-child-llm-max-calls N`,
  `--arc-paper-child-llm-max-tokens N`, and
  `--arc-paper-child-llm-output-reserve-tokens N`: together opt in to managed
  child LLM/job operations. All three positive finite values and
  `--arc-paper-access full` are required. The private run-shared budget,
  reservations, tickets, and settlements persist across resume; unknown usage
  is charged at the reservation and uncertain submitted work needs supervision.
- `--skip-translation`: omit translation only after the workflow agent has
  confirmed from beginning, middle, and end body samples that the source and
  target have the same base language. This also disables bilingual glossary
  generation, reuse, migration, projection, prompt context, and output; a
  source Index remains source-only content.
- `--reference-translation-id <cached-id>`: use one cached parsed translation
  as a non-authoritative working draft without fetching or refreshing it.
- `--reference-translation-map <source-chapter-id>=<reference-chapter-id>`:
  repeat for a complete explicit map when strict automatic `1..N` leading
  ordinal alignment is unavailable. Reference mode cannot be combined with
  `--skip-translation`.
- `--source-language <BCP47-tag>`: pass the language established by the
  beginning/middle/end source sampling. The CLI does not detect it.
- `--user-intent <text>`: freeze the exact managed-run intent and generate one
  cached global guidance artifact shared by glossary, title, guide,
  translation, commentary, and segment-review workers. With authorized
  `--context-paper-id` values, only compact cached TOCs are supplied initially;
  workers later read exact selected chapters through a restricted read-only
  ARC paper policy.
- `--stop-after-first-chapter`: schedule only the first substantive chapter.
- `--recovery-policy auto|manual`: automatically recover eligible blocked
  translation/commentary lanes by default, or retain a supervised stop for an
  explicit recovery decision.
- `--max-auto-replacements N`: allow at most `N` fresh replacement generations
  for one blocked lane group; the default is `3` and the value is recovery
  state, not content identity.
- `--regenerate-segment LANE:SEGMENT_ID`: repeat to rebuild only a selected
  translation or commentary segment while locally revalidating its suffix.
- `--domain-id` or `--domain-manifest`: reuse one explicitly named domain;
  companion does not discover or build one.

`workers` is one global LLM-call concurrency budget shared by chapter
preparation, translation, commentary, and review. Changing it does
not invalidate content checkpoints.

Controller jobs persist their opaque ticket before waiting. Polling releases
the stateful lane turn and holds no provider concurrency permit, then reacquires
the turn before evidence finalization. Cancelling one waiter does not cancel a
deduplicated job that still has another active waiter.

The CLI deliberately performs no automatic language detection. The agent
running `workflows/companion.md` inspects substantive source body text near the
beginning, middle, and end, then compares normalized base languages. Language
tags are case-insensitive and `_` is equivalent to `-`: `EN_US`, `EN_UK`, and
`en-GB` are all `en`, while simplified and traditional Chinese are both `zh`.
Mixed or uncertain samples retain translation. The agent records
`source_language`, `source_base_language`, `target_language`,
`target_base_language`, `translation_mode`, and `translation_reason` in
`context.json`, and passes `--skip-translation` only for a clear same-language
decision. Pass the canonical sampled source tag through `--source-language`.

Inspect and validate without changing generation state:

```bash
arc-companion status --project-dir <dir> --json
arc-companion render-web --project-dir <dir> --json
arc-companion validate --project-dir <dir> --json
```

Status includes `current_phase`, `wait_reason`, active/queued/draining call
counts, `pending_call_count`, each pending call's submission state, recovery
action and blocking reason, `last_progress_at`, and persisted per-phase elapsed
time. Lock-file payloads are diagnostic only; the reported `active` field comes
from the OS advisory lock.

`render-web` manually rebuilds the static reader from durable checkpoints; it
does not repeat generation calls. `validate` checks both the PDF and web bundle.
Every successful full-document build and `render --format pdf|all` keeps its
immutable internal PDF revision and atomically maintains a byte-identical PDF
as the run-root delivery PDF in the resolved `--project-dir` itself, never its
parent. ARC records the delivery path and hash, repairs a missing or damaged
managed delivery on a completed build fast path without model or rendering
work, and never publishes a first-chapter preview at that path. ARC may adopt
an existing regular file only when its bytes already match the immutable
render. It replaces different content only when prior published ARC state
already owns that exact delivery path, and refuses unmanaged conflicts.
JSON output and state record the delivery as `output_run_pdf` plus
`output_run_pdf_sha256`.

### Step 2: Recover a blocked call

Routine accepted blocks automatically advance. With the default
`--recovery-policy auto`, ARC leaves accepted blocks unchanged and replays each
durable response through normal candidate selection, JSON normalization, and
the exact call-site business validation; only a response that passes that path
advances to acceptance. The owning handler records that exact control
acceptance immediately after its durable business checkpoint; finalization is
not a deferred acceptance mechanism. Reconstructing a complete failed raw
response promotes only the call checkpoint to `pending business validation and
application`; schema shape alone never accepts the control block and no repair
or alternate handler is called until normal business replay evaluates it. ARC
validates every receipt against the current registered ledger snapshot and uses
that snapshot's digest as the compare-and-swap precondition for each transition,
so a delayed generation-N callback cannot advance generation N+1. ARC correlates recovery records only by the complete
identity tuple: canonical ledger path, session key, logical unit, generation,
and idempotency key. Partial or conflicting tuples remain supervised. When a
native session is involved, its durable native-session ID must also match
exactly. Only after those gates, a call whose durable terminal progress is
specifically `idle_timeout` and has no complete response is never continued in
its old native session. ARC starts the original unresolved task in a fresh generation,
preserving the continuous accepted prefix and invalidating only the first
unaccepted block and its suffix. Other eligible failures retain native-first
recovery. Replacement requires an ARC-owned lane or side-effect-free submission
receipt and is bounded by the configured limit. An uncertain submitted call may
still be billed twice; the recovery journal records that audit warning rather
than claiming exactly-once provider execution.

```bash
arc-companion resume --project-dir <dir> --json
arc-companion resume --project-dir <dir> --action resume-native --json
arc-companion resume --project-dir <dir> --action restart-generation \
  --confirm-possible-duplicate-charge --json
```

The default `auto` action selects automatic recovery, even for a build that
previously used `--recovery-policy manual`. Explicit
`resume-native` requires the same complete identity tuple, is strict, and does
not upgrade to a replacement generation.
It deliberately retains old-session semantics even for a typed idle timeout.
Explicit `restart-generation` remains available after automatic restart budget
is exhausted and requires confirmation because an uncertain submitted call may
be billed twice. Cancellation, authentication, quota, rate-limit, missing
source, local I/O, and invalid configuration errors are not hidden by automatic
generation replacement; they remain in `needs_supervision`. Old native session,
runtime, and idempotency identities are required only by strict `resume-native`,
not by a fresh automatic replacement.

Status exposes separate bounded `control_identity` and `logical_identity`
projections and joins them only through the complete tuple above. Action-history
diagnostics are category/field projections with reason, error, and message text
redacted and bounded; prompts, responses, credentials, and arbitrary config
values are not status output.

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

Skip this step completely with `--skip-translation`. Do not inspect, migrate,
project, or expose a bilingual glossary. Record an explicit glossary
regeneration request as a zero-call `skipped` lane in the reuse plan, while
leaving any existing cache files untouched for a future translated build.

If a real Index exists, preserve its complete hierarchy: every main entry,
subentry, page or range, `see`, and `see also`. There is no entry cap. Add
standard target-language terms and short explanations in bounded batches
without deleting, merging, or deduplicating source entries. Index pages render
once as the global glossary and never enter translation or commentary lanes.

Without an Index, use the concise terminology policy: at most 50 entries for
documents through 50 pages, 100 through 100 pages, and 200 for longer documents
or unknown page counts. Personal names remain in Latin script.

### Step 2: Project terms to segments

A deterministic scanner uses only the current segment's original source blocks.
It does not scan sources or generated prose. Matching uses NFKC, case folding,
and Latin-letter/digit word boundaries. Source aliases are supported, and a
matching Index subentry retains its parent lineage. The translation and
commentary lanes receive the same ordered projection from the whole-document
glossary: canonical source and target terms plus only necessary aliases,
explanations, and protected names. These are the only two glossary levels; no
intermediate projection artifact or preparatory call is created.

### Step 3: Prepare guides and segments

Chapters may prepare concurrently under the global worker budget. In each
chapter, medium-tier segmentation and a high-tier stateful guide run in
parallel. A long chapter guide advances over bounded source windows before its
final synthesis. Guide fields are optional and selected for reader value:
motivation, contents, section logic, document position, prerequisites, and
supplementary reading.

Supplementary sources are deduplicated against the bibliography by DOI, arXiv
ID, and normalized title. Only additions whose primary page was actually read
may appear, under an explicit supplementary-reading label.

## Phase 4: Stateful Chapter Lanes

### Step 1: Advance in source order

After guide and segmentation validation, translation uses a medium-tier session
and commentary uses an independent high-tier session. The lanes may run in
parallel, while each advances strictly by segment order. With
`--skip-translation`, the translation lane is disabled completely: no
translation session, provider call, ledger, checkpoint, review overlay, or
migrated translation artifact is created. Guide, segmentation, commentary, and
companion review continue normally; bilingual glossary data is absent from
commentary and review prompts and from output.
Review uses its commentary-only contract and rejects any proposed translation
patch. Legacy migration records a receipt saying that prior translations were
not migrated. Both chaptered builds and the legacy non-chaptered path support
this mode. Omitting the flag preserves the normal two-lane behavior.

The generation bootstrap carries fixed rules, a compact chapter descriptor,
guide, static whole-document navigation, capability instructions, and the first
source segment. Delta turns carry only the current segment, its glossary
projection when translation is enabled, neighboring source anchors, bounded
sources already available for that segment, cursor, source hash, and a short
instruction. Models do not read project files. The static prefix is an audit
hash, not a substitute for the explicit bootstrap.

Every turn has a stable idempotency key. A repeated accepted turn replays its
recorded response without another provider call. The lane ledger records logical
call ID, input/output hashes, accepted-chain predecessor, session, generation,
native ID, usage, and validation receipt.

The bootstrap includes the compact, policy-filtered ARC-paper catalog once.
Workers request at most three Controller evidence rounds. Every complete
dispatch envelope is stored as a content-addressed object; responses at or
below 64 KiB may be inline, while larger results and authorized result files
use read-only handles. `artifact-read` returns verified base64 pages of at most
46 KiB and requires a new request ID for every new offset. Normal companion
policies exclude admin, destructive, LLM, and job operations; managed long jobs
remain outside this workflow until the dedicated managed-job layer.

### Step 2: Validate before advancement

Each paid call moves through `prepared/not_submitted`, `submitted`,
`response_received`, and `accepted`; segment validation moves through `pending`, `schema_valid`,
`invariant_valid`, and `accepted`. ARC performs local JSON repair first, then
may send one aggregated correction turn in the same session. Validation covers
source order, block coverage, opaque tokens, language, protected names, and
direct-citation structure. Acceptance atomically updates the ledger and submits the
next block without placing the invoking agent in the critical path.

For translation token-placement and missing-coverage repairs, an already
complete structured response is reused without another provider call when it
contains every exact requested block ID plus unrelated extra IDs. ARC discards
only those extras, collapses only canonically identical duplicates, restores
the requested order, and reruns the full response schema and translation
invariants. Missing IDs, conflicting duplicates, and legacy or invalid slot
shapes remain supervised. The original response stays in its attempt marker;
a body-free normalization receipt records the original/projected hashes,
discarded IDs, validator versions, and decision under the owning checkpoint.

At the same accepted boundary, ARC atomically publishes a refreshed reader
snapshot and hashed static bundle. Readers either see the previous complete
bundle or the new complete bundle, never a partially written update. A failed
web refresh does not discard the last valid bundle; run `render-web` to refresh
it manually after the underlying issue is fixed.

### Step 3: Reuse Review segments and arbitrate exact conflicts

Translated and commentary-only builds use one segment-addressed Review planner.
Each complete response must declare the exact `reviewed_segment_ids`; ARC
splits it only after current schema, exact coverage, singleton scope, block
ownership, patch-domain, trial-application, and translation or commentary
invariants pass. Empty findings and patches are valid negative coverage.
Commentary-only Review cannot propose translation. Provider call records,
provider/model/tier, workers, prompt budget, chunk topology, session, path,
time, and unrelated segments are not semantic identity.

Validated bodies are immutable project-local objects under
`.arc-companion/review-segments/objects/`; flat acceptances bind them to exact
segment identities without copying bodies. ARC prompts only uncovered,
changed, corrupt, invalid, or explicitly regenerated segments. It sends every
valid reused and new source to T15 exactly once in document and semantic-hash
order, and never follows that with a whole-document model Review.
Canonically equivalent sources collapse; non-equivalent sources remain
candidates. Supersession is accepted only when explicit, acyclic, and within
the same segment and exact T15 target. It is never inferred from age, order,
provider, or whether a source was reused.

Each non-null annotation field and each translation block is an independent
target. Canonically identical proposals merge their origins, proposals for
different targets coexist, and only different replacements for the same exact
target conflict. ARC validates candidates locally and applies valid
non-conflicting targets first. It materializes any changed translation as the
complete ordered block list, never as an incomplete replacement.

When conflicts remain, ARC sends all of them in exactly one low-tier, stateless,
offline arbitration call. ARC disables its internet, paper Broker/CLI, MCP
exposure, and inherited host-tool routes for this call. This is an ARC call
policy, not a sandbox or a claim that provider-intrinsic configuration or
capabilities are isolated. With no conflicts, ARC makes no arbitration call.

Before submission ARC writes a private partial recovery checkpoint containing
the validated non-conflicting merge. Terminal `no_conflicts`, `resolved`, and
`needs_supervision` receipts replay locally after exact identity and hash
validation, without another arbitration call. A recovered submitted response
returns through the same schema and semantic validation path. Missing,
malformed, foreign, invalid, or explicitly unresolved decisions preserve the
partial work and stop only the exact affected paths for operator supervision;
ARC does not silently choose a candidate or automatically research a
replacement. Decision receipts, supervision records, and status references are
body-light and expose safe relative paths and hashes rather than candidate
text.

With `--skip-translation`, Review uses the commentary-only schema and rejects
every translation target before arbitration. Review arbitration also respects
the accepted first-chapter freeze; neither a local merge nor an arbitration
decision may rewrite the frozen chapter.

Before any segment submission ARC atomically seals the body-free
`review-reuse-plan.json`. The global reuse plan records its hash, counts,
estimated calls, and ordered missing chunks. After terminal T15 arbitration and
acceptance publication, `review-reuse-receipt.json` records actual calls,
identity/source/acceptance and merged-output hashes, and a safe T15 receipt
link. Cross-run acceptance requires terminal `no_conflicts` or `resolved`;
`needs_supervision` remains recoverable only in the current run. The
whole-Review fast path additionally requires matching semantic output and valid
T15 and T16 receipts. Resume revalidates response, object, T15, acceptance, and
receipt checkpoints locally; tampering, corruption, unsafe paths, or hash
mismatches fail closed.

Legacy Review v3/v4 files are read-only and remain byte-identical. ARC imports
them only with exact unique coverage, recomputed historical input/prompt/schema
hashes, a matching prompt audit, a valid accepted global Review, and matching
terminal T15 proof. Otherwise it records `legacy_proof_unavailable` and
regenerates the affected current segments. After v1 acceptances exist,
invalidation is segment-local.

An internet-enabled commentary agent searches, reads, writes, and cites sources
within one generation turn. It should prefer papers, publishers, and official
primary pages, and must not use search-results pages or snippet-only aggregators
as final URLs. `commentary_sources` supports facts in explanation/commentary;
each `prior_work` or `later_work` item carries its own `sources`. Every source
requires a title, an HTTP(S) URL, and a reader-understandable locator, with at
most three distinct sources per claim. ARC validates structure and duplicate
sources only; it does not register claims or perform a second generation pass.
In `--no-internet` mode, unsupported external claims must be omitted.

Within one native chapter session, the commentary prompt asks the model to use
the existing conversation and prioritize information not already explained.
Delta turns do not resend old commentary. ARC does not create or persist
commentary summaries, covered points, or similarity checks. A rollover or
restart carries only hash-based continuity, so modest repetition is acceptable.

### Step 4: Roll over safely

At an accepted boundary, start a new generation when a session reaches 70% of
its known context window; use a conservative 128k-token estimate if unknown.
Send fixed context and a hash-based continuity capsule, not accepted source or
commentary prose. If an earlier accepted block changes, retain the unchanged prefix,
invalidate the suffix, and rotate generation.

## Phase 5: First-Chapter Review and Rendering

### Step 1: Stop after the first chapter

With `--stop-after-first-chapter`, ARC must not schedule chapter two. It returns
`first_chapter_ready` only after the first substantive chapter's guide,
all enabled lanes, review, PDF typesetting, static-web publication, and
validation complete.
Use this flag for `interactive` mode or a one-shot first-chapter checkpoint.
Rerun without the flag after approval; that approval does not switch the
managed run to `auto`. The accepted first chapter is frozen and
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

Translated builds render the original document/Part/Chapter/section/References/
Index title followed by its translation. PDF bookmarks and web navigation use
the translated title first without creating a second navigation entry. Titles
use a separate translation artifact and never enter commentary. Figure/table
captions and cited-paper titles remain source-only. Skip mode makes no title
translation call and renders each title once.

When translation is enabled, place the PDF glossary after references and before
the document end. In the single static `index.html`, place it once as the final
`#glossary` section and add its link after chapter navigation. Lazy-mount a
large glossary when its link is used or it approaches the viewport; hash
navigation and restored reading position must still work.

Mark matched source terms, source aliases, and target terms in original,
translation, and commentary text with low-saturation blue-gray text. Hover or
keyboard focus shows the canonical `source ↔ target` pair as a plain-text,
accessible tooltip. Do not split math, URLs, citation/link text, or KaTeX DOM.

Source, guides, translations, commentary, and glossary use sans-serif text.
Choose Noto/DejaVu fallbacks for common LTR scripts and SC/TC/JP/KR fonts for
CJK according to the source and target tags. Mathematics and formula
`\\text{}` remain LaTeX serif. HTML records per-layer language and direction;
RTL PDF layout remains best-effort and produces an explicit warning.
Invisible manifest or TeX markers validate hierarchy.

PDF and Web/HTML consume one canonical source-credit object and hash. Original
source author names are permanent and visible exactly once; an explicit,
reliable localized variant is only an adjacent secondary label and never
replaces the original. ARC may pair an explicitly selected cached reference
automatically only when both documents have exactly one author. Multiple
authors require a complete explicit mapping; ARC never pairs by position,
spelling, transliteration, affiliation, or model judgment. Affiliations and
profiles always come from the original source and are never translated,
completed, or rewritten from the reference.

Reliable source block anchors control credit placement in both outputs.
Front-matter anchors map to the title/front-matter group, while a distinct
profile block remains at that source block. Only unanchored metadata uses the
fallback immediately after the title, ordered as authors, affiliations, then
profiles. Identity-and-content hashes suppress duplicate source/metadata
projections without merging distinct equal strings. Cached localization
evidence is read through the strict current-cache path: a miss, stale or
malformed entry yields no localized label and performs no fetch, cache upgrade,
or LLM call. Both output manifests bind the shared hash, ordered identities,
placement facts, and exact visible counts.

Reference localization is currently an explicit programmatic build input, not
an inference from context papers: set `BuildOptions.source_credit_reference_id`.
For multiple authors, also set
`BuildOptions.source_credit_author_mapping` with complete
`source_author_id`, `reference_author_id`, and `reference_identity` records.
The command-line interface does not currently expose these fields.

Direct citations render with the source title linked to its URL and the locator
visible to the reader. The source manifest preserves each segment's citation
objects unchanged, including claim association.

The static reader is a self-contained directory rooted at `reader/index.html`.
It uses only manifest-declared local JavaScript, CSS, KaTeX, font, and media
files; viewing it must not fetch CDN or other network resources. Its snapshot
and data script are content-addressed, and publishing the HTML entry point is
the final atomic step.

`arc-companion validate` checks exact coverage, glossary completeness, guide
placement, formulas, figures, tables, links, names, accepted ledgers, searchable
text, fonts, clipping, and overlap. In skip mode it also rejects translation IDs
in the manifest and translation markers in TeX. It additionally validates web
manifest containment and hashes, offline asset closure, snapshot coverage, and
HTML source/translation/commentary ordering. Deliver the validated full PDF and
static reader; normal non-JSON output prints the run-root delivery PDF when present
and otherwise falls back to the internal PDF path.

The stable run-root delivery PDF is the preferred handoff path. It is written
directly inside the resolved `--project-dir`, not its parent. The internal
render revision remains the authoritative `output_pdf` used by validation and
reproducibility packaging.

Create a reproducibility package only when explicitly requested:

```bash
arc-companion package --project-dir <dir> --json
```

The package contains the PDF/TeX validation set and every web-manifest file,
including the HTML entry point, snapshot, JavaScript, CSS, local KaTeX/fonts,
and copied media. Packaging rejects escaping paths or hash mismatches.
