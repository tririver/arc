# Companion-Reading Workflow

Use this workflow only for an explicit companion-reading request. It produces
an original-text, translation, and companion-commentary PDF for an arXiv paper
available through INSPIRE and ar5iv. Read `manuals/arc-companion.md` before
running a command.

## Phase 1: Set Up

### Step 1: Resolve the run

Follow the automation and project-directory rules in `rules/interaction.md`.
Set `workflow` to `companion`, preserve the requested paper identifier, and
record `annotation_language`, `provider`, `workers`, and `refresh` in
`<project-dir>/context.json`.

If the user did not specify an annotation language, print this notice before
any paper or LLM work and continue without asking:

```text
默认使用中文生成伴读；如需切换伴读语言，请直接指定目标语言。
```

Use `zh-CN` after printing the notice. This language applies to both the
translation and companion commentary; the source text remains in its original
language. Default to `workers=24`, `provider=auto`, and
`include_reproducibility_package=false`.

When the user explicitly supplies an existing domain, pass exactly one of
`--domain-id <id>` or `--domain-manifest <path>`. Never discover or build a
domain automatically for companion use.

### Step 2: Confirm the supported source

The initial implementation supports papers that INSPIRE can resolve to an
arXiv identifier and that have usable ar5iv full text. Do not substitute a
local PDF, OCR, publisher copy, or another version. Confirm that reproducing
the full source is authorized by the user or supported by a verifiable reuse
basis. Otherwise print `WARNING:` and stop.

## Phase 2: Build the Companion

### Step 1: Run the resumable build

```bash
arc-companion build <paper-id> \
  --project-dir <project-dir> \
  --annotation-language <language> \
  --provider <provider> \
  --workers <workers> \
  --json
```

Add `--recache` to rebuild the rich parsed document from cached ar5iv HTML and
retry missing assets. Add `--refresh` only when the user requested fresh
ar5iv/INSPIRE data. Do not use both.

The command must obtain the versioned rich document through `arc-paper` and
stop unless its integrity status is `complete`. It then starts two independent
tasks from the immutable paper source: medium-tier semantic segmentation and a
medium-tier comprehensive terminology glossary. The glossary records the English
term, its standard target-language equivalent, and a short target-language
explanation. Keep every personal name in its Latin-script source form,
including name roots in eponymous technical terms.
Keep only specialist concepts, methods, non-standard parameters, key
approximations, and translation-ambiguous terms useful to a field reader. Do
not pad the glossary; cap it at 50 entries through 50 pages, 100 through 100
pages, and 200 above 100 pages or when page count is unavailable.
Build the global protected-name inventory only from seed-paper metadata,
front matter, and glossary-recognized names. Ignore bibliography, reference,
and citer authors when generating that inventory; related-work authors remain
only in the per-unit evidence supplied to commentary.

Segmentation divides the immutable block order into section-aligned windows
and submits medium, stateless calls in parallel, bounded by `workers`. Each
call returns only unique internal cuts from the inventory's stable global
1-based ordinals; the controller adds window and paper endpoints and
deterministically constructs source ranges.
It locally refines any unit that exceeds the fine-grain hard limits, with at
most three refinement rounds for that unit. The 60,000-character guard
counts the serialized semantic source payload and excludes preservation-only
raw HTML from size accounting; this does not remove or alter any pinned source
content.

Require exact, ordered, one-time coverage and a validated glossary before unit
generation. Also use `arc-paper` to cache a bounded, relevance-selected set of
reference and citer full texts for evidence about prior and subsequent work;
select at most eight references and eight citers for the paper, then expose at
most two of each and at most 4,000 relevant source characters per related paper
to any one unit. Fall back to verified INSPIRE abstracts when a related full
text is unavailable.
Only the primary paper requests a rich document. References, citers, and
explicit-domain papers remain on lightweight parsed sections.

Start two independent bounded waves together. Translate every unit with the
low tier and generate companion commentary with the high tier; both receive
the same unit and frozen glossary. Each wave may use up to `workers` calls, so
the default is 24 concurrent translations plus 24 concurrent commentaries.
Each call also receives bounded full-paper navigation context: the paper's
section map plus compact neighboring and global anchors. On hosts that support
it, these two waves may use ARC-only MCP/cache access and internet lookup;
segmentation, glossary generation, and review remain tool-disabled.

Translation may use external access only to resolve standard terminology or
disambiguate the supplied source context. Its translation must remain a
complete, source-faithful rendering of the supplied blocks and must never add,
replace, or correct source content from an external page. Commentary is
optional for an already evident passage and discusses supported prior and
subsequent work only when this adds reader value. Never manufacture an
explanation merely to fill every unit. When explanation is useful, select the
relevant emphasis rather than mechanically covering a checklist: explain the
material's motivation and role in the argument, with motivation preferred at a
section or chapter opening; compare a useful alternative presentation in the
supplied references; cautiously flag deeper incompatibilities between sources
while treating mere convention, notation, normalization, and equivalent
formulations as differences rather than inconsistencies; or fill in non-evident
intermediate mathematics. Do not repeat or paraphrase an already clear source
passage. Keep stable evidence identifiers only in structured evidence fields
and manifests; never
show controller IDs or hashes to the reader. Reader-facing citations use the
source title plus a section or other human-readable location when available,
and at minimum the source title. Every external commentary claim must have captured
provenance before review and typesetting; omit a claim whose source cannot be
captured and verified. Register such material in the global `evidence.json`
registry and the unit's `segment-evidence/<segment>.json`; models may bind
claims only to controller-registered evidence IDs in structured output, never
to a URL or descriptor invented in their output.

Require each new primary low-tier translation call to check exact block
coverage/order, byte-exact opaque-token coverage/order, cross-block token
isolation, and protected-name spelling before returning. Keep existing valid
checkpoints because they already passed the unchanged deterministic validator;
do not invalidate all content merely to add this instruction checklist.

When a low-tier translation changes, drops, or reorders an opaque formula,
citation, or link token, collect every mismatched block in the segment into one
medium-tier repair call, using the same provider selection but no MCP or internet
access. Preserve all valid blocks. Do not
retranslate it. Give the specialized agent the prior text, its token-stripped
natural-language residue, inert source-run context, and `N+1` stable slot IDs
for `N` required tokens. Require the returned slots to concatenate byte-for-byte
to the prior residue; allow only exact insertion of explicitly missing protected
names, with no other textual change. The controller interleaves immutable source
tokens, preserves every other block byte-for-byte, and applies the unchanged
strict whole-block validation. Reject opaque content in slots, coverage/order
changes, rephrasing, or a second repair. Record the failure-only prompt version
and medium-tier route without changing the global prompt version or invalidating
valid content checkpoints.

Strip both well-formed and bounded malformed `[[ARC_INLINE:...]]` candidates
when deriving prior natural-language residue. Determine missing protected names
from natural text runs only; controller-owned link, citation, and formula content
must not trigger duplicate name insertion. Keep text runs separated, require
case-sensitive canonical Latin spelling, and verify insertion deltas with exact
name boundaries rather than substring counts. Run segment-wide coverage, type,
and conditional non-empty natural-residue preflight before invoking any slot
repair: require residue only when the source has natural text, and allow a pure
controller-owned link/citation block to remain token-only.

If a high-tier unit finds that a useful related-work claim needs an unregistered
source, it must leave that claim out and return at most two structured evidence
requests. Batch all requests after the first commentary wave. Run the ARC,
INSPIRE, and web-discovery verification lanes independently even when another
lane has already found a candidate. Register only validated, auditable source
content; a search snippet is discovery data, not claim evidence. Rerun only
units whose requests produced registered evidence, once, with the same high
tier. Clear unsupported related work after that round and never start a third
search/rerun cycle. Do not rerun translation, segmentation, glossary, or
unaffected commentary units.

MCP and web access are optional capabilities, not workflow prerequisites. If
the host cannot provide them, continue with the current segment, frozen
glossary, full-paper navigation context, and prepared evidence embedded in the
portable prompt. Preserve any capability diagnostic and do not infer
unsupported related-work claims. Then perform a high-tier whole-document review
of both tracks and render with the deterministic LaTeX pipeline. Never
reconstruct missing source text, equation numbers, tables, figures, or
bibliography with an LLM.

During review normalization and again during deterministic rendering, unwrap
reader-visible HTML/Markdown containers while preserving their meaningful body
text, discard machine-only container summaries, and replace legacy inline
evidence-ID markers with human-readable source-title and section/location
citations. Keep the original source blocks and structured evidence bindings
unchanged for audit.

Render every semantic unit in this order: original, translation, companion.
Use the layer styling itself to distinguish them; do not print a controller
unit heading (including its segment ID) or an `Original` field before each
source passage.
Before segmentation, exclude durable source-only table-of-contents blocks,
acknowledgment sections, and reference-list headings and entries from every LLM
lane, evidence input, and review payload. Keep title, author, and affiliation
blocks under the same existing non-generative front-matter policy. Render all
of this excluded material exactly once from the pinned source; preserve nested
TOC hierarchy and internal links instead of regenerating or translating them.
The renderer copies displayed formulas from the pinned source into the
translation for local readability but omits their equation numbers. It does not
repeat figures, tables, or other floating objects. Preserve every original
formula, visible equation number, figure, table, caption, citation, and
bibliography entry unchanged in the original track. Use distinct light
background colors for translation and companion commentary, following the
visual rhythm of the reference design. Keep original text on the plain page
without a background or left rule. Render paper headings with unnumbered
LaTeX section commands, preserving their source number and hierarchy through
explicit table-of-contents entries rather than adding a second number.

### Step 2: Handle interruption or failure

Rerun the same build command. Valid segmentation windows, glossary work,
refinements, translations, and commentaries are cached independently; the
final merged segmentation is cached only after exact coverage validation.
Translation and commentary lanes finish every submitted unit before aggregating
failures, preserving all successful checkpoints; a retry submits only missing
or stale units. Submit exactly the first `min(workers, unit_count)` source-order
units to both lanes as the first wave. Once both lanes finish that wave, the
pipeline must immediately render the source prefix through its final block,
source-fidelity check, compile, and PDF-validate the persistent first-round
preview before submitting any remaining unit, resolving evidence, or reviewing.
Inspect its `preview_pdf` path from build state when early visual QA is
requested. Treat it as diagnostic and never as the final deliverable. Preview
validation failure must stop the run at that boundary. If a traceback and checkpoint inventory show that an early
failure cancelled unrelated units, reproduce it with a minimal package test and
fix the scheduler in `packages/arc-companion`, never only in the current run
directory.
Checkpoints are keyed by source, asset, evidence, glossary, language, model,
prompt, schema, and workflow hashes, so only failed or stale work runs again.
Print every returned `WARNING:`. A segmentation warning must name the affected
window or local refinement and the rejected cut condition. Do not report the
first-round preview as the completed companion PDF.

Inspect progress without changing the run:

```bash
arc-companion status --project-dir <project-dir> --json
```

## Phase 3: Validate and Deliver

### Step 1: Validate independently

```bash
arc-companion validate --project-dir <project-dir> --json
```

Require exact ordered source-block coverage, original visible equation
numbers, table grids and spans, figure asset hashes and captions, bibliography
labels and order, resolved internal references, protected personal names,
translation and annotation-record coverage (with intentionally empty commentary
prose permitted), and a readable searchable PDF with valid
fonts and no detected clipping or overlap. Also require repeated translation
formulas to omit equation numbers and require the translation track not to
repeat floating objects.

### Step 2: Deliver only the PDF

Return the validated `<paper-safe>_companion_<language>.pdf`. Do not list
internal checkpoints, JSON, TeX, logs, or evidence files unless the user asks
or a warning requires explanation.

Only when the user explicitly requests a reproducibility package, run:

```bash
arc-companion package --project-dir <project-dir> --json
```

The ZIP contains the validated PDF and TeX, the primary paper assets used by
TeX, manifests, validation, and build state. It does not include cached related
reference/citer full texts. Test the ZIP before reporting it.

## Phase 4: Self-Reflection

Read `rules/self-reflection.md`. Record coverage, page count, warnings,
missing evidence, and improvement notes in the internal run artifacts without
putting implementation details in the companion commentary.
