# ARC Companion

`arc-companion` builds and resumes source-faithful PDF and static-web companion
readers. The reviewed-content object is the authoritative semantic input; PDF
rendering and delivery are managed local artifact operations layered on top of
it.

## Validated PDF reuse

A final PDF revision is reusable only when all five identities match exactly:

1. the 64-character SHA-256 identity of the immutable reviewed-content object;
2. the PDF render recipe identity, including the current render contract and
   144-DPI validation policy;
3. the current PDF validator version;
4. one canonical PDF SHA-256 shared by the actual file, published state, and
   validation receipt; and
5. a successful, current, final-reusable validation receipt whose own file hash
   matches published state.

The matcher also requires nonempty regular TeX, PDF, source-manifest, and
validation-receipt files. Their recorded hashes must match, and every path must
remain lexically and physically inside the project without a symlink at the
leaf or in an intermediate component. A mismatch is a render miss with a
stable, body-free reason; it never weakens an identity check.

On an exact match, `arc-companion render --format pdf` reuses the immutable
revision and only repairs the run-root delivery when necessary. An already
correct delivery is a no-op. `--format all` reuses the PDF but still publishes
the Reader, while `--format web` is independent of PDF reuse. The completed
pipeline applies the same matcher, so an exact complete project needs no model,
TeX compiler, or Poppler call.

## Reader publication

Accepted Reader-visible changes are marked dirty immediately and coalesced by
one coordinator per build. Non-final publication occurs at most once per
60-second monotonic window; final and first-chapter completion bypass the wait
but still use semantic deduplication. The semantic identity covers the visible
snapshot, web render version, bundled Reader bytes, and the sorted source-asset
records while excluding only operational timestamps and machine-local paths.

Preparation performs no writes. Publication installs exact content-addressed
objects, validates the complete candidate, and switches `reader/index.html`
last as the sole mutable commit pointer. Restart inspection treats the actual
index as authoritative, validates its manifest, snapshot, data script,
coverage, and every asset, and can adopt or repair state without model or PDF
work. Exact semantic reuse does not rewrite web files or advance the committed
UTC timestamp.

## Legacy receipts and render-only upgrades

A pre-T17 receipt or a receipt without the current schema is classified as
`legacy_receipt`. Other corrupt or mismatched receipts retain their more
specific miss reasons. Existing PDF bytes are never promoted by writing a new
receipt around them.

When the completed project's non-PDF identities still match, ARC loads the
validated immutable reviewed-content object and performs a render-only upgrade:
it regenerates TeX, recompiles and revalidates the PDF, and publishes a new
current revision without calling a model. If that content object is missing or
invalid, the upgrade fails closed as `content_bundle_invalid`. After one
successful upgrade, a second unchanged PDF render can use the exact-match fast
path.

Explicit regeneration and forced builds do not use this completion shortcut.
Preview and first-chapter artifacts are not final-reusable and never publish a
run-root final PDF.

## Temporary validation artifacts

PDF validation checks metadata, searchable text, embedded fonts, font roles,
and every page rendered at 144 DPI. Extracted text and page rasters live inside
one temporary directory and are removed after success, validation failure,
cancellation, `KeyboardInterrupt`, or a middle-page error.

A successful durable receipt contains only bounded identities and an
allowlisted summary: result, page counts, DPI, PDF/text/raster byte counts,
encryption status, embedded-font count, and bounded font roles. It contains no
temporary paths, render paths, raw commands, or raw tool output. A failed
staging attempt may leave a small bounded diagnostic receipt, but no validation
text or raster.

## Historical sidecars and cleanup ownership

T17 stops creating durable validation text and page rasters; it does not delete
files produced by older renders. Historical files such as
`*.validation.txt` and `*.validation-page-<N>.png` remain untouched during PDF
matching, rendering, validation, packaging, and delivery repair.

Latest-only garbage collection runs after a newly published final build or
render. It retains the state-selected Reader graph, current validated PDF
revision, managed run-root PDF, checkpoints, reviewed-content objects, caches,
recovery journals, and provenance. It removes only strictly recognized
historical Reader objects and render revisions, plus exact legacy validation
sidecars. A no-op render, delivery repair, preview, first-chapter build, failed
build, or supervised build does not trigger cleanup.

Inspect the exact candidate set without writing, then optionally require that
same digest while applying:

```bash
arc-companion gc --project-dir <dir> --json
arc-companion gc --project-dir <dir> --apply \
  --candidate-digest <candidate-set-sha256> --json
```

Apply uses the project build lock, revalidates every candidate, quarantines by
atomic rename, and writes recoverable transaction and terminal receipt records
under `.arc-companion/gc/`. An interrupted transaction resumes forward on the
next apply. Unknown owned paths are reported and retained. Malformed recognized
paths, symbolic links, active builds, conflicting transactions, or changed
publication roots cause a stable refusal rather than speculative deletion.
`--extra-root <project-relative-path>` may be repeated to retain additional
hash-bound roots.

Current canonical TeX, PDF, source manifest, validation receipt,
reviewed-content objects, and published state remain protected artifacts.
