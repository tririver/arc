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

Removal of those legacy sidecars belongs to the T20 garbage-collection and
retention work. Until that cleanup runs, do not treat their presence as part of
the current PDF identity and do not delete them as an incidental render or
repair step. Current canonical TeX, PDF, source manifest, validation receipt,
reviewed-content objects, and published state remain protected artifacts.
