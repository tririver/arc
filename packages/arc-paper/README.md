# arc-paper

`arc-paper` owns deterministic paper identifiers, arXiv/INSPIRE acquisition,
metadata, references, citers, paper-cache administration, and paper-specific
LLM workflows. Provider-neutral source storage, HTML/Markdown/TeX/PDF parsing,
rich-document contracts, document structure, full-text and equation search,
and approximate keyword inventory come from `ac-document` and remain exposed
through the compatible `arc-paper` API. Generic model execution and durable-run
mechanics remain in `ac-llm` and `ac-jobs`.

## Running the CLI

An installed package provides the `arc-paper` console script. The ARC Skill
runtime is the portable fallback; inside an ARC source checkout, the package
virtual environment is a direct development fallback:

```bash
arc-paper --help
<skill-dir>/scripts/arc-runtime arc-paper --help
packages/arc-paper/.venv/bin/arc-paper --help
```

Use the first available launcher consistently in later commands.

## Quick start

Fetch normalized metadata for one paper:

```bash
arc-paper get-metadata "<paper-id>"
```

Replace the quoted placeholder with an arXiv, INSPIRE, or DOI identifier.
Use `arc-paper --help` and `arc-paper get-metadata --help` for the current
paper, source, search, workflow, and cache commands.

Remote arXiv HTML is not necessarily a single file. Materialize the primary
HTML with its safe authored SVG and image targets before local document export:

```bash
arc-paper export-arxiv-html-bundle arXiv:0911.3380 \
  --output-dir ./paper-source \
  --cache-root ./.arc/cache/arc-paper

ac-document export-rich-document ./paper-source/source.html \
  --output-dir ./render-input \
  --cache-root ./.ac/cache/ac-document
```

The first command writes exact `source.html`, a versioned `manifest.json`, and
verified resources at safe authored relative paths. The second command remains
provider-neutral and network-free. Released `ac-document` 2.0.1 consumes
`img[src]` resources but does not yet admit `object[data]` SVG files as
RichDocument assets; that downstream handoff requires a later ac-document
release. Use a compatible exporter output for direct `alc-render` composition:

```bash
ac-document export-rich-document ./note.md \
  --output-dir ./render-input \
  --cache-root ./.ac/cache/ac-document

alc-render compose \
  --source ./render-input/rich-source.json \
  --metadata ./render-input/metadata.json \
  --output ./render-input/publication.json
alc-render render \
  --publication ./render-input/publication.json \
  --html ./render-input/reader.html
```

`export-rich-document` accepts Markdown, HTML, or flattened TeX and an optional
PDF `--validator`. It refuses a nonempty output directory, copies verified
assets below `resources/`, and returns the source and metadata paths at
`data.source` and `data.metadata`.

### HTML dependency bundle contract

`fetch-arxiv-auto` and reference acquisition keep their existing single-file
behavior. Dependency downloads are explicit through
`fetch-arxiv-html-bundle` or `export-arxiv-html-bundle`; this prevents an old
caller from unexpectedly turning one provider request into many. Official
arXiv HTML falls back to ar5iv only when the official HTML endpoint returns a
deterministic 404 for an unversioned/latest request.

An explicit arXiv revision such as `arXiv:2608.20415v1` is an exact bundle
identity. The official request and source/dependency cache keys retain `v1`,
and the final document URL plus authored revision signals must agree with that
version. The accepted authored signals are an exact versioned HTML base root
or the paired official abstract/PDF header links. Missing or conflicting
signals fail closed. An exact-version 404 is not retried through unversioned
ar5iv. Unversioned IDs keep the existing latest/canonical behavior, and the
legacy single-file `fetch-arxiv-auto` path is unchanged.

Bundle schema `arc.paper.html_source_bundle.v2` records the primary source,
validated final document URL, effective authored base URL, canonical bundle
digest, exact acquisition policy, ordered dependency records, and structured
warnings. Each record keeps
the authored element/attribute/target, request and final response URLs,
declared and response media types, SHA-256 digest, byte size, availability, and
typed failure provenance. One failed dependency never discards a valid primary
HTML response or fabricates a resource.

`export-arxiv-html-acquisition` is the separate ALC handoff path. It writes a
strict `ac.document.html_source_export.v1` manifest from ARC's internal ACF
sidecar and may rewrite an available absolute target to a safe local asset
path. `export-arxiv-html-bundle` remains the v2-compatible export: it preserves
the primary bytes and leaves non-relative targets as warnings. Older v2-only
caches use the handoff fallback only with an explicit sidecar-missing warning.

Extraction uses the LaTeXML `article.ltx_document` root when present and
otherwise the full document. Version 1 supports `object[data]`, `img[src]`, and
`source[src]`; `img/source[srcset]` is retained as an explicit unsupported
warning because browser candidate selection is not represented by this schema.
Only credential-free HTTPS URLs on the provider's own host and default port are
requested. Authored fragments, unsafe schemes, cross-host bases, and cross-host
redirects are rejected before the next request. Every redirect hop is checked.

The defaults are 256 dependency records, 25 MiB per unique resource, 200 MiB
total unique bytes, and five redirects. Supported media are external SVG, PNG,
JPEG, GIF, WebP, and AVIF with declared type, filename extension, and response
`Content-Type` agreement. SVG bytes are preserved as external files; ARC does
not execute, sanitize, or inline them.

Bundle metadata uses separate digest-verified remote JSON cache entries named
`arxiv-html-dependencies` and `ar5iv-html-dependencies`; primary HTML retains
the existing source cache format. The first explicit bundle request against an
older main-only cache refetches the primary once because the old mapping does
not contain a verified final response URL. Replay validates the current primary
and every resource, and requires the exact same count, per-resource byte,
document byte, and redirect limits. Schema v1 sidecars did not record these
limits and are therefore reacquired rather than replayed. Refresh and
corrupt-resource repair use the existing
cooperating-process leases and atomic manifests. Cache archive export includes
all reachable bundle resources. Removal drops mappings, but shared resource
blobs are retained because no reference-counted garbage-collection contract
exists.

Only safe scheme-less authored relative targets are materialized beside
`source.html`; absolute and traversing targets remain structured warnings.
Importing or exporting an ordinary local HTML file never invokes a provider.
An unversioned schema v2 bundle cannot be relabeled as an explicit revision:
its URL, origin metadata, cache key, policy, and bundle digest bind the original
provenance. Schema v1 bundles lack policy identity and are not replayed by the
v2 codec; reacquire instead of editing their cache or manifest.

## Cache portability

Cache entries are logical folders with stable IDs returned by `cache list`.
Export any exact selection, or the entire cache, to a verified tar.gz archive:

```bash
arc-paper cache list
arc-paper cache export <entry-id> [<entry-id> ...] --output selected.tar.gz
arc-paper cache export --all --output complete-cache.tar.gz
```

An export never overwrites an existing archive. Import validates and stages
the complete archive before changing the destination:

```bash
arc-paper cache import selected.tar.gz
arc-paper cache import selected.tar.gz --replace-conflicts
```

Identical files are reused and destination-only files are preserved. A
differing destination file rejects the whole import by default, with all
conflict paths reported and no writes performed. `--replace-conflicts`
explicitly replaces only those differing files after the same preflight.

Cached Markdown can retain its exact source identity while adopting hierarchy
from an independently cached PDF outline:

```bash
arc-paper reconstruct-cached-structure \
  --document-ref '<Markdown CachedDocumentRef JSON>' \
  --outline-document-ref '<PDF CachedDocumentRef JSON>'
arc-paper get-section \
  --document-ref '<Markdown CachedDocumentRef JSON>' \
  --structure-ref '<CachedDocumentStructureRef JSON>' \
  '<section id or exact title>'
arc-paper extract-keywords ./book.md \
  --project-dir ./project \
  --structure-ref '<CachedDocumentStructureRef JSON>' \
  --section-id '<content section id>' \
  --section-id '<another content section id>' \
  --host-authority unrestricted
```

With `--structure-ref`, keyword extraction groups text by the reconstructed
content hierarchy instead of unreliable native heading levels. Each model
request receives only its selected section text; the complete parsed document
and its assets are not attached.

References use exact identities and verified cache handles:

```bash
arc-paper lookup-reference --doi '10.1234/example'
arc-paper acquire-reference --url 'https://example.org/paper'
arc-paper admit-reference ./paper.pdf --doi '10.1234/example'
arc-paper materialize-reference \
  --resource-ref '<CachedResourceRef JSON>' \
  --output ./agent-workspace/paper.pdf
```

`arc-paper extract-keywords` accepts `--host-authority` for its LLM work. Use
`unrestricted` only when the host explicitly reports unrestricted authority;
otherwise use `unknown`. Reuse the identical value if the durable keyword run
is resumed. Authority is runtime-only rather than part of the keyword
inventory identity.

LLM workflow schemas bind request-local identities such as the requested
summary section and explicit-term source IDs. Cross-field source-coverage
errors that cannot be expressed in JSON Schema receive one deterministic fresh
retry with bounded validation feedback. If an explicit-term review remains
machine-invalid, that non-essential explicit field is discarded with a durable
warning and chapter extraction continues. A valid scientific or document
quality result such as `status="unusable"` is not a machine-output failure and
does not trigger this retry; its existing supervision path is preserved.

## Python API

The deterministic service exposes the same operation:

```python
from arc_paper import ArcPaperService

paper_id = input("Paper ID: ")
metadata: dict[str, object] = ArcPaperService().get_metadata(paper_id)
print(metadata.get("title", ""))
```

Repository-backed parsing and LLM-backed workflows are also available through
their public package facades.

### Remote HTML source bundles

```python
from arc_paper import ArcPaperService

papers = ArcPaperService(cache_root="./.arc/cache/arc-paper")
bundle = papers.fetch_arxiv_html_bundle("arXiv:0911.3380")
available = [
    dependency
    for dependency in bundle.dependencies
    if dependency.availability == "available"
]
result = papers.export_arxiv_html_bundle(
    "arXiv:0911.3380",
    output_dir="./paper-source",
)
```

`HtmlSourceBundle`, `HtmlDependency`, `HtmlDependencyWarning`, their document
codecs, and the standalone fetch/export functions are public. Ordinary local
`ac-document` parsing remains network-free. Explicit web acquisition belongs to
ACF's dedicated API and ARC adapts it only after paper-provider identity and
revision validation.

### Reference material cache

Reference acquisition is cache-first and independent of any agent host:

```python
from arc_paper import ReferenceAcquisitionService, ReferenceIdentity

references = ReferenceAcquisitionService()
cached = references.lookup_cached_reference(doi="10.1000/example")
if cached is None:
    cached = references.acquire_reference("doi:10.1000/example")

# Files already obtained through an available, authorized channel can be
# admitted without coupling that channel to arc-paper.
local = references.admit_reference_file(
    "paper.epub",
    ReferenceIdentity(dois=("10.1000/example",)),
)
readable = references.cache.read_resource(local.readable_resource)
```

`ReferenceIdentity` retains every DOI while its `doi` property projects the
first DOI for compatibility. Exact cache-only lookup accepts a DOI, arXiv ID,
HTTP(S) URL, or normalized title. Titles are never fuzzy-matched. Arbitrary
media are stored through verified `CachedResourceRef` handles; EPUB originals
remain intact and receive a deterministic readable HTML derivative assembled
in publication spine order.

The built-in network paths cover official arXiv representations, DOI metadata
from INSPIRE and Crossref, Crossref full-text or landing links, and one ordinary
HTTP(S) resource. `ReferenceAcquisitionBackend` is the small extension contract
for caller-owned authorized backends. Package code does not inspect plugins,
agent hosts, or workflow files.

## Tests

The unit suite is offline; network integration is opt-in:

```bash
python -m pytest packages/arc-paper/tests
```
