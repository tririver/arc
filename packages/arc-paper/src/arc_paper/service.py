"""Typed facade for deterministic paper access and explicit workflows.

Deterministic methods own no run state. ``extract_keywords`` is a convenience
wrapper over the package's explicit :mod:`ac_jobs` workflow; LLM execution
remains owned by :mod:`ac_llm`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ac_jobs import JsonValue
from ac_llm import HostAuthority, LLMExecutionOptions, ModelSelection
from ac_document import AcDocumentService

from ._cache_admin import (
    CacheAdministrator,
    CacheEntry,
    CacheListResult,
    CacheRemoveResult,
    CacheUpdateRecord,
    CacheUpdateResult,
    PaperCacheIndex,
)
from ._cache_archive import (
    CacheExportResult,
    CacheImportResult,
    export_cache_archive,
    import_cache_archive,
)
from ._cache_root import resolve_cache_root
from .parse import PDFTextExtractor, ParseError, ParsedDocument
from .parse.parser import PDFTextExtractionError
from .cached_full_text_search import (
    CachedFullTextContextStatus,
    CachedFullTextSearchMode,
    CachedFullTextSearcher,
    search_document_occurrences,
)
from .cached_document import (
    CachedDocumentError,
    CachedDocumentRef,
    CachedSection,
    CachedSourceRange,
    CachedTableOfContents,
)
from .document_structure import (
    CachedDocumentStructureRef,
    cached_document_structure_ref_from_document,
)
from .document_access import (
    DocumentTarget,
    DocumentTargetKind,
    PaperSection,
    PaperTableOfContents,
    ResolvedDocumentInfo,
)
from .content_search import (
    DocumentTargetFailure,
    PaperEquationSearch,
    PaperFullTextSearch,
    ResolvedSearchDocument,
)
from .document_search import (
    EquationSearchResult,
    FullTextSearchResult,
    TableOfContentsEntry,
    search_equations as _search_equations,
    search_equation_terms as _search_equation_terms,
    search_full_text as _search_full_text,
    select_section as _select_section,
    table_of_contents as _table_of_contents,
)
from .ids import (
    arxiv_path_id,
    arxiv_version,
    arxiv_version_is_invalid,
    arxiv_versioned_path_id,
    doi_value,
    inspire_recid,
    normalize_paper_id,
)
from .ids import extract_paper_ids as _extract_paper_ids
from .ids import paper_ids_safe_dir_name as _paper_ids_safe_dir_name
from .html_dependencies import (
    AR5IV_HTML_DEPENDENCY_NAMESPACE,
    ARXIV_HTML_DEPENDENCY_NAMESPACE,
    HtmlSourceBundle,
    materialize_html_source_bundle,
)
from .providers import (
    Ar5ivProvider,
    ArxivHtmlProvider,
    ArxivPdfProvider,
    InspireProvider,
    describe_inspire_citer_request,
)
from .providers.base import ProviderError
from .reference_acquisition import (
    ReferenceAcquisitionError,
    ReferenceAcquisitionService,
)
from .reference_cache import (
    CachedReferenceMaterial,
    CachedResourceRef,
    ReferenceIdentity,
    ReferenceCacheError,
    ReferenceMaterialCache,
)
from .source_repository import SourceRepository, SourceRepositoryError
from .sources import (
    ParseOutcome,
    SourceArtifact,
    SourceBundle,
    SourceFormat,
    SourceOrigin,
    SourceOriginKind,
)

if TYPE_CHECKING:
    from .terms import KeywordResult


class PaperInputError(ValueError):
    """A stable invalid-request error raised before external effects."""

    code = "invalid_request"

    def __init__(self, message: str, *, code: str = "invalid_request"):
        super().__init__(message)
        self.code = code
        self.message = message


def default_cache_root() -> Path:
    return resolve_cache_root()


class ArcPaperService(AcDocumentService):
    """Injectable facade over package-owned deterministic and workflow services."""

    def __init__(
        self,
        *,
        cache_root: str | Path | None = None,
        repository: SourceRepository | None = None,
        inspire: InspireProvider | None = None,
        arxiv_html: ArxivHtmlProvider | None = None,
        ar5iv: Ar5ivProvider | None = None,
        arxiv_pdf: ArxivPdfProvider | None = None,
        pdf_text_extractor: PDFTextExtractor | None = None,
        keyword_task_service: Any | None = None,
    ):
        try:
            root = resolve_cache_root(cache_root, repository=repository)
        except ValueError as exc:
            raise PaperInputError(
                "cache_root must match the injected SourceRepository root"
            ) from exc
        super().__init__(
            cache_root=root,
            repository=repository,
            pdf_text_extractor=pdf_text_extractor,
            keyword_task_service=keyword_task_service,
        )
        self.cache_index = PaperCacheIndex(root)
        self.cache_administrator = CacheAdministrator(root)
        self.inspire = inspire or InspireProvider(cache_root=root)
        self.arxiv_html = arxiv_html or ArxivHtmlProvider(
            cache_root=root, source_repository=self.repository
        )
        self.ar5iv = ar5iv or Ar5ivProvider(
            cache_root=root, source_repository=self.repository
        )
        self.arxiv_pdf = arxiv_pdf or ArxivPdfProvider(
            cache_root=root, source_repository=self.repository
        )
        self._cached_full_text_searcher = CachedFullTextSearcher(root)
        self._reference_acquisition_service: ReferenceAcquisitionService | None = None

    def _input_error(
        self, message: str, *, code: str = "invalid_request"
    ) -> PaperInputError:
        return PaperInputError(message, code=code)

    def resolve_local_or_arxiv_source(
        self, source: str | Path, *, refresh: bool = False
    ) -> SourceArtifact:
        """Resolve an existing local file or a syntactically valid arXiv ID."""

        source_text = str(source)
        path = Path(source_text)
        if path.is_file():
            return self.repository.import_path(path)
        if arxiv_path_id(source_text):
            return self.fetch_arxiv_auto(source_text, refresh=refresh)
        raise SourceRepositoryError(
            "source_not_found",
            "source is neither an existing local file nor a valid arXiv ID: "
            f"{source_text}",
        )

    def fetch_arxiv_auto(
        self, paper_id: str, *, refresh: bool = False
    ) -> SourceArtifact:
        """Fetch official arXiv HTML, falling back to ar5iv only on a 404."""

        source, _, _ = self._fetch_arxiv_auto_materialized(
            paper_id, refresh=refresh
        )
        return source

    def fetch_arxiv_html_bundle(
        self, paper_id: str, *, refresh: bool = False
    ) -> HtmlSourceBundle:
        """Fetch the preferred HTML source and its authored image bundle."""

        if arxiv_version_is_invalid(paper_id):
            raise ProviderError(
                "arxiv_version_invalid",
                "explicit arXiv version must be positive and have no leading zeros",
            )
        try:
            bundle = self.arxiv_html.fetch_bundle(paper_id, refresh=refresh)
        except ProviderError as exc:
            if exc.code != "arxiv_html_not_found":
                raise
            if arxiv_version(paper_id):
                raise ProviderError(
                    "arxiv_html_version_not_found",
                    "exact versioned arXiv HTML is unavailable; fallback would lose version identity",
                ) from exc
            bundle = self.ar5iv.fetch_bundle(paper_id, refresh=refresh)
        self._record_arxiv_bundle_component(paper_id, bundle)
        return bundle

    def export_arxiv_html_bundle(
        self,
        paper_id: str,
        *,
        output_dir: str | Path,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Materialize remote HTML and safe authored targets for local parsing."""

        bundle = self.fetch_arxiv_html_bundle(paper_id, refresh=refresh)
        provider = self.arxiv_html if bundle.provider == "arxiv-html" else self.ar5iv
        resource_cache = getattr(provider, "resource_cache", None)
        if not isinstance(resource_cache, ReferenceMaterialCache):
            resource_cache = ReferenceMaterialCache(self.cache_root)
        return materialize_html_source_bundle(
            bundle,
            source_repository=self.repository,
            resource_cache=resource_cache,
            output_dir=output_dir,
        )

    def fetch_arxiv_pdf(
        self, paper_id: str, *, refresh: bool = False
    ) -> SourceArtifact:
        source = self.arxiv_pdf.fetch(paper_id, refresh=refresh)
        self.parser.parse_source(source)
        self._record_remote_component(
            paper_id,
            "arxiv-pdf",
            cache=getattr(self.arxiv_pdf, "cache", None),
            kind="source",
            namespace="arxiv-pdf",
            request_key=arxiv_path_id(paper_id),
        )
        return source

    def parse_arxiv_auto(
        self,
        paper_id: str,
        *,
        refresh: bool = False,
    ) -> ParseOutcome:
        primary = self._fetch_arxiv_auto_source(paper_id, refresh=refresh)
        outcome = self.parse_bundle(SourceBundle(primary=primary))
        self._record_arxiv_auto_component(paper_id, primary)
        return outcome

    def parse_arxiv_pdf(
        self,
        paper_id: str,
        *,
        refresh: bool = False,
    ) -> ParseOutcome:
        primary = self.arxiv_pdf.fetch(paper_id, refresh=refresh)
        outcome = self.parse_bundle(SourceBundle(primary=primary))
        self._record_remote_component(
            paper_id,
            "arxiv-pdf",
            cache=getattr(self.arxiv_pdf, "cache", None),
            kind="source",
            namespace="arxiv-pdf",
            request_key=arxiv_path_id(paper_id),
        )
        return outcome

    def get_metadata(self, paper_id: str, *, refresh: bool = False) -> dict[str, Any]:
        required = _require_paper_id(paper_id)
        value = self.inspire.get_metadata(required, refresh=refresh)
        self._record_remote_component(
            required,
            "inspire-record",
            cache=getattr(self.inspire, "cache", None),
            kind="json",
            namespace="inspire-record",
            request_key=normalize_paper_id(required),
        )
        return value

    def get_title(self, paper_id: str, *, refresh: bool = False) -> str:
        return str(self.get_metadata(paper_id, refresh=refresh).get("title") or "")

    def get_abstract(self, paper_id: str, *, refresh: bool = False) -> str:
        return str(self.get_metadata(paper_id, refresh=refresh).get("abstract") or "")

    def get_authors(self, paper_id: str, *, refresh: bool = False) -> list[str]:
        value = self.get_metadata(paper_id, refresh=refresh).get("authors") or []
        return [str(item) for item in value]

    def get_references(
        self,
        paper_id: str,
        *,
        refresh: bool = False,
        enrich: bool = False,
    ) -> list[dict[str, Any]]:
        required = _require_paper_id(paper_id)
        value = self.inspire.get_references(
            required, refresh=refresh, enrich=enrich
        )
        self._record_remote_component(
            required,
            "inspire-record",
            cache=getattr(self.inspire, "cache", None),
            kind="json",
            namespace="inspire-record",
            request_key=normalize_paper_id(required),
        )
        return value

    def get_citers(
        self,
        paper_id: str,
        *,
        refresh: bool = False,
        limit: int = 1000,
        sort: str = "mostrecent",
    ) -> list[dict[str, Any]]:
        required = _require_paper_id(paper_id)
        value = self.inspire.get_citers(
            required,
            refresh=refresh,
            limit=limit,
            sort=sort,
        )
        try:
            metadata = self.inspire.get_metadata(required, refresh=False)
            recid = str(metadata.get("inspire_recid") or "")
            if recid:
                request = describe_inspire_citer_request(
                    recid,
                    sort=sort,
                    limit=limit,
                )
                self._record_remote_component(
                    required,
                    request.admin_component,
                    cache=getattr(self.inspire, "cache", None),
                    kind="json",
                    namespace="inspire-citers",
                    request_key=request.request_key,
                )
        except Exception:
            # The citer result remains valid even if optional admin indexing fails.
            pass
        return value

    def get_citer_count(self, paper_id: str, *, refresh: bool = False) -> int:
        return self.inspire.get_citer_count(
            _require_paper_id(paper_id), refresh=refresh
        )

    def search_citers(
        self,
        paper_id: str,
        terms: Sequence[str],
        *,
        refresh: bool = False,
        scan_limit: int = 1000,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Shortlist direct citers by normalized literal title/abstract terms."""

        required = _require_paper_id(paper_id)
        normalized_terms = _require_citer_search_terms(terms)
        scan_limit = _require_limit(
            scan_limit, name="scan_limit", maximum=1000
        )
        limit = _require_limit(limit, name="limit", maximum=50)
        total_citer_count = self.get_citer_count(required, refresh=refresh)

        if total_citer_count == 0:
            scanned: list[dict[str, Any]] = []
            scan_strategy = "all-mostrecent"
        elif total_citer_count <= scan_limit:
            scanned = [
                dict(record)
                for record in self.get_citers(
                    required,
                    refresh=refresh,
                    limit=scan_limit,
                    sort="mostrecent",
                )
            ]
            scan_strategy = "all-mostrecent"
        else:
            recent_limit = (scan_limit + 1) // 2
            cited_limit = scan_limit // 2
            recent = self.get_citers(
                required,
                refresh=refresh,
                limit=recent_limit,
                sort="mostrecent",
            )
            cited = (
                self.get_citers(
                    required,
                    refresh=refresh,
                    limit=cited_limit,
                    sort="mostcited",
                )
                if cited_limit
                else []
            )
            scanned = _dedupe_citer_records((*recent, *cited))
            scan_strategy = "split-mostrecent-mostcited"

        matches = _match_citer_records(scanned, normalized_terms)
        returned_matches = matches[:limit]
        scanned_count = len(scanned)
        scan_complete = (
            total_citer_count <= scan_limit
            and scanned_count >= total_citer_count
        )
        return {
            "paper_id": normalize_paper_id(required),
            "total_citer_count": total_citer_count,
            "scanned_count": scanned_count,
            "scan_complete": scan_complete,
            "scan_strategy": scan_strategy,
            "terms": [term for term, _ in normalized_terms],
            "matched_count": len(matches),
            "returned_count": len(returned_matches),
            "matches_truncated": len(matches) > len(returned_matches),
            "matches": returned_matches,
            "control_sample": _citer_control_sample(scanned),
        }

    def search_metadata(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return self.inspire.search_metadata(query, limit=limit)

    def list_cache(
        self,
        *,
        paper_ids: Sequence[str] = (),
        entry_ids: Sequence[str] = (),
        since_seconds: int | None = None,
    ) -> CacheListResult:
        return self.cache_administrator.list(
            paper_ids=paper_ids,
            entry_ids=entry_ids,
            since_seconds=since_seconds,
        )

    def export_cache(
        self,
        output: str | Path,
        *,
        entry_ids: Sequence[str] = (),
        all_entries: bool = False,
    ) -> CacheExportResult:
        return export_cache_archive(
            output,
            cache_root=self.cache_root,
            entry_ids=entry_ids,
            all_entries=all_entries,
        )

    def import_cache(
        self,
        archive: str | Path,
        *,
        replace_conflicts: bool = False,
    ) -> CacheImportResult:
        return import_cache_archive(
            archive,
            cache_root=self.cache_root,
            replace_conflicts=replace_conflicts,
        )

    def remove_cache(
        self,
        *,
        paper_ids: Sequence[str] = (),
        entry_ids: Sequence[str] = (),
        dry_run: bool = True,
    ) -> CacheRemoveResult:
        if not paper_ids and not entry_ids:
            raise PaperInputError(
                "cache remove requires at least one exact paper or entry id"
            )
        selected = self.cache_administrator.list(
            paper_ids=paper_ids,
            entry_ids=entry_ids,
        ).entries
        if dry_run:
            return CacheRemoveResult(True, selected, ())

        removed: list[str] = []
        warnings: list[str] = []
        for entry in selected:
            changed = False
            for component in entry.components:
                for storage_entry_id in component.storage_entry_ids:
                    if storage_entry_id.startswith("remote:"):
                        changed = (
                            self.cache_administrator.remote.remove_admin_entry(
                                storage_entry_id
                            )
                            or changed
                        )
                    else:
                        document_result = (
                            self.cache_administrator.document.remove(
                                entry_ids=(storage_entry_id,), dry_run=False
                            )
                        )
                        changed = bool(document_result.removed_entry_ids) or changed
            changed = self.cache_index.remove(entry.entry_id) or changed
            if changed:
                removed.append(entry.entry_id)
            else:
                warnings.append(f"cache_entry_already_absent:{entry.entry_id}")
        return CacheRemoveResult(
            False,
            selected,
            tuple(removed),
            warnings=tuple(warnings),
        )

    def update_cache(
        self,
        *,
        paper_ids: Sequence[str] = (),
        entry_ids: Sequence[str] = (),
    ) -> CacheUpdateResult:
        if not paper_ids and not entry_ids:
            raise PaperInputError(
                "cache update requires at least one exact paper or entry id"
            )
        selected = self.cache_administrator.list(
            paper_ids=paper_ids,
            entry_ids=entry_ids,
        ).entries
        records: list[CacheUpdateRecord] = []
        warnings: list[str] = []
        for entry in selected:
            if entry.kind != "paper" or entry.paper_id is None:
                records.append(
                    CacheUpdateRecord(
                        entry.entry_id,
                        "all",
                        "skipped",
                        "only paper cache entries can be refreshed",
                    )
                )
                continue
            paper_id = entry.paper_id
            metadata: dict[str, Any] | None = None
            try:
                metadata = self.get_metadata(paper_id, refresh=True)
                records.append(
                    CacheUpdateRecord(entry.entry_id, "inspire-record", "updated")
                )
            except (
                CachedDocumentError,
                PaperInputError,
                ParseError,
                PDFTextExtractionError,
                ProviderError,
                ReferenceAcquisitionError,
                ReferenceCacheError,
                SourceRepositoryError,
                ValueError,
            ) as exc:
                records.append(
                    CacheUpdateRecord(
                        entry.entry_id,
                        "inspire-record",
                        "failed",
                        _cache_error_message(exc),
                    )
                )
            for sort in ("mostrecent", "mostcited"):
                component = f"inspire-citers:{sort}:1000"
                try:
                    self.get_citers(
                        paper_id,
                        refresh=True,
                        limit=1000,
                        sort=sort,
                    )
                    records.append(
                        CacheUpdateRecord(entry.entry_id, component, "updated")
                    )
                except Exception as exc:
                    records.append(
                        CacheUpdateRecord(
                            entry.entry_id,
                            component,
                            "failed",
                            _cache_error_message(exc),
                        )
                    )
            bundle_papers = self._cached_html_bundle_paper_ids(entry)
            arxiv_id = (
                str((metadata or {}).get("arxiv_id") or "")
                or arxiv_path_id(paper_id)
            )
            arxiv_paper = f"arXiv:{arxiv_id}" if arxiv_id else paper_id
            for component, action in (
                ("arxiv-auto", self.parse_arxiv_auto),
                ("arxiv-pdf", self.parse_arxiv_pdf),
            ):
                if component == "arxiv-auto" and bundle_papers:
                    for bundle_paper in bundle_papers:
                        try:
                            bundle = self.fetch_arxiv_html_bundle(
                                bundle_paper, refresh=True
                            )
                            self.parse_bundle(SourceBundle(primary=bundle.primary))
                            records.append(
                                CacheUpdateRecord(
                                    entry.entry_id, bundle.provider, "updated"
                                )
                            )
                        except Exception as exc:
                            records.append(
                                CacheUpdateRecord(
                                    entry.entry_id,
                                    "arxiv-auto",
                                    "failed",
                                    _cache_error_message(exc),
                                )
                            )
                    continue
                try:
                    if component == "arxiv-auto":
                        result = action(arxiv_paper, refresh=True)
                        component = _auto_html_component(result.report.primary)
                    else:
                        action(arxiv_paper, refresh=True)
                    records.append(
                        CacheUpdateRecord(entry.entry_id, component, "updated")
                    )
                except Exception as exc:
                    records.append(
                        CacheUpdateRecord(
                            entry.entry_id,
                            component,
                            "failed",
                            _cache_error_message(exc),
                        )
                    )
        if not selected:
            warnings.append("cache_entry_not_found")
        warnings.extend(
            f"cache_update_failed:{item.entry_id}:{item.component}"
            for item in records
            if item.status == "failed"
        )
        return CacheUpdateResult(tuple(records), tuple(warnings))

    def search_full_text_targets(
        self,
        terms: Sequence[str],
        *,
        targets: Sequence[DocumentTarget] = (),
        source_format: SourceFormat | str | None = None,
        refresh: bool = False,
        limit: int = 100,
        context_lines: int = 0,
        case_sensitive: bool = False,
    ) -> PaperFullTextSearch:
        normalized_terms = _normalize_literal_terms(terms)
        limit = _require_limit(limit, name="limit", maximum=500)
        if not targets:
            if source_format is not None or refresh:
                raise PaperInputError(
                    "source_format and refresh require reference targets"
                )
            result = self._cached_full_text_searcher.search(
                normalized_terms,
                limit=limit,
                context_lines=context_lines,
                case_sensitive=case_sensitive,
            )
            documents = tuple(
                ResolvedSearchDocument(
                    source=ResolvedDocumentInfo(
                        document=item.document,
                        identity=(
                            ReferenceIdentity(
                                arxiv_id=arxiv_path_id(item.arxiv_ids[0])
                            )
                            if item.arxiv_ids
                            else None
                        ),
                    )
                )
                for item in result.documents
            )
            return PaperFullTextSearch(
                scope="corpus",
                mode=result.mode,
                terms=result.terms,
                limit=result.limit,
                context_lines=result.context_lines,
                case_sensitive=result.case_sensitive,
                total_occurrences=result.total_occurrences,
                matched_document_count=result.matched_document_count,
                documents=documents,
                failures=(),
                occurrences=result.occurrences,
                top_paper_titles=result.top_paper_titles,
                context_status=result.context_status,
                message=result.message,
                warnings=result.warnings,
            )

        resolved, failures = self._resolve_search_targets(
            targets, source_format=source_format, refresh=refresh
        )
        occurrences = tuple(
            occurrence
            for document, _ in resolved
            for occurrence in search_document_occurrences(
                document,
                normalized_terms,
                context_lines=context_lines,
                case_sensitive=case_sensitive,
            )
        )
        documents = tuple(item for _, item in resolved)
        matched_document_count = len(
            {item.document_digest for item in occurrences}
        )
        returned = occurrences[:limit]
        warnings = tuple(
            dict.fromkeys(
                warning
                for item in documents
                for warning in item.source.warnings
            )
        )
        return PaperFullTextSearch(
            scope="targets",
            mode=CachedFullTextSearchMode.OCCURRENCES,
            terms=normalized_terms,
            limit=limit,
            context_lines=context_lines,
            case_sensitive=case_sensitive,
            total_occurrences=len(occurrences),
            matched_document_count=matched_document_count,
            documents=documents,
            failures=failures,
            occurrences=returned,
            top_paper_titles=(),
            context_status=(
                CachedFullTextContextStatus.INCLUDED
                if context_lines
                else CachedFullTextContextStatus.NOT_REQUESTED
            ),
            message=(
                f"Found {len(occurrences)} full-text occurrence"
                f"{'' if len(occurrences) == 1 else 's'} in "
                f"{matched_document_count} document"
                f"{'' if matched_document_count == 1 else 's'}."
            ),
            warnings=warnings,
        )

    def search_equation_targets(
        self,
        targets: Sequence[DocumentTarget],
        terms: Sequence[str],
        *,
        source_format: SourceFormat | str | None = None,
        refresh: bool = False,
        limit: int = 20,
        context_lines: int = 8,
        case_sensitive: bool = False,
    ) -> PaperEquationSearch:
        if not targets:
            raise PaperInputError("search-equations requires at least one target")
        resolved, failures = self._resolve_search_targets(
            targets, source_format=source_format, refresh=refresh
        )
        documents = tuple(document for document, _ in resolved)
        result = _search_equation_terms(
            documents,
            terms,
            limit=limit,
            context_lines=context_lines,
            case_sensitive=case_sensitive,
        )
        resolved_documents = tuple(item for _, item in resolved)
        warnings = [
            warning
            for item in resolved_documents
            for warning in item.source.warnings
        ]
        if any(document.source.source_format is SourceFormat.HTML for document in documents):
            warnings.append(
                "HTML equation labels are converter-derived and may differ from printed numbering"
            )
        if any(document.source.source_format is SourceFormat.PDF for document in documents):
            warnings.append(
                "PDF equations and excerpts come from approximate layout-preserving text extraction"
            )
        return PaperEquationSearch(
            terms=result.terms,
            limit=result.limit,
            context_lines=result.context_lines,
            case_sensitive=result.case_sensitive,
            searched_document_count=result.searched_documents,
            documents=resolved_documents,
            failures=failures,
            matches=result.matches,
            truncated=result.truncated,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _resolve_search_targets(
        self,
        targets: Sequence[DocumentTarget],
        *,
        source_format: SourceFormat | str | None,
        refresh: bool,
    ) -> tuple[
        tuple[tuple[ParsedDocument, ResolvedSearchDocument], ...],
        tuple[DocumentTargetFailure, ...],
    ]:
        resolved: list[tuple[ParsedDocument, ResolvedSearchDocument]] = []
        positions: dict[str, int] = {}
        failures: list[DocumentTargetFailure] = []
        for index, target in enumerate(targets):
            try:
                is_reference = target.kind is DocumentTargetKind.REFERENCE
                document, source = self._resolve_document_target(
                    target,
                    source_format=source_format if is_reference else None,
                    refresh=refresh if is_reference else False,
                )
            except Exception as exc:
                failures.append(
                    DocumentTargetFailure(
                        target_index=index,
                        target=target,
                        code=str(getattr(exc, "code", type(exc).__name__)),
                        message=str(getattr(exc, "message", str(exc))),
                    )
                )
                continue
            previous = positions.get(document.document_digest)
            if previous is not None:
                old_document, old_info = resolved[previous]
                resolved[previous] = (
                    old_document,
                    ResolvedSearchDocument(
                        source=old_info.source,
                        target_indices=(*old_info.target_indices, index),
                    ),
                )
                continue
            positions[document.document_digest] = len(resolved)
            resolved.append(
                (
                    document,
                    ResolvedSearchDocument(source=source, target_indices=(index,)),
                )
            )
        if not resolved:
            detail = "; ".join(
                f"target {item.target_index}: {item.code}: {item.message}"
                for item in failures
            )
            raise PaperInputError(
                f"no document target resolved{': ' + detail if detail else ''}",
                code="no_document_target_resolved",
            )
        return tuple(resolved), tuple(failures)

    def lookup_reference(
        self,
        *,
        doi: str | None = None,
        arxiv_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
    ) -> CachedReferenceMaterial | None:
        return ReferenceMaterialCache(self.cache_root).lookup(
            doi=doi, arxiv_id=arxiv_id, url=url, title=title
        )

    def acquire_reference(
        self,
        *,
        doi: str | None = None,
        arxiv_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
        refresh: bool = False,
        source_format: SourceFormat | str | None = None,
    ) -> CachedReferenceMaterial:
        identity = _reference_identity(
            doi=doi, arxiv_id=arxiv_id, url=url, title=title
        )
        return self._reference_acquisition().acquire_reference(
            identity, refresh=refresh, source_format=source_format
        )

    def get_table_of_contents(
        self,
        target: DocumentTarget,
        *,
        structure: CachedDocumentStructureRef | None = None,
        source_format: SourceFormat | str | None = None,
        refresh: bool = False,
    ) -> PaperTableOfContents:
        if structure is not None:
            if target.kind is not DocumentTargetKind.DOCUMENT or target.document is None:
                raise PaperInputError("structure applies only to exact document targets")
            cached = self.get_cached_table_of_contents(
                target.document, structure=structure
            )
            source = ResolvedDocumentInfo(
                document=target.document, identity=None, warnings=cached.warnings
            )
            return PaperTableOfContents(
                source=source, entries=cached.entries, warnings=cached.warnings
            )
        parsed, source = self._resolve_document_target(
            target, source_format=source_format, refresh=refresh
        )
        return PaperTableOfContents(
            source=source,
            entries=_table_of_contents(parsed),
            warnings=source.warnings,
        )

    def get_section(
        self,
        target: DocumentTarget,
        selector: str | int,
        *,
        structure: CachedDocumentStructureRef | None = None,
        source_format: SourceFormat | str | None = None,
        refresh: bool = False,
    ) -> PaperSection:
        if structure is not None:
            if target.kind is not DocumentTargetKind.DOCUMENT or target.document is None:
                raise PaperInputError("structure applies only to exact document targets")
            cached = self.get_cached_section(
                target.document, selector, structure=structure
            )
            source = ResolvedDocumentInfo(
                document=target.document, identity=None, warnings=cached.warnings
            )
            return PaperSection(
                source=source,
                section_id=cached.section_id,
                title=cached.title,
                text=cached.text,
                level=cached.level,
                ordinal=cached.ordinal,
                page_start=cached.page_start,
                page_end=cached.page_end,
                warnings=cached.warnings,
            )
        parsed, source = self._resolve_document_target(
            target, source_format=source_format, refresh=refresh
        )
        section = _select_section(parsed, selector)
        return PaperSection(
            source=source,
            section_id=section.section_id,
            title=section.title,
            text=section.text,
            level=section.level,
            ordinal=section.ordinal,
            page_start=section.page_start,
            page_end=section.page_end,
            warnings=source.warnings,
        )

    def _resolve_document_target(
        self,
        target: DocumentTarget,
        *,
        source_format: SourceFormat | str | None = None,
        refresh: bool = False,
    ) -> tuple[ParsedDocument, ResolvedDocumentInfo]:
        if not isinstance(target, DocumentTarget):
            raise PaperInputError("target must be a DocumentTarget")
        requested_format = (
            SourceFormat(source_format) if source_format is not None else None
        )
        if target.kind is DocumentTargetKind.DOCUMENT:
            if requested_format is not None or refresh:
                raise PaperInputError(
                    "source_format and refresh apply only to reference targets"
                )
            assert target.document is not None
            parsed, warnings = self._resolve_cached_document(target.document)
            return parsed, ResolvedDocumentInfo(
                document=target.document,
                identity=None,
                warnings=warnings,
            )

        identity = _reference_identity_for_query(target.reference)
        material = None if refresh else _lookup_reference_identity(
            ReferenceMaterialCache(self.cache_root), identity
        )
        if material is None or (
            requested_format is not None
            and _select_reference_resource(material, requested_format) is None
        ):
            material = self._reference_acquisition().acquire_reference(
                identity,
                refresh=refresh,
                source_format=requested_format,
            )
        resource = _select_reference_resource(material, requested_format)
        if resource is None:
            raise PaperInputError(
                "reference contains no parseable requested representation",
                code="reference_format_unavailable",
            )
        payload = ReferenceMaterialCache(self.cache_root).read_resource(resource)
        resolved_format = _source_format_for_media_type(resource.media_type)
        source = self.repository.store_bytes(
            payload,
            source_format=resolved_format,
            media_type=resource.media_type,
            origin=SourceOrigin(
                kind=SourceOriginKind.REPOSITORY,
                locator=resource.source_locator,
                metadata=(
                    {
                        "arxiv_id": material.identity.arxiv_id,
                        "document_id": f"arXiv:{material.identity.arxiv_id}",
                    }
                    if material.identity.arxiv_id
                    else {}
                ),
            ),
        )
        parsed, parse_warnings = self.parser.materialize_source(source)
        document_ref = self._cached_document_ref(source, parsed)
        return parsed, ResolvedDocumentInfo(
            document=document_ref,
            identity=material.identity,
            requested_reference=target.reference,
            warnings=parse_warnings,
        )

    def admit_reference(
        self,
        path: str | Path,
        *,
        doi: str | None = None,
        arxiv_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
        media_type: str | None = None,
    ) -> CachedReferenceMaterial:
        identity = _reference_identity(
            doi=doi, arxiv_id=arxiv_id, url=url, title=title
        )
        return self._reference_acquisition().admit_reference_file(
            path, identity, media_type=media_type
        )

    def materialize_reference(
        self,
        resource: CachedResourceRef,
        output: str | Path,
    ) -> dict[str, Any]:
        """Verify and atomically materialize one cached resource."""

        from ac_jobs import atomic_write_bytes

        payload = ReferenceMaterialCache(self.cache_root).read_resource(resource)
        output_path = Path(output)
        atomic_write_bytes(output_path, payload)
        return {
            "resource": resource,
            "output": str(output_path),
            "bytes_written": len(payload),
        }

    def _reference_acquisition(self) -> ReferenceAcquisitionService:
        if self._reference_acquisition_service is None:
            self._reference_acquisition_service = ReferenceAcquisitionService(
                cache_root=self.cache_root,
                inspire=self.inspire,
                arxiv_html=self.arxiv_html,
                ar5iv=self.ar5iv,
                arxiv_pdf=self.arxiv_pdf,
            )
        return self._reference_acquisition_service

    def _fetch_arxiv_auto_materialized(
        self, paper_id: str, *, refresh: bool
    ) -> tuple[
        SourceArtifact,
        ParsedDocument,
        tuple[str, ...],
    ]:
        source = self._fetch_arxiv_auto_source(paper_id, refresh=refresh)
        document, warnings = self.parser.materialize_source(source)
        self._record_arxiv_auto_component(paper_id, source)
        return source, document, warnings

    def _fetch_arxiv_auto_source(
        self, paper_id: str, *, refresh: bool
    ) -> SourceArtifact:
        """Choose the official source, falling back only for an HTML 404."""

        try:
            return self.arxiv_html.fetch(paper_id, refresh=refresh)
        except ProviderError as exc:
            if exc.code != "arxiv_html_not_found":
                raise
        return self.ar5iv.fetch(paper_id, refresh=refresh)

    def _record_arxiv_auto_component(
        self, paper_id: str, source: SourceArtifact
    ) -> None:
        component = _auto_html_component(source)
        provider = self.arxiv_html if component == "arxiv-html" else self.ar5iv
        self._record_remote_component(
            paper_id,
            component,
            cache=getattr(provider, "cache", None),
            kind="source",
            namespace=component,
            request_key=arxiv_path_id(paper_id),
        )

    def _record_arxiv_bundle_component(
        self, paper_id: str, bundle: HtmlSourceBundle
    ) -> None:
        component = bundle.provider
        provider = self.arxiv_html if component == "arxiv-html" else self.ar5iv
        dependency_namespace = (
            ARXIV_HTML_DEPENDENCY_NAMESPACE
            if component == "arxiv-html"
            else AR5IV_HTML_DEPENDENCY_NAMESPACE
        )
        request_key = (
            arxiv_versioned_path_id(paper_id)
            if component == "arxiv-html"
            else arxiv_path_id(paper_id)
        )
        for kind, namespace in (
            ("source", component),
            ("json", dependency_namespace),
        ):
            self._record_remote_component(
                paper_id,
                component,
                cache=getattr(provider, "cache", None),
                kind=kind,
                namespace=namespace,
                request_key=request_key,
            )

    def _cached_html_bundle_paper_ids(
        self, entry: CacheEntry
    ) -> tuple[str, ...]:
        admin_by_id = {
            item.entry_id: item
            for item in self.cache_administrator.remote.admin_entries()
        }
        request_keys = {
            admin.request_key
            for component in entry.components
            for storage_id in component.storage_entry_ids
            if (admin := admin_by_id.get(storage_id)) is not None
            and admin.kind == "json"
            and admin.namespace
            in {
                ARXIV_HTML_DEPENDENCY_NAMESPACE,
                AR5IV_HTML_DEPENDENCY_NAMESPACE,
            }
        }
        return tuple(
            f"arXiv:{request_key}"
            for request_key in sorted(request_keys)
            if arxiv_path_id(request_key)
        )

    def _record_remote_component(
        self,
        paper_id: str,
        component: str,
        *,
        cache: Any,
        kind: str,
        namespace: str,
        request_key: str,
    ) -> None:
        if not request_key or not hasattr(cache, "admin_entry"):
            return
        try:
            entry = cache.admin_entry(kind, namespace, request_key)
            if entry is None:
                return
            self.cache_index.record_paper_component(
                paper_id,
                component,
                cached_at=entry.cached_at,
                storage_entry_ids=(entry.entry_id,),
            )
        except (OSError, TypeError, ValueError):
            return

def extract_paper_ids(text: str) -> list[str]:
    return _extract_paper_ids(str(text))


def paper_ids_safe_dir_name(ids: str | Iterable[str]) -> str:
    values = [ids] if isinstance(ids, str) else list(ids)
    normalized = [str(item) for item in values if str(item).strip()]
    if not normalized:
        raise PaperInputError("at least one paper id is required")
    return _paper_ids_safe_dir_name(normalized)


def _require_paper_id(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise PaperInputError("paper_id is required")
    return normalized


def _require_limit(value: int, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PaperInputError(f"{name} must be an integer between 1 and {maximum}")
    if not 1 <= value <= maximum:
        raise PaperInputError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _require_citer_search_terms(
    terms: Sequence[str],
) -> list[tuple[str, str]]:
    if isinstance(terms, (str, bytes)):
        raise PaperInputError("terms must be a non-empty sequence of search phrases")
    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for term in terms:
        if not isinstance(term, str):
            raise PaperInputError("citer search terms must be strings")
        surface = term.strip()
        normalized = _normalize_citer_search_text(surface)
        if not normalized:
            raise PaperInputError("citer search terms must not be empty")
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append((surface, normalized))
    if not resolved:
        raise PaperInputError("at least one citer search term is required")
    return resolved


def _normalize_citer_search_text(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    separated = "".join(
        " " if unicodedata.category(character).startswith(("P", "Z")) else character
        for character in folded
    )
    return " ".join(separated.split())


def _citer_identity(record: Mapping[str, Any]) -> str:
    for key in ("paper_id", "inspire_recid", "arxiv_id", "doi"):
        if value := str(record.get(key) or "").strip():
            return f"{key}:{value.casefold()}"
    return "record:" + json.dumps(
        dict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _dedupe_citer_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        identity = _citer_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(dict(record))
    return deduplicated


def _match_citer_records(
    records: Sequence[Mapping[str, Any]],
    terms: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for record in records:
        fields = {
            "title": _normalize_citer_search_text(str(record.get("title") or "")),
            "abstract": _normalize_citer_search_text(
                str(record.get("abstract") or "")
            ),
        }
        matched_terms: list[str] = []
        matched_fields: list[str] = []
        for surface, normalized in terms:
            term_fields = [
                field
                for field in ("title", "abstract")
                if normalized in fields[field]
            ]
            if not term_fields:
                continue
            matched_terms.append(surface)
            for field in term_fields:
                if field not in matched_fields:
                    matched_fields.append(field)
        if matched_terms:
            match = dict(record)
            match["matched_terms"] = matched_terms
            match["matched_fields"] = matched_fields
            matches.append(match)

    matches.sort(key=lambda item: str(item.get("paper_id") or ""))
    matches.sort(key=lambda item: str(item.get("published") or ""), reverse=True)
    matches.sort(key=_citer_citation_count, reverse=True)
    matches.sort(key=lambda item: len(item["matched_terms"]), reverse=True)
    matches.sort(key=lambda item: "title" in item["matched_fields"], reverse=True)
    return matches


def _citer_citation_count(record: Mapping[str, Any]) -> int:
    try:
        return int(record.get("citation_count") or 0)
    except (TypeError, ValueError):
        return 0


def _citer_control_sample(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    newest = sorted(records, key=lambda item: str(item.get("paper_id") or ""))
    newest.sort(key=_citer_citation_count, reverse=True)
    newest.sort(key=lambda item: str(item.get("published") or ""), reverse=True)

    most_cited = sorted(records, key=lambda item: str(item.get("paper_id") or ""))
    most_cited.sort(
        key=lambda item: str(item.get("published") or ""), reverse=True
    )
    most_cited.sort(key=_citer_citation_count, reverse=True)

    controls: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for reason, selected in (
        ("newest", newest[:5]),
        ("most-cited", most_cited[:5]),
    ):
        for record in selected:
            identity = _citer_identity(record)
            if identity in positions:
                controls[positions[identity]]["control_reasons"].append(reason)
                continue
            control = dict(record)
            control["control_reasons"] = [reason]
            positions[identity] = len(controls)
            controls.append(control)
    return controls


def _auto_html_component(source: SourceArtifact) -> str:
    provider = source.origin.provider
    if provider in {"arxiv-html", "ar5iv"}:
        return "arxiv-html" if provider == "arxiv-html" else "ar5iv-html"
    raise ProviderError(
        "arxiv_auto_provider_invalid",
        "automatic arXiv HTML source has an unsupported provider",
    )


def _cache_error_message(exc: Exception) -> str:
    code = str(getattr(exc, "code", type(exc).__name__))
    message = str(getattr(exc, "message", str(exc)))
    return f"{code}: {message}"[:500]


def _reference_identity(
    *,
    doi: str | None,
    arxiv_id: str | None,
    url: str | None,
    title: str | None,
) -> ReferenceIdentity:
    supplied = [value is not None for value in (doi, arxiv_id, url, title)]
    if sum(supplied) != 1:
        raise PaperInputError(
            "exactly one DOI, arXiv ID, URL, or title is required"
        )
    try:
        return ReferenceIdentity(
            arxiv_id=arxiv_id or "",
            dois=(doi,) if doi is not None else (),
            urls=(url,) if url is not None else (),
            title=title or "",
        )
    except ValueError as exc:
        raise PaperInputError(str(exc)) from exc


def _reference_identity_for_query(value: str) -> ReferenceIdentity:
    text = str(value or "").strip()
    normalized = normalize_paper_id(text)
    try:
        if arxiv_id := arxiv_path_id(normalized):
            return ReferenceIdentity(arxiv_id=arxiv_id)
        if doi := doi_value(normalized):
            return ReferenceIdentity(dois=(doi,))
        if recid := inspire_recid(normalized):
            return ReferenceIdentity(inspire_recid=recid)
        if text.casefold().startswith(("http://", "https://")):
            return ReferenceIdentity(urls=(text,))
        if text:
            return ReferenceIdentity(title=text)
    except ValueError as exc:
        raise PaperInputError(str(exc)) from exc
    raise PaperInputError("reference target is empty")


def _normalize_literal_terms(terms: Sequence[str]) -> tuple[str, ...]:
    if isinstance(terms, (str, bytes)):
        raise PaperInputError("terms must be a sequence of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in terms:
        if not isinstance(value, str) or not value.strip():
            raise PaperInputError("each term must be a non-empty string")
        term = " ".join(value.split())
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            normalized.append(term)
    if not normalized:
        raise PaperInputError("at least one term is required")
    return tuple(normalized)


def _lookup_reference_identity(
    cache: ReferenceMaterialCache, identity: ReferenceIdentity
) -> CachedReferenceMaterial | None:
    if identity.arxiv_id:
        return cache.lookup(arxiv_id=identity.arxiv_id)
    if identity.inspire_recid:
        return cache.lookup(inspire_recid=identity.inspire_recid)
    if identity.dois:
        return cache.lookup(doi=identity.dois[0])
    if identity.urls:
        return cache.lookup(url=identity.urls[0])
    return cache.lookup(title=identity.title)


def _source_format_for_media_type(media_type: str) -> SourceFormat:
    normalized = str(media_type).casefold()
    formats = {
        "text/html": SourceFormat.HTML,
        "application/xhtml+xml": SourceFormat.HTML,
        "text/markdown": SourceFormat.MARKDOWN,
        "text/x-tex": SourceFormat.TEX,
        "application/x-tex": SourceFormat.TEX,
        "application/pdf": SourceFormat.PDF,
    }
    try:
        return formats[normalized]
    except KeyError as exc:
        raise PaperInputError(
            f"reference media type is not parseable: {media_type}",
            code="reference_format_unavailable",
        ) from exc


def _select_reference_resource(
    material: CachedReferenceMaterial,
    source_format: SourceFormat | None,
) -> CachedResourceRef | None:
    for resource in material.resources:
        try:
            resolved = _source_format_for_media_type(resource.media_type)
        except PaperInputError:
            continue
        if source_format is None or resolved is source_format:
            return resource
    return None


def list_cache(
    *,
    paper_ids: Sequence[str] = (),
    entry_ids: Sequence[str] = (),
    since_seconds: int | None = None,
    cache_root: str | Path | None = None,
) -> CacheListResult:
    return ArcPaperService(cache_root=cache_root).list_cache(
        paper_ids=paper_ids,
        entry_ids=entry_ids,
        since_seconds=since_seconds,
    )


def remove_cache(
    *,
    paper_ids: Sequence[str] = (),
    entry_ids: Sequence[str] = (),
    dry_run: bool = True,
    cache_root: str | Path | None = None,
) -> CacheRemoveResult:
    return ArcPaperService(cache_root=cache_root).remove_cache(
        paper_ids=paper_ids,
        entry_ids=entry_ids,
        dry_run=dry_run,
    )


def update_cache(
    *,
    paper_ids: Sequence[str] = (),
    entry_ids: Sequence[str] = (),
    cache_root: str | Path | None = None,
) -> CacheUpdateResult:
    return ArcPaperService(cache_root=cache_root).update_cache(
        paper_ids=paper_ids,
        entry_ids=entry_ids,
    )


def export_cache(
    output: str | Path,
    *,
    entry_ids: Sequence[str] = (),
    all_entries: bool = False,
    cache_root: str | Path | None = None,
) -> CacheExportResult:
    return ArcPaperService(cache_root=cache_root).export_cache(
        output, entry_ids=entry_ids, all_entries=all_entries
    )


def import_cache(
    archive: str | Path,
    *,
    replace_conflicts: bool = False,
    cache_root: str | Path | None = None,
) -> CacheImportResult:
    return ArcPaperService(cache_root=cache_root).import_cache(
        archive, replace_conflicts=replace_conflicts
    )


def get_metadata(paper_id: str, *, refresh: bool = False) -> dict[str, Any]:
    return ArcPaperService().get_metadata(paper_id, refresh=refresh)


def get_title(paper_id: str, *, refresh: bool = False) -> str:
    return ArcPaperService().get_title(paper_id, refresh=refresh)


def get_abstract(paper_id: str, *, refresh: bool = False) -> str:
    return ArcPaperService().get_abstract(paper_id, refresh=refresh)


def get_authors(paper_id: str, *, refresh: bool = False) -> list[str]:
    return ArcPaperService().get_authors(paper_id, refresh=refresh)


def get_references(
    paper_id: str, *, refresh: bool = False, enrich: bool = False
) -> list[dict[str, Any]]:
    return ArcPaperService().get_references(
        paper_id, refresh=refresh, enrich=enrich
    )


def get_citers(
    paper_id: str,
    *,
    refresh: bool = False,
    limit: int = 1000,
    sort: str = "mostrecent",
) -> list[dict[str, Any]]:
    return ArcPaperService().get_citers(
        paper_id, refresh=refresh, limit=limit, sort=sort
    )


def get_citer_count(paper_id: str, *, refresh: bool = False) -> int:
    return ArcPaperService().get_citer_count(paper_id, refresh=refresh)


def search_citers(
    paper_id: str,
    terms: Sequence[str],
    *,
    refresh: bool = False,
    scan_limit: int = 1000,
    limit: int = 50,
) -> dict[str, Any]:
    return ArcPaperService().search_citers(
        paper_id,
        terms,
        refresh=refresh,
        scan_limit=scan_limit,
        limit=limit,
    )


def search_metadata(query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    return ArcPaperService().search_metadata(query, limit=limit)


def cache_document(
    source: SourceArtifact | ParsedDocument,
    *,
    cache_root: str | Path | None = None,
) -> CachedDocumentRef:
    return ArcPaperService(cache_root=cache_root).cache_document(source)


def reconstruct_cached_structure(
    document: CachedDocumentRef,
    outline_document: CachedDocumentRef,
    *,
    cache_root: str | Path | None = None,
) -> CachedDocumentStructureRef:
    return ArcPaperService(cache_root=cache_root).reconstruct_cached_structure(
        document, outline_document
    )


def lookup_reference_cli(
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    url: str | None = None,
    title: str | None = None,
    cache_root: str | Path | None = None,
) -> CachedReferenceMaterial | None:
    return ArcPaperService(cache_root=cache_root).lookup_reference(
        doi=doi, arxiv_id=arxiv_id, url=url, title=title
    )


def acquire_reference_cli(
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    url: str | None = None,
    title: str | None = None,
    refresh: bool = False,
    cache_root: str | Path | None = None,
) -> CachedReferenceMaterial:
    return ArcPaperService(cache_root=cache_root).acquire_reference(
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        title=title,
        refresh=refresh,
    )


def get_table_of_contents_cli(
    target: DocumentTarget,
    *,
    structure: CachedDocumentStructureRef | None = None,
    source_format: str | None = None,
    refresh: bool = False,
    cache_root: str | Path | None = None,
) -> PaperTableOfContents:
    return ArcPaperService(cache_root=cache_root).get_table_of_contents(
        target,
        structure=structure,
        source_format=source_format,
        refresh=refresh,
    )


def get_section_cli(
    target: DocumentTarget,
    selector: str | int,
    *,
    structure: CachedDocumentStructureRef | None = None,
    source_format: str | None = None,
    refresh: bool = False,
    cache_root: str | Path | None = None,
) -> PaperSection:
    return ArcPaperService(cache_root=cache_root).get_section(
        target,
        selector,
        structure=structure,
        source_format=source_format,
        refresh=refresh,
    )


def search_full_text_cli(
    terms: Sequence[str],
    *,
    targets: Sequence[DocumentTarget] = (),
    source_format: str | None = None,
    refresh: bool = False,
    limit: int = 100,
    context_lines: int = 0,
    case_sensitive: bool = False,
    cache_root: str | Path | None = None,
) -> PaperFullTextSearch:
    return ArcPaperService(cache_root=cache_root).search_full_text_targets(
        terms,
        targets=targets,
        source_format=source_format,
        refresh=refresh,
        limit=limit,
        context_lines=context_lines,
        case_sensitive=case_sensitive,
    )


def search_equations_cli(
    targets: Sequence[DocumentTarget],
    terms: Sequence[str],
    *,
    source_format: str | None = None,
    refresh: bool = False,
    limit: int = 20,
    context_lines: int = 8,
    case_sensitive: bool = False,
    cache_root: str | Path | None = None,
) -> PaperEquationSearch:
    return ArcPaperService(cache_root=cache_root).search_equation_targets(
        targets,
        terms,
        source_format=source_format,
        refresh=refresh,
        limit=limit,
        context_lines=context_lines,
        case_sensitive=case_sensitive,
    )


def admit_reference_cli(
    path: str | Path,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    url: str | None = None,
    title: str | None = None,
    media_type: str | None = None,
    cache_root: str | Path | None = None,
) -> CachedReferenceMaterial:
    return ArcPaperService(cache_root=cache_root).admit_reference(
        path,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        title=title,
        media_type=media_type,
    )


def materialize_reference_cli(
    resource: CachedResourceRef,
    output: str | Path,
    *,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    return ArcPaperService(cache_root=cache_root).materialize_reference(
        resource, output
    )


def read_cached_source_range(
    document: CachedDocumentRef,
    start_line: int,
    end_line: int,
    *,
    text_only: bool = False,
    cache_root: str | Path | None = None,
) -> CachedSourceRange:
    return ArcPaperService(cache_root=cache_root).read_cached_source_range(
        document,
        start_line,
        end_line,
        text_only=text_only,
    )


def search_full_text(
    documents: ParsedDocument | Iterable[ParsedDocument],
    query: str,
    *,
    limit: int = 20,
    context_lines: int = 1,
    case_sensitive: bool = False,
) -> FullTextSearchResult:
    return _search_full_text(
        documents,
        query,
        limit=limit,
        context_lines=context_lines,
        case_sensitive=case_sensitive,
    )


def search_equations(
    documents: ParsedDocument | Iterable[ParsedDocument],
    query: str,
    *,
    limit: int = 20,
    case_sensitive: bool = False,
) -> EquationSearchResult:
    return _search_equations(
        documents,
        query,
        limit=limit,
        case_sensitive=case_sensitive,
    )


def table_of_contents(
    document: ParsedDocument,
) -> tuple[TableOfContentsEntry, ...]:
    return _table_of_contents(document)


def select_section(
    document: ParsedDocument, selector: str | int
) -> ParsedSection:
    return _select_section(document, selector)


def import_source(
    path: str | Path,
    *,
    cache_root: str | Path | None = None,
    source_format: SourceFormat | str | None = None,
) -> SourceArtifact:
    return ArcPaperService(cache_root=cache_root).import_source(
        path, source_format=source_format
    )


def fetch_arxiv_auto(
    paper_id: str,
    *,
    cache_root: str | Path | None = None,
    refresh: bool = False,
) -> SourceArtifact:
    return ArcPaperService(cache_root=cache_root).fetch_arxiv_auto(
        paper_id, refresh=refresh
    )


def fetch_arxiv_html_bundle(
    paper_id: str,
    *,
    cache_root: str | Path | None = None,
    refresh: bool = False,
) -> HtmlSourceBundle:
    return ArcPaperService(cache_root=cache_root).fetch_arxiv_html_bundle(
        paper_id, refresh=refresh
    )


def export_arxiv_html_bundle(
    paper_id: str,
    *,
    output_dir: str | Path,
    cache_root: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    return ArcPaperService(cache_root=cache_root).export_arxiv_html_bundle(
        paper_id,
        output_dir=output_dir,
        refresh=refresh,
    )


def fetch_arxiv_pdf(
    paper_id: str,
    *,
    cache_root: str | Path | None = None,
    refresh: bool = False,
) -> SourceArtifact:
    return ArcPaperService(cache_root=cache_root).fetch_arxiv_pdf(
        paper_id, refresh=refresh
    )


def parse_local(
    primary_path: str | Path,
    *,
    validator_paths: Sequence[str | Path] = (),
    cache_root: str | Path | None = None,
    primary_format: SourceFormat | str | None = None,
    validator_formats: Sequence[SourceFormat | str | None] = (),
    policy: ValidationPolicy | str | None = None,
) -> ParseOutcome:
    return ArcPaperService(cache_root=cache_root).parse_local(
        primary_path,
        validator_paths=validator_paths,
        primary_format=primary_format,
        validator_formats=validator_formats,
        policy=policy,
    )


def export_rich_document(
    source: str | Path,
    *,
    output_dir: str | Path,
    validator: str | Path | None = None,
    cache_root: str | Path | None = None,
    source_format: SourceFormat | str | None = None,
) -> dict[str, object]:
    return ArcPaperService(cache_root=cache_root).export_rich_document(
        source,
        output_dir=output_dir,
        validator=validator,
        source_format=source_format,
    )


def extract_keywords(
    source: str | Path,
    *,
    project_dir: str | Path,
    structure_ref: Mapping[str, Any] | None = None,
    section_ids: Sequence[str] | None = None,
    approx_count: int = 50,
    cache_root: str | Path | None = None,
    refresh: bool = False,
    llm_provider: str = "auto",
    model: str | None = None,
    model_tier: str = "medium",
    run_id: str | None = None,
    resume_input: Mapping[str, JsonValue] | None = None,
    host_authority: str = HostAuthority.UNKNOWN.value,
) -> KeywordResult:
    service = ArcPaperService(cache_root=cache_root)
    source_text = str(source)
    artifact = service.resolve_local_or_arxiv_source(
        source_text, refresh=refresh
    )
    return service.extract_keywords(
        artifact,
        project_dir=project_dir,
        structure=(
            cached_document_structure_ref_from_document(structure_ref)
            if structure_ref is not None
            else None
        ),
        section_ids=section_ids,
        approx_count=approx_count,
        model=ModelSelection(
            provider=llm_provider,
            model=model,
            tier=model_tier,
        ),
        run_id=run_id,
        resume_input=resume_input,
        options=LLMExecutionOptions(host_authority=HostAuthority(host_authority)),
    )


__all__ = [
    "ArcPaperService",
    "CacheListResult",
    "CacheExportResult",
    "CacheImportResult",
    "CacheRemoveResult",
    "CacheUpdateRecord",
    "CacheUpdateResult",
    "PaperInputError",
    "acquire_reference_cli",
    "admit_reference_cli",
    "cache_document",
    "default_cache_root",
    "extract_paper_ids",
    "export_arxiv_html_bundle",
    "export_cache",
    "export_rich_document",
    "extract_keywords",
    "fetch_arxiv_auto",
    "fetch_arxiv_html_bundle",
    "fetch_arxiv_pdf",
    "get_abstract",
    "get_authors",
    "get_citer_count",
    "get_citers",
    "get_metadata",
    "get_references",
    "get_title",
    "import_source",
    "import_cache",
    "list_cache",
    "lookup_reference_cli",
    "materialize_reference_cli",
    "paper_ids_safe_dir_name",
    "parse_local",
    "remove_cache",
    "reconstruct_cached_structure",
    "select_section",
    "search_equations",
    "search_citers",
    "search_full_text",
    "search_metadata",
    "table_of_contents",
    "update_cache",
]
