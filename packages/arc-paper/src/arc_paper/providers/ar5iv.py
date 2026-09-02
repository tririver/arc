from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from ..html_dependencies import (
    AR5IV_HTML_ACQUISITION_NAMESPACE,
    AR5IV_HTML_DEPENDENCY_NAMESPACE,
    DEFAULT_MAX_DEPENDENCY_BYTES,
    DEFAULT_MAX_DEPENDENCY_COUNT,
    DEFAULT_MAX_DEPENDENCY_REDIRECTS,
    DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES,
    HtmlDependencyAcquirer,
    HtmlSourceBundle,
    HtmlSourceBundleError,
    acquire_html_dependencies,
    fetch_cached_html_bundle,
    fetch_safe_response,
)
from ..ids import arxiv_path_id
from ..reference_cache import ReferenceMaterialCache
from ..source_repository import SourceRepository
from ..sources import SourceArtifact, SourceFormat, SourceOrigin, SourceOriginKind
from ._http import require_https_host, response_media_type, validate_response_size
from ._request_gate import HostRequestGate, shared_host_gate
from .base import ProviderError
from .remote_cache import RemoteRequestCache


AR5IV_HOST = "ar5iv.labs.arxiv.org"
AR5IV_MEDIA_TYPE = "text/html"
MAX_HTML_BYTES = 50 * 1024 * 1024


def ar5iv_url(paper_id: str) -> str:
    aid = arxiv_path_id(paper_id)
    if not aid:
        raise ProviderError("not_arxiv_id", f"ar5iv requires an arXiv ID: {paper_id}")
    return f"https://{AR5IV_HOST}/html/{aid}"


class Ar5ivProvider:
    """Fetch ar5iv HTML into the content-addressed source repository."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        cache_root: str | Path | None = None,
        source_repository: SourceRepository | None = None,
        request_cache: RemoteRequestCache | None = None,
        request_gate: HostRequestGate | None = None,
        dependency_acquirer: HtmlDependencyAcquirer = acquire_html_dependencies,
        dependency_resolver: Callable[[str], Sequence[str]] | None = None,
        dependency_transport_factory: Callable[[httpx.Client, Any], Any] | None = None,
        max_dependency_count: int = DEFAULT_MAX_DEPENDENCY_COUNT,
        max_dependency_bytes: int = DEFAULT_MAX_DEPENDENCY_BYTES,
        max_total_dependency_bytes: int = DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES,
        max_dependency_redirects: int = DEFAULT_MAX_DEPENDENCY_REDIRECTS,
    ):
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.timeout = timeout
        self.cache = request_cache or RemoteRequestCache(
            cache_root, source_repository=source_repository
        )
        # ar5iv publishes no crawl-delay contract.  Keep one active request and
        # honor Retry-After without inventing arXiv's 15-second interval.
        self.request_gate = request_gate or shared_host_gate(
            self.cache.root, AR5IV_HOST, minimum_interval=0
        )
        self.resource_cache = ReferenceMaterialCache(self.cache.root)
        self.dependency_acquirer = dependency_acquirer
        self.dependency_resolver = dependency_resolver
        self.dependency_transport_factory = dependency_transport_factory
        self.max_dependency_count = max_dependency_count
        self.max_dependency_bytes = max_dependency_bytes
        self.max_total_dependency_bytes = max_total_dependency_bytes
        self.max_dependency_redirects = max_dependency_redirects

    def fetch(self, paper_id: str, *, refresh: bool = False) -> SourceArtifact:
        url = ar5iv_url(paper_id)
        aid = arxiv_path_id(paper_id)
        origin = SourceOrigin(
            kind=SourceOriginKind.REMOTE_PROVIDER,
            provider="ar5iv",
            locator=url,
            metadata={"arxiv_id": aid, "document_id": f"arXiv:{aid}"},
        )
        return self.cache.fetch_source(
            "ar5iv-html",
            aid,
            source_format=SourceFormat.HTML,
            media_type=AR5IV_MEDIA_TYPE,
            origin=origin,
            refresh=refresh,
            fetch=lambda: self._fetch_html_bytes(url, paper_id),
        )

    def fetch_bundle(
        self, paper_id: str, *, refresh: bool = False
    ) -> HtmlSourceBundle:
        """Fetch ar5iv HTML plus bounded authored image dependencies."""

        url = ar5iv_url(paper_id)
        aid = arxiv_path_id(paper_id)
        origin = SourceOrigin(
            kind=SourceOriginKind.REMOTE_PROVIDER,
            provider="ar5iv",
            locator=url,
            metadata={"arxiv_id": aid, "document_id": f"arXiv:{aid}"},
        )
        return fetch_cached_html_bundle(
            cache=self.cache,
            resource_cache=self.resource_cache,
            source_namespace="ar5iv-html",
            dependency_namespace=AR5IV_HTML_DEPENDENCY_NAMESPACE,
            request_key=aid,
            source_origin=origin,
            provider="ar5iv",
            allowed_host=AR5IV_HOST,
            client=self.client,
            request_gate=self.request_gate,
            timeout=self.timeout,
            refresh=refresh,
            fetch_main=lambda: self._fetch_html_document(url, paper_id),
            dependency_acquirer=self.dependency_acquirer,
            sidecar_namespace=AR5IV_HTML_ACQUISITION_NAMESPACE,
            requested_url=url,
            dependency_resolver=self.dependency_resolver,
            dependency_transport_factory=self.dependency_transport_factory,
            max_dependency_count=self.max_dependency_count,
            max_dependency_bytes=self.max_dependency_bytes,
            max_total_dependency_bytes=self.max_total_dependency_bytes,
            max_redirects=self.max_dependency_redirects,
        )

    def _fetch_html_bytes(self, url: str, paper_id: str) -> bytes:
        payload, _url = self._fetch_html_document(url, paper_id)
        return payload

    def _fetch_html_document(self, url: str, paper_id: str) -> tuple[bytes, str]:
        require_https_host(url, AR5IV_HOST)
        try:
            response = fetch_safe_response(
                self.client,
                self.request_gate,
                url,
                allowed_hosts=(AR5IV_HOST,),
                timeout=self.timeout,
                max_redirects=self.max_dependency_redirects,
                maximum_bytes=MAX_HTML_BYTES,
                size_error_code="ar5iv_html_too_large",
            )
        except HtmlSourceBundleError as exc:
            raise ProviderError(exc.code, exc.message) from exc
        if response.status_code == 404:
            raise ProviderError(
                "ar5iv_not_found", f"ar5iv HTML not found for {paper_id}"
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                "ar5iv_fetch_failed",
                str(exc),
                status_code=exc.response.status_code,
            ) from exc
        require_https_host(str(response.url), AR5IV_HOST)
        validate_response_size(response, MAX_HTML_BYTES, "ar5iv_html_too_large")
        media_type = response_media_type(response)
        if media_type != AR5IV_MEDIA_TYPE:
            raise ProviderError(
                "ar5iv_media_type_invalid",
                f"ar5iv returned unsupported media type: {media_type or '<missing>'}",
            )
        return bytes(response.content), str(response.url)


__all__ = [
    "AR5IV_HTML_DEPENDENCY_NAMESPACE",
    "AR5IV_MEDIA_TYPE",
    "Ar5ivProvider",
    "MAX_HTML_BYTES",
    "ar5iv_url",
]
