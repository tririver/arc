from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from ..ids import (
    arxiv_path_id,
    arxiv_version,
    arxiv_version_is_invalid,
    arxiv_versioned_path_id,
)
from ..html_dependencies import (
    ARXIV_HTML_DEPENDENCY_NAMESPACE,
    DEFAULT_MAX_DEPENDENCY_BYTES,
    DEFAULT_MAX_DEPENDENCY_COUNT,
    DEFAULT_MAX_DEPENDENCY_REDIRECTS,
    DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES,
    HtmlSourceBundle,
    HtmlSourceBundleError,
    fetch_cached_html_bundle,
    fetch_safe_response,
)
from ..reference_cache import ReferenceMaterialCache
from ..source_repository import SourceRepository
from ..sources import SourceArtifact, SourceFormat, SourceOrigin, SourceOriginKind
from ._http import require_https_host, response_media_type, validate_response_size
from ._request_gate import HostRequestGate, shared_host_gate
from .base import ProviderError
from .remote_cache import RemoteCacheError, RemoteRequestCache


ARXIV_HOST = "arxiv.org"
ARXIV_HTML_MEDIA_TYPE = "text/html"
ARXIV_HTML_AVAILABILITY_NAMESPACE = "arxiv-html-availability"
MAX_HTML_BYTES = 50 * 1024 * 1024


def arxiv_html_url(paper_id: str) -> str:
    aid = arxiv_path_id(paper_id)
    if not aid:
        raise ProviderError(
            "not_arxiv_id", f"arXiv HTML requires an arXiv ID: {paper_id}"
        )
    return f"https://{ARXIV_HOST}/html/{aid}"


def arxiv_html_bundle_url(paper_id: str) -> str:
    """Preserve an explicit version for dependency-aware bundle acquisition."""

    if arxiv_version_is_invalid(paper_id):
        raise ProviderError(
            "arxiv_version_invalid",
            f"arXiv HTML requires a positive version without leading zeros: {paper_id}",
        )
    request_key = arxiv_versioned_path_id(paper_id)
    if not request_key:
        raise ProviderError(
            "not_arxiv_id", f"arXiv HTML requires an arXiv ID: {paper_id}"
        )
    return f"https://{ARXIV_HOST}/html/{request_key}"


class ArxivHtmlProvider:
    """Fetch official arXiv HTML, caching deterministic absence separately."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        cache_root: str | Path | None = None,
        source_repository: SourceRepository | None = None,
        request_cache: RemoteRequestCache | None = None,
        request_gate: HostRequestGate | None = None,
        max_dependency_count: int = DEFAULT_MAX_DEPENDENCY_COUNT,
        max_dependency_bytes: int = DEFAULT_MAX_DEPENDENCY_BYTES,
        max_total_dependency_bytes: int = DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES,
        max_dependency_redirects: int = DEFAULT_MAX_DEPENDENCY_REDIRECTS,
    ) -> None:
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.timeout = timeout
        self.cache = request_cache or RemoteRequestCache(
            cache_root, source_repository=source_repository
        )
        self.request_gate = request_gate or shared_host_gate(
            self.cache.root, ARXIV_HOST
        )
        self.resource_cache = ReferenceMaterialCache(self.cache.root)
        self.max_dependency_count = max_dependency_count
        self.max_dependency_bytes = max_dependency_bytes
        self.max_total_dependency_bytes = max_total_dependency_bytes
        self.max_dependency_redirects = max_dependency_redirects

    def fetch(self, paper_id: str, *, refresh: bool = False) -> SourceArtifact:
        aid = arxiv_path_id(paper_id)
        url = arxiv_html_url(paper_id)
        origin = SourceOrigin(
            kind=SourceOriginKind.REMOTE_PROVIDER,
            provider="arxiv-html",
            locator=url,
            metadata={"arxiv_id": aid, "document_id": f"arXiv:{aid}"},
        )
        known_not_found = self._is_known_not_found(aid)
        if not refresh and known_not_found:
            raise _not_found_error(paper_id)
        if known_not_found:
            # Refresh must not leave a stale 404 marker after a transient error.
            self.cache.remove("json", ARXIV_HTML_AVAILABILITY_NAMESPACE, aid)
        artifact = self.cache.fetch_source(
            "arxiv-html",
            aid,
            source_format=SourceFormat.HTML,
            media_type=ARXIV_HTML_MEDIA_TYPE,
            origin=origin,
            refresh=refresh,
            fetch=lambda: self._fetch_html_bytes(
                url,
                paper_id,
                aid,
                honor_cached_not_found=not refresh,
            ),
        )
        return self._with_resolved_revision(artifact, aid, url)

    def fetch_bundle(
        self, paper_id: str, *, refresh: bool = False
    ) -> HtmlSourceBundle:
        """Fetch official HTML plus bounded authored image dependencies."""

        aid = arxiv_path_id(paper_id)
        requested_revision = arxiv_version(paper_id)
        request_key = arxiv_versioned_path_id(paper_id)
        url = arxiv_html_bundle_url(paper_id)
        requested_document_id = f"arXiv:{request_key}"
        origin = SourceOrigin(
            kind=SourceOriginKind.REMOTE_PROVIDER,
            provider="arxiv-html",
            locator=url,
            metadata={
                "arxiv_id": aid,
                "document_id": requested_document_id,
                **(
                    {"arxiv_version": requested_revision}
                    if requested_revision
                    else {}
                ),
            },
        )
        known_not_found = self._is_known_not_found(request_key)
        if not refresh and known_not_found:
            raise _not_found_error(paper_id)
        if known_not_found:
            self.cache.remove(
                "json", ARXIV_HTML_AVAILABILITY_NAMESPACE, request_key
            )
        return fetch_cached_html_bundle(
            cache=self.cache,
            resource_cache=self.resource_cache,
            source_namespace="arxiv-html",
            dependency_namespace=ARXIV_HTML_DEPENDENCY_NAMESPACE,
            request_key=request_key,
            source_origin=origin,
            provider="arxiv-html",
            allowed_host=ARXIV_HOST,
            client=self.client,
            request_gate=self.request_gate,
            timeout=self.timeout,
            refresh=refresh,
            fetch_main=lambda: self._fetch_html_document(
                url,
                paper_id,
                request_key,
                honor_cached_not_found=not refresh,
            ),
            build_origin=lambda payload, document_url: self._resolved_bundle_origin(
                payload,
                aid=aid,
                requested_revision=requested_revision,
                document_url=document_url,
            ),
            max_dependency_count=self.max_dependency_count,
            max_dependency_bytes=self.max_dependency_bytes,
            max_total_dependency_bytes=self.max_total_dependency_bytes,
            max_redirects=self.max_dependency_redirects,
        )

    def _with_resolved_revision(
        self,
        artifact: SourceArtifact,
        aid: str,
        url: str,
    ) -> SourceArtifact:
        revision = _arxiv_html_revision(
            self.cache.source_repository.read_bytes(artifact), aid
        )
        if revision is None:
            return artifact
        return SourceArtifact(
            source_format=artifact.source_format,
            artifact_digest=artifact.artifact_digest,
            size=artifact.size,
            media_type=artifact.media_type,
            origin=SourceOrigin(
                kind=SourceOriginKind.REMOTE_PROVIDER,
                provider="arxiv-html",
                locator=url,
                metadata={
                    "arxiv_id": aid,
                    "document_id": f"arXiv:{aid}",
                    "arxiv_version": revision,
                },
            ),
        )

    @staticmethod
    def _resolved_bundle_origin(
        payload: bytes,
        *,
        aid: str,
        requested_revision: str,
        document_url: str,
    ) -> SourceOrigin:
        _require_requested_revision_url(
            document_url,
            aid=aid,
            requested_revision=requested_revision,
        )
        try:
            revision = _verified_arxiv_html_revision(
                payload,
                aid,
                expected_revision=requested_revision,
            )
        except ValueError as exc:
            code = str(exc)
            raise ProviderError(code, _revision_error_message(code)) from exc
        metadata = {
            "arxiv_id": aid,
            "document_id": (
                f"arXiv:{aid}{requested_revision}"
                if requested_revision
                else f"arXiv:{aid}"
            ),
        }
        if revision:
            metadata["arxiv_version"] = revision
        return SourceOrigin(
            kind=SourceOriginKind.REMOTE_PROVIDER,
            provider="arxiv-html",
            locator=document_url,
            metadata=metadata,
        )

    def _is_known_not_found(self, request_key: str) -> bool:
        try:
            value = self.cache.get_json(
                ARXIV_HTML_AVAILABILITY_NAMESPACE, request_key
            )
        except RemoteCacheError as exc:
            if exc.code != "remote_cache_json_corrupt":
                raise
            self.cache.remove(
                "json", ARXIV_HTML_AVAILABILITY_NAMESPACE, request_key
            )
            return False
        return value == {"status": "not_found"}

    def _fetch_html_bytes(
        self,
        url: str,
        paper_id: str,
        request_key: str,
        *,
        honor_cached_not_found: bool,
    ) -> bytes:
        payload, _url = self._fetch_html_document(
            url,
            paper_id,
            request_key,
            honor_cached_not_found=honor_cached_not_found,
        )
        return payload

    def _fetch_html_document(
        self,
        url: str,
        paper_id: str,
        request_key: str,
        *,
        honor_cached_not_found: bool,
    ) -> tuple[bytes, str]:
        if honor_cached_not_found and self._is_known_not_found(request_key):
            raise _not_found_error(paper_id)
        require_https_host(url, ARXIV_HOST)
        try:
            response = fetch_safe_response(
                self.client,
                self.request_gate,
                url,
                allowed_hosts=(ARXIV_HOST,),
                timeout=self.timeout,
                max_redirects=self.max_dependency_redirects,
                maximum_bytes=MAX_HTML_BYTES,
                size_error_code="arxiv_html_too_large",
            )
        except HtmlSourceBundleError as exc:
            raise ProviderError(exc.code, exc.message) from exc
        if response.status_code == 404:
            self.cache.fetch_json(
                ARXIV_HTML_AVAILABILITY_NAMESPACE,
                request_key,
                fetch=lambda: {"status": "not_found"},
                refresh=True,
            )
            raise _not_found_error(paper_id)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                "arxiv_html_fetch_failed",
                str(exc),
                status_code=exc.response.status_code,
            ) from exc
        require_https_host(str(response.url), ARXIV_HOST)
        validate_response_size(response, MAX_HTML_BYTES, "arxiv_html_too_large")
        media_type = response_media_type(response)
        if media_type != ARXIV_HTML_MEDIA_TYPE:
            raise ProviderError(
                "arxiv_html_media_type_invalid",
                "arXiv returned unsupported media type: "
                f"{media_type or '<missing>'}",
            )
        return bytes(response.content), str(response.url)


def _not_found_error(paper_id: str) -> ProviderError:
    return ProviderError(
        "arxiv_html_not_found", f"arXiv HTML not found for {paper_id}"
    )


class _RevisionSignalParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.signals: list[tuple[str, str]] = []
        self._base_seen = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        name = tag.casefold()
        values = {
            key.casefold(): str(value or "")
            for key, value in attrs
        }
        if name == "base" and not self._base_seen:
            self._base_seen = True
            self.signals.append(("html", values.get("href", "")))
            return
        if name != "a" or "header-button" not in values.get("class", "").split():
            return
        title = values.get("title", "").strip().casefold()
        aria_label = values.get("aria-label", "").strip().casefold()
        if title == "back to abstract page" and aria_label == title:
            self.signals.append(("abs", values.get("href", "")))
        elif (
            title == "download pdf"
            and values.get("target", "").strip().casefold() == "_blank"
        ):
            self.signals.append(("pdf", values.get("href", "")))


def _arxiv_html_revision(payload: bytes, aid: str) -> str | None:
    """Read an official conversion revision without changing the canonical ID."""

    try:
        return _verified_arxiv_html_revision(payload, aid, expected_revision="")
    except ValueError:
        return None


def _verified_arxiv_html_revision(
    payload: bytes,
    aid: str,
    *,
    expected_revision: str,
) -> str | None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        if expected_revision:
            raise ValueError("arxiv_html_revision_unverified")
        return None
    parser = _RevisionSignalParser()
    try:
        parser.feed(text)
        parser.close()
    except ValueError:
        if expected_revision:
            raise ValueError("arxiv_html_revision_unverified")
        return None

    revisions: list[str] = []
    invalid = False
    header_kinds: set[str] = set()
    for kind, href in parser.signals:
        revision, relevant, valid = _revision_from_signal(href, aid=aid, kind=kind)
        if not relevant:
            continue
        if kind in {"abs", "pdf"}:
            header_kinds.add(kind)
        if not valid:
            invalid = True
        elif revision:
            revisions.append(revision)
    if header_kinds and header_kinds != {"abs", "pdf"}:
        invalid = True
    distinct = set(revisions)
    if invalid or len(distinct) > 1:
        raise ValueError("arxiv_html_revision_mismatch")
    revision = next(iter(distinct), None)
    if expected_revision:
        if revision is None:
            raise ValueError("arxiv_html_revision_unverified")
        if revision != expected_revision:
            raise ValueError("arxiv_html_revision_mismatch")
    return revision


def _revision_from_signal(
    href: str,
    *,
    aid: str,
    kind: str,
) -> tuple[str | None, bool, bool]:
    route = {"html": "html", "abs": "abs", "pdf": "pdf"}[kind]
    try:
        parsed = urlsplit(str(href or "").strip())
        port = parsed.port
    except ValueError:
        return None, True, False
    if kind == "html":
        if not parsed.path.startswith("/html/"):
            return None, False, True
        if re.fullmatch(
            r"/html/[^/]+v[1-9][0-9]*/?",
            parsed.path,
            flags=re.IGNORECASE,
        ) is None:
            return None, False, True
    if (
        parsed.scheme not in {"", "https"}
        or (parsed.hostname or ARXIV_HOST).casefold() != ARXIV_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
    ):
        return None, True, False
    match = re.fullmatch(
        rf"/{route}/{re.escape(aid)}(v[1-9][0-9]*)/?",
        parsed.path,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, True, False
    return match.group(1).casefold(), True, True


def _require_requested_revision_url(
    document_url: str,
    *,
    aid: str,
    requested_revision: str,
) -> None:
    if not requested_revision:
        return
    parsed = urlsplit(document_url)
    expected_path = f"/html/{aid}{requested_revision}"
    if parsed.path.rstrip("/") != expected_path:
        raise ProviderError(
            "arxiv_html_revision_mismatch",
            "arXiv HTML final response URL does not match the requested version",
        )


def _revision_error_message(code: str) -> str:
    if code == "arxiv_html_revision_unverified":
        return "arXiv HTML did not provide a verified revision signal"
    return "arXiv HTML revision signals do not match the requested version"


__all__ = [
    "ARXIV_HTML_AVAILABILITY_NAMESPACE",
    "ARXIV_HTML_DEPENDENCY_NAMESPACE",
    "ARXIV_HTML_MEDIA_TYPE",
    "ArxivHtmlProvider",
    "MAX_HTML_BYTES",
    "arxiv_html_bundle_url",
    "arxiv_html_url",
]
