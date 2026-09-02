"""ARC adapters for the public AC Foundation HTML acquisition contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
from typing import Any
from urllib.parse import urljoin

import httpx

from .html_dependencies import (
    DEFAULT_MAX_DEPENDENCY_BYTES,
    DEFAULT_MAX_DEPENDENCY_COUNT,
    DEFAULT_MAX_DEPENDENCY_REDIRECTS,
    DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES,
    HtmlDependency,
    HtmlDependencyAcquisitionResult,
    HtmlDependencyWarning,
    HtmlSourceBundle,
    HtmlSourceBundleError,
    _source_artifact_from_document,
    _source_artifact_to_document,
)
from .reference_cache import (
    CachedResourceRef,
    ReferenceCacheError,
    ReferenceMaterialCache,
)
from .sources import SourceArtifact


HTML_ACQUISITION_SIDECAR_SCHEMA = "arc.paper.html_acquisition_sidecar.v1"
HTML_ACQUISITION_SIDECAR_FIELDS = {
    "schema_version",
    "request_key",
    "primary",
    "bundle",
}
LEGACY_SIDECAR_MISSING_WARNING = "arc_html_acquisition_sidecar_missing"


class _AcquisitionProvenance:
    def __init__(self) -> None:
        self.current_request_url = ""
        self.storage_failures: dict[str, list[str]] = {}


class ReferenceMaterialCacheHTMLStorage:
    """Adapt ARC source and resource storage to ACF's bundle storage protocol."""

    def __init__(
        self,
        source_repository: Any,
        resource_cache: ReferenceMaterialCache,
        provenance: _AcquisitionProvenance,
    ):
        self.source_repository = source_repository
        self.resource_cache = resource_cache
        self.provenance = provenance

    def read_primary(self, primary: SourceArtifact) -> bytes:
        return self.source_repository.read_bytes(primary)

    def store_dependency(self, payload: bytes, *, media_type: str):
        from ac_document import HTMLSourceBundleError, StoredHTMLDependency

        try:
            resource = self.resource_cache.store_resource(payload, media_type=media_type)
        except (OSError, ReferenceCacheError) as exc:
            self.provenance.storage_failures.setdefault(
                self.provenance.current_request_url, []
            ).append(str(exc))
            raise HTMLSourceBundleError(
                "html_dependency_cache_write_failed",
                "dependency storage write failed",
            ) from exc
        return StoredHTMLDependency(
            artifact_digest=resource.resource_sha256,
            media_type=resource.media_type,
            size=resource.resource_size,
        )

    def read_dependency(
        self, *, artifact_digest: str, media_type: str, size: int
    ) -> bytes:
        return self.resource_cache.read_resource(
            CachedResourceRef(
                resource_sha256=artifact_digest,
                resource_size=size,
                media_type=media_type,
            )
        )


class HttpxRequestGateTransport:
    """Explicit test transport for an injected httpx client."""

    def __init__(self, client: httpx.Client, request_gate: Any):
        self.client = client
        self.request_gate = request_gate
        self.response_statuses: dict[str, int] = {}
        self.redirect_locations: dict[str, str] = {}
        self.provenance: _AcquisitionProvenance | None = None

    def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        maximum_bytes: int,
        validated_addresses: Sequence[str],
    ):
        from ac_document import HTMLSourceBundleError, WebResponse

        if self.provenance is not None:
            self.provenance.current_request_url = url

        def request() -> httpx.Response:
            return self.client.get(
                url,
                timeout=timeout_seconds,
                follow_redirects=False,
            )

        try:
            response = self.request_gate.request(request)
        except httpx.HTTPError as exc:
            raise HTMLSourceBundleError(
                "remote_fetch_failed", "remote request failed"
            ) from exc
        self.response_statuses[url] = response.status_code
        if response.headers.get("location"):
            self.redirect_locations[url] = response.headers["location"]
        if response.status_code in {301, 302, 303, 307, 308} or not 200 <= response.status_code < 300:
            return WebResponse(
                url=str(response.url),
                status=response.status_code,
                headers=dict(response.headers),
                body=b"",
            )
        body = bytes(response.content)
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError as exc:
                raise HTMLSourceBundleError(
                    "remote_content_length_invalid",
                    "remote response has an invalid Content-Length",
                ) from exc
            if declared < 0:
                raise HTMLSourceBundleError(
                    "remote_content_length_invalid",
                    "remote response has an invalid Content-Length",
                )
            if declared > maximum_bytes:
                raise HTMLSourceBundleError(
                    "html_dependency_too_large",
                    "remote response exceeds the configured byte limit",
                )
        if len(body) > maximum_bytes:
            raise HTMLSourceBundleError(
                "html_dependency_too_large",
                "remote response exceeds the configured byte limit",
            )
        return WebResponse(
            url=str(response.url),
            status=response.status_code,
            headers=dict(response.headers),
            body=body,
        )


class GatedPinnedTransport:
    """Preserve ACF's validated-address transport behind ARC's request gate."""

    def __init__(self, request_gate: Any, transport: Any):
        self.request_gate = request_gate
        self.transport = transport
        self.response_statuses: dict[str, int] = {}
        self.redirect_locations: dict[str, str] = {}
        self.provenance: _AcquisitionProvenance | None = None

    def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        maximum_bytes: int,
        validated_addresses: Sequence[str],
    ):
        if self.provenance is not None:
            self.provenance.current_request_url = url
        response = self.request_gate.request(
            lambda: self.transport.fetch(
                url,
                timeout_seconds=timeout_seconds,
                maximum_bytes=maximum_bytes,
                validated_addresses=validated_addresses,
            )
        )
        self.response_statuses[url] = response.status
        if response.headers.get("location"):
            self.redirect_locations[url] = response.headers["location"]
        return response


def acquire_html_dependencies(
    payload: bytes,
    *,
    primary: SourceArtifact,
    provider: str,
    document_url: str,
    requested_url: str | None,
    allowed_host: str,
    client: httpx.Client,
    request_gate: Any,
    resource_cache: ReferenceMaterialCache,
    source_repository: Any,
    timeout: float,
    max_dependency_count: int = DEFAULT_MAX_DEPENDENCY_COUNT,
    max_dependency_bytes: int = DEFAULT_MAX_DEPENDENCY_BYTES,
    max_total_dependency_bytes: int = DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES,
    max_redirects: int = DEFAULT_MAX_DEPENDENCY_REDIRECTS,
    resolver: Callable[[str], Sequence[str]] | None = None,
    transport_factory: Callable[[httpx.Client, Any], Any] | None = None,
) -> HtmlDependencyAcquisitionResult:
    """Acquire dependencies through ACF after ARC validates the HTML primary."""

    del payload
    from ac_document import (
        HTMLAcquisitionPolicy,
        HTMLSourceAcquisitionService,
        StdlibHTTPSWebTransport,
    )

    provenance = _AcquisitionProvenance()
    storage = ReferenceMaterialCacheHTMLStorage(
        source_repository, resource_cache, provenance
    )
    policy = HTMLAcquisitionPolicy(
        max_primary_bytes=max(primary.size, DEFAULT_MAX_DEPENDENCY_BYTES),
        max_dependency_count=max_dependency_count,
        max_dependency_bytes=max_dependency_bytes,
        max_total_dependency_bytes=max_total_dependency_bytes,
        max_redirects=max_redirects,
        timeout_seconds=timeout,
        same_origin_dependencies=True,
        allowed_origins=(f"https://{allowed_host}",),
    )
    transport = (
        transport_factory(client, request_gate)
        if transport_factory is not None
        else GatedPinnedTransport(request_gate, StdlibHTTPSWebTransport())
    )
    if hasattr(transport, "provenance"):
        transport.provenance = provenance
    service = HTMLSourceAcquisitionService(
        repository=source_repository,
        storage=storage,
        transport=transport,
        resolver=resolver,
    )
    generic = service.acquire_dependencies(
        primary,
        document_url=document_url,
        requested_url=requested_url,
        policy=policy,
        storage=storage,
    )
    bundle = _v2_bundle_from_acf(
        generic,
        provider=provider,
        response_statuses=getattr(transport, "response_statuses", {}),
        redirect_locations=getattr(transport, "redirect_locations", {}),
        max_dependency_bytes=max_dependency_bytes,
        storage_failures=provenance.storage_failures,
    )
    return HtmlDependencyAcquisitionResult(
        bundle=bundle,
        sidecar=html_acquisition_sidecar_to_document(
            generic,
            request_key=requested_url or document_url,
        ),
    )


def html_acquisition_sidecar_to_document(bundle: Any, *, request_key: str) -> dict[str, Any]:
    from ac_document import html_source_bundle_to_document

    return {
        "schema_version": HTML_ACQUISITION_SIDECAR_SCHEMA,
        "request_key": request_key,
        "primary": _source_artifact_to_document(bundle.primary),
        "bundle": html_source_bundle_to_document(bundle),
    }


def html_acquisition_sidecar_from_document(
    value: Any, *, request_key: str
) -> Any:
    from ac_document import html_source_bundle_from_document

    if not isinstance(value, Mapping) or set(value) != HTML_ACQUISITION_SIDECAR_FIELDS:
        raise ValueError("HTML acquisition sidecar has invalid fields")
    if value.get("schema_version") != HTML_ACQUISITION_SIDECAR_SCHEMA:
        raise ValueError("HTML acquisition sidecar schema is unsupported")
    if value.get("request_key") != request_key:
        raise ValueError("HTML acquisition sidecar request key does not match")
    primary = _source_artifact_from_document(value.get("primary"))
    bundle = html_source_bundle_from_document(value.get("bundle"))
    if bundle.primary != primary:
        raise ValueError("HTML acquisition sidecar primary does not match bundle")
    return bundle


def materialize_acquisition_sidecar(
    bundle: Any,
    *,
    source_repository: Any,
    resource_cache: ReferenceMaterialCache,
    output_dir: str,
) -> dict[str, Any]:
    from ac_document import (
        html_source_bundle_export_from_document,
        materialize_html_source_bundle,
        verify_html_source_bundle_export,
    )

    storage = ReferenceMaterialCacheHTMLStorage(
        source_repository, resource_cache, _AcquisitionProvenance()
    )
    materialized = materialize_html_source_bundle(bundle, storage, output_dir)
    export = html_source_bundle_export_from_document(
        verify_html_source_bundle_export(output_dir)
    )
    return {
        "source": str(materialized.source_path),
        "manifest": str(materialized.manifest_path),
        "bundle_digest": bundle.bundle_digest,
        "resources": [str(path) for path in materialized.resource_paths],
        "warnings": [_warning_to_document(item) for item in bundle.warnings],
        "export": export,
    }


def legacy_acquisition_fallback(bundle: HtmlSourceBundle, *, request_key: str) -> Any:
    """Make a visibly incomplete ACF bundle from a v2-only cache entry."""

    from ac_document import (
        HTMLAcquisitionPolicy,
        HTMLSourceBundle,
        HTMLSourceDependency,
        HTMLSourceWarning,
        materialization_path_for_target,
    )

    policy = HTMLAcquisitionPolicy(
        max_primary_bytes=max(bundle.primary.size, DEFAULT_MAX_DEPENDENCY_BYTES),
        max_dependency_count=bundle.acquisition_policy["max_dependency_count"],
        max_dependency_bytes=bundle.acquisition_policy["max_dependency_bytes"],
        max_total_dependency_bytes=bundle.acquisition_policy[
            "max_total_dependency_bytes"
        ],
        max_redirects=bundle.acquisition_policy["max_redirects"],
        same_origin_dependencies=True,
        allowed_origins=(_origin(bundle.document_url),),
    )
    dependencies = []
    warnings = [
        HTMLSourceWarning(
            code=warning.code,
            message=warning.message,
            dependency_ordinal=None,
            element=warning.element,
            attribute=warning.attribute,
            authored_target=warning.authored_target,
        )
        for warning in bundle.warnings
        if warning.dependency_ordinal is None
    ]
    for dependency in bundle.dependencies:
        if dependency.availability == "available":
            materialization_path = materialization_path_for_target(
                dependency.authored_target, dependency.resolved_url
            )
            if materialization_path == dependency.authored_target:
                dependencies.append(
                    HTMLSourceDependency(
                        ordinal=dependency.ordinal,
                        element=dependency.element,
                        attribute=dependency.attribute,
                        authored_target=dependency.authored_target,
                        request_url=dependency.request_url,
                        resolved_url=dependency.resolved_url,
                        declared_media_type=dependency.declared_media_type,
                        availability="available",
                        materialization_path=materialization_path,
                        media_type=dependency.media_type,
                        artifact_digest=dependency.artifact_digest,
                        size=dependency.size,
                    )
                )
                continue
            code = LEGACY_SIDECAR_MISSING_WARNING
            message = "strict HTML acquisition sidecar is unavailable for this legacy cache entry"
            dependencies.append(
                HTMLSourceDependency(
                    ordinal=dependency.ordinal,
                    element=dependency.element,
                    attribute=dependency.attribute,
                    authored_target=dependency.authored_target,
                    declared_media_type=dependency.declared_media_type,
                    availability="unavailable",
                    error_code=code,
                    error_message=message,
                )
            )
            warnings.append(
                HTMLSourceWarning(
                    code=code,
                    message=message,
                    dependency_ordinal=dependency.ordinal,
                    element=dependency.element,
                    attribute=dependency.attribute,
                    authored_target=dependency.authored_target,
                )
            )
            continue
        dependencies.append(
            HTMLSourceDependency(
                ordinal=dependency.ordinal,
                element=dependency.element,
                attribute=dependency.attribute,
                authored_target=dependency.authored_target,
                declared_media_type=dependency.declared_media_type,
                availability="unavailable",
                error_code=dependency.error_code,
                error_message=dependency.error_message,
            )
        )
        warnings.append(
            HTMLSourceWarning(
                code=dependency.error_code,
                message=dependency.error_message,
                dependency_ordinal=dependency.ordinal,
                element=dependency.element,
                attribute=dependency.attribute,
                authored_target=dependency.authored_target,
            )
        )
    warnings.append(
        HTMLSourceWarning(
            code=LEGACY_SIDECAR_MISSING_WARNING,
            message="strict HTML acquisition sidecar is unavailable for this legacy cache entry",
        )
    )
    return HTMLSourceBundle(
        primary=bundle.primary,
        requested_url=request_key,
        final_url=bundle.document_url,
        base_url=bundle.base_url,
        acquisition_policy=policy.to_document(),
        dependencies=tuple(dependencies),
        warnings=tuple(warnings),
    )


def _v2_bundle_from_acf(
    bundle: Any,
    *,
    provider: str,
    response_statuses: Mapping[str, int],
    redirect_locations: Mapping[str, str],
    max_dependency_bytes: int,
    storage_failures: Mapping[str, Sequence[str]],
) -> HtmlSourceBundle:
    dependencies = tuple(
        HtmlDependency(
            ordinal=item.ordinal,
            element=item.element,
            attribute=item.attribute,
            authored_target=item.authored_target,
            request_url=item.request_url,
            resolved_url=(item.resolved_url if item.availability == "available" else ""),
            declared_media_type=item.declared_media_type,
            availability=item.availability,
            media_type=item.media_type,
            artifact_digest=item.artifact_digest,
            size=item.size,
            error_code=_v2_error_code(item.error_code),
            error_message=_v2_error_message(
                item,
                response_statuses=response_statuses,
                redirect_locations=redirect_locations,
                max_dependency_bytes=max_dependency_bytes,
                storage_failures=storage_failures,
            ),
        )
        for item in bundle.dependencies
    )
    warnings = tuple(
        HtmlDependencyWarning(
            code=(
                dependencies[item.dependency_ordinal].error_code
                if item.dependency_ordinal is not None
                else _v2_error_code(item.code)
            ),
            message=(
                dependencies[item.dependency_ordinal].error_message
                if item.dependency_ordinal is not None
                else item.message
            ),
            dependency_ordinal=item.dependency_ordinal,
            element=item.element,
            attribute=item.attribute,
            authored_target=item.authored_target,
        )
        for item in bundle.warnings
    )
    return HtmlSourceBundle(
        primary=bundle.primary,
        provider=provider,
        document_url=bundle.final_url,
        base_url=bundle.base_url,
        acquisition_policy={
            "max_dependency_count": bundle.acquisition_policy[
                "max_dependency_count"
            ],
            "max_dependency_bytes": bundle.acquisition_policy[
                "max_dependency_bytes"
            ],
            "max_total_dependency_bytes": bundle.acquisition_policy[
                "max_total_dependency_bytes"
            ],
            "max_redirects": bundle.acquisition_policy["max_redirects"],
        },
        dependencies=dependencies,
        warnings=warnings,
    )


def _v2_error_code(value: str) -> str:
    return {
        "html_dependency_origin_invalid": "html_dependency_url_invalid",
        "remote_origin_not_allowed": "html_dependency_url_invalid",
        "remote_url_invalid": "html_dependency_url_invalid",
    }.get(value, value)


def _v2_error_message(
    dependency: Any,
    *,
    response_statuses: Mapping[str, int],
    redirect_locations: Mapping[str, str],
    max_dependency_bytes: int,
    storage_failures: Mapping[str, Sequence[str]],
) -> str:
    request_url = dependency.request_url
    if dependency.error_code == "html_dependency_fetch_failed":
        status_url = request_url
        seen: set[str] = set()
        while status_url in redirect_locations and status_url not in seen:
            seen.add(status_url)
            status_url = urljoin(status_url, redirect_locations[status_url])
        status = response_statuses.get(status_url)
        if status is not None and not 200 <= status < 300:
            request = httpx.Request("GET", status_url)
            response = httpx.Response(status, request=request)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                return str(exc)
    if dependency.error_code == "html_dependency_redirect_invalid":
        location = redirect_locations.get(request_url)
        if location:
            return f"remote redirect left the allowed provider boundary: {location}"
    if dependency.error_code == "html_dependency_too_large":
        return f"remote response exceeds {max_dependency_bytes} bytes"
    if dependency.error_code == "html_dependency_srcset_unsupported":
        return "srcset candidate selection is not supported by bundle schema v2"
    if dependency.error_code == "html_dependency_cache_write_failed":
        messages = storage_failures.get(request_url, ())
        if not messages:
            messages = next(iter(storage_failures.values()), ())
        return messages[0] if messages else dependency.error_message
    return dependency.error_message


def _origin(url: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _warning_to_document(value: Any) -> dict[str, Any]:
    return {
        "code": value.code,
        "message": value.message,
        "dependency_ordinal": value.dependency_ordinal,
        "element": value.element,
        "attribute": value.attribute,
        "authored_target": value.authored_target,
    }
