"""Safe acquisition and portable export of authored HTML image dependencies."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx
from ac_jobs import atomic_write_bytes

from .reference_cache import (
    CachedResourceRef,
    ReferenceCacheError,
    ReferenceMaterialCache,
)
from .sources import SourceArtifact, SourceFormat, SourceOrigin, SourceOriginKind


HTML_SOURCE_BUNDLE_SCHEMA = "arc.paper.html_source_bundle.v2"
HTML_SOURCE_EXPORT_SCHEMA = "arc.paper.html_source_export.v1"
ARXIV_HTML_DEPENDENCY_NAMESPACE = "arxiv-html-dependencies"
AR5IV_HTML_DEPENDENCY_NAMESPACE = "ar5iv-html-dependencies"
ARXIV_HTML_ACQUISITION_NAMESPACE = "arxiv-html-acquisition"
AR5IV_HTML_ACQUISITION_NAMESPACE = "ar5iv-html-acquisition"

DEFAULT_MAX_DEPENDENCY_COUNT = 256
DEFAULT_MAX_DEPENDENCY_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES = 200 * 1024 * 1024
DEFAULT_MAX_DEPENDENCY_REDIRECTS = 5

SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {
        "image/avif",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/svg+xml",
        "image/webp",
    }
)
_EXTENSION_MEDIA_TYPES = {
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_PRIMARY_FIELDS = {
    "source_format",
    "artifact_digest",
    "size",
    "media_type",
    "origin",
}
_ORIGIN_FIELDS = {"kind", "provider", "locator", "metadata"}
_DEPENDENCY_FIELDS = {
    "ordinal",
    "element",
    "attribute",
    "authored_target",
    "request_url",
    "resolved_url",
    "declared_media_type",
    "availability",
    "media_type",
    "artifact_digest",
    "size",
    "error_code",
    "error_message",
}
_WARNING_FIELDS = {
    "code",
    "message",
    "dependency_ordinal",
    "element",
    "attribute",
    "authored_target",
}
_POLICY_FIELDS = {
    "max_dependency_count",
    "max_dependency_bytes",
    "max_total_dependency_bytes",
    "max_redirects",
}
_BUNDLE_FIELDS = {
    "schema_version",
    "primary",
    "provider",
    "document_url",
    "base_url",
    "acquisition_policy",
    "dependencies",
    "warnings",
    "bundle_digest",
}


class HtmlSourceBundleError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class HtmlDependencyWarning:
    code: str
    message: str
    dependency_ordinal: int | None = None
    element: str = ""
    attribute: str = ""
    authored_target: str = ""

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, str)
            for item in (
                self.code,
                self.message,
                self.element,
                self.attribute,
                self.authored_target,
            )
        ):
            raise TypeError("HTML dependency warning string fields must be strings")
        if not self.code or not self.message:
            raise ValueError("HTML dependency warning requires code and message")
        if self.dependency_ordinal is not None and (
            isinstance(self.dependency_ordinal, bool)
            or not isinstance(self.dependency_ordinal, int)
            or self.dependency_ordinal < 0
        ):
            raise ValueError("dependency warning ordinal cannot be negative")

    def __str__(self) -> str:
        suffix = f": {self.authored_target}" if self.authored_target else ""
        return f"{self.code}: {self.message}{suffix}"


@dataclass(frozen=True)
class HtmlDependency:
    ordinal: int
    element: str
    attribute: str
    authored_target: str
    request_url: str = ""
    resolved_url: str = ""
    declared_media_type: str = ""
    availability: str = "unavailable"
    media_type: str = ""
    artifact_digest: str = ""
    size: int = 0
    error_code: str = ""
    error_message: str = ""

    def __post_init__(self) -> None:
        strings = (
            self.element,
            self.attribute,
            self.authored_target,
            self.request_url,
            self.resolved_url,
            self.declared_media_type,
            self.availability,
            self.media_type,
            self.artifact_digest,
            self.error_code,
            self.error_message,
        )
        if not all(isinstance(item, str) for item in strings):
            raise TypeError("HTML dependency string fields must be strings")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("HTML dependency ordinal cannot be negative")
        if self.element not in {"img", "object", "source"}:
            raise ValueError("HTML dependency element is unsupported")
        if self.attribute not in {"data", "src", "srcset"}:
            raise ValueError("HTML dependency attribute is unsupported")
        if self.availability not in {"available", "unavailable"}:
            raise ValueError("HTML dependency availability is invalid")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("HTML dependency size cannot be negative")
        if self.declared_media_type and (
            self.declared_media_type != self.declared_media_type.casefold()
            or ";" in self.declared_media_type
            or "/" not in self.declared_media_type
        ):
            raise ValueError("declared dependency media type must be normalized")
        if self.availability == "available":
            if (
                not self.request_url
                or not self.resolved_url
                or self.media_type not in SUPPORTED_IMAGE_MEDIA_TYPES
                or len(self.artifact_digest) != 64
                or any(char not in "0123456789abcdef" for char in self.artifact_digest)
                or self.error_code
                or self.error_message
            ):
                raise ValueError("available HTML dependency metadata is incomplete")
        elif (
            self.media_type
            or self.artifact_digest
            or self.size != 0
            or not self.error_code
            or not self.error_message
        ):
            raise ValueError("unavailable HTML dependency metadata is inconsistent")


@dataclass(frozen=True)
class HtmlSourceBundle:
    primary: SourceArtifact
    provider: str
    document_url: str
    base_url: str
    acquisition_policy: Mapping[str, int]
    dependencies: tuple[HtmlDependency, ...] = ()
    warnings: tuple[HtmlDependencyWarning, ...] = ()
    bundle_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.primary.source_format is not SourceFormat.HTML:
            raise ValueError("HTML source bundle primary must be HTML")
        if self.provider not in {"arxiv-html", "ar5iv"}:
            raise ValueError("HTML source bundle provider is unsupported")
        if not self.document_url or not self.base_url:
            raise ValueError("HTML source bundle provenance is incomplete")
        policy = _normalized_acquisition_policy(self.acquisition_policy)
        allowed_host = (
            "arxiv.org" if self.provider == "arxiv-html" else "ar5iv.labs.arxiv.org"
        )
        if self.document_url != normalize_safe_https_url(
            self.document_url, allowed_hosts=(allowed_host,)
        ) or self.base_url != normalize_safe_https_url(
            self.base_url, allowed_hosts=(allowed_host,)
        ):
            raise ValueError("HTML source bundle URLs are not normalized")
        if (
            self.primary.origin.provider != self.provider
            or self.primary.origin.locator != self.document_url
        ):
            raise ValueError("HTML source bundle primary provenance does not match")
        dependencies = tuple(self.dependencies)
        if [item.ordinal for item in dependencies] != list(range(len(dependencies))):
            raise ValueError("HTML dependency ordinals must be contiguous")
        warnings = tuple(self.warnings)
        by_ordinal = {
            item.dependency_ordinal: item
            for item in warnings
            if item.dependency_ordinal is not None
        }
        if len(by_ordinal) != sum(
            item.dependency_ordinal is not None for item in warnings
        ):
            raise ValueError("HTML source bundle has duplicate dependency warnings")
        for ordinal, warning in by_ordinal.items():
            if ordinal >= len(dependencies):
                raise ValueError("HTML dependency warning ordinal is out of range")
            dependency = dependencies[ordinal]
            if (
                dependency.availability != "unavailable"
                or warning.code != dependency.error_code
                or warning.element != dependency.element
                or warning.attribute != dependency.attribute
                or warning.authored_target != dependency.authored_target
            ):
                raise ValueError("HTML dependency warning does not match its record")
        if any(
            item.availability == "unavailable" and item.ordinal not in by_ordinal
            for item in dependencies
        ):
            raise ValueError("unavailable HTML dependency requires a warning")
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "acquisition_policy", MappingProxyType(policy))
        identity = _bundle_identity_document(self)
        object.__setattr__(
            self,
            "bundle_digest",
            hashlib.sha256(_json_bytes(identity)).hexdigest(),
        )


class HtmlDependencyAcquirer(Protocol):
    """Acquire authored dependencies for an ARC-validated HTML primary."""

    def __call__(
        self,
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
    ) -> "HtmlDependencyAcquisitionResult | HtmlSourceBundle": ...


@dataclass(frozen=True)
class HtmlDependencyAcquisitionResult:
    """ARC v2 projection plus the internal strict ACF sidecar."""

    bundle: HtmlSourceBundle
    sidecar: Mapping[str, Any] | None = None


def fetch_safe_response(
    client: httpx.Client,
    request_gate: Any,
    url: str,
    *,
    allowed_hosts: Sequence[str],
    timeout: float,
    max_redirects: int = DEFAULT_MAX_DEPENDENCY_REDIRECTS,
    redirect_error_code: str = "remote_url_invalid",
    maximum_bytes: int | None = None,
    size_error_code: str = "remote_response_too_large",
) -> httpx.Response:
    """Follow redirects only after validating each URL before its request."""

    current = normalize_safe_https_url(url, allowed_hosts=allowed_hosts)
    for redirect_count in range(max_redirects + 1):
        response = request_gate.request(
            lambda current=current: _get_bounded_response(
                client,
                current,
                timeout=timeout,
                maximum_bytes=maximum_bytes,
                size_error_code=size_error_code,
            )
        )
        normalize_safe_https_url(str(response.url), allowed_hosts=allowed_hosts)
        if response.status_code not in _REDIRECT_STATUSES:
            return response
        if redirect_count == max_redirects:
            raise HtmlSourceBundleError(
                "html_dependency_redirect_limit",
                f"remote response exceeded {max_redirects} redirects",
            )
        location = response.headers.get("location", "").strip()
        if not location:
            raise HtmlSourceBundleError(
                redirect_error_code,
                "remote redirect did not provide a Location header",
            )
        try:
            current = normalize_safe_https_url(
                urljoin(str(response.url), location),
                allowed_hosts=allowed_hosts,
            )
        except HtmlSourceBundleError as exc:
            raise HtmlSourceBundleError(
                redirect_error_code,
                f"remote redirect left the allowed provider boundary: {location}",
            ) from exc
    raise AssertionError("redirect loop did not terminate")


def normalize_safe_https_url(url: str, *, allowed_hosts: Sequence[str]) -> str:
    text = str(url or "").strip()
    if not text or any(ord(character) < 0x20 or ord(character) == 0x7F for character in text):
        raise HtmlSourceBundleError("remote_url_invalid", "remote URL contains invalid characters")
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise HtmlSourceBundleError("remote_url_invalid", "remote URL is malformed") from exc
    hosts = {str(item).strip().casefold() for item in allowed_hosts if str(item).strip()}
    hostname = (parsed.hostname or "").casefold()
    if (
        parsed.scheme.casefold() != "https"
        or hostname not in hosts
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        raise HtmlSourceBundleError(
            "remote_url_invalid",
            "remote URL must use credential-free HTTPS on an allowed provider host",
        )
    netloc = hostname if port is None else f"{hostname}:443"
    return urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))


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
    """Delegate generic dependency acquisition to AC Foundation."""

    from .html_acquisition import acquire_html_dependencies as acquire_with_acf

    return acquire_with_acf(
        payload,
        primary=primary,
        provider=provider,
        document_url=document_url,
        requested_url=requested_url,
        allowed_host=allowed_host,
        client=client,
        request_gate=request_gate,
        resource_cache=resource_cache,
        source_repository=source_repository,
        timeout=timeout,
        max_dependency_count=max_dependency_count,
        max_dependency_bytes=max_dependency_bytes,
        max_total_dependency_bytes=max_total_dependency_bytes,
        max_redirects=max_redirects,
        resolver=resolver,
        transport_factory=transport_factory,
    )


def _sidecar_is_valid(
    value: Any,
    *,
    request_key: str,
    primary: SourceArtifact,
    decoder: Callable[..., Any],
) -> bool:
    try:
        bundle = decoder(value, request_key=request_key)
    except (TypeError, ValueError):
        return False
    return bundle.primary == primary


def fetch_cached_html_bundle(
    *,
    cache: Any,
    resource_cache: ReferenceMaterialCache,
    source_namespace: str,
    dependency_namespace: str,
    request_key: str,
    source_origin: SourceOrigin,
    provider: str,
    allowed_host: str,
    client: httpx.Client,
    request_gate: Any,
    timeout: float,
    refresh: bool,
    fetch_main: Callable[[], tuple[bytes, str]],
    build_origin: Callable[[bytes, str], SourceOrigin] | None = None,
    dependency_acquirer: HtmlDependencyAcquirer = acquire_html_dependencies,
    sidecar_namespace: str | None = None,
    requested_url: str | None = None,
    dependency_resolver: Callable[[str], Sequence[str]] | None = None,
    dependency_transport_factory: Callable[[httpx.Client, Any], Any] | None = None,
    max_dependency_count: int = DEFAULT_MAX_DEPENDENCY_COUNT,
    max_dependency_bytes: int = DEFAULT_MAX_DEPENDENCY_BYTES,
    max_total_dependency_bytes: int = DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES,
    max_redirects: int = DEFAULT_MAX_DEPENDENCY_REDIRECTS,
) -> HtmlSourceBundle:
    _validate_limits(
        max_dependency_count,
        max_dependency_bytes,
        max_total_dependency_bytes,
        max_redirects,
    )
    acquisition_policy = _acquisition_policy_document(
        max_dependency_count=max_dependency_count,
        max_dependency_bytes=max_dependency_bytes,
        max_total_dependency_bytes=max_total_dependency_bytes,
        max_redirects=max_redirects,
    )

    def current_source() -> SourceArtifact | None:
        return cache.get_source(
            source_namespace,
            request_key,
            source_format=SourceFormat.HTML,
            media_type="text/html",
            origin=source_origin,
        )

    def payload_is_valid(value: Any) -> bool:
        try:
            primary = current_source()
            if primary is None:
                return False
            bundle = html_source_bundle_from_document(value)
            if bundle.primary.content_identity != primary.content_identity:
                return False
            if dict(bundle.acquisition_policy) != acquisition_policy:
                return False
            return _bundle_resources_are_valid(bundle, resource_cache)
        except (ReferenceCacheError, RuntimeError, TypeError, ValueError):
            return False

    def acquire() -> dict[str, Any]:
        payload, document_url = fetch_main()
        resolved_origin = (
            build_origin(payload, document_url)
            if build_origin is not None
            else SourceOrigin(
                SourceOriginKind.REMOTE_PROVIDER,
                provider=provider,
                locator=document_url,
                metadata=source_origin.metadata,
            )
        )
        try:
            current_source()
        except RuntimeError as exc:
            if exc.code != "remote_cache_source_corrupt":
                raise
            cache.remove("source", source_namespace, request_key)
        primary = cache.fetch_source(
            source_namespace,
            request_key,
            source_format=SourceFormat.HTML,
            media_type="text/html",
            origin=resolved_origin,
            refresh=True,
            fetch=lambda: payload,
        )
        acquired = dependency_acquirer(
            payload,
            primary=primary,
            provider=provider,
            document_url=document_url,
            requested_url=requested_url,
            allowed_host=allowed_host,
            client=client,
            request_gate=request_gate,
            resource_cache=resource_cache,
            source_repository=cache.source_repository,
            timeout=timeout,
            max_dependency_count=max_dependency_count,
            max_dependency_bytes=max_dependency_bytes,
            max_total_dependency_bytes=max_total_dependency_bytes,
            max_redirects=max_redirects,
            resolver=dependency_resolver,
            transport_factory=dependency_transport_factory,
        )
        if isinstance(acquired, HtmlDependencyAcquisitionResult):
            bundle = acquired.bundle
            if acquired.sidecar is not None and sidecar_namespace is not None:
                from .html_acquisition import html_acquisition_sidecar_from_document

                cache.fetch_json(
                    sidecar_namespace,
                    request_key,
                    fetch=lambda: {
                        **dict(acquired.sidecar or {}),
                        "request_key": request_key,
                    },
                    refresh=True,
                    payload_validator=lambda value: _sidecar_is_valid(
                        value,
                        request_key=request_key,
                        primary=primary,
                        decoder=html_acquisition_sidecar_from_document,
                    ),
                )
        else:
            bundle = acquired
        return html_source_bundle_to_document(bundle)

    document = cache.fetch_json(
        dependency_namespace,
        request_key,
        fetch=acquire,
        refresh=refresh,
        payload_validator=payload_is_valid,
    )
    return html_source_bundle_from_document(document)


def html_source_bundle_to_document(bundle: HtmlSourceBundle) -> dict[str, Any]:
    return {
        **_bundle_identity_document(bundle),
        "bundle_digest": bundle.bundle_digest,
    }


def html_source_bundle_from_document(value: Any) -> HtmlSourceBundle:
    if not isinstance(value, Mapping) or set(value) != _BUNDLE_FIELDS:
        raise ValueError("HTML source bundle has invalid fields")
    if value.get("schema_version") != HTML_SOURCE_BUNDLE_SCHEMA:
        raise ValueError("HTML source bundle has an unsupported schema")
    dependencies = value.get("dependencies")
    warnings = value.get("warnings")
    if not isinstance(dependencies, list) or not isinstance(warnings, list):
        raise ValueError("HTML source bundle collections are invalid")
    bundle = HtmlSourceBundle(
        primary=_source_artifact_from_document(value.get("primary")),
        provider=_required_string(value, "provider"),
        document_url=_required_string(value, "document_url"),
        base_url=_required_string(value, "base_url"),
        acquisition_policy=_required_policy(value, "acquisition_policy"),
        dependencies=tuple(_dependency_from_document(item) for item in dependencies),
        warnings=tuple(_warning_from_document(item) for item in warnings),
    )
    if value.get("bundle_digest") != bundle.bundle_digest:
        raise ValueError("HTML source bundle digest does not match its content")
    return bundle


def materialize_html_source_bundle(
    bundle: HtmlSourceBundle,
    *,
    source_repository: Any,
    resource_cache: ReferenceMaterialCache,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve(strict=False)
    _require_available_output(output)
    source_payload = source_repository.read_bytes(bundle.primary)
    resources_by_path: dict[str, tuple[HtmlDependency, bytes]] = {}
    warnings = [_warning_to_document(item) for item in bundle.warnings]

    for dependency in bundle.dependencies:
        if dependency.availability != "available":
            continue
        relative = _safe_local_target(dependency.authored_target)
        if relative is None:
            warnings.append(
                _warning_to_document(
                    HtmlDependencyWarning(
                        "html_dependency_target_not_materializable",
                        "authored target is not a safe source-relative path",
                        dependency_ordinal=dependency.ordinal,
                        element=dependency.element,
                        attribute=dependency.attribute,
                        authored_target=dependency.authored_target,
                    )
                )
            )
            continue
        reference = _resource_reference(dependency)
        payload = resource_cache.read_resource(reference)
        key = relative.as_posix()
        previous = resources_by_path.get(key)
        if previous is not None:
            if previous[0].artifact_digest != dependency.artifact_digest:
                raise HtmlSourceBundleError(
                    "html_bundle_path_collision",
                    f"authored targets collide with different bytes: {key}",
                )
            continue
        resources_by_path[key] = (dependency, payload)

    resource_documents = [
        {
            "ordinal": dependency.ordinal,
            "authored_target": dependency.authored_target,
            "path": path,
            "artifact_digest": dependency.artifact_digest,
            "media_type": dependency.media_type,
            "size": dependency.size,
        }
        for path, (dependency, _payload) in sorted(resources_by_path.items())
    ]
    manifest = {
        "schema_version": HTML_SOURCE_EXPORT_SCHEMA,
        "bundle": html_source_bundle_to_document(bundle),
        "source": "source.html",
        "resources": resource_documents,
        "warnings": warnings,
    }
    _publish_source_bundle(
        output,
        source_payload=source_payload,
        manifest_payload=_json_bytes(manifest) + b"\n",
        resources=resources_by_path,
    )
    return {
        "source": str(output / "source.html"),
        "manifest": str(output / "manifest.json"),
        "bundle_digest": bundle.bundle_digest,
        "resources": [
            {**item, "path": str(output / item["path"])}
            for item in resource_documents
        ],
        "warnings": warnings,
    }


def bundle_resource_identities(value: Any) -> tuple[CachedResourceRef, ...]:
    bundle = html_source_bundle_from_document(value)
    return tuple(
        _resource_reference(item)
        for item in bundle.dependencies
        if item.availability == "available"
    )


def _validate_limits(count: int, single: int, total: int, redirects: int) -> None:
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (count, single, total, redirects)):
        raise ValueError("HTML dependency limits must be integers")
    if count <= 0 or single <= 0 or total <= 0 or redirects < 0:
        raise ValueError("HTML dependency limits must be positive")
    if total < single:
        raise ValueError("total dependency byte limit cannot be smaller than one resource")


def _acquisition_policy_document(
    *,
    max_dependency_count: int,
    max_dependency_bytes: int,
    max_total_dependency_bytes: int,
    max_redirects: int,
) -> dict[str, int]:
    return {
        "max_dependency_count": max_dependency_count,
        "max_dependency_bytes": max_dependency_bytes,
        "max_total_dependency_bytes": max_total_dependency_bytes,
        "max_redirects": max_redirects,
    }


def _normalized_acquisition_policy(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _POLICY_FIELDS:
        raise ValueError("HTML source bundle acquisition policy is invalid")
    policy = {
        key: value[key]
        for key in (
            "max_dependency_count",
            "max_dependency_bytes",
            "max_total_dependency_bytes",
            "max_redirects",
        )
    }
    _validate_limits(
        policy["max_dependency_count"],
        policy["max_dependency_bytes"],
        policy["max_total_dependency_bytes"],
        policy["max_redirects"],
    )
    return policy


def _bundle_resources_are_valid(
    bundle: HtmlSourceBundle,
    resource_cache: ReferenceMaterialCache,
) -> bool:
    try:
        for dependency in bundle.dependencies:
            if dependency.availability == "available":
                resource_cache.read_resource(_resource_reference(dependency))
    except ReferenceCacheError:
        return False
    return True


def _resource_reference(dependency: HtmlDependency) -> CachedResourceRef:
    return CachedResourceRef(
        resource_sha256=dependency.artifact_digest,
        resource_size=dependency.size,
        media_type=dependency.media_type,
        source_locator=dependency.resolved_url,
        filename=PurePosixPath(unquote(urlsplit(dependency.resolved_url).path)).name,
    )


def _bundle_identity_document(bundle: HtmlSourceBundle) -> dict[str, Any]:
    return {
        "schema_version": HTML_SOURCE_BUNDLE_SCHEMA,
        "primary": _source_artifact_to_document(bundle.primary),
        "provider": bundle.provider,
        "document_url": bundle.document_url,
        "base_url": bundle.base_url,
        "acquisition_policy": dict(bundle.acquisition_policy),
        "dependencies": [_dependency_to_document(item) for item in bundle.dependencies],
        "warnings": [_warning_to_document(item) for item in bundle.warnings],
    }


def _source_artifact_to_document(value: SourceArtifact) -> dict[str, Any]:
    return {
        "source_format": value.source_format.value,
        "artifact_digest": value.artifact_digest,
        "size": value.size,
        "media_type": value.media_type,
        "origin": {
            "kind": value.origin.kind.value,
            "provider": value.origin.provider,
            "locator": value.origin.locator,
            "metadata": dict(value.origin.metadata),
        },
    }


def _source_artifact_from_document(value: Any) -> SourceArtifact:
    if not isinstance(value, Mapping) or set(value) != _PRIMARY_FIELDS:
        raise ValueError("HTML source bundle primary is invalid")
    origin = value.get("origin")
    if not isinstance(origin, Mapping) or set(origin) != _ORIGIN_FIELDS:
        raise ValueError("HTML source bundle origin is invalid")
    metadata = origin.get("metadata")
    if not isinstance(metadata, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in metadata.items()
    ):
        raise ValueError("HTML source bundle origin metadata is invalid")
    return SourceArtifact(
        source_format=value.get("source_format"),
        artifact_digest=_required_string(value, "artifact_digest"),
        size=_required_int(value, "size"),
        media_type=_required_string(value, "media_type"),
        origin=SourceOrigin(
            kind=origin.get("kind"),
            provider=str(origin.get("provider") or ""),
            locator=str(origin.get("locator") or ""),
            metadata=dict(metadata),
        ),
    )


def _dependency_to_document(value: HtmlDependency) -> dict[str, Any]:
    return {key: getattr(value, key) for key in _DEPENDENCY_FIELDS}


def _dependency_from_document(value: Any) -> HtmlDependency:
    if not isinstance(value, Mapping) or set(value) != _DEPENDENCY_FIELDS:
        raise ValueError("HTML dependency record has invalid fields")
    return HtmlDependency(**dict(value))


def _warning_to_document(value: HtmlDependencyWarning) -> dict[str, Any]:
    return {key: getattr(value, key) for key in _WARNING_FIELDS}


def _warning_from_document(value: Any) -> HtmlDependencyWarning:
    if not isinstance(value, Mapping) or set(value) != _WARNING_FIELDS:
        raise ValueError("HTML dependency warning has invalid fields")
    return HtmlDependencyWarning(**dict(value))


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"HTML source bundle {key} must be a nonempty string")
    return item


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise ValueError(f"HTML source bundle {key} must be a nonnegative integer")
    return item


def _required_policy(value: Mapping[str, Any], key: str) -> dict[str, int]:
    return _normalized_acquisition_policy(value.get(key))


def _safe_local_target(target: str) -> PurePosixPath | None:
    if "\\" in target:
        return None
    try:
        parsed = urlsplit(target)
    except ValueError:
        return None
    if parsed.scheme or parsed.netloc or parsed.fragment or parsed.path.startswith("/"):
        return None
    relative = PurePosixPath(parsed.path)
    if not parsed.path or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return relative


def _require_available_output(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise HtmlSourceBundleError(
            "html_bundle_output_exists",
            f"output path exists and is not a directory: {output}",
        )
    try:
        nonempty = next(output.iterdir(), None) is not None
    except OSError as exc:
        raise HtmlSourceBundleError(
            "html_bundle_output_unreadable",
            f"output directory cannot be inspected: {output}",
        ) from exc
    if nonempty:
        raise HtmlSourceBundleError(
            "html_bundle_output_not_empty",
            f"output directory must be absent or empty: {output}",
        )


def _publish_source_bundle(
    output: Path,
    *,
    source_payload: bytes,
    manifest_payload: bytes,
    resources: Mapping[str, tuple[HtmlDependency, bytes]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent))
    replaced_empty = False
    try:
        atomic_write_bytes(staging / "source.html", source_payload)
        for path, (_dependency, payload) in resources.items():
            atomic_write_bytes(staging / path, payload)
        atomic_write_bytes(staging / "manifest.json", manifest_payload)
        _require_available_output(output)
        if output.exists():
            output.rmdir()
            replaced_empty = True
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if replaced_empty and not output.exists():
            output.mkdir()
        raise


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _response_media_type(response: httpx.Response) -> str:
    return response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()


def _get_bounded_response(
    client: httpx.Client,
    url: str,
    *,
    timeout: float,
    maximum_bytes: int | None,
    size_error_code: str,
) -> httpx.Response:
    with client.stream(
        "GET",
        url,
        timeout=timeout,
        follow_redirects=False,
    ) as response:
        if not 200 <= response.status_code < 300:
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=b"",
                request=response.request,
                extensions=response.extensions,
            )
        content_length = response.headers.get("content-length")
        if maximum_bytes is not None and content_length is not None:
            try:
                if int(content_length) > maximum_bytes:
                    raise HtmlSourceBundleError(
                        size_error_code,
                        f"remote response exceeds {maximum_bytes} bytes",
                    )
            except ValueError as exc:
                raise HtmlSourceBundleError(
                    "remote_content_length_invalid",
                    "remote response has an invalid Content-Length",
                ) from exc
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if maximum_bytes is not None and size > maximum_bytes:
                raise HtmlSourceBundleError(
                    size_error_code,
                    f"remote response exceeds {maximum_bytes} bytes",
                )
            chunks.append(chunk)
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=b"".join(chunks),
            request=response.request,
            extensions=response.extensions,
        )


def _validate_response_size(
    response: httpx.Response,
    maximum: int,
    code: str,
) -> None:
    content_length = response.headers.get("content-length")
    try:
        if content_length is not None and int(content_length) > maximum:
            raise HtmlSourceBundleError(code, f"remote response exceeds {maximum} bytes")
    except ValueError as exc:
        raise HtmlSourceBundleError(
            "remote_content_length_invalid",
            "remote response has an invalid Content-Length",
        ) from exc
    if len(response.content) > maximum:
        raise HtmlSourceBundleError(code, f"remote response exceeds {maximum} bytes")


__all__ = [
    "AR5IV_HTML_DEPENDENCY_NAMESPACE",
    "ARXIV_HTML_DEPENDENCY_NAMESPACE",
    "DEFAULT_MAX_DEPENDENCY_BYTES",
    "DEFAULT_MAX_DEPENDENCY_COUNT",
    "DEFAULT_MAX_DEPENDENCY_REDIRECTS",
    "DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES",
    "HTML_SOURCE_BUNDLE_SCHEMA",
    "HTML_SOURCE_EXPORT_SCHEMA",
    "HtmlDependency",
    "HtmlDependencyWarning",
    "HtmlSourceBundle",
    "HtmlSourceBundleError",
    "SUPPORTED_IMAGE_MEDIA_TYPES",
    "acquire_html_dependencies",
    "bundle_resource_identities",
    "fetch_cached_html_bundle",
    "fetch_safe_response",
    "html_source_bundle_from_document",
    "html_source_bundle_to_document",
    "materialize_html_source_bundle",
    "normalize_safe_https_url",
]
