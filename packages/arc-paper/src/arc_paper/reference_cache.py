"""Host-independent identities and verified cached reference materials."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import SplitResult, urlsplit, urlunsplit

from ac_jobs import atomic_write_bytes, file_lease, file_matches_sha256

from ._cache_root import resolve_cache_root
from .ids import arxiv_path_id, doi_value


REFERENCE_IDENTITY_SCHEMA = "arc.paper.reference_identity.v1"
CACHED_RESOURCE_REF_SCHEMA = "arc.paper.cached_resource_ref.v1"
CACHED_REFERENCE_MATERIAL_SCHEMA = "arc.paper.cached_reference_material.v1"
REFERENCE_RESOURCE_OBJECT_SCHEMA = "arc.paper.reference_resource_object.v1"
REFERENCE_MATERIAL_CACHE_SCHEMA = "arc.paper.reference_material_cache.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RESOURCE_OBJECT_FIELDS = {
    "schema_version",
    "resource_sha256",
    "resource_size",
    "media_types",
}
_MATERIAL_FIELDS = {
    "schema_version",
    "record_id",
    "identity",
    "resources",
    "readable_resource",
}


class ReferenceCacheError(RuntimeError):
    """A stable cache-only reference failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ReferenceIdentity:
    """Exact, provider-neutral aliases for one reference.

    Titles are display values. Lookup uses :func:`normalize_reference_title`
    and never performs fuzzy matching.
    """

    arxiv_id: str = ""
    dois: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    title: str = ""
    inspire_recid: str = ""

    def __post_init__(self) -> None:
        arxiv_id = arxiv_path_id(str(self.arxiv_id or ""))
        raw_arxiv = str(self.arxiv_id or "").strip()
        if raw_arxiv and not arxiv_id:
            raise ValueError("arxiv_id is invalid")
        dois = _dedupe(
            normalized
            for value in self.dois
            if (normalized := doi_value(str(value)))
        )
        if len(dois) != len(tuple(self.dois)):
            invalid = [
                str(value)
                for value in self.dois
                if not doi_value(str(value))
            ]
            if invalid:
                raise ValueError("dois contains an invalid DOI")
        urls = _dedupe(normalize_reference_url(str(value)) for value in self.urls)
        title = " ".join(str(self.title or "").split())
        inspire_recid = str(self.inspire_recid or "").strip()
        if inspire_recid and not inspire_recid.isdigit():
            raise ValueError("inspire_recid must contain digits only")
        if not any((arxiv_id, dois, urls, title, inspire_recid)):
            raise ValueError("reference identity requires at least one exact alias")
        object.__setattr__(self, "arxiv_id", arxiv_id)
        object.__setattr__(self, "dois", dois)
        object.__setattr__(self, "urls", urls)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "inspire_recid", inspire_recid)

    @property
    def doi(self) -> str:
        """Compatibility projection for consumers that accept one DOI."""

        return self.dois[0] if self.dois else ""

    @property
    def canonical_key(self) -> str:
        if self.inspire_recid:
            return f"inspire:{self.inspire_recid}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.casefold()}"
        if self.dois:
            return f"doi:{self.dois[0]}"
        if self.urls:
            return f"url:{self.urls[0]}"
        return f"title:{normalize_reference_title(self.title)}"


@dataclass(frozen=True)
class CachedResourceRef:
    """Logical handle for arbitrary verified cached bytes."""

    resource_sha256: str
    resource_size: int
    media_type: str
    source_locator: str = ""
    filename: str = ""

    def __post_init__(self) -> None:
        digest = str(self.resource_sha256).casefold()
        media_type = _normalize_media_type(self.media_type)
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("resource_sha256 must be a SHA-256 digest")
        if (
            not isinstance(self.resource_size, int)
            or isinstance(self.resource_size, bool)
            or self.resource_size < 0
        ):
            raise ValueError("resource_size cannot be negative")
        object.__setattr__(self, "resource_sha256", digest)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "source_locator", str(self.source_locator or ""))
        object.__setattr__(self, "filename", Path(str(self.filename or "")).name)

    @property
    def content_identity(self) -> tuple[str, str, int]:
        return (self.media_type, self.resource_sha256, self.resource_size)


@dataclass(frozen=True)
class CachedReferenceMaterial:
    """One cached reference identity and all admitted representations."""

    identity: ReferenceIdentity
    resources: tuple[CachedResourceRef, ...]
    readable_resource: CachedResourceRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ReferenceIdentity):
            raise TypeError("identity must be a ReferenceIdentity")
        resources = tuple(self.resources)
        if not resources:
            raise ValueError("cached reference material requires a resource")
        identities = [item.content_identity for item in resources]
        if len(identities) != len(set(identities)):
            raise ValueError("cached reference material contains duplicate resources")
        readable = self.readable_resource
        if readable is not None and readable.content_identity not in set(identities):
            raise ValueError("readable_resource must be present in resources")
        object.__setattr__(self, "resources", resources)


class ReferenceMaterialCache:
    """Atomic content storage plus exact cache-only reference lookup."""

    def __init__(self, root: str | Path | None = None):
        self.root = resolve_cache_root(root)

    def store_resource(
        self,
        payload: bytes,
        *,
        media_type: str,
        source_locator: str = "",
        filename: str = "",
    ) -> CachedResourceRef:
        if not isinstance(payload, bytes):
            raise TypeError("resource payload must be bytes")
        digest = hashlib.sha256(payload).hexdigest()
        normalized_media = _normalize_media_type(media_type)
        object_dir = self._resource_dir(digest)
        manifest_path = object_dir / "manifest.json"
        payload_path = object_dir / "resource"
        with file_lease(self._resource_lock(digest), blocking=True):
            if manifest_path.exists():
                try:
                    size, media_types = self._read_resource_object(digest)
                except ReferenceCacheError as exc:
                    if exc.code not in {
                        "reference_resource_corrupt",
                        "reference_resource_manifest_invalid",
                    }:
                        raise
                    if object_dir.is_symlink():
                        raise
                    shutil.rmtree(object_dir)
                    size, media_types = -1, frozenset()
                if size != len(payload):
                    if size >= 0:
                        raise ReferenceCacheError(
                            "reference_resource_mismatch",
                            "cached resource size conflicts with admitted bytes",
                        )
            if not manifest_path.exists():
                object_dir.mkdir(parents=True, exist_ok=True)
                if not file_matches_sha256(payload_path, digest, len(payload)):
                    atomic_write_bytes(payload_path, payload)
                atomic_write_bytes(
                    manifest_path,
                    _json_bytes(
                        {
                            "schema_version": REFERENCE_RESOURCE_OBJECT_SCHEMA,
                            "resource_sha256": digest,
                            "resource_size": len(payload),
                            "media_types": [normalized_media],
                        }
                    ),
                )
            elif normalized_media not in media_types:
                atomic_write_bytes(
                    manifest_path,
                    _json_bytes(
                        {
                            "schema_version": REFERENCE_RESOURCE_OBJECT_SCHEMA,
                            "resource_sha256": digest,
                            "resource_size": len(payload),
                            "media_types": sorted(
                                {*media_types, normalized_media}
                            ),
                        }
                    ),
                )
        return CachedResourceRef(
            resource_sha256=digest,
            resource_size=len(payload),
            media_type=normalized_media,
            source_locator=source_locator,
            filename=filename,
        )

    def store_material(
        self,
        identity: ReferenceIdentity,
        resources: tuple[CachedResourceRef, ...],
        *,
        readable_resource: CachedResourceRef | None = None,
    ) -> CachedReferenceMaterial:
        incoming = CachedReferenceMaterial(identity, resources, readable_resource)
        with file_lease(self._records_lock(), blocking=True):
            existing = self._all_materials_unlocked()
            compatible = [
                item for item in existing if _identities_compatible(item.identity, identity)
            ]
            if len(compatible) > 1:
                raise ReferenceCacheError(
                    "reference_ambiguous",
                    "multiple compatible cached references require explicit resolution",
                )
            if compatible:
                current = compatible[0]
                merged = CachedReferenceMaterial(
                    identity=_merge_identities(current.identity, identity),
                    resources=_merge_resources(current.resources, resources),
                    readable_resource=readable_resource or current.readable_resource,
                )
                record_id = self._record_id(current.identity)
                old_path = self._record_path(record_id)
            else:
                merged = incoming
                record_id = self._record_id(identity)
                old_path = self._record_path(record_id)
            self._verify_material_resources(merged)
            atomic_write_bytes(
                old_path,
                _json_bytes(_material_to_document(merged, record_id=record_id)),
            )
            return merged

    def lookup(
        self,
        *,
        doi: str | None = None,
        arxiv_id: str | None = None,
        inspire_recid: str | None = None,
        url: str | None = None,
        title: str | None = None,
    ) -> CachedReferenceMaterial | None:
        supplied = [
            value is not None
            for value in (doi, arxiv_id, inspire_recid, url, title)
        ]
        if sum(supplied) != 1:
            raise ValueError(
                "lookup requires exactly one DOI, arXiv ID, INSPIRE record, URL, or title"
            )
        if doi is not None:
            normalized_doi = doi_value(doi)
            if not normalized_doi:
                raise ValueError("doi is invalid")
            predicate = lambda item: normalized_doi in item.identity.dois
        elif arxiv_id is not None:
            normalized_arxiv = arxiv_path_id(arxiv_id)
            if not normalized_arxiv:
                raise ValueError("arxiv_id is invalid")
            predicate = lambda item: item.identity.arxiv_id == normalized_arxiv
        elif inspire_recid is not None:
            normalized_recid = str(inspire_recid).strip()
            if not normalized_recid.isdigit():
                raise ValueError("inspire_recid is invalid")
            predicate = lambda item: item.identity.inspire_recid == normalized_recid
        elif url is not None:
            normalized_url = normalize_reference_url(url)
            predicate = lambda item: normalized_url in item.identity.urls
        else:
            normalized_title = normalize_reference_title(title or "")
            if not normalized_title:
                raise ValueError("title is empty after normalization")
            predicate = (
                lambda item: normalize_reference_title(item.identity.title)
                == normalized_title
            )
        with file_lease(self._records_lock(), blocking=True):
            matches = [item for item in self._all_materials_unlocked() if predicate(item)]
        if not matches:
            return None
        if len(matches) > 1:
            raise ReferenceCacheError(
                "reference_ambiguous",
                "multiple cached references match the exact lookup",
            )
        return matches[0]

    def read_resource(self, reference: CachedResourceRef) -> bytes:
        if not isinstance(reference, CachedResourceRef):
            raise TypeError("reference must be a CachedResourceRef")
        with file_lease(
            self._resource_lock(reference.resource_sha256), blocking=True
        ):
            size, media_types = self._read_resource_object(
                reference.resource_sha256
            )
            if size != reference.resource_size:
                raise ReferenceCacheError(
                    "reference_resource_mismatch",
                    "cached resource reference does not match stored bytes",
                )
            if reference.media_type not in media_types:
                raise ReferenceCacheError(
                    "reference_resource_mismatch",
                    "cached resource media type was not admitted for these bytes",
                )
            path = self._resource_dir(reference.resource_sha256) / "resource"
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise ReferenceCacheError(
                    "reference_resource_corrupt",
                    "cached resource bytes are unreadable",
                ) from exc
            if (
                len(payload) != reference.resource_size
                or hashlib.sha256(payload).hexdigest() != reference.resource_sha256
            ):
                raise ReferenceCacheError(
                    "reference_resource_corrupt",
                    "cached resource bytes do not match their digest",
                )
            return payload

    def _all_materials_unlocked(self) -> list[CachedReferenceMaterial]:
        records_root = self._base() / "records"
        if not records_root.is_dir():
            return []
        materials = []
        for path in sorted(records_root.glob("*/*/manifest.json")):
            materials.append(self._read_material(path))
        return materials

    def _read_material(self, path: Path) -> CachedReferenceMaterial:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReferenceCacheError(
                "reference_manifest_invalid",
                "reference material manifest is unreadable or malformed",
            ) from exc
        if not isinstance(value, dict) or set(value) != _MATERIAL_FIELDS:
            raise ReferenceCacheError(
                "reference_manifest_invalid",
                "reference material manifest has an invalid schema",
            )
        record_id = value.get("record_id")
        if (
            value.get("schema_version") != REFERENCE_MATERIAL_CACHE_SCHEMA
            or not isinstance(record_id, str)
            or _SHA256_RE.fullmatch(record_id) is None
            or path != self._record_path(record_id)
        ):
            raise ReferenceCacheError(
                "reference_manifest_invalid",
                "reference material manifest identity does not match its key",
            )
        try:
            material = cached_reference_material_from_document(value)
        except (TypeError, ValueError) as exc:
            raise ReferenceCacheError(
                "reference_manifest_invalid",
                f"reference material manifest is invalid: {exc}",
            ) from exc
        self._verify_material_resources(material)
        return material

    def _verify_material_resources(self, material: CachedReferenceMaterial) -> None:
        for resource in material.resources:
            size, media_types = self._read_resource_object(
                resource.resource_sha256
            )
            if (
                size != resource.resource_size
                or resource.media_type not in media_types
            ):
                raise ReferenceCacheError(
                    "reference_resource_mismatch",
                    "reference material points to mismatched resource metadata",
                )

    def _read_resource_object(self, digest: str) -> tuple[int, frozenset[str]]:
        object_dir = self._resource_dir(digest)
        try:
            value = json.loads(
                (object_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise ReferenceCacheError(
                "reference_resource_not_found",
                f"cached resource is not present: {digest}",
            ) from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReferenceCacheError(
                "reference_resource_manifest_invalid",
                "cached resource manifest is unreadable or malformed",
            ) from exc
        if (
            not isinstance(value, dict)
            or set(value) != _RESOURCE_OBJECT_FIELDS
            or value.get("schema_version") != REFERENCE_RESOURCE_OBJECT_SCHEMA
            or value.get("resource_sha256") != digest
            or not isinstance(value.get("resource_size"), int)
            or isinstance(value.get("resource_size"), bool)
            or value["resource_size"] < 0
            or not isinstance(value.get("media_types"), list)
            or not value["media_types"]
            or not _valid_media_types(value["media_types"])
            or len(value["media_types"]) != len(set(value["media_types"]))
        ):
            raise ReferenceCacheError(
                "reference_resource_manifest_invalid",
                "cached resource manifest has an invalid schema",
            )
        if not file_matches_sha256(
            object_dir / "resource", digest, value["resource_size"]
        ):
            raise ReferenceCacheError(
                "reference_resource_corrupt",
                "cached resource bytes do not match their manifest",
            )
        return value["resource_size"], frozenset(value["media_types"])

    def _base(self) -> Path:
        return self.root / "reference-material-cache" / "v1"

    def _resource_dir(self, digest: str) -> Path:
        return self._base() / "resources" / "sha256" / digest[:2] / digest

    def _resource_lock(self, digest: str) -> Path:
        return self._base() / "locks" / "resources" / f"{digest}.lock"

    def _records_lock(self) -> Path:
        return self._base() / "locks" / "records.lock"

    def _record_id(self, identity: ReferenceIdentity) -> str:
        return hashlib.sha256(
            _json_bytes(reference_identity_to_document(identity))
        ).hexdigest()

    def _record_path(self, record_id: str) -> Path:
        return self._base() / "records" / record_id[:2] / record_id / "manifest.json"


def normalize_reference_title(value: str) -> str:
    folded = unicodedata.normalize("NFKC", str(value or "")).casefold()
    separated = "".join(
        " " if unicodedata.category(character).startswith(("P", "Z")) else character
        for character in folded
    )
    return " ".join(separated.split())


def normalize_reference_url(value: str) -> str:
    text = str(value or "").strip()
    parsed = urlsplit(text)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("reference URL must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("reference URL must not contain credentials")
    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname.casefold()
    port = parsed.port
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"
    normalized = SplitResult(
        scheme,
        netloc,
        parsed.path or "/",
        parsed.query,
        "",
    )
    return urlunsplit(normalized)


def reference_identity_to_document(value: ReferenceIdentity) -> dict[str, Any]:
    return {
        "arxiv_id": value.arxiv_id,
        "dois": list(value.dois),
        "urls": list(value.urls),
        "title": value.title,
        "inspire_recid": value.inspire_recid,
    }


def reference_identity_from_document(value: Mapping[str, Any]) -> ReferenceIdentity:
    if not isinstance(value, Mapping) or set(value) != {
        "arxiv_id",
        "dois",
        "urls",
        "title",
        "inspire_recid",
    }:
        raise ValueError("reference identity has invalid fields")
    dois = value["dois"]
    urls = value["urls"]
    if not isinstance(dois, list) or not all(isinstance(item, str) for item in dois):
        raise ValueError("reference identity dois must be strings")
    if not isinstance(urls, list) or not all(isinstance(item, str) for item in urls):
        raise ValueError("reference identity urls must be strings")
    return ReferenceIdentity(
        arxiv_id=value["arxiv_id"],
        dois=tuple(dois),
        urls=tuple(urls),
        title=value["title"],
        inspire_recid=value["inspire_recid"],
    )


def cached_resource_ref_to_document(value: CachedResourceRef) -> dict[str, Any]:
    return {
        "resource_sha256": value.resource_sha256,
        "resource_size": value.resource_size,
        "media_type": value.media_type,
        "source_locator": value.source_locator,
        "filename": value.filename,
    }


def cached_resource_ref_from_document(value: Mapping[str, Any]) -> CachedResourceRef:
    if not isinstance(value, Mapping) or set(value) != {
        "resource_sha256",
        "resource_size",
        "media_type",
        "source_locator",
        "filename",
    }:
        raise ValueError("cached resource reference has invalid fields")
    return CachedResourceRef(**dict(value))


def cached_reference_material_to_document(
    value: CachedReferenceMaterial,
) -> dict[str, Any]:
    return {
        "identity": reference_identity_to_document(value.identity),
        "resources": [
            cached_resource_ref_to_document(item) for item in value.resources
        ],
        "readable_resource": (
            cached_resource_ref_to_document(value.readable_resource)
            if value.readable_resource is not None
            else None
        ),
    }


def cached_reference_material_from_document(
    value: Mapping[str, Any],
) -> CachedReferenceMaterial:
    if not isinstance(value, Mapping):
        raise ValueError("cached reference material must be an object")
    allowed = {"identity", "resources", "readable_resource"}
    if "schema_version" in value or "record_id" in value:
        allowed |= {"schema_version", "record_id"}
        if (
            value.get("schema_version") != REFERENCE_MATERIAL_CACHE_SCHEMA
            or not isinstance(value.get("record_id"), str)
            or _SHA256_RE.fullmatch(value["record_id"]) is None
        ):
            raise ValueError("cached reference material cache metadata is invalid")
    if set(value) != allowed:
        raise ValueError("cached reference material has invalid fields")
    resources = value["resources"]
    readable = value["readable_resource"]
    if not isinstance(resources, list):
        raise ValueError("cached reference resources must be an array")
    return CachedReferenceMaterial(
        identity=reference_identity_from_document(value["identity"]),
        resources=tuple(cached_resource_ref_from_document(item) for item in resources),
        readable_resource=(
            cached_resource_ref_from_document(readable)
            if readable is not None
            else None
        ),
    )


def _material_to_document(
    value: CachedReferenceMaterial, *, record_id: str
) -> dict[str, Any]:
    return {
        "schema_version": REFERENCE_MATERIAL_CACHE_SCHEMA,
        "record_id": record_id,
        **cached_reference_material_to_document(value),
    }


def _identities_compatible(
    left: ReferenceIdentity, right: ReferenceIdentity
) -> bool:
    strong_overlap = any(
        (
            left.inspire_recid
            and right.inspire_recid
            and left.inspire_recid == right.inspire_recid,
            left.arxiv_id
            and right.arxiv_id
            and left.arxiv_id == right.arxiv_id,
            bool(set(left.dois) & set(right.dois)),
            bool(set(left.urls) & set(right.urls)),
        )
    )
    conflicts = any(
        (
            left.inspire_recid
            and right.inspire_recid
            and left.inspire_recid != right.inspire_recid,
            left.arxiv_id
            and right.arxiv_id
            and left.arxiv_id != right.arxiv_id,
        )
    )
    if strong_overlap:
        return not conflicts
    return (
        reference_identity_to_document(left)
        == reference_identity_to_document(right)
    )


def _merge_identities(
    left: ReferenceIdentity, right: ReferenceIdentity
) -> ReferenceIdentity:
    return ReferenceIdentity(
        inspire_recid=left.inspire_recid or right.inspire_recid,
        arxiv_id=left.arxiv_id or right.arxiv_id,
        dois=_dedupe((*left.dois, *right.dois)),
        urls=_dedupe((*left.urls, *right.urls)),
        title=right.title or left.title,
    )


def _merge_resources(
    left: tuple[CachedResourceRef, ...],
    right: tuple[CachedResourceRef, ...],
) -> tuple[CachedResourceRef, ...]:
    values: list[CachedResourceRef] = []
    seen: set[tuple[str, str, int]] = set()
    for item in (*left, *right):
        if item.content_identity not in seen:
            seen.add(item.content_identity)
            values.append(item)
    return tuple(values)


def _dedupe(values: Any) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            out.append(normalized)
    return tuple(out)


def _normalize_media_type(value: str) -> str:
    normalized = str(value or "").split(";", 1)[0].strip().casefold()
    if not normalized or "/" not in normalized:
        raise ValueError("media_type must be a normalized MIME type")
    return normalized


def _valid_media_types(values: list[Any]) -> bool:
    try:
        return all(
            isinstance(item, str) and item == _normalize_media_type(item)
            for item in values
        )
    except ValueError:
        return False


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "CACHED_REFERENCE_MATERIAL_SCHEMA",
    "CACHED_RESOURCE_REF_SCHEMA",
    "REFERENCE_IDENTITY_SCHEMA",
    "REFERENCE_MATERIAL_CACHE_SCHEMA",
    "REFERENCE_RESOURCE_OBJECT_SCHEMA",
    "CachedReferenceMaterial",
    "CachedResourceRef",
    "ReferenceCacheError",
    "ReferenceIdentity",
    "ReferenceMaterialCache",
    "cached_reference_material_from_document",
    "cached_reference_material_to_document",
    "cached_resource_ref_from_document",
    "cached_resource_ref_to_document",
    "normalize_reference_title",
    "normalize_reference_url",
    "reference_identity_from_document",
    "reference_identity_to_document",
]
