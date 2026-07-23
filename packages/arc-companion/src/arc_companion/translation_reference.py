"""Validated, chapter-local reference translations for companion builds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from arc_llm import EvidenceRequest

from .io import canonical_json, read_json, sha256_file, sha256_json, write_json
from .paper_broker import PaperBroker


TRANSLATION_REFERENCE_MANIFEST_VERSION = (
    "arc.companion.translation-reference-manifest.v1"
)
TRANSLATION_REFERENCE_ALIGNMENT_VERSION = (
    "arc.companion.translation-reference-alignment.v1"
)
TRANSLATION_REFERENCE_VALIDATION_VERSION = (
    "arc.companion.translation-reference-validation.v1"
)
TRANSLATION_REFERENCE_CHAPTER_VERSION = (
    "arc.companion.translation-reference-chapter.v1"
)
TRANSLATION_REFERENCE_PROMPT_VERSION = (
    "arc.companion.translation-reference-prompt.v1"
)
TRANSLATION_REFERENCE_PROVENANCE_VERSION = (
    "arc.companion.translation-reference-provenance.v1"
)
TRANSLATION_REFERENCE_REBINDING_VERSION = (
    "arc.companion.translation-reference-rebinding.v1"
)
PARSED_STRUCTURE_VIEW_VERSION = "arc.paper.parsed-structure-view.v1"
STRUCTURE_VERSION = "arc.paper.structure.v1"
AGGREGATE_NAMESPACE = "translation-reference"
MAX_REFERENCE_CHAPTER_BYTES = 16 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_PATH = re.compile(
    r"\.arc-companion/translation-references/manifests/"
    r"([0-9a-f]{64})\.json"
)
_BODY_KEYS = {
    "body", "text", "content", "payload", "sections", "blocks", "runs",
    "pages", "content_base64", "prompt",
}
_PROVENANCE_KEYS = {
    "schema_version", "manifest_path", "manifest_sha256",
    "requested_reference_id", "canonical_reference_id",
    "reference_source_sha256", "reference_document_sha256",
    "alignment_method", "alignment_version", "alignment_input_sha256",
    "alignment_validation_receipt_sha256", "mappings",
}
_PROVENANCE_MAPPING_KEYS = {
    "source_chapter_id", "source_chapter_content_sha256",
    "reference_chapter_id", "reference_section_ids",
    "reference_section_payload_sha256s",
    "reference_chapter_content_sha256", "object_id", "object_sha256",
    "object_size_bytes", "lookup_receipt_sha256",
    "rebinding_receipt_sha256",
}
_MANIFEST_MAPPING_KEYS = _PROVENANCE_MAPPING_KEYS - {
    "rebinding_receipt_sha256",
}


class TranslationReferenceError(RuntimeError):
    """A stable, normally non-retryable translation-reference failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.retryable = bool(retryable)
        self.provenance = dict(provenance or {})


@dataclass(frozen=True)
class ChapterReference:
    source_chapter_id: str
    source_chapter_content_sha256: str
    reference_chapter_id: str
    reference_section_ids: tuple[str, ...]
    reference_section_payload_sha256s: tuple[str, ...]
    reference_chapter_content_sha256: str
    content_artifact_id: str
    content_artifact_bytes: int
    lookup_receipt_sha256: str
    alignment_method: str
    alignment_version: str = TRANSLATION_REFERENCE_ALIGNMENT_VERSION

    def semantic_identity(self) -> dict[str, Any]:
        """Return only identities capable of invalidating this source chapter."""

        return {
            "schema_version": TRANSLATION_REFERENCE_CHAPTER_VERSION,
            "source_chapter_id": self.source_chapter_id,
            "source_chapter_content_sha256": (
                self.source_chapter_content_sha256
            ),
            "reference_chapter_id": self.reference_chapter_id,
            "reference_section_ids": list(self.reference_section_ids),
            "reference_section_payload_sha256s": list(
                self.reference_section_payload_sha256s
            ),
            "reference_chapter_content_sha256": (
                self.reference_chapter_content_sha256
            ),
            "content_artifact_id": self.content_artifact_id,
            "alignment_method": self.alignment_method,
            "alignment_version": self.alignment_version,
        }

    @property
    def semantic_identity_sha256(self) -> str:
        return sha256_json(self.semantic_identity())


@dataclass(frozen=True)
class TranslationReferenceBundle:
    manifest: Mapping[str, Any]
    manifest_sha256: str
    manifest_path: Path
    chapters: Mapping[str, ChapterReference]
    compact_provenance: Mapping[str, Any]
    project_root: Path

    def chapter(self, source_chapter_id: str) -> ChapterReference:
        try:
            return self.chapters[source_chapter_id]
        except KeyError as exc:
            raise TranslationReferenceError(
                "translation_reference_mapping_invalid",
                f"No translation reference maps source chapter {source_chapter_id}.",
            ) from exc


def normalize_mapping_options(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen_raw: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if value in seen_raw:
            raise ValueError("reference translation mappings must be unique")
        seen_raw.add(value)
        if value.count("=") != 1:
            raise ValueError(
                "reference translation mappings must be SOURCE_CHAPTER=REFERENCE_CHAPTER"
            )
        source, reference = (part.strip() for part in value.split("=", 1))
        if not source or not reference:
            raise ValueError(
                "reference translation mappings require two non-empty chapter IDs"
            )
        normalized.append(f"{source}={reference}")
    return tuple(normalized)


def build_primary_structure_view(
    parsed: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the already accepted primary parse to the closed structure view."""

    structure = parsed.get("structure")
    if not isinstance(structure, Mapping):
        raise TranslationReferenceError(
            "translation_reference_structure_invalid",
            "Primary source has no authoritative structure.",
        )
    sections = [
        dict(item) for item in parsed.get("sections") or []
        if isinstance(item, Mapping)
    ]
    by_id = {
        str(item.get("section_id") or ""): item
        for item in sections if str(item.get("section_id") or "")
    }
    chapters = []
    for raw in structure.get("chapters") or []:
        if not isinstance(raw, Mapping):
            continue
        title = _normalized_title(raw.get("title"))
        chapters.append({
            "chapter_id": str(raw.get("chapter_id") or ""),
            "title": title,
            "level": _integer(raw.get("level"), minimum=1),
            "leading_decimal_ordinal": leading_decimal_ordinal(title),
            "section_ids": [
                str(value) for value in raw.get("section_ids") or []
            ],
        })
    section_views = []
    for ordinal, raw in enumerate(sections):
        section_id = str(raw.get("section_id") or "")
        if not section_id:
            continue
        section_views.append({
            "section_id": section_id,
            "title": _normalized_title(raw.get("title")),
            "level": _integer(raw.get("level"), minimum=1),
            "ordinal": ordinal,
            "section_payload_sha256": sha256_json(by_id[section_id]),
        })
    coverage = _closed_coverage(structure.get("coverage"))
    return {
        "schema_version": PARSED_STRUCTURE_VIEW_VERSION,
        "requested_source_id": str(parsed.get("paper_id") or ""),
        "canonical_source_id": str(parsed.get("paper_id") or ""),
        "parser_version": str(parsed.get("parser_version") or ""),
        "source_hash": str(parsed.get("source_hash") or ""),
        "document_hash": str(
            parsed.get("document_hash") or parsed.get("source_hash") or ""
        ),
        "structure_schema_version": str(
            structure.get("schema_version") or ""
        ),
        "requested_document_kind": str(
            structure.get("requested_document_kind") or ""
        ),
        "document_kind": str(structure.get("document_kind") or ""),
        "structure_source": str(structure.get("structure_source") or ""),
        "chapters": chapters,
        "sections": section_views,
        "coverage": coverage,
    }


def leading_decimal_ordinal(title: Any) -> int | None:
    value = str(title or "").strip()
    match = re.match(r"([0-9]+)(.*)", value, flags=re.DOTALL)
    if match is None:
        return None
    digits, remainder = match.groups()
    if digits == "0" or digits.startswith("0"):
        return None
    if remainder:
        first = remainder[0]
        if not first.isspace() and not unicodedata.category(first).startswith("P"):
            return None
        after = remainder[1:]
        if (
            unicodedata.category(first).startswith("P")
            and after[:1].isascii()
            and after[:1].isdigit()
        ):
            return None
    return int(digits)


def align_translation_chapters(
    *,
    chapters_pack: Mapping[str, Any],
    primary_structure: Mapping[str, Any],
    reference_structure: Mapping[str, Any],
    explicit_mappings: Sequence[str] = (),
) -> dict[str, Any]:
    source_chapters = [
        dict(item) for item in chapters_pack.get("chapters") or []
        if isinstance(item, Mapping)
    ]
    reference_chapters = [
        dict(item) for item in reference_structure.get("chapters") or []
        if isinstance(item, Mapping)
    ]
    if not source_chapters or not reference_chapters:
        raise TranslationReferenceError(
            "translation_reference_structure_invalid",
            "Translation-reference alignment requires non-empty chapter structures.",
        )
    if explicit_mappings:
        try:
            normalized_mappings = normalize_mapping_options(explicit_mappings)
        except ValueError as exc:
            raise TranslationReferenceError(
                "translation_reference_mapping_invalid",
                str(exc),
            ) from exc
        pairs = _explicit_alignment(
            source_chapters, reference_chapters, normalized_mappings
        )
        method = "explicit"
    else:
        pairs = _automatic_alignment(
            source_chapters, primary_structure, reference_structure
        )
        method = "leading-decimal-ordinal"
    input_value = {
        "schema_version": TRANSLATION_REFERENCE_ALIGNMENT_VERSION,
        "method": method,
        "source_chapter_ids": [
            str(item.get("chapter_id") or "") for item in source_chapters
        ],
        "reference": [{
            "chapter_id": str(item.get("chapter_id") or ""),
            "section_ids": [
                str(value) for value in item.get("section_ids") or []
            ],
            "leading_decimal_ordinal": item.get("leading_decimal_ordinal"),
        } for item in reference_chapters],
        "pairs": pairs,
    }
    return {
        "method": method,
        "version": TRANSLATION_REFERENCE_ALIGNMENT_VERSION,
        "input_sha256": sha256_json(input_value),
        "validation_receipt": True,
        "validation_receipt_sha256": sha256_json({
            "schema_version": TRANSLATION_REFERENCE_VALIDATION_VERSION,
            "valid": True,
            "input_sha256": sha256_json(input_value),
        }),
        "pairs": pairs,
    }


def resolve_translation_reference(
    *,
    project_dir: Path,
    checkpoint_dir: Path | None,
    primary_parsed: Mapping[str, Any],
    primary_document: Mapping[str, Any],
    chapters_pack: Mapping[str, Any],
    requested_reference_id: str | None,
    explicit_mappings: Sequence[str] = (),
    broker: PaperBroker | Any | None = None,
) -> TranslationReferenceBundle | None:
    """Resolve one reference before intent guidance or any provider call."""

    reference_id = str(requested_reference_id or "").strip()
    if not reference_id:
        return None
    if broker is None:
        raise TranslationReferenceError(
            "translation_reference_source_unavailable",
            "Translation-reference resolution requires a Controller Broker.",
        )
    identity = _broker_data(
        broker, "get-parsed-identity", reference_id,
        error_code="translation_reference_source_unavailable",
    )
    reference_structure = _broker_data(
        broker, "get-parsed-structure", reference_id,
        error_code="translation_reference_structure_invalid",
    )
    _validate_structure_view(reference_structure)
    canonical_id = str(reference_structure["canonical_source_id"])
    if any((
        str(identity.get("paper_id") or "") != canonical_id,
        str(identity.get("source_hash") or "")
        != str(reference_structure["source_hash"]),
        str(identity.get("document_hash") or "")
        != str(reference_structure["document_hash"]),
    )):
        raise TranslationReferenceError(
            "translation_reference_source_changed",
            "Reference identity and body-free structure disagree.",
        )
    primary_structure = build_primary_structure_view(primary_parsed)
    _validate_structure_view(primary_structure, require_section_hashes=False)
    alignment = align_translation_chapters(
        chapters_pack=chapters_pack,
        primary_structure=primary_structure,
        reference_structure=reference_structure,
        explicit_mappings=explicit_mappings,
    )
    primary_hashes = _primary_chapter_hashes(
        chapters_pack, primary_document
    )
    reference_by_id = {
        str(item["chapter_id"]): dict(item)
        for item in reference_structure["chapters"]
    }
    section_by_id = {
        str(item["section_id"]): dict(item)
        for item in reference_structure["sections"]
    }
    prior_payloads = _prior_payload_hashes(project_dir, canonical_id)
    fetched_sections: dict[str, dict[str, Any]] = {}
    chapter_records: list[dict[str, Any]] = []
    chapter_refs: dict[str, ChapterReference] = {}
    for pair in alignment["pairs"]:
        source_id = str(pair["source_chapter_id"])
        reference_chapter_id = str(pair["reference_chapter_id"])
        reference_chapter = reference_by_id[reference_chapter_id]
        section_ids = tuple(
            str(value) for value in reference_chapter["section_ids"]
        )
        section_hashes = tuple(
            str(section_by_id[value]["section_payload_sha256"])
            for value in section_ids
        )
        lookup_identity = _chapter_lookup_identity(
            canonical_reference_source_id=canonical_id,
            reference_chapter_id=reference_chapter_id,
            section_ids=section_ids,
            section_hashes=section_hashes,
        )
        lookup_sha = sha256_json(lookup_identity)
        stored = None
        expected_payload_sha = prior_payloads.get(lookup_sha)
        if expected_payload_sha is not None:
            try:
                stored = broker.load_controller_aggregate_json(
                    namespace=AGGREGATE_NAMESPACE,
                    lookup_identity=lookup_identity,
                    expected_payload_sha256=expected_payload_sha,
                    max_bytes=MAX_REFERENCE_CHAPTER_BYTES,
                )
            except Exception as exc:
                raise _artifact_error(exc) from exc
        if stored is None:
            section_payloads = []
            for section_id, expected_sha in zip(
                section_ids, section_hashes, strict=True
            ):
                payload = fetched_sections.get(section_id)
                if payload is None:
                    payload = _broker_data(
                        broker, "get-parsed-section", canonical_id,
                        arguments={"section": section_id},
                        error_code="translation_reference_section_unavailable",
                    )
                    fetched_sections[section_id] = payload
                if sha256_json(payload) != expected_sha:
                    raise TranslationReferenceError(
                        "translation_reference_source_changed",
                        f"Reference section {section_id} changed after alignment.",
                    )
                section_payloads.append({
                    "section_id": section_id,
                    "section_payload_sha256": expected_sha,
                    "payload": payload,
                })
            aggregate = {
                "schema_version": TRANSLATION_REFERENCE_CHAPTER_VERSION,
                "canonical_reference_source_id": canonical_id,
                "reference_chapter_id": reference_chapter_id,
                "sections": section_payloads,
            }
            try:
                stored = broker.store_controller_aggregate_json(
                    namespace=AGGREGATE_NAMESPACE,
                    lookup_identity=lookup_identity,
                    payload=aggregate,
                    max_bytes=MAX_REFERENCE_CHAPTER_BYTES,
                )
                stored = {
                    **stored,
                    "payload": aggregate,
                }
            except Exception as exc:
                raise _artifact_error(exc) from exc
        _validate_aggregate(
            stored["payload"],
            canonical_source_id=canonical_id,
            reference_chapter_id=reference_chapter_id,
            section_ids=section_ids,
            section_hashes=section_hashes,
        )
        object_record = stored["object"]
        lookup_receipt = stored["lookup_receipt"]
        content_sha = str(object_record["payload_sha256"])
        reference = ChapterReference(
            source_chapter_id=source_id,
            source_chapter_content_sha256=primary_hashes[source_id],
            reference_chapter_id=reference_chapter_id,
            reference_section_ids=section_ids,
            reference_section_payload_sha256s=section_hashes,
            reference_chapter_content_sha256=content_sha,
            content_artifact_id=f"sha256-{content_sha}",
            content_artifact_bytes=int(object_record["size_bytes"]),
            lookup_receipt_sha256=str(stored["lookup_receipt_sha256"]),
            alignment_method=str(alignment["method"]),
        )
        chapter_refs[source_id] = reference
        chapter_records.append({
            "source_chapter_id": source_id,
            "source_chapter_content_sha256": primary_hashes[source_id],
            "reference_chapter_id": reference_chapter_id,
            "reference_section_ids": list(section_ids),
            "reference_section_payload_sha256s": list(section_hashes),
            "reference_chapter_content_sha256": content_sha,
            "object_id": reference.content_artifact_id,
            "object_path": str(object_record["object_path"]),
            "object_sha256": content_sha,
            "object_size_bytes": int(object_record["size_bytes"]),
            "lookup_identity_sha256": lookup_sha,
            "lookup_receipt_path": str(stored["lookup_receipt_path"]),
            "lookup_receipt_sha256": str(
                stored["lookup_receipt_sha256"]
            ),
        })
        if lookup_receipt.get("payload_sha256") != content_sha:
            raise TranslationReferenceError(
                "translation_reference_artifact_invalid",
                "Reference lookup receipt does not bind its aggregate.",
            )
    manifest_records = [{
        key: record[key] for key in _MANIFEST_MAPPING_KEYS
    } for record in chapter_records]
    manifest = {
        "schema_version": TRANSLATION_REFERENCE_MANIFEST_VERSION,
        "source": {
            "paper_id": str(primary_parsed.get("paper_id") or ""),
            "source_hash": str(primary_parsed.get("source_hash") or ""),
            "chapters_pack_sha256": sha256_json(chapters_pack),
        },
        "reference": {
            "requested_id": reference_id,
            "canonical_id": canonical_id,
            "source_hash": str(reference_structure["source_hash"]),
            "document_hash": str(reference_structure["document_hash"]),
        },
        "alignment": {
            key: alignment[key] for key in (
                "method", "version", "input_sha256", "validation_receipt",
                "validation_receipt_sha256",
            )
        },
        "mappings": manifest_records,
    }
    manifest_sha = sha256_json(manifest)
    manifest_relative = (
        ".arc-companion/translation-references/manifests/"
        f"{manifest_sha}.json"
    )
    manifest_path = _contained_path(project_dir, manifest_relative)
    _write_immutable_canonical_json(
        manifest_path,
        manifest,
        expected_name=f"{manifest_sha}.json",
    )
    enriched_mappings = []
    for record in chapter_records:
        reference = chapter_refs[str(record["source_chapter_id"])]
        rebinding = {
            "schema_version": TRANSLATION_REFERENCE_REBINDING_VERSION,
            "current_manifest_sha256": manifest_sha,
            "chapter_semantic_identity": reference.semantic_identity(),
            "chapter_semantic_identity_sha256": (
                reference.semantic_identity_sha256
            ),
            "lookup_receipt_path": record["lookup_receipt_path"],
            "lookup_receipt_sha256": record["lookup_receipt_sha256"],
            "object_path": record["object_path"],
            "object_sha256": record["object_sha256"],
            "object_size_bytes": record["object_size_bytes"],
        }
        binding_sha = sha256_json({
            "manifest_sha256": manifest_sha,
            "chapter_semantic_identity_sha256": (
                reference.semantic_identity_sha256
            ),
        })
        rebinding_relative = (
            ".arc-companion/paper-broker/controller-objects/"
            f"{AGGREGATE_NAMESPACE}/rebindings/sha256-{binding_sha}.json"
        )
        rebinding_path = _contained_path(project_dir, rebinding_relative)
        _write_immutable_canonical_json(
            rebinding_path,
            rebinding,
            expected_name=f"sha256-{binding_sha}.json",
        )
        enriched_mappings.append({
            **record,
            "rebinding_receipt_path": rebinding_relative,
            "rebinding_receipt_sha256": sha256_json(rebinding),
        })
    compact = _compact_provenance(
        manifest_relative=manifest_relative,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        mappings=enriched_mappings,
    )
    bundle = TranslationReferenceBundle(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        manifest_path=manifest_path,
        chapters=chapter_refs,
        compact_provenance=compact,
        project_root=project_dir.resolve(),
    )
    validate_translation_reference_bundle(bundle)
    binding = {
        "schema_version": TRANSLATION_REFERENCE_VALIDATION_VERSION,
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_sha,
        "compact_provenance": compact,
        "rebinding_receipts": [{
            "source_chapter_id": item["source_chapter_id"],
            "path": item["rebinding_receipt_path"],
            "sha256": item["rebinding_receipt_sha256"],
        } for item in enriched_mappings],
    }
    if checkpoint_dir is not None:
        write_json(checkpoint_dir / "translation-reference.json", binding)
    return bundle


def load_reference_chapter_payload(
    bundle: TranslationReferenceBundle,
    chapter_id: str,
) -> dict[str, Any]:
    """Read a verified immutable chapter object without contacting ARC-paper."""

    reference = bundle.chapter(chapter_id)
    path = _contained_path(
        bundle.project_root,
        (
            ".arc-companion/paper-broker/controller-objects/"
            f"{AGGREGATE_NAMESPACE}/objects/"
            f"sha256-{reference.reference_chapter_content_sha256}.json"
        ),
    )
    if (
        not path.is_file()
        or path.stat().st_size != reference.content_artifact_bytes
        or sha256_file(path) != reference.reference_chapter_content_sha256
    ):
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Reference chapter aggregate is missing or changed.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Reference chapter aggregate is unreadable.",
        ) from exc
    if not isinstance(payload, dict):
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Reference chapter aggregate has invalid shape.",
        )
    _validate_aggregate(
        payload,
        canonical_source_id=str(
            bundle.compact_provenance.get("canonical_reference_id") or ""
        ),
        reference_chapter_id=reference.reference_chapter_id,
        section_ids=reference.reference_section_ids,
        section_hashes=reference.reference_section_payload_sha256s,
    )
    return payload


def validate_translation_reference_bundle(
    bundle: TranslationReferenceBundle,
) -> None:
    if (
        sha256_json(bundle.manifest) != bundle.manifest_sha256
        or bundle.manifest_path.name != f"{bundle.manifest_sha256}.json"
        or not bundle.manifest_path.is_file()
        or sha256_file(bundle.manifest_path) != bundle.manifest_sha256
    ):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Translation-reference manifest is invalid.",
        )
    _validate_manifest(bundle.manifest)
    validate_translation_reference_provenance(
        bundle.compact_provenance,
        project_root=bundle.project_root,
        expected_chapter_ids=tuple(bundle.chapters),
    )
    for chapter_id in bundle.chapters:
        load_reference_chapter_payload(bundle, chapter_id)


def validate_translation_reference_provenance(
    value: Any,
    *,
    project_root: Path,
    expected_chapter_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the one shared body-free provenance contract."""

    if not isinstance(value, Mapping) or set(value) != _PROVENANCE_KEYS:
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Compact translation-reference provenance has unknown fields.",
        )
    compact = json.loads(canonical_json(dict(value)))
    _reject_body_keys(compact)
    if (
        compact.get("schema_version")
        != TRANSLATION_REFERENCE_PROVENANCE_VERSION
        or not _SHA256.fullmatch(str(compact.get("manifest_sha256") or ""))
        or not _MANIFEST_PATH.fullmatch(str(compact.get("manifest_path") or ""))
        or _MANIFEST_PATH.fullmatch(str(compact["manifest_path"])).group(1)
        != compact["manifest_sha256"]
    ):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Compact translation-reference provenance identity is invalid.",
        )
    manifest_path = _contained_path(project_root, compact["manifest_path"])
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != compact["manifest_sha256"]
    ):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Compact provenance manifest is missing or changed.",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Compact provenance manifest is unreadable.",
        ) from exc
    if (
        not isinstance(manifest, dict)
        or sha256_json(manifest) != compact["manifest_sha256"]
    ):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Compact provenance manifest has invalid content.",
        )
    _validate_manifest(manifest)
    mappings = compact.get("mappings")
    if (
        not isinstance(mappings, list)
        or not mappings
        or any(
            not isinstance(item, dict)
            or set(item) != _PROVENANCE_MAPPING_KEYS
            for item in mappings
        )
    ):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Compact provenance mappings are invalid.",
        )
    chapter_ids = [str(item["source_chapter_id"]) for item in mappings]
    if (
        len(chapter_ids) != len(set(chapter_ids))
        or (
            expected_chapter_ids is not None
            and chapter_ids != list(expected_chapter_ids)
        )
    ):
        raise TranslationReferenceError(
            "translation_reference_mapping_invalid",
            "Compact provenance chapter coverage is invalid.",
        )
    manifest_mappings = {
        str(item.get("source_chapter_id") or ""): item
        for item in manifest.get("mappings") or []
        if isinstance(item, Mapping)
    }
    for item in mappings:
        _validate_compact_mapping(
            item,
            manifest_sha256=str(compact["manifest_sha256"]),
            manifest_mapping=manifest_mappings.get(
                str(item["source_chapter_id"])
            ),
            manifest=manifest,
            project_root=project_root,
        )
    reference = manifest.get("reference") or {}
    alignment = manifest.get("alignment") or {}
    expected_top = {
        "requested_reference_id": reference.get("requested_id"),
        "canonical_reference_id": reference.get("canonical_id"),
        "reference_source_sha256": reference.get("source_hash"),
        "reference_document_sha256": reference.get("document_hash"),
        "alignment_method": alignment.get("method"),
        "alignment_version": alignment.get("version"),
        "alignment_input_sha256": alignment.get("input_sha256"),
        "alignment_validation_receipt_sha256": alignment.get(
            "validation_receipt_sha256"
        ),
    }
    if any(compact.get(key) != expected for key, expected in expected_top.items()):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Compact provenance does not match its manifest.",
        )
    return compact


def _explicit_alignment(
    source_chapters: list[dict[str, Any]],
    reference_chapters: list[dict[str, Any]],
    mappings: Sequence[str],
) -> list[dict[str, str]]:
    parsed = [tuple(value.split("=", 1)) for value in mappings]
    source_ids = [str(item.get("chapter_id") or "") for item in source_chapters]
    reference_order = {
        str(item.get("chapter_id") or ""): index
        for index, item in enumerate(reference_chapters)
    }
    mapped_source = [source for source, _reference in parsed]
    mapped_reference = [reference for _source, reference in parsed]
    if (
        mapped_source != source_ids
        or len(mapped_source) != len(set(mapped_source))
        or len(mapped_reference) != len(set(mapped_reference))
        or any(value not in reference_order for value in mapped_reference)
        or [reference_order[value] for value in mapped_reference]
        != sorted(reference_order[value] for value in mapped_reference)
    ):
        raise TranslationReferenceError(
            "translation_reference_mapping_invalid",
            "Explicit translation-reference mapping is incomplete or unordered.",
        )
    return [
        {"source_chapter_id": source, "reference_chapter_id": reference}
        for source, reference in parsed
    ]


def _automatic_alignment(
    source_chapters: list[dict[str, Any]],
    primary: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> list[dict[str, str]]:
    try:
        _validate_structure_for_automatic(primary)
        _validate_structure_for_automatic(reference)
        primary_chapters = list(primary["chapters"])
        reference_chapters = list(reference["chapters"])
        pack_ids = [str(item.get("chapter_id") or "") for item in source_chapters]
        primary_ids = [str(item["chapter_id"]) for item in primary_chapters]
        if (
            pack_ids != primary_ids
            or len(primary_chapters) != len(reference_chapters)
            or primary["document_kind"] != reference["document_kind"]
        ):
            raise ValueError("chapter universe differs")
        expected = list(range(1, len(primary_chapters) + 1))
        primary_ordinals = [
            item.get("leading_decimal_ordinal") for item in primary_chapters
        ]
        reference_ordinals = [
            item.get("leading_decimal_ordinal") for item in reference_chapters
        ]
        if primary_ordinals != expected or reference_ordinals != expected:
            raise ValueError("leading decimal ordinals are not canonical 1..N")
    except (KeyError, TypeError, ValueError) as exc:
        raise TranslationReferenceError(
            "translation_reference_alignment_ambiguous",
            "Automatic translation-reference alignment is ambiguous.",
        ) from exc
    return [{
        "source_chapter_id": str(source["chapter_id"]),
        "reference_chapter_id": str(reference_item["chapter_id"]),
    } for source, reference_item in zip(
        primary_chapters, reference_chapters, strict=True
    )]


def _validate_structure_for_automatic(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != PARSED_STRUCTURE_VIEW_VERSION
        or value.get("structure_schema_version") != STRUCTURE_VERSION
        or value.get("document_kind") not in {"article", "book"}
    ):
        raise ValueError("structure version or document kind is invalid")
    coverage = value.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("status") != "complete"
        or coverage.get("missing") != []
        or coverage.get("unexpected") != []
        or coverage.get("duplicates") != []
        or coverage.get("monotonic_order") is not True
        or coverage.get("expected_count") != coverage.get("covered_count")
    ):
        raise ValueError("structure coverage is incomplete")
    groups = [
        tuple(str(section) for section in item.get("section_ids") or [])
        for item in value.get("chapters") or []
    ]
    flattened = [section for group in groups for section in group]
    if (
        not groups or any(not group for group in groups)
        or len(flattened) != len(set(flattened))
        or len(groups) != len(set(groups))
    ):
        raise ValueError("chapter section groups are empty or ambiguous")


def _broker_data(
    broker: Any,
    operation: str,
    source_id: str,
    *,
    arguments: Mapping[str, Any] | None = None,
    error_code: str,
) -> dict[str, Any]:
    request = EvidenceRequest(
        request_id=f"translation-reference-{operation}-{sha256_json([source_id, arguments])[:12]}",
        operation=operation,
        arguments={"source_id": source_id, **dict(arguments or {})},
        reason="Resolve the explicitly configured translation reference.",
        worker_id="translation-reference-controller",
        role="translation",
    )
    try:
        responses = broker.resolve_round((request,), round_number=0)
    except Exception as exc:
        retryable = bool(getattr(exc, "retryable", False))
        raise TranslationReferenceError(
            str(getattr(exc, "code", error_code)) if retryable else error_code,
            str(exc),
            retryable=retryable,
            provenance=dict(getattr(exc, "provenance", {}) or {}),
        ) from exc
    if len(responses) != 1 or not responses[0].ok:
        response = responses[0] if responses else None
        provenance = (
            dict(response.provenance)
            if response is not None and isinstance(response.provenance, Mapping)
            else {}
        )
        broker_error = provenance.get("error")
        broker_error = broker_error if isinstance(broker_error, Mapping) else {}
        retryable = broker_error.get("retryable") is True
        raise TranslationReferenceError(
            str(broker_error.get("code") or error_code)
            if retryable else error_code,
            str(response.error if response is not None else "Broker returned no response"),
            retryable=retryable,
            provenance=provenance,
        )
    envelope = responses[0].data
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("ok") is not True
        or not isinstance(envelope.get("data"), Mapping)
    ):
        raise TranslationReferenceError(
            error_code, "Broker returned an invalid reference payload."
        )
    return dict(envelope["data"])


def _validate_structure_view(
    value: Mapping[str, Any], *, require_section_hashes: bool = True,
) -> None:
    keys = {
        "schema_version", "requested_source_id", "canonical_source_id",
        "parser_version", "source_hash", "document_hash",
        "structure_schema_version", "requested_document_kind", "document_kind",
        "structure_source", "chapters", "sections", "coverage",
    }
    chapter_keys = {
        "chapter_id", "title", "level", "leading_decimal_ordinal",
        "section_ids",
    }
    section_keys = {
        "section_id", "title", "level", "ordinal", "section_payload_sha256",
    }
    coverage_keys = {
        "status", "expected_count", "covered_count", "duplicates", "missing",
        "unexpected", "monotonic_order",
    }
    if (
        set(value) != keys
        or value.get("schema_version") != PARSED_STRUCTURE_VIEW_VERSION
        or value.get("structure_schema_version") != STRUCTURE_VERSION
        or value.get("document_kind") not in {"article", "book"}
        or not isinstance(value.get("chapters"), list)
        or not isinstance(value.get("sections"), list)
        or not isinstance(value.get("coverage"), Mapping)
        or set(value["coverage"]) != coverage_keys
        or any(
            not isinstance(item, Mapping) or set(item) != chapter_keys
            for item in value["chapters"]
        )
        or any(
            not isinstance(item, Mapping) or set(item) != section_keys
            for item in value["sections"]
        )
        or (
            require_section_hashes
            and any(
                not _SHA256.fullmatch(
                    str(item.get("section_payload_sha256") or "")
                )
                for item in value["sections"]
            )
        )
    ):
        raise TranslationReferenceError(
            "translation_reference_structure_invalid",
            "Parsed reference structure view is invalid.",
        )
    chapter_ids = [str(item["chapter_id"]) for item in value["chapters"]]
    section_ids = [str(item["section_id"]) for item in value["sections"]]
    chapter_section_ids = [
        str(section_id)
        for chapter in value["chapters"]
        for section_id in chapter["section_ids"]
    ]
    if (
        any(not item for item in chapter_ids)
        or len(chapter_ids) != len(set(chapter_ids))
        or any(not item for item in section_ids)
        or len(section_ids) != len(set(section_ids))
        or [item["ordinal"] for item in value["sections"]]
        != list(range(len(value["sections"])))
        or any(
            not isinstance(item["section_ids"], list)
            or not item["section_ids"]
            for item in value["chapters"]
        )
        or any(section_id not in set(section_ids) for section_id in chapter_section_ids)
    ):
        raise TranslationReferenceError(
            "translation_reference_structure_invalid",
            "Parsed reference structure identities are invalid.",
        )


def _closed_coverage(value: Any) -> dict[str, Any]:
    source = dict(value) if isinstance(value, Mapping) else {}
    return {
        "status": str(source.get("status") or ""),
        "expected_count": _integer(source.get("expected_count"), minimum=0),
        "covered_count": _integer(source.get("covered_count"), minimum=0),
        "duplicates": [str(item) for item in source.get("duplicates") or []],
        "missing": [str(item) for item in source.get("missing") or []],
        "unexpected": [str(item) for item in source.get("unexpected") or []],
        "monotonic_order": source.get("monotonic_order") is True,
    }


def _primary_chapter_hashes(
    chapters_pack: Mapping[str, Any], document: Mapping[str, Any],
) -> dict[str, str]:
    blocks = {
        str(item.get("block_id") or item.get("source_id") or ""): dict(item)
        for item in document.get("blocks") or []
        if isinstance(item, Mapping)
    }
    result = {}
    for chapter in chapters_pack.get("chapters") or []:
        chapter_id = str(chapter.get("chapter_id") or "")
        block_ids = [str(value) for value in chapter.get("block_ids") or []]
        if not chapter_id or any(value not in blocks for value in block_ids):
            raise TranslationReferenceError(
                "translation_reference_structure_invalid",
                "Primary chapter content cannot be identified exactly.",
            )
        result[chapter_id] = sha256_json({
            "chapter_id": chapter_id,
            "block_ids": block_ids,
            "blocks": [blocks[value] for value in block_ids],
        })
    return result


def _prior_payload_hashes(
    project_dir: Path, canonical_reference_id: str,
) -> dict[str, str]:
    root = (
        project_dir.resolve() / ".arc-companion"
        / "translation-references" / "manifests"
    )
    result: dict[str, str] = {}
    if not root.is_dir():
        return result
    for path in sorted(root.glob("[0-9a-f]" * 64 + ".json")):
        try:
            value = read_json(path)
        except (OSError, ValueError):
            continue
        if (
            not isinstance(value, Mapping)
            or (value.get("reference") or {}).get("canonical_id")
            != canonical_reference_id
        ):
            continue
        for mapping in value.get("mappings") or []:
            if not isinstance(mapping, Mapping):
                continue
            object_sha256 = str(mapping.get("object_sha256") or "")
            section_ids = mapping.get("reference_section_ids")
            section_hashes = mapping.get(
                "reference_section_payload_sha256s"
            )
            if (
                not _SHA256.fullmatch(object_sha256)
                or not isinstance(section_ids, list)
                or not isinstance(section_hashes, list)
                or len(section_ids) != len(section_hashes)
            ):
                continue
            lookup_identity = _chapter_lookup_identity(
                canonical_reference_source_id=canonical_reference_id,
                reference_chapter_id=str(
                    mapping.get("reference_chapter_id") or ""
                ),
                section_ids=[str(item) for item in section_ids],
                section_hashes=[str(item) for item in section_hashes],
            )
            result[sha256_json(lookup_identity)] = object_sha256
    return result


def _validate_aggregate(
    value: Any,
    *,
    canonical_source_id: str,
    reference_chapter_id: str,
    section_ids: Sequence[str],
    section_hashes: Sequence[str],
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "schema_version", "canonical_reference_source_id",
            "reference_chapter_id", "sections",
        }
        or value.get("schema_version") != TRANSLATION_REFERENCE_CHAPTER_VERSION
        or value.get("canonical_reference_source_id") != canonical_source_id
        or value.get("reference_chapter_id") != reference_chapter_id
        or not isinstance(value.get("sections"), list)
        or [
            str(item.get("section_id") or "")
            for item in value["sections"] if isinstance(item, Mapping)
        ] != list(section_ids)
    ):
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Reference chapter aggregate shape is invalid.",
        )
    for item, section_id, expected_hash in zip(
        value["sections"], section_ids, section_hashes, strict=True
    ):
        if (
            not isinstance(item, Mapping)
            or set(item) != {
                "section_id", "section_payload_sha256", "payload",
            }
            or item.get("section_id") != section_id
            or item.get("section_payload_sha256") != expected_hash
            or not isinstance(item.get("payload"), Mapping)
            or sha256_json(item["payload"]) != expected_hash
        ):
            raise TranslationReferenceError(
                "translation_reference_artifact_invalid",
                "Reference aggregate section binding is invalid.",
            )


def _compact_provenance(
    *,
    manifest_relative: str,
    manifest_sha256: str,
    manifest: Mapping[str, Any],
    mappings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference = manifest["reference"]
    alignment = manifest["alignment"]
    return {
        "schema_version": TRANSLATION_REFERENCE_PROVENANCE_VERSION,
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_sha256,
        "requested_reference_id": reference["requested_id"],
        "canonical_reference_id": reference["canonical_id"],
        "reference_source_sha256": reference["source_hash"],
        "reference_document_sha256": reference["document_hash"],
        "alignment_method": alignment["method"],
        "alignment_version": alignment["version"],
        "alignment_input_sha256": alignment["input_sha256"],
        "alignment_validation_receipt_sha256": alignment[
            "validation_receipt_sha256"
        ],
        "mappings": [{
            key: item[key] for key in _PROVENANCE_MAPPING_KEYS
        } for item in mappings],
    }


def _validate_manifest(value: Mapping[str, Any]) -> None:
    source = value.get("source")
    reference = value.get("reference")
    alignment = value.get("alignment")
    mappings = value.get("mappings")
    if (
        set(value) != {
            "schema_version", "source", "reference", "alignment", "mappings",
        }
        or value.get("schema_version")
        != TRANSLATION_REFERENCE_MANIFEST_VERSION
        or not isinstance(source, Mapping)
        or set(source) != {"paper_id", "source_hash", "chapters_pack_sha256"}
        or not isinstance(reference, Mapping)
        or set(reference) != {
            "requested_id", "canonical_id", "source_hash", "document_hash",
        }
        or not isinstance(alignment, Mapping)
        or set(alignment) != {
            "method", "version", "input_sha256", "validation_receipt",
            "validation_receipt_sha256",
        }
        or alignment.get("method") not in {
            "explicit", "leading-decimal-ordinal",
        }
        or alignment.get("version")
        != TRANSLATION_REFERENCE_ALIGNMENT_VERSION
        or alignment.get("validation_receipt") is not True
        or not isinstance(mappings, list)
        or not mappings
        or any(
            not isinstance(item, Mapping)
            or set(item) != _MANIFEST_MAPPING_KEYS
            for item in mappings
        )
    ):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Translation-reference manifest has invalid shape.",
        )
    _reject_body_keys(value)
    sha_values = (
        source.get("source_hash"),
        source.get("chapters_pack_sha256"),
        reference.get("source_hash"),
        reference.get("document_hash"),
        alignment.get("input_sha256"),
        alignment.get("validation_receipt_sha256"),
    )
    if (
        any(not str(item or "").strip() for item in (
            source.get("paper_id"),
            reference.get("requested_id"),
            reference.get("canonical_id"),
        ))
        or any(not _SHA256.fullmatch(str(item or "")) for item in sha_values)
    ):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Translation-reference manifest identity is invalid.",
        )
    source_ids: list[str] = []
    reference_ids: list[str] = []
    for item in mappings:
        source_id = str(item.get("source_chapter_id") or "")
        reference_id = str(item.get("reference_chapter_id") or "")
        section_ids = item.get("reference_section_ids")
        section_hashes = item.get("reference_section_payload_sha256s")
        sha_fields = (
            item.get("source_chapter_content_sha256"),
            item.get("reference_chapter_content_sha256"),
            item.get("object_sha256"),
            item.get("lookup_receipt_sha256"),
        )
        if (
            not source_id
            or not reference_id
            or not isinstance(section_ids, list)
            or not section_ids
            or len(section_ids) != len(set(map(str, section_ids)))
            or not isinstance(section_hashes, list)
            or len(section_ids) != len(section_hashes)
            or any(not _SHA256.fullmatch(str(item or "")) for item in sha_fields)
            or any(
                not _SHA256.fullmatch(str(item or ""))
                for item in section_hashes
            )
            or item.get("reference_chapter_content_sha256")
            != item.get("object_sha256")
            or item.get("object_id")
            != f"sha256-{item.get('object_sha256')}"
            or isinstance(item.get("object_size_bytes"), bool)
            or not isinstance(item.get("object_size_bytes"), int)
            or item.get("object_size_bytes") < 2
        ):
            raise TranslationReferenceError(
                "translation_reference_manifest_invalid",
                "Translation-reference manifest mapping is invalid.",
            )
        source_ids.append(source_id)
        reference_ids.append(reference_id)
    if (
        len(source_ids) != len(set(source_ids))
        or len(reference_ids) != len(set(reference_ids))
    ):
        raise TranslationReferenceError(
            "translation_reference_mapping_invalid",
            "Translation-reference manifest mappings are not one-to-one.",
        )


def _validate_compact_mapping(
    item: Mapping[str, Any],
    *,
    manifest_sha256: str,
    manifest_mapping: Any,
    manifest: Mapping[str, Any],
    project_root: Path,
) -> None:
    sha_keys = (
        "source_chapter_content_sha256",
        "reference_chapter_content_sha256", "object_sha256",
        "lookup_receipt_sha256", "rebinding_receipt_sha256",
    )
    if (
        not isinstance(manifest_mapping, Mapping)
        or any(
            not _SHA256.fullmatch(str(item.get(key) or ""))
            for key in sha_keys
        )
        or not isinstance(item.get("reference_section_ids"), list)
        or not isinstance(
            item.get("reference_section_payload_sha256s"), list
        )
        or len(item["reference_section_ids"])
        != len(item["reference_section_payload_sha256s"])
        or any(
            not _SHA256.fullmatch(str(value))
            for value in item["reference_section_payload_sha256s"]
        )
        or item.get("object_id") != f"sha256-{item.get('object_sha256')}"
        or isinstance(item.get("object_size_bytes"), bool)
        or not isinstance(item.get("object_size_bytes"), int)
        or item.get("object_size_bytes") < 2
    ):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Compact provenance mapping identity is invalid.",
        )
    common = (
        "source_chapter_id", "source_chapter_content_sha256",
        "reference_chapter_id", "reference_section_ids",
        "reference_section_payload_sha256s",
        "reference_chapter_content_sha256", "object_id", "object_sha256",
        "object_size_bytes", "lookup_receipt_sha256",
    )
    if any(item.get(key) != manifest_mapping.get(key) for key in common):
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Compact provenance mapping does not match its manifest.",
        )
    reference = manifest["reference"]
    lookup_identity = _chapter_lookup_identity(
        canonical_reference_source_id=str(reference["canonical_id"]),
        reference_chapter_id=str(item["reference_chapter_id"]),
        section_ids=item["reference_section_ids"],
        section_hashes=item["reference_section_payload_sha256s"],
    )
    lookup_identity_sha256 = sha256_json(lookup_identity)
    object_relative = (
        ".arc-companion/paper-broker/controller-objects/"
        f"{AGGREGATE_NAMESPACE}/objects/"
        f"sha256-{item['object_sha256']}.json"
    )
    lookup_relative = (
        ".arc-companion/paper-broker/controller-objects/"
        f"{AGGREGATE_NAMESPACE}/lookups/"
        f"sha256-{lookup_identity_sha256}.json"
    )
    object_path = _contained_path(project_root, object_relative)
    lookup_path = _contained_path(
        project_root, lookup_relative
    )
    if (
        not object_path.is_file()
        or object_path.stat().st_size != item["object_size_bytes"]
        or sha256_file(object_path) != item["object_sha256"]
        or not lookup_path.is_file()
        or sha256_file(lookup_path) != item["lookup_receipt_sha256"]
    ):
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Compact provenance object or lookup receipt changed.",
        )
    try:
        lookup_receipt = json.loads(lookup_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Compact provenance lookup receipt is unreadable.",
        ) from exc
    expected_lookup = {
        "schema_version": (
            "arc.companion.paper-broker-controller-aggregate-lookup.v1"
        ),
        "namespace": AGGREGATE_NAMESPACE,
        "lookup_identity": lookup_identity,
        "lookup_identity_sha256": lookup_identity_sha256,
        "object_path": object_relative,
        "payload_sha256": item["object_sha256"],
        "size_bytes": item["object_size_bytes"],
        "media_type": "application/json",
    }
    if lookup_receipt != expected_lookup:
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Compact provenance lookup receipt has invalid content.",
        )
    semantic_identity = {
        "schema_version": TRANSLATION_REFERENCE_CHAPTER_VERSION,
        "source_chapter_id": item["source_chapter_id"],
        "source_chapter_content_sha256": item[
            "source_chapter_content_sha256"
        ],
        "reference_chapter_id": item["reference_chapter_id"],
        "reference_section_ids": item["reference_section_ids"],
        "reference_section_payload_sha256s": item[
            "reference_section_payload_sha256s"
        ],
        "reference_chapter_content_sha256": item[
            "reference_chapter_content_sha256"
        ],
        "content_artifact_id": item["object_id"],
        "alignment_method": str(manifest["alignment"]["method"]),
        "alignment_version": TRANSLATION_REFERENCE_ALIGNMENT_VERSION,
    }
    semantic_sha = sha256_json(semantic_identity)
    binding_sha = sha256_json({
        "manifest_sha256": manifest_sha256,
        "chapter_semantic_identity_sha256": semantic_sha,
    })
    rebinding_path = _contained_path(
        project_root,
        (
            ".arc-companion/paper-broker/controller-objects/"
            f"{AGGREGATE_NAMESPACE}/rebindings/sha256-{binding_sha}.json"
        ),
    )
    if (
        not rebinding_path.is_file()
        or sha256_file(rebinding_path) != item["rebinding_receipt_sha256"]
    ):
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Compact provenance rebinding receipt changed.",
        )
    try:
        rebinding = json.loads(rebinding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Compact provenance rebinding receipt is unreadable.",
        ) from exc
    expected_rebinding = {
        "schema_version": TRANSLATION_REFERENCE_REBINDING_VERSION,
        "current_manifest_sha256": manifest_sha256,
        "chapter_semantic_identity": semantic_identity,
        "chapter_semantic_identity_sha256": semantic_sha,
        "lookup_receipt_path": lookup_relative,
        "lookup_receipt_sha256": item["lookup_receipt_sha256"],
        "object_path": object_relative,
        "object_sha256": item["object_sha256"],
        "object_size_bytes": item["object_size_bytes"],
    }
    if rebinding != expected_rebinding:
        raise TranslationReferenceError(
            "translation_reference_artifact_invalid",
            "Compact provenance rebinding receipt has invalid content.",
        )


def _chapter_lookup_identity(
    *,
    canonical_reference_source_id: str,
    reference_chapter_id: str,
    section_ids: Sequence[str],
    section_hashes: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": TRANSLATION_REFERENCE_CHAPTER_VERSION,
        "canonical_reference_source_id": canonical_reference_source_id,
        "reference_chapter_id": reference_chapter_id,
        "sections": [
            {
                "section_id": section_id,
                "section_payload_sha256": payload_sha256,
            }
            for section_id, payload_sha256
            in zip(section_ids, section_hashes, strict=True)
        ],
    }


def _artifact_error(exc: Exception) -> TranslationReferenceError:
    retryable = bool(getattr(exc, "retryable", False))
    return TranslationReferenceError(
        (
            str(getattr(exc, "code", "translation_reference_artifact_invalid"))
            if retryable
            else "translation_reference_artifact_invalid"
        ),
        str(exc),
        retryable=retryable,
        provenance=dict(getattr(exc, "provenance", {}) or {}),
    )


def _reject_body_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = _BODY_KEYS.intersection(map(str, value))
        if forbidden:
            raise TranslationReferenceError(
                "translation_reference_manifest_invalid",
                "Compact provenance contains body-bearing fields.",
            )
        for item in value.values():
            _reject_body_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_body_keys(item)


def _contained_path(project_root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Translation-reference path is missing.",
        )
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Translation-reference path is not a normalized relative path.",
        )
    root = project_root.expanduser().resolve()
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Translation-reference path escapes the project.",
        ) from exc
    return path


def _write_immutable_canonical_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_name: str,
) -> None:
    payload = canonical_json(dict(value)).encode("utf-8")
    if path.name != expected_name:
        raise TranslationReferenceError(
            "translation_reference_manifest_invalid",
            "Immutable JSON path does not match its identity.",
        )
    if path.exists():
        if path.read_bytes() != payload:
            raise TranslationReferenceError(
                "translation_reference_artifact_invalid",
                "Immutable translation-reference receipt conflicts.",
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _normalized_title(value: Any) -> str:
    return " ".join(str(value or "").split())


def _integer(value: Any, *, minimum: int) -> int:
    if isinstance(value, bool):
        return minimum
    try:
        result = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, result)
