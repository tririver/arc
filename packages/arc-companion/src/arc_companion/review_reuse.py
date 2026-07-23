"""Segment-local identity, source, planning, and persistence for Review reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from arc_llm import strip_arc_llm_call_records
from jsonschema import ValidationError, validate as validate_json_schema

from .review_arbitration import (
    REVIEW_ARBITRATION_RECEIPT_VERSION,
    ReviewArbitrationError,
    ReviewPatchSource,
    atomize_review_sources,
    canonical_json,
    canonical_sha256,
)
from .secure_io import SecureReadError, read_bounded_file, read_bounded_json


REVIEW_SEGMENT_IDENTITY_VERSION = "arc.companion.review-segment-identity.v1"
REVIEW_SEGMENT_SOURCE_VERSION = "arc.companion.review-segment-source.v1"
REVIEW_SEGMENT_ACCEPTANCE_VERSION = (
    "arc.companion.review-segment-acceptance.v1"
)
REVIEW_SEGMENT_VALIDATION_VERSION = (
    "arc.companion.review-segment-validation.v1"
)
REVIEW_REUSE_PLAN_VERSION = "arc.companion.review-reuse-plan.v1"
REVIEW_REUSE_RECEIPT_VERSION = "arc.companion.review-reuse-receipt.v1"
REVIEW_LEGACY_IMPORT_VERSION = "arc.companion.review-legacy-import.v1"
REVIEW_SEGMENT_RULE_VERSION = "arc.companion.review-segment-rule.v1"

REVIEW_REUSE_OBJECT_MAX_BYTES = 2 * 1024 * 1024
REVIEW_REUSE_ACCEPTANCE_MAX_BYTES = 256 * 1024
REVIEW_REUSE_MAX_ACCEPTANCES = 4096
REVIEW_REUSE_MAX_OBJECTS = 8192
REVIEW_REUSE_MAX_TOTAL_BYTES = 64 * 1024 * 1024
REVIEW_REUSE_MAX_OBJECT_LINKS = 64
REVIEW_REUSE_MAX_FINDINGS = 256
REVIEW_REUSE_MAX_PATCHES = 256
REVIEW_REUSE_MAX_SOURCES_PER_SEGMENT = 64
REVIEW_REUSE_AUDIT_MAX_BYTES = 16 * 1024

REVIEW_REUSE_INVALIDATION_CODES = frozenset({
    "source_changed",
    "translation_changed",
    "commentary_changed",
    "glossary_changed",
    "evidence_changed",
    "reference_changed",
    "intent_changed",
    "rule_changed",
    "schema_changed",
    "coverage_invalid",
    "source_schema_invalid",
    "target_invalid",
    "domain_invalid",
    "supersession_invalid",
    "acceptance_corrupt",
    "legacy_proof_unavailable",
    "legacy_unscoped_issue",
})

_SHA256_CHARS = frozenset("0123456789abcdef")
_IDENTITY_FIELDS = (
    "identity_version",
    "segment_id",
    "mode",
    "semantic_segment_sha256",
    "augmentation_block_annotation_input_sha256",
    "current_translation_sha256",
    "current_annotation_sha256",
    "local_glossary_sha256",
    "ordered_protected_names_sha256",
    "segment_evidence_sha256",
    "t14_reference_identity_sha256",
    "t14_reference_artifact_sha256",
    "intent_guidance_sha256",
    "annotation_language",
    "segment_rule",
    "t15_contracts",
    "provider_output_schema_version",
    "provider_output_schema_sha256",
    "validation_version",
)
_SOURCE_FIELDS = (
    "schema_version",
    "identity",
    "identity_sha256",
    "segment_id",
    "findings",
    "patches",
    "issues",
    "supersession_edges",
    "semantic_content_sha256",
    "validation_receipt",
    "audit",
)
_ACCEPTANCE_FIELDS = (
    "schema_version",
    "identity_sha256",
    "segment_id",
    "object_links",
    "validation",
    "t15_receipt",
    "accepted_merged_segment_sha256",
    "reviewed_output",
    "supersession_edges",
)
_ACCEPTANCE_VALIDATION_FIELDS = frozenset({
    "schema_version",
    "schema_valid",
    "domain_valid",
})
_T15_LINK_FIELDS = frozenset({
    "path",
    "sha256",
    "status",
    "schema_version",
    "semantic_input_sha256",
    "source_hashes",
    "merged_sha256",
    "final_review_sha256",
    "reviewed_translation_sha256",
    "reviewed_annotation_sha256",
    "supersession_edges",
})
_REVIEWED_OUTPUT_FIELDS = frozenset({
    "schema_version",
    "segments",
    "merged_output_sha256",
    "reviewed_translation_sha256",
    "reviewed_annotation_sha256",
})
REVIEW_REUSE_REVIEWED_OUTPUT_VERSION = (
    "arc.companion.review-reuse-reviewed-output.v1"
)
_REUSE_RECEIPT_FIELDS = frozenset({
    "schema_version",
    "plan_path",
    "plan_sha256",
    "identity_sha256s",
    "source_sha256s",
    "acceptance_sha256s",
    "new_segment_count",
    "reused_segment_count",
    "actual_review_calls",
    "t15_receipt",
    "merged_output_sha256",
    "merged_segment_sha256s",
    "schema_valid",
    "domain_valid",
})
_VALIDATION_FIELDS = frozenset({
    "schema_version",
    "schema_valid",
    "coverage_valid",
    "target_valid",
    "domain_valid",
    "supersession_valid",
    "candidate_count",
    "owned_block_ids",
})


class ReviewReuseError(RuntimeError):
    """A reusable Review identity, source, or binding failed closed."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - _SHA256_CHARS)
    )


def _require_sha256(value: Any, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not _is_sha256(value):
        raise ReviewReuseError(f"{label} must be a lowercase SHA-256")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ReviewReuseError(
        f"review reuse values must be JSON-compatible, got {type(value).__name__}"
    )


def sanitize_segment_evidence(value: Any) -> Any:
    """Canonicalize an already closed segment-local evidence projection."""

    return _json_value(value)


@dataclass(frozen=True)
class ReviewSegmentIdentity:
    """Immutable canonical semantic identity for one current Review segment."""

    _document_json: str = field(repr=False)
    sha256: str

    def __post_init__(self) -> None:
        document = json.loads(self._document_json)
        if set(document) != set(_IDENTITY_FIELDS):
            raise ReviewReuseError("review segment identity fields are not exact")
        if (
            document["identity_version"] != REVIEW_SEGMENT_IDENTITY_VERSION
            or not isinstance(document["segment_id"], str)
            or not document["segment_id"]
            or document["mode"] not in {
                "translation_enabled",
                "commentary_only",
            }
            or not isinstance(document["annotation_language"], str)
            or not document["annotation_language"]
            or not isinstance(document["segment_rule"], str)
            or not document["segment_rule"]
            or not isinstance(document["validation_version"], str)
            or not document["validation_version"]
        ):
            raise ReviewReuseError("review segment identity is malformed")
        for key in (
            "semantic_segment_sha256",
            "augmentation_block_annotation_input_sha256",
            "current_annotation_sha256",
            "local_glossary_sha256",
            "ordered_protected_names_sha256",
            "segment_evidence_sha256",
            "provider_output_schema_sha256",
        ):
            _require_sha256(document[key], key)
        for key in (
            "current_translation_sha256",
            "t14_reference_identity_sha256",
            "t14_reference_artifact_sha256",
            "intent_guidance_sha256",
        ):
            _require_sha256(document[key], key, nullable=True)
        if (
            document["mode"] == "commentary_only"
            and document["current_translation_sha256"] is not None
        ):
            raise ReviewReuseError(
                "commentary-only identity cannot bind a translation"
            )
        if (
            document["mode"] == "translation_enabled"
            and document["current_translation_sha256"] is None
        ):
            raise ReviewReuseError(
                "translation-enabled identity must bind a translation"
            )
        if (
            (document["t14_reference_identity_sha256"] is None)
            != (document["t14_reference_artifact_sha256"] is None)
        ):
            raise ReviewReuseError(
                "T14 identity and artifact hashes must be paired"
            )
        if (
            not isinstance(document["t15_contracts"], Mapping)
            or not document["t15_contracts"]
            or not isinstance(document["provider_output_schema_version"], str)
            or not document["provider_output_schema_version"]
        ):
            raise ReviewReuseError("review segment contracts are malformed")
        _require_sha256(self.sha256, "review segment identity hash")
        if canonical_sha256(document) != self.sha256:
            raise ReviewReuseError("review segment identity hash mismatch")

    @classmethod
    def build(
        cls,
        *,
        segment_id: str,
        mode: str,
        semantic_segment: Mapping[str, Any],
        augmentation_blocks: Sequence[Mapping[str, Any]],
        current_translation: Mapping[str, Any] | None,
        current_annotation: Mapping[str, Any],
        local_glossary: Mapping[str, Any],
        protected_names: Sequence[str],
        segment_evidence: Any,
        t14_reference_identity: Mapping[str, Any] | None,
        t14_reference_artifact_sha256: str | None,
        intent_guidance: Mapping[str, Any] | None,
        annotation_language: str,
        t15_contracts: Mapping[str, Any],
        provider_output_schema_version: str,
        provider_output_schema: Mapping[str, Any],
        segment_rule: str = REVIEW_SEGMENT_RULE_VERSION,
        validation_version: str = REVIEW_SEGMENT_VALIDATION_VERSION,
    ) -> "ReviewSegmentIdentity":
        segment_id = str(segment_id)
        if not segment_id:
            raise ReviewReuseError("review segment id is empty")
        if mode not in {"translation_enabled", "commentary_only"}:
            raise ReviewReuseError("review segment mode is invalid")
        if (
            str(semantic_segment.get("segment_id") or "")
            and str(semantic_segment.get("segment_id")) != segment_id
        ):
            raise ReviewReuseError(
                "top-level and semantic segment IDs do not match"
            )
        if mode == "translation_enabled" and current_translation is None:
            raise ReviewReuseError(
                "translation-enabled identity requires current translation"
            )
        if mode == "commentary_only" and current_translation is not None:
            raise ReviewReuseError(
                "commentary-only identity forbids current translation"
            )
        if (
            (t14_reference_identity is None)
            != (t14_reference_artifact_sha256 is None)
        ):
            raise ReviewReuseError(
                "T14 identity and artifact hashes must be paired"
            )
        if len(protected_names) != len(set(str(item) for item in protected_names)):
            raise ReviewReuseError("ordered protected names contain duplicates")
        document = {
            "identity_version": REVIEW_SEGMENT_IDENTITY_VERSION,
            "segment_id": segment_id,
            "mode": mode,
            "semantic_segment_sha256": canonical_sha256(semantic_segment),
            "augmentation_block_annotation_input_sha256": canonical_sha256(
                list(augmentation_blocks)
            ),
            "current_translation_sha256": (
                None
                if current_translation is None
                else canonical_sha256(current_translation)
            ),
            "current_annotation_sha256": canonical_sha256(current_annotation),
            "local_glossary_sha256": canonical_sha256(local_glossary),
            "ordered_protected_names_sha256": canonical_sha256(
                [str(item) for item in protected_names]
            ),
            "segment_evidence_sha256": canonical_sha256(
                sanitize_segment_evidence(segment_evidence)
            ),
            "t14_reference_identity_sha256": (
                None
                if t14_reference_identity is None
                else canonical_sha256(t14_reference_identity)
            ),
            "t14_reference_artifact_sha256": t14_reference_artifact_sha256,
            "intent_guidance_sha256": (
                None
                if intent_guidance is None
                else canonical_sha256(intent_guidance)
            ),
            "annotation_language": str(annotation_language),
            "segment_rule": str(segment_rule),
            "t15_contracts": _json_value(dict(t15_contracts)),
            "provider_output_schema_version": str(
                provider_output_schema_version
            ),
            "provider_output_schema_sha256": canonical_sha256(
                provider_output_schema
            ),
            "validation_version": str(validation_version),
        }
        encoded = canonical_json(document)
        return cls(
            _document_json=encoded,
            sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], sha256: str | None = None,
    ) -> "ReviewSegmentIdentity":
        exact = {key: _json_value(document[key]) for key in _IDENTITY_FIELDS}
        if set(document) != set(_IDENTITY_FIELDS):
            raise ReviewReuseError("review segment identity fields are not exact")
        encoded = canonical_json(exact)
        return cls(
            _document_json=encoded,
            sha256=sha256 or hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        )

    @property
    def document(self) -> Mapping[str, Any]:
        return json.loads(self._document_json)

    @property
    def segment_id(self) -> str:
        return str(self.document["segment_id"])

    @property
    def mode(self) -> str:
        return str(self.document["mode"])


def validate_current_review_identities(
    identities: Sequence[ReviewSegmentIdentity],
) -> tuple[ReviewSegmentIdentity, ...]:
    values = tuple(identities)
    segment_ids = [item.segment_id for item in values]
    if len(segment_ids) != len(set(segment_ids)):
        raise ReviewReuseError("duplicate current review segment id")
    for item in values:
        _require_sha256(item.sha256, "current review segment identity")
    return values


def _validate_supersession_edges(
    edges: Sequence[Sequence[str]],
    *,
    source: ReviewPatchSource,
    owned_block_ids: Sequence[str],
    allow_unknown: bool = False,
) -> tuple[tuple[tuple[str, str], ...], int]:
    patches = source.patches
    segment_ids = {
        str(patch.get("segment_id") or "") for patch in patches
    }
    if len(segment_ids) > 1:
        raise ReviewReuseError("segment source spans multiple segments")
    segment_id = next(iter(segment_ids), "")
    try:
        atomized = atomize_review_sources(
            [source],
            segment_order=[segment_id] if segment_id else [],
            block_order_by_segment={
                segment_id: [str(item) for item in owned_block_ids]
            } if segment_id else {},
            skip_translation=all(
                patch.get("translation_blocks") is None for patch in patches
            ),
        ) if patches else ()
    except ReviewArbitrationError as exc:
        raise ReviewReuseError(
            f"review source target validation failed: {exc}"
        ) from exc
    candidates = {item.candidate_sha256: item for item in atomized}
    graph: dict[str, set[str]] = {value: set() for value in candidates}
    validated: list[tuple[str, str]] = []
    for raw in edges:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ReviewReuseError("supersession edge is malformed")
        winner, loser = str(raw[0]), str(raw[1])
        if winner == loser:
            raise ReviewReuseError("supersession edge is not same-target")
        if winner not in candidates or loser not in candidates:
            if allow_unknown and _is_sha256(winner) and _is_sha256(loser):
                validated.append((winner, loser))
                continue
            raise ReviewReuseError("supersession edge is not same-target")
        if (
            candidates[winner].segment_id != candidates[loser].segment_id
            or candidates[winner].path != candidates[loser].path
        ):
            raise ReviewReuseError("supersession edge is not same-target")
        graph[winner].add(loser)
        validated.append((winner, loser))

    def visit(node: str, stack: set[str], seen: set[str]) -> None:
        if node in stack:
            raise ReviewReuseError("supersession graph is cyclic")
        if node in seen:
            return
        stack.add(node)
        for target in graph[node]:
            visit(target, stack, seen)
        stack.remove(node)
        seen.add(node)

    seen: set[str] = set()
    for candidate in graph:
        visit(candidate, set(), seen)
    return tuple(sorted(set(validated))), len(atomized)


@dataclass(frozen=True)
class ReviewSegmentSource:
    """One validated, scoped provider proposal for one segment."""

    identity: ReviewSegmentIdentity
    semantic_content_sha256: str
    validation_receipt: Mapping[str, Any]
    supersession_edges: tuple[tuple[str, str], ...]
    _findings_json: str = field(repr=False)
    _patches_json: str = field(repr=False)
    _issues_json: str = field(repr=False)
    _audit_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(
            self.semantic_content_sha256, "review segment source hash"
        )
        if canonical_sha256(self.semantic_body) != self.semantic_content_sha256:
            raise ReviewReuseError("review segment source hash mismatch")
        _validate_segment_validation_receipt(self.validation_receipt)
        audit_bytes = len(self._audit_json.encode("utf-8"))
        if audit_bytes > REVIEW_REUSE_AUDIT_MAX_BYTES:
            raise ReviewReuseError("review segment source audit is too large")

    @classmethod
    def build(
        cls,
        *,
        identity: ReviewSegmentIdentity,
        findings: Sequence[Mapping[str, Any]],
        patches: Sequence[Mapping[str, Any]],
        issues: Sequence[str] = (),
        supersession_edges: Sequence[Sequence[str]] = (),
        validation_receipt: Mapping[str, Any],
        audit: Mapping[str, Any] | None = None,
    ) -> "ReviewSegmentSource":
        clean_findings = strip_arc_llm_call_records(list(findings))
        clean_patches = strip_arc_llm_call_records(list(patches))
        clean_issues = strip_arc_llm_call_records(list(issues))
        if (
            len(clean_findings) > REVIEW_REUSE_MAX_FINDINGS
            or len(clean_patches) > REVIEW_REUSE_MAX_PATCHES
        ):
            raise ReviewReuseError("review segment source item bound exceeded")
        if clean_issues:
            raise ReviewReuseError("unscoped review issues are not reusable")
        validation = _validate_segment_validation_receipt(
            validation_receipt,
        )
        t15_patches = [
            (
                {"translation_blocks": None, **dict(patch)}
                if identity.mode == "commentary_only"
                and "translation_blocks" not in patch
                else dict(patch)
            )
            for patch in clean_patches
        ]
        provisional_edges: tuple[tuple[str, str], ...] = tuple(sorted({
            (str(item[0]), str(item[1]))
            for item in supersession_edges
            if isinstance(item, (list, tuple)) and len(item) == 2
        }))
        if len(provisional_edges) != len(supersession_edges):
            raise ReviewReuseError("supersession edge is malformed or duplicate")
        body = {
            "identity_sha256": identity.sha256,
            "segment_id": identity.segment_id,
            "findings": _json_value(clean_findings),
            "patches": _json_value(clean_patches),
            "issues": [],
            "supersession_edges": [list(item) for item in provisional_edges],
        }
        semantic_content_sha256 = canonical_sha256(body)
        patch_source = ReviewPatchSource.from_review(
            source_kind="segment_review",
            stable_order=0,
            review={
                "reviewed_segment_ids": [identity.segment_id],
                "findings": clean_findings,
                "patches": t15_patches,
            },
            segment_set=[identity.segment_id],
            source_semantic_identity={
                "identity_sha256": identity.sha256,
                "semantic_content_sha256": semantic_content_sha256,
            },
        )
        validated_edges, candidate_count = _validate_supersession_edges(
            provisional_edges,
            source=patch_source,
            owned_block_ids=validation["owned_block_ids"],
            allow_unknown=True,
        )
        if validation["candidate_count"] != candidate_count:
            raise ReviewReuseError(
                "validation candidate count does not match atomization"
            )
        if validated_edges != provisional_edges:
            raise ReviewReuseError("supersession canonicalization changed")
        return cls(
            identity=identity,
            semantic_content_sha256=semantic_content_sha256,
            validation_receipt=validation,
            supersession_edges=validated_edges,
            _findings_json=canonical_json(clean_findings),
            _patches_json=canonical_json(clean_patches),
            _issues_json=canonical_json(clean_issues),
            _audit_json=canonical_json(dict(audit or {})),
        )

    @property
    def findings(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(json.loads(self._findings_json))

    @property
    def patches(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(json.loads(self._patches_json))

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(json.loads(self._issues_json))

    @property
    def audit(self) -> Mapping[str, Any]:
        return json.loads(self._audit_json)

    @property
    def semantic_body(self) -> Mapping[str, Any]:
        return {
            "identity_sha256": self.identity.sha256,
            "segment_id": self.identity.segment_id,
            "findings": list(self.findings),
            "patches": list(self.patches),
            "issues": list(self.issues),
            "supersession_edges": [list(item) for item in self.supersession_edges],
        }

    @property
    def document(self) -> Mapping[str, Any]:
        return {
            "schema_version": REVIEW_SEGMENT_SOURCE_VERSION,
            "identity": dict(self.identity.document),
            "identity_sha256": self.identity.sha256,
            "segment_id": self.identity.segment_id,
            "findings": list(self.findings),
            "patches": list(self.patches),
            "issues": list(self.issues),
            "supersession_edges": [list(item) for item in self.supersession_edges],
            "semantic_content_sha256": self.semantic_content_sha256,
            "validation_receipt": dict(self.validation_receipt),
            "audit": dict(self.audit),
        }

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any],
    ) -> "ReviewSegmentSource":
        if set(document) != set(_SOURCE_FIELDS):
            raise ReviewReuseError("review segment source fields are not exact")
        if document.get("schema_version") != REVIEW_SEGMENT_SOURCE_VERSION:
            raise ReviewReuseError("review segment source version is unsupported")
        identity = ReviewSegmentIdentity.from_document(
            document["identity"], str(document["identity_sha256"]),
        )
        source = cls.build(
            identity=identity,
            findings=document["findings"],
            patches=document["patches"],
            issues=document["issues"],
            supersession_edges=document["supersession_edges"],
            validation_receipt=document["validation_receipt"],
            audit=document["audit"],
        )
        if source.semantic_content_sha256 != document["semantic_content_sha256"]:
            raise ReviewReuseError("review segment source content hash changed")
        return source

    def as_t15_source(self, *, stable_order: int) -> ReviewPatchSource:
        t15_patches = [
            (
                {"translation_blocks": None, **dict(patch)}
                if self.identity.mode == "commentary_only"
                and "translation_blocks" not in patch
                else dict(patch)
            )
            for patch in self.patches
        ]
        return ReviewPatchSource.from_review(
            source_kind="segment_review",
            stable_order=stable_order,
            review={
                "reviewed_segment_ids": [self.identity.segment_id],
                "findings": list(self.findings),
                "patches": t15_patches,
            },
            segment_set=[self.identity.segment_id],
            source_semantic_identity={
                "identity_sha256": self.identity.sha256,
                "semantic_content_sha256": self.semantic_content_sha256,
            },
        )


def _validate_segment_validation_receipt(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _VALIDATION_FIELDS
        or value.get("schema_version")
        != REVIEW_SEGMENT_VALIDATION_VERSION
        or any(
            value.get(key) is not True
            for key in (
                "schema_valid",
                "coverage_valid",
                "target_valid",
                "domain_valid",
                "supersession_valid",
            )
        )
        or type(value.get("candidate_count")) is not int
        or value["candidate_count"] < 0
        or not isinstance(value.get("owned_block_ids"), list)
        or len(value["owned_block_ids"]) != len(set(value["owned_block_ids"]))
        or not all(
            isinstance(item, str) and item
            for item in value["owned_block_ids"]
        )
    ):
        raise ReviewReuseError(
            "review segment validation receipt is invalid"
        )
    return _json_value(dict(value))


def validate_review_segment_source_set(
    sources: Sequence[ReviewSegmentSource],
) -> tuple[tuple[str, str], ...]:
    """Validate canonical supersession only after the chosen source set exists."""

    values = tuple(sources)
    if not values:
        raise ReviewReuseError("chosen review segment source set is empty")
    segment_ids = {item.identity.segment_id for item in values}
    modes = {item.identity.mode for item in values}
    owned_sets = {
        tuple(item.validation_receipt["owned_block_ids"])
        for item in values
    }
    if len(segment_ids) != 1 or len(modes) != 1 or len(owned_sets) != 1:
        raise ReviewReuseError("chosen review sources disagree on segment scope")
    segment_id = next(iter(segment_ids))
    owned_block_ids = next(iter(owned_sets))
    try:
        atoms = atomize_review_sources(
            [
                item.as_t15_source(stable_order=index)
                for index, item in enumerate(values)
            ],
            segment_order=[segment_id],
            block_order_by_segment={segment_id: list(owned_block_ids)},
            skip_translation=next(iter(modes)) == "commentary_only",
        )
    except ReviewArbitrationError as exc:
        raise ReviewReuseError(
            f"chosen review source target validation failed: {exc}"
        ) from exc
    candidates = {item.candidate_sha256: item for item in atoms}
    edges = tuple(sorted({
        edge for source in values for edge in source.supersession_edges
    }))
    graph: dict[str, set[str]] = {value: set() for value in candidates}
    for winner, loser in edges:
        if (
            winner not in candidates
            or loser not in candidates
            or winner == loser
            or candidates[winner].segment_id != candidates[loser].segment_id
            or candidates[winner].path != candidates[loser].path
        ):
            raise ReviewReuseError(
                "chosen review supersession edge is unknown or cross-target"
            )
        graph[winner].add(loser)

    def visit(node: str, stack: set[str], seen: set[str]) -> None:
        if node in stack:
            raise ReviewReuseError(
                "chosen review supersession graph is cyclic"
            )
        if node in seen:
            return
        stack.add(node)
        for target in graph[node]:
            visit(target, stack, seen)
        stack.remove(node)
        seen.add(node)

    seen: set[str] = set()
    for node in graph:
        visit(node, set(), seen)
    return edges


def split_review_segment_response(
    response: Mapping[str, Any],
    *,
    identities_by_segment: Mapping[str, ReviewSegmentIdentity],
    schema: Mapping[str, Any],
    validate_singleton: Callable[
        [ReviewSegmentIdentity, Mapping[str, Any]], Mapping[str, Any]
    ],
    audit: Mapping[str, Any] | None = None,
) -> tuple[ReviewSegmentSource, ...]:
    """Validate one complete call then split it into singleton reusable sources."""

    clean = strip_arc_llm_call_records(dict(response))
    try:
        validate_json_schema(instance=clean, schema=schema)
    except ValidationError as exc:
        raise ReviewReuseError(
            f"review source schema invalid: {exc.message}"
        ) from exc
    ordered_ids = [str(item) for item in clean.get("reviewed_segment_ids") or []]
    expected_ids = list(identities_by_segment)
    if (
        len(ordered_ids) != len(set(ordered_ids))
        or set(ordered_ids) != set(expected_ids)
    ):
        raise ReviewReuseError("review response coverage is not exact")
    findings = list(clean.get("findings") or [])
    patches = list(clean.get("patches") or [])
    if any(
        str(item.get("segment_id") or "") not in identities_by_segment
        for item in [*findings, *patches]
        if isinstance(item, Mapping)
    ):
        raise ReviewReuseError("review response contains an unknown segment id")
    result: list[ReviewSegmentSource] = []
    for segment_id in expected_ids:
        singleton = {
            "reviewed_segment_ids": [segment_id],
            "findings": [
                item for item in findings
                if str(item.get("segment_id") or "") == segment_id
            ],
            "patches": [
                item for item in patches
                if str(item.get("segment_id") or "") == segment_id
            ],
        }
        receipt = validate_singleton(
            identities_by_segment[segment_id], singleton,
        )
        result.append(ReviewSegmentSource.build(
            identity=identities_by_segment[segment_id],
            findings=singleton["findings"],
            patches=singleton["patches"],
            validation_receipt=receipt,
            audit=audit,
        ))
    return tuple(result)


@dataclass(frozen=True)
class AcceptedReviewSegment:
    identity: ReviewSegmentIdentity
    source_sha256s: tuple[str, ...]
    acceptance_sha256: str
    accepted_merged_segment_sha256: str | None = None
    reviewed_segment: Mapping[str, Any] | None = None
    valid: bool = True
    invalidation_code: str | None = None

    def __post_init__(self) -> None:
        for value in (*self.source_sha256s, self.acceptance_sha256):
            _require_sha256(value, "accepted review segment hash")
        if (
            self.accepted_merged_segment_sha256 is None
        ) != (self.reviewed_segment is None):
            raise ReviewReuseError(
                "accepted reviewed-segment proof is partial"
            )
        if self.accepted_merged_segment_sha256 is not None:
            _require_sha256(
                self.accepted_merged_segment_sha256,
                "accepted merged segment hash",
            )
            if canonical_sha256(self.reviewed_segment) != (
                self.accepted_merged_segment_sha256
            ):
                raise ReviewReuseError(
                    "accepted reviewed-segment body changed"
                )
        if (
            self.invalidation_code is not None
            and self.invalidation_code not in REVIEW_REUSE_INVALIDATION_CODES
        ):
            raise ReviewReuseError("review reuse invalidation code is unstable")


@dataclass(frozen=True)
class ReviewReusePlanEntry:
    segment_id: str
    identity_sha256: str
    disposition: str
    source_sha256s: tuple[str, ...]
    acceptance_sha256s: tuple[str, ...]
    reason: str
    estimated_calls: int
    planned_miss_chunk_index: int | None
    planned_miss_chunk_sha256: str | None
    planned_miss_logical_unit: str | None

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ReviewReuseError("review reuse plan segment id is empty")
        _require_sha256(self.identity_sha256, "review reuse plan identity")
        if self.disposition not in {
            "reused",
            "uncovered",
            "invalidated",
            "explicit_regeneration",
        }:
            raise ReviewReuseError("review reuse plan disposition is invalid")
        if not self.reason or (
            self.disposition == "invalidated"
            and self.reason not in REVIEW_REUSE_INVALIDATION_CODES
        ):
            raise ReviewReuseError("review reuse plan reason is invalid")
        if (
            type(self.estimated_calls) is not int
            or self.estimated_calls < 0
            or self.estimated_calls > 1
        ):
            raise ReviewReuseError("review reuse estimated calls are invalid")
        for values in (self.source_sha256s, self.acceptance_sha256s):
            if len(values) != len(set(values)):
                raise ReviewReuseError("review reuse plan hashes are duplicate")
            for value in values:
                _require_sha256(value, "review reuse plan link")
        chunk_values = (
            self.planned_miss_chunk_index,
            self.planned_miss_chunk_sha256,
            self.planned_miss_logical_unit,
        )
        if self.disposition == "reused":
            if (
                not self.source_sha256s
                or not self.acceptance_sha256s
                or self.estimated_calls != 0
                or any(value is not None for value in chunk_values)
            ):
                raise ReviewReuseError("reused plan entry is inconsistent")
        else:
            if self.source_sha256s or self.acceptance_sha256s:
                raise ReviewReuseError("miss plan entry contains reuse links")
            if any(value is None for value in chunk_values) and not all(
                value is None for value in chunk_values
            ):
                raise ReviewReuseError("miss chunk binding is partial")
            if self.planned_miss_chunk_index is not None:
                if (
                    type(self.planned_miss_chunk_index) is not int
                    or self.planned_miss_chunk_index < 0
                ):
                    raise ReviewReuseError("miss chunk index is invalid")
                _require_sha256(
                    self.planned_miss_chunk_sha256,
                    "miss chunk hash",
                )
                if (
                    not isinstance(self.planned_miss_logical_unit, str)
                    or not self.planned_miss_logical_unit
                ):
                    raise ReviewReuseError("miss logical unit is invalid")

    def document(self) -> Mapping[str, Any]:
        return {
            "segment_id": self.segment_id,
            "identity_sha256": self.identity_sha256,
            "disposition": self.disposition,
            "source_sha256s": list(self.source_sha256s),
            "acceptance_sha256s": list(self.acceptance_sha256s),
            "reason": self.reason,
            "estimated_calls": self.estimated_calls,
            "planned_miss_chunk_index": self.planned_miss_chunk_index,
            "planned_miss_chunk_sha256": self.planned_miss_chunk_sha256,
            "planned_miss_logical_unit": self.planned_miss_logical_unit,
        }


@dataclass(frozen=True)
class ReviewReusePlanChunk:
    ordered_identity_sha256s: tuple[str, ...]
    chunk_sha256: str
    logical_unit: str

    def __post_init__(self) -> None:
        if not self.ordered_identity_sha256s:
            raise ReviewReuseError("review reuse chunk is empty")
        for value in self.ordered_identity_sha256s:
            _require_sha256(value, "review reuse chunk identity")
        expected = canonical_sha256({
            "ordered_identity_sha256s": list(
                self.ordered_identity_sha256s
            ),
        })
        if self.chunk_sha256 != expected:
            raise ReviewReuseError("review reuse chunk hash mismatch")
        if self.logical_unit != f"review-segment-{expected}":
            raise ReviewReuseError("review reuse chunk logical unit mismatch")

    def document(self) -> Mapping[str, Any]:
        return {
            "ordered_identity_sha256s": list(
                self.ordered_identity_sha256s
            ),
            "chunk_sha256": self.chunk_sha256,
            "logical_unit": self.logical_unit,
        }


@dataclass(frozen=True)
class ReviewReusePlan:
    entries: tuple[ReviewReusePlanEntry, ...]
    chunks: tuple[ReviewReusePlanChunk, ...] = ()

    def __post_init__(self) -> None:
        segment_ids = [item.segment_id for item in self.entries]
        if len(segment_ids) != len(set(segment_ids)):
            raise ReviewReuseError("review reuse plan segments are duplicate")
        chunk_hashes = [item.chunk_sha256 for item in self.chunks]
        if len(chunk_hashes) != len(set(chunk_hashes)):
            raise ReviewReuseError("review reuse plan chunks are duplicate")
        identity_to_entry = {
            item.identity_sha256: item for item in self.entries
        }
        if len(identity_to_entry) != len(self.entries):
            raise ReviewReuseError("review reuse plan identities are duplicate")
        covered: list[str] = []
        for index, chunk in enumerate(self.chunks):
            for identity_sha256 in chunk.ordered_identity_sha256s:
                entry = identity_to_entry.get(identity_sha256)
                if (
                    entry is None
                    or entry.disposition == "reused"
                    or entry.planned_miss_chunk_index != index
                    or entry.planned_miss_chunk_sha256
                    != chunk.chunk_sha256
                    or entry.planned_miss_logical_unit != chunk.logical_unit
                ):
                    raise ReviewReuseError(
                        "review reuse entry/chunk binding mismatch"
                    )
                covered.append(identity_sha256)
        misses = [
            item for item in self.entries if item.disposition != "reused"
        ]
        if self.chunks:
            if (
                len(covered) != len(set(covered))
                or set(covered) != {
                    item.identity_sha256 for item in misses
                }
                or sum(item.estimated_calls for item in misses)
                != len(self.chunks)
            ):
                raise ReviewReuseError(
                    "review reuse chunk coverage/call estimate is invalid"
                )
        elif any(
            item.planned_miss_chunk_index is not None for item in misses
        ):
            raise ReviewReuseError("review reuse plan has orphan chunk links")

    @property
    def document(self) -> Mapping[str, Any]:
        return {
            "schema_version": REVIEW_REUSE_PLAN_VERSION,
            "entries": [dict(item.document()) for item in self.entries],
            "counts": {
                value: sum(
                    item.disposition == value for item in self.entries
                )
                for value in (
                    "reused",
                    "uncovered",
                    "invalidated",
                    "explicit_regeneration",
                )
            },
            "estimated_calls": sum(
                item.estimated_calls for item in self.entries
            ),
            "ordered_missing_chunks": [
                dict(item.document()) for item in self.chunks
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.document)

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        expected_sha256: str | None = None,
        require_sealed: bool = False,
    ) -> "ReviewReusePlan":
        if set(document) != {
            "schema_version",
            "entries",
            "counts",
            "estimated_calls",
            "ordered_missing_chunks",
        } or document.get("schema_version") != REVIEW_REUSE_PLAN_VERSION:
            raise ReviewReuseError("review reuse plan fields are not exact")
        entries = []
        for item in document.get("entries") or []:
            if not isinstance(item, Mapping) or set(item) != {
                "segment_id",
                "identity_sha256",
                "disposition",
                "source_sha256s",
                "acceptance_sha256s",
                "reason",
                "estimated_calls",
                "planned_miss_chunk_index",
                "planned_miss_chunk_sha256",
                "planned_miss_logical_unit",
            }:
                raise ReviewReuseError("review reuse plan entry is malformed")
            entries.append(ReviewReusePlanEntry(
                segment_id=str(item["segment_id"]),
                identity_sha256=str(item["identity_sha256"]),
                disposition=str(item["disposition"]),
                source_sha256s=tuple(item["source_sha256s"]),
                acceptance_sha256s=tuple(item["acceptance_sha256s"]),
                reason=str(item["reason"]),
                estimated_calls=item["estimated_calls"],
                planned_miss_chunk_index=item[
                    "planned_miss_chunk_index"
                ],
                planned_miss_chunk_sha256=item[
                    "planned_miss_chunk_sha256"
                ],
                planned_miss_logical_unit=item[
                    "planned_miss_logical_unit"
                ],
            ))
        chunks = tuple(
            ReviewReusePlanChunk(
                ordered_identity_sha256s=tuple(
                    item["ordered_identity_sha256s"]
                ),
                chunk_sha256=str(item["chunk_sha256"]),
                logical_unit=str(item["logical_unit"]),
            )
            for item in document.get("ordered_missing_chunks") or []
            if isinstance(item, Mapping)
        )
        plan = cls(tuple(entries), chunks)
        if dict(plan.document) != dict(document):
            raise ReviewReuseError("review reuse plan derived fields changed")
        if expected_sha256 is not None and plan.sha256 != expected_sha256:
            raise ReviewReuseError("review reuse plan hash mismatch")
        if require_sealed:
            misses = [
                item for item in plan.entries
                if item.disposition != "reused"
            ]
            if misses and (
                not plan.chunks
                or any(
                    item.planned_miss_chunk_index is None
                    or item.planned_miss_chunk_sha256 is None
                    or item.planned_miss_logical_unit is None
                    for item in misses
                )
            ):
                raise ReviewReuseError("review reuse plan is not sealed")
            if not misses and plan.chunks:
                raise ReviewReuseError(
                    "no-miss review reuse plan contains chunks"
                )
        return plan


def bind_review_reuse_plan_chunks(
    plan: ReviewReusePlan,
    ordered_chunk_segment_ids: Sequence[Sequence[str]],
) -> ReviewReusePlan:
    """Bind already selected misses to exact UTF-8 packing without adding bodies."""

    identity_by_segment = {
        item.segment_id: item.identity_sha256 for item in plan.entries
    }
    chunks: list[ReviewReusePlanChunk] = []
    chunk_by_segment: dict[str, tuple[int, ReviewReusePlanChunk]] = {}
    for index, segment_ids in enumerate(ordered_chunk_segment_ids):
        ordered_hashes = tuple(
            identity_by_segment[str(segment_id)]
            for segment_id in segment_ids
        )
        chunk_sha256 = canonical_sha256({
            "ordered_identity_sha256s": list(ordered_hashes),
        })
        chunk = ReviewReusePlanChunk(
            ordered_identity_sha256s=ordered_hashes,
            chunk_sha256=chunk_sha256,
            logical_unit=f"review-segment-{chunk_sha256}",
        )
        chunks.append(chunk)
        for segment_id in segment_ids:
            value = str(segment_id)
            if value in chunk_by_segment:
                raise ReviewReuseError("review miss appears in multiple chunks")
            chunk_by_segment[value] = (index, chunk)
    misses = {
        item.segment_id
        for item in plan.entries if item.disposition != "reused"
    }
    if set(chunk_by_segment) != misses:
        raise ReviewReuseError("review miss chunk coverage is not exact")
    charged: set[int] = set()
    entries = []
    for item in plan.entries:
        if item.disposition == "reused":
            entries.append(item)
            continue
        chunk_index, chunk = chunk_by_segment[item.segment_id]
        estimated_calls = 0 if chunk_index in charged else 1
        charged.add(chunk_index)
        entries.append(ReviewReusePlanEntry(
            segment_id=item.segment_id,
            identity_sha256=item.identity_sha256,
            disposition=item.disposition,
            source_sha256s=item.source_sha256s,
            acceptance_sha256s=item.acceptance_sha256s,
            reason=item.reason,
            estimated_calls=estimated_calls,
            planned_miss_chunk_index=chunk_index,
            planned_miss_chunk_sha256=chunk.chunk_sha256,
            planned_miss_logical_unit=chunk.logical_unit,
        ))
    return ReviewReusePlan(tuple(entries), tuple(chunks))


def _identity_change_code(
    current: ReviewSegmentIdentity,
    prior: ReviewSegmentIdentity,
) -> str:
    current_doc, prior_doc = current.document, prior.document
    groups = (
        ("source_changed", (
            "semantic_segment_sha256",
            "augmentation_block_annotation_input_sha256",
        )),
        ("translation_changed", ("current_translation_sha256",)),
        ("commentary_changed", ("current_annotation_sha256",)),
        ("glossary_changed", ("local_glossary_sha256",)),
        ("evidence_changed", ("segment_evidence_sha256",)),
        ("reference_changed", (
            "t14_reference_identity_sha256",
            "t14_reference_artifact_sha256",
        )),
        ("intent_changed", ("intent_guidance_sha256",)),
        ("rule_changed", ("segment_rule",)),
        ("schema_changed", (
            "mode",
            "t15_contracts",
            "provider_output_schema_version",
            "provider_output_schema_sha256",
            "validation_version",
            "annotation_language",
        )),
    )
    for code, keys in groups:
        if any(current_doc[key] != prior_doc[key] for key in keys):
            return code
    return "acceptance_corrupt"


def plan_review_reuse(
    current_identities: Sequence[ReviewSegmentIdentity],
    accepted_sources: Mapping[str, Sequence[AcceptedReviewSegment]],
    regenerate_review: bool,
) -> ReviewReusePlan:
    """Plan deterministic segment reuse without I/O or provider calls."""

    identities = validate_current_review_identities(current_identities)
    entries: list[ReviewReusePlanEntry] = []
    for identity in identities:
        candidates = tuple(accepted_sources.get(identity.segment_id, ()))
        exact = [
            item for item in candidates
            if item.valid and item.identity.sha256 == identity.sha256
        ]
        if regenerate_review:
            disposition, reason = (
                "explicit_regeneration",
                "explicit_regeneration",
            )
        elif exact:
            selected = min(exact, key=lambda item: item.acceptance_sha256)
            source_hashes = tuple(selected.source_sha256s)
            acceptance_hashes = tuple(sorted(
                item.acceptance_sha256
                for item in exact
                if item.source_sha256s == selected.source_sha256s
            ))
            entries.append(ReviewReusePlanEntry(
                segment_id=identity.segment_id,
                identity_sha256=identity.sha256,
                disposition="reused",
                source_sha256s=source_hashes,
                acceptance_sha256s=acceptance_hashes,
                reason="validated_acceptance",
                estimated_calls=0,
                planned_miss_chunk_index=None,
                planned_miss_chunk_sha256=None,
                planned_miss_logical_unit=None,
            ))
            continue
        elif candidates:
            disposition = "invalidated"
            invalid = next(
                (
                    item.invalidation_code
                    for item in candidates
                    if item.invalidation_code is not None
                ),
                None,
            )
            reason = invalid or _identity_change_code(
                identity, candidates[0].identity,
            )
        else:
            disposition, reason = "uncovered", "no_valid_acceptance"
        entries.append(ReviewReusePlanEntry(
            segment_id=identity.segment_id,
            identity_sha256=identity.sha256,
            disposition=disposition,
            source_sha256s=(),
            acceptance_sha256s=(),
            reason=reason,
            estimated_calls=0,
            planned_miss_chunk_index=None,
            planned_miss_chunk_sha256=None,
            planned_miss_logical_unit=None,
        ))
    return ReviewReusePlan(tuple(entries))


def _safe_regular_file(path: Path, root: Path, *, suffix: str) -> bool:
    try:
        if path.is_symlink() or path.suffix != suffix:
            return False
        info = path.stat(follow_symlinks=False)
        return (
            stat.S_ISREG(info.st_mode)
            and path.resolve(strict=True).is_relative_to(root.resolve(strict=True))
        )
    except (OSError, RuntimeError):
        return False


def _read_bounded_json(path: Path, *, max_bytes: int) -> Mapping[str, Any]:
    if path.stat(follow_symlinks=False).st_size > max_bytes:
        raise ReviewReuseError("review reuse file exceeds its byte bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewReuseError("review reuse JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise ReviewReuseError("review reuse artifact must be an object")
    return value


def _publish_immutable_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = (canonical_json(document) + "\n").encode("utf-8")
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
        ):
            raise ReviewReuseError("immutable review reuse path collision")
        _reconcile_publication_links(path)
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != payload
            ):
                raise ReviewReuseError(
                    "immutable review reuse publication collision"
                )
        temporary_path.unlink(missing_ok=True)
        _reconcile_publication_links(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _reconcile_publication_links(path: Path) -> None:
    try:
        published = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReviewReuseError("immutable publication is unavailable") from exc
    for staged in path.parent.glob(f".{path.name}.*.tmp"):
        try:
            staged_stat = staged.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (
            stat.S_ISREG(staged_stat.st_mode)
            and (staged_stat.st_dev, staged_stat.st_ino)
            == (published.st_dev, published.st_ino)
        ):
            staged.unlink()
    final = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(final.st_mode) or final.st_nlink != 1:
        raise ReviewReuseError(
            "immutable publication link count is not reconciled"
        )


def _ensure_private_store_directories(
    project_dir: Path,
    *relative_directories: str,
) -> None:
    current = Path(project_dir)
    for name in (
        ".arc-companion",
        "review-segments",
        *relative_directories,
    ):
        current = current / name
        try:
            info = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=0o700)
            info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ReviewReuseError("review reuse directory chain is unsafe")
        os.chmod(current, 0o700)


@contextmanager
def _identity_lock(project_dir: Path, identity_sha256: str):
    _require_sha256(identity_sha256, "review identity lock")
    _ensure_private_store_directories(project_dir, "locks")
    root = review_reuse_root(project_dir)
    lock_root = root / "locks"
    path = lock_root / f"{identity_sha256}.lock"
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def review_reuse_root(project_dir: Path) -> Path:
    return Path(project_dir) / ".arc-companion" / "review-segments"


def publish_review_segment_object(
    project_dir: Path,
    source: ReviewSegmentSource,
) -> tuple[Path, str]:
    _ensure_private_store_directories(project_dir, "objects")
    root = review_reuse_root(project_dir)
    object_sha256 = canonical_sha256(source.document)
    path = root / "objects" / f"{object_sha256}.json"
    with _identity_lock(project_dir, source.identity.sha256):
        _publish_immutable_json(path, source.document)
    return path, object_sha256


def publish_reviewed_output(
    project_dir: Path,
    *,
    segments: Mapping[str, Mapping[str, Any]],
    merged_output_sha256: str,
    reviewed_translation_sha256: str,
    reviewed_annotation_sha256: str,
) -> tuple[Path, str]:
    _require_sha256(merged_output_sha256, "reviewed merged output hash")
    _require_sha256(
        reviewed_translation_sha256, "reviewed translation hash",
    )
    _require_sha256(
        reviewed_annotation_sha256, "reviewed annotation hash",
    )
    if not segments:
        raise ReviewReuseError("reviewed output has no segments")
    derived = _reviewed_output_hashes(segments)
    if (
        derived["merged_output_sha256"] != merged_output_sha256
        or derived["reviewed_translation_sha256"]
        != reviewed_translation_sha256
        or derived["reviewed_annotation_sha256"]
        != reviewed_annotation_sha256
    ):
        raise ReviewReuseError("reviewed output hashes do not match bodies")
    document = {
        "schema_version": REVIEW_REUSE_REVIEWED_OUTPUT_VERSION,
        "segments": _json_value(dict(segments)),
        "merged_output_sha256": merged_output_sha256,
        "reviewed_translation_sha256": reviewed_translation_sha256,
        "reviewed_annotation_sha256": reviewed_annotation_sha256,
    }
    output_sha256 = canonical_sha256(document)
    _ensure_private_store_directories(project_dir, "outputs")
    path = (
        review_reuse_root(project_dir)
        / "outputs"
        / f"{output_sha256}.json"
    )
    with _identity_lock(project_dir, output_sha256):
        _publish_immutable_json(path, document)
    return path, output_sha256


def _reviewed_output_hashes(
    segments: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, str]:
    annotations = {
        str(segment_id): value["annotation"]
        for segment_id, value in segments.items()
    }
    translation_values = {
        str(segment_id): value.get("translation")
        for segment_id, value in segments.items()
    }
    translations = (
        None
        if all(value is None for value in translation_values.values())
        else translation_values
    )
    return {
        "reviewed_translation_sha256": canonical_sha256(translations),
        "reviewed_annotation_sha256": canonical_sha256(annotations),
        "merged_output_sha256": canonical_sha256({
            "translations": translations,
            "annotations": annotations,
        }),
    }


def build_review_segment_acceptance(
    *,
    identity: ReviewSegmentIdentity,
    object_links: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    t15_receipt: Mapping[str, Any],
    accepted_merged_segment_sha256: str,
    reviewed_output: Mapping[str, Any],
    supersession_edges: Sequence[Sequence[str]] = (),
) -> tuple[Mapping[str, Any], str]:
    _require_sha256(
        accepted_merged_segment_sha256, "accepted merged segment hash"
    )
    _validate_acceptance_validation(validation)
    _validate_t15_acceptance_link(
        t15_receipt,
        accepted_merged_segment_sha256=accepted_merged_segment_sha256,
    )
    if any(
        not isinstance(item, (list, tuple))
        or len(item) != 2
        for item in supersession_edges
    ):
        raise ReviewReuseError("review acceptance supersession is malformed")
    normalized_edges = tuple(sorted(
        (str(item[0]), str(item[1])) for item in supersession_edges
    ))
    if (
        len(normalized_edges) != len(set(normalized_edges))
        or any(
            not _is_sha256(item[0]) or not _is_sha256(item[1])
            for item in normalized_edges
        )
        or not set(normalized_edges).issubset({
            (str(item[0]), str(item[1]))
            for item in t15_receipt["supersession_edges"]
        })
    ):
        raise ReviewReuseError(
            "review acceptance supersession was not applied by T15"
        )
    if (
        not object_links
        or len(object_links) > REVIEW_REUSE_MAX_OBJECT_LINKS
    ):
        raise ReviewReuseError("review acceptance object-link count is invalid")
    for link in object_links:
        if (
            not isinstance(link, Mapping)
            or set(link) != {
                "path",
                "object_sha256",
                "semantic_content_sha256",
            }
            or not _is_sha256(link.get("object_sha256"))
            or not _is_sha256(link.get("semantic_content_sha256"))
        ):
            raise ReviewReuseError("review acceptance object link is malformed")
    if (
        not isinstance(reviewed_output, Mapping)
        or set(reviewed_output) != {"path", "sha256"}
        or not _is_sha256(reviewed_output.get("sha256"))
    ):
        raise ReviewReuseError("reviewed-output link is malformed")
    document = {
        "schema_version": REVIEW_SEGMENT_ACCEPTANCE_VERSION,
        "identity_sha256": identity.sha256,
        "segment_id": identity.segment_id,
        "object_links": [_json_value(item) for item in object_links],
        "validation": _json_value(dict(validation)),
        "t15_receipt": _json_value(dict(t15_receipt)),
        "accepted_merged_segment_sha256": accepted_merged_segment_sha256,
        "reviewed_output": _json_value(dict(reviewed_output)),
        "supersession_edges": [
            [item[0], item[1]] for item in normalized_edges
        ],
    }
    return document, canonical_sha256(document)


def _validate_acceptance_validation(
    value: Mapping[str, Any],
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != _ACCEPTANCE_VALIDATION_FIELDS
        or value.get("schema_version")
        != REVIEW_SEGMENT_VALIDATION_VERSION
        or value.get("schema_valid") is not True
        or value.get("domain_valid") is not True
    ):
        raise ReviewReuseError("review acceptance validation is invalid")


def _validate_t15_acceptance_link(
    value: Mapping[str, Any],
    *,
    accepted_merged_segment_sha256: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != _T15_LINK_FIELDS
        or value.get("status") not in {"no_conflicts", "resolved"}
        or value.get("schema_version")
        != REVIEW_ARBITRATION_RECEIPT_VERSION
        or not all(
            _is_sha256(value.get(key))
            for key in (
                "sha256",
                "semantic_input_sha256",
                "merged_sha256",
                "final_review_sha256",
                "reviewed_translation_sha256",
                "reviewed_annotation_sha256",
            )
        )
        or not isinstance(value.get("source_hashes"), list)
        or not value["source_hashes"]
        or any(not _is_sha256(item) for item in value["source_hashes"])
        or not isinstance(value.get("supersession_edges"), list)
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or any(not _is_sha256(part) for part in item)
            for item in value["supersession_edges"]
        )
    ):
        raise ReviewReuseError("T15 receipt link is not reusable")


def publish_review_segment_acceptance(
    project_dir: Path,
    *,
    identity: ReviewSegmentIdentity,
    document: Mapping[str, Any],
    acceptance_sha256: str,
) -> Path:
    if set(document) != set(_ACCEPTANCE_FIELDS):
        raise ReviewReuseError("review segment acceptance fields are not exact")
    if canonical_sha256(document) != acceptance_sha256:
        raise ReviewReuseError("review segment acceptance hash mismatch")
    path = (
        review_reuse_root(project_dir)
        / "acceptances"
        / f"{identity.sha256}-{acceptance_sha256}.json"
    )
    _ensure_private_store_directories(project_dir, "acceptances")
    with _identity_lock(project_dir, identity.sha256):
        _publish_immutable_json(path, document)
    return path


def load_review_segment_acceptances(
    project_dir: Path,
    current_identities: Sequence[ReviewSegmentIdentity],
) -> tuple[
    Mapping[str, tuple[AcceptedReviewSegment, ...]],
    Mapping[str, tuple[ReviewSegmentSource, ...]],
]:
    """Load bounded, fully bound acceptances and immutable source objects."""

    current = validate_current_review_identities(current_identities)
    current_segment_ids = {item.segment_id for item in current}
    current_by_segment = {item.segment_id: item for item in current}
    root = review_reuse_root(project_dir)
    acceptance_root = root / "acceptances"
    object_root = root / "objects"
    candidates: dict[
        str,
        list[tuple[AcceptedReviewSegment, tuple[ReviewSegmentSource, ...],
                   tuple[Any, ...]]],
    ] = {}
    if not acceptance_root.is_dir() or acceptance_root.is_symlink():
        return {}, {}
    for directory in (
        Path(project_dir) / ".arc-companion",
        root,
        acceptance_root,
        object_root,
        root / "outputs",
    ):
        try:
            info = directory.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ReviewReuseError(
                "review reuse directory chain is unsafe"
            )
    paths = sorted(
        path for path in acceptance_root.iterdir()
        if not path.name.endswith(".tmp")
    )
    if len(paths) > REVIEW_REUSE_MAX_ACCEPTANCES:
        raise ReviewReuseError("review acceptance file-count bound exceeded")
    if sum(
        path.stat(follow_symlinks=False).st_size for path in paths
    ) > REVIEW_REUSE_MAX_TOTAL_BYTES:
        raise ReviewReuseError("review acceptance byte bound exceeded")
    object_paths = (
        [
            path for path in object_root.iterdir()
            if not path.name.endswith(".tmp")
        ]
        if object_root.is_dir() else []
    )
    if (
        len(object_paths) > REVIEW_REUSE_MAX_OBJECTS
        or sum(
            path.stat(follow_symlinks=False).st_size
            for path in object_paths
        ) > REVIEW_REUSE_MAX_TOTAL_BYTES
    ):
        raise ReviewReuseError("review object store bound exceeded")
    for path in paths:
        identity_prefix = path.name.split("-", 1)[0]
        _require_sha256(identity_prefix, "acceptance identity filename")
        with _identity_lock(project_dir, identity_prefix):
            _reconcile_publication_links(path)
        relative_acceptance = path.relative_to(project_dir)
        try:
            document = read_bounded_json(
                Path(project_dir),
                relative_acceptance,
                max_bytes=REVIEW_REUSE_ACCEPTANCE_MAX_BYTES,
            )
        except (SecureReadError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReviewReuseError("unsafe review acceptance path") from exc
        if not isinstance(document, Mapping):
            raise ReviewReuseError("review acceptance is not an object")
        if set(document) != set(_ACCEPTANCE_FIELDS):
            raise ReviewReuseError("review acceptance fields are not exact")
        if (
            document.get("schema_version")
            != REVIEW_SEGMENT_ACCEPTANCE_VERSION
        ):
            raise ReviewReuseError("review acceptance version is unsupported")
        acceptance_sha = canonical_sha256(document)
        expected_name = (
            f"{document.get('identity_sha256')}-{acceptance_sha}.json"
        )
        if path.name != expected_name:
            raise ReviewReuseError("review acceptance filename/hash mismatch")
        segment_id = str(document.get("segment_id") or "")
        if segment_id not in current_segment_ids:
            continue
        _validate_acceptance_validation(document.get("validation"))
        output_link = document.get("reviewed_output")
        if (
            not isinstance(output_link, Mapping)
            or set(output_link) != {"path", "sha256"}
            or not _is_sha256(output_link.get("sha256"))
        ):
            raise ReviewReuseError("reviewed-output link is malformed")
        output_relative = Path(str(output_link.get("path") or ""))
        if (
            output_relative.parts[:3]
            != (".arc-companion", "review-segments", "outputs")
            or len(output_relative.parts) != 4
            or output_relative.name != f"{output_link['sha256']}.json"
        ):
            raise ReviewReuseError("reviewed-output path is not exact")
        try:
            with _identity_lock(
                project_dir, str(output_link["sha256"]),
            ):
                _reconcile_publication_links(
                    Path(project_dir) / output_relative
                )
            output_document = read_bounded_json(
                Path(project_dir),
                output_relative,
                max_bytes=REVIEW_REUSE_OBJECT_MAX_BYTES,
            )
        except SecureReadError as exc:
            raise ReviewReuseError("reviewed-output link is unsafe") from exc
        if (
            not isinstance(output_document, Mapping)
            or set(output_document) != _REVIEWED_OUTPUT_FIELDS
            or output_document.get("schema_version")
            != REVIEW_REUSE_REVIEWED_OUTPUT_VERSION
            or canonical_sha256(output_document) != output_link["sha256"]
            or not _is_sha256(output_document.get("merged_output_sha256"))
            or not isinstance(output_document.get("segments"), Mapping)
            or segment_id not in output_document["segments"]
        ):
            raise ReviewReuseError("reviewed-output binding is invalid")
        derived_output_hashes = _reviewed_output_hashes(
            output_document["segments"]
        )
        if any(
            output_document.get(key) != value
            for key, value in derived_output_hashes.items()
        ):
            raise ReviewReuseError("reviewed-output body hashes changed")
        expected_merged = canonical_sha256(
            output_document["segments"][segment_id]
        )
        if (
            document.get("accepted_merged_segment_sha256")
            != expected_merged
        ):
            raise ReviewReuseError(
                "accepted merged segment does not match owned final Review"
            )
        t15 = document.get("t15_receipt")
        _validate_t15_acceptance_link(
            t15,
            accepted_merged_segment_sha256=str(expected_merged),
        )
        t15_relative = Path(str(t15.get("path") or ""))
        if (
            not str(t15_relative)
            or t15_relative.is_absolute()
            or ".." in t15_relative.parts
        ):
            raise ReviewReuseError("review acceptance T15 path is invalid")
        try:
            t15_bytes = read_bounded_file(
                Path(project_dir),
                t15_relative,
                max_bytes=REVIEW_REUSE_ACCEPTANCE_MAX_BYTES,
                suffixes=(".json",),
            )
            t15_raw = json.loads(t15_bytes)
        except (SecureReadError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReviewReuseError(
                "review acceptance T15 receipt is unsafe"
            ) from exc
        if (
            not isinstance(t15_raw, Mapping)
            or hashlib.sha256(t15_bytes).hexdigest()
            != t15["sha256"]
            or t15_raw.get("schema_version")
            != REVIEW_ARBITRATION_RECEIPT_VERSION
            or t15_raw.get("status") != t15["status"]
            or t15_raw.get("status") not in {"no_conflicts", "resolved"}
            or t15_raw.get("unresolved_paths")
            or any(
                t15_raw.get(key) != t15.get(key)
                for key in (
                    "semantic_input_sha256",
                    "source_hashes",
                    "merged_sha256",
                    "final_review_sha256",
                    "reviewed_translation_sha256",
                    "reviewed_annotation_sha256",
                    "supersession_edges",
                )
            )
            or output_document.get("reviewed_translation_sha256")
            != t15["reviewed_translation_sha256"]
            or output_document.get("reviewed_annotation_sha256")
            != t15["reviewed_annotation_sha256"]
        ):
            raise ReviewReuseError("review acceptance T15 receipt mismatch")
        loaded_sources: list[ReviewSegmentSource] = []
        object_hashes: list[str] = []
        links = document.get("object_links")
        if (
            not isinstance(links, list)
            or not links
            or len(links) > REVIEW_REUSE_MAX_OBJECT_LINKS
        ):
            raise ReviewReuseError("review object-link count is invalid")
        total_loaded = 0
        for link in links:
            if (
                not isinstance(link, Mapping)
                or set(link) != {
                    "path",
                    "object_sha256",
                    "semantic_content_sha256",
                }
            ):
                raise ReviewReuseError("review object link is malformed")
            relative = Path(str(link["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ReviewReuseError("review object link escapes project")
            if (
                relative.parts[:3]
                != (".arc-companion", "review-segments", "objects")
                or len(relative.parts) != 4
                or relative.name != f"{link['object_sha256']}.json"
            ):
                raise ReviewReuseError("review object path is not exact")
            try:
                with _identity_lock(
                    project_dir,
                    str(document["identity_sha256"]),
                ):
                    _reconcile_publication_links(
                        Path(project_dir) / relative
                    )
                source_document = read_bounded_json(
                    Path(project_dir),
                    relative,
                    max_bytes=REVIEW_REUSE_OBJECT_MAX_BYTES,
                )
            except SecureReadError as exc:
                raise ReviewReuseError("unsafe review object path") from exc
            if not isinstance(source_document, Mapping):
                raise ReviewReuseError("review object is not an object")
            object_path = Path(project_dir) / relative
            total_loaded += object_path.stat(
                follow_symlinks=False,
            ).st_size
            if total_loaded > REVIEW_REUSE_MAX_TOTAL_BYTES:
                raise ReviewReuseError("review loaded-byte bound exceeded")
            if canonical_sha256(source_document) != link["object_sha256"]:
                raise ReviewReuseError("review object content hash mismatch")
            source = ReviewSegmentSource.from_document(source_document)
            if (
                source.identity.sha256
                != str(document.get("identity_sha256") or "")
                or source.identity.segment_id != segment_id
                or source.semantic_content_sha256
                != link["semantic_content_sha256"]
                or object_path.name
                != f"{link['object_sha256']}.json"
            ):
                raise ReviewReuseError("review object identity mismatch")
            loaded_sources.append(source)
            object_hashes.append(source.semantic_content_sha256)
        if len(set(object_hashes)) > REVIEW_REUSE_MAX_SOURCES_PER_SEGMENT:
            raise ReviewReuseError("review acceptance source set is invalid")
        collapsed_sources = tuple({
            source.semantic_content_sha256: source
            for source in loaded_sources
        }.values())
        object_hashes = [
            source.semantic_content_sha256
            for source in collapsed_sources
        ]
        identity = loaded_sources[0].identity
        t15_source_hashes = {
            source.as_t15_source(stable_order=0).semantic_source_sha256
            for source in collapsed_sources
        }
        if not t15_source_hashes.issubset(set(t15["source_hashes"])):
            raise ReviewReuseError(
                "review acceptance source set is not bound to T15"
            )
        source_edges = validate_review_segment_source_set(
            collapsed_sources
        )
        acceptance_edges = tuple(sorted(
            (str(item[0]), str(item[1]))
            for item in document.get("supersession_edges") or []
        ))
        if source_edges != acceptance_edges:
            raise ReviewReuseError(
                "review acceptance supersession binding mismatch"
            )
        t15_edges = {
            (str(item[0]), str(item[1]))
            for item in t15["supersession_edges"]
        }
        if not set(source_edges).issubset(t15_edges):
            raise ReviewReuseError(
                "review acceptance supersession was not applied by T15"
            )
        accepted_item = AcceptedReviewSegment(
            identity=identity,
            source_sha256s=tuple(object_hashes),
            acceptance_sha256=acceptance_sha,
            accepted_merged_segment_sha256=expected_merged,
            reviewed_segment=_json_value(
                output_document["segments"][segment_id]
            ),
        )
        binding = (
            tuple(object_hashes),
            tuple(t15["source_hashes"]),
            t15["semantic_input_sha256"],
            t15["merged_sha256"],
            t15["final_review_sha256"],
            expected_merged,
            source_edges,
        )
        candidates.setdefault(segment_id, []).append(
            (accepted_item, collapsed_sources, binding)
        )
    accepted_result: dict[str, tuple[AcceptedReviewSegment, ...]] = {}
    source_result: dict[str, tuple[ReviewSegmentSource, ...]] = {}
    for segment_id, values in candidates.items():
        exact_values = [
            item for item in values
            if item[0].identity.sha256
            == current_by_segment[segment_id].sha256
        ]
        selected = min(
            exact_values or values,
            key=lambda item: item[0].acceptance_sha256,
        )
        equivalent = [
            item for item in (exact_values or values)
            if item[2] == selected[2]
        ]
        accepted_result[segment_id] = tuple(
            item[0] for item in sorted(
                equivalent, key=lambda item: item[0].acceptance_sha256,
            )
        )
        source_result[segment_id] = selected[1]
    return accepted_result, source_result


def load_review_reuse_receipt(
    project_dir: Path,
    relative_path: Path | str,
) -> Mapping[str, Any]:
    """Load and verify one body-light T16 receipt and its T15/plan links."""

    try:
        receipt_bytes = read_bounded_file(
            Path(project_dir),
            relative_path,
            max_bytes=REVIEW_REUSE_ACCEPTANCE_MAX_BYTES,
            suffixes=(".json",),
        )
        receipt = json.loads(receipt_bytes)
    except (SecureReadError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewReuseError("review reuse receipt is unsafe") from exc
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != _REUSE_RECEIPT_FIELDS
        or receipt.get("schema_version") != REVIEW_REUSE_RECEIPT_VERSION
        or receipt.get("schema_valid") is not True
        or receipt.get("domain_valid") is not True
        or not all(
            _is_sha256(receipt.get(key))
            for key in (
                "plan_sha256",
                "merged_output_sha256",
            )
        )
        or not isinstance(receipt.get("merged_segment_sha256s"), Mapping)
        or any(
            not isinstance(key, str) or not key or not _is_sha256(value)
            for key, value in receipt["merged_segment_sha256s"].items()
        )
    ):
        raise ReviewReuseError("review reuse receipt is malformed")
    plan_relative = Path(str(receipt.get("plan_path") or ""))
    try:
        plan_document = read_bounded_json(
            Path(project_dir),
            plan_relative,
            max_bytes=REVIEW_REUSE_ACCEPTANCE_MAX_BYTES,
        )
    except SecureReadError as exc:
        raise ReviewReuseError("review reuse plan link is unsafe") from exc
    plan = ReviewReusePlan.from_document(
        plan_document,
        expected_sha256=str(receipt["plan_sha256"]),
        require_sealed=True,
    )
    plan_identity_sha256s = [
        item.identity_sha256 for item in plan.entries
    ]
    plan_segment_ids = [item.segment_id for item in plan.entries]
    reused_entries = [
        item for item in plan.entries if item.disposition == "reused"
    ]
    miss_entries = [
        item for item in plan.entries if item.disposition != "reused"
    ]
    if (
        not isinstance(receipt.get("identity_sha256s"), list)
        or receipt["identity_sha256s"] != plan_identity_sha256s
        or set(receipt["merged_segment_sha256s"]) != set(plan_segment_ids)
        or type(receipt.get("new_segment_count")) is not int
        or receipt["new_segment_count"] != len(miss_entries)
        or type(receipt.get("reused_segment_count")) is not int
        or receipt["reused_segment_count"] != len(reused_entries)
        or type(receipt.get("actual_review_calls")) is not int
        or not 0 <= receipt["actual_review_calls"] <= len(plan.chunks)
        or not isinstance(receipt.get("source_sha256s"), list)
        or len(receipt["source_sha256s"])
        != len(set(receipt["source_sha256s"]))
        or any(
            not _is_sha256(item) for item in receipt["source_sha256s"]
        )
        or not isinstance(receipt.get("acceptance_sha256s"), list)
        or len(receipt["acceptance_sha256s"]) != len(plan.entries)
        or len(receipt["acceptance_sha256s"])
        != len(set(receipt["acceptance_sha256s"]))
        or any(
            not _is_sha256(item)
            for item in receipt["acceptance_sha256s"]
        )
    ):
        raise ReviewReuseError("review reuse receipt/plan binding changed")
    t15 = receipt.get("t15_receipt")
    _validate_t15_acceptance_link(
        t15,
        accepted_merged_segment_sha256=next(
            iter(receipt["merged_segment_sha256s"].values()),
            "0" * 64,
        ),
    )
    try:
        t15_bytes = read_bounded_file(
            Path(project_dir),
            Path(str(t15["path"])),
            max_bytes=REVIEW_REUSE_ACCEPTANCE_MAX_BYTES,
            suffixes=(".json",),
        )
        t15_document = json.loads(t15_bytes)
    except (SecureReadError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewReuseError("review reuse T15 link is unsafe") from exc
    if (
        hashlib.sha256(t15_bytes).hexdigest() != t15["sha256"]
        or not isinstance(t15_document, Mapping)
        or t15_document.get("schema_version")
        != REVIEW_ARBITRATION_RECEIPT_VERSION
        or t15_document.get("status") != t15["status"]
        or t15_document.get("unresolved_paths")
        or any(
            t15_document.get(key) != t15.get(key)
            for key in (
                "semantic_input_sha256",
                "source_hashes",
                "merged_sha256",
                "final_review_sha256",
                "reviewed_translation_sha256",
                "reviewed_annotation_sha256",
                "supersession_edges",
            )
        )
    ):
        raise ReviewReuseError("review reuse T15 link mismatch")
    expected_source_sha256s: list[str] = []
    total_loaded = 0
    for entry, acceptance_sha256 in zip(
        plan.entries, receipt["acceptance_sha256s"],
    ):
        acceptance_relative = (
            Path(".arc-companion")
            / "review-segments"
            / "acceptances"
            / (
                f"{entry.identity_sha256}-"
                f"{acceptance_sha256}.json"
            )
        )
        try:
            acceptance_bytes = read_bounded_file(
                Path(project_dir),
                acceptance_relative,
                max_bytes=REVIEW_REUSE_ACCEPTANCE_MAX_BYTES,
                suffixes=(".json",),
            )
            acceptance = json.loads(acceptance_bytes)
        except (
            SecureReadError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ReviewReuseError(
                "review reuse acceptance link is unsafe"
            ) from exc
        total_loaded += len(acceptance_bytes)
        if (
            total_loaded > REVIEW_REUSE_MAX_TOTAL_BYTES
            or not isinstance(acceptance, Mapping)
            or canonical_sha256(acceptance) != acceptance_sha256
            or acceptance.get("schema_version")
            != REVIEW_SEGMENT_ACCEPTANCE_VERSION
            or acceptance.get("identity_sha256")
            != entry.identity_sha256
            or acceptance.get("segment_id") != entry.segment_id
            or acceptance.get("t15_receipt") != t15
            or acceptance.get("accepted_merged_segment_sha256")
            != receipt["merged_segment_sha256s"][entry.segment_id]
            or not isinstance(acceptance.get("object_links"), list)
            or not acceptance["object_links"]
        ):
            raise ReviewReuseError(
                "review reuse acceptance binding mismatch"
            )
        segment_sources: list[str] = []
        for link in acceptance["object_links"]:
            if (
                not isinstance(link, Mapping)
                or set(link) != {
                    "path",
                    "object_sha256",
                    "semantic_content_sha256",
                }
                or not _is_sha256(link.get("object_sha256"))
                or not _is_sha256(
                    link.get("semantic_content_sha256")
                )
            ):
                raise ReviewReuseError(
                    "review reuse object link is malformed"
                )
            try:
                object_bytes = read_bounded_file(
                    Path(project_dir),
                    Path(str(link["path"])),
                    max_bytes=REVIEW_REUSE_OBJECT_MAX_BYTES,
                    suffixes=(".json",),
                )
                object_document = json.loads(object_bytes)
            except (
                SecureReadError,
                UnicodeError,
                json.JSONDecodeError,
            ) as exc:
                raise ReviewReuseError(
                    "review reuse object link is unsafe"
                ) from exc
            total_loaded += len(object_bytes)
            if (
                total_loaded > REVIEW_REUSE_MAX_TOTAL_BYTES
                or not isinstance(object_document, Mapping)
                or canonical_sha256(object_document)
                != link["object_sha256"]
            ):
                raise ReviewReuseError(
                    "review reuse object content changed"
                )
            source = ReviewSegmentSource.from_document(
                object_document
            )
            if (
                source.identity.sha256 != entry.identity_sha256
                or source.identity.segment_id != entry.segment_id
                or source.semantic_content_sha256
                != link["semantic_content_sha256"]
            ):
                raise ReviewReuseError(
                    "review reuse object identity changed"
                )
            segment_sources.append(
                source.semantic_content_sha256
            )
        if segment_sources != sorted(set(segment_sources)):
            raise ReviewReuseError(
                "review reuse segment source order changed"
            )
        expected_source_sha256s.extend(segment_sources)
    if receipt["source_sha256s"] != expected_source_sha256s:
        raise ReviewReuseError(
            "review reuse source receipt set changed"
        )
    return _json_value(dict(receipt))
