"""Deterministic atomization and arbitration of complete Review responses."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

from arc_llm import strip_arc_llm_call_records
from jsonschema import ValidationError, validate as validate_json_schema
from .prompts import (
    COMMENTARY_REVIEW_SCHEMA,
    REVIEW_SCHEMA,
    SECTION_REVIEW_SCHEMA,
)


REVIEW_ARBITRATION_VERSION = "arc.companion.review-arbitration.v1"
REVIEW_ARBITRATION_OUTPUT_SCHEMA_VERSION = (
    "arc.companion.review-arbitration-output-schema.v1"
)
REVIEW_ARBITRATION_RECEIPT_VERSION = (
    "arc.companion.review-arbitration-receipt.v1"
)
REVIEW_ARBITRATION_PARTIAL_VERSION = (
    "arc.companion.review-arbitration-partial.v1"
)
REVIEW_ARBITRATION_TIER = "low"

ANNOTATION_FIELDS = (
    "commentary",
    "explanation",
    "commentary_sources",
    "prior_work",
    "later_work",
)
REVIEW_PATCH_FIELDS = (
    "segment_id",
    "translation_blocks",
    *ANNOTATION_FIELDS,
    "reason",
)
_SHA256_HEX = frozenset("0123456789abcdef")
_SOURCE_FACTORY_TOKEN = object()


def _source_citation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["title", "url", "locator"],
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "url": {"type": "string", "minLength": 1, "pattern": "^https?://"},
            "locator": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }


def _related_work_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "maxItems": 3,
        "items": {
            "type": "object",
            "required": ["text", "sources"],
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "sources": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": _source_citation_schema(),
                },
            },
            "additionalProperties": False,
        },
    }


ARBITRATION_REPLACEMENT_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": list(REVIEW_PATCH_FIELDS),
    "properties": {
        "segment_id": {"type": "string", "minLength": 1},
        "translation_blocks": {
            "type": ["array", "null"],
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "type": "object",
                "required": ["block_id", "text"],
                "properties": {
                    "block_id": {"type": "string", "minLength": 1},
                    "text": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "commentary": {"type": ["string", "null"]},
        "explanation": {"type": ["string", "null"]},
        "commentary_sources": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "maxItems": 3,
                    "items": _source_citation_schema(),
                },
            ],
        },
        "prior_work": {
            "oneOf": [{"type": "null"}, _related_work_schema()],
        },
        "later_work": {
            "oneOf": [{"type": "null"}, _related_work_schema()],
        },
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}


REVIEW_ARBITRATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "path",
                    "action",
                    "selected_candidate_hashes",
                    "replacement_patch",
                    "reason",
                ],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "action": {
                        "type": "string",
                        "enum": [
                            "select_candidate",
                            "merge_candidates",
                            "keep_original",
                            "unresolved",
                        ],
                    },
                    "selected_candidate_hashes": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                    },
                    "replacement_patch": {
                        "oneOf": [
                            {"type": "null"},
                            ARBITRATION_REPLACEMENT_PATCH_SCHEMA,
                        ],
                    },
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

def _validate_complete_review_source(
    review: Mapping[str, Any],
    *,
    source_kind: str,
    segment_set: Sequence[str] | None,
) -> None:
    source_kind = str(source_kind)
    schemas = {
        "section": SECTION_REVIEW_SCHEMA,
        "final": REVIEW_SCHEMA,
        "commentary": COMMENTARY_REVIEW_SCHEMA,
    }
    if source_kind not in schemas:
        raise ReviewArbitrationError(
            f"unknown review source kind: {source_kind}"
        )
    try:
        validate_json_schema(instance=review, schema=schemas[source_kind])
    except ValidationError as exc:
        raise ReviewArbitrationError(
            f"review source schema is invalid: {exc.message}"
        ) from exc
    if source_kind == "final" and segment_set is not None:
        raise ReviewArbitrationError(
            "final review source must not declare a segment set"
        )
    if source_kind in {"section", "commentary"}:
        canonical_segment_set = sorted(str(item) for item in (segment_set or ()))
        if (
            not canonical_segment_set
            or len(canonical_segment_set) != len(set(canonical_segment_set))
            or (
                source_kind == "section"
                and sorted(review["reviewed_segment_ids"])
                != canonical_segment_set
            )
        ):
            raise ReviewArbitrationError(
                f"{source_kind} review source does not match its segment set"
            )
        if source_kind == "section" and any(
            str(item["segment_id"]) not in set(canonical_segment_set)
            for item in review["findings"]
        ):
            raise ReviewArbitrationError(
                "section review finding is outside its segment set"
            )
        if any(
            str(item["segment_id"]) not in set(canonical_segment_set)
            for item in review["patches"]
        ):
            raise ReviewArbitrationError(
                f"{source_kind} review patch is outside its segment set"
            )


class ReviewArbitrationError(RuntimeError):
    """Review candidate identities or arbitration output are invalid."""


class ReviewArbitrationNeedsSupervision(RuntimeError):
    """Exact Review targets require operator resolution."""

    def __init__(
        self,
        *,
        paths: Sequence[str],
        reason: str,
        partial_path: str | None = None,
        receipt_path: str | None = None,
        submission_state: str = "not_submitted",
    ) -> None:
        self.paths = tuple(paths)
        self.reason = str(reason)
        self.partial_path = partial_path
        self.receipt_path = receipt_path
        self.recovery_context = {
            "submission_state": str(submission_state),
            "resumable": False,
            "recovery_action": "operator-supervision",
            "blocked_reason": self.reason,
            "review_arbitration_paths": list(self.paths),
            **({"partial_path": partial_path} if partial_path else {}),
            **({"receipt_path": receipt_path} if receipt_path else {}),
        }
        super().__init__(
            f"review arbitration needs supervision for {list(self.paths)}: "
            f"{self.reason}"
        )


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_pointer_escape(value: str) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def json_pointer_unescape(value: str) -> str:
    return str(value).replace("~1", "/").replace("~0", "~")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - _SHA256_HEX)
    )


@dataclass(frozen=True)
class ReviewPatchSource:
    """One complete Review response with only semantic source identity."""

    source_id: str
    source_kind: str
    semantic_source_sha256: str
    stable_order: int
    _patches_json: str = field(repr=False)
    _findings_json: str = field(repr=False)
    _issues_json: str = field(repr=False)
    _factory_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _SOURCE_FACTORY_TOKEN:
            raise ReviewArbitrationError(
                "ReviewPatchSource must be constructed with from_review"
            )
        if not self.source_id or not self.source_kind:
            raise ReviewArbitrationError("review source identity is empty")
        if not _is_sha256(self.semantic_source_sha256):
            raise ReviewArbitrationError("review source semantic hash is invalid")
        if (
            type(self.stable_order) is not int
            or self.stable_order < 0
        ):
            raise ReviewArbitrationError("review source stable order is invalid")

    @property
    def patches(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(json.loads(self._patches_json))

    @property
    def findings(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(json.loads(self._findings_json))

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(json.loads(self._issues_json))

    @classmethod
    def from_review(
        cls,
        *,
        source_kind: str,
        stable_order: int,
        review: Mapping[str, Any],
        segment_set: Sequence[str] | None = None,
        source_id: str | None = None,
    ) -> "ReviewPatchSource":
        clean = strip_arc_llm_call_records(dict(review))
        if not isinstance(clean, Mapping):
            raise ReviewArbitrationError("review source response is malformed")
        clean = _json_value(clean)
        _validate_complete_review_source(
            clean,
            source_kind=source_kind,
            segment_set=segment_set,
        )
        patches = clean.get("patches")
        if not isinstance(patches, list):
            raise ReviewArbitrationError("review source patches are malformed")
        findings = clean.get("findings") or []
        issues = clean.get("issues") or []
        if not isinstance(findings, list) or not isinstance(issues, list):
            raise ReviewArbitrationError("review source findings/issues are malformed")
        canonical_segment_set: list[str] | None = None
        if segment_set is None:
            identity_payload: Any = clean
        else:
            canonical_segment_set = sorted(str(item) for item in segment_set)
            identity_payload = {
                "segment_set_sha256": canonical_sha256(canonical_segment_set),
                "complete_response_sha256": canonical_sha256(clean),
            }
        semantic_source_sha256 = canonical_sha256({
            "source_kind": str(source_kind),
            "identity": identity_payload,
        })
        derived_source_id = f"{str(source_kind)}:{semantic_source_sha256}"
        if source_id is not None and source_id != derived_source_id:
            raise ReviewArbitrationError(
                "review source id does not match its semantic identity"
            )
        return cls(
            source_id=derived_source_id,
            source_kind=str(source_kind),
            semantic_source_sha256=semantic_source_sha256,
            stable_order=stable_order,
            _patches_json=canonical_json(patches),
            _findings_json=canonical_json(findings),
            _issues_json=canonical_json(issues),
            _factory_token=_SOURCE_FACTORY_TOKEN,
        )


@dataclass(frozen=True)
class ReviewAtomOrigin:
    source_id: str
    source_kind: str
    semantic_source_sha256: str
    stable_order: int
    original_ordinal: int
    patch_sha256: str
    origin_sha256: str
    reason: str

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "semantic_source_sha256": self.semantic_source_sha256,
            "original_ordinal": self.original_ordinal,
            "patch_sha256": self.patch_sha256,
            "origin_sha256": self.origin_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReviewAtom:
    path: str
    segment_id: str
    target_kind: str
    target_id: str
    replacement: Any
    candidate_sha256: str
    origin: ReviewAtomOrigin

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "segment_id": self.segment_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "replacement": deepcopy(self.replacement),
            "candidate_sha256": self.candidate_sha256,
            "origin": self.origin.semantic_payload(),
        }


@dataclass(frozen=True)
class CanonicalCandidate:
    path: str
    segment_id: str
    target_kind: str
    target_id: str
    replacement: Any
    candidate_sha256: str
    origins: tuple[ReviewAtomOrigin, ...]

    @property
    def presentation_origins(self) -> tuple[ReviewAtomOrigin, ...]:
        return tuple(sorted(
            self.origins,
            key=lambda origin: (
                origin.stable_order,
                origin.original_ordinal,
                origin.origin_sha256,
            ),
        ))

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            origin.reason
            for origin in self.presentation_origins if origin.reason
        ))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "segment_id": self.segment_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "replacement": deepcopy(self.replacement),
            "candidate_sha256": self.candidate_sha256,
            "origins": [
                item.semantic_payload()
                for item in sorted(
                    self.origins,
                    key=lambda origin: origin.origin_sha256,
                )
            ],
        }


@dataclass(frozen=True)
class ReviewConflictComponent:
    path: str
    candidates: tuple[CanonicalCandidate, ...]
    edges: tuple[tuple[str, str], ...] = ()
    pruned_candidate_hashes: tuple[str, ...] = ()

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "candidates": [item.semantic_payload() for item in self.candidates],
            "edges": [list(item) for item in self.edges],
            "pruned_candidate_hashes": list(self.pruned_candidate_hashes),
        }


@dataclass(frozen=True)
class ReviewMergePlan:
    atoms: tuple[ReviewAtom, ...]
    canonical_groups: tuple[CanonicalCandidate, ...]
    components: tuple[ReviewConflictComponent, ...]
    non_conflicting_atoms: tuple[CanonicalCandidate, ...]
    sources: tuple[ReviewPatchSource, ...]
    findings: tuple[Mapping[str, Any], ...]
    issues: tuple[str, ...]
    source_hashes: tuple[str, ...]
    finding_hashes: tuple[str, ...]
    issue_hashes: tuple[str, ...]
    supersession_edges: tuple[tuple[str, str], ...]
    pruned_candidate_hashes: tuple[str, ...]
    _original_targets_json: str = field(repr=False)
    _invariant_contexts_json: str = field(repr=False)
    semantic_input: Mapping[str, Any]
    semantic_input_sha256: str

    @property
    def original_targets(self) -> Mapping[str, Any]:
        return json.loads(self._original_targets_json)

    @property
    def invariant_contexts(self) -> Mapping[str, Mapping[str, Any]]:
        return json.loads(self._invariant_contexts_json)

    @property
    def conflict_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.components)


@dataclass(frozen=True)
class ArbitrationDecision:
    path: str
    action: str
    selected_candidate_hashes: tuple[str, ...]
    replacement_patch: Mapping[str, Any] | None
    reason: str
    replacement_value: Any = None


@dataclass(frozen=True)
class ArbitrationResolution:
    decisions: tuple[ArbitrationDecision, ...]
    resolved_candidates: tuple[CanonicalCandidate, ...]
    unresolved_paths: tuple[str, ...]
    output_sha256: str


def _normalize_supersession_edges(
    value: (
        Mapping[str, Sequence[str]]
        | Sequence[tuple[str, str]]
        | None
    ),
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    edges: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for winner, losers in value.items():
            if isinstance(losers, str) or not isinstance(losers, Sequence):
                raise ReviewArbitrationError("supersession targets are malformed")
            edges.extend((str(winner), str(loser)) for loser in losers)
    else:
        for edge in value:
            if (
                not isinstance(edge, Sequence)
                or isinstance(edge, str)
                or len(edge) != 2
            ):
                raise ReviewArbitrationError("supersession edge is malformed")
            edges.append((str(edge[0]), str(edge[1])))
    return tuple(sorted(set(edges)))


def _target_path(
    segment_id: str,
    *,
    target_kind: str,
    target_id: str,
) -> str:
    escaped_segment = json_pointer_escape(segment_id)
    escaped_target = json_pointer_escape(target_id)
    if target_kind == "annotation":
        return f"/segments/{escaped_segment}/annotation/{escaped_target}"
    if target_kind == "translation_block":
        return (
            f"/segments/{escaped_segment}/translation_blocks/{escaped_target}"
        )
    raise ReviewArbitrationError(f"unknown Review target kind {target_kind!r}")


def _patch_origin(
    source: ReviewPatchSource,
    *,
    ordinal: int,
    patch: Mapping[str, Any],
) -> ReviewAtomOrigin:
    clean_patch = strip_arc_llm_call_records(dict(patch))
    patch_sha256 = canonical_sha256(clean_patch)
    origin_sha256 = canonical_sha256({
        "semantic_source_sha256": source.semantic_source_sha256,
        "original_ordinal": ordinal,
        "patch_sha256": patch_sha256,
    })
    return ReviewAtomOrigin(
        source_id=source.source_id,
        source_kind=source.source_kind,
        semantic_source_sha256=source.semantic_source_sha256,
        stable_order=source.stable_order,
        original_ordinal=ordinal,
        patch_sha256=patch_sha256,
        origin_sha256=origin_sha256,
        reason=str(clean_patch.get("reason") or ""),
    )


def atomize_review_sources(
    sources: Iterable[ReviewPatchSource],
    *,
    segment_order: Sequence[str],
    block_order_by_segment: Mapping[str, Sequence[str]],
    skip_translation: bool = False,
) -> tuple[ReviewAtom, ...]:
    ordered_sources = tuple(sorted(
        sources,
        key=lambda item: (
            item.semantic_source_sha256,
            item.source_kind,
            item.source_id,
        ),
    ))
    segment_ids = tuple(str(item) for item in segment_order)
    if (
        any(not item for item in segment_ids)
        or len(segment_ids) != len(set(segment_ids))
    ):
        raise ReviewArbitrationError("segment order is invalid")
    segment_set = set(segment_ids)
    owned_blocks = {
        str(segment_id): tuple(str(item) for item in block_ids)
        for segment_id, block_ids in block_order_by_segment.items()
    }
    for segment_id in segment_ids:
        blocks = owned_blocks.get(segment_id, ())
        if (
            any(not item for item in blocks)
            or len(blocks) != len(set(blocks))
        ):
            raise ReviewArbitrationError(
                f"translation block ownership is invalid for {segment_id}"
            )

    atoms: list[ReviewAtom] = []
    for source in ordered_sources:
        for ordinal, raw_patch in enumerate(source.patches):
            if not isinstance(raw_patch, Mapping):
                raise ReviewArbitrationError("Review patch is not an object")
            patch = strip_arc_llm_call_records(dict(raw_patch))
            expected_fields = (
                set(REVIEW_PATCH_FIELDS) - {"translation_blocks"}
                if source.source_kind == "commentary"
                else set(REVIEW_PATCH_FIELDS)
            )
            if set(patch) != expected_fields:
                raise ReviewArbitrationError(
                    "Review patch fields do not match the closed contract"
                )
            segment_id = patch.get("segment_id")
            if (
                not isinstance(segment_id, str)
                or not segment_id
                or segment_id not in segment_set
            ):
                raise ReviewArbitrationError(
                    f"Review patch has an unknown or empty segment id: "
                    f"{segment_id!r}"
                )
            origin = _patch_origin(source, ordinal=ordinal, patch=patch)
            replacement_count = 0
            translation_blocks = patch.get("translation_blocks")
            if translation_blocks is not None:
                if skip_translation:
                    raise ReviewArbitrationError(
                        "translation patches are forbidden in skip-translation mode"
                    )
                if not isinstance(translation_blocks, list) or not translation_blocks:
                    raise ReviewArbitrationError(
                        "translation_blocks must be null or a non-empty array"
                    )
                seen_blocks: set[str] = set()
                for raw_block in translation_blocks:
                    if (
                        not isinstance(raw_block, Mapping)
                        or set(raw_block) != {"block_id", "text"}
                    ):
                        raise ReviewArbitrationError(
                            "translation replacement block is malformed"
                        )
                    block_id = raw_block.get("block_id")
                    if (
                        not isinstance(block_id, str)
                        or not block_id
                        or block_id in seen_blocks
                    ):
                        raise ReviewArbitrationError(
                            "translation replacement block identity is empty or duplicate"
                        )
                    if block_id not in set(owned_blocks.get(segment_id, ())):
                        raise ReviewArbitrationError(
                            f"translation block {block_id} is not owned by {segment_id}"
                        )
                    seen_blocks.add(block_id)
                    replacement = deepcopy(dict(raw_block))
                    path = _target_path(
                        segment_id,
                        target_kind="translation_block",
                        target_id=block_id,
                    )
                    atoms.append(ReviewAtom(
                        path=path,
                        segment_id=segment_id,
                        target_kind="translation_block",
                        target_id=block_id,
                        replacement=replacement,
                        candidate_sha256=canonical_sha256({
                            "path": path,
                            "replacement": replacement,
                        }),
                        origin=origin,
                    ))
                    replacement_count += 1
            for field in ANNOTATION_FIELDS:
                replacement = patch.get(field)
                if replacement is None:
                    continue
                path = _target_path(
                    segment_id,
                    target_kind="annotation",
                    target_id=field,
                )
                atoms.append(ReviewAtom(
                    path=path,
                    segment_id=segment_id,
                    target_kind="annotation",
                    target_id=field,
                    replacement=deepcopy(replacement),
                    candidate_sha256=canonical_sha256({
                        "path": path,
                        "replacement": replacement,
                    }),
                    origin=origin,
                ))
                replacement_count += 1
            if replacement_count == 0:
                raise ReviewArbitrationError(
                    f"Review patch for {segment_id} has no replacement target"
                )
    return tuple(atoms)


def _candidate_order_key(
    candidate: CanonicalCandidate | ReviewAtom,
    *,
    segment_rank: Mapping[str, int],
    block_rank: Mapping[str, Mapping[str, int]],
) -> tuple[Any, ...]:
    segment_id = candidate.segment_id
    if candidate.target_kind == "translation_block":
        field_rank = 0
        target_rank = block_rank.get(segment_id, {}).get(
            candidate.target_id, 10**9,
        )
    else:
        field_rank = 1 + ANNOTATION_FIELDS.index(candidate.target_id)
        target_rank = 0
    origin_key: tuple[Any, ...] = ()
    if isinstance(candidate, ReviewAtom):
        origin_key = (
            candidate.origin.original_ordinal,
            candidate.origin.origin_sha256,
        )
    return (
        segment_rank.get(segment_id, 10**9),
        field_rank,
        target_rank,
        candidate.path,
        candidate.candidate_sha256,
        *origin_key,
    )


def _canonicalize_atoms(
    atoms: Sequence[ReviewAtom],
    *,
    segment_order: Sequence[str],
    block_order_by_segment: Mapping[str, Sequence[str]],
) -> tuple[CanonicalCandidate, ...]:
    segment_rank = {
        str(segment_id): index for index, segment_id in enumerate(segment_order)
    }
    block_rank = {
        str(segment_id): {
            str(block_id): index for index, block_id in enumerate(block_ids)
        }
        for segment_id, block_ids in block_order_by_segment.items()
    }
    ordered_atoms = sorted(
        atoms,
        key=lambda item: _candidate_order_key(
            item, segment_rank=segment_rank, block_rank=block_rank,
        ),
    )
    by_identity: dict[tuple[str, str], list[ReviewAtom]] = {}
    for atom in ordered_atoms:
        by_identity.setdefault(
            (atom.path, atom.candidate_sha256), [],
        ).append(atom)
    groups = [
        CanonicalCandidate(
            path=items[0].path,
            segment_id=items[0].segment_id,
            target_kind=items[0].target_kind,
            target_id=items[0].target_id,
            replacement=deepcopy(items[0].replacement),
            candidate_sha256=items[0].candidate_sha256,
            origins=tuple(item.origin for item in items),
        )
        for items in by_identity.values()
    ]
    return tuple(sorted(
        groups,
        key=lambda item: _candidate_order_key(
            item, segment_rank=segment_rank, block_rank=block_rank,
        ),
    ))


def _validated_supersession(
    groups: Sequence[CanonicalCandidate],
    edges: Sequence[tuple[str, str]],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    by_hash = {item.candidate_sha256: item for item in groups}
    adjacency: dict[str, set[str]] = {value: set() for value in by_hash}
    validated: list[tuple[str, str]] = []
    for winner, loser in edges:
        if winner not in by_hash or loser not in by_hash:
            raise ReviewArbitrationError(
                "supersession references an unknown candidate hash"
            )
        if winner == loser:
            raise ReviewArbitrationError("supersession cannot reference itself")
        if by_hash[winner].path != by_hash[loser].path:
            raise ReviewArbitrationError(
                "supersession cannot cross Review target paths"
            )
        edge = (winner, loser)
        if edge not in validated:
            validated.append(edge)
            adjacency[winner].add(loser)

    state: dict[str, int] = {}

    def visit(node: str) -> None:
        marker = state.get(node, 0)
        if marker == 1:
            raise ReviewArbitrationError("supersession graph contains a cycle")
        if marker == 2:
            return
        state[node] = 1
        for target in adjacency[node]:
            visit(target)
        state[node] = 2

    for candidate_hash in adjacency:
        visit(candidate_hash)
    pruned = tuple(sorted({loser for _, loser in validated}))
    return tuple(validated), pruned


def plan_review_merge(
    sources: Iterable[ReviewPatchSource],
    *,
    segment_order: Sequence[str],
    block_order_by_segment: Mapping[str, Sequence[str]],
    skip_translation: bool = False,
    contract_versions: Mapping[str, str] | None = None,
    provider: str = "auto",
    model: str | None = None,
    primary_tier: str = "medium",
    arbitration_tier: str = REVIEW_ARBITRATION_TIER,
    tool_policy: Mapping[str, Any] | None = None,
    original_value_resolver: Callable[[str], Any],
    invariant_context_resolver: (
        Callable[[ReviewConflictComponent], Mapping[str, Any]]
    ),
    controller_supersession_edges: (
        Mapping[str, Sequence[str]]
        | Sequence[tuple[str, str]]
        | None
    ) = None,
    semantic_context: Mapping[str, Any] | None = None,
) -> ReviewMergePlan:
    semantic_sources = tuple(sorted(
        sources,
        key=lambda item: (
            item.semantic_source_sha256,
            item.source_kind,
            item.source_id,
        ),
    ))
    presentation_sources = tuple(sorted(
        semantic_sources,
        key=lambda item: (
            item.stable_order,
            item.semantic_source_sha256,
        ),
    ))
    atoms = atomize_review_sources(
        semantic_sources,
        segment_order=segment_order,
        block_order_by_segment=block_order_by_segment,
        skip_translation=skip_translation,
    )
    groups = _canonicalize_atoms(
        atoms,
        segment_order=segment_order,
        block_order_by_segment=block_order_by_segment,
    )
    all_edges = _normalize_supersession_edges(
        controller_supersession_edges
    )
    validated_edges, pruned = _validated_supersession(groups, all_edges)
    pruned_set = set(pruned)
    active_groups = [
        item for item in groups if item.candidate_sha256 not in pruned_set
    ]
    by_path: dict[str, list[CanonicalCandidate]] = {}
    for group in active_groups:
        by_path.setdefault(group.path, []).append(group)
    components: list[ReviewConflictComponent] = []
    non_conflicting: list[CanonicalCandidate] = []
    for path, path_groups in by_path.items():
        path_edges = tuple(
            edge
            for edge in validated_edges
            if next(
                item for item in groups
                if item.candidate_sha256 == edge[0]
            ).path == path
        )
        path_pruned = tuple(
            item.candidate_sha256
            for item in groups
            if item.path == path and item.candidate_sha256 in pruned_set
        )
        if len(path_groups) == 1:
            non_conflicting.append(path_groups[0])
        else:
            components.append(ReviewConflictComponent(
                path=path,
                candidates=tuple(path_groups),
                edges=path_edges,
                pruned_candidate_hashes=path_pruned,
            ))

    findings = tuple(
        deepcopy(dict(item)) if isinstance(item, Mapping) else item
        for source in presentation_sources
        for item in source.findings
    )
    issues = tuple(
        str(item)
        for source in presentation_sources
        for item in source.issues
    )
    source_hashes = tuple(sorted(
        source.semantic_source_sha256 for source in semantic_sources
    ))
    finding_hashes = tuple(canonical_sha256(item) for item in findings)
    issue_hashes = tuple(canonical_sha256(item) for item in issues)
    active_components = tuple(
        ReviewConflictComponent(path=path, candidates=tuple(path_groups))
        for path, path_groups in by_path.items()
    )
    original_targets = {
        component.path: _json_value(original_value_resolver(component.path))
        for component in active_components
    }
    invariant_contexts = {
        component.path: _json_value(dict(invariant_context_resolver(component)))
        for component in active_components
    }
    original_target_hashes = {
        path: canonical_sha256(value)
        for path, value in original_targets.items()
    }
    invariant_hashes = {
        path: canonical_sha256(value)
        for path, value in invariant_contexts.items()
    }
    semantic_input = {
        "schema_version": REVIEW_ARBITRATION_VERSION,
        "contract_versions": dict(sorted((contract_versions or {}).items())),
        "mode": "skip_translation" if skip_translation else "translation_enabled",
        "segment_order": [str(item) for item in segment_order],
        "block_order_by_segment": {
            str(key): [str(item) for item in value]
            for key, value in sorted(block_order_by_segment.items())
        },
        "source_hashes": list(source_hashes),
        "finding_hashes": sorted(finding_hashes),
        "issue_hashes": sorted(issue_hashes),
        "candidates": [item.semantic_payload() for item in groups],
        "components": [item.semantic_payload() for item in components],
        "ordered_paths": [item.path for item in components],
        "original_target_hashes": original_target_hashes,
        "invariant_hashes": invariant_hashes,
        "supersession_edges": [list(item) for item in validated_edges],
        "pruned_candidate_hashes": list(pruned),
        "provider": str(provider),
        "model": model,
        "primary_tier": str(primary_tier),
        "arbitration_tier": str(arbitration_tier),
        "tool_policy": dict(tool_policy or {
            "allow_internet": False,
            "paper_access": "none",
            "inherit_host_tools": False,
            "provider_tools": [],
            "session_policy": "stateless",
        }),
        "semantic_context": _json_value(dict(semantic_context or {})),
    }
    return ReviewMergePlan(
        atoms=atoms,
        canonical_groups=groups,
        components=tuple(components),
        non_conflicting_atoms=tuple(non_conflicting),
        sources=presentation_sources,
        findings=findings,
        issues=issues,
        source_hashes=source_hashes,
        finding_hashes=finding_hashes,
        issue_hashes=issue_hashes,
        supersession_edges=validated_edges,
        pruned_candidate_hashes=pruned,
        _original_targets_json=canonical_json(original_targets),
        _invariant_contexts_json=canonical_json(invariant_contexts),
        semantic_input=semantic_input,
        semantic_input_sha256=canonical_sha256(semantic_input),
    )


def apply_non_conflicting_atoms(
    plan: ReviewMergePlan,
    baseline: Any,
    *,
    apply_atom: Callable[[Any, CanonicalCandidate], None],
    candidate_validator: (
        Callable[[Any, CanonicalCandidate], None] | None
    ) = None,
) -> tuple[Any, tuple[CanonicalCandidate, ...]]:
    """Trial-apply each unique atom, returning a deep-copied partial result."""

    output = deepcopy(baseline)
    invalid: list[CanonicalCandidate] = []
    for candidate in plan.non_conflicting_atoms:
        trial = deepcopy(baseline)
        try:
            apply_atom(trial, candidate)
            if candidate_validator is not None:
                candidate_validator(trial, candidate)
        except (RuntimeError, TypeError, ValueError):
            invalid.append(candidate)
            continue
        apply_atom(output, candidate)
    return output, tuple(invalid)


def trial_validate_candidates(
    plan: ReviewMergePlan,
    baseline: Any,
    *,
    apply_atom: Callable[[Any, CanonicalCandidate], None],
    candidate_validator: (
        Callable[[Any, CanonicalCandidate], None] | None
    ) = None,
) -> tuple[CanonicalCandidate, ...]:
    """Validate every active candidate independently on the original baseline."""

    invalid: list[CanonicalCandidate] = []
    active = [
        *plan.non_conflicting_atoms,
        *(
            candidate
            for component in plan.components
            for candidate in component.candidates
        ),
    ]
    for candidate in active:
        trial = deepcopy(baseline)
        try:
            apply_atom(trial, candidate)
            if candidate_validator is not None:
                candidate_validator(trial, candidate)
        except (RuntimeError, TypeError, ValueError):
            invalid.append(candidate)
    return tuple(invalid)


def arbitration_payload(
    plan: ReviewMergePlan,
) -> dict[str, Any]:
    """Build the conflict-only, body-bounded semantic provider payload."""

    findings_by_segment: dict[str, list[Any]] = {}
    for finding in plan.findings:
        if isinstance(finding, Mapping):
            findings_by_segment.setdefault(
                str(finding.get("segment_id") or ""), [],
            ).append(deepcopy(dict(finding)))
    conflicts = []
    for component in plan.components:
        first = component.candidates[0]
        conflicts.append({
            "path": component.path,
            "original": deepcopy(plan.original_targets[component.path]),
            "candidates": [
                {
                    "candidate_sha256": candidate.candidate_sha256,
                    "replacement_patch": candidate_to_patch(candidate),
                    "reasons": list(candidate.reasons),
                    "origin_hashes": [
                        item.origin_sha256 for item in candidate.origins
                    ],
                }
                for candidate in component.candidates
            ],
            "related_findings": findings_by_segment.get(first.segment_id, []),
            "invariant_context": deepcopy(
                plan.invariant_contexts[component.path]
            ),
        })
    return {
        "schema_version": REVIEW_ARBITRATION_VERSION,
        "semantic_input_sha256": plan.semantic_input_sha256,
        "conflicts": conflicts,
    }


def empty_review_patch(segment_id: str, *, reason: str = "") -> dict[str, Any]:
    return {
        "segment_id": str(segment_id),
        "translation_blocks": None,
        **{field: None for field in ANNOTATION_FIELDS},
        "reason": str(reason),
    }


def candidate_to_patch(
    candidate: CanonicalCandidate,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    patch = empty_review_patch(
        candidate.segment_id,
        reason=(
            "; ".join(candidate.reasons)
            if reason is None else reason
        ),
    )
    if candidate.target_kind == "translation_block":
        patch["translation_blocks"] = [deepcopy(candidate.replacement)]
    else:
        patch[candidate.target_id] = deepcopy(candidate.replacement)
    return patch


def _decision_target_value(
    patch: Mapping[str, Any],
    component: ReviewConflictComponent,
) -> Any:
    candidate = component.candidates[0]
    if str(patch.get("segment_id") or "") != candidate.segment_id:
        raise ReviewArbitrationError(
            f"arbitration replacement targets the wrong segment for {component.path}"
        )
    non_null = [
        field
        for field in ("translation_blocks", *ANNOTATION_FIELDS)
        if patch.get(field) is not None
    ]
    if candidate.target_kind == "translation_block":
        if non_null != ["translation_blocks"]:
            raise ReviewArbitrationError(
                f"translation arbitration replacement targets more than {component.path}"
            )
        blocks = patch.get("translation_blocks")
        if (
            not isinstance(blocks, list)
            or len(blocks) != 1
            or not isinstance(blocks[0], Mapping)
            or str(blocks[0].get("block_id") or "") != candidate.target_id
        ):
            raise ReviewArbitrationError(
                f"translation arbitration replacement targets the wrong block for "
                f"{component.path}"
            )
        return deepcopy(dict(blocks[0]))
    if non_null != [candidate.target_id]:
        raise ReviewArbitrationError(
            f"annotation arbitration replacement targets more than {component.path}"
        )
    return deepcopy(patch[candidate.target_id])


def validate_arbitration_output(
    output: Mapping[str, Any],
    plan: ReviewMergePlan,
    *,
    synthesized_candidate_validator: (
        Callable[[ReviewConflictComponent, Any], None] | None
    ) = None,
) -> ArbitrationResolution:
    clean = strip_arc_llm_call_records(dict(output))
    try:
        validate_json_schema(instance=clean, schema=REVIEW_ARBITRATION_SCHEMA)
    except ValidationError as exc:
        raise ReviewArbitrationError(
            f"review arbitration output schema is invalid: {exc.message}"
        ) from exc
    decisions = clean["decisions"]
    paths = [str(item["path"]) for item in decisions]
    expected_paths = [item.path for item in plan.components]
    if len(paths) != len(set(paths)):
        raise ReviewArbitrationError(
            "review arbitration output contains duplicate paths"
        )
    if set(paths) != set(expected_paths) or len(paths) != len(expected_paths):
        raise ReviewArbitrationError(
            "review arbitration output paths do not match the conflict set"
        )
    by_path = {item.path: item for item in plan.components}
    resolved: list[CanonicalCandidate] = []
    unresolved: list[str] = []
    normalized: list[ArbitrationDecision] = []
    for raw in sorted(decisions, key=lambda item: expected_paths.index(item["path"])):
        path = str(raw["path"])
        component = by_path[path]
        candidates = {
            item.candidate_sha256: item for item in component.candidates
        }
        raw_selected = tuple(
            str(item) for item in raw["selected_candidate_hashes"]
        )
        if any(item not in candidates for item in raw_selected):
            raise ReviewArbitrationError(
                f"review arbitration references a foreign candidate for {path}"
            )
        selected_set = set(raw_selected)
        selected = tuple(
            candidate.candidate_sha256
            for candidate in component.candidates
            if candidate.candidate_sha256 in selected_set
        )
        action = str(raw["action"])
        patch = raw.get("replacement_patch")
        replacement_value = None
        if action == "select_candidate":
            if len(selected) != 1 or not isinstance(patch, Mapping):
                raise ReviewArbitrationError(
                    f"select_candidate is malformed for {path}"
                )
            replacement_value = _decision_target_value(patch, component)
            selected_candidate = candidates[selected[0]]
            if canonical_sha256(replacement_value) != canonical_sha256(
                selected_candidate.replacement
            ):
                raise ReviewArbitrationError(
                    f"selected replacement differs from its candidate for {path}"
                )
            resolved.append(selected_candidate)
        elif action == "merge_candidates":
            if len(selected) < 2 or not isinstance(patch, Mapping):
                raise ReviewArbitrationError(
                    f"merge_candidates must cite at least two candidates for {path}"
                )
            replacement_value = _decision_target_value(patch, component)
            if synthesized_candidate_validator is not None:
                synthesized_candidate_validator(component, replacement_value)
            first = component.candidates[0]
            merge_hash = canonical_sha256({
                "path": path,
                "replacement": replacement_value,
            })
            merged_origins = tuple(
                origin
                for candidate_hash in selected
                for origin in candidates[candidate_hash].origins
            )
            resolved.append(CanonicalCandidate(
                path=path,
                segment_id=first.segment_id,
                target_kind=first.target_kind,
                target_id=first.target_id,
                replacement=deepcopy(replacement_value),
                candidate_sha256=merge_hash,
                origins=merged_origins,
            ))
        elif action in {"keep_original", "unresolved"}:
            if selected or patch is not None:
                raise ReviewArbitrationError(
                    f"{action} must not select or replace a candidate for {path}"
                )
            if action == "unresolved":
                unresolved.append(path)
        else:  # The closed JSON schema should make this unreachable.
            raise ReviewArbitrationError(
                f"unknown arbitration action for {path}: {action}"
            )
        normalized.append(ArbitrationDecision(
            path=path,
            action=action,
            selected_candidate_hashes=selected,
            replacement_patch=deepcopy(dict(patch)) if isinstance(patch, Mapping) else None,
            reason=str(raw["reason"]),
            replacement_value=deepcopy(replacement_value),
        ))
    return ArbitrationResolution(
        decisions=tuple(normalized),
        resolved_candidates=tuple(resolved),
        unresolved_paths=tuple(unresolved),
        output_sha256=canonical_sha256(clean),
    )


def materialize_review_patches(
    candidates: Iterable[CanonicalCandidate],
    *,
    segment_order: Sequence[str],
    original_translation_blocks: Callable[[str], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Materialize at most one complete ordinary Review patch per segment."""

    segment_ids = tuple(str(item) for item in segment_order)
    if (
        any(not item for item in segment_ids)
        or len(segment_ids) != len(set(segment_ids))
    ):
        raise ReviewArbitrationError("materialization segment order is invalid")
    segment_rank = {
        segment_id: index for index, segment_id in enumerate(segment_ids)
    }
    seen_paths: set[str] = set()
    by_segment: dict[str, list[CanonicalCandidate]] = {}
    for candidate in candidates:
        if candidate.path in seen_paths:
            raise ReviewArbitrationError(
                f"duplicate materialized target {candidate.path}"
            )
        seen_paths.add(candidate.path)
        if candidate.segment_id not in segment_rank:
            raise ReviewArbitrationError(
                f"materialized candidate has unknown segment {candidate.segment_id}"
            )
        expected_path = _target_path(
            candidate.segment_id,
            target_kind=candidate.target_kind,
            target_id=candidate.target_id,
        )
        if expected_path != candidate.path:
            raise ReviewArbitrationError(
                f"materialized candidate path identity is invalid: {candidate.path}"
            )
        if candidate.target_kind == "translation_block":
            if (
                not isinstance(candidate.replacement, Mapping)
                or set(candidate.replacement) != {"block_id", "text"}
                or candidate.replacement.get("block_id") != candidate.target_id
                or not isinstance(candidate.replacement.get("text"), str)
            ):
                raise ReviewArbitrationError(
                    f"materialized translation replacement is invalid for {candidate.path}"
                )
        elif (
            candidate.target_kind != "annotation"
            or candidate.target_id not in ANNOTATION_FIELDS
        ):
            raise ReviewArbitrationError(
                f"materialized candidate target kind is invalid for {candidate.path}"
            )
        by_segment.setdefault(candidate.segment_id, []).append(candidate)
    output: list[dict[str, Any]] = []
    for segment_id in segment_ids:
        values = by_segment.get(str(segment_id), [])
        if not values:
            continue
        patch = empty_review_patch(str(segment_id))
        reasons: list[str] = []
        original_blocks = list(original_translation_blocks(str(segment_id)))
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"block_id", "text"}
            or not isinstance(item.get("block_id"), str)
            or not item.get("block_id")
            or not isinstance(item.get("text"), str)
            for item in original_blocks
        ):
            raise ReviewArbitrationError(
                f"baseline translation blocks are malformed for {segment_id}"
            )
        original_order = [str(item["block_id"]) for item in original_blocks]
        if len(original_order) != len(set(original_order)):
            raise ReviewArbitrationError(
                f"baseline translation block ids are duplicate for {segment_id}"
            )
        translation_by_id = {
            str(item["block_id"]): deepcopy(dict(item))
            for item in original_blocks
        }
        translation_changed = False
        ordered_values = sorted(
            values,
            key=lambda candidate: (
                0 if candidate.target_kind == "translation_block" else 1,
                (
                    original_order.index(candidate.target_id)
                    if candidate.target_kind == "translation_block"
                    and candidate.target_id in original_order
                    else (
                        ANNOTATION_FIELDS.index(candidate.target_id)
                        if candidate.target_kind == "annotation"
                        else 10**9
                    )
                ),
                candidate.path,
            ),
        )
        for candidate in ordered_values:
            for reason in candidate.reasons:
                if reason and reason not in reasons:
                    reasons.append(reason)
            if candidate.target_kind == "translation_block":
                if candidate.target_id not in translation_by_id:
                    raise ReviewArbitrationError(
                        f"materialized translation target is absent for {candidate.path}"
                    )
                translation_by_id[candidate.target_id] = deepcopy(
                    candidate.replacement
                )
                translation_changed = True
            else:
                patch[candidate.target_id] = deepcopy(candidate.replacement)
        if translation_changed:
            if set(translation_by_id) != set(original_order):
                raise ReviewArbitrationError(
                    f"materialized translation coverage changed for {segment_id}"
                )
            patch["translation_blocks"] = [
                translation_by_id[block_id] for block_id in original_order
            ]
        patch["reason"] = "; ".join(reasons)
        output.append(patch)
    return output


def validate_materialized_review(
    review: Mapping[str, Any],
    *,
    full_schema: Mapping[str, Any],
    baseline: Any,
    changed_paths: Sequence[str],
    apply_review: Callable[[Any, Mapping[str, Any]], None],
    project_review: (
        Callable[[Mapping[str, Any], Sequence[str]], Mapping[str, Any]]
    ),
) -> Any:
    """Validate a full merged Review and localize domain failures."""

    ordered_paths = tuple(dict.fromkeys(str(item) for item in changed_paths))
    try:
        validate_json_schema(instance=review, schema=full_schema)
    except ValidationError as exc:
        raise ReviewArbitrationNeedsSupervision(
            paths=ordered_paths or ("/review",),
            reason=f"merged Review schema validation failed: {exc.message}",
        ) from exc

    def attempt(paths: Sequence[str]) -> tuple[bool, Any]:
        projected = project_review(review, paths)
        try:
            validate_json_schema(instance=projected, schema=full_schema)
            trial = deepcopy(baseline)
            apply_review(trial, projected)
            return True, trial
        except (ValidationError, RuntimeError, TypeError, ValueError):
            return False, None

    valid, merged = attempt(ordered_paths)
    if valid:
        return merged

    by_segment: dict[str, list[str]] = {}
    for path in ordered_paths:
        parts = path.split("/")
        segment_key = parts[2] if len(parts) > 2 else ""
        by_segment.setdefault(segment_key, []).append(path)
    isolated: list[str] = []
    for segment_paths in by_segment.values():
        segment_valid, _ = attempt(segment_paths)
        if segment_valid:
            continue
        found: tuple[str, ...] | None = None
        for size in range(1, len(segment_paths) + 1):
            for subset in combinations(segment_paths, size):
                subset_valid, _ = attempt(subset)
                if not subset_valid:
                    found = subset
                    break
            if found is not None:
                break
        isolated.extend(found or tuple(segment_paths))
    raise ReviewArbitrationNeedsSupervision(
        paths=tuple(dict.fromkeys(isolated or ordered_paths)),
        reason="merged Review failed domain validation",
    )
