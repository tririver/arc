from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import unicodedata
from typing import Any, Literal, Mapping, Sequence, TypedDict, cast


SOURCE_CREDIT_VERSION = "arc.companion.source-credit.v1"


class LocalizedNameEvidence(TypedDict):
    evidence_class: Literal["source_variant", "cached_reference"]
    reference_identity: str
    field_sha256: str


class SourceCreditAuthor(TypedDict):
    id: str
    source_name: str
    localized_name: str | None
    localized_evidence: LocalizedNameEvidence | None
    anchor_id: str
    content_sha256: str


class SourceCreditAffiliation(TypedDict):
    id: str
    text: str
    anchor_id: str
    content_sha256: str


class SourceCreditProfile(TypedDict):
    id: str
    text: str
    author_id: str | None
    anchor_id: str
    content_sha256: str


class SourceCreditAssociation(TypedDict):
    id: str
    author_id: str
    affiliation_id: str
    content_sha256: str


class SourceCreditAnchor(TypedDict):
    id: str
    block_id: str | None
    order: int
    placement: Literal["source", "after_title"]
    content_sha256: str


class SourceCredit(TypedDict):
    schema_version: Literal["arc.companion.source-credit.v1"]
    authors: list[SourceCreditAuthor]
    affiliations: list[SourceCreditAffiliation]
    profiles: list[SourceCreditProfile]
    associations: list[SourceCreditAssociation]
    anchors: list[SourceCreditAnchor]
    canonical_sha256: str


class SourceCreditError(ValueError):
    """The canonical source-credit object is malformed or has been changed."""


_TOP_KEYS = {
    "schema_version", "authors", "affiliations", "profiles", "associations",
    "anchors", "canonical_sha256",
}
_AUTHOR_KEYS = {
    "id", "source_name", "localized_name", "localized_evidence", "anchor_id",
    "content_sha256",
}
_AFFILIATION_KEYS = {"id", "text", "anchor_id", "content_sha256"}
_PROFILE_KEYS = {"id", "text", "author_id", "anchor_id", "content_sha256"}
_ASSOCIATION_KEYS = {"id", "author_id", "affiliation_id", "content_sha256"}
_ANCHOR_KEYS = {"id", "block_id", "order", "placement", "content_sha256"}
_EVIDENCE_KEYS = {"evidence_class", "reference_identity", "field_sha256"}


def normalize_source_credit(
    document: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    *,
    cached_reference: Mapping[str, Any] | None = None,
    explicit_author_mapping: Sequence[Mapping[str, Any]] | None = None,
    diagnostics: list[dict[str, str]] | None = None,
) -> SourceCredit:
    """Build the output-neutral source-credit projection without any I/O.

    ``cached_reference`` must already have been selected and loaded by the
    caller.  This function never discovers references.  Multi-author documents
    accept cached localized names only through ``explicit_author_mapping``.
    """

    front = document.get("front_matter")
    front = front if isinstance(front, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    blocks = [
        item for item in document.get("blocks") or [] if isinstance(item, Mapping)
    ]
    block_order = {
        _text(item.get("block_id")): index
        for index, item in enumerate(blocks)
        if _text(item.get("block_id"))
    }
    front_block_ids = front.get("block_ids")
    front_block_ids = (
        front_block_ids if isinstance(front_block_ids, Mapping) else {}
    )

    raw_authors = _records(front.get("author_records") or front.get("authors"))
    author_source = "front"
    if not raw_authors:
        raw_authors = _records(metadata.get("authors") or metadata.get("author"))
        author_source = "metadata"
    raw_affiliations = _records(
        front.get("affiliation_records") or front.get("affiliations")
    )
    if not raw_affiliations:
        raw_affiliations = _records(metadata.get("affiliations"))
    raw_profiles = _records(
        front.get("profiles") or front.get("author_profiles")
    )
    profile_source = "front"
    if not raw_profiles:
        raw_profiles = _records(
            metadata.get("profiles") or metadata.get("author_profiles")
        )
        profile_source = "metadata"

    anchors: list[SourceCreditAnchor] = []
    authors: list[SourceCreditAuthor] = []
    affiliations: list[SourceCreditAffiliation] = []
    profiles: list[SourceCreditProfile] = []
    seen_elements: set[tuple[str, str]] = set()

    for index, raw in enumerate(raw_authors):
        source_name = _record_text(raw, ("source_name", "name", "text"))
        if not source_name:
            continue
        element_identity = _element_identity(
            "author", raw, index, source_name, source=author_source
        )
        content_identity = _sha256({"source_name": source_name})
        if (element_identity, content_identity) in seen_elements:
            continue
        seen_elements.add((element_identity, content_identity))
        anchor = _anchor_for(
            "author", raw, index, front_block_ids.get("authors"),
            block_order=block_order, fallback_order=index,
        )
        anchors.append(anchor)
        author_payload: dict[str, Any] = {
            "id": element_identity,
            "source_name": source_name,
            "localized_name": None,
            "localized_evidence": None,
            "anchor_id": anchor["id"],
        }
        author_payload["content_sha256"] = _sha256(author_payload)
        authors.append(cast(SourceCreditAuthor, author_payload))

    affiliation_offset = len(authors)
    for index, raw in enumerate(raw_affiliations):
        text = _record_text(raw, ("text", "name", "affiliation"))
        if not text:
            continue
        element_identity = _element_identity(
            "affiliation", raw, index, text, source="front"
        )
        content_identity = _sha256({"text": text})
        if (element_identity, content_identity) in seen_elements:
            continue
        seen_elements.add((element_identity, content_identity))
        anchor = _anchor_for(
            "affiliation", raw, index, front_block_ids.get("affiliations"),
            block_order=block_order, fallback_order=affiliation_offset + index,
        )
        anchors.append(anchor)
        payload: dict[str, Any] = {
            "id": element_identity,
            "text": text,
            "anchor_id": anchor["id"],
        }
        payload["content_sha256"] = _sha256(payload)
        affiliations.append(cast(SourceCreditAffiliation, payload))

    profile_offset = len(authors) + len(affiliations)
    for index, raw in enumerate(raw_profiles):
        text = _record_text(raw, ("text", "profile", "biography", "description"))
        if not text:
            continue
        element_identity = _element_identity(
            "profile", raw, index, text, source=profile_source
        )
        content_identity = _sha256({"text": text})
        if (element_identity, content_identity) in seen_elements:
            continue
        seen_elements.add((element_identity, content_identity))
        author_id = _explicit_author_id(raw, authors)
        anchor = _anchor_for(
            "profile", raw, index,
            front_block_ids.get("profiles")
            or front_block_ids.get("author_profiles"),
            block_order=block_order, fallback_order=profile_offset + index,
        )
        anchors.append(anchor)
        payload = {
            "id": element_identity,
            "text": text,
            "author_id": author_id,
            "anchor_id": anchor["id"],
        }
        payload["content_sha256"] = _sha256(payload)
        profiles.append(cast(SourceCreditProfile, payload))

    _apply_source_variants(
        authors,
        front.get("author_name_variants"),
        diagnostics=diagnostics,
    )
    _apply_cached_reference(
        authors,
        cached_reference=cached_reference,
        explicit_author_mapping=explicit_author_mapping,
        diagnostics=diagnostics,
    )

    associations = _associations(
        front.get("author_affiliations") or front.get("associations"),
        authors=authors,
        affiliations=affiliations,
    )
    anchors = _dedupe_anchors(anchors)
    value: dict[str, Any] = {
        "schema_version": SOURCE_CREDIT_VERSION,
        "authors": authors,
        "affiliations": affiliations,
        "profiles": profiles,
        "associations": associations,
        "anchors": anchors,
    }
    value["canonical_sha256"] = _sha256(value)
    return validate_source_credit(value)


def validate_source_credit(value: Mapping[str, Any]) -> SourceCredit:
    """Validate the complete closed canonical shape and all bound hashes."""

    if not isinstance(value, Mapping) or set(value) != _TOP_KEYS:
        raise SourceCreditError("source-credit shape is invalid")
    if value.get("schema_version") != SOURCE_CREDIT_VERSION:
        raise SourceCreditError("source-credit schema is invalid")
    output = deepcopy(dict(value))
    for key in ("authors", "affiliations", "profiles", "associations", "anchors"):
        if not isinstance(output.get(key), list):
            raise SourceCreditError(f"source-credit {key} are invalid")

    anchors = output["anchors"]
    _validate_records(anchors, _ANCHOR_KEYS, "anchor")
    anchor_ids = _unique_ids(anchors, "anchor")
    for anchor in anchors:
        if (
            anchor["placement"] not in {"source", "after_title"}
            or not isinstance(anchor["order"], int)
            or isinstance(anchor["order"], bool)
            or anchor["order"] < 0
            or (anchor["block_id"] is not None and not _text(anchor["block_id"]))
        ):
            raise SourceCreditError("source-credit anchor is invalid")
        _verify_record_hash(anchor)

    authors = output["authors"]
    _validate_records(authors, _AUTHOR_KEYS, "author")
    author_ids = _unique_ids(authors, "author")
    for author in authors:
        if not _text(author["source_name"]) or author["anchor_id"] not in anchor_ids:
            raise SourceCreditError("source-credit author is invalid")
        localized = author["localized_name"]
        evidence = author["localized_evidence"]
        if (localized is None) != (evidence is None):
            raise SourceCreditError("localized source-credit evidence is incomplete")
        if localized is not None:
            if not _text(localized) or not isinstance(evidence, Mapping):
                raise SourceCreditError("localized source-credit name is invalid")
            if set(evidence) != _EVIDENCE_KEYS:
                raise SourceCreditError("localized source-credit evidence shape is invalid")
            if (
                evidence["evidence_class"] not in {"source_variant", "cached_reference"}
                or not _text(evidence["reference_identity"])
                or not _digest(evidence["field_sha256"])
                or evidence["field_sha256"] != _sha256(_text(localized))
            ):
                raise SourceCreditError("localized source-credit evidence is invalid")
        _verify_record_hash(author)

    affiliations = output["affiliations"]
    _validate_records(affiliations, _AFFILIATION_KEYS, "affiliation")
    affiliation_ids = _unique_ids(affiliations, "affiliation")
    for affiliation in affiliations:
        if not _text(affiliation["text"]) or affiliation["anchor_id"] not in anchor_ids:
            raise SourceCreditError("source-credit affiliation is invalid")
        _verify_record_hash(affiliation)

    profiles = output["profiles"]
    _validate_records(profiles, _PROFILE_KEYS, "profile")
    _unique_ids(profiles, "profile")
    for profile in profiles:
        if (
            not _text(profile["text"])
            or profile["anchor_id"] not in anchor_ids
            or (
                profile["author_id"] is not None
                and profile["author_id"] not in author_ids
            )
        ):
            raise SourceCreditError("source-credit profile is invalid")
        _verify_record_hash(profile)

    associations = output["associations"]
    _validate_records(associations, _ASSOCIATION_KEYS, "association")
    _unique_ids(associations, "association")
    pairs: set[tuple[str, str]] = set()
    for association in associations:
        pair = (association["author_id"], association["affiliation_id"])
        if (
            pair[0] not in author_ids
            or pair[1] not in affiliation_ids
            or pair in pairs
        ):
            raise SourceCreditError("source-credit association is invalid")
        pairs.add(pair)
        _verify_record_hash(association)

    expected = _sha256({key: output[key] for key in _TOP_KEYS if key != "canonical_sha256"})
    if output.get("canonical_sha256") != expected:
        raise SourceCreditError("source-credit canonical hash is invalid")
    return cast(SourceCredit, output)


def project_source_credit(value: Mapping[str, Any]) -> SourceCredit:
    """Return the one identical validated projection used by Web and PDF."""

    return validate_source_credit(value)


def source_credit_hash(value: Mapping[str, Any]) -> str:
    return validate_source_credit(value)["canonical_sha256"]


def ordered_source_credit_items(
    value: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any], SourceCreditAnchor]]:
    """Return output-neutral elements in authoritative placement/order."""

    credit = validate_source_credit(value)
    anchors = {item["id"]: item for item in credit["anchors"]}
    items: list[
        tuple[int, int, int, str, Mapping[str, Any], SourceCreditAnchor]
    ] = []
    kind_order = {"author": 0, "affiliation": 1, "profile": 2}
    for kind, records in (
        ("author", credit["authors"]),
        ("affiliation", credit["affiliations"]),
        ("profile", credit["profiles"]),
    ):
        for record in records:
            anchor = anchors[record["anchor_id"]]
            placement_order = 0 if anchor["placement"] == "after_title" else 1
            items.append((
                placement_order, anchor["order"], kind_order[kind], kind,
                record, anchor,
            ))
    items.sort(key=lambda item: (item[0], item[1], item[2]))
    return [
        (kind, record, anchor)
        for _, _, _, kind, record, anchor in items
    ]


def source_credit_placement(
    value: Mapping[str, Any],
    *,
    front_matter_block_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Project canonical anchors to shared semantic document slots."""

    front_ids = {_text(item) for item in front_matter_block_ids if _text(item)}
    output: list[dict[str, Any]] = []
    for kind, record, anchor in ordered_source_credit_items(value):
        block_id = anchor["block_id"]
        if anchor["placement"] == "after_title":
            slot = "after_title"
        elif block_id in front_ids:
            slot = "front_matter"
        else:
            slot = "source_block"
        output.append({
            "kind": kind,
            "id": record["id"],
            "anchor_id": anchor["id"],
            "slot": slot,
            "block_id": block_id,
        })
    return output


def source_credit_visible_projection(
    value: Mapping[str, Any],
    *,
    front_matter_block_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return the shared text-bearing PDF/Web source-credit projection."""

    credit = validate_source_credit(value)
    order = source_credit_placement(
        credit, front_matter_block_ids=front_matter_block_ids,
    )
    records = {
        (kind, str(item["id"])): item
        for key, kind in (
            ("authors", "author"),
            ("affiliations", "affiliation"),
            ("profiles", "profile"),
        )
        for item in credit[key]
    }
    return [
        {
            "kind": str(item["kind"]),
            "id": str(item["id"]),
            "anchor_id": str(item["anchor_id"]),
            "slot": str(item["slot"]),
            "block_id": item["block_id"],
            "source_text": str(
                records[(str(item["kind"]), str(item["id"]))].get(
                    "source_name"
                )
                or records[(str(item["kind"]), str(item["id"]))].get("text")
                or ""
            ),
            "localized_text": (
                str(
                    records[(str(item["kind"]), str(item["id"]))].get(
                        "localized_name"
                    )
                )
                if records[(str(item["kind"]), str(item["id"]))].get(
                    "localized_name"
                )
                else None
            ),
        }
        for item in order
    ]


def _apply_source_variants(
    authors: list[SourceCreditAuthor],
    raw_variants: Any,
    *,
    diagnostics: list[dict[str, str]] | None,
) -> None:
    for raw in _records(raw_variants):
        author = _mapped_author(raw, authors)
        localized = _record_text(raw, ("localized_name", "name", "text", "value"))
        if author is None or not localized or author["localized_name"] is not None:
            continue
        identity = _source_variant_identity(raw, localized)
        if not identity:
            _diagnose(
                diagnostics,
                "source_credit_source_variant_identity_missing",
                (
                    "Localized source-name variant was ignored because it has no "
                    "explicit record, block, or field identity."
                ),
            )
            continue
        _set_localized(author, localized, "source_variant", identity)


def _apply_cached_reference(
    authors: list[SourceCreditAuthor],
    *,
    cached_reference: Mapping[str, Any] | None,
    explicit_author_mapping: Sequence[Mapping[str, Any]] | None,
    diagnostics: list[dict[str, str]] | None,
) -> None:
    if not isinstance(cached_reference, Mapping):
        return
    identity = _record_text(
        cached_reference, ("identity", "source_id", "paper_id")
    )
    if not identity:
        return
    reference_document = cached_reference.get("document")
    reference_document = (
        reference_document if isinstance(reference_document, Mapping) else {}
    )
    reference_front = reference_document.get("front_matter")
    reference_front = (
        reference_front if isinstance(reference_front, Mapping) else {}
    )
    reference_metadata = cached_reference.get("metadata")
    reference_metadata = (
        reference_metadata if isinstance(reference_metadata, Mapping) else {}
    )
    reference_authors = _reference_author_records(
        reference_front,
        reference_metadata,
        cached_reference,
    )
    reference_targets: dict[str, str] = {}
    for raw in reference_authors:
        target_identity = _explicit_record_identity(raw)
        target_name = _record_text(raw, ("source_name", "name", "text"))
        target_field_sha256 = (
            _record_text(raw, ("field_sha256",))
            if isinstance(raw, Mapping) else ""
        )
        if (
            not target_identity
            or not target_name
            or target_field_sha256 != _raw_text_sha256(target_name)
            or target_identity in reference_targets
        ):
            _diagnose(
                diagnostics,
                "source_credit_author_mapping_target_evidence_invalid",
                (
                    "Cached reference was ignored because an author record "
                    "lacks unique stable identity evidence."
                ),
            )
            return
        reference_targets[target_identity] = target_name
    names = list(reference_targets.values())
    if len(authors) == 1 and len(names) == 1:
        if authors[0]["localized_name"] is None:
            _set_localized(authors[0], names[0], "cached_reference", identity)
        return
    if len(authors) < 2:
        return
    if not explicit_author_mapping:
        _diagnose(
            diagnostics,
            "source_credit_author_mapping_required",
            (
                "Multi-author cached reference was ignored because no explicit "
                "identity mapping was supplied."
            ),
        )
        return
    if len(reference_authors) != len(authors):
        _diagnose(
            diagnostics,
            "source_credit_author_mapping_cardinality_mismatch",
            (
                "Multi-author cached reference was ignored because source and "
                "reference author-record cardinalities differ."
            ),
        )
        return

    pending: list[tuple[SourceCreditAuthor, str]] = []
    used_authors: set[str] = set()
    used_reference_authors: set[str] = set()
    expected_mapping_keys = {
        "source_author_id", "reference_author_id", "reference_identity",
    }
    for mapping in explicit_author_mapping:
        if not isinstance(mapping, Mapping):
            _diagnose(
                diagnostics,
                "source_credit_author_mapping_shape_invalid",
                "Multi-author identity mapping was ignored because its shape is invalid.",
            )
            return
        if set(mapping) != expected_mapping_keys:
            _diagnose(
                diagnostics,
                "source_credit_author_mapping_identity_keys_required",
                (
                    "Multi-author mapping was ignored because only stable source "
                    "and reference author identities are accepted."
                ),
            )
            return
        author = _author_by_stable_mapping_identity(
            _record_text(mapping, ("source_author_id",)),
            authors,
        )
        reference_author_id = _record_text(
            mapping, ("reference_author_id",)
        )
        mapping_identity = _record_text(mapping, ("reference_identity",))
        localized = reference_targets.get(reference_author_id, "")
        if (
            not mapping_identity
            or mapping_identity != identity
        ):
            _diagnose(
                diagnostics,
                "source_credit_author_mapping_reference_identity_invalid",
                (
                    "Multi-author mapping was ignored because its cached-reference "
                    "identity is missing or does not match."
                ),
            )
            return
        if author is None:
            _diagnose(
                diagnostics,
                "source_credit_author_mapping_source_identity_invalid",
                (
                    "Multi-author mapping was ignored because its source-author "
                    "identity is not present in the source record."
                ),
            )
            return
        if not localized:
            _diagnose(
                diagnostics,
                "source_credit_author_mapping_target_evidence_invalid",
                (
                    "Multi-author mapping was ignored because its target identity "
                    "has no validated reference author record."
                ),
            )
            return
        if (
            author["id"] in used_authors
            or reference_author_id in used_reference_authors
        ):
            _diagnose(
                diagnostics,
                "source_credit_author_mapping_duplicate_identity",
                (
                    "Multi-author mapping was ignored because an author identity "
                    "was mapped more than once."
                ),
            )
            return
        used_authors.add(author["id"])
        used_reference_authors.add(reference_author_id)
        pending.append((author, localized))
    if (
        len(pending) != len(authors)
        or len(used_reference_authors) != len(reference_targets)
    ):
        _diagnose(
            diagnostics,
            "source_credit_author_mapping_cardinality_mismatch",
            (
                "Multi-author mapping was ignored because it does not cover every "
                "source and reference author identity exactly once."
            ),
        )
        return
    for author, localized in pending:
        if author["localized_name"] is None:
            _set_localized(author, localized, "cached_reference", identity)


def _set_localized(
    author: SourceCreditAuthor,
    localized: str,
    evidence_class: Literal["source_variant", "cached_reference"],
    identity: str,
) -> None:
    author["localized_name"] = localized
    author["localized_evidence"] = {
        "evidence_class": evidence_class,
        "reference_identity": identity,
        "field_sha256": _sha256(localized),
    }
    author["content_sha256"] = _sha256({
        key: value for key, value in author.items() if key != "content_sha256"
    })


def _associations(
    raw_associations: Any,
    *,
    authors: list[SourceCreditAuthor],
    affiliations: list[SourceCreditAffiliation],
) -> list[SourceCreditAssociation]:
    output: list[SourceCreditAssociation] = []
    seen: set[tuple[str, str]] = set()
    for raw in _records(raw_associations):
        author_id = _explicit_author_id(raw, authors)
        affiliation_id = _explicit_affiliation_id(raw, affiliations)
        if not author_id or not affiliation_id or (author_id, affiliation_id) in seen:
            continue
        seen.add((author_id, affiliation_id))
        payload: dict[str, Any] = {
            "id": _stable_id(
                "association", f"{author_id}\0{affiliation_id}"
            ),
            "author_id": author_id,
            "affiliation_id": affiliation_id,
        }
        payload["content_sha256"] = _sha256(payload)
        output.append(cast(SourceCreditAssociation, payload))
    return output


def _anchor_for(
    kind: str,
    raw: Any,
    index: int,
    category_block_ids: Any,
    *,
    block_order: Mapping[str, int],
    fallback_order: int,
) -> SourceCreditAnchor:
    raw_mapping = raw if isinstance(raw, Mapping) else {}
    block_id = _record_text(raw_mapping, ("block_id",))
    if not block_id:
        block_ids = raw_mapping.get("block_ids")
        if isinstance(block_ids, Sequence) and not isinstance(block_ids, (str, bytes)):
            candidates = [_text(item) for item in block_ids if _text(item)]
            if len(candidates) == 1:
                block_id = candidates[0]
    if not block_id and isinstance(category_block_ids, Sequence) and not isinstance(
        category_block_ids, (str, bytes)
    ):
        candidates = [_text(item) for item in category_block_ids if _text(item)]
        if len(candidates) == 1:
            block_id = candidates[0]
    reliable = block_id in block_order
    payload: dict[str, Any] = {
        "id": _stable_id(
            "anchor", f"{kind}\0{block_id or 'after-title'}\0{index}"
        ),
        "block_id": block_id if reliable else None,
        "order": block_order[block_id] if reliable else fallback_order,
        "placement": "source" if reliable else "after_title",
    }
    payload["content_sha256"] = _sha256(payload)
    return cast(SourceCreditAnchor, payload)


def _dedupe_anchors(
    anchors: list[SourceCreditAnchor],
) -> list[SourceCreditAnchor]:
    output: list[SourceCreditAnchor] = []
    seen: set[tuple[str, str]] = set()
    for anchor in anchors:
        identity = (anchor["id"], anchor["content_sha256"])
        if identity not in seen:
            seen.add(identity)
            output.append(anchor)
    return output


def _element_identity(
    kind: str, raw: Any, index: int, text: str, *, source: str,
) -> str:
    if isinstance(raw, Mapping):
        explicit = _record_text(raw, ("source_id", "element_id", "id"))
        if explicit:
            return _stable_id(kind, explicit)
        block = _record_text(raw, ("block_id",))
        if block:
            return _stable_id(kind, f"{block}\0{index}")
    return _stable_id(kind, f"{source}\0{index}\0{text}")


def _explicit_author_id(
    raw: Any, authors: list[SourceCreditAuthor],
) -> str | None:
    if not isinstance(raw, Mapping):
        return None
    explicit = _record_text(raw, ("author_id", "source_author_id"))
    if explicit:
        stable = _stable_id("author", explicit)
        return stable if any(item["id"] == stable for item in authors) else None
    index = raw.get("author_index")
    if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(authors):
        return authors[index]["id"]
    return None


def _explicit_affiliation_id(
    raw: Any, affiliations: list[SourceCreditAffiliation],
) -> str | None:
    if not isinstance(raw, Mapping):
        return None
    explicit = _record_text(raw, ("affiliation_id", "source_affiliation_id"))
    if explicit:
        stable = _stable_id("affiliation", explicit)
        return stable if any(item["id"] == stable for item in affiliations) else None
    index = raw.get("affiliation_index")
    if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(affiliations):
        return affiliations[index]["id"]
    return None


def _mapped_author(
    raw: Any, authors: list[SourceCreditAuthor],
) -> SourceCreditAuthor | None:
    if not isinstance(raw, Mapping):
        return None
    author_id = _explicit_author_id(raw, authors)
    if author_id:
        return next(item for item in authors if item["id"] == author_id)
    source_name = _record_text(raw, ("source_name", "source_author"))
    matches = [item for item in authors if item["source_name"] == source_name]
    return matches[0] if source_name and len(matches) == 1 else None


def _author_by_stable_mapping_identity(
    identity: str,
    authors: list[SourceCreditAuthor],
) -> SourceCreditAuthor | None:
    if not identity:
        return None
    stable = _stable_id("author", identity)
    matches = [
        item for item in authors
        if item["id"] == identity or item["id"] == stable
    ]
    return matches[0] if len(matches) == 1 else None


def _reference_author_records(
    reference_front: Mapping[str, Any],
    reference_metadata: Mapping[str, Any],
    cached_reference: Mapping[str, Any],
) -> list[Any]:
    containers = (
        reference_front,
        reference_metadata,
        cached_reference,
    )
    for container in containers:
        records = _records(container.get("author_records"))
        if records:
            return records
    return []


def _explicit_record_identity(raw: Any) -> str:
    return _record_text(
        raw, ("source_author_id", "source_id", "element_id", "id")
    )


def _source_variant_identity(raw: Any, localized: str) -> str:
    if not isinstance(raw, Mapping):
        return ""
    explicit = _record_text(raw, ("source_identity",))
    if explicit:
        return explicit
    record = _record_text(raw, ("record_id", "variant_id", "source_id"))
    if record:
        return f"record:{record}"
    block = _record_text(raw, ("block_id",))
    if block:
        return f"block:{block}"
    field = _record_text(raw, ("field_id", "field_path"))
    if field:
        return f"field:{field}"
    field_sha256 = _record_text(raw, ("field_sha256",))
    if _digest(field_sha256) and field_sha256 == _raw_text_sha256(localized):
        return f"field-sha256:{field_sha256}"
    return ""


def _diagnose(
    diagnostics: list[dict[str, str]] | None,
    code: str,
    message: str,
) -> None:
    if diagnostics is not None:
        diagnostics.append({
            "severity": "warning",
            "code": code,
            "source": "source-credit",
            "message": message,
        })


def _validate_records(
    records: list[Any], keys: set[str], label: str,
) -> None:
    if not all(isinstance(item, dict) and set(item) == keys for item in records):
        raise SourceCreditError(f"source-credit {label} shape is invalid")


def _unique_ids(records: list[dict[str, Any]], label: str) -> set[str]:
    identities = [_text(item.get("id")) for item in records]
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise SourceCreditError(f"source-credit {label} identities are invalid")
    return set(identities)


def _verify_record_hash(record: Mapping[str, Any]) -> None:
    expected = _sha256({
        key: value for key, value in record.items() if key != "content_sha256"
    })
    if record.get("content_sha256") != expected:
        raise SourceCreditError("source-credit record hash is invalid")


def _records(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _record_text(value: Any, keys: Sequence[str]) -> str:
    if isinstance(value, Mapping):
        for key in keys:
            text = _text(value.get(key))
            if text:
                return text
        return ""
    return _text(value)


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFC", value.strip())


def _stable_id(kind: str, value: str) -> str:
    return f"{kind}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
