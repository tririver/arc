from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .io import read_json, sha256_json, write_json, write_text


INTENT_GUIDANCE_VERSION = "arc.companion.intent-guidance.v2"
LEGACY_INTENT_GUIDANCE_VERSION = "arc.companion.intent-guidance.v1"
WORKER_POLICY_VERSION = "arc.companion.intent-guidance-worker-policy.v2"
TARGET_CATALOG_VERSION = "arc.companion.intent-guidance-target-catalog.v1"
TARGET_PAGE_VERSION = "arc.companion.intent-guidance-target-page.v1"
INTENT_GUIDANCE_LANES = (
    "glossary",
    "title_translation",
    "guide",
    "translation",
    "commentary",
    "review",
)
READ_ONLY_REFERENCE_OPERATIONS = ("get-parsed-toc", "get-parsed-section")
CONTROLLER_PAGE_BYTES = 46 * 1024
GUIDANCE_MAX_CHARS = 8_000
GUIDANCE_MAX_BYTES = 12_000
TARGET_CATALOG_MAX_BYTES = 8 * 1024 * 1024
TARGET_INLINE_HARD_BYTES = 8 * 1024
TARGET_INLINE_BYTES = TARGET_INLINE_HARD_BYTES * 9 // 10
TARGET_PAGE_HARD_BYTES = 46 * 1024
TARGET_PAGE_BYTES = TARGET_PAGE_HARD_BYTES * 9 // 10
LIST_REFERENCE_TARGETS_OPERATION = "list-reference-targets"
POLICY_TARGETS_OPERATION = "policy-targets"
WORKER_ALLOWED_OPERATIONS = (
    "artifact-read", POLICY_TARGETS_OPERATION, *READ_ONLY_REFERENCE_OPERATIONS,
)

INTENT_GUIDANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["guidance", "resolution_status"],
    "properties": {
        "guidance": {
            "type": "string", "minLength": 1, "maxLength": GUIDANCE_MAX_CHARS,
        },
        "resolution_status": {"type": "string", "enum": ["resolved", "ambiguous"]},
        "reference_targets": {
            "type": "array",
            "uniqueItems": True,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source_id", "locator", "purpose", "lanes"],
                "properties": {
                    "source_id": {"type": "string", "minLength": 1},
                    "locator": {"type": "string", "minLength": 1},
                    "purpose": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "lanes": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": list(INTENT_GUIDANCE_LANES)},
                    },
                },
            },
        },
    },
}

_SAFE_METADATA_FIELDS = (
    "paper_id",
    "parser_version",
    "title",
    "authors",
    "year",
    "language",
    "source_language",
    "document_kind",
    "edition",
    "publisher",
    "isbn",
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class IntentGuidanceError(RuntimeError):
    """Raised when global intent guidance cannot be built or trusted."""


class IntentGuidanceAmbiguousError(IntentGuidanceError):
    """Raised after a valid ambiguous result is durably recorded."""

    def __init__(self, message: str, *, artifact: Mapping[str, Any]):
        super().__init__(message)
        self.artifact = dict(artifact)


def build_intent_guidance(
    user_intent: str | None,
    *,
    source_language: str,
    target_language: str,
    document_type: str,
    context_paper_ids: Iterable[str] = (),
    project_dir: Path,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
    parsed_getter: Callable[..., dict[str, Any]] | None = None,
    toc_getter: Callable[[str], dict[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    """Build or reuse the single global guidance artifact for semantic inputs.

    Empty intent deliberately returns before touching reference caches or the
    model, preserving the legacy zero-call behavior.
    """
    intent = str(user_intent or "").strip()
    if not intent:
        return None
    source_language = _required_text(source_language, "source_language")
    target_language = _required_text(target_language, "target_language")
    document_type = _required_text(document_type, "document_type")
    source_ids = _authorized_source_ids(context_paper_ids)
    if parsed_getter is None or toc_getter is None:
        from arc_paper import service

        parsed_getter = parsed_getter or service.get_parsed_source_identity
        toc_getter = toc_getter or service.get_parsed_source_compact_toc

    references = [
        _load_reference(source_id, parsed_getter=parsed_getter, toc_getter=toc_getter)
        for source_id in source_ids
    ]
    semantic_input = {
        "user_intent": intent,
        "source_language": source_language,
        "target_language": target_language,
        "document_type": document_type,
        "reference_sources": references,
        "allowed_reference_operations": list(READ_ONLY_REFERENCE_OPERATIONS),
        "guidance_contract": INTENT_GUIDANCE_VERSION,
    }
    semantic_sha256 = sha256_json(semantic_input)
    user_intent_sha256 = hashlib.sha256(intent.encode("utf-8")).hexdigest()
    artifact_dir = project_dir / ".arc-companion" / "intent-guidance" / semantic_sha256
    artifact_path = artifact_dir / "artifact.json"
    if artifact_path.is_file() and not force:
        try:
            cached = read_json(artifact_path)
        except (OSError, ValueError, TypeError):
            cached = None
        if _artifact_valid(
            cached,
            semantic_sha256=semantic_sha256,
            user_intent_sha256=user_intent_sha256,
            references=references,
        ):
            if not _target_catalog_files_valid(cached, artifact_dir=artifact_dir):
                normalized = _validate_model_result(
                    _artifact_model_result(cached), references=references,
                )
                cached = _build_artifact(
                    normalized,
                    semantic_sha256=semantic_sha256,
                    user_intent_sha256=user_intent_sha256,
                    references=references,
                    artifact_dir=artifact_dir,
                    migration=(
                        cached.get("migration")
                        if isinstance(cached.get("migration"), Mapping) else None
                    ),
                )
                write_json(artifact_path, cached)
            return _require_resolved(cached)
    if not force:
        legacy_semantic_input = {
            **semantic_input,
            "guidance_contract": LEGACY_INTENT_GUIDANCE_VERSION,
        }
        legacy_semantic_sha256 = sha256_json(legacy_semantic_input)
        legacy_path = (
            project_dir / ".arc-companion" / "intent-guidance"
            / legacy_semantic_sha256 / "artifact.json"
        )
        if legacy_path.is_file():
            try:
                legacy = read_json(legacy_path)
            except (OSError, ValueError, TypeError):
                legacy = None
            if _legacy_artifact_valid(
                legacy,
                semantic_sha256=legacy_semantic_sha256,
                user_intent_sha256=user_intent_sha256,
                references=references,
            ):
                normalized = _validate_model_result(
                    _artifact_model_result(legacy), references=references,
                )
                artifact = _build_artifact(
                    normalized,
                    semantic_sha256=semantic_sha256,
                    user_intent_sha256=user_intent_sha256,
                    references=references,
                    artifact_dir=artifact_dir,
                    migration={
                        "from_schema_version": LEGACY_INTENT_GUIDANCE_VERSION,
                        "source_semantic_input_sha256": legacy_semantic_sha256,
                    },
                )
                write_json(artifact_path, artifact)
                return _require_resolved(artifact)

    prompt = _guidance_prompt(semantic_input)
    raw = call_model(
        prompt,
        INTENT_GUIDANCE_SCHEMA,
        artifact_dir / "llm",
        "companion-intent-guidance",
    )
    normalized = _validate_model_result(raw, references=references)
    artifact = _build_artifact(
        normalized,
        semantic_sha256=semantic_sha256,
        user_intent_sha256=user_intent_sha256,
        references=references,
        artifact_dir=artifact_dir,
    )
    write_json(artifact_path, artifact)
    return _require_resolved(artifact)


def worker_guidance_payload(
    artifact: Mapping[str, Any], *, lane: str | None = None,
) -> dict[str, Any]:
    """Return a byte-bounded bootstrap payload for one worker lane.

    Small catalogs are inlined. Larger catalogs are represented only by their
    content-addressed descriptor and can be enumerated with
    :func:`list_reference_targets`.
    """
    _validate_worker_artifact(artifact)
    selected_lane = _validate_lane(lane)
    return deepcopy(_worker_payload_unvalidated(artifact, lane=selected_lane))


def worker_guidance_prompt_prefix(
    artifact: Mapping[str, Any], *, lane: str | None = None,
) -> str:
    """Render one deterministic bootstrap prefix; never use it for delta turns."""
    payload = worker_guidance_payload(artifact, lane=lane)
    return (
        "GLOBAL USER-INTENT GUIDANCE (bootstrap only)\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\nThe source document remains authoritative for facts, coverage, and structure. "
        "Reference text may influence only terminology, idiom, and style; never inherit its "
        "additions, omissions, or errors. Read only the exact authorized reference targets "
        "through the supplied read-only policy.\n"
    )


def worker_policy_descriptor(
    artifact: Mapping[str, Any], *, lane: str | None = None,
) -> dict[str, Any]:
    """Describe the portable, cache-only reference-query policy for worker envs."""
    _validate_worker_artifact(artifact)
    selected_lane = _validate_lane(lane)
    selected_targets = _targets_for_lane(artifact, selected_lane)
    source_ids = sorted({str(item["source_id"]) for item in selected_targets})
    descriptor = target_catalog_descriptor(artifact, lane=selected_lane)
    policy: dict[str, Any] = {
        "schema_version": WORKER_POLICY_VERSION,
        "access": "local-cache-read-only",
        "allowed_operations": list(WORKER_ALLOWED_OPERATIONS),
        "reference_operations": list(READ_ONLY_REFERENCE_OPERATIONS),
        "target_catalog": descriptor,
        "authorized_source_ids": source_ids,
        "authorized_section_targets": [
            {"source_id": item["source_id"], "locator": item["locator"]}
            for item in selected_targets
        ],
        "targets": [
            {
                "source_id": item["source_id"],
                "locator": item["locator"],
                "purpose": item["purpose"],
            }
            for item in selected_targets
        ],
        "network": False,
        "mutation": False,
        "transport_fallback": "controller-evidence-request",
    }
    if descriptor["serialized_bytes"] > TARGET_INLINE_BYTES:
        policy["authorized_source_ids_sha256"] = sha256_json(source_ids)
        policy["authorized_source_count"] = len(source_ids)
    return policy


def validate_worker_query(
    artifact: Mapping[str, Any],
    *,
    operation: str,
    source_id: str,
    locator: str | None = None,
    lane: str | None = None,
) -> dict[str, str]:
    """Validate a structured worker query without parsing or executing shell text."""
    _validate_worker_artifact(artifact)
    selected_lane = _validate_lane(lane)
    operation = str(operation or "")
    source_id = str(source_id or "")
    if operation not in READ_ONLY_REFERENCE_OPERATIONS:
        raise IntentGuidanceError(f"reference operation is not read-only or allowed: {operation}")
    selected_targets = _targets_for_lane(artifact, selected_lane)
    source_ids = {str(item["source_id"]) for item in selected_targets}
    if source_id not in source_ids:
        raise IntentGuidanceError(f"reference source is not authorized: {source_id}")
    if operation == "get-parsed-toc":
        if locator is not None:
            raise IntentGuidanceError("get-parsed-toc does not accept a section locator")
        return {"operation": operation, "source_id": source_id}
    target = {"source_id": source_id, "locator": str(locator or "")}
    if target not in [
        {"source_id": item["source_id"], "locator": item["locator"]}
        for item in selected_targets
    ]:
        raise IntentGuidanceError(
            f"reference section is not an exact guidance target: {source_id}:{locator or ''}"
        )
    return {"operation": operation, **target}


def target_catalog_descriptor(
    artifact: Mapping[str, Any], *, lane: str | None = None,
) -> dict[str, Any]:
    """Return the stable descriptor for all targets or one lane projection."""
    _validate_worker_artifact(artifact)
    selected_lane = _validate_lane(lane)
    catalog = artifact.get("target_catalog")
    if not isinstance(catalog, Mapping):
        raise IntentGuidanceError("intent-guidance artifact has no target catalog")
    key = "all" if selected_lane is None else selected_lane
    descriptors = catalog.get("indexes")
    descriptor = descriptors.get(key) if isinstance(descriptors, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise IntentGuidanceError(f"intent-guidance target catalog is missing lane: {key}")
    return deepcopy(dict(descriptor))


def list_reference_targets(
    artifact: Mapping[str, Any],
    *,
    lane: str,
    cursor: str | None = None,
    source_id: str | None = None,
    query: str | None = None,
    limit_bytes: int | None = None,
) -> dict[str, Any]:
    """List one authorized lane catalog using whole-record, digest-bound pages."""
    _validate_worker_artifact(artifact)
    selected_lane = _validate_lane(lane, required=True)
    descriptor = target_catalog_descriptor(artifact, lane=selected_lane)
    targets = _targets_for_lane(artifact, selected_lane)
    normalized_source = str(source_id or "").strip()
    if normalized_source:
        authorized_sources = {str(item["source_id"]) for item in targets}
        if normalized_source not in authorized_sources:
            raise IntentGuidanceError(
                f"reference source is not authorized for lane {selected_lane}: "
                f"{normalized_source}"
            )
    normalized_query = " ".join(str(query or "").split()).casefold()
    filtered = [
        item for item in targets
        if (not normalized_source or item["source_id"] == normalized_source)
        and (
            not normalized_query
            or normalized_query in " ".join((
                item["source_id"], item["locator"], item["purpose"],
            )).casefold()
        )
    ]
    filter_sha256 = sha256_json({
        "lane": selected_lane,
        "source_id": normalized_source,
        "query": normalized_query,
    })
    position = _decode_target_cursor(
        cursor,
        catalog_sha256=str(descriptor["sha256"]),
        filter_sha256=filter_sha256,
    )
    if position > len(filtered):
        raise IntentGuidanceError("reference-target cursor is beyond the filtered catalog")
    if limit_bytes is None:
        strict_limit = TARGET_PAGE_HARD_BYTES
        target_limit = TARGET_PAGE_BYTES
    else:
        try:
            strict_limit = int(limit_bytes)
        except (TypeError, ValueError) as exc:
            raise IntentGuidanceError(
                "reference-target page limit must be an integer"
            ) from exc
        if strict_limit < 1 or strict_limit > TARGET_PAGE_HARD_BYTES:
            raise IntentGuidanceError(
                f"reference-target page limit must be 1..{TARGET_PAGE_HARD_BYTES} bytes"
            )
        target_limit = min(strict_limit, TARGET_PAGE_BYTES)
    page_items: list[dict[str, Any]] = []
    next_position = position
    while next_position < len(filtered):
        candidate_items = [*page_items, filtered[next_position]]
        candidate_position = next_position + 1
        candidate = _finalize_target_page(_target_page_payload(
            lane=selected_lane,
            descriptor=descriptor,
            filter_sha256=filter_sha256,
            items=candidate_items,
            next_position=candidate_position,
            total_matches=len(filtered),
        ))
        size = int(candidate["response_bytes"])
        if size <= target_limit:
            page_items = candidate_items
            next_position = candidate_position
            continue
        if not page_items and size <= strict_limit:
            page_items = candidate_items
            next_position = candidate_position
        elif not page_items:
            raise IntentGuidanceError(
                "one reference-target record exceeds the strict page byte limit"
            )
        break
    page = _finalize_target_page(_target_page_payload(
        lane=selected_lane,
        descriptor=descriptor,
        filter_sha256=filter_sha256,
        items=page_items,
        next_position=next_position,
        total_matches=len(filtered),
    ))
    if int(page["response_bytes"]) > strict_limit:
        raise IntentGuidanceError(
            "one reference-target record exceeds the strict page byte limit"
        )
    return page


def _finalize_target_page(page: dict[str, Any]) -> dict[str, Any]:
    """Set response_bytes to the exact canonical UTF-8 size of the envelope."""
    finalized = dict(page)
    finalized["response_bytes"] = 0
    for _ in range(8):
        actual_size = _json_size(finalized)
        if finalized["response_bytes"] == actual_size:
            return finalized
        finalized["response_bytes"] = actual_size
    raise AssertionError("reference-target response byte accounting did not converge")


def _target_page_payload(
    *,
    lane: str,
    descriptor: Mapping[str, Any],
    filter_sha256: str,
    items: Sequence[Mapping[str, Any]],
    next_position: int,
    total_matches: int,
) -> dict[str, Any]:
    eof = next_position >= total_matches
    return {
        "schema_version": TARGET_PAGE_VERSION,
        "lane": lane,
        "catalog_sha256": str(descriptor["sha256"]),
        "filter_sha256": filter_sha256,
        "items": deepcopy(list(items)),
        "item_count": len(items),
        "total_matches": total_matches,
        "next_cursor": None if eof else _encode_target_cursor(
            catalog_sha256=str(descriptor["sha256"]),
            filter_sha256=filter_sha256,
            position=next_position,
        ),
        "eof": eof,
        "response_bytes": 0,
    }


def _encode_target_cursor(
    *, catalog_sha256: str, filter_sha256: str, position: int,
) -> str:
    raw = json.dumps(
        {"v": 1, "c": catalog_sha256, "f": filter_sha256, "p": position},
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_target_cursor(
    cursor: str | None, *, catalog_sha256: str, filter_sha256: str,
) -> int:
    if cursor in (None, ""):
        return 0
    encoded = str(cursor)
    try:
        raw = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True,
        )
        payload = json.loads(raw.decode("ascii"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntentGuidanceError("reference-target cursor is invalid") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"v", "c", "f", "p"}
        or payload.get("v") != 1
        or payload.get("c") != catalog_sha256
        or payload.get("f") != filter_sha256
        or not isinstance(payload.get("p"), int)
        or isinstance(payload.get("p"), bool)
        or payload["p"] < 0
    ):
        raise IntentGuidanceError(
            "reference-target cursor does not match this catalog and filter"
        )
    return int(payload["p"])


def _json_size(value: Any) -> int:
    return len(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))


def _validate_lane(lane: str | None, *, required: bool = False) -> str | None:
    if lane is None and not required:
        return None
    if lane not in INTENT_GUIDANCE_LANES:
        raise IntentGuidanceError(f"unsupported intent-guidance lane: {lane}")
    return lane


def _targets_for_lane(
    artifact: Mapping[str, Any], lane: str | None,
) -> list[dict[str, Any]]:
    targets = artifact.get("reference_targets")
    if not isinstance(targets, list):
        raise IntentGuidanceError("intent-guidance artifact has invalid reference targets")
    return [
        deepcopy(dict(item)) for item in targets
        if isinstance(item, Mapping) and (lane is None or lane in item.get("lanes", ()))
    ]


def resolve_worker_evidence_requests(
    artifact: Mapping[str, Any],
    requests: Iterable[Any],
    *,
    round_number: int,
    toc_getter: Callable[[str], dict[str, Any]] | None = None,
    section_getter: Callable[[str, str], dict[str, Any]] | None = None,
    lane: str | None = None,
) -> tuple[Any, ...]:
    """Resolve the no-shell fallback through the same exact read policy."""
    from arc_llm import EvidenceResponse
    if toc_getter is None or section_getter is None:
        from arc_paper import service

        toc_getter = toc_getter or service.get_parsed_source_toc
        section_getter = section_getter or service.get_parsed_source_section
    responses = []
    for request in requests:
        arguments = dict(getattr(request, "arguments", {}) or {})
        operation = str(getattr(request, "operation", "") or "")
        source_id = str(arguments.get("source_id") or "")
        locator = arguments.get("locator", arguments.get("section"))
        try:
            if operation == LIST_REFERENCE_TARGETS_OPERATION:
                if lane is None:
                    raise IntentGuidanceError(
                        "list-reference-targets requires an intent-guidance lane"
                    )
                page = list_reference_targets(
                    artifact,
                    lane=lane,
                    cursor=arguments.get("cursor"),
                    source_id=source_id or None,
                    query=arguments.get("query"),
                    limit_bytes=arguments.get("limit_bytes"),
                )
                responses.append(EvidenceResponse(
                    str(request.request_id), True, data=page,
                    provenance={
                        "provider": "local-cache", "operation": operation,
                        "lane": lane, "round_number": round_number,
                        "catalog_sha256": page["catalog_sha256"],
                    },
                ))
                continue
            validate_worker_query(
                artifact, operation=operation, source_id=source_id,
                locator=None if operation == "get-parsed-toc" else str(locator or ""),
                lane=lane,
            )
            result = (
                toc_getter(source_id)
                if operation == "get-parsed-toc"
                else section_getter(source_id, str(locator or ""))
            )
            data = _local_result(
                result, source_id=source_id, kind=operation, expected_mapping=False,
            )
            page = _controller_page(
                data,
                offset=arguments.get("offset", 0),
                limit=arguments.get("limit", CONTROLLER_PAGE_BYTES),
            )
            responses.append(EvidenceResponse(
                str(request.request_id), True, data=page,
                provenance={
                    "provider": "local-cache", "operation": operation,
                    "source_id": source_id, "locator": locator,
                    "round_number": round_number,
                },
            ))
        except (IntentGuidanceError, TypeError, ValueError) as exc:
            responses.append(EvidenceResponse(
                str(request.request_id), False, error=str(exc),
                provenance={
                    "provider": "local-cache", "operation": operation,
                    "source_id": source_id, "round_number": round_number,
                },
            ))
    return tuple(responses)


def _controller_page(data: Any, *, offset: Any, limit: Any) -> dict[str, Any]:
    try:
        offset_value = int(offset)
        limit_value = int(limit)
    except (TypeError, ValueError) as exc:
        raise IntentGuidanceError("controller evidence pagination must use integer offsets") from exc
    if offset_value < 0 or limit_value < 1 or limit_value > CONTROLLER_PAGE_BYTES:
        raise IntentGuidanceError(
            f"controller evidence limit must be 1..{CONTROLLER_PAGE_BYTES} bytes"
        )
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    start_offset = min(offset_value, len(payload))
    while start_offset < len(payload) and payload[start_offset] & 0xC0 == 0x80:
        start_offset += 1
    end_offset = min(len(payload), start_offset + limit_value)
    # Extend a tiny page through the current code point so every non-EOF page
    # makes progress even when limit=1 and the next character is multibyte.
    while end_offset < len(payload) and payload[end_offset] & 0xC0 == 0x80:
        end_offset += 1
    text = payload[start_offset:end_offset].decode("utf-8")
    next_offset = end_offset
    return {
        "encoding": "utf-8-json-fragment", "content": text,
        "offset": start_offset, "next_offset": next_offset,
        "size_bytes": len(payload), "eof": next_offset >= len(payload),
    }


def _load_reference(
    source_id: str,
    *,
    parsed_getter: Callable[..., dict[str, Any]],
    toc_getter: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    parsed_result = parsed_getter(source_id, include_document=False)
    parsed = _local_result(parsed_result, source_id=source_id, kind="metadata")
    toc_result = toc_getter(source_id)
    toc_data = _local_result(toc_result, source_id=source_id, kind="TOC", expected_mapping=False)
    if not isinstance(toc_data, list):
        raise IntentGuidanceError(f"local ARC TOC for {source_id} is invalid")
    toc = _compact_toc(toc_data, source_id=source_id)
    integrity = parsed.get("integrity") if isinstance(parsed.get("integrity"), Mapping) else {}
    document_hash = str(
        parsed.get("document_hash")
        or integrity.get("document_hash")
        or parsed.get("source_hash")
        or ""
    ).strip()
    return {
        "source_id": source_id,
        "source_hash": str(parsed.get("source_hash") or "").strip(),
        "document_hash": document_hash,
        "metadata": _sanitized_metadata(parsed),
        "toc": toc,
    }


def _local_result(
    result: Any, *, source_id: str, kind: str, expected_mapping: bool = True
) -> Any:
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        error = result.get("error") if isinstance(result, Mapping) else None
        message = error.get("message") if isinstance(error, Mapping) else None
        raise IntentGuidanceError(
            f"unable to load authorized reference {source_id} {kind} from the local ARC cache: "
            f"{message or 'cache entry not found'}"
        )
    data = result.get("data")
    if expected_mapping and not isinstance(data, Mapping):
        raise IntentGuidanceError(f"local ARC {kind} for {source_id} is invalid")
    return data


def _sanitized_metadata(parsed: Mapping[str, Any]) -> dict[str, Any]:
    nested = parsed.get("metadata") if isinstance(parsed.get("metadata"), Mapping) else {}
    output: dict[str, Any] = {}
    for key in _SAFE_METADATA_FIELDS:
        value = nested.get(key, parsed.get(key))
        cleaned = _compact_metadata_value(value)
        if cleaned not in (None, "", []):
            output[key] = cleaned
    return output


def _compact_metadata_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())[:1000]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        compact = []
        for item in value[:50]:
            if isinstance(item, Mapping):
                name = item.get("name") or item.get("full_name")
                if name:
                    compact.append(" ".join(str(name).split())[:300])
            elif isinstance(item, (str, int, float)) and not isinstance(item, bool):
                compact.append(" ".join(str(item).split())[:300])
        return compact
    return None


def _compact_toc(raw_toc: Sequence[Any], *, source_id: str) -> list[dict[str, Any]]:
    toc: list[dict[str, Any]] = []
    for item in raw_toc:
        if not isinstance(item, Mapping):
            continue
        locator = str(item.get("id") or item.get("section_id") or "").strip()
        title = " ".join(str(item.get("title") or "").split())
        if not locator or not title:
            continue
        entry: dict[str, Any] = {"locator": locator, "title": title[:1000]}
        level = item.get("level")
        if isinstance(level, int) and not isinstance(level, bool) and level >= 0:
            entry["level"] = level
        toc.append(entry)
    if not toc:
        raise IntentGuidanceError(f"local ARC cache for {source_id} has no usable TOC")
    return toc


def _guidance_prompt(semantic_input: Mapping[str, Any]) -> str:
    return (
        "Create one concise global intent-guidance artifact for all content workers in this managed "
        "document run. Base the strategy on the exact user intent. The reference records below contain "
        "sanitized metadata and compact tables of contents only; no section body has been supplied. "
        "When a reference chapter is useful and uniquely determined, select its exact source_id and exact "
        "TOC locator and assign only the applicable lanes. Select every necessary exact target; do not "
        "truncate the result to an arbitrary item count. Workers may later use only get-parsed-toc and "
        "get-parsed-section against the local ARC cache. If the requested chapter cannot be uniquely "
        "identified from the supplied TOC, set resolution_status to ambiguous, return no targets, and do "
        "not guess. The source document is always authoritative for facts, coverage, and structure. A "
        "reference translation may guide terminology, idiom, and style only; its additions, omissions, or "
        "errors must never be inherited. Do not ask to parse, refresh, fetch, search the network, or use any "
        "other tool. Write guidance that applies unchanged to every relevant worker session.\n"
        + json.dumps(semantic_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _validate_model_result(
    raw: Any, *, references: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise IntentGuidanceError("intent-guidance model returned a non-object")
    allowed_keys = {"guidance", "resolution_status", "reference_targets"}
    if set(raw) - allowed_keys:
        raise IntentGuidanceError("intent-guidance model returned unsupported fields")
    guidance = str(raw.get("guidance") or "").strip()
    if not guidance:
        raise IntentGuidanceError("intent-guidance model returned empty guidance")
    if len(guidance) > GUIDANCE_MAX_CHARS:
        raise IntentGuidanceError("intent-guidance model returned oversized guidance")
    if len(guidance.encode("utf-8")) > GUIDANCE_MAX_BYTES:
        raise IntentGuidanceError("intent-guidance model returned oversized UTF-8 guidance")
    status = str(raw.get("resolution_status") or "")
    if status not in {"resolved", "ambiguous"}:
        raise IntentGuidanceError("intent-guidance resolution_status is invalid")
    targets_raw = raw.get("reference_targets", [])
    if not isinstance(targets_raw, list):
        raise IntentGuidanceError("intent-guidance reference_targets must be a list")
    locator_counts = {
        str(reference["source_id"]): _locator_counts(reference["toc"])
        for reference in references
    }
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for raw_target in targets_raw:
        if not isinstance(raw_target, Mapping) or set(raw_target) != {
            "source_id", "locator", "purpose", "lanes"
        }:
            raise IntentGuidanceError("intent-guidance reference target has invalid fields")
        source_id = str(raw_target.get("source_id") or "").strip()
        locator = str(raw_target.get("locator") or "").strip()
        purpose = str(raw_target.get("purpose") or "").strip()
        lanes_raw = raw_target.get("lanes")
        if source_id not in locator_counts:
            raise IntentGuidanceError(f"intent-guidance selected unauthorized source_id: {source_id}")
        if locator_counts[source_id].get(locator, 0) != 1:
            raise IntentGuidanceError(
                f"intent-guidance locator is missing or non-unique in {source_id}: {locator}"
            )
        if not purpose:
            raise IntentGuidanceError("intent-guidance target purpose is empty")
        if len(purpose) > 1000:
            raise IntentGuidanceError("intent-guidance target purpose is oversized")
        if not isinstance(lanes_raw, list) or not lanes_raw:
            raise IntentGuidanceError("intent-guidance target lanes must be a non-empty list")
        lanes = [str(lane) for lane in lanes_raw]
        if len(set(lanes)) != len(lanes) or any(lane not in INTENT_GUIDANCE_LANES for lane in lanes):
            raise IntentGuidanceError("intent-guidance target contains an invalid or duplicate lane")
        identity = (source_id, locator, tuple(lanes))
        if identity in seen:
            raise IntentGuidanceError("intent-guidance contains a duplicate reference target")
        seen.add(identity)
        targets.append({
            "source_id": source_id,
            "locator": locator,
            "purpose": purpose,
            "lanes": lanes,
        })
    if status == "ambiguous" and targets:
        raise IntentGuidanceError("ambiguous intent-guidance must not guess reference targets")
    _target_catalog_metadata(targets)
    return {"guidance": guidance, "resolution_status": status, "reference_targets": targets}


def _locator_counts(toc: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in toc:
        locator = str(entry.get("locator") or "")
        counts[locator] = counts.get(locator, 0) + 1
    return counts


def _build_artifact(
    normalized: Mapping[str, Any],
    *,
    semantic_sha256: str,
    user_intent_sha256: str,
    references: Sequence[Mapping[str, Any]],
    artifact_dir: Path,
    migration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    targets = deepcopy(list(normalized["reference_targets"]))
    catalog = _write_target_catalog(targets, artifact_dir=artifact_dir)
    artifact: dict[str, Any] = {
        "schema_version": INTENT_GUIDANCE_VERSION,
        "semantic_input_sha256": semantic_sha256,
        "user_intent_sha256": user_intent_sha256,
        "output_sha256": sha256_json(dict(normalized)),
        "resolution_status": normalized["resolution_status"],
        "guidance": normalized["guidance"],
        "reference_targets": targets,
        "reference_sources": deepcopy(list(references)),
        "target_catalog": catalog,
    }
    artifact["worker_payload"] = _worker_payload_unvalidated(artifact, lane=None)
    if migration is not None:
        artifact["migration"] = deepcopy(dict(migration))
    return artifact


def _write_target_catalog(
    targets: Sequence[Mapping[str, Any]], *, artifact_dir: Path,
) -> dict[str, Any]:
    catalog = _target_catalog_metadata(targets)
    for descriptor in catalog["indexes"].values():
        key = str(descriptor["key"])
        projected = [
            item for item in targets
            if key == "all" or key in item.get("lanes", ())
        ]
        text = _target_catalog_text(projected)
        if len(text.encode("utf-8")) != descriptor["serialized_bytes"]:
            raise AssertionError("target catalog byte accounting drifted")
        write_text(artifact_dir / str(descriptor["catalog_file"]), text)
    return catalog


def _target_catalog_files_valid(
    artifact: Mapping[str, Any], *, artifact_dir: Path,
) -> bool:
    catalog = artifact.get("target_catalog")
    descriptors = catalog.get("indexes") if isinstance(catalog, Mapping) else None
    if not isinstance(descriptors, Mapping):
        return False
    for descriptor in descriptors.values():
        if not isinstance(descriptor, Mapping):
            return False
        relative = str(descriptor.get("catalog_file") or "")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            return False
        path = artifact_dir / relative
        try:
            payload = path.read_bytes()
        except OSError:
            return False
        if (
            len(payload) != descriptor.get("serialized_bytes")
            or hashlib.sha256(payload).hexdigest() != descriptor.get("sha256")
        ):
            return False
    return True


def _target_catalog_metadata(
    targets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_text = _target_catalog_text(targets)
    all_bytes = len(all_text.encode("utf-8"))
    if all_bytes > TARGET_CATALOG_MAX_BYTES:
        raise IntentGuidanceError(
            f"intent-guidance target catalog exceeds {TARGET_CATALOG_MAX_BYTES} UTF-8 bytes"
        )
    indexes: dict[str, Any] = {}
    for key in ("all", *INTENT_GUIDANCE_LANES):
        projected = [
            item for item in targets
            if key == "all" or key in item.get("lanes", ())
        ]
        text = _target_catalog_text(projected)
        payload = text.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        indexes[key] = {
            "schema_version": TARGET_CATALOG_VERSION,
            "key": key,
            "lane": None if key == "all" else key,
            "target_count": len(projected),
            "serialized_bytes": len(payload),
            "sha256": digest,
            "catalog_file": f"target-catalogs/{key}.{digest}.jsonl",
        }
    return {
        "schema_version": TARGET_CATALOG_VERSION,
        "max_serialized_bytes": TARGET_CATALOG_MAX_BYTES,
        "indexes": indexes,
    }


def _target_catalog_text(targets: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) + "\n"
        for item in targets
    )


def _worker_payload_unvalidated(
    artifact: Mapping[str, Any], *, lane: str | None,
) -> dict[str, Any]:
    key = "all" if lane is None else lane
    catalog = artifact["target_catalog"]
    descriptor = deepcopy(dict(catalog["indexes"][key]))
    descriptor.pop("catalog_file", None)
    payload: dict[str, Any] = {
        "guidance": str(artifact["guidance"]),
        "reference_target_catalog": descriptor,
    }
    if descriptor["serialized_bytes"] <= TARGET_INLINE_BYTES:
        payload["reference_targets"] = _targets_for_lane(artifact, lane)
        payload["reference_target_catalog"]["inline"] = True
    else:
        payload["reference_target_catalog"].update({
            "inline": False,
            "list_operations": {
                "worker_cli": POLICY_TARGETS_OPERATION,
                "controller_evidence": LIST_REFERENCE_TARGETS_OPERATION,
            },
            "page_target_bytes": TARGET_PAGE_BYTES,
            "page_hard_bytes": TARGET_PAGE_HARD_BYTES,
        })
    return payload


def _artifact_valid(
    value: Any,
    *,
    semantic_sha256: str,
    user_intent_sha256: str,
    references: Sequence[Mapping[str, Any]],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("schema_version") != INTENT_GUIDANCE_VERSION:
        return False
    if value.get("semantic_input_sha256") != semantic_sha256:
        return False
    if value.get("user_intent_sha256") != user_intent_sha256:
        return False
    try:
        normalized = _validate_model_result(_artifact_model_result(value), references=references)
    except IntentGuidanceError:
        return False
    expected_catalog = _target_catalog_metadata(normalized["reference_targets"])
    candidate = dict(value)
    candidate["target_catalog"] = expected_catalog
    expected_payload = _worker_payload_unvalidated(candidate, lane=None)
    return bool(
        value.get("output_sha256") == sha256_json(normalized)
        and value.get("reference_sources") == list(references)
        and value.get("target_catalog") == expected_catalog
        and value.get("worker_payload") == expected_payload
    )


def _legacy_artifact_valid(
    value: Any,
    *,
    semantic_sha256: str,
    user_intent_sha256: str,
    references: Sequence[Mapping[str, Any]],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("schema_version") != LEGACY_INTENT_GUIDANCE_VERSION:
        return False
    if value.get("semantic_input_sha256") != semantic_sha256:
        return False
    if value.get("user_intent_sha256") != user_intent_sha256:
        return False
    try:
        normalized = _validate_model_result(_artifact_model_result(value), references=references)
    except IntentGuidanceError:
        return False
    expected_payload = {
        "guidance": normalized["guidance"],
        "reference_targets": normalized["reference_targets"],
    }
    return bool(
        value.get("output_sha256") == sha256_json(normalized)
        and value.get("reference_sources") == list(references)
        and value.get("worker_payload") == expected_payload
    )


def _validate_worker_artifact(artifact: Mapping[str, Any]) -> None:
    if not isinstance(artifact, Mapping) or artifact.get("schema_version") != INTENT_GUIDANCE_VERSION:
        raise IntentGuidanceError("invalid intent-guidance artifact")
    references = artifact.get("reference_sources")
    if not isinstance(references, list):
        raise IntentGuidanceError("intent-guidance artifact has invalid reference sources")
    normalized = _validate_model_result(_artifact_model_result(artifact), references=references)
    if normalized["resolution_status"] != "resolved":
        raise IntentGuidanceError("ambiguous intent-guidance cannot bootstrap a worker")
    expected_catalog = _target_catalog_metadata(normalized["reference_targets"])
    if artifact.get("target_catalog") != expected_catalog:
        raise IntentGuidanceError("intent-guidance target catalog does not match its accepted output")
    expected = _worker_payload_unvalidated(artifact, lane=None)
    if artifact.get("worker_payload") != expected:
        raise IntentGuidanceError("intent-guidance worker payload does not match its accepted output")


def _artifact_model_result(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "guidance": artifact.get("guidance"),
        "resolution_status": artifact.get("resolution_status"),
        "reference_targets": artifact.get("reference_targets"),
    }


def _require_resolved(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if artifact.get("resolution_status") == "ambiguous":
        raise IntentGuidanceAmbiguousError(
            "reference chapter is ambiguous; refine the user intent instead of guessing a locator",
            artifact=artifact,
        )
    return dict(artifact)


def _authorized_source_ids(values: Iterable[str]) -> list[str]:
    source_ids: list[str] = []
    for value in values:
        source_id = str(value or "").strip()
        if not source_id:
            raise IntentGuidanceError("context paper IDs must not be empty")
        if len(source_id) > 512 or _CONTROL_CHARACTERS.search(source_id):
            raise IntentGuidanceError("context paper ID contains unsafe characters")
        if source_id in source_ids:
            raise IntentGuidanceError(f"duplicate context paper ID: {source_id}")
        source_ids.append(source_id)
    return sorted(source_ids)


def _required_text(value: str, field: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise IntentGuidanceError(f"{field} must not be empty when user_intent is provided")
    return cleaned
