from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import read_json, sha256_json, write_json


NORMALIZER_VERSION = "arc.companion.response-normalizer.v1"
RECEIPT_SCHEMA_VERSION = "arc.companion.response-normalization-receipt.v1"


class ResponseNormalizationError(RuntimeError):
    """A complete response cannot be projected without weakening its contract."""

    def __init__(
        self,
        code: str,
        receipt: Mapping[str, Any],
        *,
        detail: str | None = None,
    ) -> None:
        self.code = code
        self.receipt = dict(receipt)
        message = f"response normalization rejected: {code}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


def normalize_complete_response(
    value: Mapping[str, Any],
    collection_field: str,
    expected_ids: Sequence[str],
    id_extractor: Callable[[Mapping[str, Any]], Any],
    schema_validator: Callable[[Mapping[str, Any]], Any],
    invariant_validator: Callable[[Mapping[str, Any]], Any],
    validator_version: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project a complete response onto an exact ordered logical-ID contract.

    This function consumes an already parsed, T06-selected mapping. It neither
    parses provider output nor invokes a model or formatter.
    """

    original = dict(value) if isinstance(value, Mapping) else {}
    original_sha256 = sha256_json(value)
    expected = (
        list(expected_ids)
        if isinstance(expected_ids, Sequence)
        and not isinstance(expected_ids, (str, bytes, bytearray))
        else []
    )
    receipt = _base_receipt(
        collection_field=collection_field,
        expected_ids=expected,
        validator_version=validator_version,
        original_sha256=original_sha256,
    )

    if (
        not isinstance(value, Mapping)
        or not isinstance(collection_field, str)
        or not collection_field
        or not isinstance(validator_version, str)
        or not validator_version
    ):
        _reject(receipt, "invalid_normalization_contract")
    if (
        not expected
        or any(type(item) is not str or not item for item in expected)
        or len(set(expected)) != len(expected)
    ):
        _reject(receipt, "invalid_expected_ids")

    raw_items = value.get(collection_field)
    if not isinstance(raw_items, list):
        _reject(receipt, "invalid_collection")

    expected_set = set(expected)
    by_id: dict[str, list[tuple[Mapping[str, Any], str]]] = {
        item: [] for item in expected
    }
    returned_ids: list[str] = []
    discarded_ids: list[str] = []
    discarded_hashes: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            receipt["returned_ids"] = returned_ids
            _reject(receipt, "invalid_returned_item")
        try:
            logical_id = id_extractor(raw_item)
        except Exception:
            receipt["returned_ids"] = returned_ids
            _reject(receipt, "invalid_returned_id")
        if type(logical_id) is not str or not logical_id:
            receipt["returned_ids"] = returned_ids
            _reject(receipt, "invalid_returned_id")
        item_sha256 = sha256_json(raw_item)
        returned_ids.append(logical_id)
        if logical_id in expected_set:
            by_id[logical_id].append((raw_item, item_sha256))
        else:
            discarded_ids.append(logical_id)
            discarded_hashes.append(item_sha256)

    receipt["returned_ids"] = returned_ids
    receipt["discarded_ids"] = discarded_ids
    receipt["discarded_item_sha256s"] = discarded_hashes
    missing = [logical_id for logical_id in expected if not by_id[logical_id]]
    receipt["missing_ids"] = missing
    if missing:
        _reject(receipt, "missing_expected_ids")

    collapsed_ids: list[str] = []
    conflicting_ids: list[str] = []
    selected: dict[str, Mapping[str, Any]] = {}
    for logical_id in expected:
        candidates = by_id[logical_id]
        hashes = {item_sha256 for _, item_sha256 in candidates}
        if len(hashes) != 1:
            conflicting_ids.append(logical_id)
            continue
        selected[logical_id] = candidates[0][0]
        if len(candidates) > 1:
            collapsed_ids.append(logical_id)
    receipt["collapsed_ids"] = collapsed_ids
    receipt["conflicting_ids"] = conflicting_ids
    if conflicting_ids:
        _reject(receipt, "conflicting_duplicate_ids")

    projected = {
        **original,
        collection_field: [dict(selected[logical_id]) for logical_id in expected],
    }
    receipt["projected_response_sha256"] = sha256_json(projected)
    try:
        schema_result = schema_validator(projected)
    except Exception:
        receipt["schema_valid"] = False
        _reject(receipt, "schema_validation_failed")
    if schema_result is False:
        receipt["schema_valid"] = False
        _reject(receipt, "schema_validation_failed")
    receipt["schema_valid"] = True

    try:
        invariant_result = invariant_validator(projected)
    except Exception as exc:
        receipt["invariant_valid"] = False
        _reject(receipt, "invariant_validation_failed", detail=str(exc))
    if invariant_result is False:
        receipt["invariant_valid"] = False
        _reject(receipt, "invariant_validation_failed")
    receipt["invariant_valid"] = True

    reordered = returned_ids != expected and not discarded_ids and not collapsed_ids
    if discarded_ids and collapsed_ids:
        reason = "projected_and_collapsed"
    elif discarded_ids:
        reason = "projected_unknown_ids"
    elif collapsed_ids:
        reason = "collapsed_duplicate_ids"
    elif reordered:
        reason = "reordered_expected_ids"
    else:
        reason = "exact_expected_ids"
    receipt["decision"] = "accepted"
    receipt["reason_code"] = reason
    return projected, receipt


def normalize_complete_response_with_receipt(
    value: Mapping[str, Any],
    collection_field: str,
    expected_ids: Sequence[str],
    id_extractor: Callable[[Mapping[str, Any]], Any],
    schema_validator: Callable[[Mapping[str, Any]], Any],
    invariant_validator: Callable[[Mapping[str, Any]], Any],
    validator_version: str,
    *,
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize and atomically publish, or verify, a body-free receipt."""

    projected: dict[str, Any] | None = None
    failure: ResponseNormalizationError | None = None
    try:
        projected, receipt = normalize_complete_response(
            value,
            collection_field,
            expected_ids,
            id_extractor,
            schema_validator,
            invariant_validator,
            validator_version,
        )
    except ResponseNormalizationError as exc:
        receipt = exc.receipt
        failure = exc

    if receipt_path.is_file():
        try:
            existing = read_json(receipt_path)
        except (OSError, ValueError, TypeError) as exc:
            raise ResponseNormalizationError(
                "receipt_replay_mismatch",
                _replay_failure_receipt(receipt),
            ) from exc
        if not isinstance(existing, Mapping) or sha256_json(existing) != sha256_json(receipt):
            raise ResponseNormalizationError(
                "receipt_replay_mismatch",
                _replay_failure_receipt(receipt),
            )
    else:
        write_json(receipt_path, receipt)

    if failure is not None:
        raise failure
    assert projected is not None
    return projected, receipt


def _base_receipt(
    *,
    collection_field: Any,
    expected_ids: list[Any],
    validator_version: Any,
    original_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "validator_version": validator_version,
        "decision": "rejected",
        "reason_code": "invalid_normalization_contract",
        "collection_field": collection_field,
        "expected_ids": expected_ids,
        "returned_ids": [],
        "discarded_ids": [],
        "discarded_item_sha256s": [],
        "collapsed_ids": [],
        "missing_ids": [],
        "conflicting_ids": [],
        "original_response_sha256": original_sha256,
        "projected_response_sha256": None,
        "schema_valid": None,
        "invariant_valid": None,
    }


def _reject(
    receipt: dict[str, Any], code: str, *, detail: str | None = None,
) -> None:
    receipt["decision"] = "rejected"
    receipt["reason_code"] = code
    raise ResponseNormalizationError(code, receipt, detail=detail)


def _replay_failure_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(receipt),
        "decision": "rejected",
        "reason_code": "receipt_replay_mismatch",
    }
