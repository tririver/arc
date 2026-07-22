from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PAPER_ACCESS_POLICY_VERSION = "arc.paper.worker-read-policy.v2"
POLICY_TARGETS_OPERATION = "policy-targets"


def canonical_paper_access_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a controller policy into the v2 worker-file contract.

    The input accepts both the portable descriptor used by ARC workflows and
    the older ``operations``/``targets`` mapping.  The returned object has a
    closed, deterministic shape suitable for content addressing.
    """

    if not isinstance(policy, Mapping):
        raise ValueError("paper_access_policy must be a mapping")
    operations = _string_list(
        policy.get("operations", policy.get("allowed_operations", ())),
        field="operations",
    )
    if POLICY_TARGETS_OPERATION not in operations:
        operations.append(POLICY_TARGETS_OPERATION)

    authorized_sources = _string_list(
        policy.get("authorized_source_ids", ()), field="authorized_source_ids"
    )
    targets: list[dict[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()

    raw_targets = policy.get("targets")
    if isinstance(raw_targets, Mapping):
        for source_id, source_policy in raw_targets.items():
            normalized_source_id = _required_text(source_id, "targets source_id")
            if normalized_source_id not in authorized_sources:
                authorized_sources.append(normalized_source_id)
            if not isinstance(source_policy, Mapping):
                raise ValueError("paper_access_policy target source entries must be mappings")
            sections = _string_list(source_policy.get("sections", ()), field="sections")
            for locator in sections:
                _append_target(
                    targets,
                    seen_targets,
                    source_id=normalized_source_id,
                    locator=locator,
                    purpose="",
                )
    elif raw_targets is not None:
        if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
            raise ValueError("paper_access_policy targets must be a mapping or list")
        for item in raw_targets:
            _append_input_target(targets, seen_targets, authorized_sources, item)

    raw_section_targets = policy.get("authorized_section_targets", ())
    if not isinstance(raw_section_targets, Sequence) or isinstance(
        raw_section_targets, (str, bytes)
    ):
        raise ValueError("authorized_section_targets must be a list")
    for item in raw_section_targets:
        _append_input_target(targets, seen_targets, authorized_sources, item)

    if not operations:
        raise ValueError("paper_access_policy must authorize at least one operation")
    return {
        "schema_version": PAPER_ACCESS_POLICY_VERSION,
        "operations": operations,
        "authorized_source_ids": authorized_sources,
        "targets": targets,
    }


def validate_canonical_paper_access_policy(policy: Any) -> dict[str, Any]:
    """Validate an on-disk v2 policy without accepting descriptor aliases."""

    if not isinstance(policy, dict) or set(policy) != {
        "schema_version", "operations", "authorized_source_ids", "targets"
    }:
        raise ValueError("worker read policy has an invalid top-level shape")
    if policy.get("schema_version") != PAPER_ACCESS_POLICY_VERSION:
        raise ValueError("worker read policy schema_version is not supported")
    operations = _string_list(policy.get("operations"), field="operations")
    if operations != policy.get("operations") or not operations:
        raise ValueError("worker read policy operations are not canonical")
    sources = _string_list(
        policy.get("authorized_source_ids"), field="authorized_source_ids"
    )
    if sources != policy.get("authorized_source_ids"):
        raise ValueError("worker read policy authorized_source_ids are not canonical")
    raw_targets = policy.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("worker read policy targets must be a list")
    targets: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_targets:
        if not isinstance(item, dict) or set(item) != {"source_id", "locator", "purpose"}:
            raise ValueError("worker read policy target has an invalid shape")
        source_id = _required_text(item.get("source_id"), "target source_id")
        locator = _required_text(item.get("locator"), "target locator")
        purpose = item.get("purpose")
        if not isinstance(purpose, str):
            raise ValueError("worker read policy target purpose must be a string")
        if source_id not in sources:
            raise ValueError("worker read policy target source is not authorized")
        key = (source_id, locator)
        if key in seen:
            raise ValueError("worker read policy targets must be unique")
        seen.add(key)
        targets.append({"source_id": source_id, "locator": locator, "purpose": purpose})
    if targets != raw_targets:
        raise ValueError("worker read policy targets are not canonical")
    return policy


def _append_input_target(
    targets: list[dict[str, str]],
    seen: set[tuple[str, str]],
    authorized_sources: list[str],
    item: Any,
) -> None:
    if not isinstance(item, Mapping):
        raise ValueError("paper_access_policy targets must contain mappings")
    source_id = _required_text(item.get("source_id"), "target source_id")
    locator = _required_text(item.get("locator"), "target locator")
    purpose_value = item.get("purpose", "")
    if not isinstance(purpose_value, str):
        raise ValueError("paper_access_policy target purpose must be a string")
    if source_id not in authorized_sources:
        authorized_sources.append(source_id)
    _append_target(
        targets,
        seen,
        source_id=source_id,
        locator=locator,
        purpose=purpose_value,
    )


def _append_target(
    targets: list[dict[str, str]],
    seen: set[tuple[str, str]],
    *,
    source_id: str,
    locator: str,
    purpose: str,
) -> None:
    key = (source_id, locator)
    if key in seen:
        return
    seen.add(key)
    targets.append({"source_id": source_id, "locator": locator, "purpose": purpose})


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"paper_access_policy {field} must be a list")
    result: list[str] = []
    for item in value:
        text = _required_text(item, field)
        if text not in result:
            result.append(text)
    return result


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"paper_access_policy {field} values must be non-empty strings")
    return value
