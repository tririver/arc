from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_VOLATILE_KEYS = {
    "attempt_number",
    "round_number",
    "active_proposer_ids",
    "locked_outputs",
    "retry_feedback",
    "idea_id",
    "variant_id",
}


@dataclass(frozen=True)
class ContextPack:
    static: dict[str, Any]
    volatile: dict[str, Any]
    omitted: dict[str, Any]


def split_caller_context(caller_context: Mapping[str, Any], cache_context: Mapping[str, Any] | None) -> ContextPack:
    static_keys = _string_list(cache_context.get("static_caller_context_keys")) if cache_context else []
    volatile_keys = _string_list(cache_context.get("volatile_caller_context_keys")) if cache_context else []
    volatile_set = set(volatile_keys) | DEFAULT_VOLATILE_KEYS
    static: dict[str, Any] = {}
    volatile: dict[str, Any] = {}
    for key in static_keys:
        if key in caller_context and key not in volatile_set:
            static[key] = copy.deepcopy(caller_context[key])
    for key, value in caller_context.items():
        if key in volatile_set:
            volatile[key] = copy.deepcopy(value)
        elif key not in static:
            static[key] = copy.deepcopy(value)
    return ContextPack(static=static, volatile=volatile, omitted={})


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]
