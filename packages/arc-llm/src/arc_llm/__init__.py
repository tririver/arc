from __future__ import annotations

from importlib import import_module

from .call_record import ARC_LLM_CALL_RECORD_FIELD, ARC_LLM_CALL_RECORD_SCHEMA_VERSION, strip_arc_llm_call_records
from .host import HostDetection, ProviderSelection, detect_host, select_llm_provider
from .model import ModelTierError, resolve_model


_LAZY_EXPORTS = {
    "LLMConfig": ("arc_llm.runner", "LLMConfig"),
    "resolve_llm_config": ("arc_llm.runner", "resolve_llm_config"),
    "run_json": ("arc_llm.runner", "run_json"),
    "run_text": ("arc_llm.runner", "run_text"),
    "run_text_result": ("arc_llm.runner", "run_text_result"),
    "run_proposers_reviewer_batch": ("arc_llm.proposers_reviewer.runner", "run_proposers_reviewer_batch"),
    "run_proposers_reviewer_bench": ("arc_llm.proposers_reviewer_bench.runner", "run_proposers_reviewer_bench"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "HostDetection",
    "LLMConfig",
    "ModelTierError",
    "ProviderSelection",
    "ARC_LLM_CALL_RECORD_FIELD",
    "ARC_LLM_CALL_RECORD_SCHEMA_VERSION",
    "detect_host",
    "resolve_llm_config",
    "resolve_model",
    "run_proposers_reviewer_batch",
    "run_proposers_reviewer_bench",
    "run_json",
    "run_text",
    "run_text_result",
    "select_llm_provider",
    "strip_arc_llm_call_records",
]
