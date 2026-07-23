from __future__ import annotations

from importlib import import_module

from .call_record import ARC_LLM_CALL_RECORD_FIELD, ARC_LLM_CALL_RECORD_SCHEMA_VERSION, strip_arc_llm_call_records
from .call_checkpoint import SupervisedNativeResumeAuthorization
from .cancellation import install_signal_cancel_chain
from .evidence import (
    EVIDENCE_REQUESTS_FIELD,
    MAX_EVIDENCE_ROUNDS,
    EvidenceControllerCallback,
    EvidenceProtocolError,
    EvidenceRequest,
    EvidenceResponse,
    allow_evidence_requests,
    evidence_requests_from_output,
    resolve_evidence_round,
)
from .evidence_journal import (
    SCHEMA_VERSION as EVIDENCE_JOURNAL_SCHEMA_VERSION,
    STATES as EVIDENCE_JOURNAL_STATES,
    EvidenceExecution,
    EvidenceJournal,
    EvidenceJournalAction,
    EvidenceJournalAddress,
    EvidenceJournalContext,
    EvidenceJournalCorruptError,
    EvidenceJournalError,
    EvidenceJournalRecoveryError,
    EvidenceJournalStaleError,
    EvidenceOperationPolicy,
    canonical_hash as evidence_identity_hash,
)
from .failure_classification import classify_provider_diagnostic, disposition_error_kwargs
from .host import HostDetection, ProviderSelection, detect_host, select_llm_provider
from .model import ModelTierError, resolve_model
from .recovery_context import LLMRecoveryContext, read_recovery_context
from .runtime_manifest import (
    RUNTIME_MANIFEST_VERSION,
    runtime_manifest,
    runtime_manifest_fingerprint,
)
from .providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMFailureDisposition,
    LLMSubmissionState,
    LLMWorkerError,
    failure_disposition,
)
from .safety import LLMCircuitOpen, LLMSafetyController


_LAZY_EXPORTS = {
    "LLMConfig": ("arc_llm.runner", "LLMConfig"),
    "resolve_llm_config": ("arc_llm.runner", "resolve_llm_config"),
    "run_json": ("arc_llm.runner", "run_json"),
    "run_json_result": ("arc_llm.runner", "run_json_result"),
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
    "EVIDENCE_REQUESTS_FIELD",
    "EvidenceControllerCallback",
    "EvidenceExecution",
    "EVIDENCE_JOURNAL_SCHEMA_VERSION",
    "EVIDENCE_JOURNAL_STATES",
    "EvidenceJournal",
    "EvidenceJournalAction",
    "EvidenceJournalAddress",
    "EvidenceJournalContext",
    "EvidenceJournalCorruptError",
    "EvidenceJournalError",
    "EvidenceJournalRecoveryError",
    "EvidenceJournalStaleError",
    "EvidenceOperationPolicy",
    "EvidenceProtocolError",
    "EvidenceRequest",
    "EvidenceResponse",
    "LLMConfig",
    "LLMAbortScope",
    "LLMCircuitOpen",
    "LLMFailureCategory",
    "LLMFailureDisposition",
    "LLMRecoveryContext",
    "LLMSafetyController",
    "LLMSubmissionState",
    "LLMWorkerError",
    "RUNTIME_MANIFEST_VERSION",
    "ModelTierError",
    "MAX_EVIDENCE_ROUNDS",
    "ProviderSelection",
    "SupervisedNativeResumeAuthorization",
    "ARC_LLM_CALL_RECORD_FIELD",
    "ARC_LLM_CALL_RECORD_SCHEMA_VERSION",
    "detect_host",
    "classify_provider_diagnostic",
    "disposition_error_kwargs",
    "allow_evidence_requests",
    "evidence_requests_from_output",
    "evidence_identity_hash",
    "failure_disposition",
    "install_signal_cancel_chain",
    "resolve_llm_config",
    "resolve_evidence_round",
    "resolve_model",
    "read_recovery_context",
    "runtime_manifest",
    "runtime_manifest_fingerprint",
    "run_proposers_reviewer_batch",
    "run_proposers_reviewer_bench",
    "run_json",
    "run_json_result",
    "run_text",
    "run_text_result",
    "select_llm_provider",
    "strip_arc_llm_call_records",
]
