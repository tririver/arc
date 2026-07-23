"""Controller-owned, structured ARC-paper broker for companion workers.

The broker accepts only T09 catalog operations plus its two local controls. It
owns authorization, durable dispatch receipts, cache-session promotion, and
content-addressed response paging; worker-provided command lines and paths are
never part of this boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Mapping, Sequence
from urllib.parse import quote, urlsplit

from arc_llm import (
    EvidenceExecution,
    EvidenceJournal,
    EvidenceJournalContext,
    EvidenceJournalRecoveryError,
    EvidenceOperationPolicy,
    EvidenceRequest,
    EvidenceResponse,
    evidence_identity_hash,
)
from arc_paper.capabilities import (
    CATALOG_SCHEMA_VERSION,
    OPERATION_CATALOG,
    OperationSpec,
    dispatch_operation,
    get_operation_spec,
)
from arc_paper.ids import normalize_paper_id, paper_ids_safe_dir_name
from arc_paper.parse.document import RICH_DOCUMENT_PARSER_VERSION
from arc_paper.broker_jobs import (
    BrokerJobExecutionContext,
    BrokerJobManager,
    BrokerJobTicket,
)
from arc_llm.budget import SharedBudget
from arc_paper.worker_session import PromotionResult, WorkerCacheSession


BROKER_SCHEMA_VERSION = "arc.companion.paper-broker.v1"
BROKER_POLICY_SCHEMA_VERSION = "arc.companion.paper-broker-policy.v1"
BROKER_RECEIPT_SCHEMA_VERSION = "arc.companion.paper-broker-receipt.v1"
BROKER_HANDLE_SCHEMA_VERSION = "arc.companion.paper-broker-handle.v1"
BROKER_PAGE_SCHEMA_VERSION = "arc.companion.paper-broker-page.v1"
BROKER_OBJECT_SCHEMA_VERSION = "arc.companion.paper-broker-object.v1"
CONTROLLER_AGGREGATE_OBJECT_SCHEMA_VERSION = (
    "arc.companion.paper-broker-controller-aggregate-object.v1"
)
CONTROLLER_AGGREGATE_LOOKUP_SCHEMA_VERSION = (
    "arc.companion.paper-broker-controller-aggregate-lookup.v1"
)
LIST_REFERENCE_TARGETS_OPERATION = "list-reference-targets"
ARTIFACT_READ_OPERATION = "artifact-read"
MAX_INLINE_BYTES = 64 * 1024
MAX_PAGE_BYTES = 46 * 1024
MAX_ROUND_RESPONSE_BYTES = 64 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_PURE_CACHE_PATH_FIELDS = {"cache_path", "rich_cache_path", "summary_path"}


class PaperBrokerError(RuntimeError):
    """Stable local Broker failure with response disposition metadata."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: str = "local",
        retryable: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.category = category
        self.retryable = retryable
        self.provenance = dict(provenance or {})


@dataclass(frozen=True)
class PaperBrokerPolicy:
    schema_version: str
    access: str
    allowed_operation_ids: tuple[str, ...]
    authorized_source_ids: tuple[str, ...]
    authorized_sections: tuple[tuple[str, str], ...]
    paper_network_authorized: bool
    catalog_schema_version: str
    catalog_sha256: str
    direct_shell_requested: bool
    direct_shell_available: bool
    direct_shell_probe_id: str
    policy_sha256: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_operation_ids"] = list(self.allowed_operation_ids)
        value["authorized_source_ids"] = list(self.authorized_source_ids)
        value["authorized_sections"] = [
            {"source_id": source_id, "section": section}
            for source_id, section in self.authorized_sections
        ]
        return value


def default_operation_specs(
    *, include_managed_jobs: bool = False,
) -> tuple[OperationSpec, ...]:
    """Return normal Controller operations, excluding T13/admin boundaries."""

    return tuple(
        spec
        for spec in sorted(OPERATION_CATALOG.values(), key=lambda item: item.name)
        if not (
            spec.admin
            or spec.destructive
            or (
                not include_managed_jobs
                and (spec.uses_llm or spec.is_job)
            )
        )
    )


def compact_catalog(
    operation_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    allowed = set(
        (spec.operation_id for spec in default_operation_specs())
        if operation_ids is None
        else operation_ids
    )
    operations = []
    for spec in sorted(OPERATION_CATALOG.values(), key=lambda item: item.name):
        if spec.operation_id not in allowed:
            continue
        operations.append({
            "id": spec.operation_id,
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "aliases": list(spec.aliases),
            "parameters": dict(spec.parameter_schema),
            "classification": {
                "network": spec.network_access,
                "cache": spec.cache_access,
                "recovery": spec.recovery_class,
            },
        })
    controls = [] if not allowed else [
        {
            "id": "arc-companion.paper-broker.list-reference-targets.v1",
            "name": LIST_REFERENCE_TARGETS_OPERATION,
            "description": (
                "List the frozen reference targets authorized for this worker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cursor": {"type": "string"},
                    "source_id": {"type": "string"},
                    "query": {"type": "string"},
                    "limit_bytes": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
            "classification": {
                "network": "none", "cache": "none", "recovery": "idempotent",
            },
        },
        {
            "id": "arc-companion.paper-broker.artifact-read.v1",
            "name": ARTIFACT_READ_OPERATION,
            "description": (
                "Read one bounded page from a Controller-owned result artifact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "handle_id": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {
                        "type": "integer", "minimum": 1,
                        "maximum": MAX_PAGE_BYTES,
                    },
                },
                "required": ["handle_id"],
                "additionalProperties": False,
            },
            "classification": {
                "network": "none", "cache": "read", "recovery": "idempotent",
            },
        },
    ]
    document = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "broker_schema_version": BROKER_SCHEMA_VERSION,
        "controls": controls,
        "operations": operations,
    }
    document["catalog_sha256"] = _canonical_hash(document)
    return document


def build_paper_broker_policy(
    *,
    access: str = "full",
    allowed_operations: Iterable[str] | None = None,
    authorized_source_ids: Iterable[str] = (),
    authorized_sections: Iterable[Mapping[str, Any] | tuple[str, str]] = (),
    paper_network_authorized: bool | None = None,
    direct_shell_requested: bool = False,
    nested_shell_capability: Mapping[str, Any] | Any | None = None,
    managed_job_route: bool = False,
) -> PaperBrokerPolicy:
    normalized_access = str(access or "").strip().lower()
    if normalized_access not in {"none", "full"}:
        raise ValueError("arc_paper_access must be none or full")
    specs: list[OperationSpec] = []
    requested = list(allowed_operations) if allowed_operations is not None else [
        spec.name for spec in default_operation_specs(
            include_managed_jobs=managed_job_route,
        )
    ]
    for operation in requested:
        spec = get_operation_spec(str(operation))
        if spec is None:
            raise ValueError(f"unknown ARC-paper operation: {operation}")
        if spec.admin or spec.destructive:
            raise ValueError(f"normal Broker policy cannot authorize {spec.name}")
        if (spec.uses_llm or spec.is_job) and not managed_job_route:
            continue
        if spec.operation_id not in {item.operation_id for item in specs}:
            specs.append(spec)
    if normalized_access == "none":
        specs = []
    catalog = compact_catalog(spec.operation_id for spec in specs)
    sources = tuple(sorted(dict.fromkeys(
        normalized for value in authorized_source_ids
        if (normalized := normalize_paper_id(str(value)))
    )))
    sections: list[tuple[str, str]] = []
    for raw in authorized_sections:
        if isinstance(raw, Mapping):
            source_id = normalize_paper_id(str(raw.get("source_id") or ""))
            section = str(raw.get("section", raw.get("locator", "")) or "").strip()
        else:
            source_id = normalize_paper_id(str(raw[0]))
            section = str(raw[1]).strip()
        if not source_id or not section:
            raise ValueError("authorized sections require source_id and section")
        if source_id not in sources:
            raise ValueError("authorized section source is not authorized")
        if (source_id, section) not in sections:
            sections.append((source_id, section))
    nested_available = False
    probe_id = "none"
    if nested_shell_capability is not None:
        if isinstance(nested_shell_capability, Mapping):
            nested_available = nested_shell_capability.get("nested_sandboxed_shell") is True
            probe_id = str(
                nested_shell_capability.get("probe_identity")
                or nested_shell_capability.get("nested_shell_probe_id")
                or "none"
            )
        else:
            nested_available = bool(
                getattr(nested_shell_capability, "nested_sandboxed_shell", False)
            )
            probe_id = str(getattr(nested_shell_capability, "probe_identity", "none"))
    if direct_shell_requested and normalized_access == "none":
        raise ValueError("arc_paper_direct_shell requires arc_paper_access=full")
    if not direct_shell_requested:
        nested_available = False
        probe_id = (
            "probe-not-requested" if normalized_access == "full" else "none"
        )
    if direct_shell_requested and not nested_available:
        raise PaperBrokerError(
            "paper_direct_shell_unavailable",
            "Trusted nested shell capability is unavailable for direct ARC-paper access.",
        )
    network_authorized = (
        normalized_access == "full"
        if paper_network_authorized is None
        else bool(paper_network_authorized) and normalized_access == "full"
    )
    material = {
        "schema_version": BROKER_POLICY_SCHEMA_VERSION,
        "access": normalized_access,
        "allowed_operation_ids": sorted(spec.operation_id for spec in specs),
        "authorized_source_ids": list(sources),
        "authorized_sections": [
            {"source_id": source_id, "section": section}
            for source_id, section in sorted(sections)
        ],
        "paper_network_authorized": network_authorized,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_sha256": catalog["catalog_sha256"],
        "direct_shell_requested": bool(direct_shell_requested),
        "direct_shell_available": nested_available,
        "direct_shell_probe_id": probe_id,
    }
    return PaperBrokerPolicy(
        schema_version=BROKER_POLICY_SCHEMA_VERSION,
        access=normalized_access,
        allowed_operation_ids=tuple(material["allowed_operation_ids"]),
        authorized_source_ids=sources,
        authorized_sections=tuple(sorted(sections)),
        paper_network_authorized=network_authorized,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        catalog_sha256=str(catalog["catalog_sha256"]),
        direct_shell_requested=bool(direct_shell_requested),
        direct_shell_available=nested_available,
        direct_shell_probe_id=probe_id,
        policy_sha256=_canonical_hash(material),
    )


def paper_broker_prompt_prefix(policy: PaperBrokerPolicy) -> str:
    """Return the access-aware compact Controller contract for a worker."""

    if policy.access == "none":
        return ""
    catalog = compact_catalog(policy.allowed_operation_ids)
    route = (
        "{{ARC_NESTED_SHELL_CAPABILITY}} The direct wrapper is arc-paper-worker; use it "
        "only for policy-allowed network=none catalog operations and use Controller "
        "requests otherwise."
        if policy.direct_shell_requested
        else "Use arc_evidence_requests for Controller reads; no direct shell command is exposed."
    )
    return (
        "ARC PAPER BROKER (bootstrap only)\n"
        + _canonical_json({
            "schema_version": BROKER_SCHEMA_VERSION,
            "policy_sha256": policy.policy_sha256,
            "catalog": catalog,
            "paper_network_authorized": policy.paper_network_authorized,
        })
        + "\n"
        + route
        + "\n"
    )


def paper_broker_schema(
    schema: Mapping[str, Any] | None, *, access: str
) -> dict[str, Any] | None:
    """Add the reserved evidence field only when ARC-paper access is full."""

    if schema is None:
        return None
    from arc_llm import allow_evidence_requests

    value = json.loads(_canonical_json(dict(schema)))
    if access == "full":
        return allow_evidence_requests(value)
    if value.get("type") == "object" and isinstance(value.get("properties"), dict):
        value["properties"].pop("arc_evidence_requests", None)
        required = value.get("required")
        if isinstance(required, list):
            value["required"] = [
                item for item in required if item != "arc_evidence_requests"
            ]
    return value


class PaperBroker:
    """Durable Controller route for one build and one immutable Broker policy."""

    def __init__(
        self,
        *,
        checkpoint_root: Path | str,
        base_cache_root: Path | str,
        policy: PaperBrokerPolicy,
        run_id: str,
        generic_internet_allowed: bool,
        journal_context: EvidenceJournalContext | None = None,
        target_lister: Callable[..., Mapping[str, Any]] | None = None,
        max_parallel_fetches: int = 4,
        transition_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
        managed_job_context: BrokerJobExecutionContext | None = None,
        broker_job_manager: BrokerJobManager | None = None,
        managed_job_wait_context: Callable[[], ContextManager[None]] | None = None,
        managed_job_cancel_check: Callable[[], bool] | None = None,
        controller_project_root: Path | str | None = None,
    ) -> None:
        self.policy = policy
        self.run_id = str(run_id).strip()
        if not self.run_id:
            raise ValueError("paper Broker run_id is required")
        self.generic_internet_allowed = bool(generic_internet_allowed)
        self.root = Path(checkpoint_root).expanduser().resolve() / "paper-broker"
        self.objects_root = self.root / "objects"
        self.handles_root = self.root / "handles"
        self.receipts_root = self.root / "receipts"
        self.inputs_root = self.root / "controller-inputs"
        self.controller_project_root = (
            None
            if controller_project_root is None
            else Path(controller_project_root).expanduser().resolve()
        )
        self.controller_objects_root = (
            None
            if self.controller_project_root is None
            else self.controller_project_root
            / ".arc-companion" / "paper-broker" / "controller-objects"
        )
        for directory in (
            self.root, self.objects_root, self.handles_root,
            self.receipts_root, self.inputs_root,
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            _chmod(directory, 0o700)
        if self.controller_objects_root is not None:
            if not _is_relative_to(
                self.controller_objects_root, self.controller_project_root
            ):
                raise ValueError(
                    "Controller aggregate root escapes the resolved project root"
                )
            self.controller_objects_root.mkdir(
                parents=True, exist_ok=True, mode=0o700
            )
            _chmod(self.controller_objects_root, 0o700)
        session_key = _canonical_hash({
            "run_id": self.run_id,
            "policy_sha256": policy.policy_sha256,
        })[:20]
        self.session = WorkerCacheSession(
            base_root=base_cache_root,
            run_root=self.root,
            session_id=f"broker-{session_key}",
            max_parallel_fetches=max_parallel_fetches,
        )
        self.journal_context = (
            None
            if journal_context is None
            else replace(
                journal_context,
                policy_hash=evidence_identity_hash({
                    "embedding_policy_hash": journal_context.policy_hash,
                    "broker_policy_sha256": policy.policy_sha256,
                    "catalog_sha256": policy.catalog_sha256,
                }),
                runtime_hash=evidence_identity_hash({
                    "embedding_runtime_hash": journal_context.runtime_hash,
                    "arc_paper_access": policy.access,
                    "broker_policy_sha256": policy.policy_sha256,
                    "catalog_sha256": policy.catalog_sha256,
                    "paper_network_authorized": policy.paper_network_authorized,
                    "generic_internet_allowed": self.generic_internet_allowed,
                    "direct_shell_requested": policy.direct_shell_requested,
                    "direct_shell_available": policy.direct_shell_available,
                    "direct_shell_probe_id": policy.direct_shell_probe_id,
                }),
            )
        )
        self.target_lister = target_lister
        self._transition_hook = transition_hook
        self.managed_job_context = managed_job_context
        self.broker_job_manager = broker_job_manager
        self.managed_job_wait_context = managed_job_wait_context
        self.managed_job_cancel_check = managed_job_cancel_check
        if managed_job_context is not None and managed_job_wait_context is None:
            raise ValueError(
                "managed paper jobs require an outer wait context that releases "
                "provider/session capacity"
            )

    @property
    def catalog(self) -> dict[str, Any]:
        return compact_catalog(self.policy.allowed_operation_ids)

    def canonicalize_request(self, request: EvidenceRequest) -> EvidenceRequest:
        operation = request.operation
        if operation in {LIST_REFERENCE_TARGETS_OPERATION, ARTIFACT_READ_OPERATION}:
            canonical = operation
        else:
            spec = get_operation_spec(operation)
            if spec is None:
                raise PaperBrokerError(
                    "paper_operation_unknown", "Unknown ARC-paper operation."
                )
            canonical = spec.name
        arguments = _json_object(request.arguments, "paper Broker arguments")
        if canonical == "get-parsed-section":
            locator = arguments.pop("locator", None)
            if locator is not None and "section" in arguments and locator != arguments["section"]:
                raise PaperBrokerError(
                    "paper_operation_parameters_invalid",
                    "locator conflicts with canonical section.",
                )
            if locator is not None:
                arguments["section"] = locator
        for key in ("source_id", "paper_id"):
            if key in arguments and isinstance(arguments[key], str):
                arguments[key] = self._canonical_source(arguments[key])
        if "paper_ids" in arguments:
            raw_ids = arguments["paper_ids"]
            if isinstance(raw_ids, str):
                arguments["paper_ids"] = self._canonical_source(raw_ids)
            elif isinstance(raw_ids, list):
                arguments["paper_ids"] = list(dict.fromkeys(
                    self._canonical_source(str(item)) for item in raw_ids
                ))
        return EvidenceRequest(
            request.request_id,
            canonical,
            arguments,
            request.reason,
            request.worker_id,
            request.role,
        )

    def controller(
        self,
        requests: tuple[EvidenceRequest, ...],
        *,
        round_number: int,
    ) -> tuple[EvidenceResponse, ...]:
        material, responses = self._canonicalize_requests(
            requests, round_number=round_number,
        )
        try:
            self._preflight_page_budget(material)
        except Exception as exc:
            for request in material:
                responses[request.request_id] = self._failure_response(
                    request, exc, round_number=round_number,
                )
            material = []
        page_requests = [
            request for request in material
            if request.operation == ARTIFACT_READ_OPERATION
        ]
        ordinary_requests = [
            request for request in material
            if request.operation != ARTIFACT_READ_OPERATION
        ]
        for request in ordinary_requests:
            responses[request.request_id] = self._resolve_one(
                request, round_number=round_number
            )
        if page_requests:
            provisional = tuple(
                responses[request.request_id]
                for request in requests
                if request.request_id in responses
            )
            bounded = self._fit_round_response_budget(
                ordinary_requests,
                provisional,
                round_number=round_number,
                max_bytes=(
                    MAX_ROUND_RESPONSE_BYTES
                    - self._page_budget_estimate(page_requests)
                ),
            )
            responses.update((response.request_id, response) for response in bounded)
        for request in page_requests:
            responses[request.request_id] = self._resolve_one(
                request, round_number=round_number
            )
        ordered = tuple(responses[request.request_id] for request in requests)
        return self._fit_round_response_budget(
            material, ordered, round_number=round_number,
        )

    def resolve_round(
        self,
        requests: Iterable[EvidenceRequest],
        *,
        round_number: int,
    ) -> tuple[EvidenceResponse, ...]:
        original = tuple(requests)
        material, responses = self._canonicalize_requests(
            original, round_number=round_number,
        )
        if self.journal_context is None:
            resolved = self.controller(tuple(material), round_number=round_number)
            responses.update(
                (response.request_id, response) for response in resolved
            )
            return tuple(responses[request.request_id] for request in original)
        journal = EvidenceJournal(self.journal_context.journal_root)
        policies = {
            request.operation: self._journal_policy(request, round_number=round_number)
            for request in material
        }
        resolved = journal.resolve_round(
            self.journal_context,
            material,
            self.controller,
            round_number=round_number,
            operation_policies=policies,
        )
        responses.update((response.request_id, response) for response in resolved)
        ordered = tuple(responses[request.request_id] for request in original)
        return self._fit_round_response_budget(
            material, ordered, round_number=round_number,
        )

    def _canonicalize_requests(
        self,
        requests: Sequence[EvidenceRequest],
        *,
        round_number: int,
    ) -> tuple[list[EvidenceRequest], dict[str, EvidenceResponse]]:
        material: list[EvidenceRequest] = []
        responses: dict[str, EvidenceResponse] = {}
        for request in requests:
            try:
                material.append(self.canonicalize_request(request))
            except Exception as exc:
                responses[request.request_id] = self._failure_response(
                    request, exc, round_number=round_number,
                )
        return material, responses

    def mark_delivered(
        self,
        requests: Iterable[EvidenceRequest],
        *,
        round_number: int,
        target_generation: int,
        target_session: str,
        followup_id: str,
    ) -> None:
        if self.journal_context is None:
            return
        material: list[EvidenceRequest] = []
        for request in requests:
            try:
                material.append(self.canonicalize_request(request))
            except PaperBrokerError:
                continue
        if not material:
            return
        EvidenceJournal(self.journal_context.journal_root).mark_delivered(
            self.journal_context,
            material,
            round_number=round_number,
            target_generation=target_generation,
            target_session=target_session,
            followup_id=followup_id,
            operation_policies={
                request.operation: self._journal_policy(
                    request, round_number=round_number
                )
                for request in material
            },
        )

    def _failure_response(
        self,
        request: EvidenceRequest,
        exc: Exception,
        *,
        round_number: int,
    ) -> EvidenceResponse:
        normalized = _normalize_error(exc)
        spec = get_operation_spec(request.operation)
        provenance = self._base_provenance(round_number)
        if request.operation in {
            LIST_REFERENCE_TARGETS_OPERATION, ARTIFACT_READ_OPERATION,
        }:
            provenance.update(
                self._control_provenance(request.operation, round_number)
            )
        elif spec is not None:
            provenance.update({
                "operation_id": spec.operation_id,
                "operation_version": spec.version,
                "arguments_sha256": _canonical_hash(dict(request.arguments)),
                "recovery_class": spec.recovery_class,
                "result_sha256": None,
                "result_handle": None,
                "result_inline": False,
                "network_declared": spec.network_access,
                "paper_network_authorized": self.policy.paper_network_authorized,
                "generic_internet_allowed": self.generic_internet_allowed,
                "network_observed": "unknown",
                "cache_declared": spec.cache_access,
                "cache_observed": "unknown",
                "promotion": _bounded_promotion(PromotionResult()),
                "warnings": [],
                "source_hashes": [],
            })
        else:
            provenance["operation_id"] = None
        provenance["error"] = {
            key: normalized[key]
            for key in ("code", "category", "retryable")
        }
        extra_provenance = getattr(exc, "provenance", None)
        if isinstance(extra_provenance, Mapping):
            provenance.update(dict(extra_provenance))
        return EvidenceResponse(
            request.request_id,
            False,
            error=normalized["message"],
            provenance=provenance,
        )

    def register_input_bytes(
        self,
        content: bytes,
        *,
        operation: str,
        parameter: str,
        media_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        spec = get_operation_spec(operation)
        if spec is None or (parameter, "read") not in spec.artifact_parameters:
            raise PaperBrokerError(
                "paper_artifact_registration_invalid",
                "Artifact input is not declared by the operation catalog.",
            )
        return self._store_bytes(
            bytes(content),
            media_type=media_type,
            contexts=((spec.name, parameter, "read"),),
        )[1]

    def register_output(
        self,
        *,
        operation: str,
        parameter: str,
        media_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Allocate a Controller-owned writable artifact handle."""

        spec = get_operation_spec(operation)
        if spec is None or (parameter, "write") not in spec.artifact_parameters:
            raise PaperBrokerError(
                "paper_artifact_registration_invalid",
                "Artifact output is not declared by the operation catalog.",
            )
        token = uuid.uuid4().hex
        output_name = f"output-{token}.bin"
        path = self.inputs_root / output_name
        _atomic_bytes(path, b"", mode=0o600)
        digest = hashlib.sha256(b"").hexdigest()
        handle_id = f"w-{token}-{self.policy.policy_sha256[:12]}"
        record = {
            "schema_version": BROKER_HANDLE_SCHEMA_VERSION,
            "handle_id": handle_id,
            "sha256": digest,
            "size_bytes": 0,
            "media_type": media_type,
            "output_name": output_name,
            "run_id": self.run_id,
            "access": self.policy.access,
            "policy_sha256": self.policy.policy_sha256,
            "contexts": [[spec.name, parameter, "write"]],
        }
        _atomic_json(
            self.handles_root / f"{handle_id}.json", record, max_bytes=128 * 1024
        )
        return {
            "handle_id": handle_id,
            "sha256": digest,
            "size_bytes": 0,
            "media_type": media_type,
        }

    def read_page(
        self, handle_id: str, *, offset: int = 0, limit: int = MAX_PAGE_BYTES
    ) -> dict[str, Any]:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise PaperBrokerError("artifact_page_invalid", "Artifact offset is invalid.")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > MAX_PAGE_BYTES
        ):
            raise PaperBrokerError(
                "artifact_page_invalid",
                f"Artifact limit must be between 1 and {MAX_PAGE_BYTES} bytes.",
            )
        record = self._read_handle(handle_id)
        object_name = record.get("object_name")
        if not isinstance(object_name, str) or not object_name:
            raise PaperBrokerError(
                "artifact_handle_forbidden", "Writable artifact handle is not readable."
            )
        path = self.objects_root / object_name
        payload = _verified_bytes(path, str(record["sha256"]))
        start = min(offset, len(payload))
        chunk = payload[start:start + limit]
        next_offset = start + len(chunk)
        return {
            "schema_version": BROKER_PAGE_SCHEMA_VERSION,
            "handle_id": handle_id,
            "encoding": "base64",
            "content_base64": base64.b64encode(chunk).decode("ascii"),
            "offset": start,
            "next_offset": next_offset,
            "total_size": len(payload),
            "eof": next_offset >= len(payload),
            "sha256": record["sha256"],
        }

    def store_controller_aggregate_json(
        self,
        *,
        namespace: str,
        lookup_identity: Mapping[str, Any],
        payload: Mapping[str, Any],
        max_bytes: int,
    ) -> dict[str, Any]:
        """Store canonical JSON under a project-stable Controller identity."""

        namespace_root = self._controller_aggregate_namespace(namespace)
        limit = _positive_byte_limit(max_bytes)
        identity = _json_object(
            lookup_identity, "Controller aggregate lookup identity"
        )
        value = _json_object(payload, "Controller aggregate payload")
        payload_bytes = _canonical_json(value).encode("utf-8")
        if len(payload_bytes) > limit:
            raise PaperBrokerError(
                "paper_controller_aggregate_too_large",
                "Controller aggregate payload exceeds its byte limit.",
            )
        payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        identity_sha256 = _canonical_hash(identity)
        objects_root = namespace_root / "objects"
        lookups_root = namespace_root / "lookups"
        for directory in (objects_root, lookups_root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            _chmod(directory, 0o700)
        object_path = objects_root / f"sha256-{payload_sha256}.json"
        lookup_path = lookups_root / f"sha256-{identity_sha256}.json"
        if object_path.exists():
            if _verified_bytes(object_path, payload_sha256) != payload_bytes:
                raise PaperBrokerError(
                    "paper_controller_aggregate_object_invalid",
                    "Controller aggregate object has conflicting canonical bytes.",
                )
        else:
            _atomic_bytes(object_path, payload_bytes, mode=0o600)
        assert self.controller_project_root is not None
        object_relative = object_path.relative_to(
            self.controller_project_root
        ).as_posix()
        lookup_relative = lookup_path.relative_to(
            self.controller_project_root
        ).as_posix()
        lookup = {
            "schema_version": CONTROLLER_AGGREGATE_LOOKUP_SCHEMA_VERSION,
            "namespace": namespace,
            "lookup_identity": identity,
            "lookup_identity_sha256": identity_sha256,
            "object_path": object_relative,
            "payload_sha256": payload_sha256,
            "size_bytes": len(payload_bytes),
            "media_type": "application/json",
        }
        if lookup_path.exists():
            existing = self._read_controller_aggregate_lookup(
                namespace=namespace,
                lookup_identity=identity,
                lookup_identity_sha256=identity_sha256,
                lookup_path=lookup_path,
                max_bytes=limit,
            )
            if existing != lookup:
                raise PaperBrokerError(
                    "paper_controller_aggregate_lookup_invalid",
                    "Controller aggregate lookup conflicts with immutable content.",
                )
        else:
            lookup_bytes = _canonical_json(lookup).encode("utf-8")
            if len(lookup_bytes) > 128 * 1024:
                raise PaperBrokerError(
                    "paper_controller_aggregate_lookup_invalid",
                    "Controller aggregate lookup receipt exceeds its byte limit.",
                )
            _atomic_bytes(lookup_path, lookup_bytes, mode=0o600)
        return {
            "object": self._controller_object_descriptor(
                namespace=namespace,
                object_path=object_relative,
                payload_sha256=payload_sha256,
                size_bytes=len(payload_bytes),
            ),
            "lookup_receipt": lookup,
            "lookup_receipt_path": lookup_relative,
            "lookup_receipt_sha256": _canonical_hash(lookup),
        }

    def load_controller_aggregate_json(
        self,
        *,
        namespace: str,
        lookup_identity: Mapping[str, Any],
        expected_payload_sha256: str,
        max_bytes: int,
    ) -> dict[str, Any] | None:
        """Load and verify one project-stable Controller aggregate."""

        namespace_root = self._controller_aggregate_namespace(namespace)
        limit = _positive_byte_limit(max_bytes)
        identity = _json_object(
            lookup_identity, "Controller aggregate lookup identity"
        )
        identity_sha256 = _canonical_hash(identity)
        lookup_path = (
            namespace_root / "lookups" / f"sha256-{identity_sha256}.json"
        )
        if not lookup_path.exists():
            return None
        lookup = self._read_controller_aggregate_lookup(
            namespace=namespace,
            lookup_identity=identity,
            lookup_identity_sha256=identity_sha256,
            lookup_path=lookup_path,
            max_bytes=limit,
        )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_payload_sha256)
            or lookup["payload_sha256"] != expected_payload_sha256
        ):
            raise PaperBrokerError(
                "paper_controller_aggregate_object_invalid",
                "Controller aggregate payload identity changed.",
            )
        assert self.controller_project_root is not None
        object_path = (
            self.controller_project_root / str(lookup["object_path"])
        ).resolve()
        payload_bytes = _verified_bytes(
            object_path, str(lookup["payload_sha256"])
        )
        if len(payload_bytes) != lookup["size_bytes"] or len(payload_bytes) > limit:
            raise PaperBrokerError(
                "paper_controller_aggregate_object_invalid",
                "Controller aggregate object size is invalid.",
            )
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PaperBrokerError(
                "paper_controller_aggregate_object_invalid",
                "Controller aggregate object is not valid JSON.",
            ) from exc
        if (
            not isinstance(payload, dict)
            or _canonical_json(payload).encode("utf-8") != payload_bytes
        ):
            raise PaperBrokerError(
                "paper_controller_aggregate_object_invalid",
                "Controller aggregate object is not canonical JSON.",
            )
        return {
            "object": self._controller_object_descriptor(
                namespace=namespace,
                object_path=str(lookup["object_path"]),
                payload_sha256=str(lookup["payload_sha256"]),
                size_bytes=int(lookup["size_bytes"]),
            ),
            "lookup_receipt": lookup,
            "lookup_receipt_path": lookup_path.relative_to(
                self.controller_project_root
            ).as_posix(),
            "lookup_receipt_sha256": _canonical_hash(lookup),
            "payload": payload,
        }

    def _controller_aggregate_namespace(self, namespace: str) -> Path:
        if self.controller_project_root is None or self.controller_objects_root is None:
            raise PaperBrokerError(
                "paper_controller_aggregate_root_required",
                "Controller aggregate storage requires a resolved project root.",
            )
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", namespace):
            raise PaperBrokerError(
                "paper_controller_aggregate_namespace_invalid",
                "Controller aggregate namespace is invalid.",
            )
        root = self.controller_objects_root.resolve()
        namespace_root = (root / namespace).resolve()
        if (
            not _is_relative_to(root, self.controller_project_root)
            or not _is_relative_to(namespace_root, root)
        ):
            raise PaperBrokerError(
                "paper_controller_aggregate_root_invalid",
                "Controller aggregate storage escapes the project.",
            )
        return namespace_root

    def _read_controller_aggregate_lookup(
        self,
        *,
        namespace: str,
        lookup_identity: Mapping[str, Any],
        lookup_identity_sha256: str,
        lookup_path: Path,
        max_bytes: int,
    ) -> dict[str, Any]:
        try:
            if lookup_path.stat().st_size > 128 * 1024:
                raise ValueError("lookup receipt exceeds byte limit")
            lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PaperBrokerError(
                "paper_controller_aggregate_lookup_invalid",
                "Controller aggregate lookup receipt is unreadable.",
            ) from exc
        keys = {
            "schema_version", "namespace", "lookup_identity",
            "lookup_identity_sha256", "object_path", "payload_sha256",
            "size_bytes", "media_type",
        }
        payload_sha256 = str(
            lookup.get("payload_sha256") if isinstance(lookup, dict) else ""
        )
        assert self.controller_project_root is not None
        expected_object = (
            self.controller_objects_root / namespace / "objects"
            / f"sha256-{payload_sha256}.json"
        ).resolve()
        try:
            object_relative = expected_object.relative_to(
                self.controller_project_root
            ).as_posix()
        except ValueError as exc:
            raise PaperBrokerError(
                "paper_controller_aggregate_lookup_invalid",
                "Controller aggregate object path escapes the project.",
            ) from exc
        if (
            not isinstance(lookup, dict)
            or set(lookup) != keys
            or lookup.get("schema_version")
            != CONTROLLER_AGGREGATE_LOOKUP_SCHEMA_VERSION
            or lookup.get("namespace") != namespace
            or lookup.get("lookup_identity") != dict(lookup_identity)
            or lookup.get("lookup_identity_sha256") != lookup_identity_sha256
            or not re.fullmatch(r"[0-9a-f]{64}", payload_sha256)
            or lookup.get("object_path") != object_relative
            or lookup.get("media_type") != "application/json"
            or isinstance(lookup.get("size_bytes"), bool)
            or not isinstance(lookup.get("size_bytes"), int)
            or lookup.get("size_bytes") < 2
            or lookup.get("size_bytes") > max_bytes
        ):
            raise PaperBrokerError(
                "paper_controller_aggregate_lookup_invalid",
                "Controller aggregate lookup receipt is invalid.",
            )
        payload = _verified_bytes(expected_object, payload_sha256)
        if len(payload) != lookup["size_bytes"]:
            raise PaperBrokerError(
                "paper_controller_aggregate_lookup_invalid",
                "Controller aggregate lookup size does not match its object.",
            )
        return lookup

    @staticmethod
    def _controller_object_descriptor(
        *,
        namespace: str,
        object_path: str,
        payload_sha256: str,
        size_bytes: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": CONTROLLER_AGGREGATE_OBJECT_SCHEMA_VERSION,
            "namespace": namespace,
            "object_path": object_path,
            "payload_sha256": payload_sha256,
            "size_bytes": size_bytes,
            "media_type": "application/json",
        }

    def _resolve_one(
        self, request: EvidenceRequest, *, round_number: int
    ) -> EvidenceResponse:
        try:
            if self.policy.access != "full":
                raise PaperBrokerError(
                    "paper_access_disabled", "ARC-paper access is disabled."
                )
            if request.operation == ARTIFACT_READ_OPERATION:
                data = self.read_page(
                    str(request.arguments.get("handle_id") or ""),
                    offset=request.arguments.get("offset", 0),
                    limit=request.arguments.get("limit", MAX_PAGE_BYTES),
                )
                return EvidenceResponse(
                    request.request_id,
                    True,
                    data=data,
                    provenance=self._control_provenance(request.operation, round_number),
                )
            if request.operation == LIST_REFERENCE_TARGETS_OPERATION:
                if self.target_lister is None:
                    raise PaperBrokerError(
                        "paper_target_catalog_unavailable",
                        "Reference target catalog is unavailable.",
                    )
                data = dict(self.target_lister(**dict(request.arguments)))
                return EvidenceResponse(
                    request.request_id,
                    True,
                    data=data,
                    provenance=self._control_provenance(request.operation, round_number),
                )
            return self._dispatch(request, round_number=round_number)
        except Exception as exc:
            return self._failure_response(
                request, exc, round_number=round_number,
            )

    def _dispatch(
        self, request: EvidenceRequest, *, round_number: int
    ) -> EvidenceResponse:
        spec = get_operation_spec(request.operation)
        assert spec is not None
        managed_job = (
            spec.uses_llm or spec.is_job or spec.recovery_class == "managed_job"
        )
        if managed_job and self.managed_job_context is None:
            raise PaperBrokerError(
                "managed_job_required",
                "This ARC-paper operation requires the managed job route.",
            )
        if spec.operation_id not in self.policy.allowed_operation_ids:
            raise PaperBrokerError(
                "paper_operation_forbidden", "ARC-paper operation is not authorized."
            )
        if spec.network_access == "may" and not self.policy.paper_network_authorized:
            raise PaperBrokerError(
                "paper_network_forbidden", "ARC-paper network access is not authorized."
            )
        self._authorize_arguments(spec, request.arguments)
        address_hash = self._broker_address_hash(request, round_number)
        receipt_path = self.receipts_root / f"{address_hash}.json"
        receipt = self._load_broker_receipt(receipt_path, request)
        if receipt is not None and receipt.get("state") == "result_persisted":
            return self._response_from_receipt(receipt, request.request_id)
        if receipt is None:
            if managed_job:
                receipt, _ticket = self._ensure_managed_job_ticket(
                    request, spec, round_number, address_hash,
                )
            else:
                receipt = self._prepared_receipt(
                    request, spec, round_number, address_hash,
                )
                _atomic_json(receipt_path, receipt, max_bytes=_MAX_RECEIPT_BYTES)
            self._notify_transition("prepared", receipt)
        elif receipt.get("state") not in {
            "prepared", "object_persisted", "promotion_persisted",
        }:
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker receipt state is invalid."
            )
        ticket = (
            self._ticket_from_receipt(receipt, spec=spec, address_hash=address_hash)
            if managed_job else None
        )
        call_id = (
            f"call-{ticket.identity_sha256[:32]}"
            if ticket is not None else f"call-{address_hash[:32]}"
        )
        managed_terminal_receipt = (
            dict(receipt["managed_job_terminal"])
            if isinstance(receipt.get("managed_job_terminal"), Mapping)
            else None
        )
        state = str(receipt.get("state"))
        if state == "prepared":
            if managed_job:
                envelope, managed_terminal_receipt = self._managed_job_envelope(
                    request,
                    spec=spec,
                    ticket=ticket,
                    round_number=round_number,
                )
            else:
                with self.session.in_process(call_id):
                    envelope = dispatch_operation(
                        spec.name,
                        request.arguments,
                        artifact_resolver=self._artifact_resolver,
                    )
            self.session.record_call(
                worker_id=request.worker_id or "companion-worker",
                call_id=call_id,
                operation=spec.name,
                status="success" if envelope.get("ok") is True else "failed",
                paper_ids=self._source_ids(request.arguments),
                parameters=request.arguments,
                source={"route": "controller", "operation_id": spec.operation_id},
            )
            cleaned, warnings = self._clean_result(envelope, spec=spec)
            object_record, handle = self._store_json(cleaned)
            receipt = {
                **receipt,
                "state": "object_persisted",
                "call_id": call_id,
                "result_object": object_record,
                "handle": handle,
                "warnings": list(warnings),
                **(
                    {
                        "managed_job_terminal": dict(
                            managed_terminal_receipt
                        )
                    }
                    if managed_terminal_receipt is not None else {}
                ),
                "object_persisted_at": _now(),
            }
            _atomic_json(receipt_path, receipt, max_bytes=_MAX_RECEIPT_BYTES)
            self._notify_transition("object_persisted", receipt)
        else:
            cleaned, object_record, handle, warnings = self._object_stage(receipt)

        if receipt.get("state") == "object_persisted":
            promotion = self.session.promote_call(call_id)
            receipt = {
                **receipt,
                "state": "promotion_persisted",
                "promotion": _bounded_promotion(promotion),
                "promotion_persisted_at": _now(),
            }
            _atomic_json(receipt_path, receipt, max_bytes=_MAX_RECEIPT_BYTES)
            self._notify_transition("promotion_persisted", receipt)
        else:
            promotion = _promotion_from_receipt(receipt)
        if promotion.quarantined:
            failure = self._failure_response(
                request,
                PaperBrokerError(
                    "paper_cache_validation_failed",
                    "ARC-paper cache artifacts failed local validation.",
                ),
                round_number=round_number,
            )
            failure_provenance = {
                **dict(failure.provenance),
                "operation_version": spec.version,
                "arguments_sha256": _canonical_hash(dict(request.arguments)),
                "recovery_class": spec.recovery_class,
                "result_sha256": object_record["sha256"],
                "result_handle": dict(handle),
                "network_declared": spec.network_access,
                "paper_network_authorized": self.policy.paper_network_authorized,
                "generic_internet_allowed": self.generic_internet_allowed,
                "network_observed": "unknown",
                "cache_declared": spec.cache_access,
                "cache_observed": _cache_status(promotion, cleaned),
                "promotion": _bounded_promotion(promotion),
                "warnings": list(warnings),
                "source_hashes": _result_hashes(cleaned),
                "result_inline": False,
            }
            failure = EvidenceResponse(
                request.request_id,
                False,
                error=failure.error,
                provenance=failure_provenance,
            )
            final = {
                **receipt,
                "state": "result_persisted",
                "response": _serialize_response(failure),
                "result_persisted_at": _now(),
            }
            _atomic_json(receipt_path, final, max_bytes=_MAX_RECEIPT_BYTES)
            self._notify_transition("result_persisted", final)
            return failure
        response = self._build_response(
            request,
            spec,
            cleaned,
            object_record=object_record,
            handle=handle,
            warnings=warnings,
            promotion=promotion,
            round_number=round_number,
            managed_job_receipt=(
                {
                    **dict(managed_terminal_receipt or {}),
                    "job_id": ticket.job_id,
                    "identity_sha256": ticket.identity_sha256,
                    "ticket_sha256": receipt.get("job_ticket_sha256"),
                    "budget_identity_sha256": (
                        ticket.budget_identity_sha256
                    ),
                    "transaction_receipt_sha256": (
                        ticket.transaction_receipt_sha256
                    ),
                    "evidence_round": round_number,
                }
                if ticket is not None else None
            ),
        )
        final = {
            **receipt,
            "state": "result_persisted",
            "result_object": object_record,
            "handle": handle,
            "promotion": _bounded_promotion(promotion),
            "response": _serialize_response(response),
            "result_persisted_at": _now(),
        }
        _atomic_json(receipt_path, final, max_bytes=_MAX_RECEIPT_BYTES)
        self._notify_transition("result_persisted", final)
        return response

    def _ensure_managed_job_ticket(
        self,
        request: EvidenceRequest,
        spec: OperationSpec,
        round_number: int,
        address_hash: str,
    ) -> tuple[dict[str, Any], BrokerJobTicket]:
        context = self.managed_job_context
        if context is None:
            raise PaperBrokerError(
                "managed_job_required",
                "This ARC-paper operation requires the managed job route.",
            )
        path = self.receipts_root / f"{address_hash}.json"
        receipt = self._load_broker_receipt(path, request)
        if receipt is not None and receipt.get("job_ticket") is not None:
            return receipt, self._ticket_from_receipt(
                receipt, spec=spec, address_hash=address_hash,
            )
        if receipt is not None and receipt.get("state") != "prepared":
            raise PaperBrokerError(
                "paper_broker_job_ticket_corrupt",
                "Managed job result stages require a persisted job ticket.",
            )
        budget = SharedBudget.open(context.budget)
        source_ids = self._source_ids(request.arguments)
        cached_sources = self._managed_cached_source_identities(source_ids)
        manager = self.broker_job_manager or BrokerJobManager()
        ticket = manager.submit(
            operation=spec.name,
            arguments=request.arguments,
            budget=budget,
            output_reserve_tokens=context.output_reserve_tokens,
            parent_run_id=self.run_id,
            policy_sha256=self.policy.policy_sha256,
            runtime_sha256=(
                self.journal_context.runtime_hash
                if self.journal_context is not None
                else _canonical_hash({
                    "broker_schema_version": BROKER_SCHEMA_VERSION,
                    "policy_sha256": self.policy.policy_sha256,
                })
            ),
            transaction_receipt_sha256=address_hash,
            network_authorized=self.policy.paper_network_authorized,
            source_sha256=(
                _canonical_hash(cached_sources) if source_ids else None
            ),
            content_sha256=_canonical_hash({
                "schema_version": "arc.companion.paper-managed-content.v3",
                "operation_id": spec.operation_id,
                "arguments": dict(request.arguments),
                "cached_source_identities": cached_sources,
                "authorized_source_ids": list(
                    self.policy.authorized_source_ids
                ),
                "authorized_sections": [
                    list(item) for item in self.policy.authorized_sections
                ],
            }),
            refresh=bool(request.arguments.get("refresh", False)),
            cache_environment=self.session.environment(),
            artifact_root=self.root,
            artifact_authorizations=self._managed_artifact_authorizations(
                spec, request.arguments,
            ),
        )
        prepared = receipt or self._prepared_receipt(
            request, spec, round_number, address_hash,
        )
        prepared = {
            **prepared,
            "job_ticket": ticket.to_json(),
            "job_ticket_sha256": _canonical_hash(ticket.to_json()),
        }
        _atomic_json(path, prepared, max_bytes=_MAX_RECEIPT_BYTES)
        return prepared, ticket

    def _managed_artifact_authorizations(
        self,
        spec: OperationSpec,
        arguments: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        authorizations: dict[str, dict[str, Any]] = {}
        for parameter, access in spec.artifact_parameters:
            handle = arguments.get(parameter)
            if handle is None:
                continue
            handle_id = (
                str(handle.get("handle_id") or "")
                if isinstance(handle, Mapping) else ""
            )
            record = self._read_handle(handle_id)
            if [spec.name, parameter, access] not in record.get("contexts", []):
                raise PaperBrokerError(
                    "paper_artifact_handle_forbidden",
                    "Artifact handle is not authorized for this managed job.",
                )
            authorizations[parameter] = {
                "handle_id": handle_id,
                "operation": spec.name,
                "parameter": parameter,
                "access": access,
                "handle_receipt_sha256": _canonical_hash(record),
            }
        return authorizations

    def _ticket_from_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        spec: OperationSpec,
        address_hash: str,
    ) -> BrokerJobTicket:
        raw = receipt.get("job_ticket")
        if not isinstance(raw, Mapping):
            raise PaperBrokerError(
                "paper_broker_job_ticket_corrupt",
                "Managed job receipt has no valid ticket.",
            )
        try:
            ticket = BrokerJobTicket.from_json(raw)
        except Exception as exc:
            raise PaperBrokerError(
                "paper_broker_job_ticket_corrupt",
                "Managed job ticket is invalid.",
            ) from exc
        if (
            receipt.get("job_ticket_sha256") != _canonical_hash(ticket.to_json())
            or ticket.operation_version != spec.version
            or ticket.transaction_receipt_sha256 != address_hash
            or self.managed_job_context is None
            or ticket.budget_identity_sha256
            != self.managed_job_context.budget.identity_sha256
        ):
            raise PaperBrokerError(
                "paper_broker_job_ticket_corrupt",
                "Managed job ticket guards do not match the Broker receipt.",
            )
        return ticket

    def _managed_job_envelope(
        self,
        request: EvidenceRequest,
        *,
        spec: OperationSpec,
        ticket: BrokerJobTicket | None,
        round_number: int = 1,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        context = self.managed_job_context
        if context is None:
            raise PaperBrokerError(
                "managed_job_required",
                "This ARC-paper operation requires the managed job route.",
            )
        manager = self.broker_job_manager or BrokerJobManager()
        if ticket is None:
            raise PaperBrokerError(
                "paper_broker_job_ticket_corrupt",
                "Managed job receipt has no attachable ticket.",
            )
        terminal = manager.terminal(ticket)
        if terminal is None:
            assert self.managed_job_wait_context is not None
            with self.managed_job_wait_context():
                while terminal is None:
                    if (
                        self.managed_job_cancel_check is not None
                        and self.managed_job_cancel_check()
                    ):
                        manager.cancel(ticket)
                        raise PaperBrokerError(
                            "paper_broker_job_cancelled",
                            "Managed ARC-paper job waiter was cancelled.",
                            category="managed_job",
                            retryable=False,
                        )
                    terminal = manager.wait(ticket, timeout=30.0)
        if terminal.result is not None:
            return dict(terminal.result), dict(terminal.receipt or {})
        error = terminal.error or {
            "code": "paper_broker_job_failed",
            "message": "Managed ARC-paper job failed without a result.",
        }
        terminal_provenance = {
            **dict(terminal.receipt or {}),
            "job_id": ticket.job_id,
            "identity_sha256": ticket.identity_sha256,
            "budget_identity_sha256": ticket.budget_identity_sha256,
            "transaction_receipt_sha256": ticket.transaction_receipt_sha256,
            "evidence_round": round_number,
        }
        raise PaperBrokerError(
            str(error.get("code") or "paper_broker_job_failed"),
            _managed_job_public_error(
                str(error.get("code") or "paper_broker_job_failed"),
            ),
            category="managed_job",
            retryable=False,
            provenance={"managed_job": terminal_provenance},
        )

    def _object_stage(
        self, receipt: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[str, ...]]:
        result = receipt.get("result_object")
        handle = receipt.get("handle")
        if not isinstance(result, Mapping) or not isinstance(handle, Mapping):
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker object receipt is incomplete."
            )
        descriptor = {
            key: handle.get(key)
            for key in ("handle_id", "sha256", "size_bytes", "media_type")
        }
        if (
            not isinstance(descriptor["handle_id"], str)
            or descriptor["sha256"] != result.get("sha256")
            or descriptor["size_bytes"] != result.get("size_bytes")
            or descriptor["media_type"] != result.get("media_type")
        ):
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker handle guard is invalid."
            )
        handle_record = self._read_handle(descriptor["handle_id"])
        if any(handle_record.get(key) != value for key, value in descriptor.items()):
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker handle receipt is stale."
            )
        payload = _verified_bytes(
            self.objects_root / f"sha256-{result.get('sha256')}.json",
            str(result.get("sha256") or ""),
        )
        if result.get("size_bytes") != len(payload):
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker object size guard is invalid."
            )
        try:
            cleaned = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker result object is invalid JSON."
            ) from exc
        if not isinstance(cleaned, dict):
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker result object is invalid."
            )
        warnings = receipt.get("warnings", [])
        if not isinstance(warnings, list) or not all(
            isinstance(value, str) for value in warnings
        ):
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker warning receipt is invalid."
            )
        return cleaned, dict(result), dict(handle), tuple(warnings)

    def _notify_transition(self, state: str, receipt: Mapping[str, Any]) -> None:
        if self._transition_hook is not None:
            self._transition_hook(state, receipt)

    def _build_response(
        self,
        request: EvidenceRequest,
        spec: OperationSpec,
        cleaned: Mapping[str, Any],
        *,
        object_record: Mapping[str, Any],
        handle: Mapping[str, Any],
        warnings: Sequence[str],
        promotion: PromotionResult,
        round_number: int,
        managed_job_receipt: Mapping[str, Any] | None = None,
    ) -> EvidenceResponse:
        ok = cleaned.get("ok") is True
        error = cleaned.get("error") if isinstance(cleaned.get("error"), Mapping) else {}
        provenance = {
            **self._base_provenance(round_number),
            "operation_id": spec.operation_id,
            "operation_version": spec.version,
            "arguments_sha256": _canonical_hash(dict(request.arguments)),
            "recovery_class": spec.recovery_class,
            "result_sha256": object_record["sha256"],
            "result_handle": dict(handle),
            "network_declared": spec.network_access,
            "paper_network_authorized": self.policy.paper_network_authorized,
            "generic_internet_allowed": self.generic_internet_allowed,
            "network_observed": "unknown",
            "cache_declared": spec.cache_access,
            "cache_observed": _cache_status(promotion, cleaned),
            "promotion": _bounded_promotion(promotion),
            "warnings": list(warnings),
            "source_hashes": _result_hashes(cleaned),
        }
        if managed_job_receipt is not None:
            provenance["managed_job"] = dict(managed_job_receipt)
        if not ok:
            provenance["error"] = _envelope_error_metadata(error)
        provisional = EvidenceResponse(
            request.request_id,
            ok,
            data=dict(cleaned) if ok else None,
            error=None if ok else _envelope_error_message(error),
            provenance={**provenance, "result_inline": True},
        )
        inline = len(
            _canonical_json(_serialize_response(provisional)).encode("utf-8")
        ) <= MAX_INLINE_BYTES
        provenance["result_inline"] = inline
        return EvidenceResponse(
            request.request_id,
            ok,
            data=dict(cleaned) if inline and ok else dict(handle) if ok else None,
            error=None if ok else _envelope_error_message(error),
            provenance=provenance,
        )

    def _clean_result(
        self, envelope: Mapping[str, Any], *, spec: OperationSpec
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        warnings: list[str] = []

        def clean(value: Any, *, key: str = "") -> Any:
            if isinstance(value, Mapping):
                result = {}
                for item_key, item in value.items():
                    name = str(item_key)
                    if name in _PURE_CACHE_PATH_FIELDS:
                        warnings.append("paper_result.cache_path_omitted")
                        continue
                    cleaned = clean(item, key=name)
                    if cleaned is not _OMIT:
                        result[name] = cleaned
                return result
            if isinstance(value, list):
                return [item for raw in value if (item := clean(raw, key=key)) is not _OMIT]
            if isinstance(value, str) and _is_http_url(value):
                return _sanitized_url(value)
            if isinstance(value, str) and Path(value).is_absolute():
                path = Path(value).expanduser().resolve(strict=False)
                if self._owned_result_path(path):
                    try:
                        payload = path.read_bytes()
                    except OSError:
                        warnings.append("paper_result.path_unreadable")
                        return _OMIT
                    _record, handle = self._store_bytes(
                        payload, media_type=_media_type(path), contexts=()
                    )
                    return {"artifact": handle}
                warnings.append("paper_result.path_omitted")
                return _OMIT
            return value

        cleaned = clean(dict(envelope))
        assert isinstance(cleaned, dict)
        return cleaned, tuple(dict.fromkeys(warnings))

    def _authorize_arguments(
        self, spec: OperationSpec, arguments: Mapping[str, Any]
    ) -> None:
        source_ids = self._source_ids(arguments)
        if self.policy.authorized_source_ids and any(
            source_id not in self.policy.authorized_source_ids for source_id in source_ids
        ):
            raise PaperBrokerError(
                "paper_source_forbidden", "ARC-paper source is not authorized."
            )
        if spec.name == "get-parsed-section" and self.policy.authorized_sections:
            target = (
                str(arguments.get("source_id") or ""),
                str(arguments.get("section") or ""),
            )
            if target not in self.policy.authorized_sections:
                raise PaperBrokerError(
                    "paper_section_forbidden", "ARC-paper section is not authorized."
                )

    def _source_ids(self, arguments: Mapping[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("source_id", "paper_id", "paper_ids"):
            value = arguments.get(key)
            raw = value if isinstance(value, list) else [value]
            for item in raw:
                if isinstance(item, str) and item and item not in values:
                    values.append(item)
        return values

    def _managed_cached_source_identities(
        self,
        source_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Bind jobs to exact cached bodies as well as metadata sidecars."""

        records: list[dict[str, Any]] = []
        roots = (
            ("overlay", self.session.overlay_root),
            ("base", self.session.base_root),
        )

        def visible_file(relative: Path) -> tuple[str, bytes] | None:
            for layer, root in roots:
                path = root / relative
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    return layer, path.read_bytes()
                except OSError as exc:
                    raise PaperBrokerError(
                        "paper_cached_source_identity_corrupt",
                        "Cached parsed-source content is unreadable.",
                    ) from exc
            return None

        for source_id in source_ids:
            safe_name = paper_ids_safe_dir_name([source_id])
            identity_file = visible_file(
                Path("source-identities") / f"{safe_name}.json"
            )
            parsed_file = visible_file(Path("sources") / f"{safe_name}.json")
            selected: dict[str, Any] | None = None
            if identity_file is not None:
                layer, payload = identity_file
                try:
                    value = json.loads(payload)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise PaperBrokerError(
                        "paper_cached_source_identity_corrupt",
                        "Cached parsed-source identity is invalid.",
                    ) from exc
                if not isinstance(value, Mapping):
                    raise PaperBrokerError(
                        "paper_cached_source_identity_corrupt",
                        "Cached parsed-source identity is invalid.",
                    )
                source_hash = str(value.get("source_hash") or "")
                parsed_value: Mapping[str, Any] | None = None
                if parsed_file is not None:
                    try:
                        candidate = json.loads(parsed_file[1])
                    except (UnicodeError, json.JSONDecodeError):
                        candidate = None
                    if isinstance(candidate, Mapping):
                        parsed_value = candidate
                rich_source_hash = str(
                    (parsed_value or {}).get("source_hash") or ""
                )
                rich_file = (
                    visible_file(
                        Path("rich-sources")
                        / safe_name
                        / f"v{RICH_DOCUMENT_PARSER_VERSION}"
                        / f"{rich_source_hash}.json"
                    )
                    if rich_source_hash
                    else None
                )
                selected = {
                    "source_id": source_id,
                    "cache_state": "available",
                    "cache_layer": layer,
                    "identity_file_sha256": hashlib.sha256(payload).hexdigest(),
                    "parsed_source": (
                        {
                            "cache_layer": parsed_file[0],
                            "sha256": hashlib.sha256(parsed_file[1]).hexdigest(),
                        }
                        if parsed_file is not None else None
                    ),
                    "rich_source": (
                        {
                            "cache_layer": rich_file[0],
                            "sha256": hashlib.sha256(rich_file[1]).hexdigest(),
                        }
                        if rich_file is not None else None
                    ),
                    "declared_source_hash": source_hash,
                    "declared_document_hash": str(
                        value.get("document_hash") or ""
                    ),
                }
            records.append(selected or {
                "source_id": source_id,
                "cache_state": "missing",
                "cache_layer": None,
                "identity_file_sha256": None,
                "parsed_source": (
                    {
                        "cache_layer": parsed_file[0],
                        "sha256": hashlib.sha256(parsed_file[1]).hexdigest(),
                    }
                    if parsed_file is not None else None
                ),
                "rich_source": None,
                "declared_source_hash": None,
                "declared_document_hash": None,
            })
        return records

    def _canonical_source(self, source_id: str) -> str:
        normalized = normalize_paper_id(source_id)
        seen = set()
        while normalized and normalized not in seen:
            seen.add(normalized)
            alias_path = (
                self.session.base_root / "paper-aliases"
                / f"{quote(normalized, safe='')}.json"
            )
            try:
                alias = json.loads(alias_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                break
            next_id = normalize_paper_id(str(alias.get("canonical_id") or ""))
            if not next_id:
                break
            normalized = next_id
        return normalized

    def _artifact_resolver(
        self,
        handle_id: str,
        *,
        access: str,
        operation: str,
        parameter: str,
    ) -> Path:
        record = self._read_handle(handle_id)
        context = [operation, parameter, access]
        if context not in record.get("contexts", []):
            raise PaperBrokerError(
                "paper_artifact_handle_forbidden",
                "Artifact handle is not authorized for this operation parameter.",
            )
        if access == "write":
            output_name = record.get("output_name")
            if not isinstance(output_name, str) or Path(output_name).name != output_name:
                raise PaperBrokerError(
                    "paper_artifact_handle_forbidden",
                    "Writable artifact handle has invalid ownership.",
                )
            path = self.inputs_root / output_name
            if path.is_symlink() or not path.is_file():
                raise PaperBrokerError(
                    "paper_artifact_handle_forbidden",
                    "Writable artifact handle is unavailable.",
                )
            return path
        path = self.objects_root / str(record.get("object_name") or "")
        _verified_bytes(path, str(record["sha256"]))
        return path

    def _store_json(
        self, value: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._store_bytes(
            _canonical_json(dict(value)).encode("utf-8"),
            media_type="application/json",
            contexts=(),
        )

    def _store_bytes(
        self,
        payload: bytes,
        *,
        media_type: str,
        contexts: Sequence[tuple[str, str, str]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        digest = hashlib.sha256(payload).hexdigest()
        suffix = ".json" if media_type == "application/json" else ".bin"
        object_name = f"sha256-{digest}{suffix}"
        path = self.objects_root / object_name
        if path.exists():
            _verified_bytes(path, digest)
        else:
            _atomic_bytes(path, payload, mode=0o600)
        handle_id = f"h-{digest[:40]}-{self.policy.policy_sha256[:12]}"
        record = {
            "schema_version": BROKER_HANDLE_SCHEMA_VERSION,
            "handle_id": handle_id,
            "sha256": digest,
            "size_bytes": len(payload),
            "media_type": media_type,
            "object_name": object_name,
            "run_id": self.run_id,
            "access": self.policy.access,
            "policy_sha256": self.policy.policy_sha256,
            "contexts": [list(context) for context in contexts],
        }
        _atomic_json(
            self.handles_root / f"{handle_id}.json", record, max_bytes=128 * 1024
        )
        descriptor = {
            "handle_id": handle_id,
            "sha256": digest,
            "size_bytes": len(payload),
            "media_type": media_type,
        }
        return {
            "schema_version": BROKER_OBJECT_SCHEMA_VERSION,
            "sha256": digest,
            "size_bytes": len(payload),
            "media_type": media_type,
        }, descriptor

    def _read_handle(self, handle_id: str) -> dict[str, Any]:
        if not handle_id or not all(character.isalnum() or character in "-." for character in handle_id):
            raise PaperBrokerError("artifact_handle_invalid", "Artifact handle is invalid.")
        path = self.handles_root / f"{handle_id}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PaperBrokerError(
                "artifact_handle_not_found", "Artifact handle was not found."
            ) from exc
        expected = {
            "schema_version": BROKER_HANDLE_SCHEMA_VERSION,
            "handle_id": handle_id,
            "run_id": self.run_id,
            "access": self.policy.access,
            "policy_sha256": self.policy.policy_sha256,
        }
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in expected.items()):
            raise PaperBrokerError(
                "artifact_handle_forbidden", "Artifact handle ownership does not match this run."
            )
        return record

    def _owned_result_path(self, path: Path) -> bool:
        return any(
            _is_relative_to(path, root)
            for root in (self.session.overlay_root, self.session.base_root, self.root)
        ) and path.is_file()

    def _prepared_receipt(
        self,
        request: EvidenceRequest,
        spec: OperationSpec,
        round_number: int,
        address_hash: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": BROKER_RECEIPT_SCHEMA_VERSION,
            "state": "prepared",
            "address_sha256": address_hash,
            "request_id": request.request_id,
            "operation_id": spec.operation_id,
            "arguments_sha256": _canonical_hash(dict(request.arguments)),
            "policy_sha256": self.policy.policy_sha256,
            "catalog_sha256": self.policy.catalog_sha256,
            "runtime_sha256": (
                self.journal_context.runtime_hash if self.journal_context else "none"
            ),
            "round": round_number,
            "prepared_at": _now(),
        }

    def _load_broker_receipt(
        self, path: Path, request: EvidenceRequest
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker receipt is corrupt."
            ) from exc
        raw_ticket = value.get("job_ticket") if isinstance(value, dict) else None
        ticket_identity = (
            str(raw_ticket.get("identity_sha256") or "")
            if isinstance(raw_ticket, Mapping) else ""
        )
        expected_call_id = (
            f"call-{ticket_identity[:32]}"
            if len(ticket_identity) == 64 else f"call-{path.stem[:32]}"
        )
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != BROKER_RECEIPT_SCHEMA_VERSION
            or value.get("address_sha256") != path.stem
            or value.get("request_id") != request.request_id
            or value.get("operation_id") != get_operation_spec(request.operation).operation_id
            or value.get("arguments_sha256") != _canonical_hash(dict(request.arguments))
            or value.get("policy_sha256") != self.policy.policy_sha256
            or value.get("catalog_sha256") != self.policy.catalog_sha256
            or value.get("runtime_sha256")
            != (self.journal_context.runtime_hash if self.journal_context else "none")
            or value.get("state") not in {
                "prepared", "object_persisted", "promotion_persisted", "result_persisted",
            }
            or (
                value.get("state") != "prepared"
                and value.get("call_id") != expected_call_id
            )
        ):
            raise PaperBrokerError(
                "paper_broker_receipt_stale", "Paper Broker receipt identity is stale."
            )
        return value

    def _response_from_receipt(
        self, receipt: Mapping[str, Any], request_id: str
    ) -> EvidenceResponse:
        cleaned, result, handle, _warnings = self._object_stage(receipt)
        response = receipt.get("response")
        if not isinstance(response, Mapping) or response.get("request_id") != request_id:
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker response receipt is incomplete."
            )
        provenance = response.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("result_sha256") != result.get("sha256")
            or provenance.get("policy_sha256") != self.policy.policy_sha256
            or provenance.get("catalog_sha256") != self.policy.catalog_sha256
        ):
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker response guards are invalid."
            )
        if response.get("ok") is True:
            if provenance.get("result_inline") is True:
                if response.get("data") != cleaned:
                    raise PaperBrokerError(
                        "paper_broker_receipt_corrupt",
                        "Paper Broker inline response does not match its object.",
                    )
            elif response.get("data") != handle:
                raise PaperBrokerError(
                    "paper_broker_receipt_corrupt",
                    "Paper Broker handle response does not match its receipt.",
                )
        return EvidenceResponse(
            request_id,
            bool(response.get("ok")),
            data=response.get("data"),
            error=response.get("error"),
            provenance=dict(provenance),
        )

    def _journal_policy(
        self, request: EvidenceRequest, *, round_number: int
    ) -> EvidenceOperationPolicy:
        spec = get_operation_spec(request.operation)
        idempotent = spec is None or spec.recovery_class == "idempotent"
        managed_job = bool(
            spec is not None
            and (
                spec.uses_llm
                or spec.is_job
                or spec.recovery_class == "managed_job"
            )
        )

        def transaction(_request: EvidenceRequest) -> Mapping[str, Any]:
            canonical = self.canonicalize_request(_request)
            address_hash = self._broker_address_hash(canonical, round_number)
            path = self.receipts_root / f"{address_hash}.json"
            ticket = None
            if managed_job and spec is not None:
                _receipt, ticket = self._ensure_managed_job_ticket(
                    canonical, spec, round_number, address_hash,
                )
            elif not path.exists() and spec is not None:
                _atomic_json(
                    path,
                    self._prepared_receipt(canonical, spec, round_number, address_hash),
                    max_bytes=_MAX_RECEIPT_BYTES,
                )
            transaction_receipt = {
                "schema_version": BROKER_RECEIPT_SCHEMA_VERSION,
                "address_sha256": address_hash,
                "policy_sha256": self.policy.policy_sha256,
            }
            if ticket is not None:
                transaction_receipt.update({
                    "job_ticket": ticket.to_json(),
                    "job_ticket_sha256": _canonical_hash(ticket.to_json()),
                })
            return transaction_receipt

        def recover(
            _request: EvidenceRequest, transaction_receipt: Mapping[str, Any]
        ) -> EvidenceExecution:
            canonical = self.canonicalize_request(_request)
            digest = str(transaction_receipt.get("address_sha256") or "")
            expected = self._broker_address_hash(canonical, round_number)
            if digest != expected:
                raise EvidenceJournalRecoveryError("paper Broker receipt address changed")
            path = self.receipts_root / f"{digest}.json"
            receipt = self._load_broker_receipt(path, canonical)
            if managed_job and spec is not None:
                if receipt is None:
                    # No job ticket was durably published. Deterministic
                    # creation remains safe in this pre-job recovery window.
                    receipt, ticket = self._ensure_managed_job_ticket(
                        canonical, spec, round_number, digest,
                    )
                else:
                    ticket = self._ticket_from_receipt(
                        receipt, spec=spec, address_hash=digest,
                    )
                transaction_ticket = transaction_receipt.get("job_ticket")
                if (
                    not isinstance(transaction_ticket, Mapping)
                    or BrokerJobTicket.from_json(transaction_ticket) != ticket
                    or transaction_receipt.get("job_ticket_sha256")
                    != _canonical_hash(ticket.to_json())
                ):
                    raise EvidenceJournalRecoveryError(
                        "managed paper job transaction ticket changed"
                    )
                return EvidenceExecution(
                    self._resolve_one(canonical, round_number=round_number),
                    transaction_receipt,
                )
            if receipt is not None and receipt.get("state") == "result_persisted":
                return EvidenceExecution(
                    self._response_from_receipt(receipt, canonical.request_id),
                    transaction_receipt,
                )
            if receipt is not None and receipt.get("state") in {
                "object_persisted", "promotion_persisted",
            }:
                return EvidenceExecution(
                    self._resolve_one(canonical, round_number=round_number),
                    transaction_receipt,
                )
            if not idempotent:
                raise EvidenceJournalRecoveryError(
                    "transactional paper operation has no persisted Broker result"
                )
            return EvidenceExecution(
                self._resolve_one(canonical, round_number=round_number),
                transaction_receipt,
            )

        return EvidenceOperationPolicy(
            idempotent=idempotent,
            recover=recover,
            transaction_receipt=transaction,
        )

    def _broker_address_hash(
        self, request: EvidenceRequest, round_number: int
    ) -> str:
        if self.journal_context is not None:
            address = self.journal_context.address(
                request.request_id, evidence_round=round_number
            )
            return evidence_identity_hash(asdict(address))
        return _canonical_hash({
            "run_id": self.run_id,
            "round": round_number,
            "request_id": request.request_id,
        })

    def _preflight_page_budget(self, requests: Sequence[EvidenceRequest]) -> None:
        total = self._page_budget_estimate(requests)
        if total > MAX_ROUND_RESPONSE_BYTES:
            raise PaperBrokerError(
                "artifact_round_budget_exceeded",
                "Artifact page requests exceed the whole-round response budget.",
            )

    def _page_budget_estimate(
        self, requests: Sequence[EvidenceRequest],
    ) -> int:
        total = 0
        for request in requests:
            if request.operation != ARTIFACT_READ_OPERATION:
                continue
            limit = request.arguments.get("limit", MAX_PAGE_BYTES)
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise PaperBrokerError(
                    "artifact_page_invalid", "Artifact page limit is invalid."
                )
            # Base64 expands raw bytes to four characters per three bytes;
            # reserve the remainder for page fields and provenance.
            total += 4 * ((limit + 2) // 3) + 2048
        return total

    def _fit_round_response_budget(
        self,
        requests: Sequence[EvidenceRequest],
        responses: Sequence[EvidenceResponse],
        *,
        round_number: int,
        max_bytes: int = MAX_ROUND_RESPONSE_BYTES,
    ) -> tuple[EvidenceResponse, ...]:
        """Bound the serialized response tuple, paging ordinary results as needed."""

        adjusted = list(responses)

        def size() -> int:
            return len(_canonical_json([
                _serialize_response(response) for response in adjusted
            ]).encode("utf-8"))

        if size() <= max_bytes:
            return tuple(adjusted)
        request_by_id = {request.request_id: request for request in requests}
        candidates = sorted(
            (
                (len(_canonical_json(_serialize_response(response))), index, response)
                for index, response in enumerate(adjusted)
                if response.ok
                and response.provenance.get("result_inline") is True
                and isinstance(response.provenance.get("result_handle"), Mapping)
                and request_by_id.get(response.request_id) is not None
                and request_by_id[response.request_id].operation
                not in {LIST_REFERENCE_TARGETS_OPERATION, ARTIFACT_READ_OPERATION}
            ),
            reverse=True,
        )
        for _response_size, index, response in candidates:
            provenance = {**dict(response.provenance), "result_inline": False}
            adjusted[index] = EvidenceResponse(
                response.request_id,
                True,
                data=dict(provenance["result_handle"]),
                provenance=provenance,
            )
            self._persist_adjusted_response(
                request_by_id[response.request_id],
                adjusted[index],
                round_number=round_number,
            )
            if size() <= max_bytes:
                return tuple(adjusted)
        raise PaperBrokerError(
            "artifact_round_budget_exceeded",
            "The complete Controller response exceeds the whole-round budget.",
        )

    def _persist_adjusted_response(
        self,
        request: EvidenceRequest,
        response: EvidenceResponse,
        *,
        round_number: int,
    ) -> None:
        """Keep Broker replay receipts identical to the bounded delivered response."""

        path = self.receipts_root / (
            f"{self._broker_address_hash(request, round_number)}.json"
        )
        receipt = self._load_broker_receipt(path, request)
        if receipt is None or receipt.get("state") != "result_persisted":
            return
        _atomic_json(
            path,
            {**receipt, "response": _serialize_response(response)},
            max_bytes=_MAX_RECEIPT_BYTES,
        )

    def _base_provenance(self, round_number: int) -> dict[str, Any]:
        return {
            "broker_schema_version": BROKER_SCHEMA_VERSION,
            "broker_policy_schema_version": BROKER_POLICY_SCHEMA_VERSION,
            "catalog_schema_version": self.policy.catalog_schema_version,
            "catalog_sha256": self.policy.catalog_sha256,
            "policy_sha256": self.policy.policy_sha256,
            "runtime_sha256": (
                self.journal_context.runtime_hash if self.journal_context else "none"
            ),
            "route": "controller",
            "round": round_number,
            "direct_shell_requested": self.policy.direct_shell_requested,
            "direct_shell_available": self.policy.direct_shell_available,
            "direct_shell_probe_id": self.policy.direct_shell_probe_id,
        }

    def _control_provenance(
        self, operation: str, round_number: int
    ) -> dict[str, Any]:
        return {
            **self._base_provenance(round_number),
            "operation_id": f"arc-companion.paper-broker.{operation}.v1",
            "network_declared": "none",
            "paper_network_authorized": self.policy.paper_network_authorized,
            "generic_internet_allowed": self.generic_internet_allowed,
            "network_observed": "none",
        }


class _Omit:
    pass


_OMIT = _Omit()


def _serialize_response(response: EvidenceResponse) -> dict[str, Any]:
    return {
        "request_id": response.request_id,
        "ok": response.ok,
        "data": response.data,
        "error": response.error,
        "provenance": dict(response.provenance),
    }


def _bounded_promotion(promotion: PromotionResult) -> dict[str, Any]:
    result = {}
    for key, values in promotion.as_dict().items():
        result[key] = values[:16]
        if len(values) > 16:
            result[f"{key}_truncated"] = True
            result[f"{key}_count"] = len(values)
    return result


def _promotion_from_receipt(receipt: Mapping[str, Any]) -> PromotionResult:
    value = receipt.get("promotion")
    if not isinstance(value, Mapping):
        raise PaperBrokerError(
            "paper_broker_receipt_corrupt", "Paper Broker promotion receipt is incomplete."
        )
    fields: dict[str, tuple[str, ...]] = {}
    for key in (
        "promoted", "deduplicated", "conflicted", "quarantined", "deleted",
    ):
        items = value.get(key, [])
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise PaperBrokerError(
                "paper_broker_receipt_corrupt", "Paper Broker promotion receipt is invalid."
            )
        fields[key] = tuple(items)
    return PromotionResult(**fields)


def _cache_status(
    promotion: PromotionResult, envelope: Mapping[str, Any] | None = None,
) -> str:
    if promotion.quarantined:
        return "validation_failed"
    if promotion.conflicted:
        return "conflict_preserved"
    if promotion.promoted:
        return "promoted"
    if promotion.deduplicated:
        return "deduplicated"
    if promotion.deleted:
        return "deleted"
    meta = envelope.get("meta") if isinstance(envelope, Mapping) else None
    observed = meta.get("cache") if isinstance(meta, Mapping) else None
    if isinstance(observed, str) and observed.strip():
        return observed.strip().lower()
    return "unknown"


def _result_hashes(value: Any) -> list[dict[str, str]]:
    found: dict[tuple[str, str], dict[str, str]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                name = str(key)
                if (
                    name in {"source_hash", "document_hash", "content_hash", "sha256"}
                    and isinstance(nested, str)
                    and nested
                ):
                    found[(name, nested)] = {"kind": name, "sha256": nested}
                else:
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return [found[key] for key in sorted(found)]


def _normalize_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, PaperBrokerError):
        return {
            "code": exc.code,
            "category": exc.category,
            "retryable": exc.retryable,
            "message": str(exc),
        }
    status = getattr(exc, "status_code", None)
    if status == 429:
        return {
            "code": "paper_rate_limited", "category": "rate_limit",
            "retryable": False, "message": "ARC-paper provider rate limited the request.",
        }
    if isinstance(status, int) and status >= 500:
        return {
            "code": "paper_provider_unavailable", "category": "transport",
            "retryable": True, "message": "ARC-paper provider is temporarily unavailable.",
        }
    if isinstance(exc, TimeoutError):
        return {
            "code": "paper_timeout", "category": "timeout",
            "retryable": True, "message": "ARC-paper operation timed out.",
        }
    if isinstance(exc, (OSError, ValueError, TypeError, PermissionError)):
        return {
            "code": "paper_local_failure", "category": "local",
            "retryable": False, "message": "ARC-paper operation failed local validation.",
        }
    return {
        "code": "paper_operation_failed", "category": "operation",
        "retryable": False, "message": "ARC-paper operation failed.",
    }


def _managed_job_public_error(code: str) -> str:
    """Return a stable model-facing message without private child diagnostics."""

    if code == "paper_broker_job_cancelled":
        return "Managed ARC-paper job was cancelled."
    if code == "paper_broker_job_needs_supervision":
        return "Managed ARC-paper job requires operator supervision."
    if code in {
        "child_budget_required",
        "child_budget_exhausted",
    }:
        return "Managed ARC-paper child budget is unavailable."
    return "Managed ARC-paper job failed; inspect its private job receipt."


def _envelope_error_metadata(error: Mapping[str, Any]) -> dict[str, Any]:
    code = str(error.get("code") or "paper_operation_failed")
    if code in {"paper_rate_limited", "rate_limited"}:
        return {"code": code, "category": "rate_limit", "retryable": False}
    if code in {"paper_timeout", "timeout"}:
        return {"code": code, "category": "timeout", "retryable": True}
    if code in {"paper_transport_failed", "paper_provider_unavailable"}:
        return {"code": code, "category": "transport", "retryable": True}
    return {"code": code, "category": "local", "retryable": False}


def _envelope_error_message(error: Mapping[str, Any]) -> str:
    code = str(error.get("code") or "paper_operation_failed").lower()
    if "not_found" in code or code.endswith("_missing"):
        return "ARC-paper data was not found."
    if code in {"paper_rate_limited", "rate_limited"}:
        return "ARC-paper provider rate limited the request."
    if code in {"paper_timeout", "timeout"}:
        return "ARC-paper operation timed out."
    if code in {"paper_transport_failed", "paper_provider_unavailable"}:
        return "ARC-paper provider is temporarily unavailable."
    return "ARC-paper operation failed local validation."


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperBrokerError(
            "paper_operation_parameters_invalid", f"{label} must be an object."
        )
    try:
        result = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PaperBrokerError(
            "paper_operation_parameters_invalid", f"{label} must be finite JSON."
        ) from exc
    return result


def _positive_byte_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PaperBrokerError(
            "paper_controller_aggregate_limit_invalid",
            "Controller aggregate byte limit must be a positive integer.",
        )
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any], *, max_bytes: int) -> None:
    payload = (_canonical_json(dict(value)) + "\n").encode("utf-8")
    if len(payload) > max_bytes:
        raise PaperBrokerError(
            "paper_broker_receipt_oversized", "Paper Broker receipt is too large."
        )
    _atomic_bytes(path, payload, mode=0o600)


def _atomic_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _chmod(path.parent, 0o700)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _chmod(path, mode)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _verified_bytes(path: Path, expected_sha256: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PaperBrokerError(
            "artifact_integrity_failed", "Artifact object is unavailable."
        ) from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PaperBrokerError(
            "artifact_integrity_failed", "Artifact object failed integrity verification."
        )
    return payload


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _sanitized_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{hostname}:{port}" if port is not None else hostname
    return parsed._replace(netloc=netloc, query="", fragment="").geturl()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _media_type(path: Path) -> str:
    return "application/json" if path.suffix.lower() == ".json" else "application/octet-stream"


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
