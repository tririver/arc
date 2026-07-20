from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


MAX_EVIDENCE_ROUNDS = 3
EVIDENCE_REQUESTS_FIELD = "arc_evidence_requests"


class EvidenceProtocolError(ValueError):
    """Raised when a worker emits an invalid evidence request envelope."""


@dataclass(frozen=True)
class EvidenceRequest:
    """A protocol-neutral request from an LLM worker to its controller.

    ``operation`` and ``arguments`` belong to the embedding workflow.  Keeping
    them opaque here lets controllers call Python services, CLIs, remote APIs,
    or test doubles without coupling ``arc-llm`` to any ARC data package.
    """

    request_id: str
    operation: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    worker_id: str = ""
    role: str = ""

    def __post_init__(self) -> None:
        request_id = self.request_id.strip()
        operation = self.operation.strip()
        if not request_id:
            raise ValueError("evidence request_id is required")
        if not operation:
            raise ValueError("evidence operation is required")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "reason", self.reason.strip())
        object.__setattr__(self, "worker_id", self.worker_id.strip())
        object.__setattr__(self, "role", self.role.strip())


@dataclass(frozen=True)
class EvidenceResponse:
    """The controller result for exactly one :class:`EvidenceRequest`."""

    request_id: str
    ok: bool
    data: Any = None
    error: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        request_id = self.request_id.strip()
        if not request_id:
            raise ValueError("evidence response request_id is required")
        if self.ok and self.error:
            raise ValueError("successful evidence response cannot contain an error")
        if not self.ok and not str(self.error or "").strip():
            raise ValueError("failed evidence response requires an error")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "provenance", dict(self.provenance))


@runtime_checkable
class EvidenceControllerCallback(Protocol):
    """Controller callback invoked outside the worker's tool environment."""

    def __call__(
        self,
        requests: tuple[EvidenceRequest, ...],
        *,
        round_number: int,
    ) -> Iterable[EvidenceResponse]: ...


def resolve_evidence_round(
    requests: Iterable[EvidenceRequest],
    controller: EvidenceControllerCallback,
    *,
    round_number: int,
    max_rounds: int = MAX_EVIDENCE_ROUNDS,
) -> tuple[EvidenceResponse, ...]:
    """Resolve one bounded controller round and validate request correlation."""

    if max_rounds < 1 or max_rounds > MAX_EVIDENCE_ROUNDS:
        raise ValueError(f"max_rounds must be between 1 and {MAX_EVIDENCE_ROUNDS}")
    if round_number < 1 or round_number > max_rounds:
        raise ValueError(f"evidence round {round_number} exceeds max_rounds={max_rounds}")

    material = tuple(requests)
    request_ids = [item.request_id for item in material]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("evidence request IDs must be unique within a round")
    if not material:
        return ()

    responses = tuple(controller(material, round_number=round_number))
    response_ids = [item.request_id for item in responses]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("evidence response IDs must be unique within a round")
    missing = sorted(set(request_ids) - set(response_ids))
    unexpected = sorted(set(response_ids) - set(request_ids))
    if missing or unexpected:
        raise ValueError(
            "evidence controller response IDs do not match requests: "
            f"missing={missing}, unexpected={unexpected}"
        )
    by_id = {item.request_id: item for item in responses}
    return tuple(by_id[request_id] for request_id in request_ids)


def allow_evidence_requests(schema: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Allow ARC's optional evidence request field in an object schema.

    Worker schemas remain otherwise unchanged.  This mirrors the package's
    call-record extension and keeps workflow-authored result fields stable.
    """

    if schema is None:
        return None
    result = copy.deepcopy(dict(schema))
    if result.get("type") != "object":
        return result
    properties = result.get("properties")
    if properties is None:
        properties = {}
        result["properties"] = properties
    if not isinstance(properties, dict):
        return result
    properties.setdefault(EVIDENCE_REQUESTS_FIELD, evidence_requests_schema())
    required = result.setdefault("required", [])
    if isinstance(required, list) and EVIDENCE_REQUESTS_FIELD not in required:
        required.append(EVIDENCE_REQUESTS_FIELD)
    return result


def evidence_requests_schema() -> dict[str, Any]:
    """Return the protocol-neutral JSON Schema for worker evidence requests."""

    return {
        "type": "array",
        "maxItems": 32,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["request_id", "operation", "arguments", "reason"],
            "properties": {
                "request_id": {"type": "string", "minLength": 1},
                "operation": {"type": "string", "minLength": 1},
                "arguments": {"type": "object"},
                "reason": {"type": "string"},
            },
        },
    }


def evidence_requests_from_output(
    output: Any,
    *,
    worker_id: str,
    role: str,
) -> tuple[EvidenceRequest, ...]:
    """Parse and validate the reserved evidence field from one worker result."""

    if not isinstance(output, Mapping) or EVIDENCE_REQUESTS_FIELD not in output:
        return ()
    raw_requests = output[EVIDENCE_REQUESTS_FIELD]
    if not isinstance(raw_requests, list):
        raise EvidenceProtocolError(f"{worker_id}.{EVIDENCE_REQUESTS_FIELD} must be an array")
    if len(raw_requests) > 32:
        raise EvidenceProtocolError(f"{worker_id}.{EVIDENCE_REQUESTS_FIELD} must contain at most 32 requests")

    requests: list[EvidenceRequest] = []
    for index, raw_request in enumerate(raw_requests):
        field = f"{worker_id}.{EVIDENCE_REQUESTS_FIELD}[{index}]"
        if not isinstance(raw_request, Mapping):
            raise EvidenceProtocolError(f"{field} must be an object")
        unexpected = sorted(set(raw_request) - {"request_id", "operation", "arguments", "reason"})
        if unexpected:
            raise EvidenceProtocolError(f"{field} has unexpected fields: {', '.join(unexpected)}")
        missing = sorted({"request_id", "operation", "arguments", "reason"} - set(raw_request))
        if missing:
            raise EvidenceProtocolError(f"{field} is missing required fields: {', '.join(missing)}")
        arguments = raw_request.get("arguments")
        if not isinstance(arguments, Mapping):
            raise EvidenceProtocolError(f"{field}.arguments must be an object")
        try:
            request = EvidenceRequest(
                request_id=str(raw_request.get("request_id") or ""),
                operation=str(raw_request.get("operation") or ""),
                arguments=dict(arguments),
                reason=str(raw_request.get("reason") or ""),
                worker_id=worker_id,
                role=role,
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceProtocolError(f"invalid {field}: {exc}") from exc
        requests.append(request)

    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise EvidenceProtocolError(f"{worker_id} evidence request IDs must be unique within a worker result")
    return tuple(requests)
