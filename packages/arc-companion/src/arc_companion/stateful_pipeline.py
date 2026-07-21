from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, Iterator, Mapping
import uuid


STATEFUL_TURN_VERSION = "arc.companion.stateful-turn.v2"
STATEFUL_STREAM_STATE_VERSION = "arc.companion.stateful-stream-state.v1"
LANE_RUNTIME_PROFILE_VERSION = "arc.companion.lane-runtime-profile.v1"
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
ROLLOVER_FRACTION = 0.70


class StatefulSessionError(RuntimeError):
    """A paid stateful turn completed but its session invariant was not accepted."""


class LLMSubmissionLimiter:
    """One build-wide permit pool around provider submissions themselves."""

    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self._semaphore = BoundedSemaphore(workers)

    @contextmanager
    def permit(self) -> Iterator[None]:
        with self._semaphore:
            yield


@dataclass
class StatefulPromptStream:
    """Construct one static bootstrap followed by segment-local delta turns."""

    chapter_id: str
    lane: str
    fixed_rules: Mapping[str, Any]
    static_context: Mapping[str, Any]
    generation: int = 1
    continuity_capsule: Mapping[str, Any] | None = None
    _turns: int = 0

    @property
    def turn_count(self) -> int:
        return self._turns

    def reconcile_turn_count(self, accepted_receipt_turns: int) -> None:
        """Restore the native stream position from durable session receipts."""
        if accepted_receipt_turns < 0:
            raise ValueError("accepted_receipt_turns cannot be negative")
        self._turns = accepted_receipt_turns

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": STATEFUL_STREAM_STATE_VERSION,
            "chapter_id": self.chapter_id,
            "lane": self.lane,
            "generation": self.generation,
            "bootstrap": {
                "fixed_rules": dict(self.fixed_rules),
                "static_context": dict(self.static_context),
            },
            "turn_count": self._turns,
            "continuity_capsule": dict(self.continuity_capsule or {}),
        }

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], *, receipt_turn_count: int | None = None,
    ) -> "StatefulPromptStream":
        if state.get("schema_version") != STATEFUL_STREAM_STATE_VERSION:
            raise ValueError("unsupported stateful stream state schema")
        bootstrap = state.get("bootstrap")
        if not isinstance(bootstrap, Mapping):
            raise ValueError("stateful stream state is missing bootstrap")
        stream = cls(
            chapter_id=str(state.get("chapter_id") or ""),
            lane=str(state.get("lane") or ""),
            generation=int(state.get("generation") or 1),
            fixed_rules=dict(bootstrap.get("fixed_rules") or {}),
            static_context=dict(bootstrap.get("static_context") or {}),
            continuity_capsule=dict(state.get("continuity_capsule") or {}),
        )
        stream.reconcile_turn_count(
            int(state.get("turn_count") or 0)
            if receipt_turn_count is None else receipt_turn_count
        )
        return stream

    def request(
        self,
        request: str,
        *,
        cursor: str,
        source_sha256: str,
        current_payload: Mapping[str, Any],
    ) -> str:
        segment_payload = {"request": request, **dict(current_payload)}
        if self._turns == 0:
            payload = {
                "schema_version": STATEFUL_TURN_VERSION,
                "turn_kind": "generation_bootstrap",
                "generation": self.generation,
                "chapter_id": self.chapter_id,
                "lane": self.lane,
                "fixed_rules": dict(self.fixed_rules),
                "static_context": dict(self.static_context),
                "continuity_capsule": dict(self.continuity_capsule or {}),
                "cursor": cursor,
                "source_sha256": source_sha256,
                "current_payload": segment_payload,
            }
        else:
            payload = {
                "schema_version": STATEFUL_TURN_VERSION,
                "turn_kind": "delta",
                "cursor": cursor,
                "source_sha256": source_sha256,
                "current_payload": segment_payload,
            }
        self._turns += 1
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class ContextRolloverBudget:
    """Track one native generation and request rollover only at accepted boundaries."""

    context_window_tokens: int = field(default_factory=lambda: _configured_context_window())
    input_tokens: int = 0
    output_tokens: int = 0
    _has_known_input: bool = False
    _max_known_footprint: int = 0
    _unmeasured_tokens: int = 0

    @property
    def threshold(self) -> int:
        return int(self.context_window_tokens * ROLLOVER_FRACTION)

    def record(self, usage: Any, *, prompt_bytes: int | None = None) -> None:
        payload = usage.to_json() if hasattr(usage, "to_json") else usage
        payload = payload if isinstance(payload, Mapping) else {}
        known_input = payload.get("total_input_tokens", payload.get("input_tokens"))
        known_output = payload.get("output_tokens")
        if isinstance(known_input, int):
            # Stateful providers commonly report the complete turn input,
            # including native history.  The maximum is therefore the best
            # context-size observation and avoids double-counting history.
            self.input_tokens = max(self.input_tokens, max(0, known_input))
            self._has_known_input = True
            turn_output = max(0, known_output) if isinstance(known_output, int) else 0
            self._max_known_footprint = max(
                self._max_known_footprint, max(0, known_input) + turn_output
            )
            # A later complete-input observation already contains earlier turns.
            self._unmeasured_tokens = 0
        elif prompt_bytes:
            estimate = max(1, prompt_bytes // 4)
            self.input_tokens += estimate
            if self._has_known_input:
                self._unmeasured_tokens += estimate
        if isinstance(known_output, int):
            if isinstance(known_input, int):
                self.output_tokens = max(self.output_tokens, max(0, known_output))
            else:
                self.output_tokens += max(0, known_output)
                if self._has_known_input:
                    self._unmeasured_tokens += max(0, known_output)

    def rollover_due(self) -> bool:
        if self._has_known_input:
            return self._max_known_footprint + self._unmeasured_tokens >= self.threshold
        return self.input_tokens + self.output_tokens >= self.threshold

    def to_state(self) -> dict[str, Any]:
        return {
            "context_window_tokens": self.context_window_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "has_known_input": self._has_known_input,
            "max_known_footprint": self._max_known_footprint,
            "unmeasured_tokens": self._unmeasured_tokens,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ContextRolloverBudget":
        budget = cls(context_window_tokens=int(state.get("context_window_tokens") or _configured_context_window()))
        budget.input_tokens = int(state.get("input_tokens") or 0)
        budget.output_tokens = int(state.get("output_tokens") or 0)
        budget._has_known_input = bool(state.get("has_known_input"))
        budget._max_known_footprint = int(state.get("max_known_footprint") or 0)
        budget._unmeasured_tokens = int(state.get("unmeasured_tokens") or 0)
        return budget

    @classmethod
    def from_turn_records(cls, records: Iterator[Mapping[str, Any]] | list[Mapping[str, Any]]) -> "ContextRolloverBudget":
        budget = cls()
        for record in records:
            budget.record(record.get("usage") if isinstance(record, Mapping) else {})
        return budget


class CorrectionBudget:
    """Allow at most one model correction turn for a logical source block."""

    def __init__(self) -> None:
        self._used: set[str] = set()
        self._lock = Lock()

    def consume(self, segment_id: str) -> None:
        with self._lock:
            if segment_id in self._used:
                raise RuntimeError(
                    f"stateful correction turn already consumed for {segment_id}"
                )
            self._used.add(segment_id)


def continuity_capsule(
    *, accepted_chain_sha256: str, segment_id: str, input_sha256: str, output_sha256: str
) -> dict[str, str]:
    return {
        "accepted_chain_sha256": accepted_chain_sha256,
        "last_accepted_segment_id": segment_id,
        "last_input_sha256": input_sha256,
        "last_output_sha256": output_sha256,
    }


def write_stream_state(
    path: Path, *, stream: StatefulPromptStream, budget: ContextRolloverBudget,
) -> None:
    payload = stream.to_state()
    payload["rollover_usage"] = budget.to_state()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_stream_state(
    path: Path, *, receipt_turn_count: int | None = None,
) -> tuple[StatefulPromptStream, ContextRolloverBudget] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read stateful stream state {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"stateful stream state is not an object: {path}")
    stream = StatefulPromptStream.from_state(payload, receipt_turn_count=receipt_turn_count)
    usage = payload.get("rollover_usage")
    budget = ContextRolloverBudget.from_state(usage if isinstance(usage, Mapping) else {})
    return stream, budget


def resolve_lane_runtime_profile(
    path: Path, *, chapter_id: str, lane: str, generation: int,
    requested_allow_internet: bool, inherit_host_tools: bool,
    existing_generation: bool, recorded_runtime_fingerprint: str | None,
    provider: str = "auto", model: str | None = None,
    model_tier: str | None = None,
) -> dict[str, Any]:
    """Pin one access recipe for every primary/repair call in a generation."""
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        existing = None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read lane runtime profile {path}: {exc}") from exc
    identity = {
        "chapter_id": chapter_id, "lane": lane, "generation": generation,
    }
    if isinstance(existing, Mapping):
        if existing.get("schema_version") != LANE_RUNTIME_PROFILE_VERSION or any(
            existing.get(key) != value for key, value in identity.items()
        ):
            raise ValueError(f"lane runtime profile identity changed: {path}")
        return dict(existing)
    # A pre-profile generation may already own a paid native session. Preserve
    # the user's access choice for that generation; only newly created
    # translation generations adopt the uniform offline recipe.
    allow_internet = (
        requested_allow_internet
        if lane != "translation" or existing_generation
        else False
    )
    profile = {
        "schema_version": LANE_RUNTIME_PROFILE_VERSION,
        **identity,
        "allow_internet": bool(allow_internet),
        "inherit_host_tools": bool(inherit_host_tools),
        "provider": provider,
        "model": model,
        "model_tier": model_tier,
        "recorded_runtime_fingerprint": recorded_runtime_fingerprint,
    }
    _atomic_write_json(path, profile)
    return profile


def pin_lane_runtime_profile(
    path: Path, profile: Mapping[str, Any], *, provider: str,
    model: str | None, runtime_fingerprint: str,
    migrated_from_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Persist the provider-resolved identity after a generation's first call."""
    pinned = dict(profile)
    prior_provider = str(pinned.get("provider") or "auto")
    prior_model = pinned.get("model")
    if prior_provider not in {"auto", provider}:
        raise ValueError("lane runtime provider changed after generation start")
    if prior_model is not None and prior_model != model:
        raise ValueError("lane runtime model changed after generation start")
    recorded = pinned.get("recorded_runtime_fingerprint")
    if (
        recorded is not None and recorded != runtime_fingerprint
        and recorded != migrated_from_fingerprint
    ):
        # Profiles are initially provisional when provider auto-selection has
        # not completed.  Older rollover code could copy the preceding
        # generation's fingerprint into that otherwise-unpinned profile.  A
        # completed call may pin its actual provider manifest exactly once;
        # already provider/model-pinned profiles still reject all drift.
        if prior_provider != "auto" or prior_model is not None:
            raise ValueError("lane runtime fingerprint changed after generation start")
        pinned["migrated_from_runtime_fingerprint"] = recorded
    pinned.update({
        "provider": provider,
        "model": model,
        "recorded_runtime_fingerprint": runtime_fingerprint,
    })
    _atomic_write_json(path, pinned)
    return pinned


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _configured_context_window() -> int:
    raw = os.environ.get("ARC_LLM_CONTEXT_WINDOW_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            pass
        else:
            if value > 0:
                return value
    return DEFAULT_CONTEXT_WINDOW_TOKENS
