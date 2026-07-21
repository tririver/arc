from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Callable, Iterable, Mapping


ARTIFACT_VERSION = "arc.companion.accepted-artifact.v1"


class ArtifactStoreError(RuntimeError):
    """An accepted artifact cannot be trusted or an object was corrupted."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_id_for(
    *, kind: str, semantic_input_sha256: str, output_sha256: str, contract_version: str,
    predecessor_accepted_chain_sha256: str,
) -> str:
    return canonical_sha256({
        "kind": kind,
        "semantic_input_sha256": semantic_input_sha256,
        "output_sha256": output_sha256,
        "contract_version": contract_version,
        "predecessor_accepted_chain_sha256": predecessor_accepted_chain_sha256,
    })


class AcceptedArtifactStore:
    """Project-local, immutable storage for locally accepted model output.

    The object identity deliberately excludes recipe identity. A prompt or model
    change can therefore make an otherwise valid object recipe-stale without
    forcing another provider call.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir.resolve()
        self.root = self.project_dir / ".arc-companion" / "objects"

    def put_accepted(
        self,
        *,
        kind: str,
        semantic_input_sha256: str,
        recipe_sha256: str,
        contract_version: str,
        output: Any,
        ledger_block: Mapping[str, Any],
        provider_receipt: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        _validate_token(kind, "kind")
        for value, label in (
            (semantic_input_sha256, "semantic_input_sha256"),
            (recipe_sha256, "recipe_sha256"),
        ):
            _validate_sha(value, label)
        if not contract_version.strip():
            raise ArtifactStoreError("contract_version must be non-empty")
        if str(ledger_block.get("state") or "") != "accepted":
            raise ArtifactStoreError("only an accepted ledger block may enter the object store")
        output_sha256 = canonical_sha256(output)
        recorded_output = str(ledger_block.get("output_sha256") or "")
        if recorded_output != output_sha256:
            raise ArtifactStoreError("ledger output hash does not match the artifact output")
        recorded_input = str(ledger_block.get("input_sha256") or "")
        if recorded_input != semantic_input_sha256:
            raise ArtifactStoreError("ledger input hash does not match the semantic input")
        validation = ledger_block.get("validation_receipt")
        if not isinstance(validation, Mapping) or not validation:
            raise ArtifactStoreError("accepted artifact requires a validation receipt")
        logical_receipt = ledger_block.get("logical_receipt")
        if not isinstance(logical_receipt, Mapping) or not logical_receipt:
            raise ArtifactStoreError("accepted artifact requires a logical call receipt")
        predecessor = str(ledger_block.get("predecessor_accepted_chain_sha256") or "")
        _validate_sha(predecessor, "predecessor_accepted_chain_sha256")
        accepted_chain = str(ledger_block.get("accepted_chain_sha256") or "")
        _validate_sha(accepted_chain, "accepted_chain_sha256")
        expected_chain = hashlib.sha256(json.dumps({
            "predecessor": predecessor,
            "segment_id": ledger_block.get("segment_id"),
            "input_sha256": semantic_input_sha256,
            "output_sha256": output_sha256,
            "generation": ledger_block.get("generation"),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if accepted_chain != expected_chain:
            raise ArtifactStoreError("accepted ledger chain receipt is invalid")
        provider_receipt = dict(provider_receipt or {})
        _validate_provider_receipt(provider_receipt)
        provenance = dict(provenance or {})
        if not any(provenance.get(key) for key in ("checkpoint", "checkpoint_dir", "run_id")):
            raise ArtifactStoreError("accepted artifact requires checkpoint or run provenance")
        artifact_id = artifact_id_for(
            kind=kind,
            semantic_input_sha256=semantic_input_sha256,
            output_sha256=output_sha256,
            contract_version=contract_version,
            predecessor_accepted_chain_sha256=predecessor,
        )
        record = {
            "schema_version": ARTIFACT_VERSION,
            "artifact_id": artifact_id,
            "kind": kind,
            "semantic_input_sha256": semantic_input_sha256,
            "recipe_sha256": recipe_sha256,
            "output_sha256": output_sha256,
            "contract_version": contract_version,
            "segment_id": str(ledger_block.get("segment_id") or ""),
            "generation": ledger_block.get("generation"),
            "predecessor_accepted_chain_sha256": predecessor,
            "accepted_chain_sha256": accepted_chain,
            "validation_receipt": dict(validation),
            "logical_receipt": dict(logical_receipt),
            "provider_receipt": provider_receipt,
            "provenance": provenance,
            "output": output,
            "created_at": time.time(),
        }
        path = self.path_for(kind, artifact_id)
        if path.is_file():
            existing = self.read(kind, artifact_id)
            # Timestamps and provenance do not change an immutable accepted object.
            immutable = (
                "schema_version", "artifact_id", "kind", "semantic_input_sha256",
                "output_sha256", "contract_version", "segment_id", "generation",
                "predecessor_accepted_chain_sha256", "accepted_chain_sha256", "output",
            )
            if any(existing.get(key) != record.get(key) for key in immutable):
                raise ArtifactStoreError(f"artifact identity collision: {artifact_id}")
            return existing
        _atomic_json(path, record, exclusive=True)
        return record

    def path_for(self, kind: str, artifact_id: str) -> Path:
        _validate_token(kind, "kind")
        _validate_sha(artifact_id, "artifact_id")
        return self.root / kind / f"{artifact_id}.json"

    def read(self, kind: str, artifact_id: str) -> dict[str, Any]:
        path = self.path_for(kind, artifact_id)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactStoreError(f"could not read artifact {path}: {exc}") from exc
        if not isinstance(record, dict):
            raise ArtifactStoreError(f"artifact is not an object: {path}")
        self.validate(record, expected_kind=kind, expected_id=artifact_id)
        return record

    def validate(
        self,
        record: Mapping[str, Any],
        *,
        expected_kind: str | None = None,
        expected_id: str | None = None,
        output_validator: Callable[[Any], bool] | None = None,
    ) -> None:
        if record.get("schema_version") != ARTIFACT_VERSION:
            raise ArtifactStoreError("unsupported accepted artifact schema")
        kind = str(record.get("kind") or "")
        semantic = str(record.get("semantic_input_sha256") or "")
        output_sha = str(record.get("output_sha256") or "")
        recipe = str(record.get("recipe_sha256") or "")
        predecessor = str(record.get("predecessor_accepted_chain_sha256") or "")
        accepted_chain = str(record.get("accepted_chain_sha256") or "")
        for digest, label in (
            (semantic, "semantic_input_sha256"), (output_sha, "output_sha256"),
            (recipe, "recipe_sha256"),
            (predecessor, "predecessor_accepted_chain_sha256"),
            (accepted_chain, "accepted_chain_sha256"),
        ):
            _validate_sha(digest, label)
        contract = str(record.get("contract_version") or "")
        actual_id = artifact_id_for(
            kind=kind,
            semantic_input_sha256=semantic,
            output_sha256=output_sha,
            contract_version=contract,
            predecessor_accepted_chain_sha256=predecessor,
        )
        if expected_kind is not None and kind != expected_kind:
            raise ArtifactStoreError("artifact kind does not match its object path")
        if expected_id is not None and actual_id != expected_id:
            raise ArtifactStoreError("artifact id does not match its object path")
        if record.get("artifact_id") != actual_id:
            raise ArtifactStoreError("artifact id receipt is invalid")
        if canonical_sha256(record.get("output")) != output_sha:
            raise ArtifactStoreError("artifact output was modified")
        expected_chain = hashlib.sha256(json.dumps({
            "predecessor": predecessor,
            "segment_id": record.get("segment_id"),
            "input_sha256": semantic,
            "output_sha256": output_sha,
            "generation": record.get("generation"),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if accepted_chain != expected_chain:
            raise ArtifactStoreError("artifact accepted chain receipt is invalid")
        validation = record.get("validation_receipt")
        if not isinstance(validation, Mapping) or not validation:
            raise ArtifactStoreError("artifact validation receipt is missing")
        logical = record.get("logical_receipt")
        if not isinstance(logical, Mapping) or not logical:
            raise ArtifactStoreError("artifact logical receipt is missing")
        _validate_provider_receipt(record.get("provider_receipt"))
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping) or not any(
            provenance.get(key) for key in ("checkpoint", "checkpoint_dir", "run_id")
        ):
            raise ArtifactStoreError("artifact provenance is missing")
        if output_validator is not None and not output_validator(record.get("output")):
            raise ArtifactStoreError("artifact fails the current output contract")

    def iter_kind(self, kind: str) -> Iterable[dict[str, Any]]:
        directory = self.root / kind
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json")):
            yield self.read(kind, path.stem)

    def find(
        self,
        *,
        kind: str,
        semantic_input_sha256: str,
        contract_version: str,
        recipe_sha256: str,
        output_validator: Callable[[Any], bool] | None = None,
        predecessor_accepted_chain_sha256: str | None = None,
    ) -> dict[str, Any] | None:
        candidates = []
        for record in self.iter_kind(kind):
            if (
                record.get("semantic_input_sha256") == semantic_input_sha256
                and record.get("contract_version") == contract_version
                and (
                    predecessor_accepted_chain_sha256 is None
                    or record.get("predecessor_accepted_chain_sha256")
                    == predecessor_accepted_chain_sha256
                )
            ):
                try:
                    self.validate(record, output_validator=output_validator)
                except ArtifactStoreError:
                    continue
                candidates.append(record)
        if not candidates:
            return None
        exact = [item for item in candidates if item.get("recipe_sha256") == recipe_sha256]
        selected = max(exact or candidates, key=lambda item: float(item.get("created_at") or 0))
        return {
            **selected,
            "reuse_status": "hit" if selected.get("recipe_sha256") == recipe_sha256 else "recipe_stale",
        }


def _validate_sha(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ArtifactStoreError(f"{label} must be a lowercase SHA-256")


def _validate_token(value: str, label: str) -> None:
    if not value or any(not (character.isalnum() or character in "-_.") for character in value):
        raise ArtifactStoreError(f"{label} contains unsafe characters")


def _validate_provider_receipt(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ArtifactStoreError("accepted artifact requires a provider receipt")
    missing = [key for key in ("provider", "model", "call_id", "usage") if key not in value]
    if missing:
        raise ArtifactStoreError(
            f"provider receipt is missing auditable fields: {', '.join(missing)}"
        )
    if not str(value.get("provider") or "") or not str(value.get("model") or ""):
        raise ArtifactStoreError("provider receipt provider and model must be non-empty")
    if not str(value.get("call_id") or ""):
        raise ArtifactStoreError("provider receipt call_id must be non-empty")
    if not isinstance(value.get("usage"), Mapping):
        raise ArtifactStoreError("provider receipt usage must be an object")


def _atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
