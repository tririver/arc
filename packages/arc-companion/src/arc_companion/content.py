from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .io import read_json, sha256_json, write_json


READER_CONTENT_VERSION = "arc.companion.reader-content.v2"
CONTENT_RECEIPT_VERSION = "arc.companion.reader-content-validation.v2"
CONTENT_OBJECT_KIND = "reader-content"
_LEGACY_READER_CONTENT_VERSIONS = {"arc.companion.reader-content.v1"}


class ContentBundleError(RuntimeError):
    """A reviewed-content object is missing, malformed, or has changed."""


def store_reader_content(
    project_dir: Path,
    *,
    content: Mapping[str, Any],
    checkpoint_dir: Path | None = None,
    review_receipts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and atomically store one immutable reviewed-content object."""
    root = project_dir.resolve()
    value = deepcopy(dict(content))
    checks = _validate_content(value)
    content_sha256 = sha256_json(value)
    receipts = deepcopy(dict(review_receipts or {}))
    provenance = {
        "checkpoint_dir": str(checkpoint_dir.resolve()) if checkpoint_dir else None,
    }
    receipt_payload = {
        "schema_version": CONTENT_RECEIPT_VERSION,
        "validator_version": CONTENT_RECEIPT_VERSION,
        "content_sha256": content_sha256,
        "checks": checks,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    bundle_sha256 = sha256_json({
        "content_sha256": content_sha256,
        "content": value,
        "review_receipts": receipts,
        "provenance": provenance,
        "validation_receipt": receipt_payload,
    })
    envelope = {
        "schema_version": READER_CONTENT_VERSION,
        "content_sha256": content_sha256,
        "bundle_sha256": bundle_sha256,
        "content": value,
        "validation_receipt": {
            **receipt_payload,
            "bundle_sha256": bundle_sha256,
        },
        "review_receipts": receipts,
        "provenance": provenance,
    }
    path = content_object_path(root, content_sha256)
    if path.is_file():
        try:
            existing = load_reader_content(root, content_sha256)
        except ContentBundleError:
            try:
                stale = read_json(path)
            except (OSError, ValueError) as exc:
                raise ContentBundleError(
                    "content-addressed reader object is unreadable"
                ) from exc
            if (
                not isinstance(stale, dict)
                or stale.get("schema_version") not in _LEGACY_READER_CONTENT_VERSIONS
                or stale.get("content") != value
                or stale.get("content_sha256") != content_sha256
                or sha256_json(stale.get("content")) != content_sha256
            ):
                raise ContentBundleError(
                    "content-addressed reader object has conflicting bytes"
                )
            # The object path is payload-addressed, so a valid legacy envelope
            # must be refreshed in place before the v2 loader can accept it.
            write_json(path, envelope)
        else:
            if existing["content"] != value:
                raise ContentBundleError("content-addressed reader object has conflicting bytes")
            return {**existing, "path": path}
        return {**load_reader_content(root, content_sha256), "path": path}
    write_json(path, envelope)
    return {**load_reader_content(root, content_sha256), "path": path}


def content_object_path(project_dir: Path, content_sha256: str) -> Path:
    digest = _digest(content_sha256)
    return project_dir.resolve() / ".arc-companion" / "objects" / CONTENT_OBJECT_KIND / f"{digest}.json"


def load_reader_content(project_dir: Path, content_sha256: str) -> dict[str, Any]:
    path = content_object_path(project_dir, content_sha256)
    try:
        envelope = read_json(path)
    except (OSError, ValueError) as exc:
        raise ContentBundleError(f"reviewed-content object is unavailable: {path}") from exc
    if not isinstance(envelope, dict) or envelope.get("schema_version") != READER_CONTENT_VERSION:
        raise ContentBundleError("reviewed-content object schema is invalid")
    content = envelope.get("content")
    if not isinstance(content, dict):
        raise ContentBundleError("reviewed-content payload is missing")
    actual = sha256_json(content)
    if actual != content_sha256 or envelope.get("content_sha256") != actual:
        raise ContentBundleError("reviewed-content object hash does not match its identity")
    _validate_content(content)
    review_receipts = envelope.get("review_receipts")
    provenance = envelope.get("provenance")
    if not isinstance(review_receipts, dict) or not isinstance(provenance, dict):
        raise ContentBundleError("reviewed-content provenance is invalid")
    for value in review_receipts.values():
        if (
            not isinstance(value, dict)
            or not _sha(value.get("sha256"))
            or not isinstance(value.get("bytes"), int)
            or isinstance(value.get("bytes"), bool)
            or value.get("bytes") < 1
        ):
            raise ContentBundleError("reviewed-content review receipt is invalid")
    receipt = envelope.get("validation_receipt")
    if not isinstance(receipt, dict):
        raise ContentBundleError("reviewed-content validation receipt is invalid")
    receipt_payload = {
        key: value for key, value in receipt.items() if key != "bundle_sha256"
    }
    bundle_sha256 = sha256_json({
        "content_sha256": actual,
        "content": content,
        "review_receipts": review_receipts,
        "provenance": provenance,
        "validation_receipt": receipt_payload,
    })
    if envelope.get("bundle_sha256") != bundle_sha256:
        raise ContentBundleError("reviewed-content bundle hash does not match")
    if (
        receipt.get("schema_version") != CONTENT_RECEIPT_VERSION
        or receipt.get("validator_version") != CONTENT_RECEIPT_VERSION
        or receipt.get("content_sha256") != actual
        or receipt.get("bundle_sha256") != bundle_sha256
        or receipt.get("checks") != _validate_content(content)
        or not isinstance(receipt.get("validated_at"), str)
        or not receipt.get("validated_at")
    ):
        raise ContentBundleError("reviewed-content validation receipt is invalid")
    return envelope


def reader_content_from_overrides(
    overrides: Mapping[str, Any],
    *,
    reader_evidence_by_segment: Mapping[str, Any],
    accepted_ledger_chains: Mapping[str, Any] | None = None,
    review_overlay_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the stable render input from the post-review reader boundary."""
    content = {
        "document": deepcopy(overrides.get("document")),
        "chapters": deepcopy(overrides.get("chapters") or []),
        "segments": deepcopy(overrides.get("segments") or []),
        "chapter_guides": deepcopy(overrides.get("chapter_guides") or {}),
        "translations": deepcopy(overrides.get("translations")),
        "annotations": deepcopy(overrides.get("annotations") or {}),
        "glossary": deepcopy(overrides.get("glossary") or {}),
        "metadata": deepcopy(overrides.get("metadata") or {}),
        "reader_evidence_by_segment": deepcopy(dict(reader_evidence_by_segment)),
        "language": str(overrides.get("language") or ""),
        "source_language": str(overrides.get("source_language") or "und"),
        "translation_mode": str(overrides.get("translation_mode") or "enabled"),
        "accepted_ledger_chains": deepcopy(dict(accepted_ledger_chains or {})),
        "review_overlay_hashes": deepcopy(dict(review_overlay_hashes or {})),
    }
    # Title translation was added after the original immutable-content
    # contract.  Preserve read compatibility for callers that intentionally
    # construct a legacy payload while making every new pipeline payload
    # explicit.
    if "title_translations" in overrides:
        content["title_translations"] = deepcopy(overrides.get("title_translations"))
    return content


def checkpoint_receipts(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Collect final accepted chains and review overlay identities for provenance."""
    chains: dict[str, Any] = {}
    for path in sorted(checkpoint_dir.glob("chapters/**/*-ledger.json")):
        try:
            ledger = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(ledger, dict):
            continue
        blocks = ledger.get("blocks") or []
        if (
            isinstance(blocks, list)
            and blocks
            and all(isinstance(item, dict) and item.get("state") == "accepted" for item in blocks)
            and isinstance(ledger.get("accepted_chain_sha256"), str)
        ):
            predecessor = hashlib.sha256(b"").hexdigest()
            valid_chain = True
            for block in blocks:
                expected = hashlib.sha256(json.dumps({
                    "predecessor": predecessor,
                    "segment_id": block.get("segment_id"),
                    "input_sha256": block.get("input_sha256"),
                    "output_sha256": block.get("output_sha256"),
                    "generation": block.get("generation"),
                }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                if (
                    block.get("predecessor_accepted_chain_sha256") != predecessor
                    or block.get("accepted_chain_sha256") != expected
                ):
                    valid_chain = False
                    break
                predecessor = expected
            if not valid_chain or ledger.get("accepted_chain_sha256") != predecessor:
                continue
            key = path.relative_to(checkpoint_dir).as_posix()
            chains[key] = {
                "generation": ledger.get("generation"),
                "accepted_chain_sha256": ledger.get("accepted_chain_sha256"),
                "segment_ids": [str(item.get("segment_id") or "") for item in blocks],
            }
    overlays = {
        path.relative_to(checkpoint_dir).as_posix(): sha256_json(read_json(path))
        for path in sorted(checkpoint_dir.glob("chapters/**/*-review-overlay.json"))
    }
    return chains, overlays


def _validate_content(content: Mapping[str, Any]) -> list[str]:
    document = content.get("document")
    segments = content.get("segments")
    annotations = content.get("annotations")
    if not isinstance(document, Mapping) or not isinstance(document.get("blocks"), list):
        raise ContentBundleError("reviewed content has no source document blocks")
    if not isinstance(segments, list) or not segments:
        raise ContentBundleError("reviewed content has no segments")
    segment_ids = [str(item.get("segment_id") or "") for item in segments if isinstance(item, Mapping)]
    if len(segment_ids) != len(segments) or not all(segment_ids) or len(set(segment_ids)) != len(segment_ids):
        raise ContentBundleError("reviewed content segment identities are invalid")
    if not isinstance(annotations, Mapping) or set(map(str, annotations)) != set(segment_ids):
        raise ContentBundleError("reviewed commentary does not cover the segment set")
    mode = str(content.get("translation_mode") or "")
    translations = content.get("translations")
    if mode == "skipped":
        if translations is not None:
            raise ContentBundleError("skip-translation content must store translations as null")
    elif mode == "enabled":
        if not isinstance(translations, Mapping) or set(map(str, translations)) != set(segment_ids):
            raise ContentBundleError("reviewed translations do not cover the segment set")
    else:
        raise ContentBundleError("reviewed content translation mode is invalid")
    evidence = content.get("reader_evidence_by_segment")
    if not isinstance(evidence, Mapping) or set(map(str, evidence)) != set(segment_ids):
        raise ContentBundleError("reader evidence projection does not cover the segment set")
    if not isinstance(content.get("language"), str) or not content.get("language"):
        raise ContentBundleError("reviewed content language is missing")
    if "source_language" in content and (
        not isinstance(content.get("source_language"), str)
        or not str(content.get("source_language") or "").strip()
    ):
        raise ContentBundleError("reviewed content source language is invalid")
    chapters = content.get("chapters")
    guides = content.get("chapter_guides")
    if not isinstance(chapters, list) or not all(isinstance(item, Mapping) for item in chapters):
        raise ContentBundleError("reviewed content chapters are invalid")
    if not isinstance(guides, Mapping):
        raise ContentBundleError("reviewed content chapter guides are invalid")
    chapter_ids = {str(item.get("chapter_id") or "") for item in chapters}
    if "" in chapter_ids or (chapter_ids and set(map(str, guides)) != chapter_ids):
        raise ContentBundleError("reviewed chapter guides do not match the chapter set")
    if not isinstance(content.get("glossary"), Mapping):
        raise ContentBundleError("reviewed content glossary is invalid")
    if not isinstance(content.get("metadata"), Mapping):
        raise ContentBundleError("reviewed content metadata is invalid")
    if "title_translations" in content:
        title_translations = content.get("title_translations")
        if mode == "skipped":
            if title_translations is not None:
                raise ContentBundleError(
                    "skip-translation content must store title translations as null"
                )
        else:
            _validate_title_translation_content(
                content, title_translations=title_translations,
            )
    chains = content.get("accepted_ledger_chains")
    if not isinstance(chains, Mapping):
        raise ContentBundleError("accepted ledger chain receipts are invalid")
    for receipt in chains.values():
        if (
            not isinstance(receipt, Mapping)
            or not _sha(receipt.get("accepted_chain_sha256"))
            or not isinstance(receipt.get("segment_ids"), list)
        ):
            raise ContentBundleError("accepted ledger chain receipt is malformed")
    overlays = content.get("review_overlay_hashes")
    if not isinstance(overlays, Mapping) or any(
        not _sha(value) for value in overlays.values()
    ):
        raise ContentBundleError("review overlay receipts are invalid")
    checks = [
        "source_document_present",
        "segment_ids_unique",
        "commentary_coverage_exact",
        "translation_mode_consistent",
        "reader_evidence_coverage_exact",
        "chapter_guide_shape_valid",
        "glossary_and_metadata_valid",
        "accepted_chain_receipts_valid",
        "review_overlay_receipts_valid",
    ]
    if "title_translations" in content:
        checks.insert(-2, "title_translation_contract_valid")
    return checks


def _validate_title_translation_content(
    content: Mapping[str, Any], *, title_translations: Any,
) -> None:
    if not isinstance(title_translations, Mapping):
        raise ContentBundleError("reviewed title translations are missing")
    titles = title_translations.get("titles")
    if not isinstance(titles, list):
        raise ContentBundleError("reviewed title translations are malformed")
    try:
        from .title_translation import collect_title_records

        projection_document = {
            **dict(content.get("document") or {}),
            "metadata": dict(content.get("metadata") or {}),
        }
        records = collect_title_records(
            projection_document,
            list(content.get("chapters") or []),
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise ContentBundleError("reviewed title projection is invalid") from exc
    expected = [str(item.get("title_id") or "") for item in records]
    provided = [
        str(item.get("title_id") or "")
        for item in titles if isinstance(item, Mapping)
    ]
    if (
        len(provided) != len(titles)
        or provided != expected
        or len(provided) != len(set(provided))
        or any(
            not isinstance(item.get("text"), str) or not item.get("text", "").strip()
            for item in titles if isinstance(item, Mapping)
        )
    ):
        raise ContentBundleError(
            "reviewed title translations do not exactly cover the source title projection"
        )


def _digest(value: str) -> str:
    if not _sha(value):
        raise ContentBundleError("content SHA-256 is invalid")
    return value


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
