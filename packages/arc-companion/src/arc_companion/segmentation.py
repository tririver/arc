from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from typing import Any, Callable

from .io import read_json, sha256_json, write_json
from .prompts import CUT_SCHEMA, segmentation_prompt
from .projection import annotation_input_block, is_translatable, translation_input_block
from .source import block_id


SEGMENTATION_VERSION = "arc.companion.segmentation.v5"
WINDOW_MAX_BLOCKS = 100
WINDOW_MAX_PROJECTED_CHARS = 30_000
SEGMENT_HARD_MAX_BLOCKS = 24
SEGMENT_HARD_MAX_SOURCE_CHARS = 60_000
MAX_VALIDATION_ATTEMPTS = 3


class SegmentationError(RuntimeError):
    """Raised when medium-model semantic cuts cannot be validated safely."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context or {})

    def diagnostic(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "severity": "error",
            "code": "companion_segmentation_failed",
            "source": "arc-companion",
            "message": str(self),
        }
        if self.context:
            record["context"] = dict(self.context)
        return record


def build_block_inventory(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the compact, type-aware, ordinal inventory shown to the model."""
    blocks = document.get("blocks") or []
    section_titles: dict[str, str] = {}
    section_ordinals: dict[str, int] = {}
    for block in blocks:
        kind = str(block.get("type") or block.get("kind") or "").lower()
        section_id = str(block.get("section_id") or "")
        if section_id and section_id not in section_ordinals:
            section_ordinals[section_id] = len(section_ordinals) + 1
        if section_id and kind in {"heading", "section", "subsection", "subsubsection"}:
            section_titles.setdefault(section_id, _preview(block.get("title") or block.get("text"), 300))
    entities = {
        "equation": _entity_index(document.get("equations") or []),
        "figure": _entity_index(document.get("figures") or []),
        "table": _entity_index(document.get("tables") or []),
        "bibliography": _entity_index(document.get("bibliography") or []),
    }
    inventory: list[dict[str, Any]] = []
    for ordinal, block in enumerate(blocks, start=1):
        kind = str(block.get("type") or block.get("kind") or "prose").lower()
        entity = _entity_for_block(block, entities.get(kind, {}))
        record: dict[str, Any] = {
            "ordinal": ordinal,
            "type": kind,
        }
        section_id = str(block.get("section_id") or "")
        if section_id:
            record["section_ordinal"] = section_ordinals[section_id]
            section_title = (
                str(block.get("section_title") or "") or section_titles.get(section_id, "")
            )
            if section_title:
                record["section_title"] = section_title
        if kind in {"heading", "section", "subsection", "subsubsection"}:
            record["title"] = _preview(block.get("title") or block.get("text"), 1_200)
        elif kind in {"equation", "math", "display_math"}:
            tex = (entity or {}).get("tex") or block.get("tex")
            if isinstance(tex, list):
                tex = " ; ".join(str(value) for value in tex)
            record["formula"] = _preview(tex or block.get("text"), 800)
            numbers = (entity or {}).get("printed_equation_numbers") or []
            if numbers:
                record["numbers"] = list(numbers) if isinstance(numbers, list) else [str(numbers)]
        elif kind in {"figure", "table"}:
            source = entity or block
            record["tag"] = _preview(source.get("tag") or source.get("number"), 120)
            record["caption"] = _preview(source.get("caption") or block.get("text"), 800)
        elif kind in {"bibliography", "bibliography_item", "reference"}:
            source = entity or block
            record["label"] = _preview(source.get("label"), 120)
            record["citation"] = _preview(source.get("text") or block.get("text"), 800)
        elif kind == "list":
            record["text"] = _preview(block.get("text"), 1_200)
            record["item_count"] = len(block.get("items") or block.get("list_items") or [])
        else:
            record["text"] = _preview(block.get("text") or block.get("title"), 1_200)
        inventory.append(record)
    return inventory


def build_segmentation_windows(
    inventory: list[dict[str, Any]],
    *,
    max_blocks: int = WINDOW_MAX_BLOCKS,
    max_projected_chars: int = WINDOW_MAX_PROJECTED_CHARS,
) -> list[dict[str, Any]]:
    """Partition model ownership while the controller closes every window with a cut."""
    if not inventory:
        raise SegmentationError("block inventory is empty")
    windows: list[dict[str, Any]] = []
    start = 0
    size = 0
    section = inventory[0].get("section_ordinal")
    for index, record in enumerate(inventory):
        record_size = len(json.dumps(record, ensure_ascii=False, sort_keys=True))
        changed_section = index > start and record.get("section_ordinal") != section
        exceeds_cap = index > start and (
            index - start >= max_blocks or size + record_size > max_projected_chars
        )
        if changed_section or exceeds_cap:
            windows.append(_window_record(inventory, start, index, len(windows) + 1))
            start = index
            size = 0
            section = record.get("section_ordinal")
        size += record_size
    windows.append(_window_record(inventory, start, len(inventory), len(windows) + 1))
    return windows


def construct_segments_from_cuts(
    cuts: list[int],
    document: dict[str, Any],
    *,
    inventory: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Construct complete source ranges; model cuts are internal 1-based ordinals."""
    blocks = document.get("blocks") or []
    inventory = inventory or build_block_inventory(document)
    total = len(blocks)
    if total == 0 or len(inventory) != total:
        raise SegmentationError("block inventory does not match the source document")
    invalid = [value for value in cuts if isinstance(value, bool) or not isinstance(value, int)]
    if invalid:
        raise SegmentationError(f"cut ordinals must be integers: {invalid[:5]}")
    if len(cuts) != len(set(cuts)):
        duplicates = sorted({value for value in cuts if cuts.count(value) > 1})
        raise SegmentationError(f"cut ordinals must be unique; duplicates: {duplicates[:8]}")
    outside = sorted({value for value in cuts if value < 1 or value >= total})
    if outside:
        raise SegmentationError(
            f"cut ordinals must lie between 1 and {total - 1}: {outside[:8]}"
        )
    canonical_cuts = sorted(cuts) + [total]
    segments: list[dict[str, Any]] = []
    start = 1
    for index, end in enumerate(canonical_cuts, start=1):
        selected = blocks[start - 1:end]
        if not selected:
            raise SegmentationError(f"cut after ordinal {end} creates an empty segment")
        selected_inventory = inventory[start - 1:end]
        segments.append({
            "segment_id": f"seg-{index:04d}",
            "title": _segment_title(selected_inventory, start=start, end=end),
            "start_block_id": block_id(selected[0]),
            "end_block_id": block_id(selected[-1]),
            "block_ids": [block_id(block) for block in selected],
        })
        start = end + 1
    validate_exact_coverage(segments, blocks)
    return segments


def validate_exact_coverage(
    segments: list[dict[str, Any]], blocks: list[dict[str, Any]]
) -> None:
    expected = [block_id(block) for block in blocks]
    actual: list[str] = []
    seen_segment_ids: set[str] = set()
    for segment in segments:
        segment_id = str(segment.get("segment_id") or "")
        block_ids = [str(value) for value in segment.get("block_ids") or []]
        if not segment_id or segment_id in seen_segment_ids:
            raise SegmentationError(f"invalid or duplicate segment id: {segment_id}")
        if not block_ids:
            raise SegmentationError(f"segment {segment_id} contains no source blocks")
        if str(segment.get("start_block_id") or "") != block_ids[0]:
            raise SegmentationError(f"segment {segment_id} has an inconsistent start block")
        if str(segment.get("end_block_id") or "") != block_ids[-1]:
            raise SegmentationError(f"segment {segment_id} has an inconsistent end block")
        actual.extend(block_ids)
        seen_segment_ids.add(segment_id)
    if actual != expected:
        mismatch = next(
            (index for index, pair in enumerate(zip(actual, expected), start=1) if pair[0] != pair[1]),
            min(len(actual), len(expected)) + 1,
        )
        raise SegmentationError(
            f"final segmentation does not cover source blocks exactly once in order; "
            f"first mismatch at ordinal {mismatch}"
        )


def segment_document(
    document: dict[str, Any],
    *,
    checkpoint_dir: Path,
    workers: int,
    force: bool,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
    seed_cuts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run bounded medium calls and return validated downstream segment records."""
    inventory = build_block_inventory(document)
    windows = build_segmentation_windows(inventory)
    inventory_hash = sha256_json(inventory)
    windows_hash = sha256_json(windows)
    final_path = checkpoint_dir / "segmentation.json"
    if seed_cuts is not None:
        canonical = (
            not any(isinstance(value, bool) or not isinstance(value, int) for value in seed_cuts)
            and seed_cuts == sorted(set(seed_cuts))
        )
        if not canonical:
            raise SegmentationError("migrated segmentation cuts are not canonical")
        segments = construct_segments_from_cuts(seed_cuts, document, inventory=inventory)
        if _oversized_segments(segments, document):
            raise SegmentationError("migrated segmentation exceeds current hard limits")
        write_json(final_path, {
            "schema_version": SEGMENTATION_VERSION,
            "inventory_sha256": inventory_hash,
            "windows_sha256": windows_hash,
            "cuts": list(seed_cuts),
            "segments": segments,
            "refinements": [],
            "migration": {
                "kind": "validated_legacy_cuts",
                "provider_calls": 0,
            },
        })
        return segments
    if final_path.is_file() and not force:
        cached = _read_optional_json(final_path)
        if _valid_final_envelope(cached, inventory_hash=inventory_hash, windows_hash=windows_hash):
            cached_cuts = cached.get("cuts") or []
            canonical = (
                isinstance(cached_cuts, list)
                and not any(isinstance(value, bool) or not isinstance(value, int) for value in cached_cuts)
                and cached_cuts == sorted(set(cached_cuts))
            )
            if canonical:
                try:
                    reconstructed = construct_segments_from_cuts(cached_cuts, document, inventory=inventory)
                    if reconstructed == cached.get("segments") and not _oversized_segments(reconstructed, document):
                        return reconstructed
                except SegmentationError:
                    pass

    cuts: set[int] = set()
    segmentation_dir = checkpoint_dir / "segmentation"

    def process_window(window: dict[str, Any]) -> list[int]:
        owned_count = len(window["owned_blocks"])
        deterministic_end = (
            [int(window["end_ordinal"])]
            if int(window["end_ordinal"]) < len(inventory)
            else []
        )
        window_hash = sha256_json(window)
        path = segmentation_dir / "windows" / f"{window['window_id']}-{window_hash}.json"
        if path.is_file() and not force:
            cached = _read_optional_json(path)
            if _valid_window_envelope(cached, inventory_hash=inventory_hash, window_hash=window_hash):
                try:
                    model_cuts = _validate_window_cuts(cached.get("cuts") or [], window, len(inventory))
                    return sorted(set(model_cuts + deterministic_end))
                except SegmentationError as exc:
                    write_json(
                        segmentation_dir / "attempts" / str(window["window_id"]) / "rejected-cache.json",
                        {
                            "schema_version": SEGMENTATION_VERSION,
                            "accepted": False,
                            "error": str(exc),
                            "inventory_sha256": inventory_hash,
                            "window_sha256": window_hash,
                        },
                    )
        if owned_count == 1:
            write_json(path, {
                "schema_version": SEGMENTATION_VERSION,
                "inventory_sha256": inventory_hash,
                "window_sha256": window_hash,
                "cuts": [],
                "deterministic_end_cut": deterministic_end,
            })
            return deterministic_end
        model_cuts = _call_for_cuts(
            inventory=inventory,
            window=window,
            inventory_hash=inventory_hash,
            window_hash=window_hash,
            checkpoint_path=path,
            attempts_dir=segmentation_dir / "attempts" / str(window["window_id"]),
            call_model=call_model,
            label_prefix=f"companion-segmentation-{window['window_id']}",
            refinement=False,
            reuse_attempts=not force,
        )
        return sorted(set(model_cuts + deterministic_end))

    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(windows))) as executor:
        futures = {executor.submit(process_window, window): window for window in windows}
        try:
            for future in as_completed(futures):
                cuts.update(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    refinement_records: list[dict[str, Any]] = []
    for round_number in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        segments = construct_segments_from_cuts(sorted(cuts), document, inventory=inventory)
        oversized = _oversized_segments(segments, document)
        if not oversized:
            break

        def refine(item: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
            start = item["start_ordinal"] - 1
            end = item["end_ordinal"]
            window = _window_record(inventory, start, end, item["segment_id"])
            window_hash = sha256_json({"round": round_number, "window": window})
            checkpoint_path = (
                segmentation_dir / "refinements" / f"round-{round_number}" /
                f"{item['segment_id']}-{window_hash}.json"
            )
            attempts_dir = (
                segmentation_dir / "attempts" / "refinements" /
                f"round-{round_number}-{item['segment_id']}"
            )
            if checkpoint_path.is_file() and not force:
                cached = _read_optional_json(checkpoint_path)
                if _valid_window_envelope(
                    cached, inventory_hash=inventory_hash, window_hash=window_hash
                ):
                    try:
                        cached_cuts = _validate_window_cuts(
                            cached.get("cuts") or [], window, len(inventory)
                        )
                        if not cached_cuts:
                            raise SegmentationError(
                                f"refinement window {window['window_id']} has no internal cuts"
                            )
                        return item, cached_cuts
                    except SegmentationError as exc:
                        write_json(attempts_dir / "rejected-cache.json", {
                            "schema_version": SEGMENTATION_VERSION,
                            "accepted": False,
                            "error": str(exc),
                            "inventory_sha256": inventory_hash,
                            "window_sha256": window_hash,
                        })
            result = _call_for_cuts(
                inventory=inventory,
                window=window,
                inventory_hash=inventory_hash,
                window_hash=window_hash,
                checkpoint_path=checkpoint_path,
                attempts_dir=attempts_dir,
                call_model=call_model,
                label_prefix=f"companion-segmentation-refine-{round_number}-{item['segment_id']}",
                refinement=True,
                refinement_round=round_number,
                reuse_attempts=not force,
            )
            return item, result

        added = 0
        with ThreadPoolExecutor(max_workers=min(max(1, workers), len(oversized))) as executor:
            futures = {executor.submit(refine, item): item for item in oversized}
            for future in as_completed(futures):
                item, new_cuts = future.result()
                before = len(cuts)
                cuts.update(new_cuts)
                added += len(cuts) - before
                refinement_records.append({
                    "round": round_number,
                    "segment_id": item["segment_id"],
                    "cuts": sorted(new_cuts),
                })
        if added == 0:
            raise SegmentationError(
                f"semantic refinement round {round_number} added no cuts to oversized intervals",
                context={
                    "phase": "refinement",
                    "round": round_number,
                    "intervals": [item["segment_id"] for item in oversized],
                },
            )
    else:
        segments = construct_segments_from_cuts(sorted(cuts), document, inventory=inventory)

    oversized = _oversized_segments(segments, document)
    if oversized:
        interval_ranges = [
            f"{item['segment_id']}[{item['start_ordinal']}..{item['end_ordinal']}]"
            for item in oversized[:8]
        ]
        descriptions = [
            (
                f"{item['segment_id']}[{item['start_ordinal']}..{item['end_ordinal']}; "
                f"blocks={item['block_count']}; "
                f"prompt_projection_chars={item['prompt_projection_chars']}]"
            )
            for item in oversized[:8]
        ]
        raise SegmentationError(
            "semantic segmentation remains above the hard size limit after "
            f"{MAX_VALIDATION_ATTEMPTS} refinement rounds: {', '.join(descriptions)}",
            context={
                "phase": "refinement",
                "round": MAX_VALIDATION_ATTEMPTS,
                "intervals": interval_ranges,
            },
        )
    validate_exact_coverage(segments, document.get("blocks") or [])
    envelope = {
        "schema_version": SEGMENTATION_VERSION,
        "inventory_sha256": inventory_hash,
        "windows_sha256": windows_hash,
        "cuts": sorted(cuts),
        "segments": segments,
        "refinements": refinement_records,
    }
    write_json(final_path, envelope)
    return segments


def _call_for_cuts(
    *,
    inventory: list[dict[str, Any]],
    window: dict[str, Any],
    inventory_hash: str,
    window_hash: str,
    checkpoint_path: Path,
    attempts_dir: Path,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
    label_prefix: str,
    refinement: bool,
    refinement_round: int | None = None,
    reuse_attempts: bool = True,
) -> list[int]:
    last_error = "unknown validation failure"
    completed_attempts = 0
    attempt_paths = attempts_dir.glob("attempt-*.json") if reuse_attempts else ()
    for attempt_path in sorted(attempt_paths, key=_attempt_number):
        record = _read_optional_json(attempt_path)
        if not isinstance(record, dict):
            continue
        if (
            record.get("schema_version") != SEGMENTATION_VERSION
            or record.get("inventory_sha256") != inventory_hash
            or record.get("window_sha256") != window_hash
        ):
            continue
        attempt_number = _attempt_number(attempt_path)
        if attempt_number < 1 or attempt_number > MAX_VALIDATION_ATTEMPTS:
            continue
        completed_attempts = max(completed_attempts, attempt_number)
        response = record.get("response")
        if record.get("accepted") is True and isinstance(response, dict):
            try:
                cuts = _validate_window_cuts(
                    response.get("cut_after_ordinals"), window, len(inventory)
                )
                if refinement and not cuts:
                    raise SegmentationError(
                        f"refinement window {window['window_id']} must add at least one internal cut"
                    )
            except (SegmentationError, KeyError, TypeError, ValueError):
                pass
            else:
                write_json(checkpoint_path, {
                    "schema_version": SEGMENTATION_VERSION,
                    "inventory_sha256": inventory_hash,
                    "window_sha256": window_hash,
                    "cuts": cuts,
                })
                return cuts
        last_error = str(record.get("error") or last_error)

    if completed_attempts >= MAX_VALIDATION_ATTEMPTS:
        raise SegmentationError(
            f"window {window['window_id']} exhausted its lifetime semantic cut "
            f"validation budget of {MAX_VALIDATION_ATTEMPTS} attempts: {last_error}",
            context=_window_failure_context(
                window,
                attempt=completed_attempts,
                refinement=refinement,
                refinement_round=refinement_round,
            ),
        )

    for attempt in range(completed_attempts + 1, MAX_VALIDATION_ATTEMPTS + 1):
        prompt = segmentation_prompt(
            window, total_blocks=len(inventory), refinement=refinement
        )
        if attempt > 1:
            prompt += (
                "\n\nCORRECTION REQUIRED: The previous response was rejected by the "
                f"deterministic validator: {last_error}. Return a corrected cut list for "
                "the same unchanged source inventory and ownership interval."
            )
        # Provider, transport, quota, and cancellation failures are owned by
        # arc-llm. They must escape immediately rather than being multiplied by
        # this semantic-correction loop.
        response: Any = call_model(
            prompt,
            CUT_SCHEMA,
            attempts_dir / f"attempt-{attempt}" / "llm",
            f"{label_prefix}-attempt-{attempt}",
        )
        try:
            if not isinstance(response, dict):
                raise SegmentationError("segmentation response must be an object")
            if "cut_after_ordinals" not in response:
                raise SegmentationError("segmentation response is missing cut_after_ordinals")
            cuts = _validate_window_cuts(
                response["cut_after_ordinals"], window, len(inventory)
            )
            if refinement and not cuts:
                raise SegmentationError(
                    f"refinement window {window['window_id']} must add at least one internal cut"
                )
            write_json(attempts_dir / f"attempt-{attempt}.json", {
                "schema_version": SEGMENTATION_VERSION,
                "accepted": True,
                "response": response,
                "inventory_sha256": inventory_hash,
                "window_sha256": window_hash,
            })
            write_json(checkpoint_path, {
                "schema_version": SEGMENTATION_VERSION,
                "inventory_sha256": inventory_hash,
                "window_sha256": window_hash,
                "cuts": cuts,
            })
            return cuts
        except (SegmentationError, KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)
            write_json(attempts_dir / f"attempt-{attempt}.json", {
                "schema_version": SEGMENTATION_VERSION,
                "accepted": False,
                "response": response,
                "error": last_error,
                "inventory_sha256": inventory_hash,
                "window_sha256": window_hash,
            })
    raise SegmentationError(
        f"window {window['window_id']} failed semantic cut validation after "
        f"{MAX_VALIDATION_ATTEMPTS} attempts: {last_error}",
        context=_window_failure_context(
            window,
            attempt=MAX_VALIDATION_ATTEMPTS,
            refinement=refinement,
            refinement_round=refinement_round,
        ),
    )


def _attempt_number(path: Path) -> int:
    match = re.fullmatch(r"attempt-(\d+)\.json", path.name)
    return int(match.group(1)) if match else 0


def _validate_window_cuts(
    values: Any, window: dict[str, Any], total_blocks: int
) -> list[int]:
    if not isinstance(values, list):
        raise SegmentationError("cut_after_ordinals must be an array")
    invalid = [value for value in values if isinstance(value, bool) or not isinstance(value, int)]
    if invalid:
        raise SegmentationError(f"cut_after_ordinals contains non-integers: {invalid[:5]}")
    if len(values) != len(set(values)):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        raise SegmentationError(
            f"window {window['window_id']} returned duplicate cut ordinals: {duplicates[:8]}"
        )
    first = int(window["start_ordinal"])
    last = int(window["end_ordinal"])
    last_internal = min(last - 1, total_blocks - 1)
    outside = sorted({
        value for value in values
        if value < first or value > last_internal
    })
    if outside:
        raise SegmentationError(
            f"window {window['window_id']} may cut only after owned ordinals "
            f"{first}..{last_internal}; got {outside[:8]}"
        )
    return sorted(values)


def _oversized_segments(
    segments: list[dict[str, Any]], document: dict[str, Any]
) -> list[dict[str, Any]]:
    blocks = document.get("blocks") or []
    positions = {block_id(block): index + 1 for index, block in enumerate(blocks)}
    oversized: list[dict[str, Any]] = []
    for segment in segments:
        start = positions[str(segment["start_block_id"])]
        end = positions[str(segment["end_block_id"])]
        selected = blocks[start - 1:end]
        annotation_projection = [
            annotation_input_block(block, document) for block in selected
        ]
        translation_projection = [
            translation_input_block(block) for block in selected if is_translatable(block)
        ]
        prompt_projection_chars = max(
            _serialized_chars(annotation_projection),
            _serialized_chars(translation_projection),
        )
        if len(selected) > 1 and (
            len(selected) > SEGMENT_HARD_MAX_BLOCKS
            or prompt_projection_chars > SEGMENT_HARD_MAX_SOURCE_CHARS
        ):
            oversized.append({
                "segment_id": str(segment["segment_id"]),
                "start_ordinal": start,
                "end_ordinal": end,
                "block_count": len(selected),
                "prompt_projection_chars": prompt_projection_chars,
            })
    return oversized


def _serialized_chars(value: Any) -> int:
    return len(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ))


def _window_record(
    inventory: list[dict[str, Any]], start: int, end: int, identifier: Any
) -> dict[str, Any]:
    owned = inventory[start:end]
    if isinstance(identifier, int):
        window_id = f"w-{identifier:04d}"
    else:
        window_id = f"refine-{identifier}"
    return {
        "window_id": window_id,
        "start_ordinal": int(owned[0]["ordinal"]),
        "end_ordinal": int(owned[-1]["ordinal"]),
        "owned_blocks": owned,
        "context_before": inventory[start - 1:start] if start else [],
        "context_after": inventory[end:end + 1],
    }


def _window_failure_context(
    window: dict[str, Any],
    *,
    attempt: int,
    refinement: bool,
    refinement_round: int | None = None,
) -> dict[str, Any]:
    owned = window.get("owned_blocks") or []
    section_ordinals = sorted({
        int(item["section_ordinal"])
        for item in owned
        if isinstance(item, dict) and item.get("section_ordinal") is not None
    })
    section_titles = list(dict.fromkeys(
        str(item.get("section_title") or "").strip()
        for item in owned
        if isinstance(item, dict) and str(item.get("section_title") or "").strip()
    ))
    context: dict[str, Any] = {
        "phase": "refinement" if refinement else "window",
        "window_id": str(window["window_id"]),
        "start_ordinal": int(window["start_ordinal"]),
        "end_ordinal": int(window["end_ordinal"]),
        "attempt": attempt,
        "refinement": refinement,
    }
    if section_ordinals:
        context["section_ordinals"] = section_ordinals
    if section_titles:
        context["section_titles"] = section_titles
    if refinement and refinement_round is not None:
        context["round"] = refinement_round
    return context


def _valid_window_envelope(
    value: Any, *, inventory_hash: str, window_hash: str
) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("schema_version") == SEGMENTATION_VERSION
        and value.get("inventory_sha256") == inventory_hash
        and value.get("window_sha256") == window_hash
        and isinstance(value.get("cuts"), list)
    )


def _valid_final_envelope(
    value: Any, *, inventory_hash: str, windows_hash: str
) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("schema_version") == SEGMENTATION_VERSION
        and value.get("inventory_sha256") == inventory_hash
        and value.get("windows_sha256") == windows_hash
        and isinstance(value.get("segments"), list)
    )


def _read_optional_json(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _segment_title(
    records: list[dict[str, Any]], *, start: int, end: int
) -> str:
    heading = next((str(item.get("title") or "").strip() for item in records if item.get("title")), "")
    if heading:
        return heading
    section_title = next(
        (str(item.get("section_title") or "").strip() for item in records if item.get("section_title")),
        "",
    )
    if section_title:
        return f"{section_title} · blocks {start}–{end}"
    section_ordinal = next((item.get("section_ordinal") for item in records if item.get("section_ordinal")), None)
    if section_ordinal:
        return f"Section {section_ordinal} · blocks {start}–{end}"
    return f"Blocks {start}–{end}"


def _entity_index(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in items:
        for key in ("id", "block_id", "equation_id", "figure_id", "table_id", "bib_id"):
            if item.get(key):
                output[str(item[key])] = item
    return output


def _entity_for_block(
    block: dict[str, Any], entities: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    for key in ("entity_id", "source_id", "equation_id", "figure_id", "table_id", "id", "block_id"):
        value = block.get(key)
        if value is not None and str(value) in entities:
            return entities[str(value)]
    return None


def _preview(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"
