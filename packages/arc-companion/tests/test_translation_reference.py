from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from arc_companion.paper_broker import (
    PaperBroker,
    PaperBrokerError,
    build_paper_broker_policy,
)
from arc_companion.translation_reference import (
    TranslationReferenceError,
    align_translation_chapters,
    leading_decimal_ordinal,
    resolve_translation_reference,
    validate_translation_reference_bundle,
    validate_translation_reference_provenance,
)
from arc_companion.io import sha256_json


def _structure(
    titles: tuple[str, ...] = ("1 First", "2 Second"),
    *,
    prefix: str,
    document_kind: str = "book",
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chapters = []
    sections = []
    for index, title in enumerate(titles):
        chapter_id = f"{prefix}-chapter-{index + 1}"
        section_id = f"{prefix}-section-{index + 1}"
        chapters.append({
            "chapter_id": chapter_id,
            "title": title,
            "level": 1,
            "leading_decimal_ordinal": leading_decimal_ordinal(title),
            "section_ids": [section_id],
        })
        sections.append({
            "section_id": section_id,
            "title": title,
            "level": 1,
            "ordinal": index,
            "section_payload_sha256": "a" * 64,
        })
    default_coverage = {
        "status": "complete",
        "expected_count": len(sections),
        "covered_count": len(sections),
        "duplicates": [],
        "missing": [],
        "unexpected": [],
        "monotonic_order": True,
    }
    default_coverage.update(coverage or {})
    return {
        "schema_version": "arc.paper.parsed-structure-view.v1",
        "requested_source_id": prefix,
        "canonical_source_id": prefix,
        "parser_version": "arc.paper.rich-document.v1",
        "source_hash": "b" * 64,
        "document_hash": "c" * 64,
        "structure_schema_version": "arc.paper.structure.v1",
        "requested_document_kind": document_kind,
        "document_kind": document_kind,
        "structure_source": "rich_source_headings",
        "chapters": chapters,
        "sections": sections,
        "coverage": default_coverage,
    }


def _chapters_pack(*chapter_ids: str) -> dict[str, Any]:
    return {
        "schema_version": "arc.companion.chapters.v2",
        "chapters": [
            {"chapter_id": chapter_id, "block_ids": [f"block-{index + 1}"]}
            for index, chapter_id in enumerate(chapter_ids)
        ],
    }


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("1", 1),
        ("\u20031. Introduction", 1),
        ("2：方法", 2),
        ("3. 2-adic examples", 3),
        ("0 Zero", None),
        ("01 Leading", None),
        ("+1 Signed", None),
        ("-1 Signed", None),
        ("1.2 Decimal", None),
        ("1/2 Fraction", None),
        ("IV Roman", None),
        ("one Word", None),
        ("Chapter 1 Embedded", None),
        ("Heading 1", None),
        ("１ Fullwidth", None),
        ("1a Joined", None),
    ],
)
def test_leading_decimal_ordinal_is_strict(
    heading: str, expected: int | None,
) -> None:
    assert leading_decimal_ordinal(heading) == expected


def test_explicit_alignment_uses_exact_pack_universe_and_reference_order() -> None:
    primary = _structure(prefix="source")
    reference = _structure(
        ("Front", "1 First", "2 Second", "Back"),
        prefix="reference",
    )
    pack = _chapters_pack("source-chapter-1", "source-chapter-2")

    aligned = align_translation_chapters(
        chapters_pack=pack,
        primary_structure=primary,
        reference_structure=reference,
        explicit_mappings=(
            " source-chapter-1 = reference-chapter-2 ",
            "source-chapter-2=reference-chapter-3",
        ),
    )

    assert aligned["method"] == "explicit"
    assert aligned["pairs"] == [
        {
            "source_chapter_id": "source-chapter-1",
            "reference_chapter_id": "reference-chapter-2",
        },
        {
            "source_chapter_id": "source-chapter-2",
            "reference_chapter_id": "reference-chapter-3",
        },
    ]


@pytest.mark.parametrize(
    "mappings",
    [
        ("source-chapter-1=reference-chapter-2",),
        (
            "source-chapter-1=reference-chapter-2",
            "source-chapter-1=reference-chapter-3",
        ),
        (
            "source-chapter-1=reference-chapter-2",
            "source-chapter-2=reference-chapter-2",
        ),
        (
            "source-chapter-1=missing",
            "source-chapter-2=reference-chapter-3",
        ),
        (
            "source-chapter-1=reference-chapter-3",
            "source-chapter-2=reference-chapter-2",
        ),
        ("source-chapter-1", "source-chapter-2=reference-chapter-3"),
    ],
)
def test_explicit_alignment_rejects_incomplete_duplicate_or_unordered_mapping(
    mappings: tuple[str, ...],
) -> None:
    with pytest.raises(TranslationReferenceError) as raised:
        align_translation_chapters(
            chapters_pack=_chapters_pack(
                "source-chapter-1", "source-chapter-2",
            ),
            primary_structure=_structure(prefix="source"),
            reference_structure=_structure(
                ("Front", "1 First", "2 Second"),
                prefix="reference",
            ),
            explicit_mappings=mappings,
        )
    assert raised.value.code == "translation_reference_mapping_invalid"


def test_automatic_alignment_accepts_exact_complete_numbered_structures() -> None:
    primary = _structure(prefix="source")
    reference = _structure(prefix="reference")

    aligned = align_translation_chapters(
        chapters_pack=_chapters_pack(
            "source-chapter-1", "source-chapter-2",
        ),
        primary_structure=primary,
        reference_structure=reference,
    )

    assert aligned["method"] == "leading-decimal-ordinal"
    assert [item["reference_chapter_id"] for item in aligned["pairs"]] == [
        "reference-chapter-1", "reference-chapter-2",
    ]


def test_resolver_preserves_only_broker_proven_transient_errors(
    tmp_path: Path,
) -> None:
    class FailingBroker:
        def __init__(self, *, retryable: bool) -> None:
            self.retryable = retryable

        def resolve_round(self, _requests, *, round_number: int):
            del round_number
            raise PaperBrokerError(
                "paper_transport_failed",
                "temporary" if self.retryable else "local",
                retryable=self.retryable,
            )

    for retryable, expected_code in (
        (True, "paper_transport_failed"),
        (False, "translation_reference_source_unavailable"),
    ):
        with pytest.raises(TranslationReferenceError) as raised:
            resolve_translation_reference(
                project_dir=tmp_path,
                checkpoint_dir=None,
                primary_parsed={},
                primary_document={},
                chapters_pack={},
                requested_reference_id="reference",
                broker=FailingBroker(retryable=retryable),
            )
        assert raised.value.code == expected_code
        assert raised.value.retryable is retryable


@pytest.mark.parametrize(
    "titles",
    [
        ("0 Zero", "2 Second"),
        ("01 Leading", "2 Second"),
        ("+1 Signed", "2 Second"),
        ("1.2 Decimal", "2 Second"),
        ("1/2 Fraction", "2 Second"),
        ("I Roman", "2 Second"),
        ("one Word", "2 Second"),
        ("Chapter 1", "2 Second"),
        ("Heading", "2"),
        ("1 First", "1 Duplicate"),
        ("1 First", "3 Gap"),
        ("2 Non-one", "3 Third"),
    ],
)
def test_automatic_alignment_rejects_noncanonical_ordinals(
    titles: tuple[str, ...],
) -> None:
    with pytest.raises(TranslationReferenceError) as raised:
        align_translation_chapters(
            chapters_pack=_chapters_pack(
                "source-chapter-1", "source-chapter-2",
            ),
            primary_structure=_structure(prefix="source"),
            reference_structure=_structure(titles, prefix="reference"),
        )
    assert raised.value.code == "translation_reference_alignment_ambiguous"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["coverage"].update(status="invalid"),
        lambda value: value["coverage"]["missing"].append("reference-section-2"),
        lambda value: value["coverage"].update(monotonic_order=False),
        lambda value: value["chapters"][1].update(
            section_ids=["reference-section-1"]
        ),
        lambda value: value.update(document_kind="article"),
    ],
)
def test_automatic_alignment_rejects_coverage_boundary_and_kind_ambiguity(
    mutator,
) -> None:
    reference = _structure(prefix="reference")
    mutator(reference)
    with pytest.raises(TranslationReferenceError) as raised:
        align_translation_chapters(
            chapters_pack=_chapters_pack(
                "source-chapter-1", "source-chapter-2",
            ),
            primary_structure=_structure(prefix="source"),
            reference_structure=reference,
        )
    assert raised.value.code == "translation_reference_alignment_ambiguous"


class _ReferenceBroker:
    def __init__(
        self,
        aggregate_broker: PaperBroker,
        structure: dict[str, Any],
        section_payloads: dict[str, dict[str, Any]],
    ) -> None:
        self.aggregate_broker = aggregate_broker
        self.structure = structure
        self.section_payloads = section_payloads
        self.section_calls = 0

    def resolve_round(self, requests, *, round_number: int):
        del round_number
        request = requests[0]
        if request.operation == "get-parsed-identity":
            data = {
                "paper_id": self.structure["canonical_source_id"],
                "source_hash": self.structure["source_hash"],
                "document_hash": self.structure["document_hash"],
            }
        elif request.operation == "get-parsed-structure":
            data = self.structure
        elif request.operation == "get-parsed-section":
            self.section_calls += 1
            data = self.section_payloads[request.arguments["section"]]
        else:
            raise AssertionError(request.operation)
        return (
            SimpleNamespace(
                ok=True,
                data={"ok": True, "data": data},
                error=None,
                provenance={},
            ),
        )

    def store_controller_aggregate_json(self, **kwargs):
        return self.aggregate_broker.store_controller_aggregate_json(**kwargs)

    def load_controller_aggregate_json(self, **kwargs):
        return self.aggregate_broker.load_controller_aggregate_json(**kwargs)


def _aggregate_broker(
    root: Path, *, run_id: str, checkpoint_name: str,
) -> PaperBroker:
    return PaperBroker(
        checkpoint_root=root / checkpoint_name,
        base_cache_root=root / "cache",
        policy=build_paper_broker_policy(access="none"),
        run_id=run_id,
        generic_internet_allowed=False,
        controller_project_root=root / "project",
    )


def _resolver_inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any],
]:
    parsed = {
        "paper_id": "primary",
        "parser_version": "arc.paper.rich-document.v1",
        "source_hash": "1" * 64,
        "document_hash": "2" * 64,
        "sections": [
            {"section_id": "source-section-1", "title": "1 First", "level": 1},
            {"section_id": "source-section-2", "title": "2 Second", "level": 1},
        ],
        "structure": {
            "schema_version": "arc.paper.structure.v1",
            "requested_document_kind": "book",
            "document_kind": "book",
            "structure_source": "rich_source_headings",
            "chapters": [
                {
                    "chapter_id": "source-chapter-1",
                    "title": "1 First",
                    "level": 1,
                    "section_ids": ["source-section-1"],
                },
                {
                    "chapter_id": "source-chapter-2",
                    "title": "2 Second",
                    "level": 1,
                    "section_ids": ["source-section-2"],
                },
            ],
            "coverage": {
                "status": "complete",
                "expected_count": 2,
                "covered_count": 2,
                "duplicates": [],
                "missing": [],
                "unexpected": [],
                "monotonic_order": True,
            },
        },
    }
    document = {
        "blocks": [
            {"block_id": "block-1", "text": "Primary one."},
            {"block_id": "block-2", "text": "Primary two."},
        ],
    }
    return (
        parsed,
        document,
        _chapters_pack("source-chapter-1", "source-chapter-2"),
    )


def _reference_structure(
    payloads: dict[str, dict[str, Any]],
    *,
    source_hash: str,
) -> dict[str, Any]:
    structure = _structure(prefix="reference")
    structure["source_hash"] = source_hash
    structure["document_hash"] = source_hash
    for section in structure["sections"]:
        section["section_payload_sha256"] = sha256_json(
            payloads[section["section_id"]]
        )
    return structure


def test_resolver_reuses_aggregates_and_validates_compact_provenance(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    primary, document, pack = _resolver_inputs()
    payloads = {
        "reference-section-1": {
            "section_id": "reference-section-1",
            "text": "Reference one.",
        },
        "reference-section-2": {
            "section_id": "reference-section-2",
            "text": "Reference two.",
        },
    }
    first_broker = _ReferenceBroker(
        _aggregate_broker(
            tmp_path, run_id="first", checkpoint_name="checkpoint-first",
        ),
        _reference_structure(payloads, source_hash="3" * 64),
        payloads,
    )
    first = resolve_translation_reference(
        project_dir=project,
        checkpoint_dir=None,
        primary_parsed=primary,
        primary_document=document,
        chapters_pack=pack,
        requested_reference_id="reference",
        broker=first_broker,
    )
    assert first is not None
    assert first_broker.section_calls == 2
    assert "text" not in repr(first.compact_provenance)
    validate_translation_reference_bundle(first)

    second_broker = _ReferenceBroker(
        _aggregate_broker(
            tmp_path, run_id="second", checkpoint_name="checkpoint-second",
        ),
        _reference_structure(payloads, source_hash="4" * 64),
        payloads,
    )
    second = resolve_translation_reference(
        project_dir=project,
        checkpoint_dir=None,
        primary_parsed=primary,
        primary_document=document,
        chapters_pack=pack,
        requested_reference_id="reference",
        broker=second_broker,
    )
    assert second is not None
    assert second_broker.section_calls == 0
    assert second.manifest_sha256 != first.manifest_sha256
    assert (
        second.chapter("source-chapter-1").semantic_identity()
        == first.chapter("source-chapter-1").semantic_identity()
    )

    body_bearing = deepcopy(second.compact_provenance)
    body_bearing["mappings"][0]["source_chapter_id"] = {"body": "forbidden"}
    with pytest.raises(TranslationReferenceError) as body_error:
        validate_translation_reference_provenance(
            body_bearing,
            project_root=project,
        )
    assert body_error.value.code == "translation_reference_manifest_invalid"

    wrong_path = deepcopy(second.compact_provenance)
    wrong_path["manifest_path"] = str(second.manifest_path)
    with pytest.raises(TranslationReferenceError) as path_error:
        validate_translation_reference_provenance(
            wrong_path,
            project_root=project,
        )
    assert path_error.value.code == "translation_reference_manifest_invalid"

    rebinding_paths = list((
        project
        / ".arc-companion/paper-broker/controller-objects/"
        "translation-reference/rebindings"
    ).glob("*.json"))
    rebinding_path = next(
        path
        for path in rebinding_paths
        if (
            json.loads(path.read_text(encoding="utf-8")).get(
                "current_manifest_sha256"
            )
            == second.manifest_sha256
            and json.loads(path.read_text(encoding="utf-8"))[
                "chapter_semantic_identity"
            ]["source_chapter_id"]
            == "source-chapter-1"
        )
    )
    rebinding_bytes = rebinding_path.read_bytes()
    rebinding_path.write_text("{}", encoding="utf-8")
    with pytest.raises(TranslationReferenceError) as rebinding_tamper:
        validate_translation_reference_bundle(second)
    assert rebinding_tamper.value.code == "translation_reference_artifact_invalid"
    rebinding_path.write_bytes(rebinding_bytes)

    object_path = project / (
        ".arc-companion/paper-broker/controller-objects/"
        "translation-reference/objects/"
        f"{second.compact_provenance['mappings'][0]['object_id']}.json"
    )
    object_path.write_text("{}", encoding="utf-8")
    third_broker = _ReferenceBroker(
        _aggregate_broker(
            tmp_path, run_id="third", checkpoint_name="checkpoint-third",
        ),
        _reference_structure(payloads, source_hash="4" * 64),
        payloads,
    )
    with pytest.raises(TranslationReferenceError) as tampered:
        resolve_translation_reference(
            project_dir=project,
            checkpoint_dir=None,
            primary_parsed=primary,
            primary_document=document,
            chapters_pack=pack,
            requested_reference_id="reference",
            broker=third_broker,
        )
    assert tampered.value.code == "translation_reference_artifact_invalid"
    assert third_broker.section_calls == 0
