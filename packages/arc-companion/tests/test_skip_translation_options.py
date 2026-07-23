from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from arc_companion.pipeline import (
    BuildOptions,
    _fingerprint,
    _options_from_recovery,
    _recovery_options,
    _with_effective_source_language,
    _review,
    _verify_frozen_first_chapter_final,
    _verify_frozen_first_chapter_pre_review,
    build_companion,
)
from arc_companion.latex import (
    render_companion_tex as _render_companion_tex,
    validate_tex_fidelity,
)
from arc_companion.source import SourceBundle
from arc_companion.source_credit import normalize_source_credit


def render_companion_tex(document, *args, source_credit=None, metadata=None, **kwargs):
    return _render_companion_tex(
        document,
        *args,
        source_credit=source_credit or normalize_source_credit(document, metadata),
        metadata=metadata,
        **kwargs,
    )


def _bundle() -> SourceBundle:
    document = {
        "schema_version": "arc.paper.document.v2",
        "blocks": [{"block_id": "body", "type": "text", "text": "Source text."}],
        "integrity": {"status": "complete", "document_hash": "skip-translation-fixture"},
    }
    return SourceBundle(
        paper_id="local:skip-translation",
        parsed={"paper_id": "local:skip-translation", "document": document},
        document=document,
        metadata={"title": "Skip translation fixture"},
        references=[],
        citers=[],
    )


def _segment_review_response(prompt: str) -> dict[str, object]:
    marker = "PORTION:\n" if "PORTION:\n" in prompt else "COMPANION:\n"
    payload = json.loads(prompt.split(marker, 1)[1])
    return {
        "reviewed_segment_ids": [
            item["segment"]["segment_id"]
            for item in payload["segments"]
        ],
        "findings": [],
        "patches": [],
    }


def test_skip_translation_defaults_off_without_changing_source_fingerprint(tmp_path: Path) -> None:
    bundle = _bundle()
    evidence = {"records": []}
    translated = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "translated")
    commentary_only = BuildOptions(
        paper_id=bundle.paper_id,
        project_dir=tmp_path / "commentary-only",
        skip_translation=True,
    )

    assert translated.skip_translation is False
    assert commentary_only.skip_translation is True
    assert _fingerprint(bundle, translated, evidence=evidence) == _fingerprint(
        bundle,
        commentary_only,
        evidence=evidence,
    )
    assert _fingerprint(bundle, translated, evidence=evidence) == _fingerprint(
        bundle,
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "different-recovery-budget",
            max_auto_replacements=9,
        ),
        evidence=evidence,
    )


def test_skip_translation_round_trips_through_recovery_options(tmp_path: Path) -> None:
    original = BuildOptions(
        paper_id="local:skip-translation",
        project_dir=tmp_path,
        skip_translation=True,
    )

    serialized = _recovery_options(original)
    recovered = _options_from_recovery(tmp_path, serialized)

    assert serialized["skip_translation"] is True
    assert recovered.skip_translation is True
    assert serialized["recovery_policy"] == "auto"
    assert recovered.recovery_policy == "auto"
    assert serialized["max_auto_replacements"] == 3
    assert recovered.max_auto_replacements == 3


def test_reference_translation_options_normalize_and_round_trip(
    tmp_path: Path,
) -> None:
    options = BuildOptions(
        paper_id="local:primary",
        project_dir=tmp_path,
        reference_translation_id=" local:translated ",
        reference_translation_mappings=(" ch-1 = ref-1 ", "ch-2=ref-2"),
    )
    recovered = _options_from_recovery(tmp_path, _recovery_options(options))
    assert options.reference_translation_id == "local:translated"
    assert options.reference_translation_mappings == (
        "ch-1=ref-1", "ch-2=ref-2",
    )
    assert recovered.reference_translation_id == "local:translated"
    assert recovered.reference_translation_mappings == (
        "ch-1=ref-1", "ch-2=ref-2",
    )


def test_reference_translation_options_reject_invalid_combinations(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires reference_translation_id"):
        BuildOptions(
            paper_id="local:primary",
            project_dir=tmp_path,
            reference_translation_mappings=("ch-1=ref-1",),
        )
    with pytest.raises(ValueError, match="differ from the primary"):
        BuildOptions(
            paper_id="local:primary",
            project_dir=tmp_path,
            reference_translation_id="local:primary",
        )
    with pytest.raises(ValueError, match="skip_translation"):
        BuildOptions(
            paper_id="local:primary",
            project_dir=tmp_path,
            reference_translation_id="local:translated",
            skip_translation=True,
        )


def test_source_language_round_trips_and_old_skip_infers_target(tmp_path: Path) -> None:
    original = BuildOptions(
        paper_id="local:skip-translation", project_dir=tmp_path,
        source_language="JA_jp",
    )
    recovered = _options_from_recovery(tmp_path, _recovery_options(original))
    assert recovered.source_language == "JA_jp"
    inferred = _with_effective_source_language(BuildOptions(
        paper_id="local:skip-translation", project_dir=tmp_path,
        annotation_language="fr-FR", skip_translation=True,
    ))
    assert inferred.source_language == "fr-FR"


def test_skip_translation_rejects_known_different_base_languages(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same base"):
        _with_effective_source_language(BuildOptions(
            paper_id="local:skip-translation", project_dir=tmp_path,
            source_language="en", annotation_language="zh-CN",
            skip_translation=True,
        ))


def test_source_language_falls_back_to_workflow_context(tmp_path: Path) -> None:
    (tmp_path / "context.json").write_text(
        json.dumps({"source_language": "English", "source_base_language": "en"}),
        encoding="utf-8",
    )
    resolved = _with_effective_source_language(BuildOptions(
        paper_id="local:paper", project_dir=tmp_path,
    ))
    assert resolved.source_language == "en"


def test_old_recovery_options_keep_translation_enabled(tmp_path: Path) -> None:
    recovered = _options_from_recovery(
        tmp_path,
        {"paper_id": "local:skip-translation", "workers": 1},
    )

    assert recovered.skip_translation is False
    assert recovered.recovery_policy == "auto"


def test_arc_paper_access_round_trips_and_old_recovery_defaults_full(
    tmp_path: Path,
) -> None:
    original = BuildOptions(
        paper_id="local:skip-translation",
        project_dir=tmp_path,
        arc_paper_access="none",
    )

    serialized = _recovery_options(original)
    recovered = _options_from_recovery(tmp_path, serialized)
    old = _options_from_recovery(
        tmp_path, {"paper_id": "local:skip-translation", "workers": 1},
    )
    legacy = _options_from_recovery(
        tmp_path,
        {
            "paper_id": "local:skip-translation",
            "arc_paper_cli_access": "none",
        },
    )

    assert serialized["arc_paper_access"] == "none"
    assert "arc_paper_cli_access" not in serialized
    assert recovered.arc_paper_access == "none"
    assert old.arc_paper_access == "full"
    assert legacy.arc_paper_access == "none"


def test_arc_paper_recovery_rejects_alias_conflict_and_direct_none(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        _options_from_recovery(
            tmp_path,
            {
                "paper_id": "local:skip-translation",
                "arc_paper_access": "full",
                "arc_paper_cli_access": "none",
            },
        )
    with pytest.raises(ValueError, match="requires arc_paper_access=full"):
        _options_from_recovery(
            tmp_path,
            {
                "paper_id": "local:skip-translation",
                "arc_paper_access": "none",
                "arc_paper_direct_shell": True,
            },
        )


def test_recovery_policy_round_trips_and_validates(tmp_path: Path) -> None:
    original = BuildOptions(
        paper_id="local:skip-translation",
        project_dir=tmp_path,
        recovery_policy="manual",
        max_auto_replacements=5,
        regenerate_segments=("translation:ch-0001.seg-0002",),
    )

    recovered = _options_from_recovery(tmp_path, _recovery_options(original))

    assert recovered.recovery_policy == "manual"
    assert recovered.max_auto_replacements == 5
    assert recovered.regenerate_segments == ("translation:ch-0001.seg-0002",)
    with pytest.raises(ValueError, match="recovery_policy"):
        BuildOptions(
            paper_id="local:skip-translation",
            project_dir=tmp_path,
            recovery_policy="invalid",
        )
    with pytest.raises(ValueError, match="max_auto_replacements"):
        BuildOptions(
            paper_id="local:skip-translation", project_dir=tmp_path,
            max_auto_replacements=0,
        )
    with pytest.raises(ValueError, match="regenerate_segments"):
        BuildOptions(
            paper_id="local:skip-translation", project_dir=tmp_path,
            regenerate_segments=("review:seg-1",),
        )


def test_commentary_only_render_has_no_translation_ids_or_tex_markers(tmp_path: Path) -> None:
    bundle = _bundle()
    segments = [{"segment_id": "seg-0001", "block_ids": ["body"]}]
    annotations = {
        "seg-0001": {
            "explanation": "Commentary remains available.",
            "commentary": "",
            "prior_work": [],
            "later_work": [],
            "evidence_ids": [],
            "key_points": [],
            "source_notes": [],
        }
    }

    tex, manifest = render_companion_tex(
        bundle.document,
        segments,
        annotations,
        translations=None,
        output_dir=tmp_path,
        language="en",
    )

    layers = manifest["companion_layers"]
    assert layers["translation_mode"] is False
    assert layers["provided_translation_segment_ids"] == []
    assert layers["rendered_translation_segment_ids"] == []
    assert layers["translations"] == []
    assert "ARC-TRANSLATION-" not in tex
    assert "arctranslation" not in tex
    assert r"\begin{arctranslation}" not in tex
    assert validate_tex_fidelity(tex, bundle.document, manifest) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("provided", "provided translation segment ids"),
        ("rendered", "rendered translation segment ids"),
        ("audit", "translation audit records"),
        ("marker", "translation markers"),
        ("environment", "arctranslation environment"),
    ],
)
def test_commentary_only_fidelity_rejects_translation_layer_mutations(
    tmp_path: Path, mutation: str, expected: str,
) -> None:
    bundle = _bundle()
    tex, manifest = render_companion_tex(
        bundle.document,
        [{"segment_id": "seg-0001", "block_ids": ["body"]}],
        {"seg-0001": {
            "explanation": "Commentary.", "commentary": "", "prior_work": [],
            "later_work": [], "evidence_ids": [], "key_points": [], "source_notes": [],
        }},
        translations=None,
        output_dir=tmp_path,
        language="en",
    )
    mutated_tex = tex
    mutated_manifest = deepcopy(manifest)
    layers = mutated_manifest["companion_layers"]
    if mutation == "provided":
        layers["provided_translation_segment_ids"] = ["seg-0001"]
    elif mutation == "rendered":
        layers["rendered_translation_segment_ids"] = ["seg-0001"]
    elif mutation == "audit":
        layers["translations"] = [{"segment_id": "seg-0001"}]
    elif mutation == "marker":
        mutated_tex += "\n% ARC-TRANSLATION-BEGIN injected\n"
    else:
        mutated_tex += "\n\\begin{arctranslation}injected\\end{arctranslation}\n"

    errors = validate_tex_fidelity(mutated_tex, bundle.document, mutated_manifest)
    assert any(expected in error for error in errors)


def test_legacy_pipeline_skip_translation_keeps_commentary_and_pdf(tmp_path: Path) -> None:
    bundle = _bundle()
    labels: list[str] = []
    prompts: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        labels.append(label)
        prompts.append(prompt)
        if "translation" in label:
            raise AssertionError(f"translation call was submitted: {label}")
        if label.startswith("companion-segmentation-"):
            return {"cut_after_ordinals": []}
        if label.startswith("companion-glossary-"):
            return {"entries": []}
        if label.startswith("companion-annotation-"):
            return {
                "explanation": "Commentary remains available.",
                "commentary": "",
                "prior_work": [],
                "later_work": [],
                "context_claims": [],
                "evidence_ids": [],
                "key_points": [],
                "source_notes": [],
                "evidence_requests": [],
            }
        if label.startswith("companion-review-segment-"):
            return _segment_review_response(prompt)
        raise AssertionError(label)

    options = BuildOptions(
        paper_id=bundle.paper_id,
        project_dir=tmp_path / "legacy-run",
        annotation_language="en",
        workers=2,
        skip_translation=True,
        regenerate_lanes=("glossary",),
    )
    result = build_companion(
        options,
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["ok"], result
    checkpoint = Path(result["data"]["checkpoint_dir"])
    assert not any("translation" in label for label in labels)
    assert not any("glossary" in label for label in labels)
    assert all("GLOSSARY:\n" not in prompt for prompt in prompts)
    assert any(label.startswith("companion-annotation-") for label in labels)
    assert any(label.startswith("companion-review-segment-") for label in labels)
    assert not list(checkpoint.rglob("translations*"))
    assert not (checkpoint / "llm" / "translation").exists()
    reuse_plan = json.loads((checkpoint / "reuse-plan.json").read_text())
    glossary_entry = next(item for item in reuse_plan["entries"] if item["lane"] == "glossary")
    translation_entry = next(item for item in reuse_plan["entries"] if item["lane"] == "translation")
    review_entry = next(
        item for item in reuse_plan["entries"]
        if item["lane"] == "review"
    )
    assert glossary_entry["status"] == "skipped"
    assert glossary_entry["reason"] == "glossary_disabled_for_same_language_source"
    assert glossary_entry["estimated_provider_calls"] == 0
    assert translation_entry["status"] == "skipped"
    assert review_entry["review_segment_plan"] == {
        "plan_sha256": json.loads(
            (checkpoint / "review-reuse-receipt.json").read_text()
        )["plan_sha256"],
        "counts": {
            "reused": 0,
            "uncovered": 1,
            "invalidated": 0,
            "explicit_regeneration": 0,
        },
        "estimated_calls": 1,
        "ordered_missing_chunks": json.loads(
            (checkpoint / "review-reuse-plan.json").read_text()
        )["ordered_missing_chunks"],
    }
    reader_final = json.loads((checkpoint / "reader-final.json").read_text())
    assert reader_final["final_overrides"]["glossary"] == {}
    assert Path(result["data"]["output_pdf"]).is_file()
    tex = Path(result["data"]["output_tex"]).read_text(encoding="utf-8")
    manifest = json.loads(
        Path(result["data"]["source_manifest_path"]).read_text(encoding="utf-8")
    )
    assert "Commentary remains available" in tex
    assert "ARC-TRANSLATION-" not in tex
    assert manifest["companion_layers"]["translation_mode"] is False

    call_count = len(labels)
    Path(result["data"]["output_pdf"]).unlink()
    resumed = build_companion(
        options,
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert resumed["ok"] and resumed["data"]["status"] == "complete"
    assert resumed["meta"]["resumed"] is False
    assert len(labels) == call_count


def test_commentary_only_review_rejects_translation_patch(tmp_path: Path) -> None:
    bundle = _bundle()
    segment = {"segment_id": "seg-0001", "block_ids": ["body"]}
    annotation = {
        "explanation": "Commentary.",
        "commentary": "",
        "prior_work": [],
        "later_work": [],
        "context_claims": [],
        "evidence_ids": [],
        "key_points": [],
        "source_notes": [],
        "evidence_requests": [],
    }

    def invalid_review(prompt: str, **_kwargs):
        response = _segment_review_response(prompt)
        response["patches"] = [{
            "segment_id": "seg-0001",
            "commentary": None,
            "explanation": "Revised commentary.",
            "prior_work": None,
            "later_work": None,
            "commentary_sources": None,
            "reason": "attempted mixed-layer patch",
            "translation_blocks": [{"block_id": "body", "text": "Forbidden"}],
        }]
        return {
            **response,
        }

    with pytest.raises(
        RuntimeError, match="schema|translation"
    ):
        _review(
            [segment],
            None,
            {"seg-0001": annotation},
            document=bundle.document,
            glossary={"entries": []},
            protected_names=[],
            evidence={"papers": [], "records": []},
            options=BuildOptions(
                paper_id=bundle.paper_id,
                project_dir=tmp_path,
                annotation_language="en",
                skip_translation=True,
            ),
            llm=invalid_review,
            checkpoint_dir=tmp_path / "checkpoint",
        )


def test_large_commentary_only_review_chunks_without_translation_fields(
    tmp_path: Path,
) -> None:
    document = {
        "blocks": [
            {"block_id": "b1", "type": "text", "text": "First source block."},
            {"block_id": "b2", "type": "text", "text": "Second source block."},
        ],
        "integrity": {"status": "complete", "document_hash": "hierarchical-skip"},
    }
    segments = [
        {"segment_id": "s1", "block_ids": ["b1"]},
        {"segment_id": "s2", "block_ids": ["b2"]},
    ]
    annotation = {
        "explanation": "Commentary.", "commentary": "", "prior_work": [],
        "later_work": [], "context_claims": [], "evidence_ids": [],
        "key_points": [], "source_notes": [], "evidence_requests": [],
    }
    prompts: list[str] = []
    labels: list[str] = []

    def review(prompt: str, **kwargs):
        prompts.append(prompt)
        labels.append(str(kwargs["call_label"]))
        return _segment_review_response(prompt)

    translations, reviewed, result = _review(
        segments,
        None,
        {"s1": dict(annotation), "s2": dict(annotation)},
        document=document,
        glossary={"entries": []},
        protected_names=[],
        evidence={"papers": [], "records": []},
        options=BuildOptions(
            paper_id="local:hierarchical-skip",
            project_dir=tmp_path,
            annotation_language="en",
            skip_translation=True,
            review_context_chars=1,
        ),
        llm=review,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    assert translations is None
    assert set(reviewed) == {"s1", "s2"}
    assert result["hierarchical"] is False
    assert len(labels) == 1
    assert all(label.startswith("companion-review-segment-") for label in labels)
    assert all('"translation"' not in prompt for prompt in prompts)
    prompt_audit = result["prompt_budget_audit"]
    assert prompt_audit["routing"]["mode"] == "segment-reuse"
    assert len(prompt_audit["calls"]) == 1
    assert all(
        call["prompt_bytes"] <= prompt_audit["budget"]["strict_limit_bytes"]
        for call in prompt_audit["calls"]
    )


def test_hierarchical_commentary_review_uses_relevant_glossary_and_checkpoints(
    tmp_path: Path,
) -> None:
    document = {
        "blocks": [
            {"block_id": "b1", "type": "text", "text": "Gauge symmetry."},
            {"block_id": "b2", "type": "text", "text": "Vacuum energy."},
        ],
        "integrity": {"status": "complete", "document_hash": "review-cache"},
    }
    segments = [
        {"segment_id": "s1", "block_ids": ["b1"]},
        {"segment_id": "s2", "block_ids": ["b2"]},
    ]
    annotation = {
        "explanation": "Commentary.", "commentary": "", "prior_work": [],
        "later_work": [], "context_claims": [], "evidence_ids": [],
        "key_points": [], "source_notes": [], "evidence_requests": [],
    }
    glossary = {"entries": [
        {"source_term": "gauge symmetry", "target_term": "gauge symmetry",
         "aliases": [], "brief_explanation": "RELEVANT", "first_block_id": "b1"},
        *[
            {"source_term": f"unrelated-{index}", "target_term": f"unused-{index}",
             "aliases": [], "brief_explanation": "UNRELATED-" + "x" * 500,
             "first_block_id": "elsewhere"}
            for index in range(50)
        ],
    ]}
    prompts: list[str] = []

    def review(prompt: str, **_kwargs):
        prompts.append(prompt)
        return _segment_review_response(prompt)

    kwargs = dict(
        segments=segments,
        translations=None,
        annotations={"s1": dict(annotation), "s2": dict(annotation)},
        document=document,
        glossary=glossary,
        protected_names=[],
        evidence={"papers": [], "records": []},
        options=BuildOptions(
            paper_id="local:review-cache", project_dir=tmp_path,
            annotation_language="en", skip_translation=True, review_context_chars=1,
        ),
        checkpoint_dir=tmp_path / "checkpoint",
    )
    _review(llm=review, **kwargs)

    assert len(prompts) == 1
    assert all("GLOSSARY" not in prompt for prompt in prompts)
    assert all("RELEVANT" in prompt for prompt in prompts)
    assert all("UNRELATED-" not in prompt for prompt in prompts)
    checkpoints = sorted(
        (tmp_path / "checkpoint" / "review-segment-responses").glob("*.json")
    )
    assert len(checkpoints) == 1
    assert all(json.loads(path.read_text())["input_sha256"] for path in checkpoints)

    _review(llm=lambda *_args, **_kwargs: pytest.fail("checkpoint was not reused"), **kwargs)


def test_hierarchical_commentary_review_rejects_patch_from_another_group(
    tmp_path: Path,
) -> None:
    document = {
        "blocks": [
            {"block_id": "b1", "type": "text", "text": "One." + "x" * 16_000},
            {"block_id": "b2", "type": "text", "text": "Two." + "y" * 16_000},
        ],
        "integrity": {"status": "complete", "document_hash": "wrong-group"},
    }
    segments = [
        {"segment_id": "s1", "block_ids": ["b1"]},
        {"segment_id": "s2", "block_ids": ["b2"]},
    ]
    annotation = {
        "explanation": "Commentary.", "commentary": "", "prior_work": [],
        "later_work": [], "context_claims": [], "evidence_ids": [],
        "key_points": [], "source_notes": [], "evidence_requests": [],
    }

    def review(prompt: str, **_kwargs):
        response = _segment_review_response(prompt)
        if response["reviewed_segment_ids"] == ["s1"]:
            response["patches"] = [{
                "segment_id": "s2",
                "commentary": "wrong group",
                "explanation": None,
                "prior_work": None,
                "later_work": None,
                "commentary_sources": None,
                "reason": "invalid cross-group patch",
            }]
        return response

    with pytest.raises(RuntimeError, match="scope|coverage|unknown segment id"):
        _review(
            segments, None, {"s1": dict(annotation), "s2": dict(annotation)},
            document=document, glossary={"entries": []}, protected_names=[],
            evidence={"papers": [], "records": []},
            options=BuildOptions(
                paper_id="local:wrong-group", project_dir=tmp_path,
                annotation_language="en", skip_translation=True, review_context_chars=1,
            ),
            llm=review, checkpoint_dir=tmp_path / "checkpoint",
        )


def test_freeze_verifiers_report_missing_first_chapter_result() -> None:
    freeze = {"chapter_id": "ch-0001", "translation_mode": "skipped"}

    with pytest.raises(RuntimeError, match="result is missing before review"):
        _verify_frozen_first_chapter_pre_review(freeze, {})
    with pytest.raises(RuntimeError, match="result is missing after review"):
        _verify_frozen_first_chapter_final(
            freeze, {}, translations=None, annotations={},
        )
