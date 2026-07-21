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
    _review,
    _verify_frozen_first_chapter_final,
    _verify_frozen_first_chapter_pre_review,
    build_companion,
)
from arc_companion.latex import render_companion_tex, validate_tex_fidelity
from arc_companion.source import SourceBundle


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


def test_old_recovery_options_keep_translation_enabled(tmp_path: Path) -> None:
    recovered = _options_from_recovery(
        tmp_path,
        {"paper_id": "local:skip-translation", "workers": 1},
    )

    assert recovered.skip_translation is False
    assert recovered.recovery_policy == "auto"


def test_recovery_policy_round_trips_and_validates(tmp_path: Path) -> None:
    original = BuildOptions(
        paper_id="local:skip-translation",
        project_dir=tmp_path,
        recovery_policy="manual",
    )

    recovered = _options_from_recovery(tmp_path, _recovery_options(original))

    assert recovered.recovery_policy == "manual"
    with pytest.raises(ValueError, match="recovery_policy"):
        BuildOptions(
            paper_id="local:skip-translation",
            project_dir=tmp_path,
            recovery_policy="invalid",
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
        if label.startswith("companion-commentary-review-"):
            return {"issues": [], "patches": []}
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
    assert any(label.startswith("companion-commentary-review-") for label in labels)
    assert not list(checkpoint.rglob("translations*"))
    assert not (checkpoint / "llm" / "translation").exists()
    reuse_plan = json.loads((checkpoint / "reuse-plan.json").read_text())
    glossary_entry = next(item for item in reuse_plan["entries"] if item["lane"] == "glossary")
    translation_entry = next(item for item in reuse_plan["entries"] if item["lane"] == "translation")
    assert glossary_entry["status"] == "skipped"
    assert glossary_entry["reason"] == "glossary_disabled_for_same_language_source"
    assert glossary_entry["estimated_provider_calls"] == 0
    assert translation_entry["status"] == "skipped"
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

    def invalid_review(_prompt: str, **_kwargs):
        return {
            "issues": [],
            "patches": [{
                "segment_id": "seg-0001",
                "commentary": None,
                "explanation": "Revised commentary.",
                "prior_work": None,
                "later_work": None,
                "evidence_ids": None,
                "reason": "attempted mixed-layer patch",
                "translation_blocks": [{"block_id": "body", "text": "Forbidden"}],
            }],
        }

    with pytest.raises(
        RuntimeError, match="commentary-only review attempted a translation patch"
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
        return {"issues": [], "patches": []}

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
    assert result["hierarchical"] is True
    assert sorted(labels) == ["companion-commentary-review-0", "companion-commentary-review-1"]
    assert all('"translation"' not in prompt for prompt in prompts)


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
        return {"issues": [], "patches": []}

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

    assert len(prompts) == 2
    assert all("GLOSSARY" not in prompt for prompt in prompts)
    assert all("RELEVANT" not in prompt for prompt in prompts)
    assert all("UNRELATED-" not in prompt for prompt in prompts)
    checkpoints = sorted((tmp_path / "checkpoint" / "commentary-reviews").glob("*.json"))
    assert len(checkpoints) == 2
    assert all(json.loads(path.read_text())["input_sha256"] for path in checkpoints)

    _review(llm=lambda *_args, **_kwargs: pytest.fail("checkpoint was not reused"), **kwargs)


def test_hierarchical_commentary_review_rejects_patch_from_another_group(
    tmp_path: Path,
) -> None:
    document = {
        "blocks": [
            {"block_id": "b1", "type": "text", "text": "One."},
            {"block_id": "b2", "type": "text", "text": "Two."},
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

    def review(_prompt: str, **kwargs):
        if str(kwargs["call_label"]).endswith("-0"):
            return {"issues": [], "patches": [{
                "segment_id": "s2", "commentary": "wrong group", "explanation": None,
                "prior_work": None, "later_work": None, "evidence_ids": None,
                "reason": "invalid cross-group patch",
            }]}
        return {"issues": [], "patches": []}

    with pytest.raises(RuntimeError, match="outside its review group"):
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
