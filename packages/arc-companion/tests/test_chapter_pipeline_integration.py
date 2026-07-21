from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import arc_companion.pipeline as pipeline
from arc_companion.ledger import initialize_lane_ledger, mark_needs_supervision
from arc_companion.pipeline import BuildOptions, build_companion
from arc_companion.source import SourceBundle
from arc_companion.io import sha256_json
from arc_llm.sessions import LLMSessionManager


def test_chaptered_skip_translation_omits_lane_artifacts_and_migration(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "Chapter One"},
        {"block_id": "p1", "type": "text", "text": "Energy is conserved."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2",
        "front_matter": {},
        "blocks": blocks,
        "equations": [],
        "figures": [],
        "tables": [],
        "assets": [],
        "links": [],
        "bibliography": [],
        "integrity": {"status": "complete", "document_hash": "skip-chapter"},
    }
    bundle = SourceBundle(
        paper_id="local:skip-chapter",
        document=document,
        parsed={
            "paper_id": "local:skip-chapter",
            "document": document,
            "source_hash": "skip-chapter",
            "structure": {
                "document_kind": "book",
                "chapters": [{"title": "Chapter One", "block_ids": ["c1", "p1"]}],
            },
        },
        metadata={"title": "Book"},
        references=[],
        citers=[],
    )
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({
        "metadata": {"source_hash": "skip-chapter", "language": "en"},
        "translations": {
            "old": {"segment_id": "old", "translation": {"blocks": [
                {"block_id": "p1", "text": "This must not migrate."},
            ]}},
        },
    }), encoding="utf-8")
    labels: list[str] = []
    result_calls: list[dict[str, object]] = []

    def llm(_prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        labels.append(label)
        if "translation" in label:
            raise AssertionError(f"translation call was submitted: {label}")
        if label.startswith("companion-glossary-"):
            return {"entries": []}
        if label.startswith("companion-guide-"):
            return {
                "motivation": "Motivation.", "main_content": "Conservation.",
                "section_logic": None, "book_position": None, "prerequisites": None,
                "supplementary_reading": [],
            }
        if label.startswith("companion-segmentation-"):
            return {"cut_after_ordinals": []}
        if label.startswith("companion-annotation-"):
            return {
                "explanation": "Commentary remains.", "commentary": "",
                "prior_work": [], "later_work": [], "context_claims": [],
                "evidence_ids": [], "key_points": [], "source_notes": [],
                "evidence_requests": [],
            }
        if label.startswith("companion-commentary-review-"):
            return {"issues": [], "patches": []}
        raise AssertionError(label)

    def result_llm(prompt: str, **kwargs):
        result_calls.append(dict(kwargs))
        return SimpleNamespace(
            value=llm(prompt, **kwargs),
            logical_receipt={"idempotency_key": kwargs["idempotency_key"]},
        )

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            annotation_language="en",
            workers=2,
            skip_translation=True,
            stop_after_first_chapter=True,
            legacy_checkpoint=legacy,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["ok"], result
    checkpoint = Path(result["data"]["checkpoint_dir"])
    assert not any("translation" in label for label in labels)
    assert not any(":translation:" in str(call.get("idempotency_key")) for call in result_calls)
    assert not list((checkpoint / "chapters").rglob("translation-ledger.json"))
    assert not list(checkpoint.rglob("translations*"))
    receipt = json.loads((checkpoint / "legacy-migration.json").read_text())
    assert receipt["translations"]["ledgers"] == {}
    assert receipt["translations"]["receipts"] == [{
        "status": "skipped",
        "reason": "translation_disabled_for_same_language_source",
    }]
    freeze = json.loads((checkpoint / "first-chapter-freeze.json").read_text())
    assert freeze["translation_mode"] == "skipped"
    assert freeze["pre_review_translation_sha256"] is None
    assert freeze["translation_sha256"] is None
    tex = Path(result["data"]["output_tex"]).read_text(encoding="utf-8")
    manifest = json.loads(Path(result["data"]["source_manifest_path"]).read_text())
    assert "ARC-TRANSLATION-" not in tex
    assert manifest["companion_layers"]["translation_mode"] is False
    assert manifest["companion_layers"]["rendered_translation_segment_ids"] == []


def test_chaptered_interactive_build_runs_only_first_chapter(tmp_path: Path) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "Chapter One"},
        {"block_id": "p1", "type": "text", "text": "Energy is conserved."},
        {"block_id": "c2", "type": "section", "title": "Chapter Two"},
        {"block_id": "p2", "type": "text", "text": "Momentum is conserved."},
        {"block_id": "idx", "type": "text", "source_role": "index", "text": "Energy, 1"},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {}, "blocks": blocks,
        "equations": [], "figures": [], "tables": [], "assets": [], "links": [],
        "bibliography": [], "integrity": {"status": "complete", "document_hash": "chapters"},
    }
    structure = {"document_kind": "book", "index_block_ids": ["idx"], "chapters": [
        {"title": "Chapter One", "block_ids": ["c1", "p1"]},
        {"title": "Chapter Two", "block_ids": ["c2", "p2"]},
    ]}
    bundle = SourceBundle(
        paper_id="local:book", document=document,
        parsed={"paper_id": "local:book", "document": document, "structure": structure,
                "index_entries": {"schema_version": "arc.paper.index_entries.v1",
                                  "entries": [{"term": "Energy", "pages": [1]}]}},
        metadata={"title": "Book"}, references=[], citers=[],
    )
    labels: list[str] = []
    result_calls: list[dict[str, object]] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"]); labels.append(label)
        if label.startswith("companion-index-glossary-"):
            return {"entries": [{"entry_id": "index-00001", "target": "能量", "explanation": "守恒量"}]}
        if label.startswith("companion-guide-"):
            return {"motivation": "理解守恒。", "main_content": "能量守恒。", "section_logic": None,
                    "book_position": None, "prerequisites": None, "supplementary_reading": []}
        if label.startswith("companion-segmentation-"):
            return {"cut_after_ordinals": []}
        if label.startswith("companion-translation-"):
            return {"blocks": [{"block_id": "c1", "text": "第一章"},
                               {"block_id": "p1", "text": "能量守恒。"}]}
        if label.startswith("companion-annotation-"):
            return {"explanation": "解释守恒的意义。", "commentary": "", "prior_work": [],
                    "later_work": [], "context_claims": [], "evidence_ids": [], "key_points": [],
                    "source_notes": [], "evidence_requests": []}
        if label == "companion-final-review":
            return {"issues": [], "patches": []}
        raise AssertionError(label)

    def result_llm(prompt: str, **kwargs):
        result_calls.append(dict(kwargs))
        value = llm(prompt, **kwargs)
        return SimpleNamespace(value=value, logical_receipt={"idempotency_key": kwargs["idempotency_key"]})

    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run",
                     workers=4, stop_after_first_chapter=True),
        source_loader=lambda *args, **kwargs: bundle, llm=llm,
        result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["status"] == "first_chapter_ready"
    assert not any("ch-0002" in label for label in labels)
    assert not (
        Path(result["data"]["checkpoint_dir"]) / "chapters" / "ch-0002"
    ).exists()
    lane_calls = [call for call in result_calls if ":translation:" in str(call.get("idempotency_key"))
                  or ":companion:" in str(call.get("idempotency_key"))]
    assert {call["session_key"] for call in lane_calls} == {
        "ch-0001:translation", "ch-0001:companion",
    }
    assert all(call["session_policy"] == "stateful" for call in lane_calls)
    assert all(call["progress_contract_scope"] == "session" for call in lane_calls)
    assert all(call["schema_formatter_enabled"] is False for call in lane_calls)
    assert len({call["idempotency_key"] for call in lane_calls}) == len(lane_calls)
    guide_calls = [call for call in result_calls if str(call.get("session_key", "")).endswith(":guide")]
    assert len(guide_calls) == 1
    assert guide_calls[0]["session_key"] == "ch-0001:guide"
    assert guide_calls[0]["session_policy"] == "stateful"
    assert guide_calls[0]["model_tier"] == "high"
    guide_ledger = json.loads(
        (Path(result["data"]["checkpoint_dir"]) / "chapters" / "ch-0001" / "guide-ledger.json").read_text()
    )
    assert guide_ledger["blocks"][0]["state"] == "accepted"
    tex = Path(result["data"]["output_tex"]).read_text(encoding="utf-8")
    assert tex.count("ARC-CHAPTER-GUIDE-BEGIN") == 1
    assert tex.index("Chapter One") < tex.index("章导读") < tex.index("Energy is conserved")
    assert "Energy, 1" not in tex
    assert (tmp_path / "run" / "state.json").is_file()
    freeze = json.loads(
        (Path(result["data"]["checkpoint_dir"]) / "first-chapter-freeze.json").read_text()
    )
    assert freeze["schema_version"] == "arc.companion.first-chapter-freeze.v3"
    assert freeze["translation_mode"] == "enabled"
    assert freeze["translation_sha256"] and freeze["annotation_sha256"]

    freeze["guide_sha256"] = "tampered"
    freeze_path = Path(result["data"]["checkpoint_dir"]) / "first-chapter-freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    labels.clear()
    rejected = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run", workers=4),
        source_loader=lambda *args, **kwargs: bundle, llm=llm, result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert not rejected["ok"]
    assert "confirmed first chapter changed" in rejected["error"]["message"]
    assert not any("ch-0002" in label for label in labels)


def test_chaptered_build_applies_read_only_legacy_migration_before_lanes(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "Chapter One"},
        {"block_id": "p1", "type": "text", "text": "Energy is conserved."},
        {"block_id": "c2", "type": "section", "title": "Chapter Two"},
        {"block_id": "p2", "type": "text", "text": "Momentum is conserved."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {}, "blocks": blocks,
        "equations": [], "figures": [], "tables": [], "assets": [], "links": [],
        "bibliography": [], "integrity": {"status": "complete", "document_hash": "chapters"},
    }
    structure = {"document_kind": "book", "chapters": [
        {"title": "Chapter One", "block_ids": ["c1", "p1"]},
        {"title": "Chapter Two", "block_ids": ["c2", "p2"]},
    ]}
    bundle = SourceBundle(
        paper_id="local:book", document=document,
        parsed={"paper_id": "local:book", "document": document, "structure": structure,
                "source_hash": "chapters"},
        metadata={"title": "Book"}, references=[], citers=[],
    )
    legacy = tmp_path / "legacy"
    (legacy / "translations").mkdir(parents=True)
    prompt_hash = sha256_json({
        "prompt_version": pipeline.PROMPT_VERSION,
        "schema_version": pipeline.SCHEMA_VERSION,
        "workflow_version": pipeline.WORKFLOW_VERSION,
    })
    validator_hash = sha256_json({
        "validator_version": pipeline.LEGACY_MIGRATION_VALIDATOR_VERSION,
        "translation_retry_prompt_version": pipeline.TRANSLATION_RETRY_PROMPT_VERSION,
        "translation_token_repair_version": pipeline.TRANSLATION_TOKEN_REPAIR_VERSION,
    })
    metadata = {
        "source_hash": "chapters", "language": "zh-CN",
        "prompt_hash": prompt_hash, "validator_hash": validator_hash,
    }
    (legacy / "document.json").write_text(json.dumps({"source_hash": "chapters"}))
    (legacy / "migration-metadata.json").write_text(json.dumps(metadata))
    (legacy / "segmentation.json").write_text(json.dumps({
        "cuts": [2], "segments": [
            {"segment_id": "old-1", "block_ids": ["c1", "p1"]},
            {"segment_id": "old-2", "block_ids": ["c2", "p2"]},
        ],
    }))
    (legacy / "glossary.json").write_text(json.dumps({
        **metadata, "entries": [{"source": "Energy", "target": "能量"}],
    }))
    (legacy / "translations" / "old-1.json").write_text(json.dumps({
        "segment_id": "old-1", "translation": {"blocks": [
            {"block_id": "c1", "text": "第一章"},
            {"block_id": "p1", "text": "能量守恒。"},
        ]},
    }))
    # These generated layers are intentionally invalid: the migration reader must not open them.
    (legacy / "chapter-guide.json").write_text("not json")
    (legacy / "review.v4.json").write_text("not json")
    before = {path.relative_to(legacy): path.read_bytes() for path in legacy.rglob("*") if path.is_file()}
    labels: list[str] = []

    def llm(_prompt: str, **kwargs):
        label = str(kwargs["call_label"]); labels.append(label)
        if label.startswith("companion-guide-"):
            return {"motivation": None, "main_content": "能量守恒。", "section_logic": None,
                    "book_position": None, "prerequisites": None, "supplementary_reading": []}
        if label.startswith("companion-annotation-"):
            return {"explanation": "解释。", "commentary": "", "prior_work": [],
                    "later_work": [], "context_claims": [], "evidence_ids": [], "key_points": [],
                    "source_notes": [], "evidence_requests": []}
        if label == "companion-final-review":
            return {"issues": [], "patches": []}
        raise AssertionError(f"migration should have avoided call: {label}")

    def result_llm(prompt: str, **kwargs):
        return SimpleNamespace(
            value=llm(prompt, **kwargs),
            logical_receipt={"idempotency_key": kwargs["idempotency_key"]},
        )

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id, project_dir=tmp_path / "run", workers=2,
            stop_after_first_chapter=True, legacy_checkpoint=legacy,
        ),
        source_loader=lambda *args, **kwargs: bundle, llm=llm, result_llm=result_llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["status"] == "first_chapter_ready"
    assert not any("translation" in label or "segmentation" in label or "glossary" in label
                   for label in labels)
    checkpoint = Path(result["data"]["checkpoint_dir"])
    receipt = json.loads((checkpoint / "legacy-migration.json").read_text())
    assert receipt["read_only_source"] is True
    assert receipt["cuts"]["reused"]["ch-0001"] == []
    assert receipt["glossary"]["accepted"] is True
    assert receipt["translations"]["receipts"][0]["accepted"] is True
    assert receipt["never_migrated"] == ["guides", "annotations", "reviews", "tex", "pdf"]
    ledger = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "translation-ledger.json").read_text()
    )
    assert ledger["blocks"][0]["logical_receipt"]["provider_calls"] == 0
    chapter_glossary = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "chapter-glossary.json").read_text()
    )
    migrated_segment = json.loads(
        (checkpoint / "chapters" / "ch-0001" / "segmentation.json").read_text()
    )["segments"][0]
    migrated_segment = {**migrated_segment, "chapter_id": "ch-0001",
                        "segment_id": "ch-0001.seg-0001"}
    assert ledger["blocks"][0]["input_sha256"] == pipeline._segment_input_hash(
        migrated_segment,
        {item["block_id"]: item for item in blocks}, glossary=chapter_glossary,
    )
    assert {path.relative_to(legacy): path.read_bytes() for path in legacy.rglob("*") if path.is_file()} == before


def test_restart_generation_rotates_session_and_invalidates_suffix(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "run"
    checkpoint = project / "checkpoint"
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger_path, chapter_id="ch-0001", lane="translation", segment_ids=["s1", "s2"],
    )
    mark_needs_supervision(
        ledger_path, segment_id="s1", reason="unknown submission",
        recovery_context={"resumable": True, "native_session_id": "native-1"},
    )
    manager = LLMSessionManager(checkpoint / "sessions")
    manager.get_or_create(
        key="ch-0001:translation", provider="codex", model="m",
        runtime_fingerprint="runtime",
    )
    manager.update_native_session_id("ch-0001:translation", "native-1")
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision", "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    captured = {}
    monkeypatch.setattr(
        pipeline, "build_companion",
        lambda options: captured.setdefault("options", options) or {"ok": True},
    )

    result = pipeline.resume_companion(
        project, action="restart-generation", confirm_possible_duplicate_charge=True,
    )

    assert result is captured["options"]  # the stub returns the saved options object
    ledger = json.loads(ledger_path.read_text())
    assert ledger["generation"] == 2
    assert ledger["blocks"][0] == {"segment_id": "s1", "state": "pending", "generation": 2}
    assert LLMSessionManager(checkpoint / "sessions").get_existing(
        "ch-0001:translation"
    ).native_session_id is None


def test_resume_native_passes_one_shot_call_authorization_to_build(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "run"
    checkpoint = project / "checkpoint"
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger_path, chapter_id="ch-0001", lane="translation", segment_ids=["s1"],
    )
    logical_key = "ch-0001:translation:companion-translation-s1:generation-1"
    mark_needs_supervision(
        ledger_path,
        segment_id="s1",
        reason="unknown submitted response",
        recovery_context={
            "idempotency_key": logical_key,
            "resumable": True,
            "native_session_id": "native-1",
            "submission_state": "submitted",
        },
    )
    manager = LLMSessionManager(checkpoint / "sessions")
    manager.get_or_create(
        key="ch-0001:translation", provider="codex", model="m",
        runtime_fingerprint="runtime",
    )
    manager.update_native_session_id("ch-0001:translation", "native-1")
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    captured = {}

    def resumed_build(options):
        captured["options"] = options
        return {"ok": True, "status": "complete"}

    monkeypatch.setattr(pipeline, "build_companion", resumed_build)

    result = pipeline.resume_companion(project, action="resume-native")

    assert result["ok"] is True
    assert captured["options"].supervised_native_resume_keys == (logical_key,)
    assert json.loads(ledger_path.read_text())["needs_supervision"] is None


def test_resume_native_rejects_supervision_without_logical_call_key(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "run"
    checkpoint = project / "checkpoint"
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    initialize_lane_ledger(
        ledger_path, chapter_id="ch-0001", lane="translation", segment_ids=["s1"],
    )
    mark_needs_supervision(
        ledger_path, segment_id="s1", reason="unknown",
        recovery_context={"resumable": True, "native_session_id": "native-1"},
    )
    project.mkdir(parents=True, exist_ok=True)
    (project / "state.json").write_text(json.dumps({
        "status": "needs_supervision",
        "checkpoint_dir": str(checkpoint),
        "recovery_options": {"paper_id": "local:book", "workers": 1},
    }))
    monkeypatch.setattr(
        pipeline, "build_companion",
        lambda _options: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    result = pipeline.resume_companion(project, action="resume-native")

    assert result["error"]["code"] == "native_resume_idempotency_key_missing"


def test_stateful_lane_timeout_persists_recovery_and_stops_other_paid_lane(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": "c1", "type": "section", "title": "One"},
        {"block_id": "p1", "type": "text", "text": "A conserved quantity."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2", "front_matter": {}, "blocks": blocks,
        "equations": [], "figures": [], "tables": [], "assets": [], "links": [],
        "bibliography": [], "integrity": {"status": "complete", "document_hash": "one"},
    }
    bundle = SourceBundle(
        paper_id="local:one", document=document,
        parsed={"paper_id": "local:one", "document": document,
                "structure": {"document_kind": "book", "chapters": [
                    {"title": "One", "block_ids": ["c1", "p1"]},
                ]}},
        metadata={"title": "One"}, references=[], citers=[],
    )
    paid: list[str] = []

    def llm(_prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith("companion-glossary"):
            return {"entries": []}
        if label.startswith("companion-segmentation"):
            return {"cut_after_ordinals": []}
        raise AssertionError(label)

    def result_llm(_prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        paid.append(label)
        if label.startswith("companion-guide"):
            return SimpleNamespace(value={
                "motivation": None, "main_content": "One", "section_logic": None,
                "book_position": None, "prerequisites": None,
                "supplementary_reading": [],
            }, logical_receipt={"idempotency_key": kwargs["idempotency_key"]})
        raise TimeoutError("unknown submitted call")

    project = tmp_path / "supervised"
    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project, workers=1),
        source_loader=lambda *args, **kwargs: bundle, llm=llm, result_llm=result_llm,
    )

    assert result["status"] == "needs_supervision"
    assert len([label for label in paid if "translation" in label or "annotation" in label]) == 1
    checkpoint = Path(result["data"]["checkpoint_dir"])
    ledgers = [json.loads(path.read_text()) for path in
               (checkpoint / "chapters" / "ch-0001").glob("*-ledger.json")]
    supervised = [value for value in ledgers if value.get("needs_supervision")]
    assert len(supervised) == 1
    context = supervised[0]["needs_supervision"]["recovery_context"]
    assert context["idempotency_key"].endswith("generation-1")
    assert context["submission_state"] is None
    assert json.loads((project / "state.json").read_text())["recovery_options"]["paper_id"] == "local:one"
