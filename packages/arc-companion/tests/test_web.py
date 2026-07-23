from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import threading
from copy import deepcopy

import pytest

from arc_companion.io import read_json, sha256_json, write_json
from arc_companion.latex import (
    render_companion_tex,
    validate_pdf_source_credit_text,
)
from arc_companion.web import (
    READER_FINAL_VERSION,
    READER_SNAPSHOT_VERSION,
    WEB_MANIFEST_VERSION,
    WEB_RENDER_VERSION,
    WebReaderError,
    _source_credit_visible_projection,
    build_reader_snapshot,
    create_reader_publish_coordinator,
    inspect_reader_publish,
    prepare_reader_publish,
    publish_prepared_reader,
    publish_reader,
    validate_reader_project,
)
from arc_companion.reader_publish import READER_PUBLISH_STATE_VERSION
from arc_companion.source_credit import (
    normalize_source_credit,
    source_credit_placement,
)
from arc_paper.parse.source import parse_source_input


def _segment_name(segment_id: str) -> str:
    return hashlib.sha256(segment_id.encode("utf-8")).hexdigest()


def _project(tmp_path: Path, *, translation_state: str = "accepted") -> tuple[Path, Path, str]:
    project = tmp_path / "project"
    checkpoint = project / ".arc-companion" / "checkpoints" / "fingerprint"
    chapter_dir = checkpoint / "chapters" / "ch-0001"
    for path in (chapter_dir, checkpoint / "translations", checkpoint / "annotations"):
        path.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "arc.companion.state.v2",
        "status": "active",
        "paper_id": "local:reader-test",
        "checkpoint_dir": str(checkpoint),
        "translation_mode": "enabled",
        "updated_at": "2026-07-21T00:00:00+00:00",
    }
    write_json(project / "state.json", state)
    math_hash = "a" * 64
    blocks = [
        {
            "block_id": "b1",
            "type": "paragraph",
            "text": "Energy E equals mass.",
            "inline_runs": [
                {"kind": "text", "content": "Energy "},
                {
                    "kind": "math",
                    "content": "E=mc^2",
                    "tex": "E=mc^2",
                    "token_id": "b1.token-0002",
                    "content_hash": math_hash,
                },
                {"kind": "text", "content": " equals mass."},
            ],
        },
        {
            "block_id": "b2",
            "type": "paragraph",
            "text": "Second paragraph.",
            "inline_runs": [{"kind": "text", "content": "Second paragraph."}],
        },
    ]
    write_json(
        checkpoint / "document.json",
        {
            "metadata": {"title": "A Safe Reader", "authors": [{"name": "A. Author"}]},
            "document": {
                "blocks": blocks,
                "equations": [],
                "figures": [],
                "tables": [],
                "assets": [],
            },
        },
    )
    document_envelope = read_json(checkpoint / "document.json")
    write_json(
        checkpoint / "source-credit.json",
        normalize_source_credit(
            document_envelope["document"], document_envelope["metadata"],
        ),
    )
    write_json(
        checkpoint / "chapters.json",
        {
            "schema_version": "arc.companion.chapters.v1",
            "chapters": [
                {
                    "chapter_id": "ch-0001",
                    "title": "Foundations",
                    "block_ids": ["b1", "b2"],
                    "start_block_id": "b1",
                    "end_block_id": "b2",
                }
            ],
        },
    )
    write_json(
        chapter_dir / "segmentation.json",
        {
            "schema_version": "arc.companion.segmentation.v5",
            "segments": [
                {
                    "segment_id": "seg-0001",
                    "title": "First",
                    "block_ids": ["b1"],
                    "start_block_id": "b1",
                    "end_block_id": "b1",
                },
                {
                    "segment_id": "seg-0002",
                    "title": "Second",
                    "block_ids": ["b2"],
                    "start_block_id": "b2",
                    "end_block_id": "b2",
                },
            ],
        },
    )
    write_json(
        chapter_dir / "chapter-guide.json",
        {
            "schema_version": "arc.companion.chapter-guide.v3",
            "chapter_id": "ch-0001",
            "motivation": "Why $E$ matters.",
            "main_content": None,
            "section_logic": None,
            "prerequisites": None,
            "pedagogical_comparison": None,
            "historical_context": [],
            "supplementary_reading": [],
        },
    )
    segment_id = "ch-0001.seg-0001"
    translation = {
        "blocks": [
            {
                "block_id": "b1",
                "text": (
                    "能量 [[ARC_INLINE:b1.token-0002:"
                    + math_hash
                    + "]] 等于质量。"
                ),
            }
        ]
    }
    annotation = {
        "explanation": "The relation uses $c^2$.",
        "commentary": "",
        "commentary_sources": [
            {
                "title": "Primary source",
                "url": "https://example.test/paper",
                "locator": "Section 1",
            }
        ],
        "prior_work": [],
        "later_work": [],
    }
    name = _segment_name(segment_id)
    write_json(
        checkpoint / "translations" / f"{name}.json",
        {
            "schema_version": "arc.companion.translation-checkpoint.v2",
            "segment_id": segment_id,
            "translation": translation,
        },
    )
    write_json(
        checkpoint / "annotations" / f"{name}.json",
        {
            "schema_version": "arc.companion.annotation-checkpoint.v7",
            "segment_id": segment_id,
            "annotation": annotation,
        },
    )
    write_json(
        chapter_dir / "translation-ledger.json",
        {
            "schema_version": "arc.companion.chapter-lane-ledger.v1",
            "chapter_id": "ch-0001",
            "lane": "translation",
            "blocks": [
                {
                    "segment_id": segment_id,
                    "state": translation_state,
                    "output_sha256": sha256_json(translation),
                },
                {"segment_id": "ch-0001.seg-0002", "state": "pending"},
            ],
        },
    )
    write_json(
        chapter_dir / "companion-ledger.json",
        {
            "schema_version": "arc.companion.chapter-lane-ledger.v1",
            "chapter_id": "ch-0001",
            "lane": "companion",
            "blocks": [
                {
                    "segment_id": segment_id,
                    "state": "accepted",
                    "output_sha256": sha256_json(annotation),
                },
                {"segment_id": "ch-0001.seg-0002", "state": "pending"},
            ],
        },
    )
    write_json(
        checkpoint / "glossary.json",
        {
            "schema_version": "arc.companion.glossary.v7",
            "entries": [
                {
                    "source_term": "energy",
                    "target_term": "能量",
                    "explanation": "A conserved quantity.",
                }
            ],
        },
    )
    return project, checkpoint, segment_id


def _install_legacy_reader_bundle(
    project: Path,
    *,
    malformed_revision: bool = False,
) -> dict[str, object]:
    import arc_companion.web as web

    candidate = prepare_reader_publish(
        project,
        created_at=datetime(2026, 7, 22, tzinfo=timezone.utc),
    )
    for item in candidate.objects:
        if item.kind in {"builtin-asset", "source-asset"}:
            path = project / item.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item.data)

    snapshot = deepcopy(dict(candidate.semantic.snapshot))
    snapshot["schema_version"] = "arc.companion.reader-snapshot.v3"
    snapshot["web_render_version"] = "arc.companion.web-render.v4"
    for key in (
        "source_credit",
        "source_credit_sha256",
        "source_credit_order",
        "source_credit_visible_projection",
        "source_credit_front_matter_block_ids",
        "source_credit_replaced_block_ids",
        "translation_reference",
    ):
        snapshot.pop(key, None)
    snapshot["revision"] = sha256_json({
        key: value for key, value in snapshot.items() if key != "revision"
    })
    if malformed_revision:
        snapshot["revision"] = "0" * 64
    snapshot_bytes = web._json_file_bytes(snapshot)
    snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot_relative = f"reader/data/snapshot-{snapshot_hash}.json"
    snapshot_path = project / snapshot_relative
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(snapshot_bytes)

    data_bytes = (
        "window.__ARC_COMPANION_SNAPSHOT__ = "
        + web._safe_script_json(snapshot)
        + ";\n"
    ).encode()
    data_hash = hashlib.sha256(data_bytes).hexdigest()
    data_relative = f"reader/data/snapshot-{data_hash}.js"
    (project / data_relative).write_bytes(data_bytes)

    current_manifest = json.loads(next(
        item.data for item in candidate.objects if item.kind == "manifest"
    ))
    old_data_name = Path(current_manifest["data_script"]["path"]).name
    index_bytes = candidate.index.data.replace(
        old_data_name.encode(), Path(data_relative).name.encode(),
    )
    index_path = project / "reader" / "index.html"
    index_path.write_bytes(index_bytes)

    def record(relative: str, value: bytes) -> dict[str, object]:
        return {
            "path": relative,
            "sha256": hashlib.sha256(value).hexdigest(),
            "bytes": len(value),
        }

    manifest = deepcopy(current_manifest)
    manifest["schema_version"] = "arc.companion.web-manifest.v2"
    manifest["web_render_version"] = "arc.companion.web-render.v4"
    manifest["snapshot"] = record(snapshot_relative, snapshot_bytes)
    manifest["data_script"] = record(data_relative, data_bytes)
    manifest["index"] = record("reader/index.html", index_bytes)
    for key in (
        "reader_semantic_sha256",
        "source_credit",
        "translation_reference",
    ):
        manifest.pop(key, None)
    manifest_bytes = web._json_file_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_path = (
        project / "reader" / "data" / f"manifest-{manifest_hash}.json"
    )
    manifest_path.write_bytes(manifest_bytes)

    identity: dict[str, object] = {
        "output_html": str(index_path),
        "output_html_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "reader_snapshot_path": str(snapshot_path),
        "reader_snapshot_sha256": snapshot_hash,
        "web_manifest_path": str(manifest_path),
        "web_manifest_sha256": manifest_hash,
        "web_render_version": "arc.companion.web-render.v4",
    }
    state = {
        **read_json(project / "state.json"),
        **identity,
        "published": {"web": dict(identity)},
    }
    write_json(project / "state.json", state)
    return state


def test_snapshot_discovers_only_hash_verified_accepted_lane_values(tmp_path: Path) -> None:
    project, _checkpoint, segment_id = _project(tmp_path)

    snapshot = build_reader_snapshot(project)

    assert snapshot["schema_version"] == READER_SNAPSHOT_VERSION
    assert snapshot["coverage"] == {
        "chapter_ids": ["ch-0001"],
        "segment_ids": ["ch-0001.seg-0001", "ch-0001.seg-0002"],
        "translation_segment_ids": [segment_id],
        "annotation_segment_ids": [segment_id],
    }
    first, second = snapshot["chapters"][0]["segments"]
    assert next(
        item for item in first["translation"]["blocks"][0]["runs"]
        if item["type"] == "math"
    ) == {
        "type": "math",
        "tex": "E=mc^2",
        "display": False,
    }
    assert first["source"][0]["runs"][0] == {
        "type": "term",
        "text": "Energy",
        "entry_id": "term-0001",
        "source": "energy",
        "target": "能量",
    }
    assert first["companion"]["sections"][0]["sources"][0]["locator"] == "Section 1"
    assert second["translation"] is None and second["companion"] is None
    assert snapshot["revision"] == sha256_json(
        {key: value for key, value in snapshot.items() if key != "revision"}
    )


def test_math_runs_trim_all_supported_delimiters() -> None:
    import arc_companion.web as web

    runs = web._text_math_runs(
        r"inline $a+b$, display $$c+d$$, paren \(e+f\), bracket \[g+h\]"
    )
    math = [item for item in runs if item["type"] == "math"]

    assert [item["tex"] for item in math] == ["a+b", "c+d", "e+f", "g+h"]
    assert [item["display"] for item in math] == [False, True, False, True]


def test_inline_separator_metadata_drives_projection_and_web_without_heuristics() -> None:
    import arc_companion.web as web
    from arc_companion.projection import translation_input_block

    digest = "b" * 64
    block = {
        "block_id": "b1",
        "kind": "paragraph",
        "text": "plane ξ=0 be",
        "inline_runs": [
            {"kind": "text", "content": "plane"},
            {
                "kind": "math", "content": r"\xi=0", "tex": r"\xi=0",
                "token_id": "b1.token-0002", "content_hash": digest,
                "separator_before": " ",
            },
            {"kind": "text", "content": "be", "separator_before": " "},
        ],
    }

    projected = translation_input_block(block)["text"]
    assert projected == f"plane [[ARC_INLINE:b1.token-0002:{digest}]] be"
    assert web._inline_runs(block) == [
        {"type": "text", "text": "plane"},
        {"type": "text", "text": " "},
        {"type": "math", "tex": r"\xi=0", "display": False},
        {"type": "text", "text": " "},
        {"type": "text", "text": "be"},
    ]

    adjacent = {**block, "inline_runs": [
        {key: value for key, value in run.items() if key != "separator_before"}
        for run in block["inline_runs"]
    ]}
    assert " " not in translation_input_block(adjacent)["text"]


def test_term_runs_are_bilingual_normalized_bounded_and_deterministic() -> None:
    import arc_companion.web as web

    glossary = web._glossary_view({"entries": [
        {
            "entry_id": "short",
            "source": "field",
            "target": "场",
            "aliases": ["FIELD"],
        },
        {
            "entry_id": "long",
            "source": "gauge field",
            "target": "规范场",
            "source_aliases": ["gauge-field"],
        },
        {"source": "résumé", "target": "简历"},
        {"source": "same", "target": "ＳＡＭＥ"},
        {"source": "empty", "target": ""},
    ]})

    runs = web._term_runs(
        "ＧＡＵＧＥ ＦＩＥＬＤ / gauge-field / 规范场; re\u0301sume\u0301 but résumés field2.",
        glossary,
    )
    terms = [item for item in runs if item["type"] == "term"]

    assert [(item["text"], item["entry_id"]) for item in terms] == [
        ("ＧＡＵＧＥ ＦＩＥＬＤ", "long"),
        ("gauge-field", "long"),
        ("规范场", "long"),
        ("re\u0301sume\u0301", "term-0003"),
    ]
    assert all(item["entry_id"] not in {"short", "term-0004", "term-0005"} for item in terms)


def test_term_annotation_leaves_math_and_links_opaque() -> None:
    import arc_companion.web as web

    value = {"runs": [
        {"type": "text", "text": "energy"},
        {"type": "math", "tex": r"\text{energy}", "display": False},
        {"type": "link", "text": "energy", "href": "https://example.test"},
    ]}
    glossary = web._glossary_view({"entries": [{"source": "energy", "target": "能量"}]})

    web._annotate_term_runs(value, glossary)

    assert [item["type"] for item in value["runs"]] == ["term", "math", "link"]


def test_pending_ledger_hides_an_existing_translation_checkpoint(tmp_path: Path) -> None:
    project, _checkpoint, _segment_id = _project(
        tmp_path, translation_state="schema_valid"
    )

    snapshot = build_reader_snapshot(project)

    assert snapshot["coverage"]["translation_segment_ids"] == []
    assert snapshot["chapters"][0]["segments"][0]["translation"] is None


def test_accepted_output_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    project, checkpoint, _segment_id = _project(tmp_path)
    ledger_path = checkpoint / "chapters" / "ch-0001" / "translation-ledger.json"
    ledger = read_json(ledger_path)
    ledger["blocks"][0]["output_sha256"] = "0" * 64
    write_json(ledger_path, ledger)

    with pytest.raises(WebReaderError, match="accepted translation hash mismatch"):
        build_reader_snapshot(project)


def test_reader_final_checkpoint_overrides_live_state_and_supports_legacy_segments(
    tmp_path: Path,
) -> None:
    project, checkpoint, _segment_id = _project(tmp_path)
    document = read_json(checkpoint / "document.json")["document"]
    final_annotation = {
        "explanation": "Reviewed explanation.",
        "commentary": "",
        "commentary_sources": [],
        "prior_work": [],
        "later_work": [],
    }
    write_json(
        checkpoint / "reader-final.json",
        {
            "schema_version": READER_FINAL_VERSION,
            "final_overrides": {
                "status": "complete",
                "language": "zh-CN",
                "document": document,
                "chapters": [],
                "segments": [
                    {
                        "segment_id": "seg-0001",
                        "block_ids": ["b1", "b2"],
                        "start_block_id": "b1",
                        "end_block_id": "b2",
                    }
                ],
                "chapter_guides": {},
                "translations": None,
                "annotations": {"seg-0001": final_annotation},
                "glossary": {"entries": []},
                "metadata": {"title": "Reviewed Reader"},
                "translation_mode": "skipped",
            },
        },
    )

    snapshot = build_reader_snapshot(project)

    assert snapshot["status"] == "complete"
    assert snapshot["language"] == "zh-CN"
    assert snapshot["title"] == "Reviewed Reader"
    assert snapshot["coverage"]["segment_ids"] == ["seg-0001"]
    assert snapshot["chapters"][0]["segments"][0]["companion"]["sections"][0][
        "runs"
    ][0]["text"] == "Reviewed explanation."
    assert snapshot["chapters"][0]["segments"][0]["lane_status"]["companion"] == "accepted"


def test_explicit_current_overrides_read_legacy_reader_final_without_rewrite(
    tmp_path: Path,
) -> None:
    project, checkpoint, _segment_id = _project(tmp_path)
    path = checkpoint / "reader-final.json"
    write_json(path, {
        "schema_version": "arc.companion.reader-final.v3",
        "final_overrides": {"metadata": {"title": "Legacy checkpoint"}},
    })
    before = path.read_bytes()

    snapshot = build_reader_snapshot(
        project,
        final_overrides={"metadata": {"title": "Current render"}},
    )

    assert snapshot["title"] == "Current render"
    assert path.read_bytes() == before
    with pytest.raises(WebReaderError, match="checkpoint schema"):
        build_reader_snapshot(project)
    with pytest.raises(WebReaderError, match="checkpoint schema"):
        build_reader_snapshot(project, final_overrides={})


@pytest.mark.parametrize(
    ("schema_version", "final_payload", "error"),
    [
        ("arc.companion.reader-final.v2", {}, "checkpoint schema"),
        ("arc.companion.reader-final.v5", {}, "checkpoint schema"),
        ("arc.companion.reader-final.v3", [], "no final_overrides"),
    ],
)
def test_explicit_current_overrides_reject_other_or_malformed_reader_final(
    tmp_path: Path,
    schema_version: str,
    final_payload: object,
    error: str,
) -> None:
    project, checkpoint, _segment_id = _project(tmp_path)
    write_json(checkpoint / "reader-final.json", {
        "schema_version": schema_version,
        "final_overrides": final_payload,
    })

    with pytest.raises(WebReaderError, match=error):
        build_reader_snapshot(
            project,
            final_overrides={"metadata": {"title": "Current render"}},
        )


def test_skipped_snapshot_hides_stale_glossary_terms_and_keeps_source_index(
    tmp_path: Path,
) -> None:
    import arc_companion.web as web

    project, checkpoint, _segment_id = _project(tmp_path)
    envelope = read_json(checkpoint / "document.json")
    envelope["document"]["blocks"].extend([
        {
            "block_id": "index-heading",
            "type": "heading",
            "title": "Index",
            "text": "Index",
            "source_role": "index",
        },
        {
            "block_id": "index-entry",
            "type": "paragraph",
            "text": "Energy, 1",
            "source_role": "index",
        },
    ])
    write_json(checkpoint / "document.json", envelope)

    snapshot = build_reader_snapshot(
        project,
        final_overrides={
            "translation_mode": "skipped",
            # A stale or accidentally supplied glossary must be ignored.
            "glossary": {"entries": [{"source": "energy", "target": "能量"}]},
            "translations": None,
        },
    )

    assert snapshot["glossary"] == []
    assert snapshot["coverage"]["translation_segment_ids"] == []
    assert snapshot["appendices"][0]["kind"] == "source_only_index"
    assert snapshot["appendices"][0]["source"][1]["runs"] == [
        {"type": "text", "text": "Energy, 1"}
    ]
    assert not list(web._walk_term_runs(snapshot))


def test_active_override_without_checkpoint_proof_is_preview_only(tmp_path: Path) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    preview = {
        "explanation": "Uncheckpointed preview.",
        "commentary": "",
        "commentary_sources": [],
        "prior_work": [],
        "later_work": [],
    }

    snapshot = build_reader_snapshot(
        project,
        final_overrides={"annotations": {"ch-0001.seg-0002": preview}},
    )

    segment = snapshot["chapters"][0]["segments"][1]
    assert segment["companion"] is not None
    assert segment["lane_status"]["companion"] == "preview"


@pytest.mark.parametrize("status", ["complete", "first_chapter_ready"])
def test_terminal_state_requires_explicit_or_checkpointed_final_payload(
    tmp_path: Path, status: str
) -> None:
    project, checkpoint, _segment_id = _project(tmp_path)
    state_path = project / "state.json"
    state = read_json(state_path)
    state["status"] = status
    write_json(state_path, state)

    with pytest.raises(WebReaderError, match="requires final_overrides"):
        build_reader_snapshot(project)
    with pytest.raises(WebReaderError, match="requires final_overrides"):
        publish_reader(project)

    explicit_annotation = {
        "explanation": "Explicit final payload.",
        "commentary": "",
        "commentary_sources": [],
        "prior_work": [],
        "later_work": [],
    }
    explicit = build_reader_snapshot(
        project,
        final_overrides={
            "annotations": {"ch-0001.seg-0002": explicit_annotation}
        },
    )
    assert explicit["status"] == status
    assert explicit["chapters"][0]["segments"][1]["lane_status"]["companion"] == "accepted"
    write_json(
        checkpoint / "reader-final.json",
        {
            "schema_version": READER_FINAL_VERSION,
            "final_overrides": {"status": status},
        },
    )
    assert build_reader_snapshot(project)["status"] == status


def test_publish_is_static_local_content_addressed_and_index_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    import arc_companion.web as web

    writes: list[Path] = []
    real_write_text = web.write_text

    def recording_write_text(path: Path, text: str) -> None:
        writes.append(Path(path))
        real_write_text(path, text)

    monkeypatch.setattr(web, "write_text", recording_write_text)
    result = publish_reader(project)

    index = Path(result["output_html"])
    snapshot_path = Path(result["reader_snapshot_path"])
    manifest_path = Path(result["web_manifest_path"])
    assert writes[-1] == index
    assert index.is_file() and snapshot_path.is_file() and manifest_path.is_file()
    html = index.read_text(encoding="utf-8")
    manifest = read_json(manifest_path)
    assert manifest["schema_version"] == WEB_MANIFEST_VERSION
    assert manifest["web_render_version"] == WEB_RENDER_VERSION
    assert snapshot_path.parent.name == "data"
    assert snapshot_path.name == f"snapshot-{result['reader_snapshot_sha256']}.json"
    assert manifest_path.parent.name == "data"
    assert manifest_path.name == f"manifest-{result['web_manifest_sha256']}.json"
    assert manifest["data_script"]["path"].startswith("reader/data/snapshot-")
    assert Path(project / manifest["data_script"]["path"]).name in html
    asset_paths = {item["path"] for item in manifest["assets"]}
    assert any(path.endswith("/reader.js") for path in asset_paths)
    assert any(path.endswith("/reader.css") for path in asset_paths)
    assert any(path.endswith("/katex/katex.min.js") for path in asset_paths)
    assert any(path.endswith("/katex/katex.min.css") for path in asset_paths)
    assert any("/katex/fonts/" in path for path in asset_paths)
    assert all("/builtin-" in path for path in asset_paths)
    assert "https://cdn" not in html and "unpkg.com" not in html
    assert validate_reader_project(project, state={
        **read_json(project / "state.json"), **result,
    })["ok"] is True


def test_prepare_reader_publish_is_zero_write_and_candidate_is_fixed(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    before = {
        path.relative_to(project): (path.stat().st_mtime_ns, path.read_bytes())
        for path in project.rglob("*")
        if path.is_file()
    }

    candidate = prepare_reader_publish(
        project,
        created_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    after = {
        path.relative_to(project): (path.stat().st_mtime_ns, path.read_bytes())
        for path in project.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert candidate.semantic.semantic_sha256
    assert all(item.data for item in candidate.objects)


def test_immutable_targets_adopt_exact_bytes_and_reject_conflicts(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    candidate = prepare_reader_publish(project)
    first = publish_prepared_reader(candidate)
    immutable = Path(first["reader_snapshot_path"])
    before_mtime = immutable.stat().st_mtime_ns

    publish_prepared_reader(candidate)
    assert immutable.stat().st_mtime_ns == before_mtime

    other_project, _checkpoint, _segment_id = _project(
        tmp_path / "other",
    )
    conflicting = prepare_reader_publish(other_project)
    target = other_project / conflicting.objects[0].relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"conflict")
    with pytest.raises(WebReaderError, match="conflicts"):
        publish_prepared_reader(conflicting)
    assert not (other_project / "reader" / "index.html").exists()


def test_publish_rejects_symlink_target_and_parent(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    candidate = prepare_reader_publish(project)
    target = project / candidate.objects[0].relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.write_bytes(candidate.objects[0].data)
    target.symlink_to(outside)
    with pytest.raises(WebReaderError, match="cannot be adopted"):
        publish_prepared_reader(candidate)

    other, _checkpoint, _segment_id = _project(tmp_path / "parent")
    parent_candidate = prepare_reader_publish(other)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (other / "reader").symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(WebReaderError, match="regular directory"):
        publish_prepared_reader(parent_candidate)


def test_inspector_recomputes_semantic_instead_of_trusting_manifest(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    published = publish_reader(project)
    manifest_path = Path(published["web_manifest_path"])
    manifest = read_json(manifest_path)
    manifest["reader_semantic_sha256"] = "0" * 64
    write_json(manifest_path, manifest)

    with pytest.raises(WebReaderError, match="semantic digest"):
        inspect_reader_publish(project)


def test_snapshot_source_assets_must_match_manifest_all_and_only(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    snapshot = build_reader_snapshot(project)
    snapshot["appendices"] = [
        {
            "source": [
                {
                    "assets": [
                        {
                            "url": "assets/source/missing.png",
                            "sha256": "0" * 64,
                        }
                    ]
                }
            ]
        }
    ]
    snapshot["revision"] = sha256_json({
        key: value for key, value in snapshot.items()
        if key != "revision"
    })

    with pytest.raises(
        WebReaderError,
        match="snapshot and manifest source assets differ",
    ):
        publish_reader(project, snapshot=snapshot)
    assert not (project / "reader" / "index.html").exists()


def test_coordinator_adopts_index_before_state_without_web_rewrite(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    published = publish_reader(project)
    files = [
        path for path in (project / "reader").rglob("*") if path.is_file()
    ]
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files}
    state: dict[str, object] = {}
    reconciled = datetime(2026, 7, 23, 4, tzinfo=timezone.utc)

    def merge(values):
        state.update(values)
        return dict(state)

    create_reader_publish_coordinator(
        project,
        state_loader=lambda: dict(state),
        state_merger=merge,
        utc_now=lambda: reconciled,
        monotonic=lambda: 10.0,
    )

    assert state["reader_publish_state_version"] == READER_PUBLISH_STATE_VERSION
    assert state["reader_committed_at"] == reconciled.isoformat()
    assert state["web_manifest_path"] == published["web_manifest_path"]
    assert state["reader_committed_semantic_sha256"]
    assert {
        path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files
    } == before


def test_adoption_repairs_reconcile_utc_but_exact_startup_preserves_it(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    publish_reader(project)
    actual = inspect_reader_publish(project)
    assert actual is not None
    old = datetime(2026, 7, 22, tzinfo=timezone.utc).isoformat()
    state: dict[str, object] = {
        **actual,
        "reader_publish_state_version": READER_PUBLISH_STATE_VERSION,
        "reader_dirty": False,
        "reader_committed_at": old,
        "web_manifest_path": "stale",
    }

    def merge(values):
        state.update(values)
        return dict(state)

    now = datetime(2026, 7, 23, tzinfo=timezone.utc)
    create_reader_publish_coordinator(
        project,
        state_loader=lambda: dict(state),
        state_merger=merge,
        utc_now=lambda: now,
        monotonic=lambda: 1.0,
    )
    assert state["reader_committed_at"] == now.isoformat()

    state["reader_committed_at"] = old
    create_reader_publish_coordinator(
        project,
        state_loader=lambda: dict(state),
        state_merger=merge,
        utc_now=lambda: now,
        monotonic=lambda: 1.0,
    )
    assert state["reader_committed_at"] == old


def test_same_candidate_publish_race_adopts_exact_targets(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    candidate = prepare_reader_publish(project)
    barrier = threading.Barrier(2)
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=2)
            results.append(publish_prepared_reader(candidate))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert len(results) == 2
    assert inspect_reader_publish(project) is not None


def test_concurrent_failed_publish_cannot_rollback_successful_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arc_companion.web as web_module

    project, _checkpoint, _segment_id = _project(tmp_path)
    candidate = prepare_reader_publish(project)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def fault(label: str) -> None:
        if (
            threading.current_thread().name == "failing-publisher"
            and label == "post-index-validation"
        ):
            raise RuntimeError("injected post-index failure")

    monkeypatch.setattr(web_module, "_publish_fault_point", fault)

    def worker() -> None:
        barrier.wait(timeout=2)
        try:
            publish_prepared_reader(candidate)
        except RuntimeError:
            outcomes.append("failed")
        else:
            outcomes.append("published")

    threads = [
        threading.Thread(target=worker, name="failing-publisher"),
        threading.Thread(target=worker, name="successful-publisher"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert sorted(outcomes) == ["failed", "published"]
    assert inspect_reader_publish(project) is not None


def test_current_bundle_without_semantic_field_is_adopted_without_rewrite(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    published = publish_reader(project)
    old_manifest = Path(published["web_manifest_path"])
    manifest = read_json(old_manifest)
    manifest.pop("reader_semantic_sha256")
    manifest_bytes = (
        json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True,
            default=str,
        )
        + "\n"
    ).encode()
    legacy_path = old_manifest.with_name(
        f"manifest-{hashlib.sha256(manifest_bytes).hexdigest()}.json"
    )
    legacy_path.write_bytes(manifest_bytes)
    old_manifest.unlink()
    files = [
        path for path in (project / "reader").rglob("*") if path.is_file()
    ]
    before = {path: path.stat().st_mtime_ns for path in files}
    state: dict[str, object] = {}

    def merge(values):
        state.update(values)
        return dict(state)

    create_reader_publish_coordinator(
        project,
        state_loader=lambda: dict(state),
        state_merger=merge,
        utc_now=lambda: datetime(2026, 7, 23, tzinfo=timezone.utc),
        monotonic=lambda: 1.0,
    )

    assert state["reader_committed_semantic_sha256"]
    assert {path: path.stat().st_mtime_ns for path in files} == before


def test_state_bound_legacy_bundle_upgrades_through_coordinator(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    state = _install_legacy_reader_bundle(project)
    old_index = (project / "reader" / "index.html").read_bytes()
    for key in (
        "output_html",
        "output_html_sha256",
        "reader_snapshot_path",
        "reader_snapshot_sha256",
        "web_manifest_path",
        "web_manifest_sha256",
        "web_render_version",
    ):
        state.pop(key)
    write_json(project / "state.json", state)

    def load() -> dict[str, object]:
        return dict(state)

    def merge(values) -> dict[str, object]:
        state.update(values)
        write_json(project / "state.json", state)
        return dict(state)

    coordinator = create_reader_publish_coordinator(
        project,
        state_loader=load,
        state_merger=merge,
        utc_now=lambda: datetime(2026, 7, 23, tzinfo=timezone.utc),
        monotonic=lambda: 1.0,
    )
    assert state["reader_dirty"] is True
    assert state["reader_committed_semantic_sha256"] == ""

    result = coordinator.request(lambda: None, final=True, strict=True)

    assert result.published is True
    assert (project / "reader" / "index.html").read_bytes() != old_index
    inspected = inspect_reader_publish(project)
    assert inspected is not None
    assert inspected["web_render_version"] == WEB_RENDER_VERSION


def test_legacy_upgrade_refuses_disagreeing_state_and_preserves_index(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    state = _install_legacy_reader_bundle(project)
    candidate = prepare_reader_publish(project, state=state)
    old_index = (project / "reader" / "index.html").read_bytes()
    disk_state = read_json(project / "state.json")
    disk_state["published"]["web"]["output_html_sha256"] = "f" * 64
    write_json(project / "state.json", disk_state)

    with pytest.raises(WebReaderError):
        publish_prepared_reader(candidate)

    assert (project / "reader" / "index.html").read_bytes() == old_index


def test_legacy_upgrade_refuses_malformed_bound_bundle_and_preserves_index(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    state = _install_legacy_reader_bundle(
        project, malformed_revision=True,
    )
    candidate = prepare_reader_publish(project, state=state)
    old_index = (project / "reader" / "index.html").read_bytes()

    with pytest.raises(WebReaderError, match="snapshot schema"):
        publish_prepared_reader(candidate)

    assert (project / "reader" / "index.html").read_bytes() == old_index


def test_legacy_upgrade_post_index_failure_restores_then_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arc_companion.web as web

    project, _checkpoint, _segment_id = _project(tmp_path)
    state = _install_legacy_reader_bundle(project)
    candidate = prepare_reader_publish(project, state=state)
    index_path = project / "reader" / "index.html"
    old_index = index_path.read_bytes()

    def fail_after_index(label: str) -> None:
        if label == "post-index-validation":
            raise RuntimeError("injected legacy upgrade failure")

    monkeypatch.setattr(web, "_publish_fault_point", fail_after_index)
    with pytest.raises(RuntimeError, match="legacy upgrade failure"):
        publish_prepared_reader(candidate)
    assert index_path.read_bytes() == old_index

    monkeypatch.setattr(web, "_publish_fault_point", lambda _label: None)
    result = publish_prepared_reader(candidate)
    assert index_path.read_bytes() != old_index
    assert result["web_render_version"] == WEB_RENDER_VERSION


def test_post_index_state_failure_retries_by_adoption_without_web_rewrite(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    state: dict[str, object] = dict(read_json(project / "state.json"))

    def failing_merge(values):
        if values.get("reader_committed_semantic_sha256"):
            raise RuntimeError("state merge failed")
        state.update(values)
        return dict(state)

    coordinator = create_reader_publish_coordinator(
        project,
        state_loader=lambda: dict(state),
        state_merger=failing_merge,
        utc_now=lambda: datetime(2026, 7, 23, tzinfo=timezone.utc),
        monotonic=lambda: 1.0,
    )
    with pytest.raises(RuntimeError, match="state merge failed"):
        coordinator.request(lambda: None, final=True, strict=True)
    index = project / "reader" / "index.html"
    assert index.is_file()
    before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in (project / "reader").rglob("*")
        if path.is_file()
    }

    def merge(values):
        state.update(values)
        return dict(state)

    retry = create_reader_publish_coordinator(
        project,
        state_loader=lambda: dict(state),
        state_merger=merge,
        utc_now=lambda: datetime(2026, 7, 23, 0, 1, tzinfo=timezone.utc),
        monotonic=lambda: 2.0,
    )
    result = retry.request(lambda: None, final=True, strict=True)

    assert result.status == "deduplicated"
    assert {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in (project / "reader").rglob("*")
        if path.is_file()
    } == before


def test_coordinator_rejects_unexplained_regular_index(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    index = project / "reader" / "index.html"
    index.parent.mkdir(parents=True)
    index.write_text("unexplained", encoding="utf-8")
    state: dict[str, object] = {}

    with pytest.raises(
        WebReaderError,
        match="no web manifest matches",
    ):
        create_reader_publish_coordinator(
            project,
            state_loader=lambda: dict(state),
            state_merger=lambda values: dict(values),
        )
    assert index.read_text(encoding="utf-8") == "unexplained"


def test_web_snapshot_and_manifest_bind_shared_source_credit_and_reject_tamper(
    tmp_path: Path,
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    credit = normalize_source_credit({
        "front_matter": {
            "authors": [
                {"source_id": "a", "name": "Original <A>"},
                {"source_id": "b", "name": "اسم ب"},
            ],
            "affiliations": [{"source_id": "i", "text": "R&D Institute"}],
            "profiles": [{
                "source_id": "p",
                "text": "Long profile 第一行\nSecond line.",
                "author_id": "a",
            }],
            "author_name_variants": [{
                "author_id": "a",
                "localized_name": "本地名",
                "source_identity": "source:localized-a",
            }],
        },
        "blocks": [],
    })
    snapshot = build_reader_snapshot(
        project, final_overrides={"source_credit": credit},
    )

    assert snapshot["source_credit"] == credit
    assert snapshot["source_credit_sha256"] == credit["canonical_sha256"]
    assert snapshot["authors"] == ["Original <A>", "اسم ب"]
    assert [item["kind"] for item in snapshot["source_credit_order"]] == [
        "author", "author", "affiliation", "profile",
    ]
    result = publish_reader(project, snapshot=snapshot)
    manifest = read_json(Path(result["web_manifest_path"]))
    assert manifest["source_credit"]["canonical_sha256"] == credit[
        "canonical_sha256"
    ]
    assert manifest["source_credit"]["visible_counts"] == {
        "authors": 2, "affiliations": 1, "profiles": 1,
    }

    tampered = deepcopy(snapshot)
    tampered["source_credit"]["authors"][0]["source_name"] = "Changed"
    with pytest.raises(WebReaderError, match="source credit"):
        publish_reader(project, snapshot=tampered)


def test_every_publish_fault_point_preserves_the_previous_valid_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    import arc_companion.web as web

    first = publish_reader(project)
    previous_state = {**read_json(project / "state.json"), **first}
    index_path = Path(first["output_html"])
    previous_index = index_path.read_bytes()
    labels: list[str] = []
    monkeypatch.setattr(web, "_publish_fault_point", labels.append)
    publish_reader(project, state=previous_state)
    labels = list(dict.fromkeys(labels))
    assert {"snapshot", "data-script", "manifest", "index", "post-index-validation"} <= set(labels)
    assert any(label.startswith("builtin-asset:") for label in labels)

    for ordinal, target in enumerate(labels):
        def fail_at(label: str, *, expected: str = target) -> None:
            if label == expected:
                raise RuntimeError(f"injected publish failure at {expected}")

        monkeypatch.setattr(web, "_publish_fault_point", fail_at)
        candidate_state = {
            **previous_state,
            "updated_at": f"2026-07-21T00:00:{ordinal:02d}+00:00",
        }
        with pytest.raises(RuntimeError, match="injected publish failure"):
            publish_reader(project, state=candidate_state)
        assert index_path.read_bytes() == previous_index
        assert validate_reader_project(project, state=previous_state)["ok"] is True


def test_web_assets_use_container_layout_lazy_mount_text_nodes_and_katex() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "arc_companion" / "web_assets"
    css = (root / "reader.css").read_text(encoding="utf-8")
    javascript = (root / "reader.js").read_text(encoding="utf-8")
    katex = (root / "katex" / "katex.min.js").read_text(encoding="utf-8")

    assert "@container" in css
    assert "--translation: #f2f7ff" in css
    assert "--companion: #fff8e8" in css
    assert "IntersectionObserver" in javascript
    assert "history.replaceState" in javascript
    assert "restoreReadingPosition" in javascript
    assert "textContent" in javascript
    assert "innerHTML" not in javascript
    assert "window.katex.render" in javascript
    assert "localStorage" in javascript
    assert 'snapshot.translation_mode !== "skipped"' in javascript
    assert 'link.href = "#glossary"' in javascript
    assert "mountGlossary" in javascript
    assert "data-tooltip" in css
    assert "#36586b" in css
    assert 'run.type === "term"' in javascript
    assert "KaTeX" in katex
    assert 'font: 1rem/1.7 Inter' in css
    assert "padding: 2.6rem 1rem 2rem;" in css
    assert (
        ".sidebar h2 { margin: 0 0 .75rem; font-size: 1rem; "
        "color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }"
    ) in css
    assert """.sidebar a {
  display: block;
  padding: .42rem .55rem;
  border-radius: .4rem;
  color: #344250;
  font-size: .85rem;
  line-height: 1.35;
  text-decoration: none;
}""" in css
    assert """.sidebar-toggle {
  position: fixed;
  z-index: 30;
  top: .25rem;
  left: .25rem;
  min-width: 1.8rem;
  height: 1.8rem;
  margin: 0;
  padding: 0 .35rem;
  border: 1px solid #cbd3dc;
  border-radius: .3rem;
  background: rgba(255,255,255,.96);
  color: #2b3b49;
  font-size: .68rem;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(28,39,50,.1);
}""" in css
    assert (
        ".guide-label, .annotation-label { display: block; margin-bottom: .15rem; "
        "font-size: 1rem; font-weight: 700; color: #526474; letter-spacing: .03em; }"
    ) in css
    assert ".paper-header h1 { margin: 0; font-size: 1.35rem" in css
    assert ".chapter > h2 { margin: 0 0 1.3rem; font-size: 1.20rem" in css
    assert "font-size: 1.08rem" in css
    assert 'toggle.textContent = toggleLabel' in javascript
    assert 'toggle.setAttribute("aria-label", toggleLabel)' in javascript
    assert 'toggle.setAttribute("title", toggleLabel)' in javascript
    assert "segment.title" not in javascript
    assert "bilingualHeading" in javascript
    assert 'node.setAttribute("lang"' in javascript
    assert 'node.setAttribute("dir"' in javascript
    assert ".title-source, .title-translation" in css


@pytest.mark.parametrize(("language", "open_label", "closed_label"), [
    ("zh-CN", "收起侧栏", "展开侧栏"),
    ("en", "Collapse sidebar", "Expand sidebar"),
])
def test_sidebar_toggle_updates_dom_accessibility_storage_and_mobile_navigation(
    language: str, open_label: str, closed_label: str,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    javascript = (
        Path(__file__).resolve().parents[1]
        / "src" / "arc_companion" / "web_assets" / "reader.js"
    ).read_text(encoding="utf-8")
    harness = r'''
class Classes {
  constructor(value = "") { this.values = new Set(String(value).split(/\s+/).filter(Boolean)); }
  contains(value) { return this.values.has(value); }
  add(value) { this.values.add(value); }
  remove(value) { this.values.delete(value); }
  toggle(value, force) {
    if (force === undefined) force = !this.values.has(value);
    if (force) this.values.add(value); else this.values.delete(value);
  }
}
class Node {
  constructor(tag = "div", id = "") {
    this.tagName = tag.toUpperCase(); this.id = id; this.children = []; this.dataset = {};
    this.attributes = {}; this.listeners = {}; this.classList = new Classes(); this.textContent = "";
  }
  set className(value) { this._className = value; this.classList = new Classes(value); }
  get className() { return this._className || ""; }
  get childNodes() { return this.children; }
  append(...values) { this.children.push(...values); }
  replaceChildren(...values) { this.children = values; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(kind, fn) { this.listeners[kind] = fn; }
  querySelectorAll() { return []; }
  scrollIntoView() {}
}
const nodes = Object.fromEntries(["reader-app", "reader-main", "chapter-sidebar", "sidebar-toggle"].map(id => [id, new Node("div", id)]));
nodes["reader-app"].classList.add("reader-shell");
const store = new Map();
global.localStorage = {getItem: key => store.get(key) || null, setItem: (key, value) => store.set(key, value)};
global.location = {hash: ""}; global.history = {state: null, replaceState() {}};
global.requestAnimationFrame = fn => fn();
global.document = {
  getElementById: id => nodes[id] || null,
  createElement: tag => new Node(tag),
  createTextNode: text => ({textContent: String(text)})
};
global.window = {
  __ARC_COMPANION_SNAPSHOT__: {language: "zh-CN", title: "T", paper_id: "p", chapters: [{chapter_id: "c1", title: "C", structural_only: true, segments: []}], appendices: [], glossary: [], coverage: {}},
  matchMedia: () => ({matches: true})
};
'''
    harness = harness.replace('language: "zh-CN"', f"language: {json.dumps(language)}")
    assertions = r'''
const app = nodes["reader-app"], toggle = nodes["sidebar-toggle"];
if (toggle.textContent !== "收起侧栏" || toggle.attributes["aria-label"] !== "收起侧栏" || toggle.attributes.title !== "收起侧栏" || toggle.attributes["aria-expanded"] !== "true") process.exit(11);
toggle.listeners.click();
if (!app.classList.contains("sidebar-collapsed") || toggle.textContent !== "展开侧栏" || toggle.attributes["aria-expanded"] !== "false" || store.get("arc-reader-sidebar") !== "closed") process.exit(12);
toggle.listeners.click();
const find = (root, tag) => { for (const child of root.children) { if (child && child.tagName === tag) return child; if (child && child.children) { const found = find(child, tag); if (found) return found; } } return null; };
const link = find(nodes["chapter-sidebar"], "A");
link.listeners.click({preventDefault() {}});
if (!app.classList.contains("sidebar-collapsed") || toggle.textContent !== "展开侧栏" || toggle.attributes.title !== "展开侧栏" || store.get("arc-reader-sidebar") !== "closed") process.exit(13);
'''
    assertions = assertions.replace("收起侧栏", open_label).replace("展开侧栏", closed_label)
    result = subprocess.run(
        [node, "-e", harness + "\n" + javascript + "\n" + assertions],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_source_credit_dom_uses_text_nodes_and_exactly_once_counts(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    javascript = (
        Path(__file__).resolve().parents[1]
        / "src" / "arc_companion" / "web_assets" / "reader.js"
    ).read_text(encoding="utf-8")
    document = parse_source_input(
        html_text="""
        <article class="ltx_document">
          <h1 class="ltx_title_document">T</h1>
          <div class="ltx_authors">
            <p class="ltx_creator_author">
              <span class="ltx_personname">Original &lt;A&gt;</span>
            </p>
            <p class="ltx_affiliation">R&amp;D Institute</p>
            <p id="profile" class="ltx_role_profile">
              Profile 第一行 Second line.
            </p>
          </div>
          <p id="body">Body.</p>
        </article>
        """,
        source_id="combined-source-credit",
    )["document"]
    author_id = document["front_matter"]["author_records"][0]["source_id"]
    document["front_matter"]["author_name_variants"] = [{
        "author_id": author_id,
        "localized_name": "本地名",
        "source_identity": "source:localized-a",
    }]
    assert document["front_matter"]["block_ids"]["profiles"] == ["profile"]
    credit = normalize_source_credit(document)
    front_ids = [
        str(value)
        for key, values in document["front_matter"]["block_ids"].items()
        if key in {"title", "authors", "affiliations"}
        for value in values
    ]
    order = source_credit_placement(
        credit, front_matter_block_ids=front_ids,
    )
    snapshot = {
        "language": "zh-CN",
        "source_language": "en",
        "direction": "ltr",
        "source_direction": "ltr",
        "title": "T",
        "source_title": "T",
        "paper_id": "p",
        "status": "complete",
        "chapters": [{
            "chapter_id": "c1",
            "title": "C",
            "source_title": "C",
            "structural_only": True,
            "guide": [],
            "segments": [{
                "segment_id": "s1",
                "structural_only": True,
                "source": [{
                    "block_id": "profile",
                    "kind": "text",
                    "runs": [{"type": "text", "text": "Profile 第一行\nSecond line."}],
                    "language": "en",
                    "direction": "ltr",
                }],
                "lane_status": {},
            }],
        }],
        "appendices": [],
        "glossary": [],
        "coverage": {},
        "source_credit": credit,
        "source_credit_sha256": credit["canonical_sha256"],
        "source_credit_order": order,
        "source_credit_visible_projection": _source_credit_visible_projection(
            credit, order,
        ),
        "source_credit_front_matter_block_ids": sorted(front_ids),
        "source_credit_replaced_block_ids": ["profile"],
    }
    harness = r'''
class Classes {
  constructor(value = "") { this.values = new Set(String(value).split(/\s+/).filter(Boolean)); }
  contains(value) { return this.values.has(value); }
  add(value) { this.values.add(value); }
  remove(value) { this.values.delete(value); }
  toggle(value, force) { if (force === undefined) force = !this.values.has(value); if (force) this.values.add(value); else this.values.delete(value); }
}
class Node {
  constructor(tag = "div", id = "") { this.tagName = tag.toUpperCase(); this.id = id; this.children = []; this.dataset = {}; this.attributes = {}; this.listeners = {}; this.classList = new Classes(); this.textContent = ""; }
  set className(value) { this._className = value; this.classList = new Classes(value); }
  get className() { return this._className || ""; }
  get childNodes() { return this.children; }
  append(...values) { this.children.push(...values); }
  replaceChildren(...values) { this.children = values; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(kind, fn) { this.listeners[kind] = fn; }
  querySelectorAll() { return []; }
  scrollIntoView() {}
}
const nodes = Object.fromEntries(["reader-app", "reader-main", "chapter-sidebar", "sidebar-toggle"].map(id => [id, new Node("div", id)]));
nodes["reader-app"].classList.add("reader-shell");
global.localStorage = {getItem: () => null, setItem() {}};
global.location = {hash: ""}; global.history = {state: null, replaceState() {}};
global.requestAnimationFrame = fn => fn();
global.document = {getElementById: id => nodes[id] || null, createElement: tag => new Node(tag), createTextNode: text => ({textContent: String(text)})};
global.window = {__ARC_COMPANION_SNAPSHOT__: __PAYLOAD__, matchMedia: () => ({matches: false})};
'''.replace("__PAYLOAD__", json.dumps(snapshot, ensure_ascii=False))
    assertions = r'''
const collect = (root, cls, out = []) => { for (const child of root.children || []) { if (child && child.classList && child.classList.contains(cls)) out.push(child); if (child && child.children) collect(child, cls, out); } return out; };
const rows = collect(nodes["reader-main"], "source-credit-entry");
if (rows.length !== 3) process.exit(21);
const authors = collect(nodes["reader-main"], "source-credit-author");
if (authors.length !== 1 || authors[0].children[0].textContent !== "Original <A>" || authors[0].children[2].textContent !== "本地名") process.exit(22);
if (collect(nodes["reader-main"], "source-credit-affiliation").length !== 1 || collect(nodes["reader-main"], "source-credit-profile").length !== 1) process.exit(23);
'''
    result = subprocess.run(
        [node, "-e", harness + "\n" + javascript + "\n" + assertions],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    if shutil.which("xelatex") and shutil.which("pdftotext"):
        tex, manifest = render_companion_tex(
            document,
            [{
                "segment_id": "s1",
                "block_ids": [
                    str(item["block_id"]) for item in document["blocks"]
                ],
            }],
            {"s1": {"explanation": "", "commentary": ""}},
            output_dir=tmp_path,
            language="zh-CN",
            source_credit=credit,
        )
        tex_path = tmp_path / "combined-source-credit.tex"
        tex_path.write_text(tex, encoding="utf-8")
        compiled = subprocess.run(
            [
                shutil.which("xelatex") or "xelatex",
                "-halt-on-error",
                "-interaction=nonstopmode",
                f"-output-directory={tmp_path}",
                str(tex_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        pdf_observation = validate_pdf_source_credit_text(
            tmp_path / "combined-source-credit.pdf", document, manifest,
        )
        assert pdf_observation["canonical_sha256"] == credit["canonical_sha256"]
        assert pdf_observation["visible_projection_sha256"] == sha256_json(
            snapshot["source_credit_visible_projection"]
        )


def test_snapshot_hides_formal_chapter_heading_but_keeps_lower_headings(tmp_path: Path) -> None:
    project, checkpoint, _segment_id = _project(tmp_path)
    envelope = read_json(checkpoint / "document.json")
    envelope["document"]["blocks"] = [
        {"block_id": "paper-title", "type": "heading", "text": "A Safe Reader", "source_role": "front_matter_title"},
        {"block_id": "chapter-title", "type": "chapter", "text": "Foundations"},
        {"block_id": "section-title", "type": "section", "text": "First section"},
        *envelope["document"]["blocks"],
    ]
    write_json(checkpoint / "document.json", envelope)
    chapters = read_json(checkpoint / "chapters.json")
    chapter = chapters["chapters"][0]
    chapter["title_block_ids"] = ["chapter-title"]
    chapter["structural_block_ids"] = ["chapter-title", "section-title"]
    chapter["content_block_ids"] = ["b1", "b2"]
    chapter["block_ids"] = ["paper-title", "chapter-title", "section-title", "b1", "b2"]
    write_json(checkpoint / "chapters.json", chapters)
    segmentation = read_json(checkpoint / "chapters" / "ch-0001" / "segmentation.json")
    segmentation["segments"][0]["block_ids"] = ["paper-title", "chapter-title", "section-title", "b1"]
    write_json(checkpoint / "chapters" / "ch-0001" / "segmentation.json", segmentation)

    snapshot = build_reader_snapshot(project)

    source = snapshot["chapters"][0]["segments"][0]["source"]
    assert [item["kind"] for item in source] == ["section", "paragraph"]
    assert all(item.get("title") != "Foundations" for item in source)
    assert all("A Safe Reader" not in str(item) for item in source)


def test_snapshot_exposes_bilingual_titles_and_language_direction(tmp_path: Path) -> None:
    project, checkpoint, _segment_id = _project(tmp_path)
    envelope = read_json(checkpoint / "document.json")
    envelope["document"]["front_matter"] = {
        "title": "Die Relativitätstheorie",
        "block_ids": {"title": ["paper-title"]},
    }
    envelope["document"]["blocks"] = [
        {"block_id": "paper-title", "type": "heading", "text": "Die Relativitätstheorie", "source_role": "front_matter_title"},
        {"block_id": "chapter-title", "type": "chapter", "text": "Grundlagen"},
        {"block_id": "section-title", "type": "section", "text": "Kinematik"},
        *envelope["document"]["blocks"],
        {"block_id": "index-title", "type": "section", "text": "Register", "source_role": "index"},
    ]
    envelope["metadata"]["title"] = "Die Relativitätstheorie"
    write_json(checkpoint / "document.json", envelope)
    chapters = read_json(checkpoint / "chapters.json")
    chapter = chapters["chapters"][0]
    chapter.update({
        "title": "Grundlagen",
        "title_block_ids": ["chapter-title"],
        "block_ids": ["paper-title", "chapter-title", "section-title", "b1", "b2"],
    })
    write_json(checkpoint / "chapters.json", chapters)
    segmentation = read_json(checkpoint / "chapters" / "ch-0001" / "segmentation.json")
    segmentation["segments"][0]["block_ids"] = ["paper-title", "chapter-title", "section-title", "b1"]
    write_json(checkpoint / "chapters" / "ch-0001" / "segmentation.json", segmentation)
    title_translations = {
        "schema_version": "arc.companion.title-translations.v1",
        "source_language": "de", "target_language": "zh-CN", "source_sha256": "0" * 64,
        "titles": [
            {"title_id": "document:title", "role": "document_title", "block_id": "paper-title", "text": "相对论"},
            {"title_id": "block:chapter-title", "role": "chapter", "block_id": "chapter-title", "chapter_id": "ch-0001", "text": "基础"},
            {"title_id": "block:section-title", "role": "section", "block_id": "section-title", "chapter_id": "ch-0001", "text": "运动学"},
            {"title_id": "block:index-title", "role": "index", "block_id": "index-title", "text": "索引"},
        ],
    }

    snapshot = build_reader_snapshot(project, final_overrides={
        "source_language": "de-DE",
        "language": "zh_cn",
        "title_translations": title_translations,
    })

    assert snapshot["title"] == "相对论"
    assert snapshot["source_title"] == "Die Relativitätstheorie"
    assert snapshot["translated_title"] == "相对论"
    assert snapshot["source_language"] == "de-DE"
    assert snapshot["language"] == "zh-CN"
    assert snapshot["source_direction"] == "ltr"
    assert snapshot["direction"] == "ltr"
    rendered_chapter = snapshot["chapters"][0]
    assert rendered_chapter["title"] == "基础"
    assert rendered_chapter["source_title"] == "Grundlagen"
    section = rendered_chapter["segments"][0]["source"][0]
    assert section["source_title"] == "Kinematik"
    assert section["translated_title"] == "运动学"
    assert section["language"] == "de-DE"
    assert snapshot["appendices"][-1] == {
        "appendix_id": "source-heading-index-title",
        "kind": "source_only_structural_heading",
        "title": "索引",
        "source_title": "Register",
        "translated_title": "索引",
        "source": [],
    }


def test_index_html_uses_target_language_and_direction() -> None:
    import arc_companion.web as web

    html = web._index_html(
        data_script="data/snapshot.js", asset_root="assets/hash",
        title="النسبية", language="ar",
    )
    assert '<html lang="ar" dir="rtl">' in html
    assert "<title>النسبية</title>" in html


def test_guide_view_omits_empty_guides_but_keeps_supplementary_reading() -> None:
    import arc_companion.web as web

    assert web._guide_view({
        "motivation": None, "main_content": None, "section_logic": None,
        "prerequisites": None, "pedagogical_comparison": None,
        "historical_context": [], "supplementary_reading": [],
    }) == []
    view = web._guide_view({
        "supplementary_reading": [{"title": "Text", "reason": "A useful derivation"}],
    })
    assert view == [{
        "kind": "supplementary_reading",
        "runs": [{"type": "text", "text": "Text: A useful derivation"}],
    }]


def test_svg_sanitizer_removes_active_and_external_content() -> None:
    import arc_companion.web as web

    unsafe = """<svg xmlns="http://www.w3.org/2000/svg">
      <style>.x { fill: red }</style>
      <script>alert(1)</script>
      <foreignObject><p>HTML</p></foreignObject>
      <object data="https://example.test/object"></object>
      <embed src="https://example.test/embed" />
      <iframe src="https://example.test/frame"></iframe>
      <a href="https://example.test/"><rect style="fill: blue" onclick="go()" /></a>
      <use xlink:href="javascript:alert(1)" />
      <use href="#safe-shape" />
      <rect fill="url(https://example.test/paint)" />
      <rect filter="url(#safe-filter)" />
    </svg>"""

    rendered = web._safe_svg(unsafe)

    for forbidden in (
        "<style", "<script", "foreignObject", "<object", "<embed", "<iframe",
        "https://", "javascript:", " style=", " onclick=",
    ):
        assert forbidden not in rendered
    assert 'href="#safe-shape"' in rendered
    assert 'filter="url(#safe-filter)"' in rendered
    assert " fill=" not in rendered


def test_unsafe_checkpoint_and_urls_are_not_exposed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    write_json(
        project / "state.json",
        {
            "schema_version": "arc.companion.state.v2",
            "status": "active",
            "checkpoint_dir": str(outside),
        },
    )
    with pytest.raises(WebReaderError, match="escapes companion project"):
        build_reader_snapshot(project)

    project, checkpoint, segment_id = _project(tmp_path / "safe")
    annotation_path = checkpoint / "annotations" / f"{_segment_name(segment_id)}.json"
    envelope = read_json(annotation_path)
    envelope["annotation"]["commentary_sources"][0]["url"] = "javascript:alert(1)"
    write_json(annotation_path, envelope)
    ledger_path = checkpoint / "chapters/ch-0001/companion-ledger.json"
    ledger = read_json(ledger_path)
    ledger["blocks"][0]["output_sha256"] = sha256_json(envelope["annotation"])
    write_json(ledger_path, ledger)
    snapshot = build_reader_snapshot(project)
    assert snapshot["chapters"][0]["segments"][0]["companion"]["sections"][0][
        "sources"
    ] == []
    assert "javascript:" not in json.dumps(snapshot)


def test_validation_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    project, _checkpoint, _segment_id = _project(tmp_path)
    result = publish_reader(project)
    manifest_path = Path(result["web_manifest_path"])
    manifest = read_json(manifest_path)
    manifest["assets"][0]["path"] = "../outside.js"
    write_json(manifest_path, manifest)

    with pytest.raises(WebReaderError, match="unsafe path"):
        validate_reader_project(project)
