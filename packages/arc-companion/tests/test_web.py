from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from arc_companion.io import read_json, sha256_json, write_json
from arc_companion.web import (
    READER_FINAL_VERSION,
    READER_SNAPSHOT_VERSION,
    WEB_MANIFEST_VERSION,
    WEB_RENDER_VERSION,
    WebReaderError,
    build_reader_snapshot,
    publish_reader,
    validate_reader_project,
)


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
            "schema_version": "arc.companion.chapter-guide.v2",
            "chapter_id": "ch-0001",
            "motivation": "Why $E$ matters.",
            "main_content": None,
            "section_logic": None,
            "book_position": None,
            "prerequisites": None,
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
