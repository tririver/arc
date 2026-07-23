from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from arc_companion import pdf as pdf_module
from arc_companion.io import canonical_json, sha256_file, sha256_json
from arc_companion.package import _WEB_STATE_KEYS, package_project
from arc_companion.pdf import (
    PDFError,
    compile_latex,
    managed_run_root_pdf_path,
    normalize_run_root_pdf_state,
    publish_run_root_pdf,
    validate_pdf,
)
from arc_companion.web import (
    READER_SNAPSHOT_VERSION,
    WEB_MANIFEST_VERSION,
    WEB_RENDER_VERSION,
)
from arc_companion.source_credit import normalize_source_credit


PDFINFO = """Title: fixture
Pages:          3
Encrypted:      no
"""
FONT_REPORT = """name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
ABCDEE+FandolSong-Regular            CID TrueType      Identity-H       yes yes yes      8  0
XYZZY+LatinModernRoman               Type 1C           Custom           yes yes yes      9  0
QWERTY+NotoSansCJKSC                 CID TrueType      Identity-H       yes yes yes     10  0
"""


def test_publish_run_root_pdf_creates_stable_byte_identical_copy(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / ".arc-companion" / "renders" / "pdf" / "rev" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF stable")

    published = publish_run_root_pdf(source, project)

    target = project / "paper.pdf"
    assert published == {
        "output_run_pdf": str(target),
        "output_run_pdf_sha256": sha256_file(source),
    }
    assert target.read_bytes() == source.read_bytes()
    assert managed_run_root_pdf_path({"published": {"pdf": published}}) == target
    assert not (project.parent / target.name).exists()


def test_publish_run_root_pdf_resolves_exact_root_and_never_writes_parent(
    tmp_path: Path,
) -> None:
    resolved_root = tmp_path / "runs" / "companion"
    source = (
        resolved_root / ".arc-companion" / "renders" / "pdf" / "rev" / "paper.pdf"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF stable")
    linked_root = tmp_path / "companion-link"
    linked_root.symlink_to(resolved_root, target_is_directory=True)

    published = publish_run_root_pdf(source, linked_root)

    target = resolved_root.resolve() / "paper.pdf"
    assert published["output_run_pdf"] == str(target)
    assert target.read_bytes() == source.read_bytes()
    assert not (resolved_root.parent / target.name).exists()
    assert not (tmp_path / target.name).exists()


def test_publish_run_root_pdf_refuses_unmanaged_conflict_and_repairs_managed_copy(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / ".arc-companion" / "renders" / "pdf" / "rev" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF new")
    target = project / "paper.pdf"
    target.write_bytes(b"user file")

    with pytest.raises(PDFError, match="unmanaged run-root delivery PDF"):
        publish_run_root_pdf(source, project)
    assert target.read_bytes() == b"user file"

    published = publish_run_root_pdf(source, project, managed_path=target)

    assert target.read_bytes() == source.read_bytes()
    assert published["output_run_pdf_sha256"] == sha256_file(source)


def test_publish_run_root_pdf_adopts_byte_identical_unmanaged_file(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / ".arc-companion" / "renders" / "pdf" / "rev" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF same")
    target = project / "paper.pdf"
    target.write_bytes(source.read_bytes())

    published = publish_run_root_pdf(source, project)

    assert target.read_bytes() == b"%PDF same"
    assert published["output_run_pdf"] == str(target)
    assert published["output_run_pdf_sha256"] == sha256_file(target)


def test_publish_run_root_pdf_does_not_adopt_unmanaged_symlink(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / ".arc-companion" / "renders" / "pdf" / "rev" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF stable")
    external = tmp_path / "external.pdf"
    external.write_bytes(source.read_bytes())
    target = project / "paper.pdf"
    target.symlink_to(external)

    with pytest.raises(PDFError, match="unmanaged run-root delivery PDF"):
        publish_run_root_pdf(source, project)

    assert target.is_symlink()
    assert external.read_bytes() == b"%PDF stable"


def test_publish_run_root_pdf_replace_failure_preserves_managed_copy(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "project"
    source = project / ".arc-companion" / "renders" / "pdf" / "rev" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF new")
    target = project / "paper.pdf"
    target.write_bytes(b"%PDF old")
    monkeypatch.setattr(
        pdf_module,
        "_publish_run_root_pdf_replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        publish_run_root_pdf(source, project, managed_path=target)

    assert target.read_bytes() == b"%PDF old"
    assert not list(project.glob("*.arc-companion-delivery-*.tmp"))


def test_publish_run_root_pdf_does_not_overwrite_racing_unmanaged_file(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "project"
    source = project / ".arc-companion" / "renders" / "pdf" / "rev" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF source")
    target = project / "paper.pdf"

    def race_create(_source: Path, destination: Path) -> None:
        destination.write_bytes(b"user race")
        raise FileExistsError(destination)

    monkeypatch.setattr(pdf_module, "_publish_run_root_pdf_create", race_create)
    with pytest.raises(PDFError, match="unmanaged run-root delivery PDF"):
        publish_run_root_pdf(source, project)

    assert target.read_bytes() == b"user race"
    assert not list(project.glob("*.arc-companion-delivery-*.tmp"))


def test_early_project_pdf_draft_fields_migrate_to_run_root_contract(
    tmp_path: Path,
) -> None:
    delivery = tmp_path / "paper.pdf"
    legacy = {
        "project_pdf_managed_path": str(delivery),
        "output_project_pdf": str(delivery),
        "output_project_pdf_sha256": "a" * 64,
        "published": {
            "pdf": {
                "output_project_pdf": str(delivery),
                "output_project_pdf_sha256": "a" * 64,
            }
        },
    }

    normalized = normalize_run_root_pdf_state(legacy)

    assert normalized["run_pdf_managed_path"] == str(delivery)
    assert normalized["output_run_pdf"] == str(delivery)
    assert normalized["output_run_pdf_sha256"] == "a" * 64
    assert normalized["published"]["pdf"]["output_run_pdf"] == str(delivery)
    assert normalized["published"]["pdf"]["output_run_pdf_sha256"] == "a" * 64
    assert managed_run_root_pdf_path(legacy) == delivery
    assert "project_pdf_managed_path" not in normalized
    assert "output_project_pdf" not in normalized["published"]["pdf"]


def _pdf_runner(tmp_path: Path, *, info: str = PDFINFO, fonts: str = FONT_REPORT):
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        name = Path(command[0]).name
        if name == "pdfinfo":
            stdout = info
        elif name == "pdffonts":
            stdout = fonts
        elif name == "pdftotext":
            Path(command[-1]).write_text("searchable text", encoding="utf-8")
            stdout = ""
        elif name == "pdftoppm":
            Path(command[-1] + ".png").write_bytes(b"png")
            stdout = ""
        else:  # pragma: no cover - protects the fake itself
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    return runner, calls


def test_validate_pdf_checks_fonts_and_renders_every_page(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF fixture")
    monkeypatch.setattr(pdf_module.shutil, "which", lambda name: f"/tools/{name}")
    runner, calls = _pdf_runner(tmp_path)

    report = validate_pdf(source, runner=runner)

    assert report["pages"] == 3
    assert report["encrypted"] is False
    assert report["embedded_font_count"] == 3
    assert report["font_roles"]["sans"] == ["NotoSansCJKSC"]
    assert len(report["render_paths"]) == 3
    render_calls = [call for call in calls if Path(call[0]).name == "pdftoppm"]
    assert [call[call.index("-f") + 1] for call in render_calls] == ["1", "2", "3"]
    assert all(Path(path).is_file() for path in report["render_paths"])
    assert all(call[call.index("-r") + 1] == "144" for call in render_calls)


def test_validate_pdf_rejects_removed_visible_labels(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF fixture")
    monkeypatch.setattr(pdf_module.shutil, "which", lambda name: f"/tools/{name}")
    runner, _ = _pdf_runner(tmp_path)
    def labeled_runner(command, **kwargs):
        result = runner(command, **kwargs)
        if Path(command[0]).name == "pdftotext":
            Path(command[-1]).write_text("原文\n译文\n", encoding="utf-8")
        return result
    with pytest.raises(PDFError, match="visible layer labels"):
        validate_pdf(source, runner=labeled_runner)


@pytest.mark.parametrize(
    "text",
    [
        "伴读语言: zh-CN\n可检索正文。",
        "本文讨论译文的准确性以及伴读材料的作用。",
        "正文按“原文—译文—伴读”的顺序编排。",
        "这里不是本段解释，而是对该术语的普通引用。",
    ],
)
def test_validate_pdf_allows_layer_words_in_metadata_and_prose(
    tmp_path: Path, monkeypatch, text: str,
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF fixture")
    monkeypatch.setattr(pdf_module.shutil, "which", lambda name: f"/tools/{name}")
    runner, _ = _pdf_runner(tmp_path)

    def prose_runner(command, **kwargs):
        result = runner(command, **kwargs)
        if Path(command[0]).name == "pdftotext":
            Path(command[-1]).write_text(text, encoding="utf-8")
        return result

    report = validate_pdf(source, runner=prose_runner)

    assert report["pages"] == 3


@pytest.mark.parametrize("label", ["译 文：", "【伴读】", "## 本段解释", "— 译文 —"])
def test_validate_pdf_rejects_decorated_standalone_layer_labels(
    tmp_path: Path, monkeypatch, label: str,
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF fixture")
    monkeypatch.setattr(pdf_module.shutil, "which", lambda name: f"/tools/{name}")
    runner, _ = _pdf_runner(tmp_path)

    def labeled_runner(command, **kwargs):
        result = runner(command, **kwargs)
        if Path(command[0]).name == "pdftotext":
            Path(command[-1]).write_text(f"正文\n{label}\n", encoding="utf-8")
        return result

    with pytest.raises(PDFError, match="visible layer labels"):
        validate_pdf(source, runner=labeled_runner)


def test_compile_latex_uses_non_hidden_unique_jobname_and_cleans_sidecars(
    tmp_path: Path, monkeypatch,
) -> None:
    tex_path = tmp_path / ".hidden-building.tex"
    tex_path.write_text("fixture", encoding="utf-8")
    final_pdf = tmp_path / "paper.pdf"
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(pdf_module.shutil, "which", lambda name: f"/tools/{name}")

    def runner(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        calls.append((command, cwd))
        jobname = next(value.split("=", 1)[1] for value in command if value.startswith("-jobname="))
        assert not jobname.startswith(".")
        assert "/" not in jobname
        for suffix, content in (
            (".pdf", b"%PDF fixture"), (".aux", b"aux"), (".fdb_latexmk", b"fdb"),
            (".fls", b"fls"), (".log", b"log"), (".xdv", b"xdv"),
        ):
            (cwd / f"{jobname}{suffix}").write_bytes(content)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(pdf_module.subprocess, "run", runner)
    compile_latex(tex_path, final_pdf)

    assert final_pdf.read_bytes() == b"%PDF fixture"
    assert calls[0][1] == tmp_path
    command = calls[0][0]
    assert tex_path.name == command[-1]
    assert not list(tmp_path.glob("arc-companion-hidden-building-*.*"))


def test_compile_latex_cleans_unique_sidecars_after_failure(tmp_path: Path, monkeypatch) -> None:
    tex_path = tmp_path / "building.tex"
    tex_path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(pdf_module.shutil, "which", lambda name: f"/tools/{name}")

    def runner(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        jobname = next(value.split("=", 1)[1] for value in command if value.startswith("-jobname="))
        (cwd / f"{jobname}.aux").write_bytes(b"aux")
        (cwd / f"{jobname}.log").write_text(
            "Missing character: harmless warning\n"
            "context before\n"
            "! Missing \\endcsname inserted.\n"
            "<to be read again>\n"
            "                   \\TU\\textasciicircum\n"
            "l.4878 [\\textasciicircum{}1]\n",
            encoding="utf-8",
        )
        warning_tail = "\n".join(f"warning {index}" for index in range(40))
        return subprocess.CompletedProcess(command, 1, stdout=warning_tail, stderr="")

    monkeypatch.setattr(pdf_module.subprocess, "run", runner)
    with pytest.raises(PDFError) as caught:
        compile_latex(tex_path, tmp_path / "paper.pdf")
    message = str(caught.value)
    assert "First XeLaTeX error:" in message
    assert "! Missing \\endcsname inserted." in message
    assert "l.4878" in message
    assert "warning 39" in message
    assert not list(tmp_path.glob("arc-companion-building-*.*"))


@pytest.mark.parametrize(
    ("info", "fonts", "message"),
    [
        ("Pages: 2\nEncrypted: yes (print:yes)\n", FONT_REPORT, "encrypted"),
        ("Encrypted: no\n", FONT_REPORT, "page count"),
        (
            PDFINFO,
            FONT_REPORT.replace("yes yes yes      8", "no  yes yes      8"),
            "not embedded",
        ),
    ],
)
def test_validate_pdf_rejects_invalid_metadata_or_fonts(
    tmp_path: Path, monkeypatch, info: str, fonts: str, message: str
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF fixture")
    monkeypatch.setattr(pdf_module.shutil, "which", lambda name: f"/tools/{name}")
    runner, _ = _pdf_runner(tmp_path, info=info, fonts=fonts)
    with pytest.raises(PDFError, match=message):
        validate_pdf(source, runner=runner)


def _complete_project(root: Path, *, asset_path: Path | None = None) -> None:
    pdf = root / "deliverables" / "paper.pdf"
    tex = root / "deliverables" / "paper.tex"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF fixture")
    tex.write_text(r"\includegraphics{assets/figure.png}", encoding="utf-8")
    assets = []
    if asset_path is not None:
        assets.append(
            {
                "output_path": str(asset_path),
                "output_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            }
        )
    (root / "source-manifest.json").write_text(json.dumps({"assets": assets}), encoding="utf-8")
    (root / "validation.json").write_text('{"ok":true}', encoding="utf-8")
    (root / "state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "paper_id": "arXiv:1",
                "fingerprint": "abc",
                "output_pdf": str(pdf),
                "output_tex": "deliverables/paper.tex",
            }
        ),
        encoding="utf-8",
    )


def test_package_preserves_tex_asset_paths_and_verifies_members(tmp_path: Path) -> None:
    asset = tmp_path / "assets" / "nested" / "figure.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"figure")
    _complete_project(tmp_path, asset_path=asset)

    result = package_project(tmp_path)

    assert result["ok"] is True
    archive = Path(result["data"]["archive_path"])
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        assert "deliverables/paper.tex" in names
        assert "deliverables/paper.pdf" in names
        assert "assets/nested/figure.png" in names
        assert "package-manifest.json" in names
        manifest = json.loads(handle.read("package-manifest.json"))
        listed = {item["path"] for item in manifest["files"]}
        assert "assets/nested/figure.png" in listed
        assert handle.testzip() is None


def test_package_rejects_state_path_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _complete_project(project)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF fixture")
    state = json.loads((project / "state.json").read_text(encoding="utf-8"))
    state["output_pdf"] = str(outside)
    (project / "state.json").write_text(json.dumps(state), encoding="utf-8")

    result = package_project(project)

    assert result["ok"] is False
    assert "escapes companion project" in result["errors"][0]["message"]


def test_package_rejects_asset_path_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"figure")
    _complete_project(project, asset_path=outside)

    result = package_project(project)

    assert result["ok"] is False
    assert "escapes companion project" in result["errors"][0]["message"]


def _add_web_reader(root: Path) -> dict[str, Path]:
    reader = root / "reader"
    data = reader / "data"
    asset_root = reader / "assets" / f"builtin-{'a' * 64}"
    assets = asset_root / "katex" / "fonts"
    data.mkdir(parents=True)
    assets.mkdir(parents=True)
    source_credit = normalize_source_credit(
        {"front_matter": {}, "blocks": []}
    )
    source_credit_manifest = {
        "schema_version": source_credit["schema_version"],
        "canonical_sha256": source_credit["canonical_sha256"],
        "front_matter_block_ids": [],
        "replaced_block_ids": [],
        "ordered_items": [],
        "placements": [],
        "visible_counts": {
            "authors": 0, "affiliations": 0, "profiles": 0,
        },
    }
    snapshot = {
        "schema_version": READER_SNAPSHOT_VERSION,
        "translation_mode": "enabled",
        "glossary": [],
        "chapters": [],
        "appendices": [],
        "authors": [],
        "source_credit": source_credit,
        "source_credit_sha256": source_credit["canonical_sha256"],
        "source_credit_order": [],
        "source_credit_front_matter_block_ids": [],
        "source_credit_replaced_block_ids": [],
        "coverage": {
            "chapter_ids": [], "segment_ids": [],
            "translation_segment_ids": [], "annotation_segment_ids": [],
        },
    }
    snapshot["revision"] = sha256_json(snapshot)
    snapshot_path = _write_content_addressed_json(data, "snapshot", snapshot)
    data_text = "window.__ARC_COMPANION_SNAPSHOT__ = " + canonical_json(snapshot) + ";\n"
    data_hash = hashlib.sha256(data_text.encode("utf-8")).hexdigest()
    paths = {
        "index": reader / "index.html",
        "snapshot": snapshot_path,
        "data_script": data / f"snapshot-{data_hash}.js",
        "reader_css": asset_root / "reader.css",
        "reader_js": asset_root / "reader.js",
        "katex_css": asset_root / "katex" / "katex.min.css",
        "katex_js": asset_root / "katex" / "katex.min.js",
        "asset": assets / "KaTeX_Main-Regular.woff2",
    }
    asset_relative = asset_root.relative_to(reader).as_posix()
    paths["index"].write_text(
        f"""<html><head>
<link href="{asset_relative}/reader.css"><link href="{asset_relative}/katex/katex.min.css">
<script src="{asset_relative}/katex/katex.min.js"></script>
<script src="data/{paths['data_script'].name}"></script>
<script src="{asset_relative}/reader.js"></script>
</head></html>""",
        encoding="utf-8",
    )
    paths["data_script"].write_text(data_text, encoding="utf-8")
    paths["reader_css"].write_text("body{}", encoding="utf-8")
    paths["reader_js"].write_text("void 0;", encoding="utf-8")
    paths["katex_css"].write_text(".katex{}", encoding="utf-8")
    paths["katex_js"].write_text("window.katex={};", encoding="utf-8")
    paths["asset"].write_bytes(b"font")

    def record(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "schema_version": WEB_MANIFEST_VERSION,
        "web_render_version": WEB_RENDER_VERSION,
        "index": record(paths["index"]),
        "snapshot": record(paths["snapshot"]),
        "data_script": record(paths["data_script"]),
        "assets": [
            record(paths[key]) for key in (
                "reader_css", "reader_js", "katex_css", "katex_js", "asset"
            )
        ],
        "coverage": {
            "chapter_ids": [], "segment_ids": [],
            "translation_segment_ids": [], "annotation_segment_ids": [],
        },
        "source_credit": source_credit_manifest,
    }
    manifest_path = _write_content_addressed_json(data, "manifest", manifest)
    paths["manifest"] = manifest_path
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "output_html": str(paths["index"]),
        "output_html_sha256": record(paths["index"])["sha256"],
        "reader_snapshot_path": str(paths["snapshot"]),
        "reader_snapshot_sha256": record(paths["snapshot"])["sha256"],
        "web_manifest_path": str(manifest_path),
        "web_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "web_render_version": WEB_RENDER_VERSION,
    })
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return paths


def _write_content_addressed_json(directory: Path, prefix: str, value: object) -> Path:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")
    path = directory / f"{prefix}-{hashlib.sha256(payload).hexdigest()}.json"
    path.write_bytes(payload)
    return path


def _replace_web_manifest(
    root: Path, manifest: dict[str, object], *, state_updates: dict[str, object] | None = None,
) -> Path:
    manifest_path = _write_content_addressed_json(root / "reader" / "data", "manifest", manifest)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(state_updates or {})
    state["web_manifest_path"] = str(manifest_path)
    state["web_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return manifest_path


def test_package_collects_every_web_manifest_file(tmp_path: Path) -> None:
    _complete_project(tmp_path)
    paths = _add_web_reader(tmp_path)

    result = package_project(tmp_path)

    assert result["ok"] is True
    with zipfile.ZipFile(result["data"]["archive_path"]) as handle:
        names = set(handle.namelist())
    assert {path.relative_to(tmp_path).as_posix() for path in paths.values()} <= names
    assert any(name.startswith("reader/data/manifest-") for name in names)


def test_package_rejects_web_asset_hash_mismatch(tmp_path: Path) -> None:
    _complete_project(tmp_path)
    paths = _add_web_reader(tmp_path)
    paths["asset"].write_bytes(b"tampered")

    result = package_project(tmp_path)

    assert result["ok"] is False
    assert "hash mismatch" in result["errors"][0]["message"]


def test_package_rejects_web_manifest_path_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _complete_project(project)
    _add_web_reader(project)
    outside = tmp_path / "outside.css"
    outside.write_text("body{}", encoding="utf-8")
    state = json.loads((project / "state.json").read_text(encoding="utf-8"))
    manifest_path = Path(state["web_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"].append({
        "path": str(outside),
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "bytes": outside.stat().st_size,
    })
    _replace_web_manifest(project, manifest)

    result = package_project(project)

    assert result["ok"] is False
    assert "unsafe path" in result["errors"][0]["message"]


def test_package_rejects_semantically_incoherent_web_bundle(tmp_path: Path) -> None:
    _complete_project(tmp_path)
    paths = _add_web_reader(tmp_path)
    snapshot = json.loads(paths["snapshot"].read_text(encoding="utf-8"))
    snapshot["coverage"]["segment_ids"] = ["missing-segment"]
    snapshot["revision"] = sha256_json({
        key: value for key, value in snapshot.items() if key != "revision"
    })
    snapshot_path = _write_content_addressed_json(
        tmp_path / "reader" / "data", "snapshot", snapshot
    )

    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest_path = Path(state["web_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot"]["path"] = snapshot_path.relative_to(tmp_path).as_posix()
    manifest["snapshot"]["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    manifest["snapshot"]["bytes"] = snapshot_path.stat().st_size
    manifest["coverage"] = snapshot["coverage"]
    _replace_web_manifest(tmp_path, manifest, state_updates={
        "reader_snapshot_path": str(snapshot_path),
        "reader_snapshot_sha256": manifest["snapshot"]["sha256"],
    })

    result = package_project(tmp_path)

    assert result["ok"] is False
    assert "reader chapter content differs" in result["errors"][0]["message"]


@pytest.mark.parametrize("field", ["manifest", "render"])
def test_package_state_v2_requires_current_web_versions(tmp_path: Path, field: str) -> None:
    _complete_project(tmp_path)
    _add_web_reader(tmp_path)
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["schema_version"] = "arc.companion.state.v2"
    if field == "render":
        state["web_render_version"] = "arc.companion.web-render.v999"
    else:
        manifest_path = Path(state["web_manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "arc.companion.web-manifest.v999"
        _replace_web_manifest(tmp_path, manifest, state_updates={
            "schema_version": "arc.companion.state.v2",
        })
    if field == "render":
        state_path.write_text(json.dumps(state), encoding="utf-8")

    result = package_project(tmp_path)

    assert result["ok"] is False
    assert "current" in result["errors"][0]["message"] or "schema is invalid" in result["errors"][0]["message"]


def test_package_requires_complete_web_contract_for_state_v2(tmp_path: Path) -> None:
    _complete_project(tmp_path)
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["schema_version"] = "arc.companion.state.v2"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = package_project(tmp_path)

    assert result["ok"] is False
    assert "State v2 is missing the required web reader contract" in result["errors"][0]["message"]


def test_package_keeps_pdf_only_compatibility_for_legacy_state(tmp_path: Path) -> None:
    _complete_project(tmp_path)
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["schema_version"] = "arc.companion.state.v1"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = package_project(tmp_path)

    assert result["ok"] is True


def test_package_uses_published_last_good_while_active_run_failed(tmp_path: Path) -> None:
    _complete_project(tmp_path)
    _add_web_reader(tmp_path)
    state_path = tmp_path / "state.json"
    flat = json.loads(state_path.read_text(encoding="utf-8"))
    pdf = Path(flat["output_pdf"])
    tex = tmp_path / "deliverables" / "paper.tex"
    source_manifest = tmp_path / "source-manifest.json"
    validation = tmp_path / "validation.json"
    published_pdf = {
        "output_pdf": str(pdf),
        "output_pdf_sha256": sha256_file(pdf),
        "output_tex": str(tex),
        "output_tex_sha256": sha256_file(tex),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "validation_path": str(validation),
        "validation_sha256": sha256_file(validation),
    }
    web_keys = {
        key: flat[key] for key in _WEB_STATE_KEYS
    }
    active = {
        key: value for key, value in flat.items()
        if key not in {
            "output_pdf", "output_tex", "source_manifest_path", "validation_path",
            *_WEB_STATE_KEYS,
        }
    }
    active.update({
        "schema_version": "arc.companion.state.v3",
        "status": "failed",
        "fingerprint": "new-active-fingerprint",
        "checkpoint_dir": str(tmp_path / "missing-active-checkpoint"),
        "published": {"content_sha256": "a" * 64, "pdf": published_pdf, "web": web_keys},
    })
    state_path.write_text(json.dumps(active), encoding="utf-8")

    result = package_project(tmp_path)

    assert result["ok"] is True
    with zipfile.ZipFile(result["data"]["archive_path"]) as handle:
        names = set(handle.namelist())
    assert "deliverables/paper.pdf" in names
    assert "reader/index.html" in names
