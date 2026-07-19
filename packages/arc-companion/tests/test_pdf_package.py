from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from arc_companion import pdf as pdf_module
from arc_companion.package import package_project
from arc_companion.pdf import PDFError, validate_pdf


PDFINFO = """Title: fixture
Pages:          3
Encrypted:      no
"""
FONT_REPORT = """name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
ABCDEE+FandolSong-Regular            CID TrueType      Identity-H       yes yes yes      8  0
XYZZY+LatinModernRoman               Type 1C           Custom           yes yes yes      9  0
"""


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
    assert report["embedded_font_count"] == 2
    assert len(report["render_paths"]) == 3
    render_calls = [call for call in calls if Path(call[0]).name == "pdftoppm"]
    assert [call[call.index("-f") + 1] for call in render_calls] == ["1", "2", "3"]
    assert all(Path(path).is_file() for path in report["render_paths"])


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
