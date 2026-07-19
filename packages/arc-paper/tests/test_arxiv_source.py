from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from arc_paper import cli, service
from arc_paper.cache import CachePaths
from arc_paper.providers import arxiv_source as source_module
from arc_paper.providers.arxiv_source import ArxivSourceProvider, unpack_source_archive


def _tar_bytes(files: dict[str, bytes], *, link: tuple[str, str, bytes] | None = None) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if link is not None:
            name, target, kind = link
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = target
            archive.addfile(info)
    return output.getvalue()


def test_explicit_source_cache_records_version_license_hash_manifest_and_main_tex(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    payload = _tar_bytes({
        "sections/body.tex": b"body",
        "main.tex": b"\\documentclass{article}\n\\begin{document}Hello\\end{document}",
        "figure.pdf": b"pdf",
    })
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=payload, headers={"x-arxiv-license": "https://arxiv.org/licenses/nonexclusive-distrib/1.0/"})

    provider = ArxivSourceProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(service, "_arxiv_source", provider)

    first = service.cache_arxiv_source("0911.3380", version=2)
    second = service.cache_arxiv_source("0911.3380", version=2)
    probe = service.probe_arxiv_source("0911.3380", version=2)

    assert first["ok"] and second["ok"] and probe["ok"]
    manifest = first["data"]
    assert calls == ["https://export.arxiv.org/e-print/0911.3380v2"]
    assert manifest["versioned_id"] == "arXiv:0911.3380v2"
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["bytes"] == len(payload)
    assert manifest["license"].startswith("https://arxiv.org/licenses/")
    assert manifest["main_tex"] == "main.tex"
    assert {item["path"] for item in manifest["files"]} == {
        "figure.pdf", "main.tex", "sections/body.tex"
    }
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    assert "no TeX is executed" in manifest["execution_policy"]
    assert json.loads(CachePaths.for_paper("0911.3380").arxiv_source_manifest(2).read_text())["sha256"] == manifest["sha256"]


@pytest.mark.parametrize(
    "payload",
    [
        _tar_bytes({"../escape.tex": b"bad"}),
        _tar_bytes({"/absolute.tex": b"bad"}),
        _tar_bytes({}, link=("link.tex", "main.tex", tarfile.SYMTYPE)),
        _tar_bytes({}, link=("hard.tex", "main.tex", tarfile.LNKTYPE)),
    ],
)
def test_unpack_rejects_traversal_and_links(payload, tmp_path):
    with pytest.raises(ValueError):
        unpack_source_archive(payload, destination=tmp_path / "files")


def test_unpack_rejects_device_entry(tmp_path):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        info = tarfile.TarInfo("device")
        info.type = tarfile.CHRTYPE
        archive.addfile(info)
    with pytest.raises(ValueError, match="unsupported entry"):
        unpack_source_archive(output.getvalue(), destination=tmp_path / "files")


def test_unpack_rejects_expansion_and_file_count_limits(monkeypatch, tmp_path):
    payload = _tar_bytes({"a.tex": b"a", "b.tex": b"b"})
    monkeypatch.setattr(source_module, "MAX_FILES", 1)
    with pytest.raises(ValueError, match="more than 1 files"):
        unpack_source_archive(payload, destination=tmp_path / "files")

    monkeypatch.setattr(source_module, "MAX_FILES", 10)
    monkeypatch.setattr(source_module, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="expanded source"):
        unpack_source_archive(payload, destination=tmp_path / "files-2")

    monkeypatch.setattr(source_module, "MAX_TOTAL_BYTES", 10_000)
    monkeypatch.setattr(source_module, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="source file exceeds"):
        unpack_source_archive(_tar_bytes({"large.tex": b"xx"}), destination=tmp_path / "files-3")

    bomb = _tar_bytes({"repeated.tex": b"x" * 10_000})
    monkeypatch.setattr(source_module, "MAX_FILE_BYTES", 100_000)
    monkeypatch.setattr(source_module, "MAX_EXPANSION_RATIO", 1)
    with pytest.raises(ValueError, match="expansion ratio"):
        unpack_source_archive(bomb, destination=tmp_path / "files-4")


def test_source_probe_is_read_only_and_missing_cache_does_not_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))

    class NoNetwork:
        def cache_source(self, *args, **kwargs):
            raise AssertionError("probe must not fetch")

        def probe_source(self, paper_id, *, version):
            return None

    monkeypatch.setattr(service, "_arxiv_source", NoNetwork())
    result = service.probe_arxiv_source("0911.3380", version=1)
    assert result["ok"] is False
    assert result["error"]["code"] == "arxiv_source_not_cached"


def test_ordinary_parse_never_calls_arxiv_source_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))

    class NoSourceDownload:
        def cache_source(self, *args, **kwargs):
            raise AssertionError("ordinary parse must not download author source")

    class CachedAr5iv:
        def get_html(self, paper_id, *, refresh=False):
            return "<article><section id='S1'><h2>Intro</h2><p>Text.</p></section></article>"

    monkeypatch.setattr(service, "_arxiv_source", NoSourceDownload())
    monkeypatch.setattr(service, "_ar5iv", CachedAr5iv())
    result = service.parse_source(source="ar5iv", paper_id="0911.3380")
    assert result["ok"] is True


def test_cli_source_commands_are_explicit(monkeypatch, capsys):
    captured = []
    monkeypatch.setattr(
        cli.service,
        "cache_arxiv_source",
        lambda paper_id, *, version, refresh=False, license_url="": (
            captured.append(("cache", paper_id, version, refresh, license_url))
            or {"ok": True, "data": {}, "errors": [], "meta": {}}
        ),
    )
    monkeypatch.setattr(
        cli.service,
        "probe_arxiv_source",
        lambda paper_id, *, version: (
            captured.append(("probe", paper_id, version))
            or {"ok": True, "data": {}, "errors": [], "meta": {}}
        ),
    )

    assert cli.main(["source-cache", "0911.3380", "--version", "2", "--license", "L", "--refresh", "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["source-probe", "0911.3380", "--version", "2", "--json"]) == 0
    capsys.readouterr()
    assert captured == [
        ("cache", "0911.3380", 2, True, "L"),
        ("probe", "0911.3380", 2),
    ]
