import httpx
from pathlib import Path

from arc_paper.cache import CachePaths, read_json, write_json
from arc_paper.providers.ar5iv import Ar5ivProvider, ar5iv_url


def test_ar5iv_url_uses_arxiv_path_id():
    assert ar5iv_url("arXiv:0911.3380") == "https://ar5iv.labs.arxiv.org/html/0911.3380"
    assert ar5iv_url("arXiv:hep-th/0601001") == "https://ar5iv.labs.arxiv.org/html/hep-th/0601001"


def test_ar5iv_get_html_uses_html_cache_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text="<html>paper</html>")

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.get_html("arXiv:0911.3380") == "<html>paper</html>"
    assert provider.get_html("arXiv:0911.3380") == "<html>paper</html>"
    assert calls == ["https://ar5iv.labs.arxiv.org/html/0911.3380"]
    assert CachePaths.for_paper("arXiv:0911.3380").ar5iv_html.read_text(encoding="utf-8") == "<html>paper</html>"


def test_ar5iv_get_html_reads_existing_html_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    paths = CachePaths.for_paper("arXiv:0911.3380")
    paths.ar5iv_html.parent.mkdir(parents=True, exist_ok=True)
    paths.ar5iv_html.write_text("<html>stale cache</html>", encoding="utf-8")
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text="<html>fresh network</html>")

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.get_html("arXiv:0911.3380") == "<html>stale cache</html>"
    assert calls == []


def test_ar5iv_get_html_refresh_overwrites_existing_html_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    paths = CachePaths.for_paper("arXiv:0911.3380")
    paths.ar5iv_html.parent.mkdir(parents=True, exist_ok=True)
    paths.ar5iv_html.write_text("<html>stale cache</html>", encoding="utf-8")
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text="<html>fresh network</html>")

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.get_html("arXiv:0911.3380", refresh=True) == "<html>fresh network</html>"
    assert calls == ["https://ar5iv.labs.arxiv.org/html/0911.3380"]
    assert paths.ar5iv_html.read_text(encoding="utf-8") == "<html>fresh network</html>"


def test_ar5iv_assets_are_content_addressed_deduplicated_and_reused(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    calls = []
    image = b"\x89PNG\r\n\x1a\nfixture"

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=image, headers={"content-type": "image/png"})

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    html = '<article><figure><img src="/html/0911.3380/assets/plot.png"/><img src="/html/0911.3380/assets/plot.png"/></figure></article>'

    first = provider.cache_assets("arXiv:0911.3380", html)
    second = provider.cache_assets("arXiv:0911.3380", html)

    assert len(first) == 1
    assert first == second
    assert first[0]["status"] == "cached"
    assert first[0]["asset_id"] == f"sha256:{first[0]['sha256']}"
    assert Path(first[0]["cache_path"]).read_bytes() == image
    assert calls == ["https://ar5iv.labs.arxiv.org/html/0911.3380/assets/plot.png"]
    manifest = read_json(CachePaths.for_paper("arXiv:0911.3380").ar5iv_asset_manifest)
    assert manifest["assets"] == first


def test_ar5iv_asset_failure_is_recorded_without_fabricating_content(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))))

    records = provider.cache_assets(
        "arXiv:0911.3380",
        '<article><img src="https://example.org/not-ar5iv.png"/></article>',
    )

    assert records[0]["status"] == "missing"
    assert records[0]["sha256"] == ""
    assert records[0]["cache_path"] == ""
    assert "same-origin" in records[0]["error"]


def test_ar5iv_asset_discovery_excludes_site_chrome(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"paper-image", headers={"content-type": "image/png"})

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    records = provider.cache_assets(
        "arXiv:0911.3380",
        """
        <html><body>
          <nav><img src="/site-logo.png"/></nav>
          <article class="ltx_document"><img src="/html/0911.3380/assets/paper.png"/></article>
        </body></html>
        """,
    )

    assert len(records) == 1
    assert calls == ["https://ar5iv.labs.arxiv.org/html/0911.3380/assets/paper.png"]


def test_ar5iv_asset_redirect_does_not_request_a_cross_origin_target(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://example.org/escaped.png"})

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    records = provider.cache_assets(
        "arXiv:0911.3380",
        '<article class="ltx_document"><img src="/html/0911.3380/assets/paper.png"/></article>',
    )

    assert records[0]["status"] == "missing"
    assert "same-origin" in records[0]["error"]
    assert calls == ["https://ar5iv.labs.arxiv.org/html/0911.3380/assets/paper.png"]


def test_ar5iv_manifest_reuse_revalidates_cached_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"trusted-image", headers={"content-type": "image/png"})

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    html = '<article class="ltx_document"><img src="/html/0911.3380/assets/paper.png"/></article>'
    first = provider.cache_assets("arXiv:0911.3380", html)
    Path(first[0]["cache_path"]).write_bytes(b"tampered")

    second = provider.cache_assets("arXiv:0911.3380", html)

    assert len(calls) == 2
    assert Path(second[0]["cache_path"]).read_bytes() == b"trusted-image"
    assert second[0]["sha256"] == first[0]["sha256"]


def test_ar5iv_manifest_reuse_rejects_path_outside_current_asset_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"trusted-image")
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"trusted-image", headers={"content-type": "image/png"})

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    html = '<article class="ltx_document"><img src="/html/0911.3380/assets/paper.png"/></article>'
    first = provider.cache_assets("arXiv:0911.3380", html)
    paths = CachePaths.for_paper("arXiv:0911.3380")
    manifest = read_json(paths.ar5iv_asset_manifest)
    manifest["assets"][0]["cache_path"] = str(outside)
    write_json(paths.ar5iv_asset_manifest, manifest)

    second = provider.cache_assets("arXiv:0911.3380", html)

    # Manifest paths are never trusted; reuse is derived from the validated
    # content hash and the current paper's canonical asset root.
    assert len(calls) == 1
    assert Path(second[0]["cache_path"]).is_relative_to(paths.ar5iv_assets)
    assert second[0]["cache_path"] != str(outside)
    assert first[0]["sha256"] == second[0]["sha256"]
