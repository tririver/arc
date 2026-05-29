import httpx

from arc_paper.cache import CachePaths
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
