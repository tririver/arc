import httpx

from arc_paper_query.providers.ar5iv import Ar5ivProvider, ar5iv_url


def test_ar5iv_url_uses_arxiv_path_id():
    assert ar5iv_url("arXiv:0911.3380") == "https://ar5iv.labs.arxiv.org/html/0911.3380"
    assert ar5iv_url("arXiv:hep-th/0601001") == "https://ar5iv.labs.arxiv.org/html/hep-th/0601001"


def test_ar5iv_get_html_writes_and_reuses_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text="<html>paper</html>")

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.get_html("arXiv:0911.3380") == "<html>paper</html>"
    assert calls == ["https://ar5iv.labs.arxiv.org/html/0911.3380"]

    def failing_handler(request):
        raise AssertionError("cache hit should not call network")

    cached_provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(failing_handler)))
    assert cached_provider.get_html("arXiv:0911.3380") == "<html>paper</html>"
