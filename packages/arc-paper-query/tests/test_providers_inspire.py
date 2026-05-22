import urllib.parse

import httpx

from arc_paper_query.providers.inspire import InspireProvider


INSPIRE_RECORD = {
    "id": "123",
    "metadata": {
        "control_number": 123,
        "titles": [{"title": "A Test Paper"}],
        "authors": [{"full_name": "Alice A."}, {"full_name": "Bob B."}],
        "abstracts": [{"value": "This is the abstract."}],
        "arxiv_eprints": [{"value": "0911.3380"}],
        "citation_count": 7,
        "references": [
            {
                "record": {"$ref": "https://inspirehep.net/api/literature/456"},
                "reference": {"title": "A Reference", "arxiv_eprint": "0801.0001"},
            }
        ],
    },
}


def test_inspire_metadata_and_references_are_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        assert str(request.url) == "https://inspirehep.net/api/arxiv/0911.3380"
        return httpx.Response(200, json=INSPIRE_RECORD)

    provider = InspireProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    metadata = provider.get_metadata("arXiv:0911.3380")
    references = provider.get_references("arXiv:0911.3380")

    assert metadata["title"] == "A Test Paper"
    assert metadata["authors"] == ["Alice A.", "Bob B."]
    assert metadata["abstract"] == "This is the abstract."
    assert metadata["citation_count"] == 7
    assert references == [
        {
            "paper_id": "arXiv:0801.0001",
            "title": "A Reference",
            "inspire_recid": "456",
        }
    ]
    assert calls == ["https://inspirehep.net/api/arxiv/0911.3380"]

    cached_provider = InspireProvider(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError()))
        )
    )
    assert cached_provider.get_metadata("arXiv:0911.3380")["title"] == "A Test Paper"


def test_inspire_citers_use_recid_query_and_month_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if str(request.url) == "https://inspirehep.net/api/arxiv/0911.3380":
            return httpx.Response(200, json=INSPIRE_RECORD)
        query = urllib.parse.parse_qs(request.url.query.decode())
        assert query["q"] == ["refersto:recid:123"]
        assert query["size"] == ["1000"]
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "id": "789",
                            "metadata": {
                                "titles": [{"title": "A Citer"}],
                                "arxiv_eprints": [{"value": "2210.00001"}],
                                "authors": [{"full_name": "Carol C."}],
                                "citation_count": 3,
                            },
                        }
                    ]
                }
            },
        )

    provider = InspireProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    citers = provider.get_citers("arXiv:0911.3380")
    assert citers[0]["paper_id"] == "arXiv:2210.00001"
    assert citers[0]["title"] == "A Citer"
    assert len(calls) == 2

    cached_provider = InspireProvider(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError()))
        )
    )
    assert cached_provider.get_citers("arXiv:0911.3380")[0]["title"] == "A Citer"
