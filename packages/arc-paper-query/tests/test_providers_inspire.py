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

FULL_REFERENCE_RECORD = {
    "id": "456",
    "metadata": {
        "control_number": 456,
        "titles": [{"title": "A Full Reference"}],
        "authors": [{"full_name": "Ref Author"}],
        "abstracts": [{"value": "Reference abstract."}],
        "arxiv_eprints": [{"value": "0801.0001"}],
        "citation_count": 11,
        "earliest_date": "2008-01-01",
    },
}


def test_inspire_mathml_text_is_normalized(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    planck = '<math display="inline"><mi>P</mi><mi>l</mi><mi>a</mi><mi>n</mi><mi>c</mi><mi>k</mi></math>'
    record = {
        "id": "2878726",
        "metadata": {
            "control_number": 2878726,
            "titles": [{"title": f"Searching for inflationary physics. Constraints from {planck}"}],
            "abstracts": [
                {
                    "value": (
                        'We analyze <math display="inline"><mo>ℓ</mo><mo>∈</mo>'
                        '<mo stretchy="false">[</mo><mn>2</mn><mo>,</mo><mn>2048</mn>'
                        '<mo stretchy="false">]</mo></math> and '
                        '<math display="inline"><msubsup><mi>g</mi><mrow><mi>NL</mi></mrow>'
                        '<mrow><mi>loc</mi></mrow></msubsup></math>.'
                    )
                }
            ],
            "arxiv_eprints": [{"value": "2502.06931"}],
            "references": [
                {
                    "record": {"$ref": "https://inspirehep.net/api/literature/1"},
                    "reference": {"title": f"Reference to {planck}", "arxiv_eprint": "2501.00001"},
                }
            ],
        },
    }

    provider = InspireProvider(client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=record))))

    metadata = provider.get_metadata("2502.06931")
    references = provider.get_references("2502.06931")

    assert metadata["title"] == "Searching for inflationary physics. Constraints from Planck"
    assert "<math" not in metadata["abstract"]
    assert "ℓ∈[2,2048]" in metadata["abstract"]
    assert "g_NL^loc" in metadata["abstract"]
    assert references[0]["title"] == "Reference to Planck"

    cached_provider = InspireProvider(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError()))
        )
    )
    assert cached_provider.get_metadata("arXiv:2502.06931")["title"].endswith("Planck")


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
    assert metadata["identifiers"]["arxiv"] == "arXiv:0911.3380"
    assert references == [
        {
            "paper_id": "arXiv:0801.0001",
            "title": "A Reference",
            "raw_inspire_reference": {
                "record": {"$ref": "https://inspirehep.net/api/literature/456"},
                "reference": {"title": "A Reference", "arxiv_eprint": "0801.0001"},
            },
            "record_ref": "https://inspirehep.net/api/literature/456",
            "arxiv_id": "0801.0001",
            "inspire_recid": "456",
            "identifiers": {
                "paper_id": "arXiv:0801.0001",
                "arxiv": "arXiv:0801.0001",
                "arxiv_id": "0801.0001",
                "inspire": "inspire:456",
                "inspire_recid": "456",
            },
        }
    ]
    assert calls == ["https://inspirehep.net/api/arxiv/0911.3380"]

    cached_provider = InspireProvider(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError()))
        )
    )
    assert cached_provider.get_metadata("arXiv:0911.3380")["title"] == "A Test Paper"
    assert cached_provider.get_references("arXiv:0911.3380")[0]["title"] == "A Reference"


def test_inspire_references_can_be_enriched_through_single_paper_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if str(request.url) == "https://inspirehep.net/api/arxiv/0911.3380":
            return httpx.Response(200, json=INSPIRE_RECORD)
        if str(request.url) == "https://inspirehep.net/api/literature/456":
            return httpx.Response(200, json=FULL_REFERENCE_RECORD)
        raise AssertionError(f"unexpected request: {request.url}")

    provider = InspireProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    references = provider.get_references("0911.3380", enrich=True)

    assert calls == [
        "https://inspirehep.net/api/arxiv/0911.3380",
        "https://inspirehep.net/api/literature/456",
    ]
    reference = references[0]
    assert reference["paper_id"] == "arXiv:0801.0001"
    assert reference["title"] == "A Full Reference"
    assert reference["abstract"] == "Reference abstract."
    assert reference["authors"] == ["Ref Author"]
    assert reference["citation_count"] == 11
    assert reference["metadata_enriched"] is True
    assert reference["identifiers"]["arxiv"] == "arXiv:0801.0001"
    assert reference["identifiers"]["inspire"] == "inspire:456"

    cached_provider = InspireProvider(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError()))
        )
    )
    assert cached_provider.get_references("arXiv:0911.3380", enrich=True)[0]["abstract"] == "Reference abstract."
    assert cached_provider.get_metadata("inspire:456")["title"] == "A Full Reference"
    assert cached_provider.get_metadata("arXiv:0801.0001")["title"] == "A Full Reference"


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
        assert query["sort"] == ["mostrecent"]
        assert "abstracts" in query["fields"][0]
        assert "arxiv_eprints" in query["fields"][0]
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
                                "abstracts": [{"value": "Citer abstract."}],
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
    assert citers[0]["abstract"] == "Citer abstract."
    assert len(calls) == 2

    cached_provider = InspireProvider(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError()))
        )
    )
    assert cached_provider.get_citers("arXiv:0911.3380")[0]["title"] == "A Citer"


def test_inspire_citers_support_limit_sort_and_separate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if str(request.url) == "https://inspirehep.net/api/arxiv/0911.3380":
            return httpx.Response(200, json=INSPIRE_RECORD)
        query = urllib.parse.parse_qs(request.url.query.decode())
        assert query["q"] == ["refersto:recid:123"]
        assert query["size"] == ["3"]
        assert query["sort"] == ["mostcited"]
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "id": str(index),
                            "metadata": {
                                "titles": [{"title": f"Citer {index}"}],
                                "citation_count": 10 - index,
                            },
                        }
                        for index in range(3)
                    ]
                }
            },
        )

    provider = InspireProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    citers = provider.get_citers("arXiv:0911.3380", limit=3, sort="mostcited")

    assert [item["title"] for item in citers] == ["Citer 0", "Citer 1", "Citer 2"]
    assert len(calls) == 2

    cached_provider = InspireProvider(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError()))
        )
    )
    assert cached_provider.get_citers("arXiv:0911.3380", limit=3, sort="mostcited")[1]["title"] == "Citer 1"
