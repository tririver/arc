from arc_paper_query.summary.input_pack import build_input_pack


def test_build_input_pack_includes_hash_and_truncated_sections():
    metadata = {"title": "A Test Paper", "authors": ["Alice"], "abstract": "Abstract"}
    parsed = {
        "toc": [{"id": "S1", "title": "1 Intro", "level": 2}],
        "sections": [{"section_id": "S1", "title": "1 Intro", "text": "a" * 120}],
    }
    references = [{"paper_id": "arXiv:0801.0001", "title": "Reference"}]

    pack = build_input_pack(
        "arXiv:0911.3380",
        metadata=metadata,
        parsed=parsed,
        references=references,
        max_section_chars=50,
    )

    assert pack["paper_id"] == "arXiv:0911.3380"
    assert len(pack["source_hash"]) == 64
    assert pack["metadata"] == metadata
    assert pack["references"] == references
    assert len(pack["sections"][0]["text"]) <= 80
    assert "[truncated]" in pack["sections"][0]["text"]
