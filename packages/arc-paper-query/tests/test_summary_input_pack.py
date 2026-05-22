from arc_paper_query.summary.input_pack import build_input_pack


def test_build_input_pack_includes_hash_and_truncated_sections():
    metadata = {"title": "A Test Paper", "authors": ["Alice"], "abstract": "Abstract"}
    parsed = {
        "toc": [
            {"id": "S1", "title": "1 Intro", "level": 2},
            {"id": "S1.SS1", "title": "1.1 Details", "level": 3},
            {"id": "bib", "title": "References", "level": 2},
        ],
        "sections": [
            {"section_id": "S1", "title": "1 Intro", "level": 2, "text": "a" * 120},
            {"section_id": "S1.SS1", "title": "1.1 Details", "level": 3, "text": "b" * 120},
            {"section_id": "bib", "title": "References", "level": 2, "text": "c" * 120},
        ],
    }

    pack = build_input_pack(
        "arXiv:0911.3380",
        metadata=metadata,
        parsed=parsed,
        max_section_chars=50,
    )

    assert pack["paper_id"] == "arXiv:0911.3380"
    assert len(pack["source_hash"]) == 64
    assert pack["metadata"] == metadata
    assert "references" not in pack
    assert [item["id"] for item in pack["toc"]] == ["S1"]
    assert [section["section_id"] for section in pack["sections"]] == ["S1"]
    assert len(pack["sections"][0]["text"]) <= 80
    assert "[truncated]" in pack["sections"][0]["text"]
