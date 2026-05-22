import pytest

from arc_paper_query.summary.store import SummaryStoreError, read_summary, store_summary


def valid_summary():
    return {
        "schema_version": "arc.paper_llm_summary.v1",
        "paper_id": "arXiv:0911.3380",
        "title": "A Test Paper",
        "authors_short": "Alice and Bob",
        "high_value_summary": ["The paper computes a useful result."],
        "toc": [
            {
                "section_id": "S1",
                "title": "1 Introduction",
                "one_sentence_summary": "Introduces the problem.",
            }
        ],
        "reading_guide": [
            {
                "purpose": "Understand the main result",
                "sections": ["S1"],
                "reason": "This section defines the setup.",
            }
        ],
        "warnings": [],
        "provenance": {
            "created_at": "2026-05-22T00:00:00Z",
            "method": "manual",
            "model": "test-model",
            "prompt_version": "paper-summary-v1",
            "source_hash": "a" * 64,
        },
    }


def test_store_summary_validates_and_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    summary = valid_summary()

    path = store_summary("arXiv:0911.3380", summary)

    assert path.exists()
    cached = read_summary(
        "arXiv:0911.3380",
        prompt_version="paper-summary-v1",
        source_hash="a" * 64,
    )
    assert cached["title"] == "A Test Paper"


def test_store_summary_rejects_mismatched_paper_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    summary = valid_summary()

    with pytest.raises(SummaryStoreError):
        store_summary("arXiv:0000.0000", summary)
