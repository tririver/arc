import json

from arc_paper_query import cli, service


class FakeInspire:
    def get_metadata(self, paper_id, *, refresh=False):
        return {
            "paper_id": "arXiv:0911.3380",
            "title": "A Test Paper",
            "abstract": "A useful abstract.",
            "authors": ["Alice", "Bob"],
            "citation_count": 5,
        }

    def get_references(self, paper_id, *, refresh=False):
        return [{"paper_id": "arXiv:0801.0001", "title": "Reference"}]


class FakeAr5iv:
    def get_html(self, paper_id, *, refresh=False):
        return """
        <html><body>
          <section id="S1"><h2>1 Introduction</h2><p>Intro.</p></section>
        </body></html>
        """


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


class FakeSummaryProvider:
    name = "fake"

    def generate_summary(self, task, *, model=None):
        summary = valid_summary()
        summary["provenance"]["source_hash"] = task["input_pack"]["source_hash"]
        return summary


def test_get_llm_summary_returns_needs_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())

    result = service.get_llm_summary("0911.3380")

    assert result["ok"] is False
    assert result["status"] == "needs_llm"
    assert result["llm_task"]["input_pack"]["paper_id"] == "arXiv:0911.3380"
    assert result["llm_task"]["output_schema"]["$id"] == "arc.paper-summary-v1"


def test_generate_llm_summary_uses_provider_and_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    monkeypatch.setattr(service, "select_summary_provider", lambda provider: FakeSummaryProvider())

    result = service.generate_llm_summary("0911.3380", provider="manual")

    assert result["ok"] is True
    assert result["data"]["title"] == "A Test Paper"
    assert service.get_llm_summary("0911.3380")["meta"]["cache"] == "hit"


def test_cli_store_llm_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(valid_summary()), encoding="utf-8")

    assert cli.main(["store-llm-summary", "arXiv:0911.3380", "--summary-json", str(summary_path), "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["data"]["summary"]["title"] == "A Test Paper"
