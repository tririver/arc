import json

from arc_paper import cli, service


class FakeInspire:
    def get_metadata(self, paper_id, *, refresh=False):
        return {
            "paper_id": "arXiv:0911.3380",
            "title": "A Test Paper",
            "abstract": "A useful abstract.",
            "authors": ["Alice", "Bob"],
            "citation_count": 5,
        }

    def get_references(self, paper_id, *, refresh=False, enrich=False):
        raise AssertionError("summary input packs should not fetch references")


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
                "level": 2,
            }
        ],
        "section_summaries": [
            {
                "section_id": "S1",
                "title": "1 Introduction",
                "summary": "Introduces the problem.",
                "warnings": [],
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

    def generate_summary(self, task, *, model=None, progress_callback=None):
        summary = valid_summary()
        summary["provenance"]["source_hash"] = task["input_pack"]["source_hash"]
        if model:
            summary["provenance"]["model"] = model
        return summary


class RecordingSummaryProvider(FakeSummaryProvider):
    def __init__(self):
        self.tasks = []

    def generate_summary(self, task, *, model=None, progress_callback=None):
        self.tasks.append(task)
        return super().generate_summary(task, model=model, progress_callback=progress_callback)


class ModelRecordingSummaryProvider(FakeSummaryProvider):
    name = "codex-cli"

    def __init__(self):
        self.models = []

    def generate_summary(self, task, *, model=None, progress_callback=None):
        self.models.append(model)
        summary = super().generate_summary(task, model=model, progress_callback=progress_callback)
        summary["provenance"]["method"] = self.name
        summary["provenance"]["model"] = model or "default-model"
        return summary


class ManualSummaryProvider:
    name = "manual"


def test_get_llm_summary_returns_needs_llm_when_provider_is_manual(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    monkeypatch.setattr(service, "select_summary_provider", lambda provider: ManualSummaryProvider())

    result = service.get_llm_summary("0911.3380")

    assert result["ok"] is False
    assert result["status"] == "needs_llm"
    assert result["llm_task"]["input_pack"]["paper_id"] == "arXiv:0911.3380"
    assert "references" not in result["llm_task"]["input_pack"]
    assert result["llm_task"]["output_schema"]["$id"] == "arc.paper-summary-v1"


def test_get_llm_summary_autogenerates_and_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    monkeypatch.setattr(service, "select_summary_provider", lambda provider: FakeSummaryProvider())

    result = service.get_llm_summary("0911.3380")

    assert result["ok"] is True
    assert result["data"]["title"] == "A Test Paper"
    assert result["meta"]["cache"] == "write"
    assert service.get_llm_summary("0911.3380")["meta"]["cache"] == "hit"


def test_get_llm_summary_uses_canonical_cache_key_for_aliases(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    provider = RecordingSummaryProvider()
    monkeypatch.setattr(service, "select_summary_provider", lambda provider_name: provider)

    result = service.get_llm_summary("inspire:837197")

    assert result["ok"] is True
    assert result["data"]["paper_id"] == "arXiv:0911.3380"
    assert provider.tasks[0]["input_pack"]["paper_id"] == "arXiv:0911.3380"
    assert service.get_llm_summary("0911.3380")["meta"]["cache"] == "hit"
    assert len(provider.tasks) == 1


def test_generate_llm_summary_uses_provider_and_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    monkeypatch.setattr(service, "select_summary_provider", lambda provider: FakeSummaryProvider())

    result = service.generate_llm_summary("0911.3380", provider="manual")

    assert result["ok"] is True
    assert result["data"]["title"] == "A Test Paper"
    assert service.get_llm_summary("0911.3380")["meta"]["cache"] == "hit"
    assert service.get_cached_llm_summary("0911.3380")["meta"]["cache"] == "hit"


def test_generate_llm_summary_keeps_generation_cache_model_specific(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    provider = ModelRecordingSummaryProvider()
    monkeypatch.setattr(service, "select_summary_provider", lambda provider_name: provider)

    first = service.generate_llm_summary("0911.3380", provider="codex-cli", model="cheap-model")
    second = service.generate_llm_summary("0911.3380", provider="codex-cli", model="quality-model")

    assert first["ok"] is True
    assert second["ok"] is True
    assert provider.models == ["cheap-model", "quality-model"]
    assert second["data"]["provenance"]["model"] == "quality-model"


def test_generate_llm_summary_rejects_auto_provider_with_exact_model(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())

    result = service.generate_llm_summary("0911.3380", provider="auto", model="gpt-5.5")

    assert result["ok"] is False
    assert "Exact model requires explicit provider" in result["error"]["message"]


def test_generate_llm_summary_resolves_model_tier_before_summary_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setenv("ARC_AGENT_HOST", "codex")
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    captured = {}

    def select_summary_provider(provider):
        captured["provider"] = provider
        fake = FakeSummaryProvider()
        fake.name = provider
        return fake

    monkeypatch.setattr(service, "select_summary_provider", select_summary_provider)

    result = service.generate_llm_summary("0911.3380", provider="auto", model_tier="high")

    assert result["ok"] is True
    assert result["meta"]["provider"] == "codex-cli"
    assert captured["provider"] == "codex-cli"
    assert result["data"]["provenance"]["model"] == "gpt-5.6-sol"


def test_cli_get_llm_summary_can_force_manual_provider(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(service, "_inspire", FakeInspire())
    monkeypatch.setattr(service, "_ar5iv", FakeAr5iv())
    monkeypatch.setattr(service, "select_summary_provider", lambda provider: ManualSummaryProvider())

    assert cli.main(["get-llm-summary", "0911.3380", "--provider", "manual", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "needs_llm"


def test_cli_generate_llm_summary_passes_model_tier(monkeypatch, capsys):
    captured = {}

    def fake_generate(paper_ids, **kwargs):
        captured["paper_ids"] = paper_ids
        captured.update(kwargs)
        return {"ok": True, "data": {"paper_ids": paper_ids}}

    monkeypatch.setattr(service, "generate_llm_summary", fake_generate)

    assert cli.main(["llm-generate-summary", "0911.3380", "--model-tier", "high", "--json"]) == 0

    assert captured["paper_ids"] == "0911.3380"
    assert captured["model_tier"] == "high"
    json.loads(capsys.readouterr().out)


def test_cli_store_llm_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(valid_summary()), encoding="utf-8")

    assert cli.main(["store-llm-summary", "arXiv:0911.3380", "--summary-json", str(summary_path), "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["data"]["summary"]["title"] == "A Test Paper"
