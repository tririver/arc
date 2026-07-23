import json

import pytest
from arc_llm import runner as llm_runner
from arc_llm.budget import SharedBudget, shared_budget_context
from arc_llm.providers.base import LLMWorkerError
from arc_llm.usage import LLMProviderResponse, LLMUsage

from arc_paper.batch.db import BatchDB
from arc_paper.batch import runner
from arc_paper.summary.checkpoint import current_schema_canary_root
from arc_paper.execution import (
    ManagedExecutionCancelled,
    current_progress_callback,
    managed_execution_scope,
)


def test_prefetch_marks_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")

    monkeypatch.setattr(runner.service, "get_title", lambda paper_id, refresh=False: {"ok": True})
    monkeypatch.setattr(runner.service, "get_abstract", lambda paper_id, refresh=False: {"ok": True})
    monkeypatch.setattr(runner.service, "get_toc", lambda paper_id, refresh=False: {"ok": True})

    result = runner.prefetch_batch("qft", workers=1, db=db)

    assert result["counts"] == {"ready": 1}


def test_run_batch_marks_done_and_export(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")
    db.mark_status("qft", "arXiv:0911.3380", "ready")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"title": "Done"}), encoding="utf-8")

    monkeypatch.setattr(
        runner.service,
        "generate_llm_summary",
        lambda paper_id, provider="auto", model=None, model_tier=None, refresh=False: {
            "ok": True,
            "data": {"title": "Done"},
            "meta": {"summary_path": str(summary_path)},
        },
    )

    run_result = runner.run_batch("qft", provider="auto", concurrency=1, db=db)
    assert run_result["counts"] == {"done": 1}

    output = tmp_path / "summaries.jsonl"
    export_result = runner.export_batch("qft", output=output, db=db)
    assert export_result["exported"] == 1
    assert json.loads(output.read_text().strip())["title"] == "Done"


def test_run_batch_passes_model_tier(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")
    db.mark_status("qft", "arXiv:0911.3380", "ready")
    captured = {}

    def generate_llm_summary(paper_id, *, provider="auto", model=None, model_tier=None, refresh=False):
        captured.update(
            {
                "paper_id": paper_id,
                "provider": provider,
                "model": model,
                "model_tier": model_tier,
            }
        )
        return {"ok": True, "data": {"title": "Done"}, "meta": {"summary_path": str(tmp_path / "summary.json")}}

    monkeypatch.setattr(runner.service, "generate_llm_summary", generate_llm_summary)

    runner.run_batch("qft", provider="auto", model_tier="high", concurrency=1, db=db)

    assert captured == {
        "paper_id": "arXiv:0911.3380",
        "provider": "auto",
        "model": None,
        "model_tier": "high",
    }


def test_run_batch_claims_only_available_executor_slots(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380", "hep-th/0601001"], "paper-summary-v1")
    db.mark_status("qft", "arXiv:0911.3380", "ready")
    db.mark_status("qft", "arXiv:hep-th/0601001", "ready")
    counts_seen = []

    def generate_llm_summary(paper_id, *, provider="auto", model=None, model_tier=None, refresh=False):
        counts_seen.append(db.status_counts("qft"))
        return {"ok": True, "data": {"title": "Done"}, "meta": {"summary_path": str(tmp_path / "summary.json")}}

    monkeypatch.setattr(runner.service, "generate_llm_summary", generate_llm_summary)

    run_result = runner.run_batch("qft", provider="auto", concurrency=1, max_items=2, db=db)

    assert counts_seen[0] == {"ready": 1, "running": 1}
    assert run_result["counts"] == {"done": 2}


def test_run_batch_gives_all_items_one_schema_canary_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    paper_ids = [f"2001.{index:04d}" for index in range(1, 25)]
    db.create_batch("qft", paper_ids, "paper-summary-v1")
    for paper_id in paper_ids:
        db.mark_status("qft", paper_id, "ready")
    roots = []

    def generate_llm_summary(paper_id, **kwargs):
        roots.append(current_schema_canary_root())
        return {
            "ok": True,
            "data": {"title": "Done"},
            "meta": {"summary_path": str(tmp_path / f"{paper_id}.json")},
        }

    monkeypatch.setattr(runner.service, "generate_llm_summary", generate_llm_summary)

    runner.run_batch("qft", provider="auto", concurrency=24, db=db)

    assert len(roots) == 24
    assert set(roots) == {runner._schema_canary_root(db, "qft")}


def test_run_batch_rejects_auto_provider_with_exact_model_before_status_mutation(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")
    db.mark_status("qft", "arXiv:0911.3380", "ready")

    with pytest.raises(ValueError, match="Exact model requires explicit provider"):
        runner.run_batch("qft", provider="auto", model="gpt-5.5", concurrency=1, db=db)

    assert db.status_counts("qft") == {"ready": 1}


def test_run_batch_max_items_zero_does_not_claim_or_call(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")
    db.mark_status("qft", "0911.3380", "ready")
    monkeypatch.setattr(
        runner.service,
        "generate_llm_summary",
        lambda *args, **kwargs: pytest.fail("LLM must not be called"),
    )

    result = runner.run_batch("qft", max_items=0, db=db)

    assert result["counts"] == {"ready": 1}


def test_run_batch_stops_refilling_after_batch_fatal(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380", "hep-th/0601001"], "paper-summary-v1")
    db.mark_status("qft", "0911.3380", "ready")
    db.mark_status("qft", "hep-th/0601001", "ready")
    calls = []

    def fatal(paper_id, **kwargs):
        calls.append(paper_id)
        raise LLMWorkerError("quota exhausted", abort_batch=True)

    monkeypatch.setattr(runner.service, "generate_llm_summary", fatal)

    with pytest.raises(LLMWorkerError, match="quota exhausted"):
        runner.run_batch("qft", concurrency=1, db=db)

    assert calls == ["arXiv:0911.3380"]
    assert db.status_counts("qft") == {"failed": 1, "ready": 1}


def test_run_batch_propagates_progress_context_into_worker_thread(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380"], "paper-summary-v1")
    db.mark_status("qft", "0911.3380", "ready")
    events = []

    def generate_llm_summary(paper_id, **kwargs):
        callback = current_progress_callback()
        assert callback is not None
        callback({"event": "section_completed", "sections_completed": 1})
        return {
            "ok": True,
            "data": {"title": "Done"},
            "meta": {"summary_path": str(tmp_path / "summary.json")},
        }

    monkeypatch.setattr(runner.service, "generate_llm_summary", generate_llm_summary)
    with managed_execution_scope(
        progress_callback=events.append,
        cancel_check=lambda: False,
    ):
        runner.run_batch("qft", concurrency=1, db=db)

    assert events == [{"event": "section_completed", "sections_completed": 1}]


def test_run_batch_cancel_before_claim_leaves_items_ready(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380", "hep-th/0601001"], "paper-summary-v1")
    db.mark_status("qft", "0911.3380", "ready")
    db.mark_status("qft", "hep-th/0601001", "ready")
    monkeypatch.setattr(
        runner.service,
        "generate_llm_summary",
        lambda *args, **kwargs: pytest.fail("cancelled batch must not submit"),
    )

    with (
        managed_execution_scope(
            progress_callback=None,
            cancel_check=lambda: True,
        ),
        pytest.raises(ManagedExecutionCancelled),
    ):
        runner.run_batch("qft", concurrency=2, db=db)

    assert db.status_counts("qft") == {"ready": 2}


def test_concurrent_batch_items_charge_one_shared_budget(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    db = BatchDB.default()
    db.create_batch("qft", ["0911.3380", "hep-th/0601001"], "paper-summary-v1")
    db.mark_status("qft", "0911.3380", "ready")
    db.mark_status("qft", "hep-th/0601001", "ready")

    class Provider:
        name = "codex-cli"

        def generate_json_result(self, prompt, **kwargs):
            return LLMProviderResponse(
                {"ok": True},
                usage=LLMUsage(input_tokens=4, output_tokens=1),
            )

    monkeypatch.setattr(
        llm_runner, "select_provider", lambda *args, **kwargs: Provider(),
    )

    def generate_llm_summary(paper_id, **kwargs):
        llm_runner.run_json(
            f"summarize {paper_id}",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            provider="codex-cli",
            env={},
            process_chain=[],
            artifact_dir=tmp_path / "calls" / paper_id.replace(":", "-"),
            call_label="summary",
            idempotency_key=paper_id,
        )
        return {
            "ok": True,
            "data": {"title": "Done"},
            "meta": {"summary_path": str(tmp_path / f"{paper_id}.json")},
        }

    monkeypatch.setattr(runner.service, "generate_llm_summary", generate_llm_summary)
    budget = SharedBudget.create(
        tmp_path / "budget.sqlite3",
        budget_id="batch-parent",
        max_calls=2,
        max_tokens=1_000,
    )
    with shared_budget_context(budget, output_reserve_tokens=50):
        result = runner.run_batch("qft", concurrency=2, db=db)

    assert result["counts"] == {"done": 2}
    snapshot = budget.snapshot()
    assert snapshot.charged_calls == 2
    assert snapshot.charged_tokens == 10
    assert snapshot.outstanding_calls == 0
