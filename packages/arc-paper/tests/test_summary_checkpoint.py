from __future__ import annotations

import json

import pytest

from arc_paper.summary import checkpoint


def _valid(value):
    if not isinstance(value.get("answer"), str):
        raise ValueError("invalid answer")


def test_response_is_replayed_without_second_provider_call(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    calls = 0

    def provider(prompt, schema, model):
        nonlocal calls
        calls += 1
        return {"answer": "saved"}

    kwargs = dict(
        paper_id="arXiv:0911.3380",
        call_kind="test",
        identity={"source_hash": "abc"},
        prompt="prompt",
        schema={"type": "object"},
        model="model",
        run_json=provider,
        validate=_valid,
    )
    assert checkpoint.run_json_checkpointed(**kwargs) == {"answer": "saved"}
    assert checkpoint.run_json_checkpointed(**kwargs) == {"answer": "saved"}
    assert calls == 1


def test_invalid_paid_response_is_checkpointed_before_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    calls = 0

    def provider(prompt, schema, model):
        nonlocal calls
        calls += 1
        return {"answer": 42}

    kwargs = dict(
        paper_id="arXiv:0911.3380",
        call_kind="test-invalid",
        identity={"source_hash": "def"},
        prompt="prompt",
        schema={"type": "object"},
        model="model",
        run_json=provider,
        validate=_valid,
    )
    with pytest.raises(ValueError, match="invalid answer"):
        checkpoint.run_json_checkpointed(**kwargs)
    with pytest.raises(ValueError, match="invalid answer"):
        checkpoint.run_json_checkpointed(**kwargs)
    assert calls == 1
    files = list(tmp_path.rglob("*.json"))
    payloads = [json.loads(path.read_text()) for path in files]
    assert any(item.get("status") == "response_received" for item in payloads)


def test_uncertain_call_waits_one_hour_then_retries_only_once(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    monkeypatch.setattr(checkpoint, "UNCERTAIN_RETRY_SECONDS", 3600)

    def provider(prompt, schema, model):
        raise RuntimeError("connection vanished")

    kwargs = dict(
        paper_id="arXiv:0911.3380",
        call_kind="test-uncertain",
        identity={"source_hash": "ghi"},
        prompt="prompt",
        schema={"type": "object"},
        model="model",
        run_json=provider,
        validate=_valid,
    )
    with pytest.raises(RuntimeError, match="connection vanished"):
        checkpoint.run_json_checkpointed(**kwargs)
    with pytest.raises(checkpoint.CallCheckpointUncertain, match="quarantined"):
        checkpoint.run_json_checkpointed(**kwargs)

    path = next(tmp_path.rglob("*.json"))
    payload = json.loads(path.read_text())
    payload["started_at"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="connection vanished"):
        checkpoint.run_json_checkpointed(**kwargs)
    payload = json.loads(path.read_text())
    payload["started_at"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(checkpoint.CallCheckpointUncertain, match="remained uncertain"):
        checkpoint.run_json_checkpointed(**kwargs)
