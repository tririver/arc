from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from arc_companion import pipeline
from arc_companion.pipeline import BuildOptions, CompanionLaneError, SourceBundle
from arc_llm import run_json


FAKE_KIMI = (
    Path(__file__).parents[2]
    / "arc-llm"
    / "tests"
    / "fixtures"
    / "fake_kimi_acp.py"
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:kimi-code-cli is experimental.*:RuntimeWarning"
)


class _CrashAfterCallCheckpoint(RuntimeError):
    pass


def _case(tmp_path: Path) -> tuple[SourceBundle, dict[str, Any], BuildOptions, Path]:
    block = {
        "block_id": "body-1",
        "type": "text",
        "text": "A short source sentence.",
    }
    document = {
        "front_matter": {},
        "blocks": [block],
        "equations": [],
        "figures": [],
        "tables": [],
        "bibliography": [],
        "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="local:checkpoint-business-replay",
        parsed={"document": document},
        document=document,
        metadata={},
        references=[],
        citers=[],
    )
    segment = {"segment_id": "segment-1", "block_ids": ["body-1"]}
    options = BuildOptions(
        paper_id=bundle.paper_id,
        project_dir=tmp_path,
        provider="kimi-code-cli",
        workers=1,
        allow_internet=False,
        idle_timeout_seconds=5,
    )
    return bundle, segment, options, tmp_path / "checkpoint"


def _configure_fake_kimi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    output: dict[str, Any],
) -> Path:
    record_path = tmp_path / "fake-kimi.jsonl"
    for key in (
        "ARC_LLM_TIMEOUT_SECONDS",
        "ARC_CODEX_TIMEOUT_SECONDS",
        "ARC_CLAUDE_TIMEOUT_SECONDS",
        "ARC_KIMI_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    values = {
        "ARC_KIMI_BIN": str(FAKE_KIMI),
        "ARC_HOME": str(tmp_path / "arc-home"),
        "ARC_LLM_CACHE": str(tmp_path / "arc-home" / "cache" / "arc-llm"),
        "ARC_KIMI_WORK_DIR": str(tmp_path),
        "ARC_KIMI_IDLE_TIMEOUT_SECONDS": "5",
        "FAKE_KIMI_RECORD": str(record_path),
        "FAKE_KIMI_SCENARIO": "happy",
        "FAKE_KIMI_OUTPUT": json.dumps(
            output, ensure_ascii=False, separators=(",", ":")
        ),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return record_path


def _provider_counts(record_path: Path) -> dict[str, int]:
    records = (
        [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]
        if record_path.is_file()
        else []
    )
    methods = [
        str(item["message"].get("method") or "")
        for item in records
        if item.get("kind") == "client_message"
        and isinstance(item.get("message"), dict)
    ]
    return {
        "provider_prompt": methods.count("session/prompt"),
        "old_native_resume": methods.count("session/resume"),
        "fresh_session": methods.count("session/new"),
    }


def _call_checkpoint(checkpoint_dir: Path) -> Path:
    paths = [
        path
        for path in checkpoint_dir.rglob("call-checkpoints/*.json")
        if not path.name.endswith(".candidate-selection.json")
    ]
    assert len(paths) == 1
    return paths[0]


def _seed_checkpoint_then_crash(
    *,
    bundle: SourceBundle,
    segment: dict[str, Any],
    options: BuildOptions,
    checkpoint_dir: Path,
) -> None:
    def crash_after_checkpoint(prompt: str, **kwargs: Any) -> dict[str, Any]:
        run_json(prompt, **kwargs)
        raise _CrashAfterCallCheckpoint(
            "simulated controller crash before translation business validation"
        )

    with pytest.raises(CompanionLaneError, match="simulated controller crash"):
        pipeline._generate_translations(
            [segment],
            options=options,
            bundle=bundle,
            glossary={"entries": []},
            protected_names=[],
            checkpoint_dir=checkpoint_dir,
            llm=crash_after_checkpoint,
        )


@pytest.mark.parametrize("checkpoint_state", ["response_received", "validated"])
def test_real_call_checkpoint_replay_reenters_translation_and_accepts_without_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_state: str,
) -> None:
    bundle, segment, options, checkpoint_dir = _case(tmp_path)
    expected = {"blocks": [{"block_id": "body-1", "text": "一条简短的源句。"}]}
    record_path = _configure_fake_kimi(monkeypatch, tmp_path, output=expected)
    _seed_checkpoint_then_crash(
        bundle=bundle,
        segment=segment,
        options=options,
        checkpoint_dir=checkpoint_dir,
    )

    call_checkpoint = _call_checkpoint(checkpoint_dir)
    durable = json.loads(call_checkpoint.read_text(encoding="utf-8"))
    assert durable["state"] == "validated"
    if checkpoint_state == "response_received":
        durable["state"] = "response_received"
        durable.pop("validated_at", None)
        pipeline.write_json(call_checkpoint, durable)
    counts_before_replay = _provider_counts(record_path)
    accepted_callbacks: list[tuple[str, str, dict[str, Any]]] = []

    result = pipeline._generate_translations(
        [segment],
        options=options,
        bundle=bundle,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=checkpoint_dir,
        llm=run_json,
        accepted_callback=lambda lane, logical_unit, value: accepted_callbacks.append(
            (lane, logical_unit, value)
        ),
    )

    assert result == {"segment-1": expected}
    assert accepted_callbacks == [("translation", "segment-1", expected)]
    counts_after_replay = _provider_counts(record_path)
    assert {
        key: counts_after_replay[key] - counts_before_replay[key]
        for key in ("provider_prompt", "old_native_resume", "fresh_session")
    } == {
        "provider_prompt": 0,
        "old_native_resume": 0,
        "fresh_session": 0,
    }
    # The owning translation handler must close its stateless control before
    # returning; no final sweep is allowed to do deferred business acceptance.
    assert pipeline._accept_completed_pipeline_controls(checkpoint_dir) == 0
    ledger_paths = list(
        (checkpoint_dir / "recovery-controls" / "translation").glob("*-ledger.json")
    )
    assert len(ledger_paths) == 1
    ledger = json.loads(ledger_paths[0].read_text(encoding="utf-8"))
    assert ledger["blocks"][0]["state"] == "accepted"
    assert json.loads(call_checkpoint.read_text(encoding="utf-8"))["state"] == "validated"


def test_schema_shaped_replay_fails_business_validation_and_is_not_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, segment, options, checkpoint_dir = _case(tmp_path)
    citation_digest = hashlib.sha256(b"citation:1").hexdigest()
    bundle.document["blocks"][0].update({
        "text": "See [1].",
        "inline_runs": [
            {"kind": "text", "content": "See [", "order": 1},
            {
                "kind": "citation",
                "content": "1",
                "order": 2,
                "token_id": "body-1.citation-0001",
                "content_hash": citation_digest,
            },
            {"kind": "text", "content": "].", "order": 3},
        ],
    })
    citation_token = pipeline._opaque_inline_tokens(bundle.document["blocks"][0])[0]
    schema_shaped_but_wrong = {
        "blocks": [{
            "block_id": "body-1",
            "text": f"形式正确但括号归属含混。[]{citation_token}[]",
        }]
    }
    record_path = _configure_fake_kimi(
        monkeypatch, tmp_path, output=schema_shaped_but_wrong
    )
    _seed_checkpoint_then_crash(
        bundle=bundle,
        segment=segment,
        options=options,
        checkpoint_dir=checkpoint_dir,
    )
    counts_before_replay = _provider_counts(record_path)

    with pytest.raises(CompanionLaneError, match="ambiguous adjacent brackets"):
        pipeline._generate_translations(
            [segment],
            options=options,
            bundle=bundle,
            glossary={"entries": []},
            protected_names=[],
            checkpoint_dir=checkpoint_dir,
            llm=run_json,
        )

    counts_after_replay = _provider_counts(record_path)
    assert {
        key: counts_after_replay[key] - counts_before_replay[key]
        for key in ("provider_prompt", "old_native_resume", "fresh_session")
    } == {
        "provider_prompt": 0,
        "old_native_resume": 0,
        "fresh_session": 0,
    }
    assert pipeline._accept_completed_pipeline_controls(checkpoint_dir) == 0
    ledger_path = next(
        (checkpoint_dir / "recovery-controls" / "translation").glob("*-ledger.json")
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["blocks"][0]["state"] != "accepted"
    acceptance_path = checkpoint_dir / "translations" / (
        pipeline._segment_checkpoint_name(str(segment["segment_id"])) + ".json"
    )
    assert not acceptance_path.exists()
