from __future__ import annotations

from arc_llm.progress_prompt import (
    RUNTIME_PROGRESS_CONTRACT_MARKER,
    ensure_runtime_progress_contract,
    has_runtime_progress_contract,
    runtime_progress_contract,
)
from arc_llm.proposers_reviewer.config import load_batch_config
from arc_llm.proposers_reviewer.dialogue import (
    render_initial_worker_prompt,
    render_proposer_delta_prompt,
)
from arc_llm.proposers_reviewer.prompts import proposer_context, render_prompt
from arc_llm.schema_cache import sha256_text


def _batch_payload(tmp_path):
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "progress-test",
        "run_dir": str(tmp_path),
        "defaults": {"provider": "manual"},
        "loops": [
            {
                "loop_id": "loop_001",
                "max_rounds": 2,
                "proposers": [
                    {
                        "id": "proposer_001",
                        "prompt": {"system": "system", "template": "work round {round_number}"},
                        "output_schema": {"type": "object"},
                    }
                ],
                "reviewers": [
                    {
                        "id": "reviewer_001",
                        "prompt": {"system": "system", "template": "review"},
                        "output_schema": {"type": "object"},
                    }
                ],
                "caller_context": {"user_intent": "test"},
            }
        ],
    }


def test_progress_contract_requires_meaningful_out_of_band_milestones() -> None:
    contract = runtime_progress_contract()

    assert "out-of-band" in contract
    assert "what completed" in contract
    assert "concrete result or evidence" in contract
    assert "artifact or checkpoint" in contract
    assert "what happens next" in contract
    assert "blocker" in contract
    assert "still alive" in contract
    assert "private chain-of-thought" in contract
    assert "strict structured data" in contract


def test_progress_contract_injection_is_idempotent() -> None:
    effective = ensure_runtime_progress_contract("Do the task.\n")

    assert has_runtime_progress_contract(effective)
    assert effective.count(RUNTIME_PROGRESS_CONTRACT_MARKER) == 1
    assert ensure_runtime_progress_contract(effective) == effective


def test_progress_contract_changes_effective_prompt_without_changing_caller_text() -> None:
    caller_prompt = "Return exactly one JSON object."
    effective = ensure_runtime_progress_contract(caller_prompt)

    assert effective.startswith(caller_prompt)
    assert effective != caller_prompt
    assert effective.endswith("</arc_llm_runtime_progress_contract>\n")
    assert sha256_text(effective) != sha256_text(caller_prompt)
    assert sha256_text(ensure_runtime_progress_contract(effective)) == sha256_text(effective)


def test_worker_prompts_include_contract_once_and_initial_contract_is_static(tmp_path) -> None:
    loop = load_batch_config(_batch_payload(tmp_path)).loops[0]
    worker = loop.proposers[0]
    context = proposer_context(loop=loop, worker=worker, round_number=1, correspondence=[])

    legacy = render_prompt(worker, context)
    initial, _context, static_prefix = render_initial_worker_prompt(
        loop=loop,
        worker=worker,
        role="proposer",
        round_number=1,
    )
    delta, _context = render_proposer_delta_prompt(
        loop=loop,
        worker=worker,
        round_number=2,
        correspondence=[],
    )

    assert legacy.count(RUNTIME_PROGRESS_CONTRACT_MARKER) == 1
    assert initial.count(RUNTIME_PROGRESS_CONTRACT_MARKER) == 1
    assert static_prefix.count(RUNTIME_PROGRESS_CONTRACT_MARKER) == 1
    assert delta.count(RUNTIME_PROGRESS_CONTRACT_MARKER) == 1
    assert initial.startswith(static_prefix)
