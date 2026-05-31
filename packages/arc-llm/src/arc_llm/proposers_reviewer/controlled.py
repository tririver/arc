from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .template_materializer import materialize_batch, materialize_loop


class ReviewController(Protocol):
    def initial_state(self) -> dict[str, Any]: ...

    def build_attempt_context(self, *, attempt_number: int, state: Mapping[str, Any]) -> dict[str, Any]: ...

    def select_active_proposers(self, *, attempt_number: int, state: Mapping[str, Any]) -> list[str]: ...

    def on_attempt_result(
        self,
        *,
        attempt_number: int,
        batch_result: Mapping[str, Any],
        proposer_outputs: Mapping[str, Any],
        review: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> dict[str, Any]: ...


def run_controlled_proposers_reviewer(
    *,
    run_id: str,
    run_dir: Path | str,
    controller: ReviewController,
    loop_template: Mapping[str, Any],
    proposer_templates: Mapping[str, Mapping[str, Any]],
    reviewer_template: Mapping[str, Any],
    max_attempts: int,
    session_scope_id: str,
    defaults: Mapping[str, Any] | None = None,
    artifact_options: Mapping[str, Any] | None = None,
    batch_runner: Callable[..., dict[str, Any]] | None = None,
    json_runner: Callable[..., dict[str, Any]] | None = None,
    base_env: Mapping[str, str] | None = None,
    process_chain: list[str] | None = None,
) -> dict[str, Any]:
    from .runner import run_proposers_reviewer_batch

    run_batch = batch_runner or run_proposers_reviewer_batch
    root = Path(run_dir)
    state = controller.initial_state()
    attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, max_attempts + 1):
        active = controller.select_active_proposers(attempt_number=attempt_number, state=state) or sorted(proposer_templates)
        missing = [worker_id for worker_id in active if worker_id not in proposer_templates]
        if missing:
            raise ValueError(f"controller selected unknown proposer ids: {', '.join(missing)}")
        caller_context = controller.build_attempt_context(attempt_number=attempt_number, state=state)
        loop = materialize_loop(
            loop_template,
            loop_id=f"{run_id}_attempt_{attempt_number:03d}",
            caller_context=caller_context,
            proposers=[proposer_templates[item] for item in active],
            reviewers=[reviewer_template],
        )
        batch = materialize_batch(
            run_id=f"{run_id}_attempt_{attempt_number:03d}",
            run_dir=root / "attempt_batches",
            loops=[loop],
            defaults=defaults,
            artifact_options=artifact_options,
            session={
                "policy": "stateful",
                "history_mode": "delta",
                "scope_id": session_scope_id,
                "reuse_across_batch_calls": True,
                "root": str(root / "llm_sessions"),
            },
            max_concurrent_loops=1,
        )
        batch_result = run_batch(
            batch,
            json_runner=json_runner,
            base_env=base_env,
            process_chain=process_chain,
        )
        proposer_outputs, review = _extract_attempt_artifacts(batch_result)
        decision = controller.on_attempt_result(
            attempt_number=attempt_number,
            batch_result=batch_result,
            proposer_outputs=proposer_outputs,
            review=review,
            state=state,
        )
        attempts.append({"attempt_number": attempt_number, "batch_result": batch_result, "decision": decision})
        state = dict(decision.get("next_state") or state)
        if decision.get("action") in {"accept", "fail", "block"}:
            return {"schema_version": "arc.llm.controlled_result.v1", "status": decision["action"], "attempts": attempts}
    return {"schema_version": "arc.llm.controlled_result.v1", "status": "exhausted", "attempts": attempts}


class JsonPolicyReviewController:
    def __init__(self, policy: Mapping[str, Any], initial_state: Mapping[str, Any] | None = None) -> None:
        self.policy = dict(policy)
        self._initial_state = dict(initial_state or {})

    def initial_state(self) -> dict[str, Any]:
        return dict(self._initial_state)

    def build_attempt_context(self, *, attempt_number: int, state: Mapping[str, Any]) -> dict[str, Any]:
        context = dict(state)
        context["attempt_number"] = attempt_number
        return context

    def select_active_proposers(self, *, attempt_number: int, state: Mapping[str, Any]) -> list[str]:
        active = state.get("next_active_proposer_ids") or self.policy.get("initial_active_proposer_ids") or []
        return _string_list(active)

    def on_attempt_result(
        self,
        *,
        attempt_number: int,
        batch_result: Mapping[str, Any],
        proposer_outputs: Mapping[str, Any],
        review: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        status = _get_path(review, str(self.policy.get("status_path") or ""))
        accept_statuses = set(_string_list(self.policy.get("accept_statuses") or []))
        retry_statuses = set(_string_list(self.policy.get("retry_statuses") or []))
        if status in accept_statuses:
            return {
                "action": "accept",
                "accepted_output": _get_path(review, str(self.policy.get("accepted_output_path") or "")),
                "next_state": dict(state),
                "result": {"status": status},
            }
        if status in retry_statuses:
            next_state = dict(state)
            next_state["last_status"] = status
            next_state["retry_feedback"] = _retry_feedback(self.policy, review)
            next_active = _string_list(_get_path(review, str(self.policy.get("next_active_proposer_ids_path") or "")))
            if next_active:
                next_state["next_active_proposer_ids"] = next_active
            locked = dict(state.get("locked_outputs") or {})
            locked.update(_locked_outputs(self.policy, str(status), proposer_outputs, review))
            if locked:
                next_state["locked_outputs"] = locked
            return {
                "action": "retry",
                "next_state": next_state,
                "retry_feedback": next_state["retry_feedback"],
                "result": {"status": status},
            }
        return {
            "action": "block",
            "next_state": dict(state),
            "result": {"status": status, "batch_status": batch_result.get("status")},
        }


def _extract_attempt_artifacts(batch_result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_root = batch_result.get("run_root")
    loops = batch_result.get("loops")
    if not run_root or not isinstance(loops, list) or not loops:
        return {}, {}
    loop = loops[0] if isinstance(loops[0], Mapping) else {}
    loop_id = str(loop.get("loop_id") or "")
    rounds_completed = int(loop.get("rounds_completed") or 0)
    if not loop_id or rounds_completed <= 0:
        return {}, {}
    round_root = Path(str(run_root)) / "loops" / loop_id / "rounds" / f"round_{rounds_completed:03d}"
    proposer_outputs = {
        path.stem: payload
        for path in sorted((round_root / "proposer_outputs").glob("*.json"))
        if isinstance((payload := _read_json(path)), dict)
    }
    reviews = [
        payload
        for path in sorted((round_root / "reviews").glob("*.json"))
        if isinstance((payload := _read_json(path)), dict)
    ]
    return proposer_outputs, (reviews[0] if reviews else {})


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _retry_feedback(policy: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    paths = policy.get("retry_feedback_paths")
    if not isinstance(paths, Mapping):
        return {}
    return {str(name): _get_path(review, str(path)) for name, path in paths.items()}


def _locked_outputs(
    policy: Mapping[str, Any],
    status: str,
    proposer_outputs: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    lock_policy = policy.get("lock_outputs_when_status")
    if not isinstance(lock_policy, Mapping):
        return {}
    status_policy = lock_policy.get(status)
    if not isinstance(status_policy, Mapping):
        return {}
    ids = _string_list(_get_path(review, str(status_policy.get("ids_path") or "")))
    return {proposer_id: proposer_outputs[proposer_id] for proposer_id in ids if proposer_id in proposer_outputs}


def _get_path(value: Any, dotted_path: str) -> Any:
    current = value
    if not dotted_path:
        return None
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]
