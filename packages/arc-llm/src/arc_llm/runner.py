from __future__ import annotations

import json
import hashlib
import re
import time
from dataclasses import dataclass, replace
import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .call_record import ARC_LLM_CALL_RECORD_SCHEMA_VERSION, attach_arc_llm_call_record, strip_arc_llm_call_records
from .attempt_diagnostics import (
    AttemptDiagnosticRef,
    AttemptDiagnostics,
    bind_attempt_diagnostics,
    current_attempt_diagnostics,
    sanitize_diagnostic_text,
)
from .call_checkpoint import (
    LLMCallNeedsSupervision,
    LLMCallRetryExhausted,
    SupervisedNativeResumeAuthorization,
    checkpoint_path,
    normalize_supervised_native_resume_authorization,
    prepare_call,
    record_failure,
    record_response,
    record_submitted,
    record_validated,
)
from .host import HostDetection, select_llm_provider
from .json_schema import (
    ProviderJSONSchemaPlan,
    plan_provider_json_schema,
    provider_uses_native_schema,
    validate_local_json_schema,
    with_canonical_json_schema_contract,
)
from .model import reasoning_effort_for_model_tier, resolve_model_with_warnings
from .nested_shell_capability import (
    NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
    NESTED_SHELL_PROBE_ID,
    NestedShellCapability,
    apply_runtime_capability,
    capability_runtime_identity,
    clear_runtime_capability_values,
    nested_shell_warning_from_codex_events,
    render_nested_shell_prompt,
    resolve_nested_shell_capability,
)
from .paper_access_policy import (
    PAPER_ACCESS_POLICY_VERSION,
    canonical_paper_access_policy,
)
from .providers.activity import resolve_idle_timeout_seconds
from .providers.base import (
    LLMConfigurationError,
    LLMSchemaError,
    LLMWorkerCancelled,
    LLMWorkerError,
    LLMWorkerTimeout,
)
from .progress_journal import ProgressJournal
from .progress_prompt import apply_runtime_progress_contract
from .response_candidates import (
    LLMResponseCandidateConflict,
    LLMResponseCandidateReceiptError,
    persist_selection_receipt,
    select_response_candidate,
)
from .runtime_manifest import RUNTIME_MANIFEST_VERSION
from .providers.registry import get_provider_spec
from .providers.select import select_provider
from .schema_cache import canonical_json, schema_hash, sha256_text
from .schema_canary import SchemaCanaryIdentity, run_schema_canary
from .safety import LLMSafetyController
from .sessions import (
    LLMSessionManager, LLMSessionRef, legacy_runtime_fingerprint,
    runtime_fingerprint,
)
from .structured_recovery import recover_json_output, structured_metadata
from .usage import LLMProviderResponse, LLMUsage


MAX_ATTEMPTS_PER_PROVIDER = 1
RETRY_INTERVAL_SECONDS = 10
LOW_CONTENT_TOKEN_THRESHOLD = 10
NATIVE_RESUME_RECONCILIATION_PROMPT = (
    "Supervised native-session recovery: the preceding turn may already have run. "
    "Do not repeat its work. Reconcile the native session and return the final answer "
    "for that preceding request in the required format."
)
def _runtime_capabilities(env: Mapping[str, str] | None) -> dict[str, Any]:
    values = env or {}
    return {
        "arc_paper_cli_access": values.get("ARC_PAPER_CLI_ACCESS", "full"),
        "arc_paper_access": "full",
        "inherit_host_tools": values.get("ARC_LLM_INHERIT_HOST_TOOLS", "false").strip().lower()
        == "true",
        **capability_runtime_identity(values),
    }


def _runtime_compatibility_policy(
    env: Mapping[str, str] | None,
    *,
    session_policy: str,
    session_manager: LLMSessionManager | None,
    session_key: str | None,
    session_metadata: Mapping[str, Any] | None,
    artifact_dir: Path | None,
    idempotency_key: str | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Default new calls to paper access while keeping legacy resumptions closed."""
    effective_env = dict(os.environ if env is None else env)
    clear_runtime_capability_values(effective_env)
    effective_env.setdefault("ARC_PAPER_CLI_ACCESS", "full")
    effective_env.setdefault("ARC_LLM_INHERIT_HOST_TOOLS", "false")
    legacy_resume = False

    if session_policy == "stateful" and session_manager is not None and session_key:
        existing = session_manager.get_existing(session_key)
        if existing is not None and "arc_runtime_capabilities" not in existing.metadata:
            legacy_resume = True

    if artifact_dir is not None and idempotency_key:
        checkpoint = artifact_dir / "call-checkpoints" / f"idempotency-{sha256_text(idempotency_key)}.json"
        if checkpoint.exists():
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if "runtime_capabilities" not in payload:
                legacy_resume = True

    if legacy_resume:
        # Before this capability was serialized, ARC workers had no direct
        # paper CLI. Never enlarge such a run merely because it was resumed by
        # a newer package version.
        effective_env["ARC_PAPER_CLI_ACCESS"] = "none"
        effective_env["ARC_LLM_INHERIT_HOST_TOOLS"] = "false"

    effective_metadata = dict(session_metadata or {})
    effective_metadata["arc_runtime_capabilities"] = _runtime_capabilities(effective_env)
    effective_metadata["arc_runtime_manifest_version"] = RUNTIME_MANIFEST_VERSION
    return effective_env, effective_metadata


def _resolve_request_nested_shell(
    configs: Sequence["LLMConfig"], env: dict[str, str]
) -> NestedShellCapability:
    provider = configs[0].provider
    if env.get("ARC_PAPER_CLI_ACCESS", "full") != "full":
        capability = NestedShellCapability(
            schema_version=NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
            provider=provider,
            nested_sandboxed_shell=False,
            status="not_requested",
            probe_kind="none",
            probe_identity=NESTED_SHELL_PROBE_ID,
            warning="nested_shell.not_requested",
        )
    else:
        capability = resolve_nested_shell_capability(
            provider=provider,
            env=env,
            cwd=env.get("ARC_CODEX_WORK_DIR"),
        )
    apply_runtime_capability(env, capability)
    return capability


def _controller_evidence_exposed(schema: Mapping[str, Any] | None) -> bool:
    if not isinstance(schema, Mapping):
        return False
    properties = schema.get("properties")
    return isinstance(properties, Mapping) and "arc_evidence_requests" in properties


def prepare_runtime_prompt(
    prompt: str,
    *,
    provider: str,
    model: str | None,
    model_tier: str | None,
    env: dict[str, str],
    process_chain: Sequence[str] | None = None,
    artifact_dir: Path | None = None,
    session_manager: LLMSessionManager | None = None,
    schema: Mapping[str, Any] | None = None,
    static_prefix: str | None = None,
    stage_paper_worker: bool = True,
) -> tuple[str, str | None, NestedShellCapability]:
    """Render the portable marker before custom-runner artifacts or hashing."""

    clear_runtime_capability_values(env)
    configs = resolve_llm_configs(
        provider=provider,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    if stage_paper_worker:
        _configure_paper_worker_session(
            env, artifact_dir=artifact_dir, session_manager=session_manager
        )
    capability = _resolve_request_nested_shell(configs, env)
    exposed = _controller_evidence_exposed(schema)
    rendered = render_nested_shell_prompt(
        prompt, capability, controller_evidence_exposed=exposed
    )
    rendered_prefix = (
        render_nested_shell_prompt(
            static_prefix, capability, controller_evidence_exposed=exposed
        )
        if static_prefix is not None
        else None
    )
    return rendered, rendered_prefix, capability


def _nested_shell_warnings(
    env: Mapping[str, str] | None,
    *,
    raw_events: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, ...]:
    identity = capability_runtime_identity(env)
    warnings: list[str] = []
    if (
        (env or {}).get("ARC_PAPER_CLI_ACCESS", "full") == "full"
        and identity["nested_shell_status"] != "available"
    ):
        warnings.append(f"nested_shell.{identity['nested_shell_status']}")
    typed_warning = nested_shell_warning_from_codex_events(raw_events)
    if typed_warning is not None:
        warnings.append(typed_warning)
    return tuple(dict.fromkeys(warnings))


def _configure_paper_worker_session(
    env: dict[str, str],
    *,
    artifact_dir: Path | None,
    session_manager: LLMSessionManager | None,
) -> None:
    """Create the arc-paper overlay contract without importing arc-paper."""
    access = env.get("ARC_PAPER_CLI_ACCESS", "none")
    if access == "full" and env.get("ARC_PAPER_WORKER_SESSION_DIR"):
        return
    # A stateful session can span calls whose artifact directories differ (for
    # example, one directory per segment or round).  Keep its paper-worker
    # overlay rooted with the session manager so controller paths and the
    # runtime fingerprint remain stable for every turn in that session.
    location = session_manager.root if session_manager is not None else artifact_dir
    if location is None:
        from .paths import llm_tmp_root

        location = llm_tmp_root(env) / "paper-worker-isolation"

    run_root = _paper_worker_run_root(location)
    if access != "full":
        disabled_cache = run_root / "paper-cache-disabled"
        disabled_cache.mkdir(parents=True, exist_ok=True)
        env["ARC_PAPER_CACHE"] = str(disabled_cache)
        env["ARC_LLM_WORKER_CONTEXT"] = "true"
        for key in (
            "ARC_PAPER_WORKER_BASE_CACHE",
            "ARC_PAPER_WORKER_SESSION_DIR",
            "ARC_PAPER_WORKER_TOMBSTONE_DIR",
            "ARC_PAPER_WORKER_SESSION_ID",
        ):
            env.pop(key, None)
        return

    base_cache = _paper_base_cache(env)
    overlay = run_root / "paper-cache-overlay"
    state = overlay / ".arc-paper-worker"
    tombstones = state / "tombstones"
    for path in (overlay, state, tombstones):
        path.mkdir(parents=True, exist_ok=True)
    session_id = f"arc-llm-{sha256_text(str(run_root.resolve(strict=False)))[:20]}"
    env.update({
        "ARC_PAPER_WORKER_BASE_CACHE": str(base_cache),
        "ARC_PAPER_WORKER_SESSION_DIR": str(run_root),
        "ARC_PAPER_WORKER_TOMBSTONE_DIR": str(tombstones),
        "ARC_PAPER_WORKER_SESSION_ID": session_id,
        "ARC_LLM_WORKER_CONTEXT": "true",
        "ARC_PAPER_CACHE": str(overlay),
    })
    if env.get("ARC_LLM_INHERIT_HOST_TOOLS", "false").strip().lower() != "true":
        # Keep Codex writes inside the run tree. The global base cache remains
        # outside the sandbox and is reachable only through the wrapper's
        # validated read/promotion contract.
        env["ARC_CODEX_SANDBOX"] = "workspace-write"
        env["ARC_CODEX_WORK_DIR"] = str(run_root)
        env.pop("ARC_CODEX_ADD_DIRS", None)


def _stage_paper_access_policy(
    env: dict[str, str], policy: Mapping[str, Any] | None
) -> None:
    """Stage a content-addressed read policy inside the authorized run root."""
    if policy is None:
        return
    if env.get("ARC_PAPER_CLI_ACCESS") != "full":
        raise ValueError("paper_access_policy requires ARC paper CLI access")
    raw_root = env.get("ARC_PAPER_WORKER_SESSION_DIR")
    if not raw_root:
        raise ValueError("paper_access_policy requires a paper worker session directory")
    canonical = canonical_paper_access_policy(policy)
    payload = canonical_json(canonical).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    root = Path(raw_root).expanduser().resolve(strict=False)
    policy_dir = root / "read-policies"
    if policy_dir.is_symlink():
        raise ValueError("paper access policy directory must not be a symbolic link")
    policy_dir.mkdir(parents=True, exist_ok=True)
    if not policy_dir.is_dir() or policy_dir.is_symlink():
        raise ValueError("paper access policy directory is not a regular directory")
    destination = policy_dir / f"sha256-{digest}.json"
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise ValueError("paper access policy destination is not a regular file")
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            _atomic_write_bytes(destination, payload)
    else:
        _atomic_write_bytes(destination, payload)
    env.pop("ARC_PAPER_WORKER_ALLOWED_OPERATIONS_JSON", None)
    env.pop("ARC_PAPER_WORKER_ALLOWED_TARGETS_JSON", None)
    env["ARC_PAPER_WORKER_READ_POLICY_PATH"] = str(destination)
    env["ARC_PAPER_WORKER_READ_POLICY_SHA256"] = digest
    env["ARC_PAPER_WORKER_READ_POLICY_SCHEMA"] = PAPER_ACCESS_POLICY_VERSION


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass




def _finalize_paper_worker_call(
    env: Mapping[str, str] | None,
    *,
    status: str,
    worker_id: str | None,
    call_id: str | None,
) -> None:
    values = env or {}
    if values.get("ARC_PAPER_CLI_ACCESS") != "full":
        return
    session_id = values.get("ARC_PAPER_WORKER_SESSION_ID")
    run_root = values.get("ARC_PAPER_WORKER_SESSION_DIR")
    base_root = values.get("ARC_PAPER_WORKER_BASE_CACHE")
    if not session_id or not run_root or not base_root:
        return
    if not _paper_overlay_has_staged_changes(Path(run_root)):
        return
    command = [
        sys.executable,
        "-m",
        "arc_paper.worker_controller",
        "finalize",
        "--run-root",
        run_root,
        "--base-root",
        base_root,
        "--session-id",
        str(session_id),
        "--worker-id",
        worker_id or "",
        "--call-id",
        call_id or "",
        "--status",
        status,
    ]
    controller_env = dict(os.environ)
    for key in tuple(controller_env):
        if key.startswith("ARC_PAPER_WORKER_") or key in {
            "ARC_PAPER_CLI_ACCESS",
            "ARC_PAPER_CACHE",
            "ARC_LLM_WORKER_CONTEXT",
        }:
            controller_env.pop(key, None)
    completed = subprocess.run(
        command,
        env=controller_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "controller finalizer failed"
        raise RuntimeError(detail)


def _paper_overlay_has_staged_changes(run_root: Path) -> bool:
    overlay = run_root / "paper-cache-overlay"
    if not overlay.is_dir():
        return False
    return any(
        path.is_file()
        and not (path.parent.name == ".arc-paper-worker" and path.name == "session.json")
        for path in overlay.rglob("*")
    )


def _paper_worker_run_root(location: Path) -> Path:
    candidate = location.expanduser().resolve(strict=False)
    if candidate.suffix:
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / "manifest.json").exists() and (parent / "config.json").exists():
            return parent
    return candidate


def _paper_base_cache(env: Mapping[str, str]) -> Path:
    if value := env.get("ARC_PAPER_WORKER_BASE_CACHE"):
        return Path(value).expanduser().resolve(strict=False)
    if value := env.get("ARC_PAPER_CACHE"):
        return Path(value).expanduser().resolve(strict=False)
    if value := env.get("ARC_HOME"):
        return (Path(value).expanduser() / "cache" / "arc-paper").resolve(strict=False)
    if value := env.get("XDG_CACHE_HOME"):
        return (Path(value).expanduser() / "arc" / "arc-paper").resolve(strict=False)
    home = Path(env["HOME"]).expanduser() if env.get("HOME") else Path.home()
    return (home / ".cache" / "arc" / "arc-paper").resolve(strict=False)


class LLMTaskError(RuntimeError):
    pass


class LLMNeedsLLM(LLMTaskError):
    """Automatic provider selection found no runnable host LLM."""

    def __init__(self, config: "LLMConfig") -> None:
        super().__init__(
            "provider=auto resolved to manual; run from a supported agent host "
            "or select an explicit provider"
        )
        self.config = config


class LLMOutputValidationError(RuntimeError):
    pass


class LLMRetryableProviderOutputError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None
    host: HostDetection
    signals: list[str]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMAttemptFailure:
    provider: str
    attempt: int
    error: str


@dataclass(frozen=True)
class LLMCallOutcome:
    value: Any
    usage: LLMUsage
    native_session_id: str | None
    session_policy: str
    session_key: str | None
    call_label: str | None
    prompt_sha256: str | None
    static_prefix_sha256: str | None
    schema_sha256: str | None
    runtime_fingerprint: str | None
    idempotency_key: str | None = None
    generation: int | None = None
    prompt_bytes: int | None = None
    logical_receipt: dict[str, Any] | None = None
    call_record: dict[str, Any] | None = None
    structured_output: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()


def resolve_llm_config(
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
) -> LLMConfig:
    if provider == "auto" and model:
        raise ValueError("Exact model requires explicit provider; use provider=<provider> or model_tier=<low|medium|high|max>.")
    selected = select_llm_provider(
        env=env,
        process_chain=process_chain,
        explicit_provider=None if provider == "auto" else provider,
    )
    model_resolution = resolve_model_with_warnings(
        selected.provider,
        model,
        model_tier=model_tier,
        env=env,
    )
    spec = get_provider_spec(selected.provider)
    return LLMConfig(
        provider=selected.provider,
        model=model_resolution.model,
        host=selected.host,
        signals=selected.signals,
        warnings=(*spec.warning_codes, *model_resolution.warnings),
    )


def resolve_llm_configs(
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
) -> list[LLMConfig]:
    return [
        resolve_llm_config(
            provider=provider,
            model=model,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
        )
    ]


def run_json(
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    validate_schema: bool = True,
    output_recovery: str = "strict",
    schema_formatter_enabled: bool = True,
    role_hint: str | None = None,
    paper_access_policy: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    session_policy: str = "stateless",
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
    session_name: str | None = None,
    session_metadata: Mapping[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    schema_canary_root: Path | str | None = None,
    call_label: str | None = None,
    static_prefix: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    progress_contract_scope: str = "call",
    supervised_native_resume: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
    initial_native_authorization: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
    validated_legacy_logical_identity: Mapping[str, Any] | None = None,
    validated_legacy_runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning the JSON value with its audit record."""

    outcome = run_json_result(
        prompt,
        schema=schema,
        provider=provider,
        model=model,
        model_tier=model_tier,
        validate_schema=validate_schema,
        output_recovery=output_recovery,
        schema_formatter_enabled=schema_formatter_enabled,
        role_hint=role_hint,
        paper_access_policy=paper_access_policy,
        env=env,
        process_chain=process_chain,
        session_policy=session_policy,
        session_manager=session_manager,
        session_key=session_key,
        session_name=session_name,
        session_metadata=session_metadata,
        artifact_dir=artifact_dir,
        schema_canary_root=schema_canary_root,
        call_label=call_label,
        static_prefix=static_prefix,
        idle_timeout_seconds=idle_timeout_seconds,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        idempotency_key=idempotency_key,
        progress_contract_scope=progress_contract_scope,
        supervised_native_resume=supervised_native_resume,
        initial_native_authorization=initial_native_authorization,
        validated_legacy_logical_identity=validated_legacy_logical_identity,
        validated_legacy_runtime_identity=validated_legacy_runtime_identity,
    )
    return attach_arc_llm_call_record(outcome.value, dict(outcome.call_record or {}))


def run_json_result(
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    validate_schema: bool = True,
    output_recovery: str = "strict",
    schema_formatter_enabled: bool = True,
    role_hint: str | None = None,
    paper_access_policy: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    session_policy: str = "stateless",
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
    session_name: str | None = None,
    session_metadata: Mapping[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    schema_canary_root: Path | str | None = None,
    call_label: str | None = None,
    static_prefix: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    progress_contract_scope: str = "call",
    supervised_native_resume: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
    initial_native_authorization: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
    validated_legacy_logical_identity: Mapping[str, Any] | None = None,
    validated_legacy_runtime_identity: Mapping[str, Any] | None = None,
) -> LLMCallOutcome:
    supervised_native_resume = normalize_supervised_native_resume_authorization(
        supervised_native_resume
    )
    initial_native_authorization = normalize_supervised_native_resume_authorization(
        initial_native_authorization
    )
    if session_policy not in {"stateless", "stateful"}:
        raise ValueError("session_policy must be stateless or stateful")
    if output_recovery not in {"strict", "warn"}:
        raise ValueError("output_recovery must be strict or warn")
    if session_policy == "stateful" and (session_manager is None or not session_key):
        raise ValueError("stateful run_json requires session_manager and session_key")
    if progress_contract_scope not in {"call", "session"}:
        raise ValueError("progress_contract_scope must be call or session")
    if session_policy == "stateful" and (not idempotency_key or artifact_dir is None):
        raise ValueError("stateful run_json requires idempotency_key and artifact_dir")
    if idempotency_key and artifact_dir is None:
        raise ValueError("idempotency_key requires artifact_dir")
    if (
        supervised_native_resume is not None
        or initial_native_authorization is not None
    ) and session_policy != "stateful":
        raise ValueError("native authorization requires a stateful session")
    if initial_native_authorization is not None and (
        initial_native_authorization.session_key != session_key
        or initial_native_authorization.idempotency_key != idempotency_key
    ):
        raise ValueError(
            "initial_native_authorization does not match session_key/idempotency_key"
        )
    if supervised_native_resume is not None and (
        initial_native_authorization is None
        or supervised_native_resume != initial_native_authorization
    ):
        raise ValueError(
            "supervised_native_resume requires the exact complete initial_native_authorization"
        )
    env, session_metadata = _runtime_compatibility_policy(
        env,
        session_policy=session_policy,
        session_manager=session_manager,
        session_key=session_key,
        session_metadata=session_metadata,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        idempotency_key=idempotency_key,
    )
    configs = resolve_llm_configs(
        provider=provider,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    _raise_if_auto_resolved_manual(provider, configs)
    try:
        validate_local_json_schema(schema)
        schema_plans = {
            config.provider: plan_provider_json_schema(
                schema,
                provider=config.provider,
                uses_native_schema=provider_uses_native_schema(
                    config.provider,
                    supports_native_schema=get_provider_spec(
                        config.provider
                    ).supports_native_schema,
                    env=env,
                    output_recovery=output_recovery,
                ),
            )
            for config in configs
        }
    except ValueError as exc:
        raise LLMSchemaError(str(exc)) from exc
    _validate_runtime_inputs_before_artifacts(
        configs, env=env, idle_timeout_seconds=idle_timeout_seconds
    )
    # Controller staging below injects call-local cache/work paths. They must
    # remain in the ordinary runtime fingerprint used for session safety, but
    # must not split one batch-wide provider/schema admission contract into a
    # separate canary per worker artifact directory.
    schema_canary_runtime_env = dict(env)
    _configure_paper_worker_session(
        env,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        session_manager=session_manager,
    )
    _stage_paper_access_policy(env, paper_access_policy)
    _validate_runtime_preflight(configs, env=env, idle_timeout_seconds=idle_timeout_seconds)
    nested_shell = _resolve_request_nested_shell(configs, env)
    apply_runtime_capability(schema_canary_runtime_env, nested_shell)
    controller_exposed = _controller_evidence_exposed(schema)
    prompt = render_nested_shell_prompt(
        prompt, nested_shell, controller_evidence_exposed=controller_exposed
    )
    if static_prefix is not None:
        static_prefix = render_nested_shell_prompt(
            static_prefix, nested_shell, controller_evidence_exposed=controller_exposed
        )
    session_metadata["arc_runtime_capabilities"] = _runtime_capabilities(env)
    return _run_with_retries(
        configs,
        provider_requested=provider,
        model_requested=model,
        model_tier_requested=model_tier,
        attach_call_record=False,
        env=env,
        process_chain=process_chain,
        max_attempts=MAX_ATTEMPTS_PER_PROVIDER,
        return_outcome=True,
        idle_timeout_seconds=idle_timeout_seconds,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        paper_worker_id=session_name,
        paper_call_id=call_label or idempotency_key,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        diagnostic_call_label=call_label or idempotency_key,
        call=lambda selected, config, effective_idle_timeout_seconds: _generate_json(
            selected,
            prompt,
            schema=schema,
            schema_plan=schema_plans[config.provider],
            model=config.model,
            validate_schema=validate_schema,
            output_recovery=output_recovery,
            schema_formatter_enabled=schema_formatter_enabled,
            role_hint=role_hint,
            provider_used=config.provider,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
            session_policy=session_policy,
            session_manager=session_manager,
            session_key=session_key,
            session_name=session_name,
            session_metadata=session_metadata,
            artifact_dir=Path(artifact_dir) if artifact_dir else None,
            schema_canary_root=(
                Path(schema_canary_root)
                if schema_canary_root is not None
                else None
            ),
            schema_canary_runtime_fingerprint=_runtime_fp(
                provider_used=config.provider,
                model=config.model,
                model_tier=model_tier,
                env=schema_canary_runtime_env,
                process_chain=process_chain,
            ),
            call_label=call_label,
            static_prefix=static_prefix,
            idle_timeout_seconds=effective_idle_timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            idempotency_key=idempotency_key,
            progress_contract_scope=progress_contract_scope,
            supervised_native_resume=supervised_native_resume,
            initial_native_authorization=initial_native_authorization,
            validated_legacy_logical_identity=validated_legacy_logical_identity,
            validated_legacy_runtime_identity=validated_legacy_runtime_identity,
        ),
    )


def run_text(
    prompt: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    paper_access_policy: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    session_policy: str = "stateless",
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
    session_name: str | None = None,
    session_metadata: Mapping[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    call_label: str | None = None,
    static_prefix: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    progress_contract_scope: str = "call",
    supervised_native_resume: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
    initial_native_authorization: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
    validated_legacy_logical_identity: Mapping[str, Any] | None = None,
    validated_legacy_runtime_identity: Mapping[str, Any] | None = None,
) -> str:
    return run_text_result(
        prompt,
        provider=provider,
        model=model,
        model_tier=model_tier,
        paper_access_policy=paper_access_policy,
        env=env,
        process_chain=process_chain,
        session_policy=session_policy,
        session_manager=session_manager,
        session_key=session_key,
        session_name=session_name,
        session_metadata=session_metadata,
        artifact_dir=artifact_dir,
        call_label=call_label,
        static_prefix=static_prefix,
        idle_timeout_seconds=idle_timeout_seconds,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        idempotency_key=idempotency_key,
        progress_contract_scope=progress_contract_scope,
        supervised_native_resume=supervised_native_resume,
        initial_native_authorization=initial_native_authorization,
        validated_legacy_logical_identity=validated_legacy_logical_identity,
        validated_legacy_runtime_identity=validated_legacy_runtime_identity,
    ).value


def run_text_result(
    prompt: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    paper_access_policy: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    session_policy: str = "stateless",
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
    session_name: str | None = None,
    session_metadata: Mapping[str, Any] | None = None,
    artifact_dir: Path | str | None = None,
    call_label: str | None = None,
    static_prefix: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    progress_contract_scope: str = "call",
    supervised_native_resume: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
    initial_native_authorization: (
        SupervisedNativeResumeAuthorization | tuple[Any, ...] | None
    ) = None,
    validated_legacy_logical_identity: Mapping[str, Any] | None = None,
    validated_legacy_runtime_identity: Mapping[str, Any] | None = None,
) -> LLMCallOutcome:
    supervised_native_resume = normalize_supervised_native_resume_authorization(
        supervised_native_resume
    )
    initial_native_authorization = normalize_supervised_native_resume_authorization(
        initial_native_authorization
    )
    if session_policy not in {"stateless", "stateful"}:
        raise ValueError("session_policy must be stateless or stateful")
    if session_policy == "stateful" and (session_manager is None or not session_key):
        raise ValueError("stateful run_text requires session_manager and session_key")
    if progress_contract_scope not in {"call", "session"}:
        raise ValueError("progress_contract_scope must be call or session")
    if session_policy == "stateful" and (not idempotency_key or artifact_dir is None):
        raise ValueError("stateful run_text requires idempotency_key and artifact_dir")
    if idempotency_key and artifact_dir is None:
        raise ValueError("idempotency_key requires artifact_dir")
    if (
        supervised_native_resume is not None
        or initial_native_authorization is not None
    ) and session_policy != "stateful":
        raise ValueError("native authorization requires a stateful session")
    if initial_native_authorization is not None and (
        initial_native_authorization.session_key != session_key
        or initial_native_authorization.idempotency_key != idempotency_key
    ):
        raise ValueError(
            "initial_native_authorization does not match session_key/idempotency_key"
        )
    if supervised_native_resume is not None and (
        initial_native_authorization is None
        or supervised_native_resume != initial_native_authorization
    ):
        raise ValueError(
            "supervised_native_resume requires the exact complete initial_native_authorization"
        )
    env, session_metadata = _runtime_compatibility_policy(
        env,
        session_policy=session_policy,
        session_manager=session_manager,
        session_key=session_key,
        session_metadata=session_metadata,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        idempotency_key=idempotency_key,
    )
    configs = resolve_llm_configs(
        provider=provider,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    _raise_if_auto_resolved_manual(provider, configs)
    _validate_runtime_inputs_before_artifacts(
        configs, env=env, idle_timeout_seconds=idle_timeout_seconds
    )
    _configure_paper_worker_session(
        env,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        session_manager=session_manager,
    )
    _stage_paper_access_policy(env, paper_access_policy)
    _validate_runtime_preflight(configs, env=env, idle_timeout_seconds=idle_timeout_seconds)
    nested_shell = _resolve_request_nested_shell(configs, env)
    prompt = render_nested_shell_prompt(
        prompt, nested_shell, controller_evidence_exposed=False
    )
    if static_prefix is not None:
        static_prefix = render_nested_shell_prompt(
            static_prefix, nested_shell, controller_evidence_exposed=False
        )
    session_metadata["arc_runtime_capabilities"] = _runtime_capabilities(env)
    return _run_with_retries(
        configs,
        provider_requested=provider,
        model_requested=model,
        model_tier_requested=model_tier,
        attach_call_record=False,
        env=env,
        process_chain=process_chain,
        max_attempts=1 if session_policy == "stateful" else MAX_ATTEMPTS_PER_PROVIDER,
        return_outcome=True,
        idle_timeout_seconds=idle_timeout_seconds,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        paper_worker_id=session_name,
        paper_call_id=call_label or idempotency_key,
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        diagnostic_call_label=call_label or idempotency_key,
        call=lambda selected, config, effective_idle_timeout_seconds: _generate_text(
            selected,
            prompt,
            model=config.model,
            provider_used=config.provider,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
            session_policy=session_policy,
            session_manager=session_manager,
            session_key=session_key,
            session_name=session_name,
            session_metadata=session_metadata,
            artifact_dir=Path(artifact_dir) if artifact_dir else None,
            call_label=call_label,
            static_prefix=static_prefix,
            idle_timeout_seconds=effective_idle_timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            idempotency_key=idempotency_key,
            progress_contract_scope=progress_contract_scope,
            supervised_native_resume=supervised_native_resume,
            initial_native_authorization=initial_native_authorization,
            validated_legacy_logical_identity=validated_legacy_logical_identity,
            validated_legacy_runtime_identity=validated_legacy_runtime_identity,
        ),
    )


def _raise_if_auto_resolved_manual(provider_requested: str, configs: Sequence[LLMConfig]) -> None:
    if provider_requested == "auto" and configs and configs[0].provider == "manual":
        raise LLMNeedsLLM(configs[0])


def _validate_runtime_preflight(
    configs: Sequence[LLMConfig],
    *,
    env: Mapping[str, str],
    idle_timeout_seconds: float | None,
) -> None:
    """Validate the complete provider runtime after controller staging."""
    _validate_runtime_inputs_before_artifacts(
        configs, env=env, idle_timeout_seconds=idle_timeout_seconds
    )


def _validate_runtime_inputs_before_artifacts(
    configs: Sequence[LLMConfig],
    *,
    env: Mapping[str, str],
    idle_timeout_seconds: float | None,
) -> None:
    """Reject malformed side-effect-free inputs before creating run artifacts."""
    for config in configs:
        provider_env = _env_with_tier_reasoning_default(env, config.provider, None)
        try:
            resolve_idle_timeout_seconds(
                idle_timeout_seconds,
                env=provider_env,
                provider=config.provider,
            )
        except (TypeError, ValueError) as exc:
            raise LLMConfigurationError(str(exc)) from exc


def _run_with_retries(
    configs: Sequence[LLMConfig],
    *,
    provider_requested: str,
    model_requested: str | None,
    model_tier_requested: str | None,
    attach_call_record: bool,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
    max_attempts: int = MAX_ATTEMPTS_PER_PROVIDER,
    return_outcome: bool = False,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    paper_worker_id: str | None = None,
    paper_call_id: str | None = None,
    artifact_dir: Path | None = None,
    diagnostic_call_label: str | None = None,
    call: Callable[[Any, LLMConfig, float], Any],
) -> Any:
    del progress_callback
    failures: list[LLMAttemptFailure] = []
    attempt_records: list[dict[str, Any]] = []
    attempt_diagnostic_refs: list[dict[str, str]] = []
    last_error: Exception | None = None
    for fallback_index, config in enumerate(configs):
        try:
            _check_cancel(cancel_check)
        except BaseException as exc:
            _attach_attempt_diagnostic_refs(exc, attempt_diagnostic_refs)
            raise
        try:
            provider_env = _env_with_tier_reasoning_default(
                env, config.provider, model_tier_requested
            )
            diagnostic_env = os.environ if provider_env is None else provider_env
            effective_idle_timeout_seconds = resolve_idle_timeout_seconds(
                idle_timeout_seconds,
                env=provider_env,
                provider=config.provider,
            )
            selected = select_provider(
                config.provider, env=provider_env, process_chain=process_chain
            )
        except BaseException as exc:
            _attach_attempt_diagnostic_refs(exc, attempt_diagnostic_refs)
            if isinstance(exc, (TypeError, ValueError)):
                configured = LLMConfigurationError(str(exc))
                _attach_attempt_diagnostic_refs(configured, attempt_diagnostic_refs)
                raise configured from exc
            raise
        for attempt in range(1, max_attempts + 1):
            try:
                diagnostics = (
                    AttemptDiagnostics(
                        artifact_dir,
                        provider=config.provider,
                        model=config.model,
                        fallback_index=fallback_index,
                        attempt=attempt,
                        call_label=diagnostic_call_label,
                        env=diagnostic_env,
                    )
                    if artifact_dir is not None
                    else None
                )
            except BaseException as exc:
                _attach_attempt_diagnostic_refs(exc, attempt_diagnostic_refs)
                raise
            diagnostic_ref: AttemptDiagnosticRef | None = None
            diagnostic_error: Exception | None = None
            try:
                with bind_attempt_diagnostics(diagnostics):
                    _check_cancel(cancel_check)
                    try:
                        result = call(selected, config, effective_idle_timeout_seconds)
                    except BaseException as call_exc:
                        final_status = (
                            "cancelled" if isinstance(call_exc, (LLMWorkerCancelled, KeyboardInterrupt)) else "failed"
                        )
                        try:
                            _finalize_paper_worker_call(
                                env,
                                status=final_status,
                                worker_id=paper_worker_id,
                                call_id=paper_call_id,
                            )
                        except Exception as finalize_exc:
                            call_exc.add_note(f"paper overlay finalization failed: {finalize_exc}")
                        raise
                    _finalize_paper_worker_call(
                        env,
                        status="success",
                        worker_id=paper_worker_id,
                        call_id=paper_call_id,
                    )
                    if diagnostics is not None:
                        try:
                            replayed = bool(
                                isinstance(result, LLMCallOutcome)
                                and isinstance(result.logical_receipt, Mapping)
                                and result.logical_receipt.get("replayed") is True
                            )
                            if replayed:
                                diagnostics.event("checkpoint_replayed")
                            diagnostic_ref = diagnostics.finalize(
                                outcome="replayed" if replayed else "success",
                                native_session_id=(
                                    result.native_session_id
                                    if isinstance(result, LLMCallOutcome)
                                    else None
                                ),
                            )
                        except Exception as diagnostic_exc:
                            # A paid provider result must never be discarded or
                            # retried because its diagnostic sidecar failed.
                            diagnostic_error = diagnostic_exc
                value = result.value if isinstance(result, LLMCallOutcome) else result
                if diagnostic_error is not None and isinstance(result, LLMCallOutcome):
                    result = replace(
                        result,
                        warnings=tuple(
                            dict.fromkeys(
                                (*result.warnings, "attempt_diagnostics.persistence_failed")
                            )
                        ),
                    )
                attempt_record = _attempt_record(
                    config,
                    fallback_index=fallback_index,
                    attempt=attempt,
                    status=(
                        "replayed"
                        if isinstance(result, LLMCallOutcome)
                        and isinstance(result.logical_receipt, Mapping)
                        and result.logical_receipt.get("replayed") is True
                        else "success"
                    ),
                    diagnostic_ref=diagnostic_ref,
                    diagnostic_error=diagnostic_error,
                    env=diagnostic_env,
                )
                attempt_records.append(attempt_record)
                if isinstance(result, LLMCallOutcome):
                    result = replace(
                        result,
                        call_record=_call_record(
                            config,
                            provider_requested=provider_requested,
                            model_requested=model_requested,
                            model_tier_requested=model_tier_requested,
                            fallback_index=fallback_index,
                            attempt=attempt,
                            attempts=attempt_records,
                            outcome=result,
                        ),
                    )
                    value = result.value
                if attach_call_record and isinstance(value, dict):
                    outcome = result if isinstance(result, LLMCallOutcome) else None
                    return attach_arc_llm_call_record(
                        value,
                        _call_record(
                            config,
                            provider_requested=provider_requested,
                            model_requested=model_requested,
                            model_tier_requested=model_tier_requested,
                            fallback_index=fallback_index,
                            attempt=attempt,
                            attempts=attempt_records,
                            outcome=outcome,
                        ),
                    )
                return result if return_outcome and isinstance(result, LLMCallOutcome) else value
            except BaseException as exc:
                if diagnostics is not None and diagnostic_ref is None:
                    with bind_attempt_diagnostics(diagnostics):
                        try:
                            diagnostic_ref = diagnostics.finalize(
                                outcome=_diagnostic_outcome(exc), error=exc
                            )
                        except Exception as diagnostic_exc:
                            diagnostic_error = diagnostic_exc
                            exc.add_note(
                                "attempt diagnostics persistence failed: "
                                + sanitize_diagnostic_text(diagnostic_exc, diagnostic_env)[:4096]
                            )
                if diagnostic_ref is not None:
                    _append_attempt_diagnostic_ref(
                        attempt_diagnostic_refs, diagnostic_ref,
                    )
                _merge_exception_attempt_refs(attempt_diagnostic_refs, exc)
                _attach_attempt_diagnostic_refs(exc, attempt_diagnostic_refs)
                if not isinstance(exc, Exception):
                    raise
                last_error = exc
                failures.append(LLMAttemptFailure(provider=config.provider, attempt=attempt, error=str(exc)))
                attempt_records.append(
                    _attempt_record(
                        config,
                        fallback_index=fallback_index,
                        attempt=attempt,
                        status="failed",
                        error=exc,
                        diagnostic_ref=diagnostic_ref,
                        diagnostic_error=diagnostic_error,
                        env=diagnostic_env,
                    )
                )
                if isinstance(
                    exc,
                    (
                        LLMSchemaError,
                        LLMWorkerCancelled,
                        LLMWorkerTimeout,
                        LLMCallNeedsSupervision,
                        LLMCallRetryExhausted,
                        LLMResponseCandidateConflict,
                        LLMResponseCandidateReceiptError,
                    ),
                ):
                    raise
                if isinstance(exc, LLMOutputValidationError) or (
                    isinstance(exc, LLMWorkerError) and not exc.retryable
                ):
                    terminal = LLMTaskError(
                        _failure_message(failures, max_attempts=max_attempts)
                    )
                    _attach_attempt_diagnostic_refs(
                        terminal, attempt_diagnostic_refs,
                    )
                    raise terminal from exc
                if _has_remaining_attempt(configs, fallback_index=fallback_index, attempt=attempt, max_attempts=max_attempts):
                    time.sleep(RETRY_INTERVAL_SECONDS)
                    try:
                        _check_cancel(cancel_check)
                    except BaseException as cancel_exc:
                        _attach_attempt_diagnostic_refs(
                            cancel_exc, attempt_diagnostic_refs,
                        )
                        raise
    terminal = LLMTaskError(_failure_message(failures, max_attempts=max_attempts))
    _attach_attempt_diagnostic_refs(terminal, attempt_diagnostic_refs)
    if last_error is not None:
        raise terminal from last_error
    raise terminal


def _append_attempt_diagnostic_ref(
    refs: list[dict[str, str]], diagnostic_ref: AttemptDiagnosticRef,
) -> None:
    candidate = {"path": diagnostic_ref.path, "sha256": diagnostic_ref.sha256}
    if candidate not in refs:
        refs.append(candidate)


def _merge_exception_attempt_refs(
    refs: list[dict[str, str]], exc: BaseException,
) -> None:
    for raw in tuple(getattr(exc, "attempt_diagnostic_refs", ()) or ()):
        if not isinstance(raw, Mapping):
            continue
        path = raw.get("path")
        digest = raw.get("sha256")
        if (
            isinstance(path, str)
            and path
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            candidate = {"path": path, "sha256": digest}
            if candidate not in refs:
                refs.append(candidate)


def _attach_attempt_diagnostic_refs(
    exc: BaseException, refs: Sequence[Mapping[str, str]],
) -> None:
    # A tuple of detached objects prevents a later retry from mutating the
    # reference set already attached to an earlier terminating exception.
    detached = tuple(
        {"path": item["path"], "sha256": item["sha256"]} for item in refs
    )
    setattr(exc, "attempt_diagnostic_refs", detached)
    # Compatibility for consumers that historically inspected only the most
    # recent immutable diagnostic record.
    setattr(exc, "diagnostic_ref", dict(detached[-1]) if detached else None)


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise LLMWorkerCancelled("LLM worker call was cancelled")


def _env_with_tier_reasoning_default(
    env: Mapping[str, str] | None,
    provider: str,
    model_tier: str | None,
) -> Mapping[str, str] | None:
    effort = reasoning_effort_for_model_tier(provider, model_tier)
    if effort is None:
        return env
    resolved = dict(env) if env is not None else dict(os.environ)
    if provider == "codex-cli":
        resolved.setdefault("ARC_CODEX_REASONING_EFFORT", effort)
    elif provider == "claude-cli":
        resolved.setdefault("ARC_CLAUDE_EFFORT", effort)
    return resolved


def _has_remaining_attempt(
    configs: Sequence[LLMConfig],
    *,
    fallback_index: int,
    attempt: int,
    max_attempts: int,
) -> bool:
    return attempt < max_attempts or fallback_index < len(configs) - 1


def _generate_json(
    selected: Any,
    prompt: str,
    *,
    schema: dict[str, Any] | None,
    schema_plan: ProviderJSONSchemaPlan,
    model: str | None,
    validate_schema: bool,
    output_recovery: str,
    schema_formatter_enabled: bool,
    role_hint: str | None,
    provider_used: str,
    model_tier: str | None,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
    session_policy: str,
    session_manager: LLMSessionManager | None,
    session_key: str | None,
    session_name: str | None,
    session_metadata: Mapping[str, Any] | None,
    artifact_dir: Path | None,
    schema_canary_root: Path | None,
    schema_canary_runtime_fingerprint: str,
    call_label: str | None,
    static_prefix: str | None,
    idle_timeout_seconds: float | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_check: Callable[[], bool] | None,
    idempotency_key: str | None,
    progress_contract_scope: str,
    supervised_native_resume: SupervisedNativeResumeAuthorization | None,
    initial_native_authorization: SupervisedNativeResumeAuthorization | None,
    validated_legacy_logical_identity: Mapping[str, Any] | None,
    validated_legacy_runtime_identity: Mapping[str, Any] | None,
) -> LLMCallOutcome:
    runtime_fp = _runtime_fp(
        provider_used=provider_used,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    if session_policy == "stateful" and not hasattr(selected, "generate_json_result"):
        raise LLMTaskError(f"Provider {provider_used} does not support stateful sessions")
    provider_schema = schema_plan.provider_schema
    prepared_checkpoint = None
    transport_prompt = (
        with_canonical_json_schema_contract(prompt, schema_plan.checkpoint_schema)
        if schema_plan.prompt_fallback and schema_plan.checkpoint_schema is not None
        else prompt
    )
    effective_prompt = apply_runtime_progress_contract(
        transport_prompt, scope=progress_contract_scope, generation_bootstrap=True
    )
    generation: int | None = None
    progress = _progress_journal(
        artifact_dir=artifact_dir,
        call_label=call_label,
        provider=provider_used,
        callback=progress_callback,
        identity={
            "idempotency_key": idempotency_key,
            "session_key": session_key,
            "provider": provider_used,
            "model": model,
            "runtime_fingerprint": runtime_fp,
        },
    )

    def call_provider(
        session: LLMSessionRef | None, session_turn: int | None = None
    ) -> LLMProviderResponse[dict[str, Any]]:
        nonlocal prepared_checkpoint
        progress.update_identity(
            session_key=session.key if session is not None else None,
            generation=session.generation if session is not None else None,
        )
        if artifact_dir is not None and (call_label or idempotency_key):
            path, identity = checkpoint_path(
                artifact_dir,
                prompt=prompt,
                schema=schema_plan.checkpoint_schema,
                provider=provider_used,
                model=model,
                call_label=call_label,
                session_policy=session_policy,
                session_key=session.key if session is not None else None,
                session_turn=session_turn,
                runtime_fingerprint=runtime_fp,
                idempotency_key=idempotency_key,
                generation=session.generation if session is not None else None,
                progress_contract_scope=progress_contract_scope,
                initial_native_authorization=initial_native_authorization,
            )
            prepared_checkpoint = prepare_call(
                path,
                identity=identity,
                cancel_check=cancel_check,
                supervised_native_resume=supervised_native_resume,
                native_session_available=bool(session and session.native_session_id),
                runtime_capabilities=_runtime_capabilities(env),
                validated_legacy_logical_identity=validated_legacy_logical_identity,
            )
            diagnostics = current_attempt_diagnostics()
            if diagnostics is not None:
                diagnostics.bind_checkpoint(
                    prepared_checkpoint.recomputation_binding
                )
            progress.update_identity(checkpoint_identity=prepared_checkpoint.identity)
            if prepared_checkpoint.replay_response is not None:
                selection = select_response_candidate(
                    prepared_checkpoint.replay_response,
                    schema=schema,
                    checkpoint_identity=prepared_checkpoint.identity,
                    replayed=True,
                )
                persist_selection_receipt(
                    prepared_checkpoint.path, selection.receipt, replayed=True
                )
                replay_response = selection.response
                diagnostics = current_attempt_diagnostics()
                if diagnostics is not None:
                    diagnostics.event("checkpoint_replayed")
                    diagnostics.record_native_session_id(
                        replay_response.native_session_id
                    )
                    for ordinal, digest, source, value in selection.diagnostic_candidates:
                        diagnostics.record_candidate(
                            value,
                            source=(
                                "checkpoint_replayed_response"
                                if source == "generic.provider_value"
                                else source
                            ),
                            ordinal=ordinal,
                            canonical_sha256=digest,
                        )
                if selection.conflict is not None:
                    raise selection.conflict
                return replay_response
            progress.bind_submission_callback(lambda: record_submitted(prepared_checkpoint))
            if supervised_native_resume is not None and (
                session is None or not session.native_session_id
            ):
                prepared_checkpoint.release_lock()
                raise LLMTaskError(
                    "supervised native resume requires an existing provider session id"
                )
        provider_prompt = effective_prompt
        if supervised_native_resume is not None:
            provider_prompt = NATIVE_RESUME_RECONCILIATION_PROMPT
            if schema_plan.prompt_fallback and schema_plan.checkpoint_schema is not None:
                provider_prompt = with_canonical_json_schema_contract(
                    provider_prompt, schema_plan.checkpoint_schema
                )
        def invoke() -> LLMProviderResponse[dict[str, Any]]:
            if hasattr(selected, "generate_json_result"):
                kwargs = {
                    "schema": provider_schema,
                    "model": model,
                    "session": session,
                    "session_policy": session_policy,
                    "schema_cache_dir": _schema_cache_dir(artifact_dir),
                    "artifact_dir": artifact_dir,
                }
                if _accepts_keyword(selected.generate_json_result, "output_recovery"):
                    kwargs["output_recovery"] = (
                        "warn" if schema_plan.prompt_fallback else output_recovery
                    )
                if _accepts_keyword(selected.generate_json_result, "defer_output_errors"):
                    kwargs["defer_output_errors"] = True
                if (
                    schema_plan.prompt_fallback
                    and _accepts_keyword(selected.generate_json_result, "schema_transport")
                ):
                    kwargs["schema_transport"] = "prompt"
                if _accepts_keyword(selected.generate_json_result, "idle_timeout_seconds"):
                    kwargs["idle_timeout_seconds"] = idle_timeout_seconds
                if _accepts_keyword(selected.generate_json_result, "progress_callback"):
                    kwargs["progress_callback"] = progress
                if _accepts_keyword(selected.generate_json_result, "cancel_check"):
                    kwargs["cancel_check"] = cancel_check
                if (
                    supervised_native_resume is not None
                    and _accepts_keyword(
                        selected.generate_json_result, "supervised_native_resume"
                    )
                ):
                    # Never reduce controller authorization to a boolean at a
                    # provider-capability boundary. Providers that opt in see
                    # the exact, already-normalized five-field value.
                    kwargs["supervised_native_resume"] = supervised_native_resume
                if (
                    initial_native_authorization is not None
                    and _accepts_keyword(
                        selected.generate_json_result, "initial_native_authorization"
                    )
                ):
                    kwargs["initial_native_authorization"] = (
                        initial_native_authorization
                    )
                return selected.generate_json_result(provider_prompt, **kwargs)
            return LLMProviderResponse(selected.generate_json(provider_prompt, schema=provider_schema, model=model))

        selection = None

        def validate_provider_response(
            candidate_response: LLMProviderResponse[dict[str, Any]],
        ) -> LLMProviderResponse[dict[str, Any]]:
            nonlocal selection
            if prepared_checkpoint is not None and prepared_checkpoint.owns_lock:
                record_submitted(prepared_checkpoint)
            if (
                candidate_response.prompt_sent_bytes is None
                or candidate_response.prompt_sent_sha256 is None
            ):
                candidate_response = replace(
                    candidate_response,
                    prompt_sent_bytes=(
                        candidate_response.prompt_sent_bytes
                        or len(provider_prompt.encode("utf-8"))
                    ),
                    prompt_sent_sha256=(
                        candidate_response.prompt_sent_sha256
                        or sha256_text(provider_prompt)
                    ),
                )
            selection = select_response_candidate(
                candidate_response,
                schema=schema,
                checkpoint_identity=(
                    prepared_checkpoint.identity
                    if prepared_checkpoint is not None
                    else None
                ),
            )
            candidate_response = selection.response
            diagnostics = current_attempt_diagnostics()
            if diagnostics is not None:
                diagnostics.record_native_session_id(candidate_response.native_session_id)
                for ordinal, digest, source, value in selection.diagnostic_candidates:
                    diagnostics.record_candidate(
                        value,
                        source=(
                            "provider_parsed_response"
                            if source == "generic.provider_value"
                            else source
                        ),
                        ordinal=ordinal,
                        canonical_sha256=digest,
                    )
            if (
                candidate_response.deferred_output_error is not None
                and selection.receipt["decision"] == "no_schema_valid_candidate"
            ):
                raise candidate_response.deferred_output_error
            if candidate_response.deferred_output_error is not None:
                candidate_response = replace(
                    candidate_response, deferred_output_error=None
                )
            return candidate_response

        try:
            response = _controlled_provider_call(
                provider_used,
                env=env,
                cancel_check=cancel_check,
                call_label=call_label,
                invoke=invoke,
                schema_canary=(
                    SchemaCanaryIdentity(
                        provider_id=provider_used,
                        runtime_fingerprint=schema_canary_runtime_fingerprint,
                        effective_model=model,
                        effective_schema_sha256=schema_hash(
                            schema_plan.provider_schema
                            if schema_plan.transport_mode == "strict"
                            else schema_plan.checkpoint_schema
                        )
                        or "",
                        transport_mode=schema_plan.transport_mode,
                    )
                    if schema_canary_root is not None
                    and schema_plan.checkpoint_schema is not None
                    and schema_plan.transport_mode in {"strict", "prompt"}
                    else None
                ),
                schema_canary_root=schema_canary_root,
                response_validator=validate_provider_response,
            )
        except BaseException as exc:
            if prepared_checkpoint is not None and prepared_checkpoint.owns_lock:
                record_failure(prepared_checkpoint, exc)
            raise
        assert selection is not None
        try:
            if prepared_checkpoint is not None:
                # Write the atomic checkpoint first, then publish the body-free
                # sidecar before releasing ownership. A crash between those writes
                # can rebuild the sidecar from the retained material on replay.
                publish_receipt = lambda: persist_selection_receipt(
                    prepared_checkpoint.path, selection.receipt
                )
                if _accepts_keyword(record_response, "after_write"):
                    record_response(
                        prepared_checkpoint,
                        response,
                        after_write=publish_receipt,
                    )
                else:
                    # Preserve compatible test/host wrappers around the historical
                    # two-argument checkpoint hook. The atomic response is still
                    # authoritative and replay can recreate a missing sidecar.
                    record_response(prepared_checkpoint, response)
                    publish_receipt()
        except BaseException as exc:
            if (
                prepared_checkpoint is not None
                and prepared_checkpoint.owns_lock
            ):
                record_failure(prepared_checkpoint, exc)
            raise
        if selection.conflict is not None:
            raise selection.conflict
        progress({"event": "call_finished", "substantive": False, "resumable": False})
        return response

    if session_policy == "stateful":
        assert session_manager is not None
        assert session_key is not None
        _migrate_legacy_session_runtime(
            session_manager, session_key=session_key, provider=provider_used,
            model=model, model_tier=model_tier, env=env,
            process_chain=process_chain, runtime_fp=runtime_fp,
            validated_runtime_identity=validated_legacy_runtime_identity,
        )
        with session_manager.locked_turn(
            key=session_key,
            provider=provider_used,
            model=model,
            runtime_fingerprint=runtime_fp,
            name=session_name,
            metadata=session_metadata,
            required_generation=(
                initial_native_authorization.generation
                if initial_native_authorization is not None else None
            ),
            initial_generation=(
                initial_native_authorization.generation
                if initial_native_authorization is not None else None
            ),
        ) as (session, turn_count):
            generation = session.generation
            effective_prompt = apply_runtime_progress_contract(
                transport_prompt,
                scope=progress_contract_scope,
                generation_bootstrap=turn_count == 0,
            )
            response = call_provider(session, turn_count)
            result = response.value
            prompt_sha = response.prompt_sent_sha256 or sha256_text(effective_prompt)
            native_session_id = response.native_session_id
            session_warnings: list[str] = []
            if native_session_id:
                warning = _update_native_session_id_with_self_heal(
                    session_manager,
                    session=session,
                    native_session_id=native_session_id,
                    provider=provider_used,
                    model=model,
                    runtime_fingerprint=runtime_fp,
                    name=session_name,
                    metadata=session_metadata,
                )
                if warning:
                    session_warnings.append(warning)
            recorded_native_session_id = native_session_id or session.native_session_id

            def record_turn(structured_output: dict[str, Any] | None = None) -> None:
                extra = {"runtime_fingerprint": runtime_fp}
                if structured_output:
                    extra["structured_output"] = structured_output
                if session_warnings:
                    extra["session_warnings"] = list(session_warnings)
                session_manager.record_turn(
                    session.key,
                    call_label=call_label or "",
                    prompt_sha256=prompt_sha,
                    static_prefix_sha256=sha256_text(static_prefix) if static_prefix else None,
                    schema_sha256=schema_hash(schema),
                    usage=response.usage.to_json(),
                    provider_used=provider_used,
                    model_used=model,
                    native_session_id=recorded_native_session_id,
                    idempotency_key=idempotency_key,
                    generation=session.generation,
                    extra=extra,
                )

            try:
                result, structured_output = _recover_or_validate_json_output(
                    result,
                    schema=schema,
                    validate_schema=validate_schema,
                    output_recovery=output_recovery,
                    schema_formatter_enabled=schema_formatter_enabled,
                    role_hint=role_hint,
                    response=response,
                    provider=provider_used,
                    model=model,
                    model_tier=model_tier,
                    env=env,
                    process_chain=process_chain,
                    artifact_dir=artifact_dir,
                    schema_canary_root=schema_canary_root,
                    call_label=call_label,
                    idle_timeout_seconds=idle_timeout_seconds,
                    progress_callback=progress,
                    cancel_check=cancel_check,
                    idempotency_key=idempotency_key,
                    prompt_schema_fallback=schema_plan.prompt_fallback,
                )
            except Exception:
                record_turn(response.structured_output)
                raise
            record_turn(structured_output)
    else:
        session = None
        response = call_provider(None)
        result = response.value
        result, structured_output = _recover_or_validate_json_output(
            result,
            schema=schema,
            validate_schema=validate_schema,
            output_recovery=output_recovery,
            schema_formatter_enabled=schema_formatter_enabled,
            role_hint=role_hint,
            response=response,
            provider=provider_used,
            model=model,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
            artifact_dir=artifact_dir,
            schema_canary_root=schema_canary_root,
            call_label=call_label,
            idle_timeout_seconds=idle_timeout_seconds,
            progress_callback=progress,
            cancel_check=cancel_check,
            idempotency_key=idempotency_key,
            prompt_schema_fallback=schema_plan.prompt_fallback,
        )
        native_session_id = response.native_session_id
        prompt_sha = response.prompt_sent_sha256 or sha256_text(effective_prompt)
    if prepared_checkpoint is not None:
        record_validated(prepared_checkpoint)
    return LLMCallOutcome(
        value=result,
        usage=response.usage,
        native_session_id=(response.native_session_id or session.native_session_id) if session else response.native_session_id,
        session_policy=session_policy,
        session_key=session.key if session else None,
        call_label=call_label,
        prompt_sha256=prompt_sha,
        static_prefix_sha256=sha256_text(static_prefix) if static_prefix else None,
        schema_sha256=schema_hash(schema),
        runtime_fingerprint=runtime_fp,
        idempotency_key=idempotency_key,
        generation=generation,
        prompt_bytes=response.prompt_sent_bytes,
        logical_receipt=_logical_receipt(
            idempotency_key=idempotency_key,
            generation=generation,
            prepared=prepared_checkpoint,
            response=response,
        ),
        structured_output=structured_output,
        warnings=tuple(dict.fromkeys((
            *schema_plan.warnings,
            *_nested_shell_warnings(env, raw_events=response.raw_events),
        ))),
    )


def _generate_text(
    selected: Any,
    prompt: str,
    *,
    model: str | None,
    provider_used: str,
    model_tier: str | None,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
    session_policy: str,
    session_manager: LLMSessionManager | None,
    session_key: str | None,
    session_name: str | None,
    session_metadata: Mapping[str, Any] | None,
    artifact_dir: Path | None,
    call_label: str | None,
    static_prefix: str | None,
    idle_timeout_seconds: float | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_check: Callable[[], bool] | None,
    idempotency_key: str | None,
    progress_contract_scope: str,
    supervised_native_resume: SupervisedNativeResumeAuthorization | None,
    initial_native_authorization: SupervisedNativeResumeAuthorization | None,
    validated_legacy_logical_identity: Mapping[str, Any] | None,
    validated_legacy_runtime_identity: Mapping[str, Any] | None,
) -> LLMCallOutcome:
    runtime_fp = _runtime_fp(
        provider_used=provider_used,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )
    if session_policy == "stateful" and not hasattr(selected, "generate_text_result"):
        raise LLMTaskError(f"Provider {provider_used} does not support stateful sessions")

    prepared_checkpoint = None
    effective_prompt = apply_runtime_progress_contract(
        prompt, scope=progress_contract_scope, generation_bootstrap=True
    )
    generation: int | None = None
    progress = _progress_journal(
        artifact_dir=artifact_dir,
        call_label=call_label,
        provider=provider_used,
        callback=progress_callback,
        identity={
            "idempotency_key": idempotency_key,
            "session_key": session_key,
            "provider": provider_used,
            "model": model,
            "runtime_fingerprint": runtime_fp,
        },
    )

    def call_provider(
        session: LLMSessionRef | None, session_turn: int | None = None
    ) -> LLMProviderResponse[str]:
        nonlocal prepared_checkpoint
        progress.update_identity(
            session_key=session.key if session is not None else None,
            generation=session.generation if session is not None else None,
        )
        if artifact_dir is not None and (call_label or idempotency_key):
            path, identity = checkpoint_path(
                artifact_dir,
                prompt=prompt,
                schema=None,
                provider=provider_used,
                model=model,
                call_label=call_label,
                session_policy=session_policy,
                session_key=session.key if session is not None else None,
                session_turn=session_turn,
                runtime_fingerprint=runtime_fp,
                idempotency_key=idempotency_key,
                generation=session.generation if session is not None else None,
                progress_contract_scope=progress_contract_scope,
                initial_native_authorization=initial_native_authorization,
            )
            prepared_checkpoint = prepare_call(
                path,
                identity=identity,
                cancel_check=cancel_check,
                supervised_native_resume=supervised_native_resume,
                native_session_available=bool(session and session.native_session_id),
                runtime_capabilities=_runtime_capabilities(env),
                validated_legacy_logical_identity=validated_legacy_logical_identity,
            )
            diagnostics = current_attempt_diagnostics()
            if diagnostics is not None:
                diagnostics.bind_checkpoint(
                    prepared_checkpoint.recomputation_binding
                )
            progress.update_identity(checkpoint_identity=prepared_checkpoint.identity)
            if prepared_checkpoint.replay_response is not None:
                return prepared_checkpoint.replay_response
            progress.bind_submission_callback(lambda: record_submitted(prepared_checkpoint))
            if supervised_native_resume is not None and (
                session is None or not session.native_session_id
            ):
                prepared_checkpoint.release_lock()
                raise LLMTaskError(
                    "supervised native resume requires an existing provider session id"
                )
        provider_prompt = (
            NATIVE_RESUME_RECONCILIATION_PROMPT
            if supervised_native_resume is not None
            else effective_prompt
        )
        def invoke() -> LLMProviderResponse[str]:
            if hasattr(selected, "generate_text_result"):
                kwargs = {
                    "model": model,
                    "session": session,
                    "session_policy": session_policy,
                    "artifact_dir": artifact_dir,
                }
                if _accepts_keyword(selected.generate_text_result, "idle_timeout_seconds"):
                    kwargs["idle_timeout_seconds"] = idle_timeout_seconds
                if _accepts_keyword(selected.generate_text_result, "progress_callback"):
                    kwargs["progress_callback"] = progress
                if _accepts_keyword(selected.generate_text_result, "cancel_check"):
                    kwargs["cancel_check"] = cancel_check
                if (
                    supervised_native_resume is not None
                    and _accepts_keyword(
                        selected.generate_text_result, "supervised_native_resume"
                    )
                ):
                    kwargs["supervised_native_resume"] = supervised_native_resume
                if (
                    initial_native_authorization is not None
                    and _accepts_keyword(
                        selected.generate_text_result, "initial_native_authorization"
                    )
                ):
                    kwargs["initial_native_authorization"] = (
                        initial_native_authorization
                    )
                return selected.generate_text_result(provider_prompt, **kwargs)
            return LLMProviderResponse(selected.generate_text(provider_prompt, model=model))

        try:
            response = _controlled_provider_call(
                provider_used,
                env=env,
                cancel_check=cancel_check,
                call_label=call_label,
                invoke=invoke,
            )
        except BaseException as exc:
            if prepared_checkpoint is not None:
                record_failure(prepared_checkpoint, exc)
            raise
        if response.prompt_sent_bytes is None or response.prompt_sent_sha256 is None:
            response = replace(
                response,
                prompt_sent_bytes=response.prompt_sent_bytes or len(provider_prompt.encode("utf-8")),
                prompt_sent_sha256=response.prompt_sent_sha256 or sha256_text(provider_prompt),
            )
        if prepared_checkpoint is not None:
            record_response(prepared_checkpoint, response)
        progress({"event": "call_finished", "substantive": False, "resumable": False})
        return response

    if session_policy == "stateful":
        assert session_manager is not None
        assert session_key is not None
        _migrate_legacy_session_runtime(
            session_manager, session_key=session_key, provider=provider_used,
            model=model, model_tier=model_tier, env=env,
            process_chain=process_chain, runtime_fp=runtime_fp,
            validated_runtime_identity=validated_legacy_runtime_identity,
        )
        with session_manager.locked_turn(
            key=session_key,
            provider=provider_used,
            model=model,
            runtime_fingerprint=runtime_fp,
            name=session_name,
            metadata=session_metadata,
            required_generation=(
                initial_native_authorization.generation
                if initial_native_authorization is not None else None
            ),
            initial_generation=(
                initial_native_authorization.generation
                if initial_native_authorization is not None else None
            ),
        ) as (session, turn_count):
            generation = session.generation
            effective_prompt = apply_runtime_progress_contract(
                prompt,
                scope=progress_contract_scope,
                generation_bootstrap=turn_count == 0,
            )
            response = call_provider(session, turn_count)
            prompt_sha = response.prompt_sent_sha256 or sha256_text(effective_prompt)
            session_warnings: list[str] = []
            if response.native_session_id:
                warning = _update_native_session_id_with_self_heal(
                    session_manager,
                    session=session,
                    native_session_id=response.native_session_id,
                    provider=provider_used,
                    model=model,
                    runtime_fingerprint=runtime_fp,
                    name=session_name,
                    metadata=session_metadata,
                )
                if warning:
                    session_warnings.append(warning)
            recorded_native_session_id = response.native_session_id or session.native_session_id
            extra = {"runtime_fingerprint": runtime_fp}
            if session_warnings:
                extra["session_warnings"] = list(session_warnings)
            session_manager.record_turn(
                session.key,
                call_label=call_label or "",
                prompt_sha256=prompt_sha,
                static_prefix_sha256=sha256_text(static_prefix) if static_prefix else None,
                schema_sha256=None,
                usage=response.usage.to_json(),
                provider_used=provider_used,
                model_used=model,
                native_session_id=recorded_native_session_id,
                idempotency_key=idempotency_key,
                generation=session.generation,
                extra=extra,
            )
    else:
        session = None
        response = call_provider(None)
        prompt_sha = response.prompt_sent_sha256 or sha256_text(effective_prompt)
    if not str(response.value or "").strip():
        raise LLMOutputValidationError("LLM text output was empty")
    if prepared_checkpoint is not None:
        record_validated(prepared_checkpoint)
    return LLMCallOutcome(
        value=response.value,
        usage=response.usage,
        native_session_id=(response.native_session_id or session.native_session_id) if session else response.native_session_id,
        session_policy=session_policy,
        session_key=session.key if session else None,
        call_label=call_label,
        prompt_sha256=prompt_sha,
        static_prefix_sha256=sha256_text(static_prefix) if static_prefix else None,
        schema_sha256=None,
        runtime_fingerprint=runtime_fp,
        idempotency_key=idempotency_key,
        generation=generation,
        prompt_bytes=response.prompt_sent_bytes,
        logical_receipt=_logical_receipt(
            idempotency_key=idempotency_key,
            generation=generation,
            prepared=prepared_checkpoint,
            response=response,
        ),
        warnings=_nested_shell_warnings(env, raw_events=response.raw_events),
    )


def _runtime_fp(
    *,
    provider_used: str,
    model: str | None,
    model_tier: str | None,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
) -> str:
    return runtime_fingerprint(
        provider=provider_used,
        model=model,
        model_tier=model_tier,
        env=env,
        process_chain=process_chain,
    )


def _migrate_legacy_session_runtime(
    session_manager: LLMSessionManager, *, session_key: str, provider: str,
    model: str | None, model_tier: str | None, env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None, runtime_fp: str,
    validated_runtime_identity: Mapping[str, Any] | None,
) -> None:
    existing = session_manager.get_existing(session_key)
    if existing is None or existing.runtime_fingerprint == runtime_fp:
        return
    if (
        existing.native_session_id
        and existing.metadata.get("arc_runtime_manifest_version")
        != RUNTIME_MANIFEST_VERSION
    ):
        return
    if validated_runtime_identity is not None:
        migrated = session_manager.migrate_validated_runtime_identity(
            session_key, identity=validated_runtime_identity,
            runtime_fingerprint=runtime_fp,
        )
        if migrated is not None:
            return
    # Runtime-manifest v1 did not record whether direct nested-shell
    # instructions were safe. It must not be proof-migrated into v2.
    if capability_runtime_identity(env)["nested_shell_status"] != "not_requested":
        return
    legacy_fp = legacy_runtime_fingerprint(
        provider=provider, model=model, model_tier=model_tier,
        env=env, process_chain=process_chain,
    )
    session_manager.migrate_legacy_runtime_fingerprint(
        session_key, expected_legacy_fingerprint=legacy_fp,
        runtime_fingerprint=runtime_fp, provider=provider, model=model,
    )


def _logical_receipt(
    *,
    idempotency_key: str | None,
    generation: int | None,
    prepared: Any,
    response: LLMProviderResponse[Any],
) -> dict[str, Any] | None:
    if idempotency_key is None:
        return None
    value = response.value
    if isinstance(value, str):
        response_sha = sha256_text(value)
    else:
        from .schema_cache import canonical_json

        response_sha = sha256_text(canonical_json(value))
    candidate_receipt = None
    if prepared is not None and response.candidate_selection is not None:
        selection_path = prepared.path.with_name(
            f"{prepared.path.stem}.candidate-selection.json"
        )
        if selection_path.is_file():
            candidate_receipt = {
                "path": selection_path.relative_to(prepared.path.parent.parent).as_posix(),
                "sha256": _sha256_file(selection_path),
            }
    return {
        "schema_version": "arc.llm.logical_receipt.v1",
        "idempotency_key": idempotency_key,
        "generation": generation,
        "checkpoint_state": "validated",
        "replayed": bool(prepared and prepared.replayed),
        "response_sha256": response_sha,
        "candidate_selection_receipt": candidate_receipt,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress_journal(
    *,
    artifact_dir: Path | None,
    call_label: str | None,
    provider: str,
    callback: Callable[[dict[str, Any]], None] | None,
    identity: Mapping[str, Any] | None = None,
) -> ProgressJournal:
    if isinstance(callback, ProgressJournal):
        return callback
    return ProgressJournal(
        artifact_dir=artifact_dir,
        call_label=call_label,
        provider=provider,
        callback=callback,
        identity=identity,
    )


def _controlled_provider_call(
    provider: str,
    *,
    env: Mapping[str, str] | None,
    cancel_check: Callable[[], bool] | None,
    call_label: str | None,
    invoke: Callable[[], LLMProviderResponse[Any]],
    schema_canary: SchemaCanaryIdentity | None = None,
    schema_canary_root: Path | None = None,
    response_validator: Callable[[LLMProviderResponse[Any]], LLMProviderResponse[Any]] | None = None,
) -> LLMProviderResponse[Any]:
    def validate(response: LLMProviderResponse[Any]) -> LLMProviderResponse[Any]:
        return response_validator(response) if response_validator is not None else response

    def submit() -> LLMProviderResponse[Any]:
        if provider == "manual":
            return validate(invoke())
        safety_env = dict(os.environ)
        if env is not None:
            safety_env.update(env)
        controller = LLMSafetyController(env=safety_env)
        with controller.acquire_call(
            provider,
            timeout_seconds=None,
            cancel_check=cancel_check,
            call_label=call_label,
        ) as permit:
            response = validate(invoke())
            permit.report_success()
            return response

    if schema_canary is None or schema_canary_root is None or provider == "manual":
        return submit()
    return run_schema_canary(
        root=schema_canary_root,
        identity=schema_canary,
        invoke=submit,
        cancel_check=cancel_check,
    )


def _schema_cache_dir(artifact_dir: Path | None) -> Path | None:
    if artifact_dir is None:
        return None
    return artifact_dir / "schemas"


def _update_native_session_id_with_self_heal(
    session_manager: LLMSessionManager,
    *,
    session: LLMSessionRef,
    native_session_id: str,
    provider: str,
    model: str | None,
    runtime_fingerprint: str,
    name: str | None,
    metadata: Mapping[str, Any] | None,
) -> str | None:
    try:
        session_manager.update_native_session_id(session.key, native_session_id)
        return None
    except KeyError:
        session_manager.get_or_create(
            key=session.key,
            provider=provider,
            model=model,
            runtime_fingerprint=runtime_fingerprint,
            name=name,
            metadata=metadata,
        )
        session_manager.update_native_session_id(session.key, native_session_id)
        return f"self_healed_missing_session_record:{session.key}"


def _validate_json_output(result: dict[str, Any], schema: dict[str, Any]) -> None:
    from jsonschema import ValidationError as JsonSchemaValidationError
    from jsonschema import validate as validate_json_schema
    from jsonschema.exceptions import SchemaError as JsonSchemaError

    try:
        validate_json_schema(instance=result, schema=schema)
    except JsonSchemaValidationError as exc:
        raise LLMOutputValidationError(f"JSON output failed schema validation: {exc.message}") from exc
    except JsonSchemaError as exc:
        raise LLMOutputValidationError(f"JSON schema is invalid: {exc.message}") from exc


def format_to_schema_or_retry(*args: Any, **kwargs: Any):
    from .schema_formatter import format_to_schema_or_retry as formatter

    return formatter(*args, **kwargs)


def _recover_or_validate_json_output(
    result: Any,
    *,
    schema: dict[str, Any] | None,
    validate_schema: bool,
    output_recovery: str,
    schema_formatter_enabled: bool = True,
    role_hint: str | None,
    response: LLMProviderResponse[dict[str, Any]],
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    artifact_dir: Path | None = None,
    schema_canary_root: Path | None = None,
    call_label: str | None = None,
    idle_timeout_seconds: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    idempotency_key: str | None = None,
    prompt_schema_fallback: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    structured_output = response.structured_output
    if (
        prompt_schema_fallback
        and output_recovery == "strict"
        and isinstance(structured_output, Mapping)
        and structured_output.get("recovery_strategy")
        in {"natural_language_fallback", "schema_default"}
    ):
        raise LLMOutputValidationError(
            "Prompt-schema fallback output did not contain a recoverable JSON object"
        )
    provider_recovered = isinstance(structured_output, Mapping) and structured_output.get("mode") == "recovered"
    if not validate_schema:
        if isinstance(result, dict):
            if schema is not None and output_recovery == "warn":
                try:
                    _validate_json_output(result, schema)
                except Exception as exc:
                    structured_output = _merge_structured_output_warning(
                        structured_output,
                        severity="minor",
                        warnings=[
                            "JSON object did not satisfy schema, but validate_schema=False allowed continuation.",
                            str(exc),
                        ],
                        raw_text=response.raw_output,
                        strategy="schema_warning_no_validation",
                        provider_error_type=type(exc).__name__,
                    )
            return result, structured_output
        if output_recovery != "warn":
            raise LLMOutputValidationError("JSON output was not an object")
        return {}, _recovered_natural_language_metadata(result, response)
    if not isinstance(result, dict):
        if output_recovery != "warn":
            raise LLMOutputValidationError("JSON output was not an object")
        if schema is None:
            return {}, _recovered_natural_language_metadata(result, response)
        result, structured_output = _recover_warn_schema_output(
            result,
            schema=schema,
            role_hint=role_hint,
            response=response,
            error=LLMOutputValidationError("JSON output was not an object"),
            provider_metadata=structured_output,
            schema_formatter_enabled=schema_formatter_enabled,
            provider=provider,
            model=model,
            model_tier=model_tier,
            env=env,
            process_chain=process_chain,
            artifact_dir=artifact_dir,
            schema_canary_root=schema_canary_root,
            call_label=call_label,
            idle_timeout_seconds=idle_timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            idempotency_key=idempotency_key,
        )
        if schema is not None:
            _validate_json_output(result, schema)
        return result, structured_output
    if schema is None:
        return result, structured_output
    if validate_schema:
        try:
            _validate_json_output(result, schema)
            return result, structured_output
        except Exception as exc:
            if output_recovery != "warn":
                raise
            result, structured_output = _recover_warn_schema_output(
                result,
                schema=schema,
                role_hint=role_hint,
                response=response,
                error=exc,
                provider_metadata=structured_output,
                schema_formatter_enabled=schema_formatter_enabled,
                provider=provider,
                model=model,
                model_tier=model_tier,
                env=env,
                process_chain=process_chain,
                artifact_dir=artifact_dir,
                schema_canary_root=schema_canary_root,
                call_label=call_label,
                idle_timeout_seconds=idle_timeout_seconds,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                idempotency_key=idempotency_key,
            )
            _validate_json_output(result, schema)
            return result, structured_output
    if provider_recovered and output_recovery == "warn":
        recovered = recover_json_output(
            value=result,
            schema=schema,
            raw_text=response.raw_output,
            role_hint=role_hint,
            provider_metadata=structured_output,
        )
        result = recovered.value
        structured_output = recovered.structured_output or structured_output
    return result, structured_output


def _recover_warn_schema_output(
    result: Any,
    *,
    schema: dict[str, Any] | None,
    role_hint: str | None,
    response: LLMProviderResponse[dict[str, Any]],
    error: Exception,
    provider_metadata: Any,
    schema_formatter_enabled: bool,
    provider: str,
    model: str | None,
    model_tier: str | None,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None,
    artifact_dir: Path | None,
    schema_canary_root: Path | None,
    call_label: str | None,
    idle_timeout_seconds: float | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_check: Callable[[], bool] | None,
    idempotency_key: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    source = _schema_recovery_source_text(result, response=response, provider_metadata=provider_metadata)
    if _is_low_content_source(source):
        raise LLMRetryableProviderOutputError(
            "JSON output failed schema validation: empty or low-content output; retry original worker"
        )
    try:
        recovered = recover_json_output(
            value=result,
            schema=schema,
            raw_text=source,
            error=error,
            role_hint=role_hint,
            provider_metadata=provider_metadata if isinstance(provider_metadata, Mapping) else None,
            allow_schema_fallback=False,
        )
        if schema is None or _json_output_validates(recovered.value, schema):
            return recovered.value, recovered.structured_output or provider_metadata
    except Exception:
        pass
    if not schema_formatter_enabled:
        raise LLMOutputValidationError("JSON output failed schema validation and schema formatter is disabled")
    try:
        def formatter_runner(format_prompt: str, **formatter_kwargs: Any) -> dict[str, Any]:
            formatter_kwargs["cancel_check"] = cancel_check
            formatter_kwargs["idle_timeout_seconds"] = idle_timeout_seconds
            formatter_kwargs["progress_callback"] = progress_callback
            # A formatter is the final paid recovery step. It must never invoke
            # another formatter or cause the original worker to be replayed.
            formatter_kwargs["schema_formatter_enabled"] = False
            if schema_canary_root is not None:
                formatter_kwargs["schema_canary_root"] = schema_canary_root
            if artifact_dir is not None and call_label:
                formatter_kwargs["artifact_dir"] = artifact_dir
                formatter_kwargs["call_label"] = f"{call_label}/schema_formatter"
            if idempotency_key:
                formatter_kwargs["idempotency_key"] = f"{idempotency_key}/schema_formatter"
            return run_json(format_prompt, **formatter_kwargs)

        # Schema formatting is an isolated recovery stage. It must not inherit
        # paper access or the user's host tools from the scientific worker.
        from .proposers_reviewer.config import isolated_worker_env

        formatted = format_to_schema_or_retry(
            raw_text=source,
            schema=schema or {"type": "object"},
            role_hint=role_hint,
            json_runner=formatter_runner,
            provider=provider,
            model=model,
            model_tier=model_tier,
            env=isolated_worker_env(env),
            process_chain=list(process_chain) if process_chain is not None else None,
        )
    except (LLMWorkerCancelled, LLMWorkerTimeout, LLMSchemaError):
        raise
    except Exception as exc:
        raise LLMOutputValidationError(
            f"JSON output failed schema validation: schema formatter failed: {exc}"
        ) from exc
    if formatted.action == "retry":
        reason = getattr(formatted, "reason", "")
        raise LLMOutputValidationError(
            f"JSON output failed schema validation: schema formatter could not repair output: {reason}"
        )
    if not isinstance(formatted.value, dict):
        raise LLMOutputValidationError(
            "JSON output failed schema validation: schema formatter returned no formatted object"
        )
    return formatted.value, formatted.structured_output


def _json_output_validates(value: Any, schema: Mapping[str, Any]) -> bool:
    try:
        _validate_json_output(value, dict(schema))
        return True
    except Exception:
        return False


def _schema_recovery_source_text(
    result: Any,
    *,
    response: LLMProviderResponse[dict[str, Any]],
    provider_metadata: Any,
) -> str:
    metadata = provider_metadata if isinstance(provider_metadata, Mapping) else {}
    raw_model_output = str(response.raw_model_output or "")
    if raw_model_output.strip():
        return raw_model_output
    raw_output = response.raw_output or ""
    if metadata.get("provider_error_type") == "error_max_structured_output_retries":
        return _model_text_from_raw_output(raw_output)
    model_text = _model_text_from_raw_output(raw_output)
    if model_text.strip():
        return model_text
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(strip_arc_llm_call_records(result), ensure_ascii=False, sort_keys=True, default=str)
    raw_excerpt = metadata.get("raw_text_excerpt")
    if isinstance(raw_excerpt, str) and raw_excerpt.strip():
        return raw_excerpt
    return str(result or "")


def _model_text_from_raw_output(raw_output: str | None) -> str:
    raw = str(raw_output or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, Mapping):
        return raw
    result = payload.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        return json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    return ""


def _is_low_content_source(source: str) -> bool:
    return _content_token_count(source) < LOW_CONTENT_TOKEN_THRESHOLD


def _content_token_count(source: str) -> int:
    text = str(source or "").strip()
    if not text or text in {"{}", "[]"}:
        return 0
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        parsed = strip_arc_llm_call_records(dict(parsed))
        text = json.dumps(parsed, ensure_ascii=False, sort_keys=True, default=str)
    return sum(1 for character in text if character.isalnum())


def _recovered_natural_language_metadata(result: Any, response: LLMProviderResponse[Any]) -> dict[str, Any]:
    existing = response.structured_output if isinstance(response.structured_output, Mapping) else None
    if isinstance(existing, Mapping):
        return dict(existing)
    raw_text = response.raw_output or (result if isinstance(result, str) else repr(result))
    return structured_metadata(
        severity="major",
        warnings=["Provider returned non-object output; accepted because schema validation is disabled for this call."],
        raw_text=str(raw_text),
        strategy="natural_language_fallback",
        provider_error_type=type(result).__name__,
    )


def _merge_structured_output_warning(
    existing: Any,
    *,
    severity: str,
    warnings: list[str],
    raw_text: str | None,
    strategy: str,
    provider_error_type: str | None,
) -> dict[str, Any]:
    if isinstance(existing, Mapping):
        merged = dict(existing)
        old_warnings = merged.get("warnings") if isinstance(merged.get("warnings"), list) else []
        merged["warnings"] = [*old_warnings, *warnings]
        if not merged.get("raw_text_excerpt") and raw_text:
            merged["raw_text_excerpt"] = str(raw_text)[:4000]
        merged.setdefault("mode", "recovered")
        merged.setdefault("severity", severity)
        merged.setdefault("recovery_strategy", strategy)
        merged.setdefault("provider_error_type", provider_error_type)
        return merged
    return structured_metadata(
        severity=severity,
        warnings=warnings,
        raw_text=raw_text,
        strategy=strategy,
        provider_error_type=provider_error_type,
    )


def _call_record(
    config: LLMConfig,
    *,
    provider_requested: str,
    model_requested: str | None,
    model_tier_requested: str | None,
    fallback_index: int,
    attempt: int,
    attempts: Sequence[dict[str, Any]],
    outcome: LLMCallOutcome | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ARC_LLM_CALL_RECORD_SCHEMA_VERSION,
        "provider_requested": provider_requested,
        "model_requested": model_requested,
        "model_tier_requested": model_tier_requested,
        "provider_used": config.provider,
        "model_used": config.model,
        "fallback_index": fallback_index,
        "attempt": attempt,
        "host": config.host.host,
        "signals": list(config.signals),
        "attempts": [dict(item) for item in attempts],
        "session_policy": outcome.session_policy if outcome else "stateless",
        "session_key": outcome.session_key if outcome else None,
        "native_session_id": outcome.native_session_id if outcome else None,
        "call_label": outcome.call_label if outcome else None,
        "prompt_sha256": outcome.prompt_sha256 if outcome else None,
        "static_prefix_sha256": outcome.static_prefix_sha256 if outcome else None,
        "schema_sha256": outcome.schema_sha256 if outcome else None,
        "runtime_fingerprint": outcome.runtime_fingerprint if outcome else None,
        "idempotency_key": outcome.idempotency_key if outcome else None,
        "generation": outcome.generation if outcome else None,
        "prompt_bytes": outcome.prompt_bytes if outcome else None,
        "logical_receipt": outcome.logical_receipt if outcome else None,
        "usage": outcome.usage.to_json() if outcome else LLMUsage().to_json(),
        "structured_output": outcome.structured_output if outcome else None,
        "warnings": list(
            dict.fromkeys((*config.warnings, *(outcome.warnings if outcome else ())))
        ),
        "call_status": (
            "recovered"
            if outcome and isinstance(outcome.structured_output, Mapping)
            and outcome.structured_output.get("mode") == "recovered"
            else "valid"
        ),
    }


def _attempt_record(
    config: LLMConfig,
    *,
    fallback_index: int,
    attempt: int,
    status: str,
    error: Exception | None = None,
    diagnostic_ref: AttemptDiagnosticRef | None = None,
    diagnostic_error: Exception | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    record = {
        "provider": config.provider,
        "model": config.model,
        "fallback_index": fallback_index,
        "attempt": attempt,
        "status": status,
        "error_type": None,
        "message": None,
        "diagnostic_path": diagnostic_ref.path if diagnostic_ref else None,
        "diagnostic_sha256": diagnostic_ref.sha256 if diagnostic_ref else None,
        "diagnostic_error_type": (
            type(diagnostic_error).__name__ if diagnostic_error is not None else None
        ),
        "diagnostic_error_message": (
            sanitize_diagnostic_text(diagnostic_error, env)[:4096]
            if diagnostic_error is not None
            else None
        ),
    }
    if error is not None:
        record["error_type"] = type(error).__name__
        record["message"] = sanitize_diagnostic_text(error, env)[:4096]
    return record


def _diagnostic_outcome(exc: BaseException) -> str:
    if isinstance(exc, (LLMWorkerCancelled, KeyboardInterrupt)):
        return "cancelled"
    if isinstance(exc, LLMWorkerTimeout):
        return "timeout"
    return "error"


def _failure_message(failures: Sequence[LLMAttemptFailure], *, max_attempts: int = MAX_ATTEMPTS_PER_PROVIDER) -> str:
    provider_count = len({failure.provider for failure in failures})
    lines = [
        f"LLM task failed after {len(failures)} attempt(s) across {provider_count} provider(s).",
        "Failures:",
    ]
    for failure in failures:
        lines.append(f"- {failure.provider} attempt {failure.attempt}/{max_attempts}: {failure.error}")
    return "\n".join(lines)


def _accepts_keyword(callable_obj: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.kind in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            if parameter.name == name:
                return True
    return False
