from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .base import LLMFailureCategory, LLMSubmissionState, LLMWorkerError


def _configuration_error(message: str) -> LLMWorkerError:
    return LLMWorkerError(
        message,
        retryable=False,
        abort_batch=True,
        category=LLMFailureCategory.INVALID_REQUEST,
        submission_state=LLMSubmissionState.NOT_SUBMITTED,
    )


KIMI_INTERNAL_RETRY_ESCAPE_HATCH = "ARC_KIMI_ALLOW_INTERNAL_RETRIES"
SAFE_MAX_RETRIES_PER_STEP = 1


@dataclass(frozen=True)
class KimiRetrySafety:
    command: tuple[str, ...]
    config_path: Path | None
    configured_retries: int | None
    enforced_by: str
    warning: str | None = None


def resolve_kimi_retry_safety(
    env: Mapping[str, str] | None = None,
    *,
    help_runner=None,
) -> KimiRetrySafety:
    """Resolve a Kimi ACP command that cannot silently amplify one ARC call.

    Newer Kimi CLIs can expose a per-process retry override. Kimi Code 0.28
    cannot, so ARC must verify the inherited user configuration before it
    starts the ACP process. The configuration is inspected in place and is
    never copied because it may contain credentials.
    """

    resolved_env = os.environ if env is None else env
    binary = (resolved_env.get("ARC_KIMI_BIN") or "kimi").strip() or "kimi"
    if shutil.which(binary, path=resolved_env.get("PATH")) is None:
        raise _configuration_error(
            f"Kimi Code binary not found: {binary}. Install @moonshot-ai/kimi-code >=0.28.0."
        )
    runner = help_runner or subprocess.run
    if _supports_retry_override(binary, resolved_env, runner=runner):
        return KimiRetrySafety(
            command=(binary, "--max-retries-per-step", "1", "acp"),
            config_path=None,
            configured_retries=SAFE_MAX_RETRIES_PER_STEP,
            enforced_by="cli_override",
        )

    config_path = _kimi_config_path(resolved_env)
    configured = _read_configured_retries(config_path)
    if configured == SAFE_MAX_RETRIES_PER_STEP:
        return KimiRetrySafety(
            command=(binary, "acp"),
            config_path=config_path,
            configured_retries=configured,
            enforced_by="user_config",
        )

    if _truthy(resolved_env.get(KIMI_INTERNAL_RETRY_ESCAPE_HATCH)):
        value = "missing" if configured is None else str(configured)
        return KimiRetrySafety(
            command=(binary, "acp"),
            config_path=config_path,
            configured_retries=configured,
            enforced_by="risk_override",
            warning=(
                "Kimi internal retries are not limited by ARC "
                f"(max_retries_per_step={value}); one ARC call may create multiple provider requests."
            ),
        )

    value = "missing" if configured is None else str(configured)
    raise _configuration_error(
        "Unsafe Kimi retry configuration: "
        f"{config_path} has loop_control.max_retries_per_step={value}. "
        "Set [loop_control] max_retries_per_step = 1 before using ARC, or explicitly accept "
        f"the cost risk with {KIMI_INTERNAL_RETRY_ESCAPE_HATCH}=1.",
    )


def kimi_retry_safety_diagnostic(env: Mapping[str, str] | None = None) -> dict[str, object]:
    try:
        result = resolve_kimi_retry_safety(env)
    except LLMWorkerError as exc:
        resolved_env = os.environ if env is None else env
        return {
            "safe": False,
            "config_path": str(_kimi_config_path(resolved_env)),
            "error": str(exc),
            "remediation": "[loop_control]\nmax_retries_per_step = 1",
        }
    return {
        "safe": result.enforced_by != "risk_override",
        "enforced_by": result.enforced_by,
        "config_path": str(result.config_path) if result.config_path else None,
        "max_retries_per_step": result.configured_retries,
        "warning": result.warning,
    }


def _supports_retry_override(binary: str, env: Mapping[str, str], *, runner) -> bool:
    """Feature-detect without starting an agent session or reading credentials."""

    try:
        result = runner(
            [binary, "--help"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=dict(env),
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    output = str(getattr(result, "stdout", "") or "")
    return "--max-retries-per-step" in output


def _kimi_config_path(env: Mapping[str, str]) -> Path:
    raw_home = str(env.get("KIMI_CODE_HOME") or "").strip()
    if raw_home:
        home = Path(raw_home).expanduser()
    else:
        user_home = Path(str(env.get("HOME") or Path.home())).expanduser()
        home = user_home / ".kimi-code"
    return home.resolve(strict=False) / "config.toml"


def _read_configured_retries(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise _configuration_error(
            f"Could not safely inspect Kimi retry configuration at {path}: {exc}"
        ) from exc
    loop_control = payload.get("loop_control")
    if not isinstance(loop_control, dict):
        return None
    value = loop_control.get("max_retries_per_step")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
