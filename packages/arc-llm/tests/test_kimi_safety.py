from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from arc_llm.providers.base import LLMWorkerError
from arc_llm.providers.kimi_safety import resolve_kimi_retry_safety


def _no_cli_override(*_args, **_kwargs):
    return SimpleNamespace(stdout="usage: kimi acp")


@pytest.fixture(autouse=True)
def fake_binary_lookup(monkeypatch) -> None:
    monkeypatch.setattr("arc_llm.providers.kimi_safety.shutil.which", lambda *_args, **_kwargs: "/bin/kimi")


def test_kimi_retry_safety_rejects_missing_config_before_launch(tmp_path: Path) -> None:
    with pytest.raises(LLMWorkerError, match="max_retries_per_step=missing") as caught:
        resolve_kimi_retry_safety(
            {"HOME": str(tmp_path), "ARC_KIMI_BIN": "kimi"},
            help_runner=_no_cli_override,
        )
    assert caught.value.retryable is False
    assert caught.value.abort_batch is True


@pytest.mark.parametrize("value", [2, 10])
def test_kimi_retry_safety_rejects_amplifying_config(tmp_path: Path, value: int) -> None:
    home = tmp_path / "kimi"
    home.mkdir()
    (home / "config.toml").write_text(
        f"[loop_control]\nmax_retries_per_step = {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(LLMWorkerError, match=f"max_retries_per_step={value}"):
        resolve_kimi_retry_safety(
            {"KIMI_CODE_HOME": str(home), "ARC_KIMI_BIN": "kimi"},
            help_runner=_no_cli_override,
        )


def test_kimi_retry_safety_accepts_one_without_copying_config(tmp_path: Path) -> None:
    home = tmp_path / "kimi"
    home.mkdir()
    config = home / "config.toml"
    secret = "api_key = 'do-not-copy'\n[loop_control]\nmax_retries_per_step = 1\n"
    config.write_text(secret, encoding="utf-8")

    result = resolve_kimi_retry_safety(
        {"KIMI_CODE_HOME": str(home), "ARC_KIMI_BIN": "/opt/kimi"},
        help_runner=_no_cli_override,
    )

    assert result.command == ("/opt/kimi", "acp")
    assert result.enforced_by == "user_config"
    assert config.read_text(encoding="utf-8") == secret


def test_kimi_retry_safety_uses_supported_per_call_override(tmp_path: Path) -> None:
    def supported(*_args, **_kwargs):
        return SimpleNamespace(stdout="--max-retries-per-step N")

    result = resolve_kimi_retry_safety(
        {"HOME": str(tmp_path), "ARC_KIMI_BIN": "kimi"},
        help_runner=supported,
    )
    assert result.command == ("kimi", "--max-retries-per-step", "1", "acp")
    assert result.enforced_by == "cli_override"


def test_kimi_retry_safety_escape_hatch_is_explicit_and_warns(tmp_path: Path) -> None:
    result = resolve_kimi_retry_safety(
        {
            "HOME": str(tmp_path),
            "ARC_KIMI_BIN": "kimi",
            "ARC_KIMI_ALLOW_INTERNAL_RETRIES": "1",
        },
        help_runner=_no_cli_override,
    )
    assert result.enforced_by == "risk_override"
    assert "may create multiple provider requests" in str(result.warning)
