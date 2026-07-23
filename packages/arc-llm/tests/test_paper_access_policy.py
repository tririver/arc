from __future__ import annotations

import pytest

from arc_llm.paper_access_policy import (
    PAPER_ACCESS_INTERNAL_MIRROR,
    PAPER_ACCESS_LEGACY_WARNING,
    apply_arc_paper_access,
    resolve_arc_paper_access,
)
from arc_llm import runner


def test_resolve_canonical_default_and_legacy_access():
    assert resolve_arc_paper_access().access == "full"
    assert resolve_arc_paper_access({"arc_paper_access": "none"}).access == "none"

    legacy = resolve_arc_paper_access(env={"ARC_PAPER_CLI_ACCESS": "none"})
    assert legacy.access == "none"
    assert legacy.warnings == (PAPER_ACCESS_LEGACY_WARNING,)

    equal = resolve_arc_paper_access(
        {"arc_paper_access": "full"},
        {"ARC_PAPER_CLI_ACCESS": "full"},
    )
    assert equal.access == "full"
    assert equal.warnings == (PAPER_ACCESS_LEGACY_WARNING,)


@pytest.mark.parametrize(
    ("config", "env"),
    [
        ({"arc_paper_access": "full", "arc_paper_cli_access": "none"}, {}),
        ({"arc_paper_access": "full"}, {"ARC_PAPER_CLI_ACCESS": "none"}),
        ({}, {"ARC_PAPER_ACCESS": "full", "ARC_PAPER_CLI_ACCESS": "none"}),
        ({"arc_paper_access": "sometimes"}, {}),
    ],
)
def test_resolve_access_rejects_conflict_or_invalid_before_runtime(config, env):
    with pytest.raises(ValueError):
        resolve_arc_paper_access(config, env)


def test_compatibility_environment_is_a_nonsemantic_equal_mirror():
    env = {"ARC_PAPER_CLI_ACCESS": "none"}
    apply_arc_paper_access(env, resolve_arc_paper_access({"arc_paper_access": "full"}))
    assert env == {
        "ARC_PAPER_ACCESS": "full",
        "ARC_PAPER_CLI_ACCESS": "full",
        PAPER_ACCESS_INTERNAL_MIRROR: "true",
    }
    assert resolve_arc_paper_access(env=env).warnings == ()


def test_runtime_rejects_direct_shell_with_disabled_access_before_provider():
    with pytest.raises(ValueError, match="requires ARC_PAPER_ACCESS=full"):
        runner._runtime_compatibility_policy(
            {
                "ARC_PAPER_ACCESS": "none",
                "ARC_PAPER_DIRECT_SHELL": "true",
            },
            session_policy="stateless",
            session_manager=None,
            session_key=None,
            session_metadata=None,
            artifact_dir=None,
            idempotency_key=None,
        )
