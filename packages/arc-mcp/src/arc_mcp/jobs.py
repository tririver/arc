from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Mapping

from arc_jobs import JobCancelled, JobManager


# MCP-facing names stay local to this adapter; job persistence and execution live
# in arc-jobs and use only its protocol-neutral schemas and environment variables.
MCPJobCancelled = JobCancelled
MCPJobManager = JobManager


def resolve_inline_wait_seconds(
    *,
    env: Mapping[str, str] | None = None,
    server_name: str = "arc",
    default: float = 90.0,
) -> float:
    """Resolve the MCP inline wait budget before returning a background job."""
    env = env if env is not None else os.environ
    explicit = _float_env(env, "ARC_MCP_INLINE_WAIT_SEC")
    if explicit is not None:
        return max(0.0, explicit)
    margin = _float_env(env, "ARC_MCP_BACKGROUND_MARGIN_SEC")
    if margin is None:
        margin = 10.0
    timeout = _float_env(env, "ARC_MCP_TOOL_TIMEOUT_SEC")
    if timeout is None:
        timeout = _codex_mcp_tool_timeout(env=env, server_name=server_name)
    if timeout is None:
        return max(0.0, default)
    return max(0.0, timeout - margin)


def _codex_mcp_tool_timeout(*, env: Mapping[str, str], server_name: str) -> float | None:
    config_path = Path(env.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    server = servers.get(server_name)
    if not isinstance(server, dict):
        return None
    value = server.get("tool_timeout_sec")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_env(env: Mapping[str, str], key: str) -> float | None:
    value = env.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
