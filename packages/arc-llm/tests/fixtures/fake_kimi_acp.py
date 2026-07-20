#!/usr/bin/env python3
"""Offline NDJSON ACP server used by the arc-llm Kimi provider tests."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RECORD_PATH = Path(os.environ["FAKE_KIMI_RECORD"])
SCENARIO = os.environ.get("FAKE_KIMI_SCENARIO", "happy")
SESSION_ID = os.environ.get("FAKE_KIMI_SESSION_ID", "fake-kimi-session-1")


def record(kind: str, **payload: Any) -> None:
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RECORD_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": kind, **payload}, ensure_ascii=False) + "\n")
        handle.flush()


def send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    record("server_message", message=payload)


def respond(request: dict[str, Any], result: Any) -> None:
    send({"jsonrpc": "2.0", "id": request["id"], "result": result})


def respond_error(request: dict[str, Any], code: int, message: str) -> None:
    send(
        {
            "jsonrpc": "2.0",
            "id": request["id"],
            "error": {"code": code, "message": message},
        }
    )


def read_message() -> dict[str, Any] | None:
    line = sys.stdin.readline()
    if not line:
        return None
    message = json.loads(line)
    record("client_message", message=message)
    return message


def reverse_request(method: str, params: dict[str, Any], request_id: str) -> dict[str, Any]:
    send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
    response = read_message()
    if response is None:
        raise RuntimeError(f"client closed before replying to {method}")
    return response


def emit_chunks(session_id: str) -> None:
    output = os.environ.get("FAKE_KIMI_OUTPUT", "hello")
    split_at = int(os.environ.get("FAKE_KIMI_SPLIT_AT", str(max(1, len(output) // 2))))
    chunks = [output[:split_at], output[split_at:]]
    for chunk in chunks:
        if not chunk:
            continue
        send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": chunk},
                    },
                },
            }
        )


def hang_with_child() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ]
    )
    record("child", pid=child.pid)
    while True:
        message = read_message()
        if message is None:
            time.sleep(60)


def run_acp() -> int:
    record(
        "boot",
        argv=sys.argv,
        cwd=os.getcwd(),
        env={
            "KIMI_CODE_NO_AUTO_UPDATE": os.environ.get("KIMI_CODE_NO_AUTO_UPDATE"),
            "KIMI_DISABLE_TELEMETRY": os.environ.get("KIMI_DISABLE_TELEMETRY"),
            "KIMI_DISABLE_CRON": os.environ.get("KIMI_DISABLE_CRON"),
            "KIMI_CODE_HOME": os.environ.get("KIMI_CODE_HOME"),
        },
    )
    if SCENARIO == "stderr_flood":
        sys.stderr.write("D" * 256_000 + "\nFAKE_STDERR_END\n")
        sys.stderr.flush()

    while True:
        request = read_message()
        if request is None:
            return 0
        method = request.get("method")

        if method == "initialize":
            if SCENARIO == "invalid_json":
                sys.stdout.write("not-json\n")
                sys.stdout.flush()
                continue
            version = "0.27.0" if SCENARIO == "old_version" else "0.28.0"
            respond(
                request,
                {
                    "protocolVersion": 1,
                    "agentInfo": {"name": "Kimi Code CLI", "version": version},
                    "agentCapabilities": {"sessionCapabilities": {"resume": {}}},
                    "authMethods": [{"id": "login", "type": "terminal", "args": ["--login"]}],
                },
            )
        elif method == "authenticate":
            if SCENARIO == "auth_error":
                respond_error(request, -32000, "login required")
            else:
                respond(request, {})
        elif method == "session/new":
            respond(
                request,
                {
                    "sessionId": SESSION_ID,
                    "configOptions": [
                        {
                            "id": "model",
                            "name": "Model",
                            "type": "select",
                            "currentValue": "kimi-code/k3",
                            "options": [],
                        }
                    ],
                },
            )
        elif method == "session/resume":
            if SCENARIO == "invalid_session":
                respond_error(request, -32602, "unknown session")
            else:
                respond(request, {"configOptions": []})
        elif method == "session/set_config_option":
            respond(request, {"configOptions": []})
        elif method == "session/prompt":
            session_id = str(request.get("params", {}).get("sessionId") or SESSION_ID)
            if SCENARIO == "transport_eof":
                return 7
            if SCENARIO == "transport_usage_limit":
                sys.stderr.write("403 You've reached your usage limit\n")
                sys.stderr.flush()
                return 7
            if SCENARIO == "rpc_usage_limit":
                respond_error(request, -32099, "403 You've reached your usage limit")
                continue
            if SCENARIO == "rpc_quota_exhausted":
                respond_error(request, -32000, "quota-exhausted")
                continue
            if SCENARIO == "rpc_rate_limit":
                respond_error(request, -32000, "429 rate-limit exceeded")
                continue
            if SCENARIO == "timeout":
                hang_with_child()
            if SCENARIO == "reverse_permission":
                response = reverse_request(
                    "session/request_permission",
                    {
                        "sessionId": session_id,
                        "toolCall": {"toolCallId": "tool-1", "status": "pending"},
                        "options": [{"optionId": "allow", "name": "Allow", "kind": "allow_once"}],
                    },
                    "reverse-permission-1",
                )
                record("reverse_response", method="session/request_permission", response=response)
            if SCENARIO == "reverse_fs":
                read_response = reverse_request(
                    "fs/read_text_file",
                    {"sessionId": session_id, "path": "/secret.txt"},
                    "reverse-read-1",
                )
                record("reverse_response", method="fs/read_text_file", response=read_response)
                write_response = reverse_request(
                    "fs/write_text_file",
                    {"sessionId": session_id, "path": "/secret.txt", "content": "changed"},
                    "reverse-write-1",
                )
                record("reverse_response", method="fs/write_text_file", response=write_response)
            if SCENARIO != "empty":
                emit_chunks(session_id)
            respond(request, {"stopReason": "end_turn"})
        elif method == "session/cancel":
            record("cancel", params=request.get("params"))
        elif "id" in request:
            respond_error(request, -32601, f"unsupported fake method: {method}")


def main() -> int:
    if sys.argv[1:] == ["--version"]:
        print("0.28.0")
        return 0
    if sys.argv[1:] != ["acp"]:
        print("expected `acp`", file=sys.stderr)
        return 2
    return run_acp()


if __name__ == "__main__":
    raise SystemExit(main())
