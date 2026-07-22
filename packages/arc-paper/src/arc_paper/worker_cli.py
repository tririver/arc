"""Restricted ``arc-paper`` entry point for ARC LLM workers.

The worker entry point deliberately classifies commands rather than trying to
hide provider binaries from a child process.  Only deterministic paper
operations are delegated to :mod:`arc_paper.cli`; commands which can start an
LLM are rejected before the ordinary CLI parser or service layer is entered.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import uuid
from typing import Any

from arc_llm.paper_access_policy import (
    PAPER_ACCESS_POLICY_VERSION,
    POLICY_TARGETS_OPERATION,
    validate_canonical_paper_access_policy,
)

from . import cli
from .ids import extract_paper_ids
from .results import err, ok
from .worker_guard import authorized_wrapper_call
from .worker_session import WorkerCacheSession


MAX_INLINE_BYTES = 64 * 1024
# Base64 adds one third to the response.  Keep a complete page envelope below
# the same 64 KiB inline boundary rather than returning an oversized page.
MAX_PAGE_BYTES = 46 * 1024
DEFAULT_PAGE_BYTES = MAX_PAGE_BYTES
POLICY_TARGETS_MAX_BYTES = 46 * 1024
POLICY_TARGETS_DEFAULT_BYTES = POLICY_TARGETS_MAX_BYTES * 9 // 10

HANDLE_RE = re.compile(r"^sha256-[0-9a-f]{64}\.json$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    settings, command_argv, parse_error = _parse_worker_options(raw_argv)
    session_dir = _session_dir(settings.session_dir)
    if os.environ.get("ARC_PAPER_CLI_ACCESS", "").strip().lower() != "full":
        result = _worker_error(
            "paper_cli_disabled",
            "arc-paper-worker is disabled for this worker or execution stage",
        )
        return _finish(result, session_dir, command_argv, settings, None)
    try:
        cache_session = WorkerCacheSession.open_or_create_from_environment()
    except Exception as exc:
        result = _worker_error(
            "worker_session_invalid",
            str(exc) or exc.__class__.__name__,
            error_type=exc.__class__.__name__,
        )
        return _finish(result, session_dir, command_argv, settings, None)
    if cache_session is not None and session_dir != cache_session.run_root:
        result = _worker_error(
            "worker_session_invalid",
            "--session-dir must match the controller-authorized worker run directory",
        )
        return _finish(result, cache_session.run_root, command_argv, settings, cache_session)
    if parse_error is not None:
        return _finish(parse_error, session_dir, [], settings, cache_session)

    if not command_argv:
        result = _worker_error("worker_command_required", "An arc-paper command is required")
        return _finish(result, session_dir, command_argv, settings, cache_session)

    if cache_session is not None and not settings.call_id:
        settings.call_id = f"call-{uuid.uuid4().hex}"

    command = command_argv[0]
    policy_error = _policy_error(command_argv, cache_session, session_dir)
    if policy_error is not None:
        return _finish(policy_error, session_dir, command_argv, settings, cache_session)
    if command == "artifact-read":
        result = _read_artifact(command_argv[1:], session_dir)
        return _finish(result, session_dir, command_argv, settings, cache_session)
    if command == POLICY_TARGETS_OPERATION:
        result = _read_policy_targets(
            command_argv[1:], session_dir, include_session_meta=cache_session is not None
        )
        return _finish(result, session_dir, command_argv, settings, cache_session)

    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        activation = cache_session.activated() if cache_session is not None else nullcontext()
        call_scope = cache_session.call_scope(settings.call_id) if cache_session is not None else nullcontext()
        with (
            activation,
            call_scope,
            authorized_wrapper_call(),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(command_argv)
        result = json.loads(stdout.getvalue())
        if not isinstance(result, dict):
            raise ValueError("arc-paper returned a non-object JSON result")
        if exit_code and result.get("ok") is not False:
            result = _worker_error(
                "worker_command_failed",
                f"arc-paper exited with status {exit_code}",
            )
    except KeyboardInterrupt:
        result = _worker_error("worker_command_cancelled", "arc-paper command was cancelled")
        result["status"] = "cancelled"
    except SystemExit as exc:
        result = _worker_error(
            "worker_arguments_invalid", f"arc-paper rejected the command arguments (status {exc.code})"
        )
    except Exception as exc:
        result = _worker_error(
            "worker_command_failed", str(exc) or exc.__class__.__name__, error_type=exc.__class__.__name__
        )
    finally:
        diagnostics = stderr.getvalue()
        if diagnostics:
            print(diagnostics, file=sys.stderr, end="")

    return _finish(result, session_dir, command_argv, settings, cache_session)


def _parse_worker_options(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str], dict[str, Any] | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session-dir")
    parser.add_argument("--worker-id", default=os.environ.get("ARC_LLM_WORKER_ID", ""))
    parser.add_argument("--call-id", default=os.environ.get("ARC_LLM_CALL_ID", ""))
    parser.add_argument("--max-inline-bytes", type=int, default=MAX_INLINE_BYTES)
    try:
        settings, remainder = parser.parse_known_args(argv)
    except SystemExit:
        settings = argparse.Namespace(
            session_dir=None, worker_id="", call_id="", max_inline_bytes=MAX_INLINE_BYTES
        )
        return settings, [], _worker_error("worker_arguments_invalid", "Invalid worker options")
    if settings.max_inline_bytes < 1 or settings.max_inline_bytes > MAX_INLINE_BYTES:
        return settings, remainder, _worker_error(
            "worker_arguments_invalid",
            f"--max-inline-bytes must be between 1 and {MAX_INLINE_BYTES}",
        )
    return settings, remainder, None


def _session_dir(explicit: str | None) -> Path | None:
    value = explicit or os.environ.get("ARC_PAPER_WORKER_SESSION_DIR")
    return Path(value).expanduser().resolve() if value else None


def _policy_error(
    argv: list[str],
    cache_session: WorkerCacheSession | None,
    session_dir: Path | None,
) -> dict[str, Any] | None:
    command = argv[0]
    restricted_error = _restricted_read_policy_error(argv, session_dir=session_dir)
    if restricted_error is not None:
        return restricted_error
    if command == "summary-batch":
        subcommand = argv[1] if len(argv) > 1 else ""
        if subcommand == "export":
            run_root = cache_session.run_root if cache_session is not None else session_dir
            path_error = _export_path_error(argv, run_root)
            if path_error is not None:
                return path_error
    if cli.RECURSIVE_LLM_CAPABILITY in cli.command_capabilities(argv):
        operation = command if command != "summary-batch" else f"summary-batch {subcommand}"
        return _nested_llm_error(operation)
    return None


def _restricted_read_policy_error(
    argv: list[str], *, session_dir: Path | None
) -> dict[str, Any] | None:
    """Enforce an optional controller-authored, fail-closed read policy."""
    file_policy, file_error, file_configured = _read_policy_file(session_dir)
    if file_configured:
        if file_error is not None:
            return file_error
        assert file_policy is not None
        return _v2_read_policy_error(argv, file_policy)

    raw_operations = os.environ.get("ARC_PAPER_WORKER_ALLOWED_OPERATIONS_JSON")
    raw_targets = os.environ.get("ARC_PAPER_WORKER_ALLOWED_TARGETS_JSON")
    if raw_operations is None and raw_targets is None:
        return None
    try:
        operations = json.loads(raw_operations or "[]")
        targets = json.loads(raw_targets or "{}")
    except json.JSONDecodeError:
        return _worker_error(
            "worker_read_policy_invalid", "The controller read policy is not valid JSON"
        )
    if (
        not isinstance(operations, list)
        or not all(isinstance(value, str) and value for value in operations)
        or not isinstance(targets, dict)
    ):
        return _worker_error(
            "worker_read_policy_invalid", "The controller read policy has an invalid shape"
        )
    command = argv[0] if argv else ""
    if command not in set(operations):
        return _worker_error(
            "worker_operation_forbidden",
            f"{command!r} is not authorized by the controller read policy",
        )
    if command == "artifact-read":
        return None
    if command not in {"get-parsed-toc", "get-parsed-section"}:
        return _worker_error(
            "worker_operation_forbidden",
            f"{command!r} is not a supported restricted read operation",
        )
    source_id, locator, argument_error = _restricted_read_arguments(argv, command=command)
    if argument_error is not None:
        return argument_error
    source_policy = targets.get(source_id)
    if not isinstance(source_policy, dict):
        return _worker_error(
            "worker_source_forbidden",
            f"Source {source_id!r} is not authorized by the controller read policy",
        )
    if command == "get-parsed-toc":
        return None
    sections = source_policy.get("sections")
    if not isinstance(sections, list) or locator not in sections:
        return _worker_error(
            "worker_section_forbidden",
            f"Section {locator!r} is not authorized for source {source_id!r}",
        )
    return None


def _v2_read_policy_error(
    argv: list[str], policy: dict[str, Any]
) -> dict[str, Any] | None:
    command = argv[0] if argv else ""
    if command not in set(policy["operations"]):
        return _worker_error(
            "worker_operation_forbidden",
            f"{command!r} is not authorized by the controller read policy",
        )
    if command in {"artifact-read", POLICY_TARGETS_OPERATION}:
        return None
    if command not in {"get-parsed-toc", "get-parsed-section"}:
        return _worker_error(
            "worker_operation_forbidden",
            f"{command!r} is not a supported restricted read operation",
        )
    source_id, locator, argument_error = _restricted_read_arguments(argv, command=command)
    if argument_error is not None:
        return argument_error
    if source_id not in set(policy["authorized_source_ids"]):
        return _worker_error(
            "worker_source_forbidden",
            f"Source {source_id!r} is not authorized by the controller read policy",
        )
    if command == "get-parsed-toc":
        return None
    if not any(
        target["source_id"] == source_id and target["locator"] == locator
        for target in policy["targets"]
    ):
        return _worker_error(
            "worker_section_forbidden",
            f"Section {locator!r} is not authorized for source {source_id!r}",
        )
    return None


def _read_policy_file(
    session_dir: Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    raw_path = os.environ.get("ARC_PAPER_WORKER_READ_POLICY_PATH")
    expected_digest = os.environ.get("ARC_PAPER_WORKER_READ_POLICY_SHA256")
    expected_schema = os.environ.get("ARC_PAPER_WORKER_READ_POLICY_SCHEMA")
    configured = any(value is not None for value in (raw_path, expected_digest, expected_schema))
    if not configured:
        return None, None, False
    if not raw_path or not expected_digest or not expected_schema:
        return None, _worker_error(
            "worker_read_policy_invalid", "The controller read policy file contract is incomplete"
        ), True
    if expected_schema != PAPER_ACCESS_POLICY_VERSION or not SHA256_RE.fullmatch(expected_digest):
        return None, _worker_error(
            "worker_read_policy_invalid", "The controller read policy identity is invalid"
        ), True
    if session_dir is None:
        return None, _worker_error(
            "worker_session_required", "A worker session directory is required for read policy"
        ), True
    root = session_dir.expanduser().resolve(strict=False)
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        return None, _worker_error(
            "worker_read_policy_invalid", "The controller read policy path must be absolute"
        ), True
    try:
        if candidate.is_symlink():
            raise ValueError("policy file must not be a symbolic link")
        resolved = candidate.resolve(strict=True)
        expected_path = root / "read-policies" / f"sha256-{expected_digest}.json"
        if candidate != resolved or resolved != expected_path:
            raise ValueError("policy file is not at its canonical session path")
        metadata = resolved.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("policy path is not a regular file")
        payload = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        return None, _worker_error(
            "worker_read_policy_invalid", f"The controller read policy file is invalid: {exc}"
        ), True
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != expected_digest:
        return None, _worker_error(
            "worker_read_policy_integrity_failed",
            "The controller read policy failed its content hash check",
        ), True
    try:
        decoded = json.loads(payload.decode("utf-8"))
        policy = validate_canonical_paper_access_policy(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return None, _worker_error(
            "worker_read_policy_invalid", f"The controller read policy is invalid: {exc}"
        ), True
    return policy, None, True


def _restricted_read_arguments(
    argv: list[str], *, command: str
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Extract security-relevant arguments only when each has one meaning."""
    source_ids: list[str] = []
    sections: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--json":
            index += 1
            continue
        if token == "--section":
            if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
                return "", None, _worker_error(
                    "worker_section_forbidden",
                    "A controller-authorized --section locator is required",
                )
            sections.append(str(argv[index + 1]))
            index += 2
            continue
        if token.startswith("--section="):
            sections.append(token.split("=", 1)[1])
            index += 1
            continue
        if token in {"--source", "--source-id", "--id", "--paper-id"} or any(
            token.startswith(f"{flag}=")
            for flag in ("--source", "--source-id", "--id", "--paper-id")
        ):
            return "", None, _worker_error(
                "worker_source_forbidden",
                "Restricted reads require exactly one positional source ID",
            )
        if token.startswith("-"):
            return "", None, _worker_error(
                "worker_arguments_invalid",
                f"Unsupported restricted-read argument: {token}",
            )
        source_ids.append(str(token))
        index += 1

    if len(source_ids) != 1:
        return "", None, _worker_error(
            "worker_source_forbidden",
            "Restricted reads require exactly one positional source ID",
        )
    if command == "get-parsed-toc":
        if sections:
            return "", None, _worker_error(
                "worker_arguments_invalid", "get-parsed-toc does not accept --section"
            )
        return source_ids[0], None, None
    if len(sections) != 1 or not sections[0]:
        return "", None, _worker_error(
            "worker_section_forbidden",
            "Restricted reads require exactly one controller-authorized --section locator",
        )
    return source_ids[0], sections[0], None


def _export_path_error(argv: list[str], run_root: Path | None) -> dict[str, Any] | None:
    output: str | None = None
    output_index: int | None = None
    joined = False
    for index, token in enumerate(argv):
        if token == "--output" and index + 1 < len(argv):
            output = argv[index + 1]
            output_index = index + 1
        elif token.startswith("--output="):
            output = token.split("=", 1)[1]
            output_index = index
            joined = True
    if output is None:
        return None  # The ordinary parser will report the missing required argument.
    if run_root is None:
        return _worker_error(
            "worker_session_required",
            "summary-batch export requires a worker run directory",
        )
    root = run_root.expanduser().resolve()
    destination = Path(output).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if destination == root or not destination.is_relative_to(root):
        return _worker_error(
            "worker_output_path_forbidden",
            "summary-batch export output must remain inside the worker run directory",
        )
    if output_index is not None:
        argv[output_index] = (
            f"--output={destination}" if joined else str(destination)
        )
    return None


def _nested_llm_error(operation: str) -> dict[str, Any]:
    return _worker_error(
        "nested_llm_forbidden",
        f"{operation!r} can start an LLM and is disabled in arc-paper-worker",
    )


def _worker_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    result = err(code, message)
    result["status"] = "error"
    if details:
        result["error"].update(details)
    return result


def _externalize_large_result(
    result: dict[str, Any], session_dir: Path | None, *, max_inline_bytes: int
) -> dict[str, Any]:
    payload = _canonical_json(result)
    if len(_display_json(result)) + 1 <= max_inline_bytes:
        return result
    if session_dir is None:
        return _worker_error(
            "worker_session_required",
            "A worker session directory is required for results larger than 64 KiB",
        )

    digest = hashlib.sha256(payload).hexdigest()
    handle = f"sha256-{digest}.json"
    artifact_dir = session_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / handle
    if destination.exists() and _sha256_file(destination) != digest:
        _atomic_write(destination, payload)
    elif not destination.exists():
        _atomic_write(destination, payload)
    source_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    return ok(
        {
            "artifact": {
                "handle": handle,
                "sha256": digest,
                "size_bytes": len(payload),
                "media_type": "application/json",
            },
            "summary": _result_summary(result),
            "paging": {
                "command": f"artifact-read {handle}",
                "offset_unit": "byte",
                "default_limit": DEFAULT_PAGE_BYTES,
            },
        },
        externalized=True,
        overlay_promotion=source_meta.get("overlay_promotion"),
        worker_audit=source_meta.get("worker_audit"),
    )


def _read_artifact(argv: list[str], session_dir: Path | None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="arc-paper-worker artifact-read", add_help=False)
    parser.add_argument("handle")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=DEFAULT_PAGE_BYTES)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return _worker_error("artifact_read_invalid", "Invalid artifact-read arguments")
    if session_dir is None:
        return _worker_error("worker_session_required", "A worker session directory is required")
    if not HANDLE_RE.fullmatch(args.handle):
        return _worker_error("artifact_handle_invalid", "Artifact handle is invalid")
    if args.offset < 0 or args.limit < 1 or args.limit > MAX_PAGE_BYTES:
        return _worker_error(
            "artifact_page_invalid",
            f"offset must be non-negative and limit must be between 1 and {MAX_PAGE_BYTES}",
        )
    path = session_dir / "artifacts" / args.handle
    try:
        size = path.stat().st_size
        expected_digest = args.handle.removeprefix("sha256-").removesuffix(".json")
        if _sha256_file(path) != expected_digest:
            return _worker_error(
                "artifact_integrity_failed", f"Artifact {args.handle!r} failed its content hash check"
            )
        with path.open("rb") as handle:
            handle.seek(args.offset)
            chunk = handle.read(args.limit)
    except FileNotFoundError:
        return _worker_error("artifact_not_found", f"Artifact {args.handle!r} was not found")
    return ok(
        {
            "handle": args.handle,
            "offset": args.offset,
            "next_offset": args.offset + len(chunk),
            "size_bytes": size,
            "eof": args.offset + len(chunk) >= size,
            "encoding": "base64",
            "content": base64.b64encode(chunk).decode("ascii"),
        }
    )


def _read_policy_targets(
    argv: list[str],
    session_dir: Path | None,
    *,
    include_session_meta: bool,
) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="arc-paper-worker policy-targets", add_help=False)
    parser.add_argument("--source-id")
    parser.add_argument("--query")
    parser.add_argument("--cursor")
    parser.add_argument("--limit-bytes", type=int, default=POLICY_TARGETS_DEFAULT_BYTES)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return _worker_error("policy_targets_invalid", "Invalid policy-targets arguments")
    if args.limit_bytes < 1024 or args.limit_bytes > POLICY_TARGETS_MAX_BYTES:
        return _worker_error(
            "policy_targets_page_invalid",
            f"limit-bytes must be between 1024 and {POLICY_TARGETS_MAX_BYTES}",
        )
    policy, policy_error, configured = _read_policy_file(session_dir)
    if not configured or policy_error is not None or policy is None:
        return policy_error or _worker_error(
            "worker_read_policy_required", "policy-targets requires a v2 worker read policy"
        )
    digest = str(os.environ["ARC_PAPER_WORKER_READ_POLICY_SHA256"])
    source_id = str(args.source_id or "")
    query = str(args.query or "")
    filter_sha256 = hashlib.sha256(
        _canonical_json({"source_id": source_id, "query": query})
    ).hexdigest()
    start = 0
    if args.cursor:
        cursor, cursor_error = _decode_policy_cursor(str(args.cursor))
        if cursor_error is not None:
            return cursor_error
        assert cursor is not None
        if cursor["policy_sha256"] != digest or cursor["filter_sha256"] != filter_sha256:
            return _worker_error(
                "policy_targets_cursor_invalid",
                "The policy-targets cursor does not match this policy and filter",
            )
        start = cursor["next_index"]

    needle = query.casefold()
    filtered = [
        target for target in policy["targets"]
        if (not source_id or target["source_id"] == source_id)
        and (
            not needle
            or needle in target["source_id"].casefold()
            or needle in target["locator"].casefold()
            or needle in target["purpose"].casefold()
        )
    ]
    if start > len(filtered):
        return _worker_error(
            "policy_targets_cursor_invalid", "The policy-targets cursor offset is out of range"
        )

    selected: list[dict[str, str]] = []
    for target in filtered[start:]:
        candidate = [*selected, target]
        candidate_result = _policy_targets_result(
            digest=digest,
            targets=candidate,
            next_index=start + len(candidate),
            total=len(filtered),
            filter_sha256=filter_sha256,
            include_session_meta=include_session_meta,
        )
        if len(_display_json(candidate_result)) + 1 > args.limit_bytes:
            break
        selected = candidate

    if start < len(filtered) and not selected:
        return _worker_error(
            "policy_target_record_too_large",
            "The next whole policy target cannot fit within limit-bytes",
        )
    result = _policy_targets_result(
        digest=digest,
        targets=selected,
        next_index=start + len(selected),
        total=len(filtered),
        filter_sha256=filter_sha256,
        include_session_meta=include_session_meta,
    )
    if len(_display_json(result)) + 1 > args.limit_bytes:
        return _worker_error(
            "policy_targets_page_invalid", "limit-bytes is too small for the page envelope"
        )
    return result


def _policy_targets_result(
    *,
    digest: str,
    targets: list[dict[str, str]],
    next_index: int,
    total: int,
    filter_sha256: str,
    include_session_meta: bool,
) -> dict[str, Any]:
    eof = next_index >= total
    next_cursor = None if eof else _encode_policy_cursor(
        policy_sha256=digest, filter_sha256=filter_sha256, next_index=next_index
    )
    result = ok({
        "policy_sha256": digest,
        "targets": targets,
        "paging": {
            "next_cursor": next_cursor,
            "eof": eof,
            "returned": len(targets),
            "total_matching": total,
        },
    })
    if include_session_meta:
        result["meta"] = {
            "overlay_promotion": {"status": "pending_controller"},
            "worker_audit": {
                "schema_version": "arc.paper.worker-session.v1",
                "status": "recorded",
            },
        }
    return result


def _encode_policy_cursor(
    *, policy_sha256: str, filter_sha256: str, next_index: int
) -> str:
    payload = _canonical_json({
        "schema_version": "arc.paper.policy-targets-cursor.v1",
        "policy_sha256": policy_sha256,
        "filter_sha256": filter_sha256,
        "next_index": next_index,
    })
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_policy_cursor(
    raw: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        padding = "=" * (-len(raw) % 4)
        decoded = base64.b64decode(raw + padding, altchars=b"-_", validate=True)
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None, _worker_error(
            "policy_targets_cursor_invalid", "The policy-targets cursor is invalid"
        )
    if (
        not isinstance(payload, dict)
        or set(payload) != {
            "schema_version", "policy_sha256", "filter_sha256", "next_index"
        }
        or payload.get("schema_version") != "arc.paper.policy-targets-cursor.v1"
        or not isinstance(payload.get("policy_sha256"), str)
        or not SHA256_RE.fullmatch(payload["policy_sha256"])
        or not isinstance(payload.get("filter_sha256"), str)
        or not SHA256_RE.fullmatch(payload["filter_sha256"])
        or not isinstance(payload.get("next_index"), int)
        or isinstance(payload.get("next_index"), bool)
        or payload["next_index"] < 0
    ):
        return None, _worker_error(
            "policy_targets_cursor_invalid", "The policy-targets cursor is invalid"
        )
    return payload, None


def _atomic_write(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finish(
    result: dict[str, Any],
    session_dir: Path | None,
    argv: list[str],
    settings: argparse.Namespace,
    cache_session: WorkerCacheSession | None,
) -> int:
    call_status = _call_status(result)
    intended_exit_code = cli._exit_code(result)
    if cache_session is not None:
        meta = result.setdefault("meta", {})
        if not isinstance(meta, dict):
            meta = {}
            result["meta"] = meta
        meta["overlay_promotion"] = {"status": "pending_controller"}
        meta["worker_audit"] = {
            "schema_version": "arc.paper.worker-session.v1",
            "status": "recorded",
        }

    result = _externalize_large_result(
        result,
        session_dir,
        max_inline_bytes=settings.max_inline_bytes,
    )
    if cache_session is not None:
        audit_error = _record_cache_call(
            cache_session, argv, result, settings, call_status=call_status
        )
        if audit_error is not None:
            intended_exit_code = 1
            result = _externalize_large_result(
                audit_error,
                session_dir,
                max_inline_bytes=settings.max_inline_bytes,
            )
    else:
        _write_audit_event(session_dir, argv, result, settings, call_status=call_status)
    sys.stdout.write(_display_json(result).decode("utf-8") + "\n")
    return intended_exit_code


def _record_cache_call(
    session: WorkerCacheSession,
    argv: list[str],
    result: dict[str, Any],
    settings: argparse.Namespace,
    *,
    call_status: str,
) -> dict[str, Any] | None:
    result_payload = _canonical_json(result)
    data = result.get("data")
    artifact = (data.get("artifact") or {}) if isinstance(data, dict) else {}
    try:
        session.record_call(
            worker_id=settings.worker_id,
            call_id=settings.call_id,
            operation=_operation(argv),
            status=call_status,
            paper_ids=extract_paper_ids(" ".join(argv)),
            parameters=_parameter_summary(argv),
            source={
                "entrypoint": "arc-paper-worker",
                "provider": (result.get("meta") or {}).get("provider"),
            },
            artifact_hash=str(artifact.get("sha256") or ""),
            result_hash=hashlib.sha256(result_payload).hexdigest(),
        )
    except Exception as exc:
        return _worker_error(
            "worker_session_record_failed",
            str(exc) or exc.__class__.__name__,
            error_type=exc.__class__.__name__,
        )
    return None


def _call_status(result: dict[str, Any]) -> str:
    data = result.get("data")
    if result.get("status") == "cancelled" or (
        isinstance(data, dict) and data.get("cancelled") is True
    ):
        return "cancelled"
    return "success" if result.get("ok") is True else "failed"


def _operation(argv: list[str]) -> str:
    return (
        " ".join(argv[:2])
        if argv and argv[0] in {"summary-batch", "cache", "doctor"}
        else (argv[0] if argv else "")
    )


def _write_audit_event(
    session_dir: Path | None,
    argv: list[str],
    result: dict[str, Any],
    settings: argparse.Namespace,
    *,
    call_status: str,
) -> None:
    if session_dir is None:
        return
    session_dir.mkdir(parents=True, exist_ok=True)
    operation = _operation(argv)
    result_payload = _canonical_json(result)
    artifact = ((result.get("data") or {}).get("artifact") or {}) if isinstance(result.get("data"), dict) else {}
    event = {
        "schema_version": "arc.paper.worker-audit.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session": os.environ.get("ARC_LLM_SESSION_ID") or session_dir.name,
        "worker": settings.worker_id,
        "call": settings.call_id,
        "operation": operation,
        "paper_ids": extract_paper_ids(" ".join(argv)),
        "parameters": _parameter_summary(argv),
        "status": call_status,
        "error_code": (result.get("error") or {}).get("code"),
        "source": (result.get("meta") or {}).get("provider"),
        "artifact_hash": artifact.get("sha256"),
        "result_hash": hashlib.sha256(result_payload).hexdigest(),
        "overlay_promotion": (result.get("meta") or {}).get("overlay_promotion", "not_applicable"),
    }
    line = _canonical_json(event) + b"\n"
    audit_path = session_dir / "audit.jsonl"
    descriptor = os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def _parameter_summary(argv: list[str]) -> dict[str, Any]:
    # Deliberately record shapes, not values: free text, paths, queries, and
    # provider configuration can contain private data or credentials.
    flags = sorted({token.split("=", 1)[0] for token in argv[1:] if token.startswith("-")})
    positional_count = sum(1 for token in argv[1:] if not token.startswith("-"))
    return {"flags": flags, "argument_count": max(0, len(argv) - 1), "positional_count": positional_count}


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    data = result.get("data")
    summary: dict[str, Any] = {
        "ok": result.get("ok"),
        "status": result.get("status"),
        "error_code": (result.get("error") or {}).get("code"),
        "data_type": type(data).__name__,
    }
    if isinstance(data, (list, dict, str)):
        summary["data_items"] = len(data)
    return summary


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _display_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
