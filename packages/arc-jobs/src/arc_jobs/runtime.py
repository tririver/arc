from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


MIGRATION_NAME = "unified-arc-home-v1"


class RuntimeMigrationError(RuntimeError):
    """ARC cannot safely complete its one-time cache migration."""


@dataclass(frozen=True)
class RuntimePaths:
    home: Path
    runtimes: Path
    paper_cache: Path
    domain_cache: Path
    llm_cache: Path
    jobs: Path
    llm_tmp: Path
    migrations: Path
    migration_conflicts: Path


def arc_home(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("ARC_HOME"):
        return Path(env["ARC_HOME"]).expanduser()
    if env.get("XDG_DATA_HOME"):
        return Path(env["XDG_DATA_HOME"]).expanduser() / "arc"
    return Path.home() / ".local" / "share" / "arc"


def resolve_runtime_paths(env: Mapping[str, str] | None = None) -> RuntimePaths:
    env = os.environ if env is None else env
    home = arc_home(env)

    def configured(key: str, fallback: Path) -> Path:
        return Path(env[key]).expanduser() if env.get(key) else fallback

    return RuntimePaths(
        home=home,
        runtimes=configured("ARC_RUNTIME_HOME", home / "runtimes"),
        paper_cache=configured("ARC_PAPER_CACHE", home / "cache" / "arc-paper"),
        domain_cache=configured("ARC_DOMAIN_CACHE", home / "cache" / "arc-domain"),
        llm_cache=configured("ARC_LLM_CACHE", home / "cache" / "arc-llm"),
        jobs=configured("ARC_JOBS_DIR", home / "jobs"),
        llm_tmp=configured("ARC_LLM_TMP_DIR", home / "tmp" / "arc-llm"),
        migrations=home / "migrations",
        migration_conflicts=home / "migration-conflicts",
    )


def runtime_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    paths = resolve_runtime_paths(env)
    return {
        "ARC_HOME": str(paths.home),
        "ARC_RUNTIME_HOME": str(paths.runtimes),
        "ARC_PAPER_CACHE": str(paths.paper_cache),
        "ARC_DOMAIN_CACHE": str(paths.domain_cache),
        "ARC_LLM_CACHE": str(paths.llm_cache),
        "ARC_JOBS_DIR": str(paths.jobs),
        "ARC_LLM_TMP_DIR": str(paths.llm_tmp),
        "ARC_LLM_SCHEMA_CACHE_DIR": str(paths.llm_cache / "schemas"),
    }


def prepare_runtime(
    env: Mapping[str, str] | None = None,
    *,
    legacy_source_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    env = os.environ if env is None else env
    paths = resolve_runtime_paths(env)
    for path in (
        paths.home,
        paths.runtimes,
        paths.llm_tmp,
        paths.migrations,
        paths.migration_conflicts,
    ):
        _private_dir(path)
    migration = migrate_legacy_caches(env, paths=paths, source_roots=legacy_source_roots)
    for path in (paths.paper_cache, paths.domain_cache, paths.llm_cache, paths.jobs):
        _private_dir(path)
    return {
        "schema_version": "arc.runtime.v1",
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "environment": runtime_environment(env),
        "migration": migration,
    }


def migrate_legacy_caches(
    env: Mapping[str, str] | None = None,
    *,
    paths: RuntimePaths | None = None,
    source_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    env = os.environ if env is None else env
    paths = paths or resolve_runtime_paths(env)
    manifest_path = paths.migrations / f"{MIGRATION_NAME}.json"
    manifest = _read_json(manifest_path)
    if isinstance(manifest, dict) and manifest.get("status") == "completed":
        return manifest
    _private_dir(paths.migrations)
    with _lock(paths.migrations / f"{MIGRATION_NAME}.lock"):
        manifest = _read_json(manifest_path)
        if isinstance(manifest, dict) and manifest.get("status") == "completed":
            return manifest
        manifest = {
            "schema_version": "arc.runtime_migration.v1",
            "migration": MIGRATION_NAME,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "sources": [],
            "files_moved": 0,
            "files_deduplicated": 0,
            "files_conflicted": 0,
            "files_verified": 0,
            "conflicts": [],
            "errors": [],
        }
        try:
            _write_json(manifest_path, manifest)
            for source, target, label in _legacy_pairs(env, paths, source_roots):
                if not source.is_dir() or _same_path(source, target):
                    continue
                report = _migrate_tree(source, target, paths.migration_conflicts / label)
                manifest["sources"].append(
                    {"source": str(source), "target": str(target), "label": label, **report}
                )
                for key in (
                    "files_moved",
                    "files_deduplicated",
                    "files_conflicted",
                    "files_verified",
                ):
                    manifest[key] += report[key]
                manifest["conflicts"].extend(report["conflicts"])
            manifest["status"] = "completed"
            manifest["finished_at"] = _now()
            _write_json(manifest_path, manifest)
            return manifest
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["finished_at"] = _now()
            manifest["errors"].append({"type": type(exc).__name__, "message": str(exc)})
            try:
                _write_json(manifest_path, manifest)
            except OSError:
                pass
            raise RuntimeMigrationError(
                f"ARC cache migration failed; refusing split cache state: {exc}"
            ) from exc


def doctor(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    paths = resolve_runtime_paths(env)
    migration = _read_json(paths.migrations / f"{MIGRATION_NAME}.json")
    if not isinstance(migration, dict):
        migration = {"status": "not-run"}
    host, provider, signals = "unknown", "manual", []
    try:
        from arc_llm.host import select_llm_provider

        selection = select_llm_provider(env=env)
        host = selection.host.host
        provider = selection.provider
        signals = selection.signals
    except (ImportError, RuntimeError):
        pass
    return {
        "schema_version": "arc.runtime_doctor.v1",
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "migration": migration,
        "host": host,
        "provider": provider,
        "signals": signals,
    }


def detect_agent_host(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    if env.get("ARC_AGENT_HOST"):
        return str(env["ARC_AGENT_HOST"])
    try:
        from arc_llm.host import detect_host

        return detect_host(env=env).host
    except (ImportError, RuntimeError):
        return "unknown"


def _legacy_pairs(
    env: Mapping[str, str], paths: RuntimePaths, roots: Sequence[Path]
) -> list[tuple[Path, Path, str]]:
    cache_home = Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    pairs = [
        (cache_home / "arc" / "arc-paper", paths.paper_cache, "xdg-arc-paper"),
        (cache_home / "arc" / "arc-domain", paths.domain_cache, "xdg-arc-domain"),
        (cache_home / "arc" / "arc-llm", paths.llm_cache, "xdg-arc-llm"),
        (cache_home / "arc" / "arc-jobs" / "jobs", paths.jobs, "xdg-arc-jobs"),
        (cache_home / "arc" / "arc-jobs" / "stats", paths.jobs / ".stats", "xdg-jobs-stats"),
    ]
    source_roots = list(roots)
    if env.get("ARC_MIGRATION_SOURCE_ROOT"):
        source_roots.extend(
            Path(item).expanduser()
            for item in env["ARC_MIGRATION_SOURCE_ROOT"].split(os.pathsep)
            if item
        )
    for index, root in enumerate(source_roots):
        cache = root.resolve() / "cache"
        pairs.extend(
            [
                (cache / "arc-paper", paths.paper_cache, f"checkout-{index}-paper"),
                (cache / "arc-domain", paths.domain_cache, f"checkout-{index}-domain"),
                (cache / "arc-llm", paths.llm_cache, f"checkout-{index}-llm"),
                (cache / "arc-jobs" / "jobs", paths.jobs, f"checkout-{index}-jobs"),
                (cache / "arc-jobs" / "stats", paths.jobs / ".stats", f"checkout-{index}-stats"),
            ]
        )
    unique: list[tuple[Path, Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, target, label in pairs:
        key = (str(source.absolute()), str(target.absolute()))
        if key not in seen:
            unique.append((source, target, label))
            seen.add(key)
    return unique


def _migrate_tree(source: Path, target: Path, conflicts_root: Path) -> dict[str, Any]:
    for item in source.rglob("*"):
        if item.is_symlink():
            raise RuntimeMigrationError(f"legacy ARC cache contains a symlink: {item}")
    files = [item for item in sorted(source.rglob("*")) if item.is_file()]
    hashes = {item.relative_to(source): _sha256(item) for item in files}
    if not target.exists():
        _private_dir(target.parent)
        try:
            os.replace(source, target)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
        else:
            for relative, expected in hashes.items():
                if _sha256(target / relative) != expected:
                    raise RuntimeMigrationError(f"migration verification failed: {target / relative}")
            return _report(moved=len(files), verified=len(files))
    _private_dir(target)
    report = _report()
    for old in files:
        relative = old.relative_to(source)
        new = target / relative
        _private_dir(new.parent)
        old_hash = hashes[relative]
        if new.exists():
            if not new.is_file() or new.is_symlink():
                raise RuntimeMigrationError(f"migration target is not a regular file: {new}")
            if _sha256(new) == old_hash:
                old.unlink()
                report["files_deduplicated"] += 1
                report["files_verified"] += 1
                continue
            conflict = (conflicts_root / relative).with_name(
                f"{relative.name}.{old_hash[:12]}"
            )
            _private_dir(conflict.parent)
            candidate = conflict
            counter = 1
            while candidate.exists() and _sha256(candidate) != old_hash:
                candidate = conflict.with_name(f"{conflict.name}.{counter}")
                counter += 1
            conflict = candidate
            if conflict.exists():
                old.unlink()
            else:
                _move_file(old, conflict, old_hash)
            report["files_conflicted"] += 1
            report["files_verified"] += 1
            report["conflicts"].append(
                {"source": str(old), "target": str(new), "preserved_as": str(conflict)}
            )
        else:
            _move_file(old, new, old_hash)
            report["files_moved"] += 1
            report["files_verified"] += 1
    for directory in sorted(source.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass
    try:
        source.rmdir()
    except OSError:
        pass
    return report


def _report(*, moved: int = 0, verified: int = 0) -> dict[str, Any]:
    return {
        "files_moved": moved,
        "files_deduplicated": 0,
        "files_conflicted": 0,
        "files_verified": verified,
        "conflicts": [],
    }


def _move_file(source: Path, target: Path, expected_hash: str) -> None:
    try:
        os.replace(source, target)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        temporary = target.with_name(f".{target.name}.{os.getpid()}.migration")
        shutil.copy2(source, temporary)
        if _sha256(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeMigrationError(f"copied cache failed verification: {source}")
        os.replace(temporary, target)
        source.unlink()
    if _sha256(target) != expected_hash:
        raise RuntimeMigrationError(f"migrated cache failed verification: {target}")


@contextmanager
def _lock(path: Path, timeout: float = 30.0) -> Iterator[None]:
    _private_dir(path.parent)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeMigrationError(f"timed out waiting for migration lock: {path}")
            time.sleep(0.05)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        break
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_symlink():
        path.chmod(0o700)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    _private_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)
    path.chmod(0o600)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and inspect ARC_HOME")
    parser.add_argument("command", choices=("prepare", "doctor", "host"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "host":
        print(detect_agent_host())
        return 0
    try:
        result = prepare_runtime() if args.command == "prepare" else doctor()
    except RuntimeMigrationError as exc:
        print(str(exc), file=sys.stderr)
        return 78
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for key, value in result["paths"].items():
            print(f"{key}={value}")
        if args.command == "doctor":
            print(f"migration={result['migration'].get('status', 'unknown')}")
            print(f"host={result['host']}")
            print(f"provider={result['provider']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
