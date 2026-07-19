from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import zipfile

from .io import read_json, sha256_file, write_json
from .results import err, ok


def package_project(project_dir: Path) -> dict[str, object]:
    root = project_dir.resolve()
    state_path = root / "state.json"
    if not state_path.is_file():
        return err("companion_state_not_found", f"No companion state found in {root}")
    state = read_json(state_path)
    if state.get("status") != "complete":
        return err("companion_not_complete", "Only a validated, complete companion can be packaged")
    try:
        pdf_path = _state_file(root, state, "output_pdf")
        tex_path = _state_file(root, state, "output_tex")
        checkpoint = state.get("checkpoint_dir")
        if checkpoint:
            checkpoint_path = _inside_project(root, Path(str(checkpoint)))
            if not checkpoint_path.is_dir():
                raise ValueError("State checkpoint_dir is missing or is not a directory")
        source_manifest_path = root / "source-manifest.json"
        validation_path = root / "validation.json"
        if not source_manifest_path.is_file() or not validation_path.is_file():
            raise ValueError("Source manifest or validation report is missing")
        validation = read_json(validation_path)
        if not isinstance(validation, dict) or validation.get("ok") is not True:
            raise ValueError("Companion validation report is not successful")
        source_manifest = read_json(source_manifest_path)
        files = [pdf_path, tex_path, source_manifest_path, validation_path, state_path]
        for asset in source_manifest.get("assets") or []:
            if not isinstance(asset, dict) or not asset.get("output_path"):
                raise ValueError("Source manifest contains an invalid TeX asset")
            asset_path = _inside_project(root, Path(str(asset["output_path"])))
            if not asset_path.is_file():
                raise ValueError(f"TeX asset is missing: {asset_path}")
            expected_hash = str(asset.get("output_sha256") or "")
            if expected_hash and sha256_file(asset_path) != expected_hash:
                raise ValueError(f"TeX asset hash mismatch: {asset_path}")
            files.append(asset_path)
    except (OSError, ValueError, TypeError) as exc:
        return err("companion_package_failed", str(exc))

    unique_files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    manifest = {
        "schema_version": "arc.companion.package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_id": state.get("paper_id"),
        "fingerprint": state.get("fingerprint"),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in unique_files
        ],
    }
    manifest_path = root / "package-manifest.json"
    write_json(manifest_path, manifest)
    archive = root / f"{pdf_path.stem}_package.zip"
    try:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in [*unique_files, manifest_path]:
                handle.write(path, arcname=path.relative_to(root).as_posix())
        _verify_archive(archive, root=root, files=[*unique_files, manifest_path])
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        archive.unlink(missing_ok=True)
        return err("companion_package_failed", f"Package verification failed: {exc}")
    return ok({"archive_path": str(archive), "manifest_path": str(manifest_path), "files": manifest["files"]})


def _state_file(root: Path, state: dict[str, object], key: str) -> Path:
    raw = state.get(key)
    if not raw:
        raise ValueError(f"State {key} is missing")
    path = _inside_project(root, Path(str(raw)))
    if not path.is_file():
        raise ValueError(f"State {key} is missing or is not a file")
    return path


def _inside_project(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes companion project: {path}") from exc
    return resolved


def _verify_archive(archive: Path, *, root: Path, files: list[Path]) -> None:
    expected = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    with zipfile.ZipFile(archive, "r") as handle:
        names = handle.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("ZIP contents do not match package manifest")
        if handle.testzip() is not None:
            raise ValueError("ZIP contains a corrupt member")
        for name, expected_hash in expected.items():
            digest = hashlib.sha256(handle.read(name)).hexdigest()
            if digest != expected_hash:
                raise ValueError(f"ZIP member hash mismatch: {name}")
