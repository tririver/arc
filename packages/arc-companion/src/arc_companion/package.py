from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import zipfile

from .io import read_json, sha256_file, write_json
from .pdf import (
    PDF_RENDER_VERSION,
    PDF_VALIDATOR_VERSION,
    PDF_VALIDATION_RECEIPT_VERSION,
    current_pdf_validation_receipt_matches,
    match_validated_pdf_revision,
    pdf_render_recipe_sha256,
)
from .results import err, ok
from .web import (
    WEB_MANIFEST_VERSION,
    WEB_RENDER_VERSION,
    validate_reader_project,
)

_LEGACY_PDF_RENDER_VERSIONS = {
    f"arc.companion.final-render.v{version}"
    for version in range(1, 13)
}


def package_project(project_dir: Path) -> dict[str, object]:
    root = project_dir.resolve()
    state_path = root / "state.json"
    if not state_path.is_file():
        return err("companion_state_not_found", f"No companion state found in {root}")
    state = read_json(state_path)
    published_value = state.get("published") or {}
    if not isinstance(published_value, dict):
        return err("companion_package_failed", "Published companion state is invalid")
    published = dict(published_value)
    if not isinstance(published.get("pdf") or {}, dict) or not isinstance(
        published.get("web") or {}, dict
    ):
        return err("companion_package_failed", "Published companion outputs are invalid")
    published_pdf = dict(published.get("pdf") or {})
    published_web = dict(published.get("web") or {})
    effective = {**state, **published_pdf, **published_web}
    if not published_pdf and state.get("status") != "complete":
        return err("companion_not_complete", "Only a validated, complete companion can be packaged")
    try:
        strict_published = bool(published_pdf) or state.get("schema_version") == "arc.companion.state.v3"
        if strict_published:
            pdf_path = _state_hashed_file(
                root, effective, "output_pdf", "output_pdf_sha256"
            )
            tex_path = _state_hashed_file(
                root, effective, "output_tex", "output_tex_sha256"
            )
        else:
            pdf_path = _state_file(root, effective, "output_pdf")
            tex_path = _state_file(root, effective, "output_tex")
        checkpoint = effective.get("checkpoint_dir")
        if checkpoint and not published_pdf:
            checkpoint_path = _inside_project(root, Path(str(checkpoint)))
            if not checkpoint_path.is_dir():
                raise ValueError("State checkpoint_dir is missing or is not a directory")
        source_manifest_path = (
            _state_hashed_file(
                root, effective, "source_manifest_path", "source_manifest_sha256"
            )
            if strict_published
            else (
                _state_file(root, effective, "source_manifest_path")
                if effective.get("source_manifest_path") else root / "source-manifest.json"
            )
        )
        validation_path = (
            _state_hashed_file(
                root, effective, "validation_path", "validation_sha256"
            )
            if strict_published
            else (
                _state_file(root, effective, "validation_path")
                if effective.get("validation_path") else root / "validation.json"
            )
        )
        if not source_manifest_path.is_file() or not validation_path.is_file():
            raise ValueError("Source manifest or validation report is missing")
        validation = read_json(validation_path)
        legacy_validation = (
            isinstance(validation, dict)
            and validation.get("schema_version")
            != PDF_VALIDATION_RECEIPT_VERSION
            and validation.get("ok") is True
            and (
                not str(
                    published_pdf.get("render_version")
                    or effective.get("render_version")
                    or effective.get("final_render_version")
                    or ""
                )
                or str(
                    published_pdf.get("render_version")
                    or effective.get("render_version")
                    or effective.get("final_render_version")
                    or ""
                )
                in _LEGACY_PDF_RENDER_VERSIONS
            )
        )
        current_identity = (
            str(
                published_pdf.get("render_version")
                or effective.get("render_version")
                or ""
            )
            == PDF_RENDER_VERSION
            and str(
                published_pdf.get("render_recipe_sha256")
                or effective.get("render_recipe_sha256")
                or ""
            )
            == pdf_render_recipe_sha256()
            and str(
                published_pdf.get("validator_version")
                or effective.get("validator_version")
                or ""
            )
            == PDF_VALIDATOR_VERSION
        )
        current_validation = (
            isinstance(validation, dict)
            and current_identity
            and validation.get("schema_version")
            == PDF_VALIDATION_RECEIPT_VERSION
            and current_pdf_validation_receipt_matches(
                validation,
                content_sha256=str(
                    published.get("content_sha256")
                    or effective.get("content_sha256")
                    or ""
                ),
                render_recipe_sha256=pdf_render_recipe_sha256(),
                validator_version=PDF_VALIDATOR_VERSION,
                pdf_sha256=str(effective.get("output_pdf_sha256") or ""),
                tex_sha256=str(effective.get("output_tex_sha256") or ""),
                source_manifest_sha256=str(
                    effective.get("source_manifest_sha256") or ""
                ),
                pdf_bytes=pdf_path.stat().st_size,
            )
        )
        if current_validation:
            current_validation = match_validated_pdf_revision(
                root,
                state,
                content_sha256=str(
                    published.get("content_sha256")
                    or effective.get("content_sha256")
                    or ""
                ),
            ).reusable
        validation_success = legacy_validation or current_validation
        if not validation_success:
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
        files.extend(_web_files(root, effective))
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        return err("companion_package_failed", str(exc))

    unique_files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    manifest = {
        "schema_version": "arc.companion.package.v2",
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


_WEB_STATE_KEYS = {
    "output_html",
    "output_html_sha256",
    "reader_snapshot_path",
    "reader_snapshot_sha256",
    "web_manifest_path",
    "web_manifest_sha256",
    "web_render_version",
}


def _web_files(root: Path, state: dict[str, object]) -> list[Path]:
    """Resolve and verify every deliverable declared by the web manifest."""
    present = {key for key in _WEB_STATE_KEYS if state.get(key) not in (None, "")}
    if not present:
        if state.get("schema_version") == "arc.companion.state.v2":
            raise ValueError("State v2 is missing the required web reader contract")
        return []
    if present != _WEB_STATE_KEYS:
        missing = sorted(_WEB_STATE_KEYS - present)
        raise ValueError(f"State has an incomplete web reader contract: missing {missing}")

    if (
        state.get("schema_version") == "arc.companion.state.v2"
        and state.get("web_render_version") != WEB_RENDER_VERSION
    ):
        raise ValueError("State v2 web_render_version is not current")

    # Validate the browser-facing bundle as a coherent reader before collecting
    # its individual files.  Hash and containment checks below remain the
    # packaging boundary; this call additionally verifies index/data/coverage
    # relationships that a self-consistent manifest alone cannot establish.
    validate_reader_project(root, state=state)

    manifest_path = _state_hashed_file(
        root, state, "web_manifest_path", "web_manifest_sha256"
    )
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != WEB_MANIFEST_VERSION:
        raise ValueError("Web manifest has an unsupported schema")
    if manifest.get("web_render_version") != WEB_RENDER_VERSION:
        raise ValueError("Web manifest render version is not current")
    if manifest.get("web_render_version") != state.get("web_render_version"):
        raise ValueError("Web manifest render version does not match state")

    declared: list[tuple[str, dict[str, object]]] = []
    for key in ("index", "snapshot", "data_script"):
        record = manifest.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Web manifest {key} record is missing or invalid")
        declared.append((key, record))
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise ValueError("Web manifest assets must be an array")
    for index, record in enumerate(assets):
        if not isinstance(record, dict):
            raise ValueError(f"Web manifest asset {index} is invalid")
        declared.append((f"asset {index}", record))

    output: list[Path] = [manifest_path]
    seen: set[Path] = {manifest_path}
    records_by_path: dict[Path, tuple[str, str]] = {}
    for label, record in declared:
        raw_path = record.get("path")
        expected_hash = str(record.get("sha256") or "")
        if not isinstance(raw_path, str) or not raw_path or not expected_hash:
            raise ValueError(f"Web manifest {label} must contain path and sha256")
        path = _inside_project(root, Path(raw_path))
        if not path.is_file():
            raise ValueError(f"Web reader file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Web reader file hash mismatch: {path}")
        expected_bytes = record.get("bytes")
        if expected_bytes is not None and (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes != path.stat().st_size
        ):
            raise ValueError(f"Web reader file byte count mismatch: {path}")
        previous = records_by_path.get(path)
        identity = (expected_hash, label)
        if previous is not None and previous[0] != expected_hash:
            raise ValueError(f"Web manifest has conflicting duplicate path: {path}")
        records_by_path[path] = identity
        if path not in seen:
            seen.add(path)
            output.append(path)

    index_path = _state_hashed_file(root, state, "output_html", "output_html_sha256")
    snapshot_path = _state_hashed_file(
        root, state, "reader_snapshot_path", "reader_snapshot_sha256"
    )
    if index_path != _manifest_record_path(root, manifest["index"]):
        raise ValueError("State output_html does not match the web manifest index")
    if snapshot_path != _manifest_record_path(root, manifest["snapshot"]):
        raise ValueError("State reader_snapshot_path does not match the web manifest snapshot")
    return output


def _state_hashed_file(
    root: Path,
    state: dict[str, object],
    path_key: str,
    hash_key: str,
) -> Path:
    path = _state_file(root, state, path_key)
    expected_hash = str(state.get(hash_key) or "")
    if not expected_hash or sha256_file(path) != expected_hash:
        raise ValueError(f"State {hash_key} does not match {path_key}")
    return path


def _manifest_record_path(root: Path, record: object) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("Web manifest contains an invalid file record")
    return _inside_project(root, Path(record["path"]))


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
