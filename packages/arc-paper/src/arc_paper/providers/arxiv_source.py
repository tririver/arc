from __future__ import annotations

import gzip
import hashlib
import io
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from ..cache import CachePaths, now_iso, read_json, write_bytes, write_json
from ..ids import arxiv_path_id, normalize_paper_id
from .base import ProviderError


MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_FILES = 10_000
MAX_EXPANSION_RATIO = 100


class ArxivSourceProvider:
    """Explicit, cache-only-after-request access to versioned arXiv sources."""

    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 120.0):
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.timeout = timeout

    def cache_source(
        self,
        paper_id: str,
        *,
        version: int,
        refresh: bool = False,
        license_url: str = "",
    ) -> dict[str, Any]:
        arxiv_id = arxiv_path_id(paper_id)
        if not arxiv_id:
            raise ProviderError("not_arxiv_id", f"arXiv source requires an arXiv ID: {paper_id}")
        if version < 1:
            raise ValueError("arXiv source version must be a positive integer")
        paths = CachePaths.for_paper(paper_id)
        manifest_path = paths.arxiv_source_manifest(version)
        if not refresh and (cached := read_json(manifest_path)):
            if _valid_cached_manifest(cached, paper_id=paths.paper_id, version=version):
                return cached

        url = f"https://export.arxiv.org/e-print/{arxiv_id}v{version}"
        response = self.client.get(url, timeout=self.timeout)
        if response.status_code == 404:
            raise ProviderError("arxiv_source_not_found", f"arXiv source not found: {arxiv_id}v{version}")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError("arxiv_source_fetch_failed", str(exc)) from exc
        data = response.content
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ProviderError("arxiv_source_archive_too_large", f"source archive exceeds {MAX_ARCHIVE_BYTES} bytes")
        digest = hashlib.sha256(data).hexdigest()
        version_dir = paths.arxiv_source_version_dir(version)
        archive_path = version_dir / "archives" / f"{digest}.bin"
        files_dir = version_dir / "files" / digest
        records = unpack_source_archive(data, destination=files_dir)
        write_bytes(archive_path, data)
        candidates = main_tex_candidates(records, root=files_dir)
        manifest = {
            "schema_version": "arc.arxiv_source.v1",
            "paper_id": normalize_paper_id(paper_id),
            "arxiv_id": arxiv_id,
            "version": version,
            "versioned_id": f"arXiv:{arxiv_id}v{version}",
            "source_url": url,
            "license": (
                license_url.strip()
                or response.headers.get("x-arxiv-license", "").strip()
                or "unknown"
            ),
            "sha256": digest,
            "bytes": len(data),
            "archive_path": str(archive_path),
            "files_root": str(files_dir),
            "files": records,
            "main_tex_candidates": candidates,
            "main_tex": candidates[0]["path"] if candidates else "",
            "cached_at": now_iso(),
            "execution_policy": "source files are cached for inspection only; no TeX is executed",
        }
        write_json(manifest_path, manifest)
        return manifest

    def probe_source(self, paper_id: str, *, version: int) -> dict[str, Any] | None:
        if version < 1:
            raise ValueError("arXiv source version must be a positive integer")
        paths = CachePaths.for_paper(paper_id)
        cached = read_json(paths.arxiv_source_manifest(version))
        if _valid_cached_manifest(cached, paper_id=paths.paper_id, version=version):
            return cached
        return None


def unpack_source_archive(data: bytes, *, destination: Path) -> list[dict[str, Any]]:
    """Validate and unpack a tar archive without following archive-controlled links."""

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            return _unpack_tar(archive, destination=destination, compressed_bytes=len(data))
    except tarfile.ReadError:
        return _unpack_single_file(data, destination=destination)


def _unpack_tar(
    archive: tarfile.TarFile,
    *,
    destination: Path,
    compressed_bytes: int,
) -> list[dict[str, Any]]:
    files: list[tarfile.TarInfo] = []
    total = 0
    seen: set[str] = set()
    entry_count = 0
    while (member := archive.next()) is not None:
        entry_count += 1
        if entry_count > MAX_FILES * 2:
            raise ValueError(f"source archive contains more than {MAX_FILES * 2} entries")
        path = _safe_member_path(member.name)
        if member.issym() or member.islnk():
            raise ValueError(f"source archive contains a link: {member.name}")
        if member.ischr() or member.isblk() or member.isfifo() or not (member.isfile() or member.isdir()):
            raise ValueError(f"source archive contains an unsupported entry: {member.name}")
        if path in seen:
            raise ValueError(f"source archive contains a duplicate path: {member.name}")
        seen.add(path)
        if member.isfile():
            files.append(member)
            if len(files) > MAX_FILES:
                raise ValueError(f"source archive contains more than {MAX_FILES} files")
            if member.size < 0 or member.size > MAX_FILE_BYTES:
                raise ValueError(f"source file exceeds {MAX_FILE_BYTES} bytes: {member.name}")
            total += member.size
            if total > MAX_TOTAL_BYTES:
                raise ValueError(f"expanded source exceeds {MAX_TOTAL_BYTES} bytes")
    if compressed_bytes and total > compressed_bytes * MAX_EXPANSION_RATIO:
        raise ValueError(f"source archive expansion ratio exceeds {MAX_EXPANSION_RATIO}:1")

    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for member in files:
        relative = _safe_member_path(member.name)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"unable to read source file: {member.name}")
        payload = stream.read(MAX_FILE_BYTES + 1)
        if len(payload) != member.size or len(payload) > MAX_FILE_BYTES:
            raise ValueError(f"source file size mismatch: {member.name}")
        target = destination / relative
        write_bytes(target, payload)
        records.append(_file_record(relative, payload))
    return sorted(records, key=lambda item: item["path"])


def _unpack_single_file(data: bytes, *, destination: Path) -> list[dict[str, Any]]:
    payload = data
    if data.startswith(b"\x1f\x8b"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                payload = stream.read(MAX_FILE_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise ValueError("invalid gzip source payload") from exc
    if len(payload) > MAX_FILE_BYTES or (len(data) and len(payload) > len(data) * MAX_EXPANSION_RATIO):
        raise ValueError("single-file source payload exceeds safety limits")
    target = destination / "main.tex"
    write_bytes(target, payload)
    return [_file_record("main.tex", payload)]


def _safe_member_path(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe source archive path: {name}")
    if re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"unsafe source archive path: {name}")
    return path.as_posix()


def _file_record(path: str, data: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main_tex_candidates(files: list[dict[str, Any]], *, root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in files:
        relative = str(item.get("path") or "")
        if Path(relative).suffix.casefold() != ".tex":
            continue
        path = root / relative
        try:
            prefix = path.read_text(encoding="utf-8", errors="replace")[:200_000]
        except OSError:
            continue
        score = 0
        reasons: list[str] = []
        if "\\documentclass" in prefix:
            score += 100
            reasons.append("documentclass")
        if "\\begin{document}" in prefix:
            score += 60
            reasons.append("document_body")
        stem = Path(relative).stem.casefold()
        if stem in {"main", "paper", "article", "manuscript"}:
            score += 20
            reasons.append("conventional_name")
        depth = len(PurePosixPath(relative).parts) - 1
        score -= depth
        candidates.append({"path": relative, "score": score, "reasons": reasons})
    return sorted(candidates, key=lambda item: (-int(item["score"]), item["path"]))


def _valid_cached_manifest(value: Any, *, paper_id: str, version: int) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("schema_version") == "arc.arxiv_source.v1"
        and value.get("paper_id") == paper_id
        and value.get("version") == version
        and value.get("sha256")
        and value.get("files_root")
        and Path(str(value.get("files_root"))).is_dir()
    )
