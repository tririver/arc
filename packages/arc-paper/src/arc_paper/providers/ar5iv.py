from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote_to_bytes, urljoin, urlparse

import httpx

from ..cache import CachePaths, read_json, read_text, write_bytes, write_json, write_text
from ..ids import arxiv_path_id
from ..parse.document import discover_asset_urls
from ..worker_session import worker_fetch_once
from .base import ProviderError


MAX_ASSET_BYTES = 100 * 1024 * 1024


def ar5iv_url(paper_id: str) -> str:
    aid = arxiv_path_id(paper_id)
    if not aid:
        raise ProviderError("not_arxiv_id", f"ar5iv requires an arXiv ID: {paper_id}")
    return f"https://ar5iv.labs.arxiv.org/html/{aid}"


class Ar5ivProvider:
    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 60.0):
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.timeout = timeout

    def get_html(self, paper_id: str, *, refresh: bool = False) -> str:
        url = ar5iv_url(paper_id)
        paths = CachePaths.for_paper(paper_id)
        if not refresh:
            cached = read_text(paths.ar5iv_html)
            if cached is not None:
                return cached

        def fetch() -> str:
            response = self.client.get(url, timeout=self.timeout)
            if response.status_code == 404:
                raise ProviderError("ar5iv_not_found", f"ar5iv HTML not found for {paper_id}")
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    "ar5iv_fetch_failed", str(exc), status_code=exc.response.status_code
                ) from exc
            write_text(paths.ar5iv_html, response.text)
            return response.text

        return worker_fetch_once(
            paper_id, fetch, operation="ar5iv-html", replay_success=not refresh
        )

    def cache_assets(self, paper_id: str, html: str, *, refresh: bool = False) -> list[dict]:
        """Cache academic assets referenced by ar5iv HTML by content hash."""

        source_url = ar5iv_url(paper_id)
        paths = CachePaths.for_paper(paper_id)
        discovered = discover_asset_urls(html, source_url=source_url)
        cached = read_json(paths.ar5iv_asset_manifest) if not refresh else None
        if not (
            isinstance(cached, dict)
            and cached.get("schema_version") == "arc.ar5iv_assets.v1"
            and cached.get("paper_id") == paths.paper_id
            and cached.get("source_url") == source_url
        ):
            cached = None
        cached_by_url = {
            str(item.get("source_url") or ""): item
            for item in (cached or {}).get("assets", [])
            if isinstance(item, dict)
        }
        records: list[dict] = []
        for item in discovered:
            url = item["source_url"]
            prior = cached_by_url.get(url)
            if prior and prior.get("status") == "cached":
                reusable = _reusable_asset_record(
                    prior,
                    paths=paths,
                    source_url=url,
                    original_url=item["original_url"],
                )
                if reusable is not None:
                    records.append(reusable)
                    continue
            operation = f"ar5iv-asset-{hashlib.sha256(url.encode('utf-8')).hexdigest()}"
            records.append(
                worker_fetch_once(
                    paths.paper_id,
                    lambda: self._cache_asset(url, item["original_url"], paths),
                    operation=operation,
                )
            )
        write_json(
            paths.ar5iv_asset_manifest,
            {
                "schema_version": "arc.ar5iv_assets.v1",
                "paper_id": paths.paper_id,
                "source_url": source_url,
                "assets": records,
            },
        )
        return records

    def _cache_asset(self, url: str, original_url: str, paths: CachePaths) -> dict:
        try:
            data, media_type = self._fetch_asset_bytes(url)
            if len(data) > MAX_ASSET_BYTES:
                raise ValueError(f"asset exceeds {MAX_ASSET_BYTES} bytes")
            digest = hashlib.sha256(data).hexdigest()
            suffix = _asset_suffix(url, media_type)
            target = paths.ar5iv_assets / digest[:2] / f"{digest}{suffix}"
            target_is_valid = (
                target.exists()
                and target.stat().st_size == len(data)
                and hashlib.sha256(target.read_bytes()).hexdigest() == digest
            )
            if not target_is_valid:
                write_bytes(target, data)
            return {
                "asset_id": f"sha256:{digest}",
                "source_url": url,
                "original_url": original_url,
                "media_type": media_type,
                "sha256": digest,
                "bytes": len(data),
                "cache_path": str(target),
                "relative_path": str(target.relative_to(paths.paper_dir)),
                "status": "cached",
            }
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {401, 403, 429}:
                raise ProviderError(
                    "ar5iv_asset_fetch_failed", str(exc), status_code=status_code
                ) from exc
            return {
                "asset_id": "",
                "source_url": url,
                "original_url": original_url,
                "media_type": "",
                "sha256": "",
                "bytes": 0,
                "cache_path": "",
                "relative_path": "",
                "status": "missing",
                "error": str(exc),
            }

    def _fetch_asset_bytes(self, url: str) -> tuple[bytes, str]:
        if url.startswith("data:"):
            return _decode_data_url(url)
        current_url = url
        for _ in range(6):
            _validate_asset_url(current_url)
            response = self.client.get(current_url, timeout=self.timeout, follow_redirects=False)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("asset redirect has no location")
                current_url = urljoin(current_url, location)
                continue
            response.raise_for_status()
            _validate_asset_url(str(response.url))
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_ASSET_BYTES:
                raise ValueError(f"asset exceeds {MAX_ASSET_BYTES} bytes")
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            return response.content, media_type or "application/octet-stream"
        raise ValueError("asset redirect limit exceeded")


def _decode_data_url(url: str) -> tuple[bytes, str]:
    header, payload = url.split(",", 1)
    metadata = header[5:]
    is_base64 = metadata.endswith(";base64")
    media_type = metadata.removesuffix(";base64") or "text/plain"
    data = base64.b64decode(payload, validate=True) if is_base64 else unquote_to_bytes(payload)
    return data, media_type


def _validate_asset_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "ar5iv.labs.arxiv.org":
        raise ValueError("only same-origin HTTPS ar5iv assets are cached")


def _asset_suffix(url: str, media_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix and len(suffix) <= 12 and suffix.replace(".", "").isalnum():
        return suffix
    return mimetypes.guess_extension(media_type) or ""


def _reusable_asset_record(
    record: dict,
    *,
    paths: CachePaths,
    source_url: str,
    original_url: str,
) -> dict | None:
    try:
        if str(record.get("source_url") or "") != source_url:
            return None
        digest = str(record.get("sha256") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        if str(record.get("asset_id") or "") != f"sha256:{digest}":
            return None
        suffix = _asset_suffix(source_url, str(record.get("media_type") or ""))
        candidate = paths.ar5iv_assets / digest[:2] / f"{digest}{suffix}"
        if candidate.is_symlink():
            return None
        resolved_root = paths.ar5iv_assets.resolve()
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
        if not resolved.is_file():
            return None
        size = resolved.stat().st_size
        if size > MAX_ASSET_BYTES or size != int(record.get("bytes") or -1):
            return None
        actual_digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual_digest != digest:
            return None
        reusable = dict(record)
        reusable["source_url"] = source_url
        reusable["original_url"] = original_url
        reusable["cache_path"] = str(resolved)
        reusable["relative_path"] = str(resolved.relative_to(paths.paper_dir.resolve()))
        reusable["bytes"] = size
        return reusable
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
