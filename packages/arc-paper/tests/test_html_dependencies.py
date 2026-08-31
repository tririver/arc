from __future__ import annotations

import json
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from arc_paper import (
    ArcPaperService,
    OPERATION_REGISTRY,
    OperationEffect,
    export_cache,
    import_cache,
)
from arc_paper.cli import main
from arc_paper.providers import Ar5ivProvider, ArxivHtmlProvider
from arc_paper.providers._request_gate import HostRequestGate


PAPER_ID = "0911.3380"
MAIN_URL = f"https://arxiv.org/html/{PAPER_ID}"


def _response(
    request: httpx.Request,
    *,
    content: bytes = b"",
    media_type: str = "text/html",
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    values = {"content-type": media_type, **(headers or {})}
    return httpx.Response(
        status_code,
        request=request,
        content=content,
        headers=values,
    )


def _provider(
    cache_root: Path,
    handler,
    **kwargs,
) -> ArxivHtmlProvider:
    return ArxivHtmlProvider(
        cache_root=cache_root,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        request_gate=HostRequestGate(minimum_interval=0),
        **kwargs,
    )


def _article(body: str, *, base: str = "") -> bytes:
    head = f'<base href="{base}">' if base else ""
    return (
        f"<html><head>{head}</head><body>"
        '<img src="/static/site-shell.png">'
        f'<article class="ltx_document">{body}</article>'
        "</body></html>"
    ).encode()


def test_bundle_extracts_authored_root_base_redirects_and_deduplicates(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>never-run()</script></svg>'

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if url == MAIN_URL:
            return _response(
                request,
                status_code=302,
                headers={"location": f"/html/{PAPER_ID}v4/"},
            )
        if url == f"{MAIN_URL}v4/":
            return _response(
                request,
                content=_article(
                    """
                    <object data="plot.svg" type="image/svg+xml"></object>
                    <img src="same.png"><img src="same.png">
                    <source src="photo.webp">
                    <img srcset="small.png 1x, big.png 2x">
                    """,
                    base=f"/html/{PAPER_ID}v4/assets/",
                ),
            )
        if url.endswith("/assets/plot.svg"):
            return _response(
                request,
                status_code=307,
                headers={"location": "plot-final.svg"},
            )
        if url.endswith("/assets/plot-final.svg"):
            return _response(request, content=svg, media_type="image/svg+xml")
        if url.endswith("/assets/same.png"):
            return _response(request, content=b"same-png", media_type="image/png")
        if url.endswith("/assets/photo.webp"):
            return _response(request, content=b"webp", media_type="image/webp")
        raise AssertionError(f"unexpected request: {url}")

    provider = _provider(tmp_path, handler)
    bundle = provider.fetch_bundle(PAPER_ID)

    assert bundle.primary.origin.locator == f"{MAIN_URL}v4/"
    assert bundle.document_url == f"{MAIN_URL}v4/"
    assert bundle.base_url == f"{MAIN_URL}v4/assets/"
    assert bundle.provider == "arxiv-html"
    assert len(bundle.bundle_digest) == 64
    assert [item.authored_target for item in bundle.dependencies] == [
        "plot.svg",
        "same.png",
        "same.png",
        "photo.webp",
        "small.png 1x, big.png 2x",
    ]
    assert [item.availability for item in bundle.dependencies] == [
        "available",
        "available",
        "available",
        "available",
        "unavailable",
    ]
    assert bundle.dependencies[0].resolved_url.endswith("/assets/plot-final.svg")
    assert bundle.dependencies[0].media_type == "image/svg+xml"
    assert bundle.dependencies[1].artifact_digest == bundle.dependencies[2].artifact_digest
    assert bundle.dependencies[4].error_code == "html_dependency_srcset_unsupported"
    assert [item.code for item in bundle.warnings] == [
        "html_dependency_srcset_unsupported"
    ]
    assert not any("site-shell" in item for item in calls)
    assert calls.count(f"{MAIN_URL}v4/assets/same.png") == 1
    assert calls == [
        MAIN_URL,
        f"{MAIN_URL}v4/",
        f"{MAIN_URL}v4/assets/plot.svg",
        f"{MAIN_URL}v4/assets/plot-final.svg",
        f"{MAIN_URL}v4/assets/same.png",
        f"{MAIN_URL}v4/assets/photo.webp",
    ]


def test_ar5iv_bundle_recovers_malformed_html_without_provider_special_cases(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if url == f"https://ar5iv.labs.arxiv.org/html/{PAPER_ID}":
            return _response(
                request,
                content=b"<article class=ltx_document><object data='plot.svg'><div",
            )
        if url == f"https://ar5iv.labs.arxiv.org/html/plot.svg":
            return _response(request, content=b"<svg/>", media_type="image/svg+xml")
        raise AssertionError(url)

    provider = Ar5ivProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        request_gate=HostRequestGate(minimum_interval=0),
    )
    bundle = provider.fetch_bundle(PAPER_ID)

    assert bundle.provider == "ar5iv"
    assert bundle.dependencies[0].authored_target == "plot.svg"
    assert bundle.dependencies[0].availability == "available"
    assert calls == [
        f"https://ar5iv.labs.arxiv.org/html/{PAPER_ID}",
        f"https://ar5iv.labs.arxiv.org/html/plot.svg",
    ]


def test_bundle_rejects_unsafe_targets_base_and_redirects_without_host_drift(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    body = """
      <object data="https://evil.test/cross.svg"></object>
      <img src="data:image/png;base64,AAAA">
      <img src="javascript:alert(1)">
      <img src="file:///tmp/private.png">
      <img src="https://user:secret@arxiv.org/private.png">
      <img src="fragment.png#panel">
      <img src="redirect.png">
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if url == MAIN_URL:
            return _response(request, content=_article(body))
        if url == f"{MAIN_URL.rsplit('/', 1)[0]}/redirect.png":
            return _response(
                request,
                status_code=302,
                headers={"location": "https://evil.test/escaped.png"},
            )
        raise AssertionError(f"unsafe redirect or target was requested: {url}")

    bundle = _provider(tmp_path / "targets", handler).fetch_bundle(PAPER_ID)

    assert all(item.availability == "unavailable" for item in bundle.dependencies)
    assert {item.error_code for item in bundle.dependencies} == {
        "html_dependency_url_invalid",
        "html_dependency_redirect_invalid",
    }
    assert calls == [MAIN_URL, "https://arxiv.org/html/redirect.png"]

    base_calls: list[str] = []

    def invalid_base(request: httpx.Request) -> httpx.Response:
        base_calls.append(str(request.url))
        return _response(
            request,
            content=_article('<img src="paper.png">', base="https://evil.test/"),
        )

    invalid = _provider(tmp_path / "base", invalid_base).fetch_bundle(PAPER_ID)
    assert invalid.dependencies[0].error_code == "html_dependency_base_invalid"
    assert invalid.dependencies[0].availability == "unavailable"
    assert base_calls == [MAIN_URL]


def test_bundle_enforces_count_resource_total_status_and_media_contracts(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    body = """
      <img src="a.png"><img src="b.png"><img src="huge.png">
      <object data="wrong.svg" type="image/svg+xml"></object>
      <img src="missing.png"><img src="overflow.png">
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if url == MAIN_URL:
            return _response(request, content=_article(body))
        if url.endswith("/a.png"):
            return _response(request, content=b"aaaa", media_type="image/png")
        if url.endswith("/b.png"):
            return _response(request, content=b"bbbb", media_type="image/png")
        if url.endswith("/huge.png"):
            return _response(
                request,
                content=b"",
                media_type="image/png",
                headers={"content-length": "6"},
            )
        if url.endswith("/wrong.svg"):
            return _response(request, content=b"png", media_type="image/png")
        if url.endswith("/missing.png"):
            return _response(
                request,
                content=b"missing",
                media_type="image/png",
                status_code=404,
            )
        raise AssertionError(f"count-limited target was requested: {url}")

    bundle = _provider(
        tmp_path,
        handler,
        max_dependency_count=5,
        max_dependency_bytes=5,
        max_total_dependency_bytes=6,
    ).fetch_bundle(PAPER_ID)

    assert len(bundle.dependencies) == 5
    assert bundle.dependencies[0].availability == "available"
    assert bundle.dependencies[1].error_code == "html_dependency_total_too_large"
    assert bundle.dependencies[2].error_code == "html_dependency_too_large"
    assert bundle.dependencies[3].error_code == "html_dependency_media_type_mismatch"
    assert bundle.dependencies[4].error_code == "html_dependency_fetch_failed"
    assert "html_dependency_count_limit" in {item.code for item in bundle.warnings}
    assert not any("overflow.png" in item for item in calls)
    rejected_digest = hashlib.sha256(b"bbbb").hexdigest()
    assert not (
        tmp_path
        / "reference-material-cache"
        / "v1"
        / "resources"
        / "sha256"
        / rejected_digest[:2]
        / rejected_digest
    ).exists()


def test_document_byte_limit_counts_content_deduplicated_resources_once(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == MAIN_URL:
            return _response(
                request,
                content=_article('<img src="a.png"><img src="b.png">'),
            )
        return _response(request, content=b"same", media_type="image/png")

    bundle = _provider(
        tmp_path,
        handler,
        max_dependency_bytes=4,
        max_total_dependency_bytes=4,
    ).fetch_bundle(PAPER_ID)

    assert [item.availability for item in bundle.dependencies] == [
        "available",
        "available",
    ]
    assert bundle.dependencies[0].artifact_digest == bundle.dependencies[1].artifact_digest


def test_bundle_cache_identity_binds_exact_acquisition_policy(tmp_path: Path) -> None:
    from arc_paper import html_source_bundle_to_document

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if url == MAIN_URL:
            return _response(
                request,
                content=_article('<img src="a.png"><img src="b.png">'),
            )
        return _response(request, content=url.encode(), media_type="image/png")

    loose = _provider(tmp_path, handler, max_dependency_count=2).fetch_bundle(
        PAPER_ID
    )
    strict = _provider(tmp_path, handler, max_dependency_count=1).fetch_bundle(
        PAPER_ID
    )
    document = html_source_bundle_to_document(strict)
    encoded = OPERATION_REGISTRY[
        "fetch-arxiv-html-bundle"
    ].output_codec.encode(strict)

    assert document["schema_version"] == "arc.paper.html_source_bundle.v2"
    assert document["acquisition_policy"] == {
        "max_dependency_count": 1,
        "max_dependency_bytes": 25 * 1024 * 1024,
        "max_total_dependency_bytes": 200 * 1024 * 1024,
        "max_redirects": 5,
    }
    assert encoded["acquisition_policy"] == document["acquisition_policy"]
    assert strict.bundle_digest != loose.bundle_digest
    assert len(strict.dependencies) == 1
    assert {warning.code for warning in strict.warnings} == {
        "html_dependency_count_limit"
    }
    assert calls.count(MAIN_URL) == 2


def test_bundle_cache_hit_still_rejects_invalid_acquisition_policy(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == MAIN_URL:
            return _response(request, content=_article('<img src="a.png">'))
        return _response(request, content=b"png", media_type="image/png")

    _provider(tmp_path, handler).fetch_bundle(PAPER_ID)
    invalid = _provider(
        tmp_path,
        lambda request: (_ for _ in ()).throw(
            AssertionError("invalid policy must fail before cache replay")
        ),
        max_dependency_count=0,
    )

    with pytest.raises(ValueError, match="limits"):
        invalid.fetch_bundle(PAPER_ID)


def test_bundle_cache_replay_refresh_corruption_repair_and_main_survival(
    tmp_path: Path,
) -> None:
    generation = 0
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generation
        url = str(request.url)
        calls.append(url)
        if url == MAIN_URL:
            generation += 1
            return _response(
                request,
                content=_article('<object data="plot.svg"></object>'),
            )
        if url.endswith("/plot.svg"):
            return _response(
                request,
                content=f"<svg>generation-{generation}</svg>".encode(),
                media_type="image/svg+xml",
            )
        raise AssertionError(url)

    provider = _provider(tmp_path, handler)
    first = provider.fetch_bundle(PAPER_ID)
    replay = provider.fetch_bundle(PAPER_ID)
    refreshed = provider.fetch_bundle(PAPER_ID, refresh=True)

    assert first.bundle_digest == replay.bundle_digest
    assert refreshed.bundle_digest != first.bundle_digest
    assert calls == [MAIN_URL, "https://arxiv.org/html/plot.svg"] * 2

    entry = provider.cache.admin_entry(
        "json", "arxiv-html-dependencies", PAPER_ID
    )
    assert entry is not None
    entry_dir = (
        tmp_path
        / "remote-request-cache"
        / "v2"
        / "json"
        / entry.namespace
        / entry.request_digest[:2]
        / entry.request_digest
    )
    manifest = json.loads((entry_dir / "manifest.json").read_text(encoding="utf-8"))
    (entry_dir / manifest["payload_file"]).write_bytes(b"corrupt")
    repaired_manifest = provider.fetch_bundle(PAPER_ID)
    assert repaired_manifest.primary.size > 0

    resource = (
        tmp_path
        / "reference-material-cache"
        / "v1"
        / "resources"
        / "sha256"
        / repaired_manifest.dependencies[0].artifact_digest[:2]
        / repaired_manifest.dependencies[0].artifact_digest
        / "resource"
    )
    resource.write_bytes(b"corrupt")
    repaired_resource = provider.fetch_bundle(PAPER_ID)
    assert repaired_resource.dependencies[0].availability == "available"
    assert len(calls) == 8

    primary = provider.fetch(PAPER_ID)
    assert provider.cache.source_repository.read_bytes(primary).startswith(b"<html>")


def test_interrupted_bundle_refresh_keeps_main_and_repairs_mismatched_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generation
        if str(request.url) == MAIN_URL:
            generation += 1
            return _response(
                request,
                content=_article(
                    f'<p>generation-{generation}</p><img src="plot.png">'
                ),
            )
        return _response(
            request,
            content=f"png-{generation}".encode(),
            media_type="image/png",
        )

    provider = _provider(tmp_path, handler)
    first = provider.fetch_bundle(PAPER_ID)
    original_write = provider.cache._atomic_write

    def interrupt(path: Path, payload: bytes) -> None:
        if (
            path.name == "manifest.json"
            and "arxiv-html-dependencies" in path.parts
        ):
            raise OSError("simulated bundle manifest interruption")
        original_write(path, payload)

    monkeypatch.setattr(provider.cache, "_atomic_write", interrupt)
    with pytest.raises(OSError, match="bundle manifest interruption"):
        provider.fetch_bundle(PAPER_ID, refresh=True)
    monkeypatch.setattr(provider.cache, "_atomic_write", original_write)

    current_main = provider.fetch(PAPER_ID)
    assert provider.cache.source_repository.read_bytes(current_main).startswith(b"<html>")
    repaired = provider.fetch_bundle(PAPER_ID)
    assert repaired.bundle_digest != first.bundle_digest
    assert generation == 3


def test_bundle_concurrent_fill_fetches_main_and_each_unique_dependency_once(
    tmp_path: Path,
) -> None:
    counts: dict[str, int] = {}
    count_lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        with count_lock:
            counts[url] = counts.get(url, 0) + 1
        time.sleep(0.01)
        if url == MAIN_URL:
            return _response(
                request,
                content=_article('<img src="same.png"><img src="same.png">'),
            )
        return _response(request, content=b"png", media_type="image/png")

    provider = _provider(tmp_path, handler)
    with ThreadPoolExecutor(max_workers=8) as executor:
        bundles = list(executor.map(lambda _: provider.fetch_bundle(PAPER_ID), range(8)))

    assert counts == {MAIN_URL: 1, "https://arxiv.org/html/same.png": 1}
    assert len({item.bundle_digest for item in bundles}) == 1


def test_export_materializes_safe_authored_layout(
    tmp_path: Path,
) -> None:
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>never-run()</script></svg>'
    png = b"\x89PNG\r\nfixture"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == MAIN_URL:
            return _response(
                request,
                content=_article(
                    """
                    <figure id="F1"><object data="figures/plot.svg"
                      type="image/svg+xml"></object><figcaption>SVG</figcaption></figure>
                    <figure id="F2"><img src="figures/panel.png"
                      alt="Panel"><figcaption>PNG</figcaption></figure>
                    <img src="/static/absolute.png">
                    """
                ),
            )
        if url.endswith("/figures/plot.svg"):
            return _response(request, content=svg, media_type="image/svg+xml")
        if url.endswith("/figures/panel.png") or url.endswith("/static/absolute.png"):
            return _response(request, content=png, media_type="image/png")
        raise AssertionError(url)

    provider = _provider(tmp_path / "cache", handler)
    service = ArcPaperService(
        cache_root=tmp_path / "cache",
        arxiv_html=provider,
    )
    output = tmp_path / "source-bundle"
    result = service.export_arxiv_html_bundle(PAPER_ID, output_dir=output)

    assert Path(result["source"]) == output / "source.html"
    assert Path(result["manifest"]) == output / "manifest.json"
    assert (output / "figures" / "plot.svg").read_bytes() == svg
    assert (output / "figures" / "panel.png").read_bytes() == png
    assert not (output / "static" / "absolute.png").exists()
    assert {item["authored_target"] for item in result["resources"]} == {
        "figures/plot.svg",
        "figures/panel.png",
    }
    assert "html_dependency_target_not_materializable" in {
        item["code"] for item in result["warnings"]
    }
    assert (output / "figures" / "plot.svg").read_bytes() == svg

    with pytest.raises(Exception) as error:
        service.export_arxiv_html_bundle(PAPER_ID, output_dir=output)
    assert getattr(error.value, "code", "") == "html_bundle_output_not_empty"


def test_bundle_cache_archive_round_trip_replays_without_network(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == MAIN_URL:
            return _response(
                request,
                content=_article('<object data="plot.svg"></object>'),
            )
        return _response(request, content=b"<svg/>", media_type="image/svg+xml")

    source_cache = tmp_path / "source-cache"
    service = ArcPaperService(
        cache_root=source_cache,
        arxiv_html=_provider(source_cache, handler),
    )
    original = service.fetch_arxiv_html_bundle(PAPER_ID)
    entry = service.list_cache(paper_ids=(PAPER_ID,)).entries[0]
    component = next(item for item in entry.components if item.name == "arxiv-html")
    assert len(component.storage_entry_ids) == 2

    archive = tmp_path / "bundle.tar.gz"
    export_cache(archive, cache_root=source_cache, entry_ids=(entry.entry_id,))
    target = tmp_path / "target-cache"
    import_cache(archive, cache_root=target)

    replay = _provider(
        target,
        lambda request: (_ for _ in ()).throw(
            AssertionError("archive replay must not access the network")
        ),
    ).fetch_bundle(PAPER_ID)
    assert replay.bundle_digest == original.bundle_digest


def test_cache_update_refreshes_an_existing_dependency_bundle_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == MAIN_URL:
            return _response(
                request,
                content=_article('<img src="plot.png">'),
            )
        return _response(request, content=b"png", media_type="image/png")

    cache_root = tmp_path / "cache"
    service = ArcPaperService(
        cache_root=cache_root,
        arxiv_html=_provider(cache_root, handler),
    )
    bundle = service.fetch_arxiv_html_bundle(PAPER_ID)
    bundle_refreshes = 0
    parses = 0

    def refresh_bundle(paper_id: str, *, refresh: bool = False):
        nonlocal bundle_refreshes
        assert paper_id == f"arXiv:{PAPER_ID}"
        assert refresh is True
        bundle_refreshes += 1
        return bundle

    def parse_bundle(source_bundle):
        nonlocal parses
        assert source_bundle.primary == bundle.primary
        parses += 1

    monkeypatch.setattr(service, "get_metadata", lambda *args, **kwargs: {"arxiv_id": PAPER_ID})
    monkeypatch.setattr(service, "get_citers", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "fetch_arxiv_html_bundle", refresh_bundle)
    monkeypatch.setattr(service, "parse_bundle", parse_bundle)
    monkeypatch.setattr(
        service,
        "parse_arxiv_auto",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bundle-aware update must not refetch main-only HTML")
        ),
    )
    monkeypatch.setattr(service, "parse_arxiv_pdf", lambda *args, **kwargs: None)

    result = service.update_cache(paper_ids=(PAPER_ID,))

    assert bundle_refreshes == 1
    assert parses == 1
    assert any(
        item.component == "arxiv-html" and item.status == "updated"
        for item in result.records
    )


def test_cache_update_preserves_exact_bundle_version_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_id = "2608.20415"
    versioned = f"{paper_id}v1"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"https://arxiv.org/html/{versioned}"
        return _response(
            request,
            content=(
                '<html><body><a class="header-button" '
                'title="Back to abstract page" aria-label="Back to abstract page" '
                f'href="/abs/{versioned}">Back</a>'
                '<a class="header-button" title="Download PDF" target="_blank" '
                f'href="/pdf/{versioned}">PDF</a>'
                '<article class="ltx_document"></article></body></html>'
            ).encode(),
        )

    cache_root = tmp_path / "cache"
    service = ArcPaperService(
        cache_root=cache_root,
        arxiv_html=_provider(cache_root, handler),
    )
    bundle = service.fetch_arxiv_html_bundle(versioned)
    refreshed: list[str] = []

    def refresh_bundle(requested: str, *, refresh: bool = False):
        assert refresh is True
        refreshed.append(requested)
        return bundle

    monkeypatch.setattr(
        service, "get_metadata", lambda *args, **kwargs: {"arxiv_id": paper_id}
    )
    monkeypatch.setattr(service, "get_citers", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "fetch_arxiv_html_bundle", refresh_bundle)
    monkeypatch.setattr(service, "parse_bundle", lambda source_bundle: None)
    monkeypatch.setattr(service, "parse_arxiv_pdf", lambda *args, **kwargs: None)

    result = service.update_cache(paper_ids=(paper_id,))

    assert refreshed == [f"arXiv:{versioned}"]
    assert any(
        item.component == "arxiv-html" and item.status == "updated"
        for item in result.records
    )


def test_legacy_fetch_stays_single_file_and_new_cli_registry_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url) == MAIN_URL:
            return _response(
                request,
                content=_article('<img src="paper.png">'),
            )
        return _response(request, content=b"png", media_type="image/png")

    provider = _provider(tmp_path, handler)
    provider.fetch(PAPER_ID)
    assert calls == [MAIN_URL]

    assert OPERATION_REGISTRY["fetch-arxiv-html-bundle"].effect_flags == frozenset(
        {OperationEffect.NETWORK, OperationEffect.CACHE_WRITE}
    )
    assert OPERATION_REGISTRY["export-arxiv-html-bundle"].effect_flags == frozenset(
        {
            OperationEffect.NETWORK,
            OperationEffect.CACHE_WRITE,
            OperationEffect.ARBITRARY_LOCAL_PATH,
        }
    )

    observed: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "arc_paper.cli.dispatch_operation",
        lambda operation, parameters: observed.append((operation, parameters))
        or {"source": "handoff/source.html", "warnings": []},
    )
    assert main(
        [
            "export-arxiv-html-bundle",
            f"{PAPER_ID}v1",
            "--output-dir",
            "handoff",
            "--cache-root",
            "/cache",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert observed == [
        (
            "export-arxiv-html-bundle",
            {
                "paper_id": f"{PAPER_ID}v1",
                "refresh": False,
                "output_dir": "handoff",
                "cache_root": "/cache",
            },
        )
    ]
