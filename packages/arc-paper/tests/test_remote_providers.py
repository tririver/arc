from __future__ import annotations

import json
import threading
import time
import urllib.parse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from arc_paper.html_acquisition import HttpxRequestGateTransport
from arc_paper import ArcPaperService
from arc_paper.providers import (
    Ar5ivProvider,
    ArxivHtmlProvider,
    ArxivPdfProvider,
    InspireProvider,
    RemoteCacheError,
    RemoteRequestCache,
    describe_inspire_citer_request,
)
from arc_paper.providers.ar5iv import MAX_HTML_BYTES
from arc_paper.providers._request_gate import HostRequestGate
from arc_paper.providers.base import ProviderError
from arc_paper.providers.arxiv_html import arxiv_html_url
from arc_paper.providers.arxiv_pdf import arxiv_pdf_url
from arc_paper.source_repository import SourceRepository
from arc_paper.sources import SourceFormat, SourceOrigin, SourceOriginKind


def _response(
    request: httpx.Request,
    *,
    content: bytes,
    media_type: str,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=request,
        content=content,
        headers={"content-type": media_type, **(headers or {})},
    )


def _gate_response(starts: list[float], started: float) -> httpx.Response:
    starts.append(started)
    return httpx.Response(200)


def test_ar5iv_fetches_html_into_source_repository_and_replays_without_network(
    tmp_path,
):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return _response(request, content=b"<html>paper</html>", media_type="text/html")

    first_provider = Ar5ivProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    first = first_provider.fetch("arXiv:0911.3380")
    second_provider = Ar5ivProvider(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    AssertionError("cache hit must not access the network")
                )
            )
        ),
    )
    second = second_provider.fetch("0911.3380")

    assert first.content_identity == second.content_identity
    assert first.source_format is SourceFormat.HTML
    assert first.origin.provider == "ar5iv"
    assert SourceRepository(tmp_path).read_bytes(second) == b"<html>paper</html>"
    assert calls == ["https://ar5iv.labs.arxiv.org/html/0911.3380"]
    assert all("/pdf/" not in url for url in calls)


def test_official_arxiv_html_fetches_directly_and_replays_from_cache(tmp_path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return _response(
            request,
            content=(
                b'<html><head><base href="/html/0911.3380v3/"></head>'
                b"<body>paper</body></html>"
            ),
            media_type="text/html",
        )

    provider = ArxivHtmlProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        request_gate=HostRequestGate(minimum_interval=0),
        dependency_resolver=lambda _host: ("8.8.8.8",),
        dependency_transport_factory=lambda client, request_gate: HttpxRequestGateTransport(
            client, request_gate
        ),
    )
    first = provider.fetch("arXiv:0911.3380v2")
    replay = ArxivHtmlProvider(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    AssertionError("cache hit must not access the network")
                )
            )
        ),
        request_gate=HostRequestGate(minimum_interval=0),
    ).fetch("0911.3380")

    assert arxiv_html_url("arXiv:0911.3380v2") == (
        "https://arxiv.org/html/0911.3380"
    )
    assert first.content_identity == replay.content_identity
    assert replay.origin.provider == "arxiv-html"
    assert first.origin.metadata == {
        "arxiv_id": "0911.3380",
        "arxiv_version": "v3",
        "document_id": "arXiv:0911.3380",
    }
    assert replay.origin.metadata == first.origin.metadata
    assert calls == ["https://arxiv.org/html/0911.3380"]


def _versioned_official_html(
    aid: str,
    version: str,
    body: str = '<article class="ltx_document"></article>',
    *,
    base: str = "",
) -> bytes:
    base_tag = f'<base href="{base}">' if base else ""
    return (
        f"<html><head>{base_tag}</head><body>"
        '<nav class="html-header-nav">'
        '<a class="header-button" title="Back to abstract page" '
        f'aria-label="Back to abstract page" href="/abs/{aid}{version}">Back</a>'
        '<a class="header-button" title="Download PDF" '
        f'href="/pdf/{aid}{version}" target="_blank">PDF</a>'
        f"</nav>{body}</body></html>"
    ).encode()


def test_official_bundle_preserves_explicit_version_url_key_and_codec(tmp_path):
    from arc_paper import (
        html_source_bundle_from_document,
        html_source_bundle_to_document,
    )
    from arc_paper.providers.arxiv_html import arxiv_html_bundle_url

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if url == "https://arxiv.org/html/2608.20415v1":
            return _response(
                request,
                content=_versioned_official_html(
                    "2608.20415",
                    "v1",
                    '<article class="ltx_document"><img src="2608.20415v1/panel.png"></article>',
                ),
                media_type="text/html",
            )
        if url == "https://arxiv.org/html/2608.20415v1/panel.png":
            return _response(request, content=b"png-v1", media_type="image/png")
        raise AssertionError(url)

    provider = ArxivHtmlProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        request_gate=HostRequestGate(minimum_interval=0),
        dependency_resolver=lambda _host: ("8.8.8.8",),
        dependency_transport_factory=lambda client, request_gate: HttpxRequestGateTransport(
            client, request_gate
        ),
    )
    bundle = provider.fetch_bundle("arXiv:2608.20415v1")
    service = ArcPaperService(cache_root=tmp_path, arxiv_html=provider)
    service_bundle = service.fetch_arxiv_html_bundle("arXiv:2608.20415v1")

    assert arxiv_html_bundle_url("arXiv:2608.20415v1") == (
        "https://arxiv.org/html/2608.20415v1"
    )
    assert bundle.document_url == "https://arxiv.org/html/2608.20415v1"
    assert bundle.base_url == bundle.document_url
    assert bundle.primary.origin.locator == bundle.document_url
    assert bundle.primary.origin.metadata == {
        "arxiv_id": "2608.20415",
        "arxiv_version": "v1",
        "document_id": "arXiv:2608.20415v1",
    }
    assert html_source_bundle_from_document(
        html_source_bundle_to_document(bundle)
    ) == bundle
    assert service_bundle.bundle_digest == bundle.bundle_digest
    listed = service.list_cache(paper_ids=("2608.20415",)).entries[0]
    component = next(item for item in listed.components if item.name == "arxiv-html")
    assert len(component.storage_entry_ids) == 3
    assert {
        item.request_key
        for item in provider.cache.admin_entries()
        if item.namespace in {"arxiv-html", "arxiv-html-dependencies"}
    } == {"2608.20415v1"}
    assert calls == [
        "https://arxiv.org/html/2608.20415v1",
        "https://arxiv.org/html/2608.20415v1/panel.png",
    ]


def test_exact_version_bundle_does_not_fall_back_to_unversioned_ar5iv(tmp_path):
    class MissingOfficial:
        def fetch_bundle(self, paper_id: str, *, refresh: bool = False):
            raise ProviderError("arxiv_html_not_found", "missing")

    class ForbiddenFallback:
        def fetch_bundle(self, paper_id: str, *, refresh: bool = False):
            raise AssertionError("exact version must not use an unversioned fallback")

    service = ArcPaperService(
        cache_root=tmp_path,
        arxiv_html=MissingOfficial(),  # type: ignore[arg-type]
        ar5iv=ForbiddenFallback(),  # type: ignore[arg-type]
    )

    with pytest.raises(ProviderError) as error:
        service.fetch_arxiv_html_bundle("arXiv:2608.20415v1")

    assert error.value.code == "arxiv_html_version_not_found"


@pytest.mark.parametrize(
    "paper_id",
    ("arXiv:2608.20415v0", "2608.20415v01"),
)
def test_bundle_service_rejects_invalid_explicit_version_before_providers(
    tmp_path,
    paper_id: str,
):
    class ForbiddenProvider:
        def fetch_bundle(self, paper_id: str, *, refresh: bool = False):
            raise AssertionError("invalid explicit version must fail before provider access")

    service = ArcPaperService(
        cache_root=tmp_path,
        arxiv_html=ForbiddenProvider(),  # type: ignore[arg-type]
        ar5iv=ForbiddenProvider(),  # type: ignore[arg-type]
    )

    with pytest.raises(ProviderError) as error:
        service.fetch_arxiv_html_bundle(paper_id)

    assert error.value.code == "arxiv_version_invalid"


def test_official_bundle_separates_latest_and_explicit_version_cache_keys(
    tmp_path,
):
    calls: list[str] = []
    versions = {
        "https://arxiv.org/html/2608.20415": "v3",
        "https://arxiv.org/html/2608.20415v1": "v1",
        "https://arxiv.org/html/2608.20415v2": "v2",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        version = versions.get(url)
        if version is not None:
            return _response(
                request,
                content=_versioned_official_html("2608.20415", version),
                media_type="text/html",
            )
        raise AssertionError(url)

    provider = ArxivHtmlProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        request_gate=HostRequestGate(minimum_interval=0),
        dependency_resolver=lambda _host: ("8.8.8.8",),
        dependency_transport_factory=lambda client, request_gate: HttpxRequestGateTransport(
            client, request_gate
        ),
    )
    latest = provider.fetch_bundle("2608.20415")
    v1 = provider.fetch_bundle("2608.20415v1")
    v2 = provider.fetch_bundle("arXiv:2608.20415v2")
    replay = ArxivHtmlProvider(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    AssertionError("versioned bundle replay must not access HTTP")
                )
            )
        ),
        request_gate=HostRequestGate(minimum_interval=0),
    ).fetch_bundle("2608.20415v1")

    assert latest.primary.origin.metadata["arxiv_version"] == "v3"
    assert latest.primary.origin.metadata["document_id"] == "arXiv:2608.20415"
    assert v1.primary.origin.metadata["arxiv_version"] == "v1"
    assert v2.primary.origin.metadata["arxiv_version"] == "v2"
    assert replay.bundle_digest == v1.bundle_digest
    assert {
        item.request_key
        for item in provider.cache.admin_entries()
        if item.namespace in {"arxiv-html", "arxiv-html-dependencies"}
    } == {"2608.20415", "2608.20415v1", "2608.20415v2"}
    assert calls == list(versions)


@pytest.mark.parametrize(
    ("response_url", "payload", "code"),
    (
        (
            "https://arxiv.org/html/2608.20415v1",
            _versioned_official_html("2608.20415", "v2"),
            "arxiv_html_revision_mismatch",
        ),
        (
            "https://arxiv.org/html/2608.20415v1",
            b'<html><article class="ltx_document"></article></html>',
            "arxiv_html_revision_unverified",
        ),
        (
            "https://arxiv.org/html/2608.20415v2",
            _versioned_official_html("2608.20415", "v2"),
            "arxiv_html_revision_mismatch",
        ),
        (
            "https://arxiv.org/html/2608.20415v1",
            _versioned_official_html(
                "2608.20415",
                "v1",
                base="/html/2608.20415v2/",
            ),
            "arxiv_html_revision_mismatch",
        ),
    ),
)
def test_official_bundle_fails_closed_on_revision_mismatch_or_absence(
    tmp_path,
    response_url: str,
    payload: bytes,
    code: str,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://arxiv.org/html/2608.20415v1":
            if response_url != str(request.url):
                return _response(
                    request,
                    status_code=302,
                    content=b"",
                    media_type="text/html",
                    headers={"location": response_url},
                )
            return _response(request, content=payload, media_type="text/html")
        if str(request.url) == response_url:
            return _response(request, content=payload, media_type="text/html")
        raise AssertionError(str(request.url))

    provider = ArxivHtmlProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        request_gate=HostRequestGate(minimum_interval=0),
    )

    with pytest.raises(ProviderError) as error:
        provider.fetch_bundle("arXiv:2608.20415v1")

    assert error.value.code == code
    assert provider.cache.admin_entry(
        "source", "arxiv-html", "2608.20415v1"
    ) is None


@pytest.mark.parametrize(
    "payload",
    (
        b"<html><head></head><body>no base</body></html>",
        b'<html><head><base href="/html/0911.3380/"></head></html>',
        b'<html><head><base href="/html/9999.9999v2/"></head></html>',
        b'<html><head><base href="/html/0911.3380v0/"></head></html>',
    ),
)
def test_official_html_ignores_absent_or_malformed_revision_base(tmp_path, payload):
    provider = ArxivHtmlProvider(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _response(
                    request, content=payload, media_type="text/html"
                )
            )
        ),
        request_gate=HostRequestGate(minimum_interval=0),
    )

    artifact = provider.fetch("0911.3380")

    assert artifact.origin.metadata == {
        "arxiv_id": "0911.3380",
        "document_id": "arXiv:0911.3380",
    }


def test_official_arxiv_html_caches_404_and_refresh_rechecks_it(tmp_path):
    responses = iter((404, 200))
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        status_code = next(responses)
        return _response(
            request,
            content=b"" if status_code == 404 else b"<html>converted</html>",
            media_type="text/html",
            status_code=status_code,
        )

    provider = ArxivHtmlProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        request_gate=HostRequestGate(minimum_interval=0),
    )
    with pytest.raises(Exception) as first:
        provider.fetch("0911.3380")
    with pytest.raises(Exception) as cached:
        provider.fetch("0911.3380")
    refreshed = provider.fetch("0911.3380", refresh=True)
    replay = provider.fetch("0911.3380")

    assert getattr(first.value, "code", "") == "arxiv_html_not_found"
    assert getattr(cached.value, "code", "") == "arxiv_html_not_found"
    assert refreshed.content_identity == replay.content_identity
    assert (
        provider.cache.get_json("arxiv-html-availability", "0911.3380") is None
    )
    assert calls == [
        "https://arxiv.org/html/0911.3380",
        "https://arxiv.org/html/0911.3380",
    ]


def test_official_html_refresh_invalidates_a_stale_404_after_transient_error(
    tmp_path,
):
    responses = iter((404, 503, 200))
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        status_code = next(responses)
        return _response(
            request,
            content=b"<html>converted</html>",
            media_type="text/html",
            status_code=status_code,
        )

    provider = ArxivHtmlProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        request_gate=HostRequestGate(minimum_interval=0),
    )
    with pytest.raises(Exception) as missing:
        provider.fetch("0911.3380")
    with pytest.raises(Exception) as transient:
        provider.fetch("0911.3380", refresh=True)
    recovered = provider.fetch("0911.3380")

    assert getattr(missing.value, "code", "") == "arxiv_html_not_found"
    assert getattr(transient.value, "code", "") == "arxiv_html_fetch_failed"
    assert recovered.origin.provider == "arxiv-html"
    assert calls == ["https://arxiv.org/html/0911.3380"] * 3


def test_official_html_and_pdf_share_arxiv_gate_but_ar5iv_does_not(tmp_path):
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: _response(
                request,
                content=(b"%PDF-1.7\n" if "/pdf/" in str(request.url) else b"<html>ok</html>"),
                media_type=(
                    "application/pdf" if "/pdf/" in str(request.url) else "text/html"
                ),
            )
        )
    )
    official = ArxivHtmlProvider(cache_root=tmp_path, client=client)
    pdf = ArxivPdfProvider(cache_root=tmp_path, client=client)
    fallback = Ar5ivProvider(cache_root=tmp_path, client=client)

    assert official.request_gate is pdf.request_gate
    assert official.request_gate is not fallback.request_gate
    assert fallback.request_gate.minimum_interval == 0


def test_arxiv_gate_uses_fake_clock_for_interval_and_retry_after(tmp_path):
    class FakeClock:
        def __init__(self):
            self.value = 0.0
            self.sleeps: list[float] = []

        def now(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.value += seconds

    clock = FakeClock()
    gate = HostRequestGate(
        minimum_interval=15,
        clock=clock.now,
        sleeper=clock.sleep,
    )
    starts: list[tuple[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append((str(request.url), clock.now()))
        response = _response(
            request,
            content=(b"%PDF-1.7\n" if "/pdf/" in str(request.url) else b"<html>ok</html>"),
            media_type=(
                "application/pdf" if "/pdf/" in str(request.url) else "text/html"
            ),
        )
        if "/html/" in str(request.url):
            response.headers["retry-after"] = "30"
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))
    official = ArxivHtmlProvider(
        cache_root=tmp_path, client=client, request_gate=gate
    )
    pdf = ArxivPdfProvider(cache_root=tmp_path, client=client, request_gate=gate)

    official.fetch("0911.3380")
    pdf.fetch("0911.3380")

    assert starts == [
        ("https://arxiv.org/html/0911.3380", 0.0),
        ("https://arxiv.org/pdf/0911.3380", 30.0),
    ]
    assert clock.sleeps == [30.0]


def test_file_backed_host_gate_reuses_the_next_start_with_a_fake_clock(tmp_path):
    class FakeClock:
        def __init__(self):
            self.value = 0.0
            self.sleeps: list[float] = []

        def now(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.value += seconds

    clock = FakeClock()
    state_path = tmp_path / "gate" / "arxiv.next-start"
    lock_path = tmp_path / "gate" / "arxiv.lock"
    first = HostRequestGate(
        minimum_interval=15,
        clock=clock.now,
        sleeper=clock.sleep,
        state_path=state_path,
        lock_path=lock_path,
    )
    second = HostRequestGate(
        minimum_interval=15,
        clock=clock.now,
        sleeper=clock.sleep,
        state_path=state_path,
        lock_path=lock_path,
    )
    starts: list[float] = []

    first.request(lambda: _gate_response(starts, clock.now()))
    second.request(lambda: _gate_response(starts, clock.now()))

    assert starts == [0.0, 15.0]
    assert clock.sleeps == [15.0]


def test_shared_arxiv_gate_serializes_html_and_pdf_connections(tmp_path):
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()
    gate = HostRequestGate(minimum_interval=0)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with state_lock:
            active -= 1
        is_pdf = "/pdf/" in str(request.url)
        return _response(
            request,
            content=b"%PDF-1.7\n" if is_pdf else b"<html>ok</html>",
            media_type="application/pdf" if is_pdf else "text/html",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    official = ArxivHtmlProvider(
        cache_root=tmp_path, client=client, request_gate=gate
    )
    pdf = ArxivPdfProvider(cache_root=tmp_path, client=client, request_gate=gate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(official.fetch, "0911.3380"),
            executor.submit(pdf.fetch, "2402.00001"),
        )
        tuple(future.result() for future in futures)

    assert maximum_active == 1


def test_ar5iv_refresh_replaces_request_mapping_without_changing_content_identity(
    tmp_path,
):
    payloads = iter((b"<html>stale</html>", b"<html>fresh</html>"))
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return _response(
            request,
            content=next(payloads),
            media_type="text/html",
        )

    provider = Ar5ivProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        request_gate=HostRequestGate(minimum_interval=0),
    )
    stale = provider.fetch("0911.3380")
    fresh = provider.fetch("0911.3380", refresh=True)
    replay = provider.fetch("0911.3380")

    assert stale.content_identity != fresh.content_identity
    assert replay.content_identity == fresh.content_identity
    assert SourceRepository(tmp_path).read_bytes(replay) == b"<html>fresh</html>"
    assert calls == [
        "https://ar5iv.labs.arxiv.org/html/0911.3380",
        "https://ar5iv.labs.arxiv.org/html/0911.3380",
    ]


def test_ar5iv_concurrent_cache_fill_fetches_once(tmp_path):
    call_count = 0
    count_lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.02)
        return _response(request, content=b"<html>paper</html>", media_type="text/html")

    provider = Ar5ivProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        artifacts = list(
            executor.map(lambda _: provider.fetch("0911.3380"), range(8))
        )

    assert call_count == 1
    assert len({artifact.content_identity for artifact in artifacts}) == 1


def test_fetch_source_checks_reads_and_fetches_under_request_lock(
    tmp_path, monkeypatch
):
    cache = RemoteRequestCache(tmp_path)
    original_lock = cache._request_lock
    original_get = cache.get_source
    lock_depth = 0
    get_calls = 0
    fetch_calls = 0

    @contextmanager
    def tracked_lock(kind, namespace, digest):
        nonlocal lock_depth
        with original_lock(kind, namespace, digest):
            lock_depth += 1
            try:
                yield
            finally:
                lock_depth -= 1

    def checked_get(*args, **kwargs):
        nonlocal get_calls
        assert lock_depth == 1
        get_calls += 1
        return original_get(*args, **kwargs)

    def fetch():
        nonlocal fetch_calls
        assert lock_depth == 1
        fetch_calls += 1
        return f"source-{fetch_calls}".encode()

    monkeypatch.setattr(cache, "_request_lock", tracked_lock)
    monkeypatch.setattr(cache, "get_source", checked_get)
    origin = SourceOrigin(
        SourceOriginKind.REMOTE_PROVIDER,
        provider="fixture",
    )
    first = cache.fetch_source(
        "source",
        "request",
        source_format=SourceFormat.HTML,
        media_type="text/html",
        origin=origin,
        fetch=fetch,
    )
    replayed = cache.fetch_source(
        "source",
        "request",
        source_format=SourceFormat.HTML,
        media_type="text/html",
        origin=origin,
        fetch=fetch,
    )
    refreshed = cache.fetch_source(
        "source",
        "request",
        source_format=SourceFormat.HTML,
        media_type="text/html",
        origin=origin,
        fetch=fetch,
        refresh=True,
    )

    assert replayed.content_identity == first.content_identity
    assert refreshed.content_identity != first.content_identity
    assert get_calls == 2
    assert fetch_calls == 2
    assert lock_depth == 0


def test_fetch_json_checks_reads_and_fetches_under_request_lock(
    tmp_path, monkeypatch
):
    cache = RemoteRequestCache(tmp_path)
    original_lock = cache._request_lock
    original_get = cache.get_json
    lock_depth = 0
    get_calls = 0
    fetch_calls = 0

    @contextmanager
    def tracked_lock(kind, namespace, digest):
        nonlocal lock_depth
        with original_lock(kind, namespace, digest):
            lock_depth += 1
            try:
                yield
            finally:
                lock_depth -= 1

    def checked_get(*args, **kwargs):
        nonlocal get_calls
        assert lock_depth == 1
        get_calls += 1
        return original_get(*args, **kwargs)

    def fetch():
        nonlocal fetch_calls
        assert lock_depth == 1
        fetch_calls += 1
        return {"generation": fetch_calls}

    monkeypatch.setattr(cache, "_request_lock", tracked_lock)
    monkeypatch.setattr(cache, "get_json", checked_get)
    first = cache.fetch_json("json", "request", fetch=fetch)
    replayed = cache.fetch_json("json", "request", fetch=fetch)
    refreshed = cache.fetch_json(
        "json", "request", fetch=fetch, refresh=True
    )

    assert first == replayed == {"generation": 1}
    assert refreshed == {"generation": 2}
    assert get_calls == 2
    assert fetch_calls == 2
    assert lock_depth == 0


def test_ar5iv_rejects_wrong_media_and_oversized_header(tmp_path):
    wrong_media = Ar5ivProvider(
        cache_root=tmp_path / "wrong",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _response(
                    request, content=b"not html", media_type="text/plain"
                )
            )
        ),
    )
    with pytest.raises(Exception) as wrong_error:
        wrong_media.fetch("0911.3380")
    assert getattr(wrong_error.value, "code", "") == "ar5iv_media_type_invalid"

    def oversized(request: httpx.Request) -> httpx.Response:
        response = _response(request, content=b"", media_type="text/html")
        response.headers["content-length"] = str(MAX_HTML_BYTES + 1)
        return response

    too_large = Ar5ivProvider(
        cache_root=tmp_path / "large",
        client=httpx.Client(transport=httpx.MockTransport(oversized)),
    )
    with pytest.raises(Exception) as size_error:
        too_large.fetch("0911.3380")
    assert getattr(size_error.value, "code", "") == "ar5iv_html_too_large"


def test_arxiv_pdf_is_explicit_content_addressed_source(tmp_path):
    calls: list[str] = []
    pdf = b"%PDF-1.7\nfixture\n%%EOF\n"

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return _response(request, content=pdf, media_type="application/pdf")

    provider = ArxivPdfProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    artifact = provider.fetch("hep-th/0601001")

    assert arxiv_pdf_url("arXiv:hep-th/0601001") == (
        "https://arxiv.org/pdf/hep-th/0601001"
    )
    assert artifact.source_format is SourceFormat.PDF
    assert artifact.origin.provider == "arxiv-pdf"
    assert SourceRepository(tmp_path).read_bytes(artifact) == pdf
    assert calls == ["https://arxiv.org/pdf/hep-th/0601001"]


def test_inspire_metadata_cache_is_atomic_integrity_checked_and_concurrent(tmp_path):
    call_count = 0
    count_lock = threading.Lock()
    record = {
        "id": "123",
        "metadata": {
            "control_number": 123,
            "titles": [{"title": "Cached paper"}],
            "arxiv_eprints": [{"value": "0911.3380"}],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.02)
        return _response(
            request,
            content=json.dumps(record).encode(),
            media_type="application/json",
        )

    provider = InspireProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(
            executor.map(lambda _: provider.get_metadata("0911.3380"), range(8))
        )

    assert call_count == 1
    assert {value["title"] for value in values} == {"Cached paper"}

    manifest = next(
        (tmp_path / "remote-request-cache" / "v2" / "json").glob(
            "inspire-record/*/*/manifest.json"
        )
    )
    entry_dir = manifest.parent
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    entry_dir.joinpath(manifest_value["payload_file"]).write_bytes(
        b'{"tampered":true}'
    )
    cached = RemoteRequestCache(tmp_path)
    with pytest.raises(RemoteCacheError) as error:
        cached.get_json("inspire-record", "arXiv:0911.3380")
    assert error.value.code == "remote_cache_json_corrupt"


def test_json_payload_is_not_published_without_manifest(tmp_path, monkeypatch):
    cache = RemoteRequestCache(tmp_path)
    original = cache._atomic_write

    def interrupted(path, payload):
        if path.name == "manifest.json":
            raise OSError("simulated interruption")
        original(path, payload)

    monkeypatch.setattr(cache, "_atomic_write", interrupted)
    with pytest.raises(OSError, match="simulated"):
        cache.fetch_json("metadata", "request", fetch=lambda: {"ok": True})

    assert cache.get_json("metadata", "request") is None


def test_json_refresh_interruption_preserves_previous_manifest_generation(
    tmp_path, monkeypatch
):
    cache = RemoteRequestCache(tmp_path)
    assert cache.fetch_json(
        "metadata", "request", fetch=lambda: {"generation": "old"}
    ) == {"generation": "old"}

    manifest_path = next(
        (tmp_path / "remote-request-cache" / "v2" / "json").glob(
            "metadata/*/*/manifest.json"
        )
    )
    entry_dir = manifest_path.parent
    old_manifest = manifest_path.read_bytes()
    old_manifest_value = json.loads(old_manifest)
    old_payload_path = entry_dir / old_manifest_value["payload_file"]
    old_payload = old_payload_path.read_bytes()
    original = cache._atomic_write

    def interrupted(path, payload):
        if path.name == "manifest.json":
            raise OSError("simulated refresh interruption")
        original(path, payload)

    monkeypatch.setattr(cache, "_atomic_write", interrupted)
    with pytest.raises(OSError, match="simulated refresh"):
        cache.fetch_json(
            "metadata",
            "request",
            fetch=lambda: {"generation": "new"},
            refresh=True,
        )

    assert manifest_path.read_bytes() == old_manifest
    assert old_payload_path.read_bytes() == old_payload
    assert cache.get_json("metadata", "request") == {"generation": "old"}


def test_remote_source_manifest_revalidates_repository_bytes(tmp_path):
    provider = Ar5ivProvider(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _response(
                    request, content=b"<html>paper</html>", media_type="text/html"
                )
            )
        ),
    )
    artifact = provider.fetch("0911.3380")
    source_path = next(
        (tmp_path / "source-repository" / "v1" / "html" / "sha256").glob(
            "*/*/source"
        )
    )
    source_path.write_bytes(b"corrupt")

    cache = RemoteRequestCache(tmp_path)
    with pytest.raises(RemoteCacheError) as error:
        cache.get_source(
            "ar5iv-html",
            "0911.3380",
            source_format=SourceFormat.HTML,
            media_type="text/html",
            origin=artifact.origin,
        )
    assert error.value.code == "remote_cache_source_corrupt"


def test_inspire_search_preserves_query_limit_and_normalizes_mathml(tmp_path):
    observed: dict[str, str] = {}
    math = (
        '<math display="inline"><mi>P</mi><mi>l</mi><mi>a</mi>'
        "<mi>n</mi><mi>c</mi><mi>k</mi></math>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(dict(request.url.params))
        return _response(
            request,
            content=json.dumps(
                {
                    "hits": {
                        "hits": [
                            {
                                "id": "123",
                                "metadata": {
                                    "control_number": 123,
                                    "titles": [{"title": f"Constraints from {math}"}],
                                    "arxiv_eprints": [{"value": "0911.3380v2"}],
                                },
                            }
                        ]
                    }
                }
            ).encode(),
            media_type="application/json",
        )

    provider = InspireProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    values = provider.search_metadata("specific mechanism", limit=7)

    assert observed["q"] == "specific mechanism"
    assert observed["size"] == "7"
    assert values[0]["paper_id"] == "arXiv:0911.3380"
    assert values[0]["title"] == "Constraints from Planck"


def test_inspire_references_are_normalized_enriched_and_cached(tmp_path):
    record = {
        "id": "123",
        "metadata": {
            "control_number": 123,
            "titles": [{"title": "Paper"}],
            "arxiv_eprints": [{"value": "0911.3380v2"}],
            "references": [
                {
                    "record": {
                        "$ref": "https://inspirehep.net/api/literature/456"
                    },
                    "reference": {
                        "title": "Reference",
                        "arxiv_eprint": "0801.0001v3",
                        "dois": [{"value": "10.1000/reference-old"}],
                    },
                }
            ],
        },
    }
    reference = {
        "id": "456",
        "metadata": {
            "control_number": 456,
            "titles": [{"title": "Enriched reference"}],
            "authors": [{"full_name": "Ref Author"}],
            "abstracts": [{"value": "Reference abstract."}],
            "arxiv_eprints": [{"value": "0801.0001"}],
            "dois": [
                {"value": "10.1000/reference-primary"},
                {"value": "10.1000/reference-secondary"},
            ],
            "citation_count": 11,
        },
    }
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        payload = reference if request.url.path.endswith("/456") else record
        return _response(
            request,
            content=json.dumps(payload).encode(),
            media_type="application/json",
        )

    provider = InspireProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    values = provider.get_references("arXiv:0911.3380v2", enrich=True)

    assert values[0]["paper_id"] == "arXiv:0801.0001"
    assert values[0]["title"] == "Enriched reference"
    assert values[0]["authors"] == ["Ref Author"]
    assert values[0]["doi"] == "10.1000/reference-primary"
    assert values[0]["dois"] == [
        "10.1000/reference-primary",
        "10.1000/reference-secondary",
        "10.1000/reference-old",
    ]
    assert values[0]["metadata_enriched"] is True
    assert len(calls) == 2

    cached = InspireProvider(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    AssertionError("cached reference lookup must not use the network")
                )
            )
        ),
    )
    assert cached.get_references("0911.3380", enrich=True)[0]["citation_count"] == 11


def test_inspire_doi_lookup_uses_normalized_content_addressed_request_cache(tmp_path):
    calls: list[str] = []
    record = {
        "id": "123",
        "metadata": {
            "control_number": 123,
            "titles": [{"title": "DOI paper"}],
            "arxiv_eprints": [{"value": "0911.3380"}],
            "dois": [
                {"value": "10.1000/example"},
                {"value": "10.1000/ALTERNATE"},
                {"value": "10.1000/example"},
            ],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        assert request.url.path == "/api/literature"
        assert request.url.params["q"] == "doi:10.1000/example"
        return _response(
            request,
            content=json.dumps({"hits": {"hits": [record]}}).encode(),
            media_type="application/json",
        )

    provider = InspireProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    first = provider.get_metadata("doi:10.1000/EXAMPLE")
    second = provider.get_metadata("10.1000/example")

    assert first == second
    assert first["paper_id"] == "arXiv:0911.3380"
    assert first["identifiers"]["doi"] == "10.1000/example"
    assert first["doi"] == "10.1000/example"
    assert first["dois"] == [
        "10.1000/example",
        "10.1000/alternate",
    ]
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("limit", "sort"),
    [(1, "mostrecent"), (3, "mostcited")],
)
def test_inspire_citers_use_recid_query_and_request_specific_cache(
    tmp_path, limit, sort
):
    calls: list[str] = []
    record = {
        "id": "123",
        "metadata": {
            "control_number": 123,
            "titles": [{"title": "Paper"}],
            "arxiv_eprints": [{"value": "0911.3380"}],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/arxiv/0911.3380"):
            payload = record
        else:
            query = urllib.parse.parse_qs(request.url.query.decode())
            assert query["q"] == ["refersto:recid:123"]
            assert query["size"] == [str(limit)]
            assert query["sort"] == [sort]
            payload = {
                "hits": {
                    "hits": [
                        {
                            "id": str(index),
                            "metadata": {
                                "control_number": index,
                                "titles": [{"title": f"Citer {index}"}],
                            },
                        }
                        for index in range(limit)
                    ]
                }
            }
        return _response(
            request,
            content=json.dumps(payload).encode(),
            media_type="application/json",
        )

    provider = InspireProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    values = provider.get_citers("0911.3380", limit=limit, sort=sort)
    assert [item["title"] for item in values] == [
        f"Citer {index}" for index in range(limit)
    ]
    assert len(calls) == 2

    calls_before_replay = len(calls)
    replay = provider.get_citers("0911.3380", limit=limit, sort=sort)
    assert replay == values
    assert len(calls) == calls_before_replay

    refreshed = provider.get_citers(
        "0911.3380", limit=limit, sort=sort, refresh=True
    )
    assert refreshed == values
    assert len(calls) == calls_before_replay + 2


def test_inspire_citers_ignore_legacy_normalized_cache_and_store_raw_hits(tmp_path):
    cache = RemoteRequestCache(tmp_path)
    record = {
        "id": "123",
        "metadata": {
            "control_number": 123,
            "titles": [{"title": "Origin"}],
            "arxiv_eprints": [{"value": "0911.3380"}],
        },
    }
    cache.fetch_json(
        "inspire-record",
        "arXiv:0911.3380",
        fetch=lambda: record,
    )
    legacy_key = json.dumps(
        {"recid": "123", "sort": "mostrecent", "limit": 1000},
        sort_keys=True,
        separators=(",", ":"),
    )
    cache.fetch_json(
        "inspire-citers",
        legacy_key,
        fetch=lambda: [{"paper_id": "inspire:456", "title": "Legacy"}],
    )
    raw_citer = {
        "id": "456",
        "metadata": {
            "control_number": 456,
            "titles": [{"title": "Current"}],
            "dois": [{"value": "10.1000/current"}],
        },
    }
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(
            request,
            content=json.dumps({"hits": {"hits": [raw_citer]}}).encode(),
            media_type="application/json",
        )

    provider = InspireProvider(
        request_cache=cache,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.get_citers("0911.3380")[0]["dois"] == [
        "10.1000/current"
    ]
    assert calls == 1

    request = describe_inspire_citer_request("123")
    cached = cache.get_json("inspire-citers", request.request_key)
    assert cached == {"hits": {"hits": [raw_citer]}}
    assert "metadata" in cached["hits"]["hits"][0]
    assert "paper_id" not in cached["hits"]["hits"][0]

    offline = InspireProvider(
        request_cache=RemoteRequestCache(tmp_path),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    AssertionError("raw citer cache replay must be offline")
                )
            )
        ),
    )
    assert offline.get_citers("0911.3380")[0]["doi"] == "10.1000/current"


def test_inspire_citers_repair_malformed_current_cache_once_under_concurrency(
    tmp_path,
):
    cache = RemoteRequestCache(tmp_path)
    cache.fetch_json(
        "inspire-record",
        "arXiv:0911.3380",
        fetch=lambda: {
            "id": "123",
            "metadata": {
                "control_number": 123,
                "arxiv_eprints": [{"value": "0911.3380"}],
            },
        },
    )
    request = describe_inspire_citer_request("123")
    cache.fetch_json(
        "inspire-citers",
        request.request_key,
        fetch=lambda: {"hits": {"hits": ["not-an-object"]}},
    )
    call_count = 0
    call_guard = threading.Lock()

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        with call_guard:
            call_count += 1
        return _response(
            http_request,
            content=json.dumps(
                {
                    "hits": {
                        "hits": [
                            {
                                "id": "456",
                                "metadata": {
                                    "control_number": 456,
                                    "titles": [{"title": "Repaired"}],
                                },
                            }
                        ]
                    }
                }
            ).encode(),
            media_type="application/json",
        )

    provider = InspireProvider(
        request_cache=cache,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(
            executor.map(lambda _: provider.get_citers("0911.3380"), range(8))
        )

    assert call_count == 1
    assert {items[0]["title"] for items in values} == {"Repaired"}


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"hits": []}, {"hits": {}}, {"hits": {"hits": {}}}],
)
def test_inspire_citers_reject_malformed_new_payload_without_caching(
    tmp_path, payload
):
    cache = RemoteRequestCache(tmp_path)
    cache.fetch_json(
        "inspire-record",
        "arXiv:0911.3380",
        fetch=lambda: {
            "id": "123",
            "metadata": {
                "control_number": 123,
                "arxiv_eprints": [{"value": "0911.3380"}],
            },
        },
    )
    provider = InspireProvider(
        request_cache=cache,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _response(
                    request,
                    content=json.dumps(payload).encode(),
                    media_type="application/json",
                )
            )
        ),
    )

    with pytest.raises(Exception) as exc_info:
        provider.get_citers("0911.3380")

    assert getattr(exc_info.value, "code", "") == "inspire_response_invalid"
    request = describe_inspire_citer_request("123")
    assert cache.admin_entry("json", "inspire-citers", request.request_key) is None


def test_inspire_citers_reject_invalid_sort_before_metadata_lookup(tmp_path):
    provider = InspireProvider(
        cache_root=tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(
                    AssertionError("invalid sort must fail before metadata lookup")
                )
            )
        ),
    )

    with pytest.raises(Exception) as exc_info:
        provider.get_citers("0911.3380", sort="oldest")

    assert getattr(exc_info.value, "code", "") == "unsupported_citer_sort"


def test_search_citers_reuses_existing_inspire_record_and_citer_cache(tmp_path):
    calls: list[str] = []
    record = {
        "id": "123",
        "metadata": {
            "control_number": 123,
            "titles": [{"title": "Origin"}],
            "arxiv_eprints": [{"value": "0911.3380"}],
            "citation_count": 2,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/arxiv/0911.3380"):
            payload = record
        else:
            assert request.url.params["q"] == "refersto:recid:123"
            assert request.url.params["size"] == "1000"
            assert request.url.params["sort"] == "mostrecent"
            payload = {
                "hits": {
                    "hits": [
                        {
                            "id": str(index),
                            "metadata": {
                                "control_number": index,
                                "titles": [
                                    {"title": f"Specific mechanism {index}"}
                                ],
                            },
                        }
                        for index in range(2)
                    ]
                }
            }
        return _response(
            request,
            content=json.dumps(payload).encode(),
            media_type="application/json",
        )

    provider = InspireProvider(
        cache_root=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    service = ArcPaperService(cache_root=tmp_path, inspire=provider)

    first = service.search_citers("0911.3380", ["specific mechanism"])
    calls_after_first = len(calls)
    second = service.search_citers("0911.3380", ["specific mechanism"])

    assert first == second
    assert first["scan_complete"] is True
    assert first["matched_count"] == 2
    assert calls_after_first == 2
    assert len(calls) == calls_after_first
