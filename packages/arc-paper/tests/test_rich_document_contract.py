from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from arc_paper import cli, service
from arc_paper.cache import (
    CachePaths,
    parsed_source_cache_path,
    read_json,
    rich_document_cache_path,
    write_json,
    write_paper_alias,
)
from arc_paper.parse.document import DOCUMENT_SCHEMA_VERSION
from arc_paper.parse.source import PARSER_VERSION, parse_source_input
from arc_paper.providers.ar5iv import Ar5ivProvider


LEGACY_PARSED_SOURCE_KEYS = {
    "paper_id",
    "parser_version",
    "source_hash",
    "toc",
    "sections",
    "equations",
}


RICH_HTML = """
<html>
  <body>
    <article class="ltx_document">
      <h1 class="ltx_title">A rich paper</h1>
      <div class="ltx_authors">Ada Author</div>
      <section id="S1">
        <h2>1 Introduction</h2>
        <p id="p1">Before <a class="ltx_ref" href="#bib.bib1">[1]</a>.</p>
        <table class="ltx_equation" id="S1.E1">
          <tr>
            <td><math alttext="x=y"><semantics><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow></semantics></math></td>
            <td class="ltx_eqn_eqno">(1)</td>
          </tr>
        </table>
        <figure class="ltx_figure" id="S1.F1">
          <img src="assets/plot.svg" alt="A plot"/>
          <figcaption><span class="ltx_tag">Figure 1:</span> Test plot.</figcaption>
        </figure>
        <table class="ltx_table" id="S1.T1">
          <caption><span class="ltx_tag">Table 1:</span> Test table.</caption>
          <tr><th>Quantity</th><th>Value</th></tr>
          <tr><td rowspan="2">mass</td><td>1</td></tr>
          <tr><td>2</td></tr>
        </table>
      </section>
      <section class="ltx_bibliography" id="bib">
        <h2>References</h2>
        <ul class="ltx_biblist">
          <li class="ltx_bibitem" id="bib.bib1">
            <span class="ltx_tag ltx_tag_bibitem">[1]</span>
            <span class="ltx_bibblock">A. Author, A cited work.</span>
          </li>
        </ul>
      </section>
    </article>
  </body>
</html>
"""


def test_rich_html_parse_stores_versioned_document_without_removing_legacy_fields():
    parsed = parse_source_input(html_text=RICH_HTML, source_id="rich-paper")

    assert PARSER_VERSION == 14
    assert LEGACY_PARSED_SOURCE_KEYS <= set(parsed)
    assert parsed["parser_version"] == 14

    document = parsed["document"]
    assert document["schema_version"] == "arc.paper.document.v2"
    assert set(document) >= {
        "schema_version",
        "blocks",
        "equations",
        "figures",
        "tables",
        "footnotes",
        "bibliography",
        "links",
        "assets",
        "integrity",
    }
    assert document["blocks"]
    orders = [block["order"] for block in document["blocks"]]
    assert orders == sorted(orders)
    assert len(orders) == len(set(orders))
    assert document["equations"][0]["id"] == "S1.E1"
    assert document["equations"][0]["tex"] == ["x=y"]
    assert document["equations"][0]["printed_equation_number"] == "1"
    assert document["figures"][0]["id"] == "S1.F1"
    assert document["figures"][0]["tag"] == "Figure 1:"
    assert document["tables"][0]["id"] == "S1.T1"
    assert document["tables"][0]["tag"] == "Table 1:"
    assert document["tables"][0]["grid"][2][0] == {
        "text": "mass",
        "source_row": 1,
        "source_column": 0,
    }
    assert document["bibliography"][0]["id"] == "bib.bib1"


def test_inline_runs_separate_text_math_citations_and_links_with_stable_tokens():
    html = r"""
    <article class="ltx_document"><p id="p1">The
      <math alttext="t_{NL}"><semantics><mi>t</mi></semantics></math>
      term follows <a class="ltx_ref" href="#bib.bib1">[1]</a>; see
      <a href="https://example.test/paper">details</a> and
      <math><semantics><annotation encoding="application/x-tex">f_{NL}^{2}</annotation></semantics></math>.
    </p><ul><li class="ltx_bibitem" id="bib.bib1">Reference</li></ul></article>
    """

    first = parse_source_input(html_text=html, source_id="inline-runs")["document"]["blocks"][0]
    second = parse_source_input(html_text=html, source_id="inline-runs")["document"]["blocks"][0]
    runs = first["inline_runs"]

    assert [item["kind"] for item in runs] == ["text", "math", "text", "citation", "text", "link", "text", "math", "text"]
    assert [item.get("tex") for item in runs if item["kind"] == "math"] == [r"t_{NL}", r"f_{NL}^{2}"]
    assert [(item["token_id"], item["content_hash"]) for item in runs] == [
        (item["token_id"], item["content_hash"]) for item in second["inline_runs"]
    ]
    assert len({item["token_id"] for item in runs}) == len(runs)


def test_display_equation_layout_preserves_rows_cells_alignment_breaks_and_identity():
    parsed = parse_source_input(
        html_text=r"""
        <article class="ltx_document"><table id="EG" class="ltx_equationgroup">
          <tr id="E1" class="ltx_equation"><td class="ltx_align_right"><math alttext="a"/></td>
            <td class="ltx_align_left"><math alttext="=b"/></td><td class="ltx_eqn_eqno">(1a)</td></tr>
          <tr id="E2" class="ltx_equation"><td align="right"><math alttext="c"/></td>
            <td align="left"><math alttext="=d"/></td><td class="ltx_eqn_eqno">(1b)</td></tr>
        </table></article>
        """,
        source_id="layout",
    )
    equations = parsed["document"]["equations"]

    assert [item["layout"]["group_id"] for item in equations] == ["EG", "EG"]
    assert [[cell["alignment"] for cell in item["layout"]["rows"][0]["cells"]] for item in equations] == [
        ["right", "left"], ["right", "left"]
    ]
    assert [item["layout"]["rows"][0]["number"] for item in equations] == ["1a", "1b"]
    assert [item["layout"]["rows"][0]["label"] for item in equations] == ["E1", "E2"]


def test_real_ar5iv_equation_group_preserves_rows_tex_numbers_and_block_order():
    html = r"""
    <article class="ltx_document"><section id="S2">
      <p id="S2.p1">Before the coupled equations.</p>
      <table id="S2.EG1" class="ltx_equationgroup ltx_eqn_align ltx_eqn_table">
        <tbody id="S2.E1"><tr class="ltx_equation ltx_eqn_row">
          <td><math alttext="fallback-a"><semantics><mi>a</mi><annotation encoding="application/x-tex">\displaystyle a=b</annotation></semantics></math></td>
          <td class="ltx_eqn_eqno"><span class="ltx_tag ltx_tag_equation">(7a)</span></td>
        </tr></tbody>
        <tbody id="S2.E2"><tr class="ltx_equation ltx_eqn_row">
          <td><math alttext="fallback-c"><semantics><mi>c</mi><annotation encoding="application/x-tex">\displaystyle c=d</annotation></semantics></math></td>
          <td class="ltx_eqn_eqno"><span class="ltx_tag ltx_tag_equation">(7b)</span></td>
        </tr></tbody>
      </table>
      <p id="S2.p2">After the coupled equations.</p>
    </section></article>
    """

    parsed = parse_source_input(html_text=html, source_id="equation-group")
    equations = parsed["document"]["equations"]

    assert [item["id"] for item in equations] == ["S2.E1", "S2.E2"]
    assert [item["tex"] for item in equations] == [[r"\displaystyle a=b"], [r"\displaystyle c=d"]]
    assert [item["printed_equation_number"] for item in equations] == ["7a", "7b"]
    assert [(item["group_id"], item["group_row"], item["group_row_count"]) for item in equations] == [
        ("S2.EG1", 1, 2),
        ("S2.EG1", 2, 2),
    ]
    equation_blocks = [item for item in parsed["document"]["blocks"] if item["kind"] == "equation"]
    assert [item["block_id"] for item in equation_blocks] == ["S2.E1", "S2.E2"]
    assert [item["kind"] for item in parsed["document"]["blocks"]] == ["prose", "equation", "equation", "prose"]
    assert parsed["document"]["integrity"]["status"] == "complete"


def test_heading_and_list_blocks_preserve_rendering_structure():
    parsed = parse_source_input(
        html_text="""
        <article class="ltx_document"><section id="S2">
          <h3 id="S2.title"><span class="ltx_tag ltx_tag_subsection">2.1 </span>Assumptions</h3>
          <ol id="S2.list" start="3"><li id="S2.i1">First item
            <ul><li id="S2.i1.a">Nested item</li></ul>
          </li><li id="S2.i2" value="7">Second item</li></ol>
        </section></article>
        """,
        source_id="structured-blocks",
    )
    blocks = parsed["document"]["blocks"]

    assert blocks[0]["kind"] == "heading"
    assert blocks[0]["level"] == 3
    assert blocks[0]["tag"] == "2.1"
    assert blocks[0]["title"] == "Assumptions"
    assert blocks[1]["kind"] == "list"
    assert blocks[1]["list_kind"] == "ordered"
    assert blocks[1]["ordered"] is True
    assert blocks[1]["start"] == 3
    assert [item["id"] for item in blocks[1]["items"]] == ["S2.i1", "S2.i2"]
    assert blocks[1]["items"][0]["children"][0]["items"][0]["text"] == "Nested item"
    assert blocks[1]["items"][1]["value"] == "7"
    assert parsed["document"]["integrity"]["status"] == "complete"


def test_integrity_rejects_unrenderable_entities_but_preserves_idless_equation_group_rows():
    parsed = parse_source_input(
        html_text="""
        <article class="ltx_document">
          <figure class="ltx_figure" id="F1"><img src="/html/test/assets/missing.png"/></figure>
          <figure class="ltx_table" id="T1"><figcaption>Empty table</figcaption></figure>
          <table class="ltx_equationgroup" id="EG1"><tbody id="E1">
            <tr class="ltx_equation"><td><math alttext="a"><mi>a</mi></math></td></tr>
            <tr class="ltx_equation"><td><math alttext="b"><mi>b</mi></math></td></tr>
          </tbody></table>
        </article>
        """,
        source_id="unrenderable",
        source_url="https://ar5iv.labs.arxiv.org/html/test",
    )
    integrity = parsed["document"]["integrity"]
    codes = {item["code"] for item in integrity["diagnostics"]}

    assert integrity["status"] == "partial"
    assert integrity["renderable"] is False
    assert {"figure_asset_missing", "table_grid_missing"} <= codes
    assert "equation_group_gap" not in codes
    equations = parsed["document"]["equations"]
    assert [item["id"] for item in equations] == ["EG1.row-1", "EG1.row-2"]
    assert [(item["group_row"], item["group_row_count"]) for item in equations] == [(1, 2), (2, 2)]


def test_ar5iv_assets_are_content_addressed_deduplicated_and_reused(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"same-image", headers={"content-type": "image/png"})

    provider = Ar5ivProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    html = """
    <article class="ltx_document">
      <figure class="ltx_figure"><img src="/html/0911.3380/assets/a.png"/></figure>
      <figure class="ltx_figure"><img src="/html/0911.3380/assets/b.png"/></figure>
    </article>
    """

    first = provider.cache_assets("arXiv:0911.3380", html)
    second = provider.cache_assets("arXiv:0911.3380", html)

    assert len(first) == 2
    assert first[0]["asset_id"] == first[1]["asset_id"]
    assert first[0]["cache_path"] == first[1]["cache_path"]
    assert first[0]["status"] == "cached"
    assert first == second
    assert len(calls) == 2
    assert len(list(CachePaths.for_paper("arXiv:0911.3380").ar5iv_assets.rglob("*.png"))) == 1


def test_parse_and_get_parsed_default_to_light_cache_only(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    html_path = tmp_path / "paper.html"
    html_path.write_text(RICH_HTML, encoding="utf-8")

    parsed = service.parse_source(html_path=html_path, source_id="rich-paper")

    assert parsed["ok"] is True
    assert set(parsed["data"]) == LEGACY_PARSED_SOURCE_KEYS
    cached = read_json(parsed_source_cache_path("rich-paper"))
    assert set(cached) == LEGACY_PARSED_SOURCE_KEYS
    rich_path = rich_document_cache_path(
        "rich-paper", cached["source_hash"], service.RICH_PARSER_VERSION
    )
    assert not rich_path.exists()

    default_lookup = service.get_parsed_source("rich-paper")
    rich_lookup = service.get_parsed_source("rich-paper", include_document=True)

    assert set(default_lookup["data"]) == LEGACY_PARSED_SOURCE_KEYS
    assert rich_lookup["ok"] is False
    assert rich_lookup["error"]["code"] == "parsed_source_document_not_found"


def test_parse_include_document_exposes_cached_document(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    html_path = tmp_path / "paper.html"
    html_path.write_text(RICH_HTML, encoding="utf-8")

    result = service.parse_source(html_path=html_path, source_id="rich-paper", include_document=True)

    assert result["ok"] is True
    assert result["data"]["document"]["schema_version"] == DOCUMENT_SCHEMA_VERSION
    light = read_json(parsed_source_cache_path("rich-paper"))
    assert set(light) == LEGACY_PARSED_SOURCE_KEYS
    rich_path = rich_document_cache_path(
        "rich-paper", light["source_hash"], service.RICH_PARSER_VERSION
    )
    assert read_json(rich_path)["document"] == result["data"]["document"]


def test_recache_upgrades_v12_ar5iv_cache_without_refreshing_source(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    path = parsed_source_cache_path("arXiv:0911.3380")
    write_json(
        path,
        {
            "paper_id": "arXiv:0911.3380",
            "parser_version": 12,
            "source_hash": "old-hash",
            "toc": [],
            "sections": [],
            "equations": [],
        },
    )

    class CachedAr5iv:
        calls: list[bool] = []

        def get_html(self, paper_id, *, refresh=False):
            assert paper_id == "arXiv:0911.3380"
            self.calls.append(refresh)
            return RICH_HTML

    provider = CachedAr5iv()
    monkeypatch.setattr(service, "_ar5iv", provider)

    result = service.parse_source(
        paper_id="0911.3380",
        source="ar5iv",
        recache=True,
        include_document=True,
    )

    assert result["ok"] is True
    assert provider.calls == [False]
    assert result["data"]["parser_version"] == 14
    assert result["data"]["document"]["schema_version"] == DOCUMENT_SCHEMA_VERSION
    assert read_json(path)["parser_version"] == 14


def test_refresh_and_recache_are_mutually_exclusive(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))

    result = service.parse_source(
        paper_id="0911.3380",
        source="ar5iv",
        refresh=True,
        recache=True,
        include_document=True,
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "parse_source_invalid"
    assert "mutually exclusive" in result["error"]["message"].lower()


def test_equation_annotations_are_bound_to_parser_version_and_equation_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    html_path = tmp_path / "paper.html"
    html_path.write_text(RICH_HTML, encoding="utf-8")
    parsed = service.parse_source(html_path=html_path, source_id="rich-paper", include_document=True)
    equation_id = parsed["data"]["equations"][0]["id"]

    marked = service.mark_parsed_equation("rich-paper", equation_id, reason="Verify the sign.")

    assert marked["ok"] is True
    assert marked["data"]["parser_version"] == 14
    assert marked["data"]["equation_fingerprint"]
    assert service.get_parsed_source_equation("rich-paper", equation_id)["data"]["annotations"] == [marked["data"]]

    cached = read_json(parsed_source_cache_path("rich-paper"))
    cached["equations"][0]["equation"] = "x=-y"
    write_json(parsed_source_cache_path("rich-paper"), cached)

    changed = service.get_parsed_source_equation("rich-paper", equation_id)
    assert "annotations" not in changed["data"]


@pytest.mark.parametrize("flag", ["--include-document", "--recache"])
def test_cli_parse_forwards_rich_document_flags(monkeypatch, capsys, flag):
    captured = {}

    def parse_source(source_path=None, **kwargs):
        captured.update(kwargs)
        return {"ok": True, "data": {"paper_id": kwargs["paper_id"]}, "meta": {}}

    monkeypatch.setattr(cli.service, "parse_source", parse_source)

    argv = ["parse", "--paper-id", "0911.3380", "--source", "ar5iv", flag, "--json"]
    assert cli.main(argv) == 0
    json.loads(capsys.readouterr().out)

    assert captured[flag.removeprefix("--").replace("-", "_")] is True


def test_cli_get_parsed_forwards_include_document(monkeypatch, capsys):
    captured = {}

    def get_parsed_source(source_id, *, include_document=False):
        captured.update(source_id=source_id, include_document=include_document)
        return {"ok": True, "data": {"paper_id": source_id}, "meta": {}}

    monkeypatch.setattr(cli.service, "get_parsed_source", get_parsed_source)

    assert cli.main(["get-parsed", "0911.3380", "--include-document", "--json"]) == 0
    json.loads(capsys.readouterr().out)

    assert captured == {"source_id": "0911.3380", "include_document": True}


def test_doi_alias_reads_canonical_rich_parsed_cache_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    canonical = "arXiv:0911.3380"
    doi = "doi:10.1000/example"
    parsed = parse_source_input(html_text=RICH_HTML, source_id=canonical)
    write_json(parsed_source_cache_path(canonical), parsed)
    write_paper_alias(doi, canonical)

    class NoNetwork:
        def get_html(self, *args, **kwargs):
            raise AssertionError("alias cache lookup must not fetch HTML")

    monkeypatch.setattr(service, "_ar5iv", NoNetwork())

    result = service.get_parsed_source(doi, include_document=True)

    assert result["ok"] is True
    assert result["data"]["paper_id"] == canonical
    assert result["data"]["document"]["schema_version"] == DOCUMENT_SCHEMA_VERSION


def test_parsed_cache_lookup_follows_alias_chain_to_canonical_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    canonical = "arXiv:0911.3380"
    inspire_id = "inspire:12345"
    doi = "doi:10.1000/example"
    write_json(parsed_source_cache_path(canonical), parse_source_input(html_text=RICH_HTML, source_id=canonical))
    write_paper_alias(doi, inspire_id)
    write_paper_alias(inspire_id, canonical)

    result = service.get_parsed_source(doi, include_document=True)

    assert result["ok"] is True
    assert result["data"]["paper_id"] == canonical
    assert result["meta"]["cache_path"] == str(parsed_source_cache_path(canonical))


def test_default_ar5iv_parse_does_not_build_document_or_cache_assets(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))

    class RecordingAr5iv:
        asset_calls = 0

        def get_html(self, paper_id, *, refresh=False):
            return RICH_HTML

        def cache_assets(self, paper_id, html, *, refresh=False):
            self.asset_calls += 1
            return []

    provider = RecordingAr5iv()
    monkeypatch.setattr(service, "_ar5iv", provider)

    result = service.parse_source(source="ar5iv", paper_id="0911.3380")

    assert result["ok"] is True
    assert provider.asset_calls == 0
    light = read_json(parsed_source_cache_path("arXiv:0911.3380"))
    assert set(light) == LEGACY_PARSED_SOURCE_KEYS
    assert not rich_document_cache_path(
        "arXiv:0911.3380", light["source_hash"], service.RICH_PARSER_VERSION
    ).exists()


def test_explicit_rich_request_is_reused_and_version_bump_leaves_light_untouched(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))

    class RecordingAr5iv:
        html_calls = 0
        asset_calls = 0

        def get_html(self, paper_id, *, refresh=False):
            self.html_calls += 1
            return RICH_HTML

        def cache_assets(self, paper_id, html, *, refresh=False):
            self.asset_calls += 1
            return []

    provider = RecordingAr5iv()
    monkeypatch.setattr(service, "_ar5iv", provider)
    first = service.parse_source(source="ar5iv", paper_id="0911.3380", include_document=True)
    second = service.parse_source(source="ar5iv", paper_id="0911.3380", include_document=True)
    light_path = parsed_source_cache_path("arXiv:0911.3380")
    before = light_path.read_bytes()
    monkeypatch.setattr(service, "RICH_PARSER_VERSION", service.RICH_PARSER_VERSION + 1)
    light_only = service.parse_source(source="ar5iv", paper_id="0911.3380")
    rebuilt = service.parse_source(source="ar5iv", paper_id="0911.3380", include_document=True)

    assert all(item["ok"] for item in (first, second, light_only, rebuilt))
    assert provider.asset_calls == 2
    assert provider.html_calls == 2
    assert light_path.read_bytes() == before


def test_concurrent_explicit_rich_requests_build_once(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))

    class RecordingAr5iv:
        asset_calls = 0
        guard = threading.Lock()

        def get_html(self, paper_id, *, refresh=False):
            return RICH_HTML

        def cache_assets(self, paper_id, html, *, refresh=False):
            with self.guard:
                self.asset_calls += 1
            time.sleep(0.02)
            return []

    provider = RecordingAr5iv()
    monkeypatch.setattr(service, "_ar5iv", provider)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _: service.parse_source(
                source="ar5iv", paper_id="0911.3380", include_document=True
            ),
            range(8),
        ))

    assert all(item["ok"] for item in results)
    assert provider.asset_calls == 1
    assert len({item["data"]["document"]["document_hash"] for item in results}) == 1
