import json

from arc_paper.cache import (
    CachePaths,
    cache_root,
    migrate_paper_cache_dir,
    parsed_source_annotations_cache_path,
    parsed_source_cache_path,
    read_json,
    read_text,
    write_json,
    write_text,
)


def test_cache_root_prefers_package_env_over_arc_home_and_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ARC_HOME", str(tmp_path / "arc-home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "cache"


def test_cache_root_uses_arc_home_before_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC_PAPER_CACHE", raising=False)
    monkeypatch.setenv("ARC_HOME", str(tmp_path / "arc-home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert cache_root() == tmp_path / "arc-home" / "cache" / "arc-paper"


def test_cache_root_uses_xdg_when_arc_envs_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC_PAPER_CACHE", raising=False)
    monkeypatch.delenv("ARC_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "xdg" / "arc" / "arc-paper"


def test_cache_root_never_implicitly_uses_source_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC_PAPER_CACHE", raising=False)
    monkeypatch.delenv("ARC_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))

    assert cache_root() == tmp_path / "isolated-home" / ".cache" / "arc" / "arc-paper"


def test_cache_paths_quote_paper_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    paths = CachePaths.for_paper("arXiv:hep-th/0601001")
    assert paths.paper_dir == tmp_path / "papers" / "arXiv%3Ahep-th%2F0601001"
    assert paths.ar5iv_html == paths.paper_dir / "ar5iv" / "fulltext.html"
    assert paths.inspire_citers == paths.paper_dir / "inspire" / "citers.json"


def test_parsed_source_cache_path_uses_safe_paper_id_file_name(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))

    assert parsed_source_cache_path("arXiv:0911.3380") == tmp_path / "sources" / "0911.3380.json"
    assert parsed_source_cache_path("doi:10.1088/1475-7516/2010/04/027") == (
        tmp_path / "sources" / "doi_10.1088_1475-7516_2010_04_027.json"
    )


def test_parsed_source_cache_path_accepts_local_arc_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))

    assert parsed_source_cache_path("arc-12345678") == tmp_path / "sources" / "arc-12345678.json"
    assert parsed_source_cache_path("lecture 9") == tmp_path / "sources" / "lecture_9.json"


def test_parsed_source_annotations_cache_path_uses_sidecar_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))

    assert parsed_source_annotations_cache_path("lecture 9") == tmp_path / "source-annotations" / "lecture_9.json"


def test_migrate_paper_cache_dir_drops_legacy_parsed_json(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    doi_dir = CachePaths.for_paper("doi:10.1088/1475-7516/2010/04/027").paper_dir
    arxiv_dir = CachePaths.for_paper("arXiv:0911.3380").paper_dir
    write_json(doi_dir / "ar5iv" / "parsed.json", {"paper_id": "doi:10.1088/1475-7516/2010/04/027"})
    write_json(doi_dir / "inspire" / "metadata.json", {"title": "cached"})

    migrate_paper_cache_dir("doi:10.1088/1475-7516/2010/04/027", "arXiv:0911.3380")

    assert read_json(arxiv_dir / "inspire" / "metadata.json") == {"title": "cached"}
    assert not (arxiv_dir / "ar5iv" / "parsed.json").exists()


def test_json_and_text_roundtrip(tmp_path):
    json_path = tmp_path / "data" / "item.json"
    text_path = tmp_path / "data" / "item.txt"

    write_json(json_path, {"a": 1})
    write_text(text_path, "hello")

    assert json.loads(json_path.read_text()) == {"a": 1}
    assert read_json(json_path) == {"a": 1}
    assert read_text(text_path) == "hello"
