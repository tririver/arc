import json
from pathlib import Path

from arc_paper_query.cache import CachePaths, cache_root, read_json, read_text, write_json, write_text


def test_cache_root_prefers_arc_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path / "cache"))
    assert cache_root() == tmp_path / "cache"


def test_cache_root_uses_xdg_when_arc_env_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC_PAPER_QUERY_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "xdg" / "arc" / "paper-query"


def test_cache_root_uses_project_cache_in_arc_checkout(monkeypatch):
    monkeypatch.delenv("ARC_PAPER_QUERY_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    repo_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "packages" / "arc-paper-query").is_dir()
    )
    assert cache_root() == repo_root / "cache" / "paper-query"


def test_cache_paths_quote_paper_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_QUERY_CACHE", str(tmp_path))
    paths = CachePaths.for_paper("arXiv:hep-th/0601001")
    assert paths.paper_dir == tmp_path / "papers" / "arXiv%3Ahep-th%2F0601001"
    assert paths.ar5iv_html == paths.paper_dir / "ar5iv" / "fulltext.html"
    assert paths.inspire_citers == paths.paper_dir / "inspire" / "citers.json"


def test_json_and_text_roundtrip(tmp_path):
    json_path = tmp_path / "data" / "item.json"
    text_path = tmp_path / "data" / "item.txt"

    write_json(json_path, {"a": 1})
    write_text(text_path, "hello")

    assert json.loads(json_path.read_text()) == {"a": 1}
    assert read_json(json_path) == {"a": 1}
    assert read_text(text_path) == "hello"
