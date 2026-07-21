from __future__ import annotations

import pytest

from arc_companion.pipeline import (
    BuildOptions,
    _annotation_source_urls,
    _generate_annotations,
    _validate_direct_annotation_sources,
)
from arc_companion.source import SourceBundle


def _source(url: str = "https://example.org/paper") -> dict[str, str]:
    return {"title": "Primary paper", "url": url, "locator": "Section 3"}


def _annotation() -> dict[str, object]:
    return {
        "explanation": "Explanation.",
        "commentary": "Commentary.",
        "commentary_sources": [_source()],
        "prior_work": [{"text": "Prior result.", "sources": [_source("https://example.org/prior")]}],
        "later_work": [],
    }


def test_direct_sources_are_preserved_without_registration() -> None:
    value = _annotation()
    assert _validate_direct_annotation_sources(value) == value
    assert _annotation_source_urls(value) == {
        "https://example.org/paper", "https://example.org/prior",
    }


@pytest.mark.parametrize(
    "source",
    [
        {"url": "https://example.org", "locator": "Abstract"},
        {"title": "Paper", "locator": "Abstract"},
        {"title": "Paper", "url": "https://example.org"},
        {"title": "Paper", "url": "ftp://example.org", "locator": "Abstract"},
        {"title": "Paper", "url": "not a url", "locator": "Abstract"},
    ],
)
def test_direct_source_requires_title_http_url_and_locator(source: dict[str, str]) -> None:
    value = _annotation()
    value["commentary_sources"] = [source]
    with pytest.raises(RuntimeError):
        _validate_direct_annotation_sources(value)


def test_duplicate_source_and_more_than_three_are_rejected() -> None:
    value = _annotation()
    value["commentary_sources"] = [_source(), _source()]
    with pytest.raises(RuntimeError, match="duplicate"):
        _validate_direct_annotation_sources(value)
    value["commentary_sources"] = [
        _source(f"https://example.org/{index}") for index in range(4)
    ]
    with pytest.raises(RuntimeError, match="at most three"):
        _validate_direct_annotation_sources(value)


def test_offline_mode_accepts_only_bounded_source_urls() -> None:
    value = _annotation()
    allowed = {"https://example.org/paper", "https://example.org/prior"}
    assert _validate_direct_annotation_sources(value, allowed_urls=allowed) == value
    value["later_work"] = [{"text": "Unsupported.", "sources": [_source("https://new.example/x")]}]
    with pytest.raises(RuntimeError, match="offline annotation"):
        _validate_direct_annotation_sources(value, allowed_urls=allowed)


def test_removed_controller_fields_are_not_persisted_from_empty_legacy_shape() -> None:
    legacy = {
        "explanation": "", "commentary": "", "prior_work": [], "later_work": [],
        "key_points": [], "source_notes": [], "evidence_ids": [],
        "context_claims": [], "evidence_requests": [],
    }
    normalized = _validate_direct_annotation_sources(legacy)
    assert set(normalized) == {
        "explanation", "commentary", "commentary_sources", "prior_work", "later_work",
    }


def test_annotation_accepts_a_new_direct_source_in_one_call(tmp_path) -> None:
    document = {
        "front_matter": {},
        "blocks": [{"block_id": "b1", "type": "text", "text": "A physical claim."}],
        "equations": [], "figures": [], "tables": [], "assets": [], "bibliography": [],
    }
    bundle = SourceBundle(
        paper_id="local:direct-source", parsed={"document": document}, document=document,
        metadata={"title": "Source"}, references=[], citers=[],
    )
    calls = []

    def llm(_prompt: str, **kwargs):
        calls.append(kwargs["call_label"])
        return _annotation()

    result = _generate_annotations(
        [{"segment_id": "seg-1", "block_ids": ["b1"]}],
        options=BuildOptions(
            paper_id=bundle.paper_id, project_dir=tmp_path, workers=1, allow_internet=True,
        ),
        bundle=bundle, evidence={"related_papers": [], "references": [], "citers": []},
        domain_context=None, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=tmp_path / "checkpoint", llm=llm,
    )
    assert calls == ["companion-annotation-seg-1"]
    assert result["seg-1"] == _annotation()
    assert not list((tmp_path / "checkpoint").glob("evidence-resolution*"))
