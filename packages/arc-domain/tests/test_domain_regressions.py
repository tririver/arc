from __future__ import annotations

from datetime import datetime, timezone

from arc_domain import foundation
from arc_domain import network
from arc_domain.text import paper_key


def _paper(paper_id: str, *, title: str | None = None, citations: int = 1, **extra) -> dict:
    return {
        "paper_id": paper_id,
        "title": title or paper_id,
        "citation_count": citations,
        "identifiers": {"paper_id": paper_id},
        **extra,
    }


def test_merged_citers_interleaves_recent_and_cited_without_starving_cited(monkeypatch):
    foundation_id = "arXiv:2301.00001"
    recent = [
        _paper("arXiv:2401.00001", title="Recent A"),
        _paper("arXiv:2401.00002", title="Recent B"),
        _paper("arXiv:2401.00003", title="Recent C"),
    ]
    mostcited = [
        _paper("arXiv:2401.00002", title="Recent B", citations=200),
        _paper("arXiv:2001.00001", title="Cited D", citations=500),
        _paper("arXiv:2001.00002", title="Cited E", citations=400),
    ]

    def fake_citers(paper_id, *, refresh=False, limit=1000, sort="mostrecent"):
        assert paper_id == foundation_id
        return {"mostrecent": recent, "mostcited": mostcited}[sort]

    monkeypatch.setattr(network.paper, "citers", fake_citers)

    merged = network._merged_citers(foundation_id, refresh=False, limit=3)

    assert [item["paper_id"] for item in merged] == [
        "arXiv:2401.00001",
        "arXiv:2401.00002",
        "arXiv:2001.00001",
    ]
    duplicate = merged[1]
    assert duplicate["citer_sources"] == ["mostrecent", "mostcited"]
    assert duplicate["mostrecent_rank"] == 2
    assert duplicate["mostcited_rank"] == 1


def test_title_only_reference_has_no_stable_paper_key():
    assert paper_key({"title": "A title is not a stable identifier"}) == ""
    assert paper_key({"identifiers": {"arxiv": "2201.00001"}, "title": "Stable"}) == "arXiv:2201.00001"
    assert paper_key({"arxiv_id": "2201.00002", "title": "Stable"}) == "arXiv:2201.00002"
    assert paper_key({"identifiers": {"inspire_recid": "12345"}, "title": "Stable"}) == "inspire:12345"


def test_common_references_skip_title_only_entries_before_metadata_lookup(monkeypatch):
    requested_ids: list[str] = []

    def fake_fetch_many(ids, func, *, workers=8):
        requested_ids.extend(ids)
        return {
            "arXiv:2201.00001": {
                "paper_id": "arXiv:2201.00001",
                "title": "Stable Shared Reference",
                "citation_count": 42,
            }
        }

    monkeypatch.setattr(network.paper, "fetch_many", fake_fetch_many)

    refs_by_selected = {
        "arXiv:2401.00001": [
            {"title": "Title Only Shared Reference"},
            {"paper_id": "arXiv:2201.00001", "title": "Stable Shared Reference"},
        ],
        "arXiv:2401.00002": [
            {"title": "Title Only Shared Reference"},
            {"identifiers": {"arxiv": "2201.00001"}, "title": "Stable Shared Reference"},
        ],
    }

    common = network._common_references(
        foundation_id="arXiv:2301.00001",
        selected_ids=["arXiv:2401.00001", "arXiv:2401.00002"],
        refs_by_selected=refs_by_selected,
        max_extra=10,
        refresh=False,
        workers=1,
    )

    assert requested_ids == ["arXiv:2201.00001"]
    assert [item["paper_id"] for item in common] == ["arXiv:2201.00001"]


def test_foundation_citation_support_threshold_is_configurable():
    prompt = foundation._foundation_prompt(
        seed_metadata={"paper_id": "arXiv:2401.00001", "title": "Seed Paper"},
        candidates=[],
        intent="exact topic",
        min_citation_count=25,
    )
    selection = foundation._deterministic_selection(
        [
            {
                "paper_id": "arXiv:2401.00001",
                "title": "Young Exact Paper",
                "citation_count": 41,
                "witness_citation_overlap": 5,
                "intent_overlap": 1.0,
            },
            {
                "paper_id": "arXiv:2301.00001",
                "title": "Established Foundation",
                "citation_count": 150,
                "witness_citation_overlap": 5,
                "intent_overlap": 0.5,
            },
        ],
        intent="exact topic",
        min_citation_count=25,
    )

    assert "fewer than 25 citations" in prompt
    assert selection["selected_foundation"]["paper_id"] == "arXiv:2401.00001"


def test_domain_selection_keeps_all_recent_arxiv_papers_beyond_selected_count():
    now = datetime.now(timezone.utc)
    recent_id = f"arXiv:{now.year % 100:02d}{now.month:02d}.00001"
    selected = network._select_domain_papers(
        [
            _paper(
                "arXiv:2001.00001",
                title="Highly cited older paper",
                citations=10000,
                year=2020,
                published="2020-01-01",
            ),
            _paper(
                recent_id,
                title="Low citation recent paper",
                citations=0,
                year=now.year,
                published=now.date().isoformat(),
                arxiv_id=recent_id.removeprefix("arXiv:"),
            ),
        ],
        foundation_id="arXiv:1901.00001",
        intent_ranking={"ranked_paper_ids": []},
        intent="",
        selected_count=1,
    )

    assert [item["paper_id"] for item in selected] == ["arXiv:2001.00001", recent_id]
    recent = selected[1]
    assert recent["recent_arxiv"] is True
    assert "recent arXiv" in recent["selection_reason"]
