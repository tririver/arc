from arc_paper.ids import (
    arxiv_path_id,
    doi_value,
    extract_paper_ids,
    inspire_recid,
    normalize_paper_id,
    paper_ids_safe_dir_name,
)


def test_normalize_new_arxiv_id():
    assert normalize_paper_id("0911.3380") == "arXiv:0911.3380"
    assert normalize_paper_id("arxiv:0911.3380") == "arXiv:0911.3380"
    assert normalize_paper_id("arXiv:2512.06790v2") == "arXiv:2512.06790"


def test_normalize_old_arxiv_id():
    assert normalize_paper_id("hep-th/0601001") == "arXiv:hep-th/0601001"
    assert normalize_paper_id("HEP-TH/0601001") == "arXiv:hep-th/0601001"
    assert normalize_paper_id("arXiv:HEP-TH/0601001") == "arXiv:hep-th/0601001"
    assert normalize_paper_id("HTTPS://ARXIV.ORG/ABS/HEP-TH/0601001") == "arXiv:hep-th/0601001"
    assert arxiv_path_id("arXiv:hep-th/0601001") == "hep-th/0601001"


def test_arxiv_path_id_rejects_non_arxiv():
    assert arxiv_path_id("doi:10.1000/example") == ""


def test_arxiv_path_id_rejects_invalid_arxiv_like_ids():
    assert arxiv_path_id("arXiv:not-a-paper") == ""
    assert arxiv_path_id("https://arxiv.org/abs/foo/1234567") == ""
    assert arxiv_path_id("1234.5678") == ""
    assert arxiv_path_id("arXiv:2513.00001") == ""
    assert normalize_paper_id("https://arxiv.org/abs/foo/1234567") == "https://arxiv.org/abs/foo/1234567"
    assert normalize_paper_id("1234.5678") == "1234.5678"


def test_normalize_inspire_recid():
    assert normalize_paper_id("recid:154280") == "inspire:154280"
    assert normalize_paper_id("inspire:154280") == "inspire:154280"
    assert inspire_recid("recid:154280") == "154280"


def test_normalize_doi():
    assert normalize_paper_id("doi:10.1088/1475-7516/2010/04/027") == "doi:10.1088/1475-7516/2010/04/027"
    assert normalize_paper_id("https://doi.org/10.1007/JHEP01(2010)117.") == "doi:10.1007/jhep01(2010)117"
    assert doi_value("doi:10.1088/1475-7516/2010/04/027") == "10.1088/1475-7516/2010/04/027"


def test_extract_paper_ids_from_natural_language():
    text = (
        "Use arXiv:0911.3380v2, inspire:837197, and "
        "https://doi.org/10.1088/1475-7516/2010/04/027. "
        "Also compare 2512.06790. Also compare astro-ph/0610514. "
        "Do not extract the arXiv-like suffix inside doi:10.1234/2512.06790."
    )

    assert extract_paper_ids(text) == [
        "arXiv:0911.3380",
        "inspire:837197",
        "doi:10.1088/1475-7516/2010/04/027",
        "arXiv:2512.06790",
        "arXiv:astro-ph/0610514",
        "doi:10.1234/2512.06790",
    ]


def test_extract_paper_ids_deduplicates_and_accepts_urls():
    text = (
        "https://arxiv.org/abs/hep-th/0601001v3, arXiv:hep-th/0601001, "
        "https://inspirehep.net/literature/12345 and recid:12345"
    )

    assert extract_paper_ids(text) == ["arXiv:hep-th/0601001", "inspire:12345"]


def test_extract_paper_ids_rejects_invalid_new_style_months():
    assert extract_paper_ids("Do not treat 1234.5678 or arXiv:2513.00001 as papers.") == []


def test_paper_ids_safe_dir_name():
    assert paper_ids_safe_dir_name(["arXiv:0911.3380"]) == "0911.3380"
    assert (
        paper_ids_safe_dir_name(["arXiv:0911.3380", "astro-ph/0610514"])
        == "0911.3380_x_astro-ph_0610514"
    )
    assert paper_ids_safe_dir_name(["inspire:837197", "doi:10.1007/JHEP01(2010)117"]) == (
        "inspire_837197_x_doi_10.1007_jhep01_2010_117"
    )
    assert paper_ids_safe_dir_name(["doi:10.1088/1475-7516/2010/04/027"]) == (
        "doi_10.1088_1475-7516_2010_04_027"
    )
