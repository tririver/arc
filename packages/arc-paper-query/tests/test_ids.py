from arc_paper_query.ids import arxiv_path_id, normalize_paper_id


def test_normalize_new_arxiv_id():
    assert normalize_paper_id("0911.3380") == "arXiv:0911.3380"
    assert normalize_paper_id("arxiv:0911.3380") == "arXiv:0911.3380"
    assert normalize_paper_id("arXiv:2512.06790v2") == "arXiv:2512.06790"


def test_normalize_old_arxiv_id():
    assert normalize_paper_id("hep-th/0601001") == "arXiv:hep-th/0601001"
    assert arxiv_path_id("arXiv:hep-th/0601001") == "hep-th/0601001"


def test_arxiv_path_id_rejects_non_arxiv():
    assert arxiv_path_id("doi:10.1000/example") == ""
