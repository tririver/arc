import os

import pytest

from arc_paper_query.providers.ar5iv import Ar5ivProvider
from arc_paper_query.providers.inspire import InspireProvider


pytestmark = pytest.mark.skipif(
    os.environ.get("ARC_RUN_NET_TESTS") != "1",
    reason="set ARC_RUN_NET_TESTS=1 to run network integration tests",
)


def test_inspire_and_ar5iv_modern_arxiv_id():
    inspire = InspireProvider()
    ar5iv = Ar5ivProvider()

    metadata = inspire.get_metadata("arXiv:0911.3380", refresh=True)
    html = ar5iv.get_html("arXiv:0911.3380", refresh=True)

    assert metadata["title"]
    assert "<html" in html.lower()


def test_inspire_and_ar5iv_old_arxiv_id():
    inspire = InspireProvider()
    ar5iv = Ar5ivProvider()

    metadata = inspire.get_metadata("arXiv:hep-th/0601001", refresh=True)
    html = ar5iv.get_html("arXiv:hep-th/0601001", refresh=True)

    assert metadata["title"]
    assert "<html" in html.lower()
