from __future__ import annotations

import httpx

from ..ids import arxiv_path_id
from .base import ProviderError


def ar5iv_url(paper_id: str) -> str:
    aid = arxiv_path_id(paper_id)
    if not aid:
        raise ProviderError("not_arxiv_id", f"ar5iv requires an arXiv ID: {paper_id}")
    return f"https://ar5iv.labs.arxiv.org/html/{aid}"


class Ar5ivProvider:
    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 60.0):
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.timeout = timeout

    def get_html(self, paper_id: str, *, refresh: bool = False) -> str:
        response = self.client.get(ar5iv_url(paper_id), timeout=self.timeout)
        if response.status_code == 404:
            raise ProviderError("ar5iv_not_found", f"ar5iv HTML not found for {paper_id}")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError("ar5iv_fetch_failed", str(exc)) from exc

        return response.text
