from __future__ import annotations

from arc_llm.providers.base import LLMAbortScope, failure_disposition


def raise_if_llm_fatal(exc: BaseException) -> None:
    """Keep scientific fallbacks from swallowing batch/provider failures."""

    disposition = failure_disposition(exc)
    if disposition is not None and disposition.abort_scope in {
        LLMAbortScope.BATCH,
        LLMAbortScope.PROVIDER,
    }:
        raise exc
