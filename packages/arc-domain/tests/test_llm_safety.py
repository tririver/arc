from __future__ import annotations

import pytest

from arc_domain.llm_safety import raise_if_llm_fatal
from arc_llm.providers.base import LLMAbortScope, LLMFailureCategory, LLMWorkerError


def test_domain_fallback_propagates_provider_fatal_through_wrapper() -> None:
    provider_error = LLMWorkerError(
        "quota exhausted",
        category=LLMFailureCategory.QUOTA,
        abort_scope=LLMAbortScope.PROVIDER,
    )
    try:
        raise RuntimeError("domain wrapper") from provider_error
    except RuntimeError as wrapped:
        with pytest.raises(RuntimeError, match="domain wrapper"):
            raise_if_llm_fatal(wrapped)


def test_domain_fallback_may_handle_local_call_failure() -> None:
    raise_if_llm_fatal(RuntimeError("local normalization failed"))
