from __future__ import annotations

from arc_llm.usage import LLMUsage


def test_claude_cache_ratio_counts_uncached_input_tokens():
    usage = LLMUsage(input_tokens=11, cache_creation_input_tokens=1, cache_read_input_tokens=8)

    assert usage.total_input_tokens == 20
    assert usage.effective_cached_input_tokens == 8
    assert usage.cached_input_ratio == 0.4
    assert usage.to_json()["total_input_tokens"] == 20
    assert usage.to_json()["effective_cached_input_tokens"] == 8


def test_codex_cache_ratio_uses_cached_input_tokens():
    usage = LLMUsage(input_tokens=100, cached_input_tokens=70)

    assert usage.total_input_tokens == 100
    assert usage.effective_cached_input_tokens == 70
    assert usage.cached_input_ratio == 0.7
