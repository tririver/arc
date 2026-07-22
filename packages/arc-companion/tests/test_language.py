from __future__ import annotations

import pytest

from arc_companion.language import (
    base_language,
    cjk_font_region,
    contains_lexical_term,
    is_same_base_language,
    language_direction,
    normalize_language_tag,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ZH_hant_tw", "zh-Hant-TW"),
        ("sr_cyrl_rs", "sr-Cyrl-RS"),
        ("iw-IL", "he-IL"),
        ("en-US-u-NU-latn", "en-US-u-nu-latn"),
        ("x-ARC-Source", "x-arc-source"),
        (None, "und"),
        ("not valid!", "und"),
    ],
)
def test_normalize_language_tag_is_stable_and_conservative(raw, expected) -> None:
    assert normalize_language_tag(raw) == expected


def test_language_metadata_classifies_direction_and_base_language() -> None:
    assert base_language("FA_ir") == "fa"
    assert language_direction("ar-EG") == "rtl"
    assert language_direction("he") == "rtl"
    assert language_direction("mul") == "auto"
    assert language_direction("und") == "auto"
    assert language_direction("ru") == "ltr"
    assert is_same_base_language("zh-CN", "zh-Hant-TW") is True
    assert is_same_base_language("und", "und") is False


@pytest.mark.parametrize(
    ("language", "region"),
    [
        ("zh-CN", "SC"),
        ("zh-Hans", "SC"),
        ("zh-TW", "TC"),
        ("zh-Hant-HK", "TC"),
        ("yue-HK", "TC"),
        ("ja", "JP"),
        ("ko-KR", "KR"),
        ("ru", None),
    ],
)
def test_cjk_font_region_routes_common_language_tags(language, region) -> None:
    assert cjk_font_region(language) == region


def test_unicode_lexical_matching_uses_non_latin_boundaries_and_cjk_runs() -> None:
    assert contains_lexical_term("Теория Ландау работает.", "ландау")
    assert not contains_lexical_term("сЛандау", "Ландау")
    assert contains_lexical_term("Η θεωρία Λαντάου.", "Λαντάου")
    assert not contains_lexical_term("αλαντάου", "λαντάου")
    assert contains_lexical_term("量子场论", "量子")
    assert contains_lexical_term("ゲージ場の理論", "ゲージ場")
    assert contains_lexical_term("ＣＡＦÉ", "café")
    assert not contains_lexical_term("Feynmanian", "Feynman")


def test_case_sensitive_lexical_matching_supports_exact_source_terms() -> None:
    assert contains_lexical_term("identité de Ward", "identité de Ward", case_sensitive=True)
    assert not contains_lexical_term("Identité de Ward", "identité de Ward", case_sensitive=True)
