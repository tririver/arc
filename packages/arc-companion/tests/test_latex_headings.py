from pathlib import Path

from arc_companion.latex import render_companion_tex as _render_companion_tex
from arc_companion.source_credit import normalize_source_credit


def render_companion_tex(document, *args, source_credit=None, metadata=None, **kwargs):
    return _render_companion_tex(
        document,
        *args,
        source_credit=source_credit or normalize_source_credit(document, metadata),
        metadata=metadata,
        **kwargs,
    )


def test_source_heading_keeps_original_number_without_latex_renumbering(tmp_path: Path) -> None:
    document = {
        "blocks": [
            {
                "block_id": "heading-1",
                "section_id": "S1",
                "kind": "heading",
                "level": 2,
                "title": "Introduction and Summary",
                "text": "1 Introduction and Summary",
                "html": (
                    '<h2 class="ltx_title ltx_title_section">'
                    '<span class="ltx_tag ltx_tag_section">1 </span>'
                    "Introduction and Summary</h2>"
                ),
            }
        ],
        "integrity": {"status": "complete"},
    }
    segments = [
        {
            "segment_id": "segment-1",
            "start_block_id": "heading-1",
            "end_block_id": "heading-1",
            "block_ids": ["heading-1"],
        }
    ]

    tex, _ = render_companion_tex(
        document,
        segments,
        {"segment-1": {"commentary": "Commentary."}},
        output_dir=tmp_path,
        language="en",
    )

    assert r"\subsection*{1 Introduction and Summary}" in tex
    assert r"\subsection{1 Introduction and Summary}" not in tex
    assert r"\addcontentsline{toc}{subsection}{1 Introduction and Summary}" in tex
    assert r"\label{heading-1}" in tex
    assert r"\label{S1}" in tex
