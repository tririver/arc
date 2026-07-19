from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

from arc_companion import pipeline as pipeline_module
from arc_companion.cli import _emit
from arc_companion.evidence import arc_cache_descriptor, text_sha256, validate_evidence_record
from arc_companion.evidence_requests import EvidenceResolution
from arc_companion.pipeline import (
    BuildOptions,
    CompanionLaneError,
    _evidence,
    _fingerprint,
    _evidence_for_segment,
    _first_wave_preview_document,
    _full_paper_context,
    _generation_document,
    _generate_translations,
    _translation_input_block,
    _validate_translation,
    _protected_names,
    _segment_checkpoint_name,
    _state,
    build_companion,
    validate_and_expand_segments,
)
from arc_companion.source import SourceBundle


def _inline_run(kind: str, content: str, order: int, *, tex: str = "") -> dict[str, str | int]:
    digest = hashlib.sha256(f"{kind}:{content}".encode()).hexdigest()
    value: dict[str, str | int] = {
        "kind": kind,
        "content": content,
        "order": order,
        "token_id": f"b.token-{order:04d}-{digest[:12]}",
        "content_hash": digest,
    }
    if tex:
        value["tex"] = tex
    return value


def test_translation_uses_and_validates_ordered_opaque_inline_tokens() -> None:
    block = {
        "block_id": "b",
        "kind": "prose",
        "text": r"The t_NL and f_NL squared terms.",
        "inline_runs": [
            _inline_run("text", "The ", 1),
            _inline_run("math", r"t_{NL}", 2, tex=r"t_{NL}"),
            _inline_run("text", " and ", 3),
            _inline_run("math", r"f_{NL}^{2}", 4, tex=r"f_{NL}^{2}"),
            _inline_run("text", " squared terms.", 5),
        ],
    }
    projected = _translation_input_block(block)
    tokens = pipeline_module._OPAQUE_INLINE_PATTERN.findall(projected["text"])
    segment = {"segment_id": "s", "block_ids": ["b"]}

    assert len(tokens) == 2
    _validate_translation(segment, {"blocks": [{"block_id": "b", "text": f"译文 {tokens[0]} 与 {tokens[1]}。"}]}, {"b": block}, [])
    try:
        _validate_translation(segment, {"blocks": [{"block_id": "b", "text": f"译文 {tokens[1]} 与 {tokens[0]}。"}]}, {"b": block}, [])
    except RuntimeError as exc:
        assert "opaque inline tokens" in str(exc)
    else:
        raise AssertionError("reordered math tokens must be rejected")


def test_translation_rejects_an_extra_opaque_link_occurrence() -> None:
    link_run = _inline_run("link", "project page", 2)
    link_run["href"] = "https://example.test/project"
    block = {
        "block_id": "linked",
        "kind": "prose",
        "text": "See project page.",
        "inline_runs": [
            _inline_run("text", "See ", 1),
            link_run,
            _inline_run("text", ".", 3),
        ],
    }
    token = pipeline_module._opaque_inline_tokens(block)[0]
    segment = {"segment_id": "seg-link", "block_ids": ["linked"]}

    _validate_translation(
        segment, {"blocks": [{"block_id": "linked", "text": f"参见 {token}。"}]},
        {"linked": block}, [],
    )
    try:
        _validate_translation(
            segment,
            {"blocks": [{"block_id": "linked", "text": f"参见 {token} 和 {token}。"}]},
            {"linked": block},
            [],
        )
    except RuntimeError as exc:
        assert "opaque inline tokens" in str(exc)
    else:
        raise AssertionError("a duplicated rendered link token must be rejected")


def test_translation_opaque_token_retry_is_bounded_and_checkpoints_successes(tmp_path: Path) -> None:
    blocks = []
    segments = []
    required_tokens: dict[str, str] = {}
    for number in range(1, 5):
        block_id_value = f"b{number}"
        math_run = _inline_run("math", f"x_{number}", 2, tex=f"x_{number}")
        block = {
            "block_id": block_id_value,
            "type": "text",
            "text": f"Value x_{number}.",
            "inline_runs": [
                _inline_run("text", "Value ", 1),
                math_run,
                _inline_run("text", ".", 3),
            ],
        }
        blocks.append(block)
        segment_id = f"seg-{number:04d}"
        segments.append({"segment_id": segment_id, "block_ids": [block_id_value]})
        required_tokens[segment_id] = pipeline_module._opaque_inline_tokens(block)[0]

    document = {
        "schema_version": "arc.paper.document.v1",
        "front_matter": {"title": "Retry fixture", "authors": [], "affiliations": []},
        "blocks": blocks,
        "equations": [], "figures": [], "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete", "document_hash": "retry-fixture"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678",
        parsed={"paper_id": "arXiv:1234.5678", "document": document},
        document=document,
        metadata={"title": "Retry fixture"}, references=[], citers=[],
    )
    calls: dict[str, int] = {}
    calls_lock = threading.Lock()

    def retrying_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        with calls_lock:
            calls[label] = calls.get(label, 0) + 1
        is_retry = label.endswith("-retry-1")
        segment_id = label.removeprefix("companion-translation-").removesuffix("-retry-1")
        block_id_value = f"b{int(segment_id[-4:])}"
        if is_retry:
            assert "VALIDATION ERROR" in prompt
            assert "opaque_inline_token_mismatch" in prompt
            assert required_tokens[segment_id] in prompt
            assert pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION in prompt
            assert Path(kwargs["artifact_dir"]).name == "retry-1"
            assert kwargs["env"] is None
            assert kwargs["model_tier"] == pipeline_module.TRANSLATION_RETRY_TIER == "medium"
        else:
            assert kwargs["model_tier"] == pipeline_module.TRANSLATION_TIER == "low"
        valid = segment_id == "seg-0001" or (segment_id == "seg-0002" and is_retry)
        text = f"译文 {required_tokens[segment_id]}。" if valid else "缺少控制器令牌。"
        return {"blocks": [{"block_id": block_id_value, "text": text}]}

    checkpoint_dir = tmp_path / "checkpoints"
    try:
        _generate_translations(
            segments,
            options=BuildOptions(
                paper_id=bundle.paper_id, project_dir=tmp_path / "project", workers=4,
            ),
            bundle=bundle,
            glossary={"entries": []},
            protected_names=[],
            checkpoint_dir=checkpoint_dir,
            llm=retrying_llm,
        )
    except CompanionLaneError as exc:
        assert exc.lane == "translation"
        assert {segment_id for segment_id, _ in exc.failures} == {"seg-0003", "seg-0004"}
        assert all("opaque inline tokens" in str(error) for _, error in exc.failures)
    else:
        raise AssertionError("persistently invalid translations must fail the lane")

    assert calls == {
        "companion-translation-seg-0001": 1,
        "companion-translation-seg-0002": 1,
        "companion-translation-seg-0002-retry-1": 1,
        "companion-translation-seg-0003": 1,
        "companion-translation-seg-0003-retry-1": 1,
        "companion-translation-seg-0004": 1,
        "companion-translation-seg-0004-retry-1": 1,
    }
    saved_ids = {
        json.loads(path.read_text(encoding="utf-8"))["segment_id"]
        for path in (checkpoint_dir / "translations").glob("*.json")
    }
    assert saved_ids == {"seg-0001", "seg-0002"}

    recovery_calls: list[str] = []

    def recovery_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        recovery_calls.append(label)
        segment_id = label.removeprefix("companion-translation-")
        block_id_value = f"b{int(segment_id[-4:])}"
        return {
            "blocks": [{"block_id": block_id_value, "text": f"译文 {required_tokens[segment_id]}。"}]
        }

    recovered = _generate_translations(
        segments,
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "project", workers=4),
        bundle=bundle,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=checkpoint_dir,
        llm=recovery_llm,
    )
    assert list(recovered) == [segment["segment_id"] for segment in segments]
    assert set(recovery_calls) == {
        "companion-translation-seg-0003", "companion-translation-seg-0004",
    }


def test_non_token_translation_validation_errors_do_not_retry(tmp_path: Path) -> None:
    scenarios = {
        "coverage": {"blocks": []},
        "empty": {"blocks": [{"block_id": "body", "text": ""}]},
        "protected-name": {"blocks": [{"block_id": "body", "text": "此处省略姓名。"}]},
    }
    for scenario, generated in scenarios.items():
        block = {
            "block_id": "body", "type": "text", "text": "Ada presents the result.",
            "inline_runs": [
                _inline_run("text", "Ada presents the result.", 1),
            ],
        }
        document = {
            "schema_version": "arc.paper.document.v1",
            "front_matter": {"title": "Validation fixture", "authors": [], "affiliations": []},
            "blocks": [block],
            "equations": [], "figures": [], "tables": [], "bibliography": [], "assets": [],
            "integrity": {"status": "complete", "document_hash": f"fixture-{scenario}"},
        }
        bundle = SourceBundle(
            paper_id="arXiv:1234.5678",
            parsed={"paper_id": "arXiv:1234.5678", "document": document},
            document=document,
            metadata={"title": "Validation fixture"}, references=[], citers=[],
        )
        calls: list[str] = []

        def invalid_llm(prompt: str, **kwargs):
            calls.append(str(kwargs["call_label"]))
            return generated

        try:
            _generate_translations(
                [{"segment_id": "seg-0001", "block_ids": ["body"]}],
                options=BuildOptions(
                    paper_id=bundle.paper_id, project_dir=tmp_path / scenario, workers=1,
                ),
                bundle=bundle,
                glossary={"entries": []},
                protected_names=["Ada"],
                checkpoint_dir=tmp_path / f"checkpoints-{scenario}",
                llm=invalid_llm,
            )
        except CompanionLaneError as exc:
            assert len(exc.failures) == 1
            assert "retry-1" not in str(exc)
        else:
            raise AssertionError(f"{scenario} validation failure must fail the lane")
        assert calls == ["companion-translation-seg-0001"]


def test_generation_document_excludes_front_matter_and_all_source_only_sections() -> None:
    document = {
        "front_matter": {
            "title": "Paper Title",
            "authors": ["Ada Author"],
            "affiliations": ["Theory Institute"],
            "abstract": "Abstract remains generative.",
        },
        "blocks": [
            {"block_id": "title", "text": "Paper Title"},
            {"block_id": "author", "text": "Ada Author"},
            {"block_id": "aff", "text": "Theory Institute"},
            {"block_id": "abstract", "text": "Abstract remains generative."},
            {
                "block_id": "toc-title", "kind": "heading", "text": "Contents",
                "html": '<h6 class="ltx_title_contents">Contents</h6>',
            },
            {
                "block_id": "toc-list", "kind": "list", "text": "1 Body",
                "html": '<ol class="ltx_toclist"><li><a href="#S1">1 Body</a></li></ol>',
            },
            {"block_id": "body", "kind": "prose", "section_id": "S1", "text": "Body."},
            {"block_id": "ack-title", "kind": "heading", "section_id": "Sx", "text": "Acknowledgments"},
            {"block_id": "ack-body", "kind": "prose", "section_id": "Sx", "text": "We thank Ada."},
            {"block_id": "refs-title", "kind": "heading", "section_id": "bib", "text": "References"},
            {"block_id": "ref-1", "kind": "bibliography", "section_id": "bib", "text": "[1] Work."},
        ],
    }

    assert [item["block_id"] for item in _generation_document(document)["blocks"]] == ["abstract", "body"]


def _bundle(tmp_path: Path) -> SourceBundle:
    image = tmp_path / "cached.png"
    image.write_bytes(b"valid-png-fixture")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    document = {
        "schema_version": "arc.paper.document.v1",
        "front_matter": {
            "title": "A&B",
            "authors": [{"name": "Ada"}],
            "affiliations": [{"name": "Institute & Lab"}],
            "abstract": {"text": "An abstract with x & y."},
        },
        "blocks": [
            {"block_id": "b1", "type": "section", "title": "Setup"},
            {"block_id": "b2", "type": "text", "text": "Let x < y & y > 0."},
            {"block_id": "b3", "type": "equation", "equation_id": "eq1"},
            {"block_id": "b4", "type": "figure", "figure_id": "fig1"},
            {"block_id": "b5", "type": "table", "table_id": "tab1"},
            {"block_id": "b6", "type": "bibliography_item", "text": "Original visible reference"},
        ],
        "equations": [{
            "id": "eq1",
            "tex": ["E=mc^2", "p=mv"],
            "printed_equation_numbers": ["(7)", "(8)"],
            "label": "eq:energy",
        }],
        "figures": [{"id": "fig1", "asset_ids": ["a1"], "tag": "Figure 2:", "caption": "A plot"}],
        "tables": [{
            "id": "tab1",
            "tag": "Table 1:",
            "caption": "Values",
            "column_count": 2,
            "rows": [[{"text": "A", "rowspan": 2}, {"text": "B"}], [{"text": "C", "colspan": 1}]],
        }],
        "bibliography": [{"id": "ref1", "label": "[1]", "text": "Original visible reference"}],
        "assets": [{"asset_id": "a1", "cache_path": str(image), "sha256": digest}],
        "integrity": {"status": "complete", "document_hash": "document-fixture"},
    }
    return SourceBundle(
        paper_id="arXiv:1234.5678",
        parsed={"paper_id": "arXiv:1234.5678", "document": document},
        document=document,
        metadata={"title": "Fixture"},
        references=[{"paper_id": "arXiv:0001.0001", "title": "Prior"}],
        citers=[{"paper_id": "arXiv:9999.9999", "title": "Later"}],
    )


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.lock = threading.Lock()
        self.annotation_barrier = threading.Barrier(2)
        self.annotation_started = threading.Event()

    def __call__(self, prompt: str, **kwargs):
        with self.lock:
            self.calls.append({"prompt": prompt, **kwargs, "thread": threading.current_thread().name})
        label = kwargs["call_label"]
        if label.startswith("companion-segmentation-w-"):
            return {"cut_after_ordinals": [3]}
        if label.startswith("companion-glossary-window-"):
            return {"entries": [{
                "source_term": "energy", "target_term": "能量",
                "brief_explanation": "物理系统的能量", "aliases": [],
                "protected_names": [], "first_block_id": "b2",
            }]}
        if label == "companion-glossary-consolidation":
            return {"entries": [{
                "source_term": "energy", "target_term": "能量",
                "brief_explanation": "物理系统的能量", "aliases": [],
                "protected_names": [], "first_block_id": "b2",
            }]}
        if label.startswith("companion-translation-"):
            assert self.annotation_started.wait(timeout=5), "translation and commentary lanes did not overlap"
            return {"blocks": [
                {"block_id": "b1", "text": "设定"},
                {"block_id": "b2", "text": "令 x < y 且 y > 0。"},
            ]}
        if label.startswith("companion-annotation-"):
            self.annotation_started.set()
            self.annotation_barrier.wait(timeout=5)
            segment_id = label.rsplit("-", 1)[-1]
            return {
                "explanation": f"解释 {segment_id}", "prior_work": "", "later_work": "",
                "commentary": f"伴读 {segment_id}", "evidence_ids": [],
                "key_points": [], "source_notes": [],
            }
        if label.startswith("companion-section-review-"):
            return {
                "findings": [{"segment_id": "seg-0001", "issue": "check terminology"}],
                "reviewed_segments": [
                    {
                        "segment_id": "seg-0001",
                        "translation": {"blocks": [
                            {"block_id": "b1", "text": "设定"},
                            {"block_id": "b2", "text": "令 x < y 且 y > 0。"},
                        ]},
                        "annotation": {
                            "explanation": "解释 0001", "prior_work": "", "later_work": "",
                            "commentary": "伴读 0001", "evidence_ids": [],
                            "key_points": [], "source_notes": [],
                        },
                    },
                    {
                        "segment_id": "seg-0002", "translation": {"blocks": []},
                        "annotation": {
                            "explanation": "解释 0002", "prior_work": "", "later_work": "",
                            "commentary": "伴读 0002", "evidence_ids": [],
                            "key_points": [], "source_notes": [],
                        },
                    },
                ],
            }
        if label == "companion-final-review":
            return {"patches": [{
                "segment_id": "seg-0001",
                "translation_blocks": None,
                "commentary": "审校后的伴读",
                "explanation": "审校后的解释",
                "prior_work": None,
                "later_work": None,
                "evidence_ids": None,
                "reason": "precision",
            }], "issues": []}
        raise AssertionError(label)


def test_build_uses_tiered_parallel_lanes_and_is_source_faithful_and_resumable(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()

    def source_loader(*args, **kwargs):
        return bundle

    def compiler(tex_path: Path, pdf_path: Path) -> None:
        assert tex_path.is_file()
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            language_was_defaulted=True,
            workers=12,
            review_context_chars=1,
        ),
        source_loader=source_loader,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert result["ok"], result
    assert result["meta"]["notice"].startswith("默认使用中文")
    tiers = {str(call["call_label"]): call["model_tier"] for call in fake.calls}
    assert all(tier == "medium" for label, tier in tiers.items() if "segmentation" in label)
    assert all(tier == "medium" for label, tier in tiers.items() if "glossary" in label)
    assert all(tier == "high" for label, tier in tiers.items() if "annotation" in label)
    assert all(tier == "high" for label, tier in tiers.items() if "review" in label)
    assert all(tier == "low" for label, tier in tiers.items() if "translation" in label)
    assert all(call["session_policy"] == "stateless" for call in fake.calls)
    externally_enabled = [
        call for call in fake.calls
        if str(call["call_label"]).startswith(("companion-translation-", "companion-annotation-"))
    ]
    assert externally_enabled
    for call in externally_enabled:
        env = call["env"]
        assert env["ARC_CODEX_ALLOW_INTERNET"] == "true"
        assert env["ARC_CLAUDE_ALLOW_INTERNET"] == "true"
        assert env["ARC_CODEX_ENABLE_MCP"] == "true"
        assert env["ARC_CLAUDE_ALLOW_MCP"] == "true"
        assert env["ARC_CODEX_MCP_MODE"] == "arc-only"
        assert env["ARC_CLAUDE_MCP_MODE"] == "arc-only"
        assert "FULL-PAPER NAVIGATION CONTEXT" in str(call["prompt"])
        assert "Setup" in str(call["prompt"])
    assert all(
        call["env"] is None
        for call in fake.calls
        if call not in externally_enabled
    )
    annotation_calls = [call for call in fake.calls if str(call["call_label"]).startswith("companion-annotation-")]
    assert len(annotation_calls) == 2
    assert len({call["thread"] for call in annotation_calls}) == 2
    assert any(str(call["call_label"]).startswith("companion-section-review-") for call in fake.calls)
    section_prompts = [str(call["prompt"]) for call in fake.calls if str(call["call_label"]).startswith("companion-section-review-")]
    assert section_prompts and all('"source_blocks"' in prompt for prompt in section_prompts)
    final_prompt = next(str(call["prompt"]) for call in fake.calls if call["call_label"] == "companion-final-review")
    assert '"section_reviews"' in final_prompt
    assert '"reviewed_segments"' in final_prompt
    assert '"source_anchors"' in final_prompt
    assert '"source_excerpt"' in final_prompt
    assert '"segment_id": "seg-0001"' in final_prompt
    assert '"segment_id": "seg-0002"' in final_prompt

    data = result["data"]
    tex = Path(data["output_tex"]).read_text(encoding="utf-8")
    assert r"\tag{7}" in tex
    assert r"\tag{8}" in tex
    assert "E=mc^2" in tex
    assert "p=mv" in tex
    assert "['E=mc^2'" not in tex
    assert r"\multirow{2}{*}{A}" in tex
    assert "Figure 2: A plot" in tex
    assert "Figure Figure" not in tex
    assert "Institute \\& Lab" in tex
    assert "An abstract with x \\& y." in tex
    assert "Original visible reference" in tex
    assert "审校后的解释" in tex
    assert "解释 0002" in tex
    assert Path(data["output_pdf"]).is_file()
    saved_evidence = list((Path(data["checkpoint_dir"]) / "segment-evidence").glob("*.json"))
    assert len(saved_evidence) == 2
    assert all("evidence" in json.loads(path.read_text(encoding="utf-8")) for path in saved_evidence)

    call_count = len(fake.calls)
    resumed = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            language_was_defaulted=True,
            workers=12,
            review_context_chars=1,
        ),
        source_loader=source_loader,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {},
    )
    assert resumed["ok"] and resumed["meta"]["resumed"] is True
    assert len(fake.calls) == call_count

    state_path = tmp_path / "run" / "state.json"
    stale_preview_state = json.loads(state_path.read_text(encoding="utf-8"))
    stale_preview_state.pop("first_wave_preview_version")
    state_path.write_text(json.dumps(stale_preview_state), encoding="utf-8")
    regenerated_preview = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            language_was_defaulted=True,
            workers=12,
            review_context_chars=1,
        ),
        source_loader=source_loader,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {},
    )
    assert regenerated_preview["ok"] and regenerated_preview["meta"]["resumed"] is False
    assert regenerated_preview["data"]["first_wave_preview_version"]
    assert len(fake.calls) == call_count


def test_first_round_preview_is_published_before_evidence_resolution_and_review(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    fake.annotation_barrier = threading.Barrier(1)
    project = tmp_path / "preview-order"
    compiler_calls: list[tuple[Path, str]] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label == "companion-final-review":
            assert (project / "arXiv-1234.5678_companion_zh-CN_first_round_preview.pdf").is_file()
            assert (project / "first-round-preview-source-manifest.json").is_file()
            assert (project / "first-round-preview-validation.json").is_file()
        value = fake(prompt, **kwargs)
        if label == "companion-annotation-seg-0001":
            value = {**value, "evidence_requests": [{
                "relation": "context",
                "needed_claim": "context claim",
                "queries": ["context query"],
                "candidate_paper_ids": [],
                "candidate_urls": [],
                "reason": "verify context before review",
            }]}
        return value

    def compiler(tex_path: Path, pdf_path: Path) -> None:
        compiler_calls.append((tex_path, tex_path.read_text(encoding="utf-8")))
        if len(compiler_calls) == 1:
            labels = [str(call["call_label"]) for call in fake.calls]
            assert "companion-translation-seg-0001" in labels
            assert "companion-annotation-seg-0001" in labels
            assert "companion-translation-seg-0002" not in labels
            assert "companion-annotation-seg-0002" not in labels
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    class Controller:
        def resolve(self, requests, *, existing_records=()):
            assert list(requests)
            assert (project / "arXiv-1234.5678_companion_zh-CN_first_round_preview.pdf").is_file()
            state = json.loads((project / "state.json").read_text(encoding="utf-8"))
            assert state["status"] == "preview_ready"
            assert state["preview_pdf_sha256"]
            return EvidenceResolution(
                records=(),
                evidence_ids_by_segment={},
                supported_request_keys=(),
                audit={"requests": [], "lanes": {}, "accepted": [], "rejected": []},
            )

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=project,
            review_context_chars=1,
            workers=1,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
        evidence_controller=Controller(),
    )

    assert result["ok"], result
    assert len(compiler_calls) == 2
    assert "解释 0001" in compiler_calls[0][1]
    assert "解释 0002" not in compiler_calls[0][1]
    assert r"\tag{7}" in compiler_calls[0][1]
    assert "Figure 2: A plot" not in compiler_calls[0][1]
    assert "审校后的解释" not in compiler_calls[0][1]
    assert "审校后的解释" in compiler_calls[1][1]
    state = result["data"]
    assert state["preview_segment_count"] == 1
    assert state["preview_segment_ids"] == ["seg-0001"]
    for key in (
        "preview_tex", "preview_pdf", "preview_source_manifest_path", "preview_validation_path",
    ):
        assert Path(state[key]).is_file()
    for key in (
        "preview_tex_sha256", "preview_pdf_sha256",
        "preview_source_manifest_sha256", "preview_validation_sha256",
    ):
        assert state[key]


def test_first_wave_preview_preserves_rich_entities_and_exact_link_occurrences() -> None:
    repeated_link = '<a href="https://example.test/paper">paper</a>'
    document = {
        "blocks": [
            {
                "block_id": "kept", "kind": "heading", "html": '<h2 id="kept">Kept</h2>',
            },
            {
                "block_id": "eq-block", "source_id": "eq-source", "kind": "equation",
                "html": f'<div>{repeated_link}<a href="#later">later</a></div>',
            },
            {
                "block_id": "fig-block", "entity_id": "fig-entity", "kind": "figure",
                "html": f"<figure>{repeated_link}</figure>",
            },
            {
                "block_id": "tab-block", "source_id": "tab-source", "kind": "table",
                "html": '<table><tr><td><a href="#kept">kept</a></td></tr></table>',
            },
            {
                "block_id": "later", "source_id": "later-equation", "kind": "equation",
                "html": f'<div>{repeated_link}<a href="#later">later</a></div>',
            },
        ],
        "equations": [
            {"id": "eq-source", "tex": ["x=1"]},
            {"id": "later-equation", "tex": ["y=2"]},
        ],
        "figures": [{"id": "fig-entity", "asset_ids": ["asset-kept"]}],
        "tables": [{"id": "tab-source", "rows": []}],
        "assets": [
            {"asset_id": "asset-kept", "cache_path": "/cache/kept"},
            {"asset_id": "asset-later", "cache_path": "/cache/later"},
        ],
        "bibliography": [{"id": "ref-later", "label": "[1]"}],
        "links": [
            {"id": "link-1", "href": "https://example.test/paper", "target_id": "", "text": "paper"},
            {"id": "link-2", "href": "https://example.test/paper", "target_id": "", "text": "paper"},
            {"id": "link-3", "href": "#kept", "target_id": "kept", "text": "kept"},
            {"id": "link-later-1", "href": "https://example.test/paper", "target_id": "", "text": "paper"},
            {"id": "link-later-2", "href": "#later", "target_id": "later", "text": "later"},
        ],
    }
    preview = _first_wave_preview_document(document, [{
        "segment_id": "first",
        "block_ids": ["kept", "eq-block", "fig-block", "tab-block"],
    }])

    assert [item["block_id"] for item in preview["blocks"]] == [
        "kept", "eq-block", "fig-block", "tab-block",
    ]
    assert [item["id"] for item in preview["equations"]] == ["eq-source"]
    assert [item["id"] for item in preview["figures"]] == ["fig-entity"]
    assert [item["id"] for item in preview["tables"]] == ["tab-source"]
    assert [item["asset_id"] for item in preview["assets"]] == ["asset-kept"]
    assert preview["links"] == [
        {"href": "https://example.test/paper", "target_id": "", "text": "paper"},
        {"href": "#later", "target_id": "later", "text": "later"},
        {"href": "https://example.test/paper", "target_id": "", "text": "paper"},
        {"href": "#kept", "target_id": "kept", "text": "kept"},
    ]
    assert preview["bibliography"] == []
    assert preview["preview_scope"] == {"kind": "source_prefix"}


def test_first_round_preview_failure_stops_before_evidence_and_review(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    fake.annotation_barrier = threading.Barrier(1)

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "preview-failure",
            workers=1,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=fake,
        compiler=lambda *_: (_ for _ in ()).throw(RuntimeError("preview compile failed")),
        pdf_validator=lambda _: (_ for _ in ()).throw(AssertionError("must not validate")),
    )

    assert not result["ok"]
    assert "preview compile failed" in result["error"]["message"]
    assert not any("review" in str(call["call_label"]) for call in fake.calls)
    assert not any(
        str(call["call_label"]).endswith("seg-0002") for call in fake.calls
    )
    assert json.loads(
        (tmp_path / "preview-failure" / "state.json").read_text(encoding="utf-8")
    )["status"] == "failed"


def test_non_json_status_prefers_ready_preview_path(capsys) -> None:
    _emit(
        {"ok": True, "data": {"status": "preview_ready", "preview_pdf": "/run/preview.pdf"}, "meta": {}},
        json_output=False,
    )
    assert capsys.readouterr().out == "/run/preview.pdf\n"


def test_segment_ranges_must_cover_blocks_once_in_order() -> None:
    blocks = [{"block_id": value} for value in ("a", "b", "c")]
    valid = validate_and_expand_segments(
        [
            {"segment_id": "one", "title": "One", "start_block_id": "a", "end_block_id": "b"},
            {"segment_id": "two", "title": "Two", "start_block_id": "c", "end_block_id": "c"},
        ],
        blocks,
    )
    assert valid[0]["block_ids"] == ["a", "b"]

    invalid = [{"segment_id": "one", "title": "One", "start_block_id": "b", "end_block_id": "c"}]
    try:
        validate_and_expand_segments(invalid, blocks)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("invalid segmentation was accepted")


def test_fingerprint_covers_metadata_evidence_prompts_and_checkpoint_names(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    evidence = _evidence(bundle)
    first = _fingerprint(bundle, options, evidence=evidence)
    changed_metadata = SourceBundle(
        paper_id=bundle.paper_id,
        parsed=bundle.parsed,
        document=bundle.document,
        metadata={**bundle.metadata, "title": "Changed"},
        references=bundle.references,
        citers=bundle.citers,
    )
    assert _fingerprint(changed_metadata, options, evidence=evidence) != first
    assert _fingerprint(bundle, options, evidence={**evidence, "citers": []}) != first
    assert _segment_checkpoint_name("a/b") != _segment_checkpoint_name("a b")


def test_fingerprint_invalidates_when_glossary_tier_changes(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    evidence = _evidence(bundle)
    medium = _fingerprint(bundle, options, evidence=evidence)

    monkeypatch.setattr(pipeline_module, "GLOSSARY_TIER", "high")

    assert _fingerprint(bundle, options, evidence=evidence) != medium


def test_fingerprint_invalidates_when_review_tier_changes(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    evidence = _evidence(bundle)
    high = _fingerprint(bundle, options, evidence=evidence)

    monkeypatch.setattr(pipeline_module, "REVIEW_TIER", "medium")

    assert _fingerprint(bundle, options, evidence=evidence) != high


def test_fingerprint_invalidates_when_workers_per_lane_changes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(bundle)
    default = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    old = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run", workers=12)

    assert default.workers == 24
    assert _fingerprint(bundle, default, evidence=evidence) != _fingerprint(bundle, old, evidence=evidence)


def test_evidence_keeps_optional_source_diagnostics(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    warning = {
        "severity": "warning",
        "code": "citer_context_unavailable",
        "source": "arc-paper",
        "message": "Unable to load optional seed citers: offline",
    }
    bundle = SourceBundle(
        paper_id=bundle.paper_id,
        parsed=bundle.parsed,
        document=bundle.document,
        metadata=bundle.metadata,
        references=bundle.references,
        citers=[],
        diagnostics=(warning,),
    )

    evidence = _evidence(bundle)

    assert evidence["schema_version"] == "arc.companion.evidence.v2"
    assert evidence["citers"] == []
    assert evidence["diagnostics"] == [warning]


def test_global_protected_names_exclude_reference_and_citer_authors(tmp_path: Path) -> None:
    base = _bundle(tmp_path)
    bundle = SourceBundle(
        paper_id=base.paper_id,
        parsed=base.parsed,
        document=base.document,
        metadata={**base.metadata, "authors": [{"name": "Seed Author"}]},
        references=[{"title": "Prior", "authors": [{"name": "Tie Researcher"}]}],
        citers=[{"title": "Later", "authors": [{"name": "May Scholar"}]}],
    )
    glossary = {"entries": [{
        "source_term": "Feynman diagram",
        "target_term": "Feynman 图",
        "brief_explanation": "diagrammatic expansion",
        "protected_names": ["Feynman"],
    }]}

    names = _protected_names(bundle, glossary=glossary)

    assert "Seed Author" in names
    assert "Seed" in names
    assert "Author" in names
    assert "Feynman" in names
    assert "Tie Researcher" not in names
    assert "Tie" not in names
    assert "May Scholar" not in names
    assert "May" not in names


def test_full_paper_context_is_bounded_navigable_and_excludes_raw_html(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    document = {
        **bundle.document,
        "blocks": [
            *bundle.document["blocks"],
            {
                "block_id": "b7", "type": "section", "title": "Distant physics",
                "raw_html": "<div>DO_NOT_EXPOSE_RAW_HTML</div>",
            },
            {
                "block_id": "b8", "type": "text", "text": "late-time transfer context",
                "preservation_html": "<span>DO_NOT_EXPOSE_PRESERVATION_HTML</span>",
            },
        ],
    }
    by_id = {item["block_id"]: item for item in document["blocks"]}
    segment = {
        "segment_id": "seg-0001", "title": "Local", "start_block_id": "b1",
        "end_block_id": "b2", "block_ids": ["b1", "b2"],
    }

    context = _full_paper_context(document, segment, blocks_by_id=by_id, max_chars=4_000)
    serialized = json.dumps(context, ensure_ascii=False)

    assert context["schema_version"] == "arc.companion.full-paper-context.v1"
    assert any(item["title"] == "Distant physics" for item in context["section_navigation"])
    assert "late-time transfer context" in serialized
    assert "DO_NOT_EXPOSE_RAW_HTML" not in serialized
    assert "DO_NOT_EXPOSE_PRESERVATION_HTML" not in serialized
    assert len(json.dumps(context, ensure_ascii=False, separators=(",", ":"))) <= 4_000


def test_segment_evidence_preserves_descriptor_and_hashes_selected_snippets() -> None:
    full_blocks = [{
        "block_id": "related-b1", "type": "text", "text": "transfer vertex field theory",
        "sha256": text_sha256("transfer vertex field theory"),
    }]
    paper = {
        "evidence_id": "prior-001", "relation": "prior", "paper_id": "arXiv:0001.0001",
        "title": "Transfer", "authors": ["A. Author"], "year": 2001,
        "citation_count": 10, "evidence_level": "full_text", "abstract": "",
        "blocks": full_blocks,
    }
    paper["source_descriptor"] = arc_cache_descriptor(
        paper_id=paper["paper_id"], title=paper["title"], authors=paper["authors"],
        year=paper["year"], evidence_level="full_text", content=full_blocks,
        document_hash="d" * 64,
    )
    segment = {"segment_id": "seg-0001", "block_ids": ["source-b1"]}
    by_id = {"source-b1": {"block_id": "source-b1", "type": "text", "text": "transfer vertex"}}

    selected = _evidence_for_segment(segment, by_id, {"related_papers": [paper]})

    assert selected["schema_version"] == "arc.companion.segment-evidence.v2"
    record = selected["papers"][0]
    assert record["source_descriptor"]["locator"]["document_hash"] == "d" * 64
    assert record["snippets"][0]["sha256"] == text_sha256(record["snippets"][0]["text"])
    assert validate_evidence_record(record) is record


def test_invalid_review_patch_fails_without_publishing_pdf(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    original = fake.__call__
    final_prompts = []

    def bad_llm(prompt: str, **kwargs):
        if kwargs["call_label"] == "companion-final-review":
            final_prompts.append(prompt)
            return {"patches": [{"segment_id": "source-b1", "commentary": "tamper", "reason": "bad"}], "issues": []}
        return original(prompt, **kwargs)

    compiled: list[Path] = []

    def preview_compiler(tex_path: Path, pdf_path: Path) -> None:
        compiled.append(tex_path)
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "bad"),
        source_loader=lambda *args, **kwargs: bundle,
        llm=bad_llm,
        compiler=preview_compiler,
        pdf_validator=lambda _: {},
    )
    assert not result["ok"]
    assert "invalid or duplicate annotation patch" in result["error"]["message"]
    assert final_prompts and '"source_blocks"' in final_prompts[0]
    assert json.loads((tmp_path / "bad" / "state.json").read_text())["status"] == "failed"
    assert len(compiled) == 1
    assert (tmp_path / "bad" / "arXiv-1234.5678_companion_zh-CN_first_round_preview.pdf").is_file()
    assert not (tmp_path / "bad" / "arXiv-1234.5678_companion_zh-CN.pdf").exists()


def test_generation_failure_drains_both_lanes_and_retry_only_runs_missing_segments(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    base = FakeLLM()
    failed_generation_calls: list[str] = []

    def failing_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith(("companion-translation-", "companion-annotation-")):
            failed_generation_calls.append(label)
            if label.endswith("seg-0001"):
                raise RuntimeError(f"intentional early failure: {label}")
            if label.startswith("companion-annotation-"):
                return {
                    "explanation": "later segment survived", "prior_work": "", "later_work": "",
                    "commentary": "later segment survived", "evidence_ids": [],
                    "key_points": [], "source_notes": [],
                }
            raise AssertionError(label)
        return base(prompt, **kwargs)

    project = tmp_path / "drained-lanes"
    first = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project, workers=2),
        source_loader=lambda *args, **kwargs: bundle,
        llm=failing_llm,
        compiler=lambda *_: (_ for _ in ()).throw(AssertionError("must not compile")),
        pdf_validator=lambda _: {},
    )

    assert not first["ok"]
    assert "translation lane failed" in first["error"]["message"]
    assert "annotation lane failed" in first["error"]["message"]
    checkpoint = Path(json.loads((project / "state.json").read_text())["checkpoint_dir"])
    completed_annotations = {
        json.loads(path.read_text(encoding="utf-8"))["segment_id"]
        for path in (checkpoint / "annotations").glob("*.json")
    }
    completed_translations = {
        json.loads(path.read_text(encoding="utf-8"))["segment_id"]
        for path in (checkpoint / "translations").glob("*.json")
    }
    assert completed_annotations == {"seg-0002"}
    assert completed_translations == {"seg-0002"}

    retry_generation_calls: list[str] = []

    def retry_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith("companion-translation-"):
            retry_generation_calls.append(label)
            return {"blocks": [
                {"block_id": "b1", "text": "设定"},
                {"block_id": "b2", "text": "令 x < y 且 y > 0。"},
            ]}
        if label.startswith("companion-annotation-"):
            retry_generation_calls.append(label)
            return {
                "explanation": "retry explanation", "prior_work": "", "later_work": "",
                "commentary": "retry commentary", "evidence_ids": [],
                "key_points": [], "source_notes": [],
            }
        return base(prompt, **kwargs)

    def compiler(tex_path: Path, pdf_path: Path) -> None:
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    second = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project, workers=2),
        source_loader=lambda *args, **kwargs: bundle,
        llm=retry_llm,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert second["ok"], second
    assert sorted(retry_generation_calls) == [
        "companion-annotation-seg-0001",
        "companion-translation-seg-0001",
    ]


def test_invalid_segmentation_is_not_cached(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    def invalid_llm(prompt: str, **kwargs):
        assert kwargs["call_label"].startswith("companion-segmentation-w-")
        return {"cut_after_ordinals": [99]}

    project = tmp_path / "invalid-segments"
    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project),
        source_loader=lambda *args, **kwargs: bundle,
        llm=invalid_llm,
        compiler=lambda *_: (_ for _ in ()).throw(AssertionError("must not compile")),
        pdf_validator=lambda _: {},
    )
    assert not result["ok"]
    assert result["error"]["code"] == "companion_segmentation_failed"
    diagnostic = next(
        item
        for item in result["meta"]["diagnostics"]
        if item.get("code") == "companion_segmentation_failed"
    )
    assert diagnostic["source"] == "arc-companion"
    assert diagnostic["context"] == {
        "phase": "window",
        "window_id": "w-0001",
        "start_ordinal": 1,
            "end_ordinal": 5,
        "attempt": 3,
        "refinement": False,
    }
    state = json.loads((project / "state.json").read_text(encoding="utf-8"))
    assert diagnostic in state["diagnostics"]
    assert not list(project.rglob("segmentation.json"))


def test_json_emit_warns_about_segmentation_failure_on_stderr(capsys) -> None:
    result = {
        "ok": False,
        "data": None,
        "error": {
            "code": "companion_segmentation_failed",
            "message": "window w-0002 failed after 3 attempts",
        },
        "errors": [],
        "meta": {
            "diagnostics": [{
                "severity": "error",
                "code": "companion_segmentation_failed",
                "source": "arc-companion",
                "message": "window w-0002 failed after 3 attempts",
                "context": {"window_id": "w-0002", "attempt": 3},
            }],
        },
    }

    _emit(result, json_output=True)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert captured.err == "WARNING: window w-0002 failed after 3 attempts\n"


def test_complete_resume_requires_matching_output_hashes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    compile_count = 0

    def compiler(tex_path: Path, pdf_path: Path) -> None:
        nonlocal compile_count
        compile_count += 1
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    project = tmp_path / "hashed-resume"
    first = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project),
        source_loader=lambda *args, **kwargs: bundle,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert first["ok"]
    state = json.loads((project / "state.json").read_text())
    assert state["output_pdf_sha256"] and state["output_tex_sha256"]
    Path(state["output_pdf"]).write_bytes(b"%PDF tampered")

    second = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project),
        source_loader=lambda *args, **kwargs: bundle,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert second["ok"]
    assert second["meta"]["resumed"] is False
    assert compile_count == 4


def test_non_failed_state_clears_stale_error(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _state(path, status="failed", error="old failure")

    state = _state(path, status="segmenting")

    assert state["status"] == "segmenting"
    assert "error" not in state
