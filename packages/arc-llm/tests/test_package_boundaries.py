from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/arc_llm"


def test_arc_llm_does_not_ship_workflow_consensus_module() -> None:
    assert not (SRC / "proposers_reviewer/consensus.py").exists()


def test_arc_llm_source_does_not_depend_on_arc_skill_files() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "skills/arc" in text or "integrity.md" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_arc_llm_does_not_ship_openai_compatible_provider_stack() -> None:
    removed = [
        SRC / "providers/openai_compatible.py",
        SRC / "providers/config.py",
    ]

    assert [path for path in removed if path.exists()] == []


def test_arc_llm_source_has_no_deepseek_or_provider_config_paths() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        if "deepseek" in text or "openai-compatible" in text or "arc_llm_provider_config" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
