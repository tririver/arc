from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "plugins/arc/skills/arc/workflows/companion.md"
MANUAL = ROOT / "plugins/arc/skills/arc/manuals/arc-companion.md"
README = ROOT / "README.md"


def test_companion_workflow_requires_ignored_repository_run_directory() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    manual = MANUAL.read_text(encoding="utf-8")

    required = "git check-ignore -q --no-index <resolved-project-dir>"
    assert required in workflow
    assert required in manual
    assert "before `arc-companion` or writing `context.json`" in workflow
    assert "ARC repository runs belong under\n`arc-tests/`" in workflow
    assert "Outside a Git worktree" in workflow
    assert "exact resolved `<project-dir>`" in workflow


def test_readme_companion_commands_use_ignored_arc_tests_tree() -> None:
    readme = README.read_text(encoding="utf-8")
    marker = "arc-companion build arXiv:0911.3380"
    start = readme.index(marker)
    block = readme[start : start + 500]

    assert "--project-dir ./arc-tests/companion/0911.3380" in block
    assert "--project-dir ./0911.3380-companion" not in readme
