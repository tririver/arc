from __future__ import annotations

import tomllib
from pathlib import Path


def test_arc_llm_dependency_is_version_bounded():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert "arc-llm>=0.1,<0.2" in dependencies
