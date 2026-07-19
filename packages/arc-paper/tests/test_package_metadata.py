from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOTS = {
    "arc-llm": ROOT / "packages/arc-llm",
    "arc-paper": ROOT / "packages/arc-paper",
    "arc-domain": ROOT / "packages/arc-domain",
    "arc-typeset": ROOT / "packages/arc-typeset",
    "arc-companion": ROOT / "packages/arc-companion",
    "arc-mcp": ROOT / "packages/arc-mcp",
}
EXPECTED_INTERNAL_DEPENDENCIES = {
    "arc-llm": [],
    "arc-paper": ["arc-llm>=0.9,<0.10"],
    "arc-domain": ["arc-llm>=0.9,<0.10", "arc-paper>=0.9,<0.10"],
    "arc-typeset": ["arc-llm>=0.9,<0.10"],
    "arc-companion": ["arc-llm>=0.9,<0.10", "arc-paper>=0.9,<0.10"],
    "arc-mcp": [
        "arc-domain>=0.9,<0.10",
        "arc-llm>=0.9,<0.10",
        "arc-paper>=0.9,<0.10",
        "arc-typeset>=0.9,<0.10",
    ],
}
EXPECTED_EXTERNAL_DEPENDENCIES = {
    "arc-companion": ["beautifulsoup4>=4.12"],
}


def _pyproject(package_name: str) -> dict:
    path = PACKAGE_ROOTS[package_name] / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_arc_packages_have_publish_metadata():
    for package_name in PACKAGE_ROOTS:
        pyproject = _pyproject(package_name)
        project = pyproject["project"]

        assert project["readme"] == "../../README.md"
        assert project["license"] == "MIT"
        assert project["authors"] == [{"name": "ARC"}]
        assert "Programming Language :: Python :: 3.11" in project["classifiers"]
        assert project["urls"] == {
            "Homepage": "https://github.com/tririver/arc",
            "Repository": "https://github.com/tririver/arc",
            "Issues": "https://github.com/tririver/arc/issues",
        }


def test_arc_package_internal_dependencies_are_version_bounded():
    for package_name, expected_dependencies in EXPECTED_INTERNAL_DEPENDENCIES.items():
        pyproject = _pyproject(package_name)
        dependencies = pyproject["project"].get("dependencies", [])

        for dependency in expected_dependencies:
            assert dependency in dependencies


def test_arc_packages_declare_direct_external_dependencies():
    for package_name, expected_dependencies in EXPECTED_EXTERNAL_DEPENDENCIES.items():
        dependencies = _pyproject(package_name)["project"].get("dependencies", [])

        for dependency in expected_dependencies:
            assert dependency in dependencies


def test_arc_llm_dependency_is_version_bounded():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert "arc-llm>=0.9,<0.10" in dependencies
