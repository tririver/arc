from __future__ import annotations

import ast
from pathlib import Path


SLOW_DOMAIN_TESTS = {
    "test_build_domain_writes_core_artifacts",
    "test_status_and_cached_summary",
    "test_network_marks_llm_added_foundation",
}


def test_slow_domain_tests_are_opt_in():
    source = Path("packages/arc-domain/tests/test_domain_build.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    missing = []
    for name in SLOW_DOMAIN_TESTS:
        node = functions[name]
        decorator_text = "\n".join(ast.unparse(decorator) for decorator in node.decorator_list)
        if "pytest.mark.skipif" not in decorator_text or "ARC_RUN_SLOW_DOMAIN_TESTS" not in decorator_text:
            missing.append(name)

    assert missing == []
