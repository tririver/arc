from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARC_LLM_SRC = ROOT / "packages/arc-llm/src"


def _run_import_check(source: str) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ARC_LLM_SRC)
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=True,
    )
    return result.stdout.strip()


def test_arc_llm_import_is_lightweight() -> None:
    assert _run_import_check("import sys; import arc_llm; print('jsonschema' in sys.modules)") == "False"


def test_arc_llm_runner_config_import_is_lightweight() -> None:
    source = "import sys; from arc_llm.runner import resolve_llm_config; print('jsonschema' in sys.modules)"
    assert _run_import_check(source) == "False"


def test_arc_llm_lazy_public_exports() -> None:
    from arc_llm import resolve_llm_config, run_json, run_proposers_reviewer_batch, run_text

    assert callable(run_json)
    assert callable(run_text)
    assert callable(resolve_llm_config)
    assert callable(run_proposers_reviewer_batch)
