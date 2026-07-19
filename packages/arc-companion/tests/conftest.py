from __future__ import annotations

from pathlib import Path
import sys


PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
ARC_LLM_SRC = Path(__file__).resolve().parents[2] / "arc-llm" / "src"
ARC_PAPER_SRC = Path(__file__).resolve().parents[2] / "arc-paper" / "src"
ARC_DOMAIN_SRC = Path(__file__).resolve().parents[2] / "arc-domain" / "src"
for path in (PACKAGE_SRC, ARC_LLM_SRC, ARC_PAPER_SRC, ARC_DOMAIN_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
