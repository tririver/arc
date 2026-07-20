from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))
