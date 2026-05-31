from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_hash(schema: dict[str, Any] | None) -> str | None:
    if schema is None:
        return None
    return sha256_text(canonical_json(schema))


def write_schema_cache_file(schema: dict[str, Any], *, cache_dir: Path) -> Path:
    digest = schema_hash(schema)
    assert digest is not None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{digest}.schema.json"
    if not path.exists():
        path.write_text(json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
