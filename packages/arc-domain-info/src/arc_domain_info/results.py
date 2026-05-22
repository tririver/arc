from __future__ import annotations

from typing import Any


def ok(data: Any, **meta: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "data": data,
        "errors": [],
        "meta": {key: value for key, value in meta.items() if value is not None},
    }


def err(code: str, message: str, **extra: Any) -> dict[str, Any]:
    result = {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
        "errors": [{"code": code, "message": message}],
        "meta": {},
    }
    result.update(extra)
    return result
