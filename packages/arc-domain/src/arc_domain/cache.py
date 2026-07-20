from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import get_ident
from typing import Any

from arc_paper.ids import normalize_paper_id


@dataclass(frozen=True)
class DomainPaths:
    domain_id: str
    domain_dir: Path
    config: Path
    status: Path
    foundation_pool: Path
    foundation_candidates: Path
    foundation_selection: Path
    citer_pool: Path
    intent_rankings: Path
    selected_papers: Path
    reference_overlap: Path
    domain_graph: Path
    paper_json_pack: Path
    evidence_pack: Path
    domain_summary: Path
    domain_summary_markdown: Path
    network_html: Path

    @classmethod
    def for_domain(cls, domain_id: str) -> "DomainPaths":
        domain_dir = cache_root() / "domains" / safe_domain_id(domain_id)
        return cls(
            domain_id=safe_domain_id(domain_id),
            domain_dir=domain_dir,
            config=domain_dir / "config.json",
            status=domain_dir / "status.json",
            foundation_pool=domain_dir / "foundation_pool.json",
            foundation_candidates=domain_dir / "foundation_candidates.json",
            foundation_selection=domain_dir / "foundation_selection.json",
            citer_pool=domain_dir / "citer_pool.json",
            intent_rankings=domain_dir / "intent_rankings.json",
            selected_papers=domain_dir / "selected_papers.json",
            reference_overlap=domain_dir / "reference_overlap.json",
            domain_graph=domain_dir / "domain_graph.json",
            paper_json_pack=domain_dir / "paper_json_pack.json",
            evidence_pack=domain_dir / "evidence_pack.json",
            domain_summary=domain_dir / "domain_summary.json",
            domain_summary_markdown=domain_dir / "domain_summary.md",
            network_html=domain_dir / "network.html",
        )


def domain_id_for(seed_paper: str, intent: str = "") -> str:
    normalized = normalize_paper_id(seed_paper)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_") or "domain"
    digest = hashlib.sha1(f"{normalized}\n{intent.strip()}".encode("utf-8")).hexdigest()[:10]
    return safe_domain_id(f"{stem}_{digest}")


def safe_domain_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._-") or "domain"


def cache_root() -> Path:
    if value := os.environ.get("ARC_DOMAIN_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("ARC_HOME"):
        return Path(value).expanduser() / "cache" / "arc-domain"
    if value := os.environ.get("XDG_CACHE_HOME"):
        return Path(value).expanduser() / "arc" / "arc-domain"
    return Path.home() / ".cache" / "arc" / "arc-domain"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{get_ident()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{get_ident()}.{time.time_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def update_status(paths: DomainPaths, **fields: Any) -> dict[str, Any]:
    status = read_json(paths.status, {}) or {}
    status.update(fields)
    status["updated_at"] = now_iso()
    write_json(paths.status, status)
    return status
