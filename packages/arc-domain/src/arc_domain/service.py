from __future__ import annotations

from typing import Any

from arc_paper.ids import normalize_paper_id

from .cache import DomainPaths, domain_id_for, now_iso, read_json, update_status, write_json
from .evidence import build_evidence_pack as _build_evidence_pack
from .foundation import identify_foundation as _identify_foundation
from .network import build_network as _build_network
from .render import render_network_html
from .results import err, ok
from .summary import summarize_domain as _summarize_domain


def init_domain(seed_paper: str, *, intent: str = "", domain_id: str | None = None) -> dict[str, Any]:
    try:
        seed_id = normalize_paper_id(seed_paper)
        resolved = domain_id or domain_id_for(seed_id, intent)
        paths = DomainPaths.for_domain(resolved)
        config = {
            "schema_version": "arc.domain_config.v1",
            "domain_id": paths.domain_id,
            "seed_paper": seed_id,
            "intent": intent,
            "created_at": now_iso(),
        }
        write_json(paths.config, config)
        update_status(paths, stage="initialized", seed_paper=seed_id, intent=intent)
        return ok({"domain_id": paths.domain_id, "domain_dir": str(paths.domain_dir), "config": config})
    except Exception as exc:
        return err("domain_init_failed", str(exc))


def identify_foundation(
    seed_paper: str,
    *,
    intent: str = "",
    domain_id: str | None = None,
    provider: str = "auto",
    model: str | None = None,
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    try:
        paths = _ensure_domain(seed_paper, intent=intent, domain_id=domain_id)
        data = _identify_foundation(
            seed_paper=seed_paper,
            intent=intent,
            paths=paths,
            provider=provider,
            model=model,
            refresh=refresh,
            workers=workers,
        )
        return ok(data)
    except Exception as exc:
        return err("foundation_identification_failed", str(exc))


def build_network(
    seed_paper: str,
    *,
    intent: str = "",
    domain_id: str | None = None,
    provider: str = "auto",
    model: str | None = None,
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    try:
        paths = _ensure_domain(seed_paper, intent=intent, domain_id=domain_id)
        if not paths.foundation_selection.exists():
            _identify_foundation(
                seed_paper=seed_paper,
                intent=intent,
                paths=paths,
                provider=provider,
                model=model,
                refresh=refresh,
                workers=workers,
            )
        data = _build_network(
            seed_paper=seed_paper,
            intent=intent,
            paths=paths,
            provider=provider,
            model=model,
            refresh=refresh,
            workers=workers,
        )
        render = render_network_html(paths=paths)
        data.update(render)
        return ok(data)
    except Exception as exc:
        return err("domain_network_failed", str(exc))


def build_evidence_pack(
    seed_paper: str,
    *,
    intent: str = "",
    domain_id: str | None = None,
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    try:
        paths = _ensure_domain(seed_paper, intent=intent, domain_id=domain_id)
        if not paths.domain_graph.exists():
            raise FileNotFoundError("domain_graph.json missing; run build-network first")
        return ok(_build_evidence_pack(paths=paths, refresh=refresh, workers=workers))
    except Exception as exc:
        return err("domain_evidence_failed", str(exc))


def summarize_domain(
    seed_paper: str,
    *,
    intent: str = "",
    domain_id: str | None = None,
    provider: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    try:
        paths = _ensure_domain(seed_paper, intent=intent, domain_id=domain_id)
        if not paths.evidence_pack.exists():
            raise FileNotFoundError("evidence_pack.json missing; run build-evidence first")
        return ok(_summarize_domain(paths=paths, provider=provider, model=model))
    except Exception as exc:
        return err("domain_summary_failed", str(exc))


def build_domain(
    seed_paper: str,
    *,
    intent: str = "",
    domain_id: str | None = None,
    provider: str = "auto",
    model: str | None = None,
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    try:
        paths = _ensure_domain(seed_paper, intent=intent, domain_id=domain_id)
        foundation = _identify_foundation(
            seed_paper=seed_paper,
            intent=intent,
            paths=paths,
            provider=provider,
            model=model,
            refresh=refresh,
            workers=workers,
        )
        network = _build_network(
            seed_paper=seed_paper,
            intent=intent,
            paths=paths,
            provider=provider,
            model=model,
            refresh=refresh,
            workers=workers,
        )
        html = render_network_html(paths=paths)
        evidence = _build_evidence_pack(paths=paths, refresh=refresh, workers=workers)
        summary = _summarize_domain(paths=paths, provider=provider, model=model)
        return ok(
            {
                "domain_id": paths.domain_id,
                "domain_dir": str(paths.domain_dir),
                "foundation": foundation["selection"],
                "network": {
                    "node_count": network["node_count"],
                    "edge_count": network["edge_count"],
                    "graph_path": network["graph_path"],
                    "network_html_path": html["network_html_path"],
                },
                "evidence_pack_path": evidence["evidence_pack_path"],
                "domain_summary_path": summary["domain_summary_path"],
                "summary": summary["summary"],
            }
        )
    except Exception as exc:
        return err("domain_build_failed", str(exc))


def status(seed_paper: str | None = None, *, intent: str = "", domain_id: str | None = None) -> dict[str, Any]:
    try:
        paths = _paths(seed_paper, intent=intent, domain_id=domain_id)
        data = {
            "domain_id": paths.domain_id,
            "domain_dir": str(paths.domain_dir),
            "status": read_json(paths.status, {}),
            "artifacts": {
                "config": _exists(paths.config),
                "foundation_pool": _exists(paths.foundation_pool),
                "foundation_candidates": _exists(paths.foundation_candidates),
                "foundation_selection": _exists(paths.foundation_selection),
                "citer_pool": _exists(paths.citer_pool),
                "selected_papers": _exists(paths.selected_papers),
                "reference_overlap": _exists(paths.reference_overlap),
                "domain_graph": _exists(paths.domain_graph),
                "evidence_pack": _exists(paths.evidence_pack),
                "domain_summary": _exists(paths.domain_summary),
                "network_html": _exists(paths.network_html),
            },
        }
        return ok(data)
    except Exception as exc:
        return err("domain_status_failed", str(exc))


def get_domain_summary(seed_paper: str | None = None, *, intent: str = "", domain_id: str | None = None) -> dict[str, Any]:
    try:
        paths = _paths(seed_paper, intent=intent, domain_id=domain_id)
        summary = read_json(paths.domain_summary)
        if not summary:
            return err("domain_summary_not_available", f"No domain summary exists for {paths.domain_id}")
        return ok({"domain_id": paths.domain_id, "summary": summary, "path": str(paths.domain_summary)})
    except Exception as exc:
        return err("domain_summary_read_failed", str(exc))


def get_domain_graph(seed_paper: str | None = None, *, intent: str = "", domain_id: str | None = None) -> dict[str, Any]:
    try:
        paths = _paths(seed_paper, intent=intent, domain_id=domain_id)
        graph = read_json(paths.domain_graph)
        if not graph:
            return err("domain_graph_not_available", f"No domain graph exists for {paths.domain_id}")
        return ok({"domain_id": paths.domain_id, "graph": graph, "path": str(paths.domain_graph)})
    except Exception as exc:
        return err("domain_graph_read_failed", str(exc))


def _ensure_domain(seed_paper: str, *, intent: str, domain_id: str | None) -> DomainPaths:
    seed_id = normalize_paper_id(seed_paper)
    paths = DomainPaths.for_domain(domain_id or domain_id_for(seed_id, intent))
    if not paths.config.exists():
        config = {
            "schema_version": "arc.domain_config.v1",
            "domain_id": paths.domain_id,
            "seed_paper": seed_id,
            "intent": intent,
            "created_at": now_iso(),
        }
        write_json(paths.config, config)
    return paths


def _paths(seed_paper: str | None, *, intent: str, domain_id: str | None) -> DomainPaths:
    if domain_id:
        return DomainPaths.for_domain(domain_id)
    if not seed_paper:
        raise ValueError("seed_paper or domain_id is required")
    return DomainPaths.for_domain(domain_id_for(normalize_paper_id(seed_paper), intent))


def _exists(path) -> dict[str, Any]:
    return {"exists": path.exists(), "path": str(path)}
