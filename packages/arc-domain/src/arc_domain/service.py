from __future__ import annotations

import hashlib
import json
from typing import Any

from arc_paper.ids import normalize_paper_id

from .cache import DomainPaths, domain_id_for, now_iso, read_json, update_status, write_json
from .evidence import build_evidence_pack as _build_evidence_pack
from .foundation import identify_foundation as _identify_foundation
from .network import build_network as _build_network
from .paper_pack import build_paper_json_pack as _build_paper_json_pack
from .render import render_network_html
from .results import err, ok
from .summary import summarize_domain as _summarize_domain


CONFIG_SCHEMA_VERSION = "arc.domain_config.v1"
INPUT_FINGERPRINT_SCHEMA_VERSION = "arc.domain_input_fingerprint.v1"


def init_domain(seed_paper: str, *, intent: str = "", domain_id: str | None = None) -> dict[str, Any]:
    try:
        seed_id = normalize_paper_id(seed_paper)
        resolved = domain_id or domain_id_for(seed_id, intent)
        paths = DomainPaths.for_domain(resolved)
        config = _domain_config(paths, seed_id=seed_id, intent=intent)
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
    model_tier: str | None = None,
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    try:
        paths = _ensure_domain(
            seed_paper,
            intent=intent,
            domain_id=domain_id,
            provider=provider,
            model=model,
            model_tier=model_tier,
        )
        data = _identify_foundation(
            seed_paper=seed_paper,
            intent=intent,
            paths=paths,
            provider=provider,
            model=model,
            model_tier=model_tier,
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
    model_tier: str | None = None,
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    try:
        paths = _ensure_domain(
            seed_paper,
            intent=intent,
            domain_id=domain_id,
            provider=provider,
            model=model,
            model_tier=model_tier,
        )
        if not paths.foundation_selection.exists():
            _identify_foundation(
                seed_paper=seed_paper,
                intent=intent,
                paths=paths,
                provider=provider,
                model=model,
                model_tier=model_tier,
                refresh=refresh,
                workers=workers,
            )
        data = _build_network(
            seed_paper=seed_paper,
            intent=intent,
            paths=paths,
            provider=provider,
            model=model,
            model_tier=model_tier,
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


def build_paper_json_pack(
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
        return ok(_build_paper_json_pack(paths=paths, refresh=refresh, workers=workers))
    except Exception as exc:
        return err("domain_paper_json_pack_failed", str(exc))


def summarize_domain(
    seed_paper: str,
    *,
    intent: str = "",
    domain_id: str | None = None,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
) -> dict[str, Any]:
    try:
        paths = _ensure_domain(
            seed_paper,
            intent=intent,
            domain_id=domain_id,
            provider=provider,
            model=model,
            model_tier=model_tier,
        )
        if not paths.evidence_pack.exists():
            raise FileNotFoundError("evidence_pack.json missing; run build-evidence first")
    except Exception as exc:
        return err("domain_summary_failed", str(exc))
    try:
        return ok(_summarize_domain(paths=paths, provider=provider, model=model, model_tier=model_tier))
    except Exception as exc:
        warning = _record_domain_summary_warning(paths, exc)
        return ok(
            _domain_summary_unavailable_result(paths, warning),
            warning=f"domain_summary_failed: {exc}",
        )


def build_domain(
    seed_paper: str,
    *,
    intent: str = "",
    domain_id: str | None = None,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str | None = None,
    refresh: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    try:
        paths = _ensure_domain(
            seed_paper,
            intent=intent,
            domain_id=domain_id,
            provider=provider,
            model=model,
            model_tier=model_tier,
        )
        foundation = _identify_foundation(
            seed_paper=seed_paper,
            intent=intent,
            paths=paths,
            provider=provider,
            model=model,
            model_tier=model_tier,
            refresh=refresh,
            workers=workers,
        )
        network = _build_network(
            seed_paper=seed_paper,
            intent=intent,
            paths=paths,
            provider=provider,
            model=model,
            model_tier=model_tier,
            refresh=refresh,
            workers=workers,
        )
        html = render_network_html(paths=paths)
        paper_pack = _build_paper_json_pack(paths=paths, refresh=refresh, workers=workers)
        evidence = _build_evidence_pack(paths=paths, refresh=refresh, workers=workers)
        warnings: list[dict[str, Any]] = []
        try:
            summary = _summarize_domain(paths=paths, provider=provider, model=model, model_tier=model_tier)
            summary_available = bool(summary.get("summary_available", summary.get("summary") is not None))
            if not summary_available and isinstance(summary.get("warnings"), list):
                warnings.extend(summary["warnings"])
        except Exception as exc:
            warning = _record_domain_summary_warning(paths, exc)
            warnings.append(warning)
            summary = _domain_summary_unavailable_result(paths, warning)
            summary_available = False
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
                "paper_json_pack_path": paper_pack["paper_json_pack_path"],
                "evidence_pack_path": evidence["evidence_pack_path"],
                "domain_summary_path": summary.get("domain_summary_path"),
                "domain_summary_markdown_path": summary.get("domain_summary_markdown_path"),
                "summary": summary.get("summary"),
                "summary_available": summary_available,
                "warnings": warnings,
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
                "paper_json_pack": _exists(paths.paper_json_pack),
                "evidence_pack": _exists(paths.evidence_pack),
                "domain_summary": _exists(paths.domain_summary),
                "domain_summary_markdown": _exists(paths.domain_summary_markdown),
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
        if summary.get("summary_method") == "deterministic_fallback":
            return err(
                "domain_summary_invalid",
                f"Cached deterministic fallback summary is invalid for {paths.domain_id}; rerun domain summarization.",
            )
        return ok(
            {
                "domain_id": paths.domain_id,
                "summary": summary,
                "path": str(paths.domain_summary),
                "markdown_path": str(paths.domain_summary_markdown) if paths.domain_summary_markdown.exists() else None,
            }
        )
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


def _ensure_domain(
    seed_paper: str,
    *,
    intent: str,
    domain_id: str | None,
    provider: str | None = None,
    model: str | None = None,
    model_tier: str | None = None,
) -> DomainPaths:
    seed_id = normalize_paper_id(seed_paper)
    paths = DomainPaths.for_domain(domain_id or domain_id_for(seed_id, intent))
    config = read_json(paths.config, {}) or {}
    if not config:
        config = _domain_config(
            paths,
            seed_id=seed_id,
            intent=intent,
            provider=provider,
            model=model,
            model_tier=model_tier,
        )
        write_json(paths.config, config)
        return paths
    _validate_domain_config(config, paths=paths, seed_id=seed_id, intent=intent)
    updated = _with_llm_fingerprint(config, provider=provider, model=model, model_tier=model_tier)
    if updated != config:
        write_json(paths.config, updated)
    return paths


def _domain_config(
    paths: DomainPaths,
    *,
    seed_id: str,
    intent: str,
    provider: str | None = None,
    model: str | None = None,
    model_tier: str | None = None,
) -> dict[str, Any]:
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "domain_id": paths.domain_id,
        "seed_paper": seed_id,
        "intent": intent,
        "created_at": now_iso(),
        "input_fingerprint": _input_fingerprint(seed_id=seed_id, intent=intent),
    }
    return _with_llm_fingerprint(config, provider=provider, model=model, model_tier=model_tier)


def _validate_domain_config(config: dict[str, Any], *, paths: DomainPaths, seed_id: str, intent: str) -> None:
    existing_seed = str(config.get("seed_paper") or "")
    existing_intent = str(config.get("intent") or "")
    if existing_seed != seed_id or existing_intent != intent:
        raise ValueError(
            "domain_id input mismatch: "
            f"{paths.domain_id} was created for seed_paper={existing_seed!r}, intent={existing_intent!r}; "
            f"requested seed_paper={seed_id!r}, intent={intent!r}"
        )
    expected = _input_fingerprint(seed_id=seed_id, intent=intent)
    stored = config.get("input_fingerprint")
    if isinstance(stored, dict) and stored.get("identity_hash") not in {None, expected["identity_hash"]}:
        raise ValueError(f"domain_id input hash mismatch for {paths.domain_id}")


def _with_llm_fingerprint(
    config: dict[str, Any],
    *,
    provider: str | None,
    model: str | None,
    model_tier: str | None,
) -> dict[str, Any]:
    if provider is None and model is None and model_tier is None:
        return config
    current = _llm_fingerprint(provider=provider, model=model, model_tier=model_tier)
    stored = config.get("input_fingerprint")
    if not isinstance(stored, dict):
        stored = _input_fingerprint(seed_id=str(config.get("seed_paper") or ""), intent=str(config.get("intent") or ""))
    existing_hash = stored.get("llm_hash")
    if existing_hash not in {None, current["llm_hash"]}:
        existing = stored.get("llm", {})
        raise ValueError(
            "domain_id LLM configuration mismatch: "
            f"created with {existing}; requested {current['llm']}"
        )
    if existing_hash == current["llm_hash"]:
        return config
    updated = dict(config)
    fingerprint = dict(stored)
    fingerprint.update(current)
    updated["input_fingerprint"] = fingerprint
    return updated


def _input_fingerprint(*, seed_id: str, intent: str) -> dict[str, Any]:
    identity = {
        "schema_version": INPUT_FINGERPRINT_SCHEMA_VERSION,
        "seed_paper": seed_id,
        "intent": intent,
    }
    return {
        "schema_version": INPUT_FINGERPRINT_SCHEMA_VERSION,
        "identity": identity,
        "identity_hash": _stable_hash(identity),
    }


def _llm_fingerprint(*, provider: str | None, model: str | None, model_tier: str | None) -> dict[str, Any]:
    llm = {
        "provider": provider or "auto",
        "model": model,
        "model_tier": model_tier,
    }
    return {"llm": llm, "llm_hash": _stable_hash(llm)}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _paths(seed_paper: str | None, *, intent: str, domain_id: str | None) -> DomainPaths:
    if domain_id:
        return DomainPaths.for_domain(domain_id)
    if not seed_paper:
        raise ValueError("seed_paper or domain_id is required")
    return DomainPaths.for_domain(domain_id_for(normalize_paper_id(seed_paper), intent))


def _exists(path) -> dict[str, Any]:
    return {"exists": path.exists(), "path": str(path)}


def _record_domain_summary_warning(paths: DomainPaths, exc: Exception) -> dict[str, Any]:
    _remove_stale_domain_summary_artifacts(paths)
    warning = {
        "code": "domain_summary_failed",
        "message": str(exc),
        "error_type": type(exc).__name__,
        "created_at": now_iso(),
    }
    status = read_json(paths.status, {}) or {}
    warnings = status.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    warnings.append(warning)
    update_status(
        paths,
        stage="summary_failed",
        domain_summary_available=False,
        domain_summary_path=None,
        domain_summary_markdown_path=None,
        domain_summary_error=warning,
        warnings=warnings,
    )
    return warning


def _remove_stale_domain_summary_artifacts(paths: DomainPaths) -> None:
    for path in (paths.domain_summary, paths.domain_summary_markdown):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _domain_summary_unavailable_result(paths: DomainPaths, warning: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain_id": paths.domain_id,
        "domain_summary_path": None,
        "domain_summary_markdown_path": None,
        "summary": None,
        "summary_available": False,
        "warnings": [warning],
    }
