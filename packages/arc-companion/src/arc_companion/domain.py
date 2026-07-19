from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DOMAIN_CONTEXT_VERSION = "arc.companion.domain-context.v1"
DOMAIN_MANIFEST_VERSION = "arc.workflow.domain_manifest.v1"


class DomainContextError(ValueError):
    """Raised when explicitly requested domain context cannot be read safely."""


def load_domain_context(
    *,
    domain_id: str | None = None,
    domain_manifest: Path | None = None,
) -> dict[str, Any] | None:
    """Read explicit existing domain artifacts without discovering or building a domain."""
    if domain_id and domain_manifest is not None:
        raise DomainContextError("domain_id and domain_manifest are mutually exclusive")
    if domain_id:
        return _load_domain_id(domain_id)
    if domain_manifest is not None:
        return _load_manifest(domain_manifest)
    return None


def _load_domain_id(domain_id: str) -> dict[str, Any]:
    requested = str(domain_id).strip()
    if not requested:
        raise DomainContextError("domain_id must not be empty")
    try:
        from arc_domain import service
    except ImportError as exc:  # pragma: no cover - packaging failure is environment-specific
        raise DomainContextError("arc-domain is required to load --domain-id") from exc

    summary_result = service.get_domain_summary(domain_id=requested)
    graph_result = service.get_domain_graph(domain_id=requested)
    summary_data = _result_data(summary_result, label=f"domain summary {requested}")
    graph_data = _result_data(graph_result, label=f"domain graph {requested}")
    summary = summary_data.get("summary") if isinstance(summary_data, dict) else None
    graph = graph_data.get("graph") if isinstance(graph_data, dict) else None
    if not isinstance(summary, dict) or not isinstance(graph, dict):
        raise DomainContextError(f"existing domain {requested} has incomplete summary or graph artifacts")
    papers = _papers_from_graph(graph)
    return {
        "schema_version": DOMAIN_CONTEXT_VERSION,
        "source": "domain_id",
        "domains": [{
            "domain_id": str(summary.get("domain_id") or requested),
            "summary": summary,
            "papers": papers,
            "artifact_paths": {
                key: str(value)
                for key, value in {
                    "summary_json": summary_data.get("path"),
                    "summary_markdown": summary_data.get("markdown_path"),
                    "graph_json": graph_data.get("path"),
                }.items()
                if value
            },
        }],
        "paper_ids": _paper_ids(papers),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != DOMAIN_MANIFEST_VERSION:
        raise DomainContextError(
            f"unsupported domain manifest schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}"
        )
    entries = manifest.get("domains")
    if not isinstance(entries, list) or not entries:
        raise DomainContextError(f"domain manifest has no domains: {manifest_path}")
    project_dir = manifest_path.parent.parent
    domains: list[dict[str, Any]] = []
    all_papers: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise DomainContextError(f"domain manifest entry {index} must be an object")
        domain_id = str(raw.get("domain_id") or "").strip()
        if not domain_id or domain_id in seen:
            raise DomainContextError(f"domain manifest has invalid or duplicate domain_id: {domain_id!r}")
        seen.add(domain_id)
        summary_path = _artifact_path(project_dir, raw.get("summary_json_path"), label="summary_json_path")
        pack_path = _artifact_path(project_dir, raw.get("paper_json_pack_path"), label="paper_json_pack_path")
        summary = _read_object(summary_path)
        pack = _read_object(pack_path)
        if str(summary.get("domain_id") or "") != domain_id:
            raise DomainContextError(f"domain summary ID does not match manifest entry {domain_id}")
        if str(pack.get("domain_id") or "") != domain_id:
            raise DomainContextError(f"domain paper pack ID does not match manifest entry {domain_id}")
        papers = _papers_from_pack(pack)
        all_papers.extend(papers)
        domains.append({
            "domain_id": domain_id,
            "summary": summary,
            "papers": papers,
            "artifact_paths": {
                "summary_json": str(summary_path),
                "paper_json_pack": str(pack_path),
            },
        })
    return {
        "schema_version": DOMAIN_CONTEXT_VERSION,
        "source": "domain_manifest",
        "manifest_path": str(manifest_path),
        "domains": domains,
        "paper_ids": _paper_ids(all_papers),
    }


def _result_data(result: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(result, dict) or not result.get("ok") or not isinstance(result.get("data"), dict):
        error = result.get("error") if isinstance(result, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        raise DomainContextError(f"unable to read existing {label}: {message or 'not available'}")
    return result["data"]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainContextError(f"unable to read domain artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DomainContextError(f"domain artifact root must be an object: {path}")
    return value


def _artifact_path(project_dir: Path, value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise DomainContextError(f"domain manifest entry requires {label}")
    candidate = Path(text).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_dir / candidate).resolve()


def _papers_from_graph(graph: dict[str, Any]) -> list[dict[str, str]]:
    return _paper_records(graph.get("nodes") or [])


def _papers_from_pack(pack: dict[str, Any]) -> list[dict[str, str]]:
    return _paper_records(pack.get("papers") or [])


def _paper_records(values: Any) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in values if isinstance(values, list) else []:
        if not isinstance(raw, dict):
            continue
        paper_id = str(raw.get("paper_id") or raw.get("id") or "").strip()
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        output.append({"paper_id": paper_id, "role": str(raw.get("role") or "")})
    return output


def _paper_ids(papers: list[dict[str, str]]) -> list[str]:
    return list(dict.fromkeys(item["paper_id"] for item in papers))
