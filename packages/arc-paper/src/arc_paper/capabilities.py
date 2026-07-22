"""Structured ARC-paper operation catalog and dispatcher.

The catalog is the package-owned contract used by CLIs, worker wrappers, and
controllers.  It deliberately accepts an operation name plus JSON-compatible
parameters; it never accepts or executes an arbitrary command line.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

from jsonschema import Draft202012Validator

from .results import err, ok


CATALOG_SCHEMA_VERSION = "arc.paper.capability-catalog.v1"
RECURSIVE_LLM_CAPABILITY = "recursive_llm"

NetworkAccess = Literal["none", "may"]
CacheAccess = Literal["none", "read", "write", "read-write"]
ExecutionClass = Literal["inline", "job"]
ArtifactAccess = Literal["read", "write"]
ArtifactResolver = Callable[..., str | Path]


RESULT_ENVELOPE_SCHEMA: Mapping[str, Any] = MappingProxyType({
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "data": {},
        "error": {},
        "errors": {"type": "array"},
        "meta": {"type": "object"},
    },
    "required": ["ok"],
})
CONTROLLER_RESULT_DELIVERY: Mapping[str, Any] = MappingProxyType({
    "owner": "controller",
    "pagination": {
        "mode": "byte-limited-artifact-handle",
        "max_inline_bytes": 64 * 1024,
    },
    "provenance": {
        "mode": "operation-receipt",
        "required_fields": ["operation_id", "operation_version", "arguments_sha256"],
    },
})


@dataclass(frozen=True)
class OperationSpec:
    """One callable ARC-paper operation and its externally visible effects."""

    name: str
    description: str
    parameter_schema: Mapping[str, Any]
    executor: str
    version: int = 1
    result_schema: Mapping[str, Any] = field(default_factory=lambda: RESULT_ENVELOPE_SCHEMA)
    result_serializer: str = "arc.result-envelope.v1"
    positional_parameters: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    network_access: NetworkAccess = "none"
    cache_access: CacheAccess = "none"
    uses_llm: bool = False
    is_job: bool = False
    destructive: bool = False
    admin: bool = False
    artifact_parameters: tuple[tuple[str, ArtifactAccess], ...] = ()

    @property
    def operation_id(self) -> str:
        return f"arc-paper.{self.name}.v{self.version}"

    @property
    def execution_class(self) -> ExecutionClass:
        return "job" if self.is_job else "inline"

    @property
    def capabilities(self) -> frozenset[str]:
        """Legacy policy capabilities derived from the authoritative profile."""

        return frozenset({RECURSIVE_LLM_CAPABILITY}) if self.uses_llm else frozenset()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.operation_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "aliases": list(self.aliases),
            "parameters": deepcopy(dict(self.parameter_schema)),
            "result": {
                "schema": deepcopy(dict(self.result_schema)),
                "serializer": self.result_serializer,
                "delivery": deepcopy(dict(CONTROLLER_RESULT_DELIVERY)),
            },
            "classification": {
                "execution": self.execution_class,
                "network": self.network_access,
                "cache": self.cache_access,
                "llm": self.uses_llm,
                "job": self.is_job,
                "destructive": self.destructive,
                "admin": self.admin,
            },
            "authorization": {
                "supervision_required": self.destructive or self.admin,
                "artifact_handles": {
                    parameter: access for parameter, access in self.artifact_parameters
                },
            },
        }


def _string(*, enum: list[str] | None = None, min_length: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if enum is not None:
        schema["enum"] = enum
    if min_length is not None:
        schema["minLength"] = min_length
    return schema


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [dict(schema), {"type": "null"}]}


def _array(item: Mapping[str, Any], *, min_items: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": dict(item)}
    if min_items is not None:
        schema["minItems"] = min_items
    return schema


def _object(
    properties: Mapping[str, Any] | None = None,
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": dict(properties or {}),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


NONEMPTY = _string(min_length=1)
OPTIONAL_STRING = _nullable(_string())
ARTIFACT_HANDLE = _object({"handle_id": NONEMPTY}, required=("handle_id",))
OPTIONAL_ARTIFACT_HANDLE = _nullable(ARTIFACT_HANDLE)
PAPER_IDS = {
    "oneOf": [
        NONEMPTY,
        _array(NONEMPTY, min_items=1),
    ]
}
OPTIONAL_PAPER_IDS = _nullable(PAPER_IDS)
REFRESH = {"type": "boolean", "default": False}
MODEL_TIER = _nullable(_string(enum=["max", "high", "medium", "low"]))


def _paper_schema(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    properties = {"paper_ids": PAPER_IDS, "refresh": REFRESH}
    properties.update(extra or {})
    return _object(properties, required=("paper_ids",))


def _spec(
    name: str,
    description: str,
    schema: Mapping[str, Any],
    executor: str,
    *,
    positional: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    network: NetworkAccess = "none",
    cache: CacheAccess = "none",
    llm: bool = False,
    job: bool = False,
    destructive: bool = False,
    admin: bool = False,
    artifacts: tuple[tuple[str, ArtifactAccess], ...] = (),
) -> OperationSpec:
    return OperationSpec(
        name=name,
        description=description,
        parameter_schema=schema,
        executor=executor,
        positional_parameters=positional,
        aliases=aliases,
        network_access=network,
        cache_access=cache,
        uses_llm=llm,
        is_job=job,
        destructive=destructive,
        admin=admin,
        artifact_parameters=artifacts,
    )


_SUMMARY_OPTIONS = {
    "provider": _string(),
    "model": OPTIONAL_STRING,
    "model_tier": MODEL_TIER,
    "refresh": REFRESH,
}
_SOURCE_ID = {"source_id": NONEMPTY}


_OPERATIONS = (
    _spec(
        "extract-paper-ids", "Extract normalized paper identifiers from text.",
        _object({"text": _string()}, required=("text",)), "service.extract_paper_ids",
        positional=("text",), aliases=("extract-ids",),
    ),
    _spec(
        "safe-dir-name", "Create a stable filesystem-safe name for paper identifiers.",
        _object({"paper_ids": _array(NONEMPTY, min_items=1)}, required=("paper_ids",)),
        "service.paper_ids_safe_dir_name", positional=("paper_ids",),
        aliases=("paper-ids-safe-dir-name",),
    ),
    _spec(
        "llm-infer-main-references", "Infer and verify the main references in free text.",
        _object(
            {
                "text": _string(), "provider": _string(), "model": OPTIONAL_STRING,
                "model_tier": MODEL_TIER, "refresh": REFRESH,
            },
            required=("text",),
        ),
        "service.llm_infer_main_references", positional=("text",), aliases=("infer-main-references",),
        network="may", cache="read-write", llm=True,
    ),
    _spec(
        "get-parsed-identity", "Read cached parsed-source identity metadata.",
        _object(
            {**_SOURCE_ID, "include_document": {"type": "boolean"}},
            required=("source_id",),
        ),
        "service.get_parsed_source_identity", positional=("source_id",), cache="read",
    ),
    *(
        _spec(
            name, description, _paper_schema(), executor, positional=("paper_ids",),
            network="may", cache="read-write",
        )
        for name, description, executor in (
            ("get-title", "Get paper titles.", "service.get_title"),
            ("get-abstract", "Get paper abstracts.", "service.get_abstract"),
            ("get-authors", "Get paper authors.", "service.get_authors"),
            ("get-metadata", "Get normalized paper metadata.", "service.get_metadata"),
            ("get-citer-count", "Get paper citation counts.", "service.get_citer_count"),
            ("get-toc", "Get parsed paper tables of contents.", "service.get_toc"),
        )
    ),
    _spec(
        "search-inspire", "Search INSPIRE paper metadata.",
        _object({"query": NONEMPTY, "limit": {"type": "integer", "minimum": 1, "maximum": 1000}}, required=("query",)),
        "service.search_inspire", positional=("query",), network="may", cache="read-write",
    ),
    _spec(
        "get-references", "Get paper references.", _paper_schema({"enrich": {"type": "boolean", "default": False}}),
        "service.get_references", positional=("paper_ids",), network="may", cache="read-write",
    ),
    _spec(
        "get-citers", "Get papers citing the requested papers.",
        _paper_schema({"limit": {"type": "integer", "minimum": 1}, "sort": _string(enum=["mostrecent", "mostcited"])}),
        "service.get_citers", positional=("paper_ids",), network="may", cache="read-write",
    ),
    _spec(
        "get-section", "Get a paper section by locator.",
        _object(
            {"paper_ids": PAPER_IDS, "refresh": REFRESH, "section": NONEMPTY},
            required=("paper_ids", "section"),
        ),
        "service.get_section", positional=("paper_ids", "section"),
        network="may", cache="read-write",
    ),
    _spec(
        "get-equation-context", "Find equation context in papers.",
        _object(
            {"paper_ids": PAPER_IDS, "refresh": REFRESH, "query": NONEMPTY},
            required=("paper_ids", "query"),
        ),
        "service.get_equation_context", positional=("paper_ids", "query"),
        network="may", cache="read-write",
    ),
    _spec(
        "search-full-text", "Search parsed full text.",
        _object(
            {
                "paper_ids": OPTIONAL_PAPER_IDS, "query": NONEMPTY, "refresh": REFRESH,
                "limit": {"type": "integer", "minimum": 1, "maximum": 10000},
                "context": {"type": "integer", "minimum": 0},
                "case_sensitive": {"type": "boolean"},
            },
            required=("query",),
        ),
        "service.search_full_text", positional=("paper_ids",), network="may", cache="read-write",
    ),
    _spec(
        "source-cache", "Fetch and cache a versioned arXiv source archive.",
        _object(
            {
                "paper_id": NONEMPTY, "version": {"type": "integer", "minimum": 1},
                "refresh": REFRESH, "license_url": _string(),
            },
            required=("paper_id", "version"),
        ),
        "service.cache_arxiv_source", positional=("paper_id",), network="may", cache="read-write",
    ),
    _spec(
        "source-probe", "Probe a cached versioned arXiv source archive.",
        _object(
            {"paper_id": NONEMPTY, "version": {"type": "integer", "minimum": 1}},
            required=("paper_id", "version"),
        ),
        "service.probe_arxiv_source", positional=("paper_id",), cache="read",
    ),
    _spec(
        "parse", "Parse an ar5iv or local document source.",
        _object({
            "source_path": OPTIONAL_ARTIFACT_HANDLE,
            "source": _string(
                enum=[
                    "auto", "ar5iv", "html", "tex", "markdown", "pdf", "tex-pdf",
                    "markdown-pdf",
                ]
            ),
            "source_id": OPTIONAL_STRING, "paper_id": OPTIONAL_STRING,
            "html_path": OPTIONAL_ARTIFACT_HANDLE,
            "tex_path": OPTIONAL_ARTIFACT_HANDLE,
            "markdown_path": OPTIONAL_ARTIFACT_HANDLE,
            "pdf_path": OPTIONAL_ARTIFACT_HANDLE,
            "refresh": REFRESH, "include_document": {"type": "boolean"}, "recache": {"type": "boolean"},
            "document_kind": _string(enum=["auto", "article", "book"]),
        }), "service.parse_source", positional=("source_path",), network="may", cache="read-write",
        artifacts=(
            ("source_path", "read"), ("html_path", "read"), ("tex_path", "read"),
            ("markdown_path", "read"), ("pdf_path", "read"),
        ),
    ),
    _spec(
        "get-parsed", "Read a cached parsed source.",
        _object({**_SOURCE_ID, "include_document": {"type": "boolean"}}, required=("source_id",)),
        "service.get_parsed_source", positional=("source_id",), network="may", cache="read-write",
    ),
    *(
        _spec(
            name, description, _object(_SOURCE_ID, required=("source_id",)), executor,
            positional=("source_id",), cache="read",
        )
        for name, description, executor in (
            ("get-parsed-compact-toc", "Read a body-free parsed-source TOC.", "service.get_parsed_source_compact_toc"),
            ("get-parsed-toc", "Read a cached parsed-source TOC.", "service.get_parsed_source_toc"),
            ("get-parsed-equations", "Read cached parsed-source equations.", "service.get_parsed_source_equations"),
        )
    ),
    _spec(
        "get-parsed-section", "Read a cached parsed-source section.",
        _object({**_SOURCE_ID, "section": NONEMPTY}, required=("source_id", "section")),
        "service.get_parsed_source_section", positional=("source_id", "section"), cache="read",
    ),
    _spec(
        "get-parsed-equation", "Read one cached parsed-source equation.",
        _object({**_SOURCE_ID, "equation_id": NONEMPTY}, required=("source_id", "equation_id")),
        "service.get_parsed_source_equation", positional=("source_id", "equation_id"), cache="read",
    ),
    _spec(
        "mark-parsed-equation", "Store a review annotation for a parsed equation.",
        _object(
            {
                **_SOURCE_ID, "equation_id": NONEMPTY,
                "status": _string(enum=["problematic", "needs_recache", "resolved"]),
                "reason": NONEMPTY,
            },
            required=("source_id", "equation_id", "reason"),
        ),
        "service.mark_parsed_equation", positional=("source_id", "equation_id"), cache="read-write",
    ),
    _spec(
        "search-parsed", "Search one cached parsed source.",
        _object(
            {
                **_SOURCE_ID, "query": NONEMPTY,
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                "case_sensitive": {"type": "boolean"},
            },
            required=("source_id", "query"),
        ),
        "service.search_parsed_source", positional=("source_id",), cache="read",
    ),
    _spec(
        "get-llm-summary", "Get or generate paper summaries.", _paper_schema(_SUMMARY_OPTIONS),
        "service.get_llm_summary", positional=("paper_ids",), aliases=("llm-summary",),
        network="may", cache="read-write", llm=True,
    ),
    _spec(
        "get-cached-llm-summary", "Read cached paper summaries without invoking an LLM.",
        _object({"paper_ids": PAPER_IDS}, required=("paper_ids",)), "service.get_cached_llm_summary",
        positional=("paper_ids",), cache="read",
    ),
    _spec(
        "generate-llm-summary", "Generate paper summaries with an LLM.", _paper_schema(_SUMMARY_OPTIONS),
        "service.generate_llm_summary", positional=("paper_ids",), aliases=("llm-generate-summary",),
        network="may", cache="read-write", llm=True,
    ),
    _spec(
        "store-llm-summary", "Validate and store a paper summary.",
        _object({"paper_id": NONEMPTY, "summary": {"type": "object"}}, required=("paper_id", "summary")),
        "service.store_llm_summary", positional=("paper_id", "summary"), cache="write",
    ),
    _spec(
        "cache.list", "List cached paper artifacts.",
        _object({
            "ids": _nullable(_array(NONEMPTY, min_items=1)),
            "since": OPTIONAL_STRING,
            "older_than": OPTIONAL_STRING,
        }),
        "service.list_cached_papers", cache="read", admin=True,
    ),
    _spec(
        "cache.remove", "Preview or remove selected cached paper artifacts.",
        _object({
            "ids": _nullable(_array(NONEMPTY, min_items=1)),
            "since": OPTIONAL_STRING,
            "older_than": OPTIONAL_STRING,
            "all_items": {"type": "boolean"},
            "dry_run": {"type": "boolean", "default": True},
        }),
        "service.remove_cached_papers", cache="read-write", destructive=True, admin=True,
    ),
    _spec(
        "doctor.cache", "Inspect ARC-paper cache configuration.",
        _object({"paper_id": OPTIONAL_STRING}), "service.doctor_cache",
        positional=("paper_id",), cache="read", admin=True,
    ),
    _spec("doctor.host", "Detect the current agent host.", _object(), "internal.doctor_host", admin=True),
    _spec("doctor.provider", "Resolve the current LLM provider.", _object(), "internal.doctor_provider", admin=True),
    _spec(
        "summary-batch.create", "Create a summary batch.",
        _object({"name": NONEMPTY, "paper_ids": _array(NONEMPTY, min_items=1)}, required=("name", "paper_ids")),
        "internal.batch_create", job=True, cache="write", destructive=True, admin=True,
    ),
    *(
        _spec(name, description, _object({"name": NONEMPTY}, required=("name",)), executor, job=True, cache=cache)
        for name, description, executor, cache in (
            ("summary-batch.status", "Read summary batch status.", "internal.batch_status", "read"),
            (
                "summary-batch.retry-failed", "Requeue failed summary batch items.",
                "internal.batch_retry", "read-write",
            ),
        )
    ),
    _spec(
        "summary-batch.prefetch", "Prefetch inputs for a summary batch.",
        _object({"name": NONEMPTY, "workers": {"type": "integer", "minimum": 1}}, required=("name",)),
        "internal.batch_prefetch", job=True, network="may", cache="read-write",
    ),
    _spec(
        "summary-batch.run", "Run queued summary generation work.",
        _object(
            {
                "name": NONEMPTY, "provider": _string(), "model": OPTIONAL_STRING,
                "model_tier": MODEL_TIER,
                "concurrency": {"type": "integer", "minimum": 1},
                "max_items": _nullable({"type": "integer", "minimum": 0}),
            },
            required=("name",),
        ),
        "internal.batch_run", job=True, network="may", cache="read-write", llm=True,
    ),
    _spec(
        "summary-batch.export", "Export completed summary batch results.",
        _object(
            {
                "name": NONEMPTY,
                "output": ARTIFACT_HANDLE,
                "format": _string(enum=["jsonl"]),
            },
            required=("name", "output"),
        ),
        "internal.batch_export", job=True, cache="read", artifacts=(("output", "write"),),
    ),
)


def _catalog_by_name(specs: tuple[OperationSpec, ...]) -> Mapping[str, OperationSpec]:
    catalog = {spec.name: spec for spec in specs}
    if len(catalog) != len(specs):
        raise RuntimeError("ARC-paper capability catalog contains duplicate operation names")
    return MappingProxyType(catalog)


def _catalog_aliases(specs: tuple[OperationSpec, ...]) -> Mapping[str, str]:
    canonical = {spec.name for spec in specs}
    aliases: dict[str, str] = {}
    for spec in specs:
        for alias in spec.aliases:
            if alias in canonical or alias in aliases:
                raise RuntimeError(f"ARC-paper capability catalog contains duplicate alias {alias!r}")
            aliases[alias] = spec.name
    return MappingProxyType(aliases)


OPERATION_CATALOG: Mapping[str, OperationSpec] = _catalog_by_name(_OPERATIONS)
_ALIASES: Mapping[str, str] = _catalog_aliases(_OPERATIONS)


def get_operation_spec(operation: str) -> OperationSpec | None:
    """Resolve a canonical name or declared alias to its operation spec."""

    canonical = _ALIASES.get(operation, operation)
    return OPERATION_CATALOG.get(canonical)


def operation_name_from_argv(argv: list[str]) -> str | None:
    """Identify a known CLI operation without interpreting arbitrary argv."""

    if not argv:
        return None
    name = argv[0]
    if name in {"cache", "doctor", "summary-batch"}:
        if len(argv) < 2 or argv[1].startswith("-"):
            return None
        name = f"{name}.{argv[1]}"
    spec = get_operation_spec(name)
    return spec.name if spec is not None else None


def operation_capabilities(operation: str) -> frozenset[str]:
    spec = get_operation_spec(operation)
    return spec.capabilities if spec is not None else frozenset()


def catalog_document() -> dict[str, Any]:
    """Return a deterministic JSON-compatible catalog for controllers."""

    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "operations": [OPERATION_CATALOG[name].as_dict() for name in sorted(OPERATION_CATALOG)],
    }


def validate_operation_parameters(operation: str, parameters: Mapping[str, Any]) -> list[str]:
    spec = get_operation_spec(operation)
    if spec is None:
        return [f"Unknown ARC-paper operation {operation!r}"]
    validator = Draft202012Validator(spec.parameter_schema)
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(
            validator.iter_errors(dict(parameters)),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def dispatch_operation(
    operation: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    artifact_resolver: ArtifactResolver | None = None,
) -> dict[str, Any]:
    """Validate and execute one catalog operation.

    Validation failures use stable ARC result envelopes.  Executor exceptions
    are intentionally not hidden so the calling host can apply its own retry
    and supervision policy.
    """

    spec = get_operation_spec(operation)
    if spec is None:
        return err("paper_operation_unknown", f"Unknown ARC-paper operation {operation!r}")
    if parameters is None:
        supplied: Mapping[str, Any] = {}
    elif not isinstance(parameters, Mapping):
        return err("paper_operation_parameters_invalid", "Operation parameters must be an object.")
    else:
        supplied = parameters
    validation_errors = validate_operation_parameters(spec.name, supplied)
    if validation_errors:
        return err(
            "paper_operation_parameters_invalid",
            f"Invalid parameters for ARC-paper operation {spec.name!r}.",
            validation_errors=validation_errors,
        )
    values = dict(supplied)
    for parameter, access in spec.artifact_parameters:
        handle = values.get(parameter)
        if handle is None:
            continue
        if artifact_resolver is None:
            return err(
                "paper_artifact_resolver_required",
                f"ARC-paper operation {spec.name!r} requires a Controller-issued artifact handle.",
                parameter=parameter,
            )
        resolved = artifact_resolver(
            handle["handle_id"],
            access=access,
            operation=spec.name,
            parameter=parameter,
        )
        values[parameter] = str(resolved)
    positional = [values.pop(name, None) for name in spec.positional_parameters]
    executor = _resolve_executor(spec.executor)
    result = executor(*positional, **values)
    if not isinstance(result, dict):
        raise TypeError(f"ARC-paper operation {spec.name!r} returned a non-object result")
    return result


def _resolve_executor(executor: str):
    namespace, name = executor.split(".", 1)
    if namespace == "service":
        return getattr(import_module("arc_paper.service"), name)
    if namespace == "internal":
        return globals()[f"_{name}"]
    raise RuntimeError(f"Unsupported ARC-paper executor namespace {namespace!r}")


def _doctor_host() -> dict[str, Any]:
    detected = import_module("arc_paper.host").detect_host()
    return ok({"host": detected.host, "confidence": detected.confidence, "signals": detected.signals})


def _doctor_provider() -> dict[str, Any]:
    selected = import_module("arc_paper.host").select_llm_provider()
    return ok({
        "provider": selected.provider,
        "host": selected.host.host,
        "confidence": selected.host.confidence,
        "signals": selected.signals,
    })


def _batch_create(*, name: str, paper_ids: list[str]) -> dict[str, Any]:
    db = import_module("arc_paper.batch.db").BatchDB.default()
    db.create_batch(name, paper_ids, "paper-summary-v1")
    return ok({"batch": name, "counts": db.status_counts(name)})


def _batch_status(*, name: str) -> dict[str, Any]:
    db = import_module("arc_paper.batch.db").BatchDB.default()
    return ok({"batch": name, "counts": db.status_counts(name)})


def _batch_retry(*, name: str) -> dict[str, Any]:
    db = import_module("arc_paper.batch.db").BatchDB.default()
    db.retry_failed(name)
    return ok({"batch": name, "counts": db.status_counts(name)})


def _batch_prefetch(*, name: str, workers: int = 4) -> dict[str, Any]:
    runner = import_module("arc_paper.batch.runner")
    return ok(runner.prefetch_batch(name, workers=workers))


def _batch_run(
    *, name: str, provider: str = "auto", model: str | None = None,
    model_tier: str | None = None, concurrency: int = 1, max_items: int | None = None,
) -> dict[str, Any]:
    runner = import_module("arc_paper.batch.runner")
    return ok(runner.run_batch(
        name,
        provider=provider,
        model=model,
        model_tier=model_tier,
        concurrency=concurrency,
        max_items=max_items,
    ))


def _batch_export(*, name: str, output: str, format: str = "jsonl") -> dict[str, Any]:
    del format  # The closed schema currently permits only the runner's JSONL format.
    runner = import_module("arc_paper.batch.runner")
    return ok(runner.export_batch(name, output=Path(output)))
