"""Typed operation registry and conservative resolver projection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ac_document import (
    DEFAULT_EXCLUDED_EFFECTS,
    JsonCodec,
    JsonOutputCodec,
    OperationEffect,
    OperationRequestError,
    OperationSpec,
    object_schema,
    operation_registry_document,
    operation_spec,
    registry_mapping,
    resolve_operation_specs,
    to_json_value,
)

from . import service
from .cached_document import cached_document_ref_from_document
from .document_access import DocumentTarget
from .document_structure import (
    cached_document_structure_ref_from_document,
)
from .reference_cache import cached_resource_ref_from_document


REGISTRY_SCHEMA_VERSION = "arc.paper.operation_registry.v1"


def _object(
    properties: Mapping[str, Any],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return object_schema(properties, required=required)


def _spec(
    name: str,
    schema: Mapping[str, Any],
    callable: Callable[..., Any],
    *,
    output_schema: Mapping[str, Any],
    effects: frozenset[OperationEffect] = frozenset(),
    version: int = 1,
    decoder: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> OperationSpec[Any]:
    return operation_spec(
        package_name="arc-paper",
        schema_namespace="arc.paper",
        name=name,
        schema=schema,
        callable=callable,
        output_schema=output_schema,
        effects=effects,
        version=version,
        decoder=decoder,
    )


_PAPER = {"paper_id": {"type": "string", "minLength": 1}}
_ARXIV = {"arxiv_id": {"type": "string", "minLength": 1}}
_REFRESH = {"refresh": {"type": "boolean", "default": False}}
_NETWORK_CACHE = frozenset(
    {OperationEffect.NETWORK, OperationEffect.CACHE_WRITE}
)

_STRING = {"type": "string"}
_NONEMPTY_STRING = {"type": "string", "minLength": 1}
_INTEGER = {"type": "integer"}
_NULLABLE_INTEGER = {"type": ["integer", "null"]}
_NULLABLE_POSITION = {"type": ["integer", "null"], "minimum": 1}
_STRING_ARRAY = {"type": "array", "items": _STRING}
_SOURCE_ORIGIN_SCHEMA = _object(
    {
        "kind": {"enum": ["local_import", "remote_provider", "repository"]},
        "provider": _STRING,
        "locator": _STRING,
        "metadata": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    required=("kind", "provider", "locator", "metadata"),
)
_SOURCE_ARTIFACT_SCHEMA = _object(
    {
        "source_format": {"enum": ["html", "markdown", "tex", "pdf"]},
        "artifact_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "size": {"type": "integer", "minimum": 0},
        "media_type": _NONEMPTY_STRING,
        "origin": _SOURCE_ORIGIN_SCHEMA,
    },
    required=(
        "source_format",
        "artifact_digest",
        "size",
        "media_type",
        "origin",
    ),
)
_HTML_DEPENDENCY_WARNING_SCHEMA = _object(
    {
        "code": _NONEMPTY_STRING,
        "message": _NONEMPTY_STRING,
        "dependency_ordinal": {"type": ["integer", "null"], "minimum": 0},
        "element": _STRING,
        "attribute": _STRING,
        "authored_target": _STRING,
    },
    required=(
        "code",
        "message",
        "dependency_ordinal",
        "element",
        "attribute",
        "authored_target",
    ),
)
_HTML_DEPENDENCY_SCHEMA = _object(
    {
        "ordinal": {"type": "integer", "minimum": 0},
        "element": {"enum": ["img", "object", "source"]},
        "attribute": {"enum": ["data", "src", "srcset"]},
        "authored_target": _STRING,
        "request_url": _STRING,
        "resolved_url": _STRING,
        "declared_media_type": _STRING,
        "availability": {"enum": ["available", "unavailable"]},
        "media_type": _STRING,
        "artifact_digest": _STRING,
        "size": {"type": "integer", "minimum": 0},
        "error_code": _STRING,
        "error_message": _STRING,
    },
    required=(
        "ordinal",
        "element",
        "attribute",
        "authored_target",
        "request_url",
        "resolved_url",
        "declared_media_type",
        "availability",
        "media_type",
        "artifact_digest",
        "size",
        "error_code",
        "error_message",
    ),
)
_HTML_SOURCE_BUNDLE_SCHEMA = _object(
    {
        "primary": _SOURCE_ARTIFACT_SCHEMA,
        "provider": {"enum": ["arxiv-html", "ar5iv"]},
        "document_url": _NONEMPTY_STRING,
        "base_url": _NONEMPTY_STRING,
        "acquisition_policy": _object(
            {
                "max_dependency_count": {"type": "integer", "minimum": 1},
                "max_dependency_bytes": {"type": "integer", "minimum": 1},
                "max_total_dependency_bytes": {"type": "integer", "minimum": 1},
                "max_redirects": {"type": "integer", "minimum": 0},
            },
            required=(
                "max_dependency_count",
                "max_dependency_bytes",
                "max_total_dependency_bytes",
                "max_redirects",
            ),
        ),
        "dependencies": {"type": "array", "items": _HTML_DEPENDENCY_SCHEMA},
        "warnings": {"type": "array", "items": _HTML_DEPENDENCY_WARNING_SCHEMA},
        "bundle_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    required=(
        "primary",
        "provider",
        "document_url",
        "base_url",
        "acquisition_policy",
        "dependencies",
        "warnings",
        "bundle_digest",
    ),
)
_HTML_BUNDLE_EXPORT_RESOURCE_SCHEMA = _object(
    {
        "ordinal": {"type": "integer", "minimum": 0},
        "authored_target": _NONEMPTY_STRING,
        "path": _NONEMPTY_STRING,
        "artifact_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "media_type": _NONEMPTY_STRING,
        "size": {"type": "integer", "minimum": 0},
    },
    required=(
        "ordinal",
        "authored_target",
        "path",
        "artifact_digest",
        "media_type",
        "size",
    ),
)
_HTML_BUNDLE_EXPORT_SCHEMA = _object(
    {
        "source": _NONEMPTY_STRING,
        "manifest": _NONEMPTY_STRING,
        "bundle_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "resources": {"type": "array", "items": _HTML_BUNDLE_EXPORT_RESOURCE_SCHEMA},
        "warnings": {"type": "array", "items": _HTML_DEPENDENCY_WARNING_SCHEMA},
    },
    required=("source", "manifest", "bundle_digest", "resources", "warnings"),
)
_HTML_ACQUISITION_EXPORT_SCHEMA = _object(
    {
        "source": _NONEMPTY_STRING,
        "manifest": _NONEMPTY_STRING,
        "bundle_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "resources": {"type": "array", "items": _NONEMPTY_STRING},
        "warnings": {"type": "array", "items": {"type": "object"}},
        "export": {"type": "object"},
        "sidecar_status": {"enum": ["replayed", "legacy_fallback"]},
    },
    required=(
        "source",
        "manifest",
        "bundle_digest",
        "resources",
        "warnings",
        "export",
        "sidecar_status",
    ),
)
_IDENTIFIERS_SCHEMA = {
    "type": "object",
    "additionalProperties": {"type": "string"},
}
_METADATA_SCHEMA = _object(
    {
        "paper_id": _STRING,
        "title": _STRING,
        "abstract": _STRING,
        "authors": _STRING_ARRAY,
        "arxiv_id": _STRING,
        "inspire_recid": _STRING,
        "doi": _STRING,
        "dois": _STRING_ARRAY,
        "identifiers": _IDENTIFIERS_SCHEMA,
        "year": _NULLABLE_INTEGER,
        "published": _STRING,
        "citation_count": _INTEGER,
    },
    required=(
        "paper_id",
        "title",
        "abstract",
        "authors",
        "arxiv_id",
        "inspire_recid",
        "doi",
        "dois",
        "identifiers",
        "year",
        "published",
        "citation_count",
    ),
)
_CITER_MATCH_SCHEMA = _object(
    {
        **_METADATA_SCHEMA["properties"],
        "matched_terms": {
            "type": "array",
            "items": _NONEMPTY_STRING,
            "minItems": 1,
        },
        "matched_fields": {
            "type": "array",
            "items": {"enum": ["title", "abstract"]},
            "minItems": 1,
            "uniqueItems": True,
        },
    },
    required=(
        *tuple(_METADATA_SCHEMA["required"]),
        "matched_terms",
        "matched_fields",
    ),
)
_CITER_CONTROL_SCHEMA = _object(
    {
        **_METADATA_SCHEMA["properties"],
        "control_reasons": {
            "type": "array",
            "items": {"enum": ["newest", "most-cited"]},
            "minItems": 1,
            "uniqueItems": True,
        },
    },
    required=(
        *tuple(_METADATA_SCHEMA["required"]),
        "control_reasons",
    ),
)
_CITER_SEARCH_RESULT_SCHEMA = _object(
    {
        "paper_id": _NONEMPTY_STRING,
        "total_citer_count": {"type": "integer", "minimum": 0},
        "scanned_count": {"type": "integer", "minimum": 0},
        "scan_complete": {"type": "boolean"},
        "scan_strategy": {
            "enum": ["all-mostrecent", "split-mostrecent-mostcited"]
        },
        "terms": {
            "type": "array",
            "items": _NONEMPTY_STRING,
            "minItems": 1,
        },
        "matched_count": {"type": "integer", "minimum": 0},
        "returned_count": {"type": "integer", "minimum": 0, "maximum": 50},
        "matches_truncated": {"type": "boolean"},
        "matches": {
            "type": "array",
            "items": _CITER_MATCH_SCHEMA,
            "maxItems": 50,
        },
        "control_sample": {
            "type": "array",
            "items": _CITER_CONTROL_SCHEMA,
            "maxItems": 10,
        },
    },
    required=(
        "paper_id",
        "total_citer_count",
        "scanned_count",
        "scan_complete",
        "scan_strategy",
        "terms",
        "matched_count",
        "returned_count",
        "matches_truncated",
        "matches",
        "control_sample",
    ),
)
_REFERENCE_SCHEMA = _object(
    {
        "paper_id": _STRING,
        "title": _STRING,
        "raw_inspire_reference": {"type": "object"},
        "record_ref": _STRING,
        "publication_info": {"type": ["array", "object", "string"]},
        "abstract": _STRING,
        "authors": _STRING_ARRAY,
        "arxiv_id": _STRING,
        "inspire_recid": _STRING,
        "doi": _STRING,
        "dois": _STRING_ARRAY,
        "identifiers": _IDENTIFIERS_SCHEMA,
        "year": _NULLABLE_INTEGER,
        "published": _STRING,
        "citation_count": _NULLABLE_INTEGER,
        "metadata_enriched": {"type": "boolean"},
        "metadata_enrichment_error": _object(
            {"code": _STRING, "message": _STRING},
            required=("code", "message"),
        ),
    },
    required=("paper_id", "title", "raw_inspire_reference"),
)
_SECTION_SCHEMA = _object(
    {
        "section_id": _NONEMPTY_STRING,
        "title": _STRING,
        "level": {"type": "integer", "minimum": 1},
        "text": _STRING,
        "ordinal": {"type": "integer", "minimum": 0},
        "page_start": _NULLABLE_INTEGER,
        "page_end": _NULLABLE_INTEGER,
    },
    required=(
        "section_id",
        "title",
        "level",
        "text",
        "ordinal",
        "page_start",
        "page_end",
    ),
)
_MATCHED_SENTENCE_SCHEMA = _object(
    {
        "text": {"type": "string", "maxLength": 400},
        "section_id": _NONEMPTY_STRING,
        "page_number": _NULLABLE_INTEGER,
        "matched_surface": _NONEMPTY_STRING,
        "clipped": {"type": "boolean"},
    },
    required=(
        "text",
        "section_id",
        "page_number",
        "matched_surface",
        "clipped",
    ),
)
_KEYWORD_TERM_SCHEMA = _object(
    {
        "term_id": _NONEMPTY_STRING,
        "term": {"type": "string", "minLength": 1, "maxLength": 300},
        "aliases": _STRING_ARRAY,
        "occurrence_count": {"type": "integer", "minimum": 0},
        "source_refs": _STRING_ARRAY,
        "matched_sentences": {
            "type": "array",
            "maxItems": 10,
            "items": _MATCHED_SENTENCE_SCHEMA,
        },
    },
    required=(
        "term_id",
        "term",
        "aliases",
        "occurrence_count",
        "source_refs",
        "matched_sentences",
    ),
)
_KEYWORD_RESULT_SCHEMA = _object(
    {
        "schema_version": {"const": "ac.document.keyword_result.v1"},
        "document_digest": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "source_digest": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "approx_count": {"type": "integer", "minimum": 1, "maximum": 200},
        "planned_count": {"type": "integer", "minimum": 2, "maximum": 300},
        "returned_count": {"type": "integer", "minimum": 0, "maximum": 300},
        "terms": {"type": "array", "maxItems": 300, "items": _KEYWORD_TERM_SCHEMA},
        "inventory_digest": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "warnings": _STRING_ARRAY,
    },
    required=(
        "schema_version",
        "document_digest",
        "source_digest",
        "approx_count",
        "planned_count",
        "returned_count",
        "terms",
        "inventory_digest",
        "warnings",
    ),
)
_TOC_ENTRY_SCHEMA = _object(
    {
        "section_id": _NONEMPTY_STRING,
        "title": _STRING,
        "level": {"type": "integer", "minimum": 1},
        "ordinal": {"type": "integer", "minimum": 0},
        "page_start": _NULLABLE_INTEGER,
        "page_end": _NULLABLE_INTEGER,
    },
    required=(
        "section_id",
        "title",
        "level",
        "ordinal",
        "page_start",
        "page_end",
    ),
)
_MATH_SPAN_SCHEMA = _object(
    {
        "span_id": _NONEMPTY_STRING,
        "kind": {"enum": ["inline", "display"]},
        "source_line_start": _NULLABLE_POSITION,
        "source_column_start": _NULLABLE_POSITION,
        "source_line_end": _NULLABLE_POSITION,
        "source_column_end": _NULLABLE_POSITION,
        "normalized_tex": _NONEMPTY_STRING,
        "context_before": _STRING,
        "context_after": _STRING,
        "source_label": _STRING,
    },
    required=(
        "span_id",
        "kind",
        "source_line_start",
        "source_column_start",
        "source_line_end",
        "source_column_end",
        "normalized_tex",
        "context_before",
        "context_after",
        "source_label",
    ),
)
_PARSED_DOCUMENT_SCHEMA = _object(
    {
        "source": _SOURCE_ARTIFACT_SCHEMA,
        "sections": {"type": "array", "items": _SECTION_SCHEMA},
        "math_spans": {"type": "array", "items": _MATH_SPAN_SCHEMA},
        "pages": {
            "type": "array",
            "items": _object(
                {
                    "page_number": {"type": "integer", "minimum": 1},
                    "text": _STRING,
                },
                required=("page_number", "text"),
            ),
        },
        "warnings": _STRING_ARRAY,
        "metadata": {"type": "object"},
        "schema_version": {"const": "ac.document.parsed_document.v2"},
        "document_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    required=(
        "source",
        "sections",
        "math_spans",
        "pages",
        "warnings",
        "metadata",
        "schema_version",
        "document_digest",
    ),
)
_RECONCILIATION_ENTRY_SCHEMA = _object(
    {
        "validator": _SOURCE_ARTIFACT_SCHEMA,
        "status": {
            "enum": ["verified", "mismatch", "missing", "ambiguous", "unreviewed"]
        },
        "subject_id": _STRING,
        "message": _STRING,
        "provenance": {"type": "object"},
    },
    required=("validator", "status", "subject_id", "message", "provenance"),
)
_PARSE_OUTCOME_SCHEMA = _object(
    {
        "document": _PARSED_DOCUMENT_SCHEMA,
        "report": _object(
            {
                "primary": _SOURCE_ARTIFACT_SCHEMA,
                "policy": {
                    "enum": ["none", "deterministic_only", "visual_all_pages"]
                },
                "entries": {
                    "type": "array",
                    "items": _RECONCILIATION_ENTRY_SCHEMA,
                },
            },
            required=("primary", "policy", "entries"),
        ),
        "warnings": _STRING_ARRAY,
    },
    required=("document", "report", "warnings"),
)
_RICH_DOCUMENT_EXPORT_RESOURCE_SCHEMA = _object(
    {
        "artifact_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "media_type": _NONEMPTY_STRING,
        "logical_name": _NONEMPTY_STRING,
        "size": {"type": "integer", "minimum": 0},
        "path": _NONEMPTY_STRING,
    },
    required=(
        "artifact_digest",
        "media_type",
        "logical_name",
        "size",
        "path",
    ),
)
_RICH_DOCUMENT_EXPORT_SCHEMA = _object(
    {
        "source": _NONEMPTY_STRING,
        "metadata": _NONEMPTY_STRING,
        "resources": {
            "type": "array",
            "items": _RICH_DOCUMENT_EXPORT_RESOURCE_SCHEMA,
        },
        "document_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "warnings": _STRING_ARRAY,
    },
    required=(
        "source",
        "metadata",
        "resources",
        "document_digest",
        "warnings",
    ),
)
_ARXIV_PROVENANCE_SCHEMA = _object(
    {
        "canonical_arxiv_id": {
            "type": "string",
            "pattern": "^arXiv:",
        },
        "provider": {"enum": ["arxiv-html", "ar5iv"]},
        "source_format": {"const": "html"},
        "source_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "document_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    required=(
        "canonical_arxiv_id",
        "provider",
        "source_format",
        "source_digest",
        "document_digest",
    ),
)
_FULL_TEXT_MATCH_SCHEMA = _object(
    {
        "document_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "source_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "location": {"enum": ["section", "page"]},
        "location_id": _NONEMPTY_STRING,
        "title": _STRING,
        "ordinal": {"type": "integer", "minimum": 0},
        "page_number": _NULLABLE_INTEGER,
        "matched_in": _NONEMPTY_STRING,
        "snippet": _STRING,
    },
    required=(
        "document_digest",
        "source_digest",
        "location",
        "location_id",
        "title",
        "ordinal",
        "page_number",
        "matched_in",
        "snippet",
    ),
)
_CACHED_DOCUMENT_REF_SCHEMA = _object(
    {
        "source_format": {"enum": ["html", "markdown", "tex", "pdf"]},
        "source_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "source_size": {"type": "integer", "minimum": 0},
        "media_type": _NONEMPTY_STRING,
        "parser_contract": _NONEMPTY_STRING,
        "parsed_document_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    },
    required=(
        "source_format",
        "source_sha256",
        "source_size",
        "media_type",
        "parser_contract",
        "parsed_document_sha256",
    ),
)
_CACHED_DOCUMENT_INPUT = {
    "document": _CACHED_DOCUMENT_REF_SCHEMA,
    "cache_root": {"type": ["string", "null"]},
}
_DOCUMENT_TARGET_SCHEMA = {
    "oneOf": [
        _object(
            {
                "kind": {"const": "reference"},
                "reference": _NONEMPTY_STRING,
            },
            required=("kind", "reference"),
        ),
        _object(
            {
                "kind": {"const": "document"},
                "document": _CACHED_DOCUMENT_REF_SCHEMA,
            },
            required=("kind", "document"),
        ),
    ]
}
_DOCUMENT_TARGET_RESULT_SCHEMA = _object(
    {
        "kind": {"enum": ["reference", "document"]},
        "reference": _STRING,
        "document": {"anyOf": [_CACHED_DOCUMENT_REF_SCHEMA, {"type": "null"}]},
    },
    required=("kind", "reference", "document"),
)
_REFERENCE_IDENTITY_INPUT = {
    "doi": {"type": ["string", "null"], "minLength": 1},
    "arxiv_id": {"type": ["string", "null"], "minLength": 1},
    "url": {"type": ["string", "null"], "minLength": 1},
    "title": {"type": ["string", "null"], "minLength": 1},
}
_CACHED_DOCUMENT_STRUCTURE_REF_SCHEMA = _object(
    {
        "document": _CACHED_DOCUMENT_REF_SCHEMA,
        "outline_document": _CACHED_DOCUMENT_REF_SCHEMA,
        "structure_contract": _NONEMPTY_STRING,
        "structure_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    },
    required=(
        "document",
        "outline_document",
        "structure_contract",
        "structure_sha256",
    ),
)
_REFERENCE_IDENTITY_RESULT_SCHEMA = _object(
    {
        "arxiv_id": _STRING,
        "dois": _STRING_ARRAY,
        "urls": _STRING_ARRAY,
        "title": _STRING,
        "inspire_recid": _STRING,
    },
    required=("arxiv_id", "dois", "urls", "title", "inspire_recid"),
)
_CACHED_RESOURCE_REF_SCHEMA = _object(
    {
        "resource_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "resource_size": {"type": "integer", "minimum": 0},
        "media_type": _NONEMPTY_STRING,
        "source_locator": _STRING,
        "filename": _STRING,
    },
    required=(
        "resource_sha256",
        "resource_size",
        "media_type",
        "source_locator",
        "filename",
    ),
)
_CACHED_REFERENCE_MATERIAL_RESULT_SCHEMA = _object(
    {
        "identity": _REFERENCE_IDENTITY_RESULT_SCHEMA,
        "resources": {
            "type": "array",
            "items": _CACHED_RESOURCE_REF_SCHEMA,
            "minItems": 1,
        },
        "readable_resource": {
            "anyOf": [_CACHED_RESOURCE_REF_SCHEMA, {"type": "null"}]
        },
    },
    required=("identity", "resources", "readable_resource"),
)
_RESOLVED_DOCUMENT_INFO_SCHEMA = _object(
    {
        "document": _CACHED_DOCUMENT_REF_SCHEMA,
        "identity": {
            "anyOf": [_REFERENCE_IDENTITY_RESULT_SCHEMA, {"type": "null"}]
        },
        "requested_reference": _STRING,
        "warnings": _STRING_ARRAY,
    },
    required=("document", "identity", "requested_reference", "warnings"),
)
_PAPER_TOC_SCHEMA = _object(
    {
        "source": _RESOLVED_DOCUMENT_INFO_SCHEMA,
        "entries": {"type": "array", "items": _TOC_ENTRY_SCHEMA},
        "warnings": _STRING_ARRAY,
    },
    required=("source", "entries", "warnings"),
)
_PAPER_SECTION_SCHEMA = _object(
    {
        "source": _RESOLVED_DOCUMENT_INFO_SCHEMA,
        "section_id": _NONEMPTY_STRING,
        "title": _STRING,
        "text": _STRING,
        "level": {"type": "integer", "minimum": 1},
        "ordinal": {"type": "integer", "minimum": 0},
        "page_start": _NULLABLE_INTEGER,
        "page_end": _NULLABLE_INTEGER,
        "warnings": _STRING_ARRAY,
    },
    required=(
        "source",
        "section_id",
        "title",
        "text",
        "level",
        "ordinal",
        "page_start",
        "page_end",
        "warnings",
    ),
)
_DOCUMENT_TARGET_FAILURE_SCHEMA = _object(
    {
        "target_index": {"type": "integer", "minimum": 0},
        "target": _DOCUMENT_TARGET_RESULT_SCHEMA,
        "code": _NONEMPTY_STRING,
        "message": _STRING,
    },
    required=("target_index", "target", "code", "message"),
)
_RESOLVED_SEARCH_DOCUMENT_SCHEMA = _object(
    {
        "source": _RESOLVED_DOCUMENT_INFO_SCHEMA,
        "target_indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "uniqueItems": True,
        },
    },
    required=("source", "target_indices"),
)
_CACHED_DOCUMENT_TOC_SCHEMA = _object(
    {
        "document": _CACHED_DOCUMENT_REF_SCHEMA,
        "entries": {"type": "array", "items": _TOC_ENTRY_SCHEMA},
        "warnings": _STRING_ARRAY,
    },
    required=("document", "entries", "warnings"),
)
_CACHED_DOCUMENT_SECTION_SCHEMA = _object(
    {
        "document": _CACHED_DOCUMENT_REF_SCHEMA,
        **_SECTION_SCHEMA["properties"],
        "warnings": _STRING_ARRAY,
    },
    required=(
        "document",
        *tuple(_SECTION_SCHEMA["required"]),
        "warnings",
    ),
)
_CACHED_SOURCE_RANGE_SCHEMA = _object(
    {
        "document": _CACHED_DOCUMENT_REF_SCHEMA,
        "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1},
        "total_lines": {"type": "integer", "minimum": 0},
        "text": _STRING,
    },
    required=("document", "start_line", "end_line", "total_lines", "text"),
)
_CACHED_DOCUMENT_SEARCH_SCHEMA = _object(
    {
        "document": _CACHED_DOCUMENT_REF_SCHEMA,
        "query": _NONEMPTY_STRING,
        "matches": {"type": "array", "items": _FULL_TEXT_MATCH_SCHEMA},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        "context_lines": {"type": "integer", "minimum": 0, "maximum": 5},
        "case_sensitive": {"type": "boolean"},
        "truncated": {"type": "boolean"},
        "warnings": _STRING_ARRAY,
    },
    required=(
        "document",
        "query",
        "matches",
        "limit",
        "context_lines",
        "case_sensitive",
        "truncated",
        "warnings",
    ),
)
_CACHED_FULL_TEXT_OCCURRENCE_SCHEMA = _object(
    {
        "source_kind": {"enum": ["arxiv", "local", "target"]},
        "arxiv_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": "^arXiv:"},
        },
        "source_format": {"enum": ["html", "markdown", "tex", "pdf"]},
        "source_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "document_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "location": {"enum": ["section", "page"]},
        "location_id": _NONEMPTY_STRING,
        "title": _STRING,
        "page_number": _NULLABLE_INTEGER,
        "line": {"type": "integer", "minimum": 1},
        "column": {"type": "integer", "minimum": 1},
        "matched_terms": {
            "type": "array",
            "items": _NONEMPTY_STRING,
            "minItems": 1,
        },
        "context": {"type": "string", "maxLength": 400},
    },
    required=(
        "source_kind",
        "arxiv_ids",
        "source_format",
        "source_digest",
        "document_digest",
        "location",
        "location_id",
        "title",
        "page_number",
        "line",
        "column",
        "matched_terms",
        "context",
    ),
)
_EQUATION_MATCH_SCHEMA = _object(
    {
        "document_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "source_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "span_id": _NONEMPTY_STRING,
        "kind": {"enum": ["inline", "display"]},
        "normalized_tex": _NONEMPTY_STRING,
        "source_label": _STRING,
        "source_line_start": _NULLABLE_POSITION,
        "source_column_start": _NULLABLE_POSITION,
        "source_line_end": _NULLABLE_POSITION,
        "source_column_end": _NULLABLE_POSITION,
        "context_before": _STRING,
        "context_after": _STRING,
        "matched_in": _NONEMPTY_STRING,
        "matched_terms": {"type": "array", "items": _NONEMPTY_STRING},
        "matched_fields": {"type": "array", "items": _NONEMPTY_STRING},
        "page_candidates": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
        "source_excerpt": _STRING,
    },
    required=(
        "document_digest",
        "source_digest",
        "span_id",
        "kind",
        "normalized_tex",
        "source_label",
        "source_line_start",
        "source_column_start",
        "source_line_end",
        "source_column_end",
        "context_before",
        "context_after",
        "matched_in",
        "matched_terms",
        "matched_fields",
        "page_candidates",
        "source_excerpt",
    ),
)
_PAPER_FULL_TEXT_SEARCH_SCHEMA = _object(
    {
        "scope": {"enum": ["corpus", "targets"]},
        "mode": {"enum": ["occurrences", "refinement_required"]},
        "terms": {"type": "array", "items": _NONEMPTY_STRING, "minItems": 1},
        "limit": {"type": "integer", "minimum": 1, "maximum": 500},
        "context_lines": {"type": "integer", "minimum": 0, "maximum": 2},
        "case_sensitive": {"type": "boolean"},
        "total_occurrences": {"type": "integer", "minimum": 0},
        "matched_document_count": {"type": "integer", "minimum": 0},
        "documents": {"type": "array", "items": _RESOLVED_SEARCH_DOCUMENT_SCHEMA},
        "failures": {"type": "array", "items": _DOCUMENT_TARGET_FAILURE_SCHEMA},
        "occurrences": {"type": "array", "items": _CACHED_FULL_TEXT_OCCURRENCE_SCHEMA},
        "top_paper_titles": {"type": "array", "items": _NONEMPTY_STRING, "maxItems": 50},
        "context_status": {
            "enum": [
                "not_requested",
                "included",
                "omitted_too_broad",
                "omitted_refinement_required",
            ]
        },
        "message": _NONEMPTY_STRING,
        "warnings": _STRING_ARRAY,
    },
    required=(
        "scope",
        "mode",
        "terms",
        "limit",
        "context_lines",
        "case_sensitive",
        "total_occurrences",
        "matched_document_count",
        "documents",
        "failures",
        "occurrences",
        "top_paper_titles",
        "context_status",
        "message",
        "warnings",
    ),
)
_PAPER_EQUATION_SEARCH_SCHEMA = _object(
    {
        "terms": {"type": "array", "items": _NONEMPTY_STRING, "minItems": 1},
        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
        "case_sensitive": {"type": "boolean"},
        "searched_document_count": {"type": "integer", "minimum": 1},
        "documents": {"type": "array", "items": _RESOLVED_SEARCH_DOCUMENT_SCHEMA, "minItems": 1},
        "failures": {"type": "array", "items": _DOCUMENT_TARGET_FAILURE_SCHEMA},
        "matches": {"type": "array", "items": _EQUATION_MATCH_SCHEMA},
        "truncated": {"type": "boolean"},
        "warnings": _STRING_ARRAY,
    },
    required=(
        "terms",
        "limit",
        "context_lines",
        "case_sensitive",
        "searched_document_count",
        "documents",
        "failures",
        "matches",
        "truncated",
        "warnings",
    ),
)
_CACHE_LOCAL_IDENTITY_SCHEMA = _object(
    {
        "source_format": {"enum": ["html", "markdown", "tex", "pdf"]},
        "media_type": _NONEMPTY_STRING,
        "artifact_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "size": {"type": "integer", "minimum": 0},
    },
    required=("source_format", "media_type", "artifact_digest", "size"),
)
_CACHE_COMPONENT_SCHEMA = _object(
    {
        "name": _NONEMPTY_STRING,
        "cached_at": _NONEMPTY_STRING,
        "storage_entry_ids": {"type": "array", "items": _NONEMPTY_STRING},
    },
    required=("name", "cached_at", "storage_entry_ids"),
)
_CACHE_ENTRY_SCHEMA = _object(
    {
        "entry_id": _NONEMPTY_STRING,
        "kind": {"enum": ["paper", "local"]},
        "paper_id": {"type": ["string", "null"]},
        "local_source_identity": {
            "anyOf": [_CACHE_LOCAL_IDENTITY_SCHEMA, {"type": "null"}]
        },
        "components": {
            "type": "array",
            "items": _CACHE_COMPONENT_SCHEMA,
            "minItems": 1,
        },
        "cached_at": _NONEMPTY_STRING,
        "updateable": {"type": "boolean"},
    },
    required=(
        "entry_id",
        "kind",
        "paper_id",
        "local_source_identity",
        "components",
        "cached_at",
        "updateable",
    ),
)
_CACHE_FILTER_INPUT = _object(
    {
        "paper_ids": {"type": "array", "items": _NONEMPTY_STRING},
        "entry_ids": {"type": "array", "items": _NONEMPTY_STRING},
        "since_seconds": {"type": ["integer", "null"], "minimum": 1},
        "cache_root": {"type": ["string", "null"]},
    }
)
_CACHE_ARCHIVE_RESULT_FIELDS = {
    "archive_path": _NONEMPTY_STRING,
    "archive_sha256": _NONEMPTY_STRING,
    "selection_mode": {"enum": ["all", "entries"]},
    "entry_ids": {"type": "array", "items": _NONEMPTY_STRING},
    "total_bytes": {"type": "integer", "minimum": 0},
}


def _decode_cached_document_parameters(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    decoded = dict(value)
    raw_document = decoded.get("document")
    if not isinstance(raw_document, Mapping):
        raise OperationRequestError(
            "invalid_parameters", "document must be an object"
        )
    try:
        decoded["document"] = cached_document_ref_from_document(raw_document)
    except ValueError as exc:
        raise OperationRequestError("invalid_parameters", str(exc)) from exc
    raw_structure = decoded.get("structure")
    if raw_structure is not None:
        if not isinstance(raw_structure, Mapping):
            raise OperationRequestError(
                "invalid_parameters", "structure must be an object"
            )
        try:
            decoded["structure"] = (
                cached_document_structure_ref_from_document(raw_structure)
            )
        except ValueError as exc:
            raise OperationRequestError("invalid_parameters", str(exc)) from exc
    return decoded


def _decode_document_target_parameters(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    decoded = dict(value)
    raw_target = decoded.get("target")
    if not isinstance(raw_target, Mapping):
        raise OperationRequestError("invalid_parameters", "target must be an object")
    try:
        if raw_target.get("kind") == "reference":
            decoded["target"] = DocumentTarget(
                kind="reference", reference=raw_target.get("reference", "")
            )
        else:
            raw_document = raw_target.get("document")
            if not isinstance(raw_document, Mapping):
                raise ValueError("document target requires a document object")
            decoded["target"] = DocumentTarget(
                kind="document",
                document=cached_document_ref_from_document(raw_document),
            )
    except ValueError as exc:
        raise OperationRequestError("invalid_parameters", str(exc)) from exc
    raw_structure = decoded.get("structure")
    if raw_structure is not None:
        if not isinstance(raw_structure, Mapping):
            raise OperationRequestError(
                "invalid_parameters", "structure must be an object"
            )
        try:
            decoded["structure"] = cached_document_structure_ref_from_document(
                raw_structure
            )
        except ValueError as exc:
            raise OperationRequestError("invalid_parameters", str(exc)) from exc
    return decoded


def _decode_document_targets_parameters(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    decoded = dict(value)
    raw_targets = decoded.get("targets", [])
    if not isinstance(raw_targets, list):
        raise OperationRequestError("invalid_parameters", "targets must be an array")
    targets = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, Mapping):
            raise OperationRequestError(
                "invalid_parameters", "each target must be an object"
            )
        target = _decode_document_target_parameters({"target": raw_target})["target"]
        targets.append(target)
    decoded["targets"] = tuple(targets)
    return decoded


def _decode_structure_reconstruction_parameters(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    decoded = _decode_cached_document_parameters(value)
    raw_outline = decoded.pop("outline_document")
    if not isinstance(raw_outline, Mapping):
        raise OperationRequestError(
            "invalid_parameters", "outline_document must be an object"
        )
    try:
        decoded["outline_document"] = cached_document_ref_from_document(
            raw_outline
        )
    except ValueError as exc:
        raise OperationRequestError("invalid_parameters", str(exc)) from exc
    return decoded


def _decode_cached_resource_parameters(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    decoded = dict(value)
    raw_resource = decoded.get("resource")
    if not isinstance(raw_resource, Mapping):
        raise OperationRequestError(
            "invalid_parameters", "resource must be an object"
        )
    try:
        decoded["resource"] = cached_resource_ref_from_document(raw_resource)
    except ValueError as exc:
        raise OperationRequestError("invalid_parameters", str(exc)) from exc
    return decoded


_OPERATIONS = (
    _spec(
        "extract-paper-ids",
        _object({"text": {"type": "string"}}, required=("text",)),
        service.extract_paper_ids,
        output_schema=_STRING_ARRAY,
    ),
    _spec(
        "safe-dir-name",
        _object(
            {
                "ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                }
            },
            required=("ids",),
        ),
        service.paper_ids_safe_dir_name,
        output_schema=_STRING,
    ),
    _spec(
        "get-title",
        _object({**_PAPER, **_REFRESH}, required=("paper_id",)),
        service.get_title,
        output_schema=_STRING,
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "get-abstract",
        _object({**_PAPER, **_REFRESH}, required=("paper_id",)),
        service.get_abstract,
        output_schema=_STRING,
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "get-authors",
        _object({**_PAPER, **_REFRESH}, required=("paper_id",)),
        service.get_authors,
        output_schema=_STRING_ARRAY,
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "get-metadata",
        _object({**_PAPER, **_REFRESH}, required=("paper_id",)),
        service.get_metadata,
        output_schema=_METADATA_SCHEMA,
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "get-citer-count",
        _object({**_PAPER, **_REFRESH}, required=("paper_id",)),
        service.get_citer_count,
        output_schema=_INTEGER,
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "get-references",
        _object(
            {
                **_PAPER,
                **_REFRESH,
                "enrich": {"type": "boolean", "default": False},
            },
            required=("paper_id",),
        ),
        service.get_references,
        output_schema={"type": "array", "items": _REFERENCE_SCHEMA},
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "get-citers",
        _object(
            {
                **_PAPER,
                **_REFRESH,
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                "sort": {"enum": ["mostrecent", "mostcited"]},
            },
            required=("paper_id",),
        ),
        service.get_citers,
        output_schema={"type": "array", "items": _METADATA_SCHEMA},
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "search-citers",
        _object(
            {
                **_PAPER,
                **_REFRESH,
                "terms": {
                    "type": "array",
                    "items": _NONEMPTY_STRING,
                    "minItems": 1,
                    "description": (
                        "Literal OR phrases matched against normalized citer "
                        "titles and abstracts."
                    ),
                },
                "scan_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            required=("paper_id", "terms"),
        ),
        service.search_citers,
        output_schema=_CITER_SEARCH_RESULT_SCHEMA,
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "search-metadata",
        _object(
            {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            required=("query",),
        ),
        service.search_metadata,
        output_schema={"type": "array", "items": _METADATA_SCHEMA},
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "search-full-text",
        _object(
            {
                "targets": {"type": "array", "items": _DOCUMENT_TARGET_SCHEMA},
                "terms": {
                    "type": "array",
                    "items": _NONEMPTY_STRING,
                    "minItems": 1,
                },
                "source_format": {
                    "type": ["string", "null"],
                    "enum": ["html", "markdown", "tex", "pdf", None],
                },
                "refresh": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                "context_lines": {"type": "integer", "minimum": 0, "maximum": 2},
                "case_sensitive": {"type": "boolean", "default": False},
                "cache_root": {"type": ["string", "null"]},
            },
            required=("terms",),
        ),
        service.search_full_text_cli,
        output_schema=_PAPER_FULL_TEXT_SEARCH_SCHEMA,
        effects=_NETWORK_CACHE,
        decoder=_decode_document_targets_parameters,
    ),
    _spec(
        "search-equations",
        _object(
            {
                "targets": {
                    "type": "array",
                    "items": _DOCUMENT_TARGET_SCHEMA,
                    "minItems": 1,
                },
                "terms": {"type": "array", "items": _NONEMPTY_STRING, "minItems": 1},
                "source_format": {
                    "type": ["string", "null"],
                    "enum": ["html", "markdown", "tex", "pdf", None],
                },
                "refresh": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
                "case_sensitive": {"type": "boolean", "default": False},
                "cache_root": {"type": ["string", "null"]},
            },
            required=("targets", "terms"),
        ),
        service.search_equations_cli,
        output_schema=_PAPER_EQUATION_SEARCH_SCHEMA,
        effects=_NETWORK_CACHE,
        decoder=_decode_document_targets_parameters,
    ),
    _spec(
        "reconstruct-cached-structure",
        _object(
            {
                **_CACHED_DOCUMENT_INPUT,
                "outline_document": _CACHED_DOCUMENT_REF_SCHEMA,
            },
            required=("document", "outline_document"),
        ),
        service.reconstruct_cached_structure,
        output_schema=_CACHED_DOCUMENT_STRUCTURE_REF_SCHEMA,
        effects=frozenset({OperationEffect.CACHE_WRITE}),
        decoder=_decode_structure_reconstruction_parameters,
    ),
    _spec(
        "get-table-of-contents",
        _object(
            {
                "target": _DOCUMENT_TARGET_SCHEMA,
                "structure": {
                    "anyOf": [_CACHED_DOCUMENT_STRUCTURE_REF_SCHEMA, {"type": "null"}]
                },
                "source_format": {
                    "type": ["string", "null"],
                    "enum": ["html", "markdown", "tex", "pdf", None],
                },
                "refresh": {"type": "boolean", "default": False},
                "cache_root": {"type": ["string", "null"]},
            },
            required=("target",),
        ),
        service.get_table_of_contents_cli,
        output_schema=_PAPER_TOC_SCHEMA,
        effects=_NETWORK_CACHE,
        decoder=_decode_document_target_parameters,
    ),
    _spec(
        "get-section",
        _object(
            {
                "target": _DOCUMENT_TARGET_SCHEMA,
                "structure": {
                    "anyOf": [_CACHED_DOCUMENT_STRUCTURE_REF_SCHEMA, {"type": "null"}]
                },
                "selector": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "integer", "minimum": 0},
                    ]
                },
                "source_format": {
                    "type": ["string", "null"],
                    "enum": ["html", "markdown", "tex", "pdf", None],
                },
                "refresh": {"type": "boolean", "default": False},
                "cache_root": {"type": ["string", "null"]},
            },
            required=("target", "selector"),
        ),
        service.get_section_cli,
        output_schema=_PAPER_SECTION_SCHEMA,
        effects=_NETWORK_CACHE,
        decoder=_decode_document_target_parameters,
    ),
    _spec(
        "lookup-reference",
        _object(
            {
                **_REFERENCE_IDENTITY_INPUT,
                "cache_root": {"type": ["string", "null"]},
            }
        ),
        service.lookup_reference_cli,
        output_schema={
            "anyOf": [
                _CACHED_REFERENCE_MATERIAL_RESULT_SCHEMA,
                {"type": "null"},
            ]
        },
    ),
    _spec(
        "acquire-reference",
        _object(
            {
                **{
                    key: value
                    for key, value in _REFERENCE_IDENTITY_INPUT.items()
                    if key != "title"
                },
                "refresh": {"type": "boolean", "default": False},
                "cache_root": {"type": ["string", "null"]},
            }
        ),
        service.acquire_reference_cli,
        output_schema=_CACHED_REFERENCE_MATERIAL_RESULT_SCHEMA,
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "admit-reference",
        _object(
            {
                "path": _NONEMPTY_STRING,
                **_REFERENCE_IDENTITY_INPUT,
                "media_type": {"type": ["string", "null"]},
                "cache_root": {"type": ["string", "null"]},
            },
            required=("path",),
        ),
        service.admit_reference_cli,
        output_schema=_CACHED_REFERENCE_MATERIAL_RESULT_SCHEMA,
        effects=frozenset(
            {OperationEffect.ARBITRARY_LOCAL_PATH, OperationEffect.CACHE_WRITE}
        ),
    ),
    _spec(
        "materialize-reference",
        _object(
            {
                "resource": _CACHED_RESOURCE_REF_SCHEMA,
                "output": _NONEMPTY_STRING,
                "cache_root": {"type": ["string", "null"]},
            },
            required=("resource", "output"),
        ),
        service.materialize_reference_cli,
        output_schema=_object(
            {
                "resource": _CACHED_RESOURCE_REF_SCHEMA,
                "output": _NONEMPTY_STRING,
                "bytes_written": {"type": "integer", "minimum": 0},
            },
            required=("resource", "output", "bytes_written"),
        ),
        effects=frozenset({OperationEffect.ARBITRARY_LOCAL_PATH}),
        decoder=_decode_cached_resource_parameters,
    ),
    _spec(
        "read-cached-source-range",
        _object(
            {
                **_CACHED_DOCUMENT_INPUT,
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "text_only": {"type": "boolean", "default": False},
            },
            required=("document", "start_line", "end_line"),
        ),
        service.read_cached_source_range,
        output_schema=_CACHED_SOURCE_RANGE_SCHEMA,
        effects=frozenset({OperationEffect.CACHE_WRITE}),
        version=2,
        decoder=_decode_cached_document_parameters,
    ),
    _spec(
        "fetch-arxiv-auto",
        _object(
            {
                **_PAPER,
                **_REFRESH,
                "cache_root": {"type": ["string", "null"]},
            },
            required=("paper_id",),
        ),
        service.fetch_arxiv_auto,
        output_schema=_SOURCE_ARTIFACT_SCHEMA,
        effects=_NETWORK_CACHE,
        version=2,
    ),
    _spec(
        "fetch-arxiv-pdf",
        _object(
            {
                **_PAPER,
                **_REFRESH,
                "cache_root": {"type": ["string", "null"]},
            },
            required=("paper_id",),
        ),
        service.fetch_arxiv_pdf,
        output_schema=_SOURCE_ARTIFACT_SCHEMA,
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "fetch-arxiv-html-bundle",
        _object(
            {
                **_PAPER,
                **_REFRESH,
                "cache_root": {"type": ["string", "null"]},
            },
            required=("paper_id",),
        ),
        service.fetch_arxiv_html_bundle,
        output_schema=_HTML_SOURCE_BUNDLE_SCHEMA,
        effects=_NETWORK_CACHE,
    ),
    _spec(
        "export-arxiv-html-bundle",
        _object(
            {
                **_PAPER,
                **_REFRESH,
                "output_dir": _NONEMPTY_STRING,
                "cache_root": {"type": ["string", "null"]},
            },
            required=("paper_id", "output_dir"),
        ),
        service.export_arxiv_html_bundle,
        output_schema=_HTML_BUNDLE_EXPORT_SCHEMA,
        effects=frozenset(
            {
                OperationEffect.NETWORK,
                OperationEffect.CACHE_WRITE,
                OperationEffect.ARBITRARY_LOCAL_PATH,
            }
        ),
    ),
    _spec(
        "export-arxiv-html-acquisition",
        _object(
            {
                **_PAPER,
                **_REFRESH,
                "output_dir": _NONEMPTY_STRING,
                "cache_root": {"type": ["string", "null"]},
            },
            required=("paper_id", "output_dir"),
        ),
        service.export_arxiv_html_acquisition,
        output_schema=_HTML_ACQUISITION_EXPORT_SCHEMA,
        effects=frozenset(
            {
                OperationEffect.NETWORK,
                OperationEffect.CACHE_WRITE,
                OperationEffect.ARBITRARY_LOCAL_PATH,
            }
        ),
    ),
    _spec(
        "import-source",
        _object(
            {
                "path": {"type": "string", "minLength": 1},
                "cache_root": {"type": ["string", "null"]},
                "source_format": {
                    "type": ["string", "null"],
                    "enum": ["html", "markdown", "tex", "pdf", None],
                },
            },
            required=("path",),
        ),
        service.import_source,
        output_schema=_SOURCE_ARTIFACT_SCHEMA,
        effects=frozenset(
            {OperationEffect.CACHE_WRITE, OperationEffect.ARBITRARY_LOCAL_PATH}
        ),
    ),
    _spec(
        "parse-local",
        _object(
            {
                "primary_path": {"type": "string", "minLength": 1},
                "validator_paths": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "cache_root": {"type": ["string", "null"]},
                "primary_format": {
                    "type": ["string", "null"],
                    "enum": ["html", "markdown", "tex", "pdf", None],
                },
                "validator_formats": {
                    "type": "array",
                    "items": {
                        "type": ["string", "null"],
                        "enum": ["html", "markdown", "tex", "pdf", None],
                    },
                },
                "policy": {
                    "type": ["string", "null"],
                    "enum": [
                        "none",
                        "deterministic_only",
                        "visual_all_pages",
                        None,
                    ],
                },
            },
            required=("primary_path",),
        ),
        service.parse_local,
        output_schema=_PARSE_OUTCOME_SCHEMA,
        effects=frozenset(
            {OperationEffect.CACHE_WRITE, OperationEffect.ARBITRARY_LOCAL_PATH}
        ),
        version=2,
    ),
    _spec(
        "export-rich-document",
        _object(
            {
                "source": _NONEMPTY_STRING,
                "output_dir": _NONEMPTY_STRING,
                "validator": {"type": ["string", "null"], "minLength": 1},
                "source_format": {
                    "type": ["string", "null"],
                    "enum": ["html", "markdown", "tex", None],
                },
                "cache_root": {"type": ["string", "null"]},
            },
            required=("source", "output_dir"),
        ),
        service.export_rich_document,
        output_schema=_RICH_DOCUMENT_EXPORT_SCHEMA,
        effects=frozenset(
            {OperationEffect.CACHE_WRITE, OperationEffect.ARBITRARY_LOCAL_PATH}
        ),
    ),
    _spec(
        "extract-keywords",
        _object(
            {
                "source": _NONEMPTY_STRING,
                "project_dir": _NONEMPTY_STRING,
                "approx_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                },
                "cache_root": {"type": ["string", "null"]},
                "refresh": {"type": "boolean", "default": False},
                "llm_provider": _NONEMPTY_STRING,
                "model": {"type": ["string", "null"]},
                "model_tier": {
                    "enum": ["low", "medium", "high", "xhigh"],
                },
                "run_id": {"type": ["string", "null"]},
                "resume_input": {"type": ["object", "null"]},
                "structure_ref": {
                    "anyOf": [
                        _CACHED_DOCUMENT_STRUCTURE_REF_SCHEMA,
                        {"type": "null"},
                    ]
                },
                "section_ids": {
                    "type": ["array", "null"],
                    "items": _NONEMPTY_STRING,
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "host_authority": {
                    "enum": ["unknown", "restricted", "unrestricted"],
                },
            },
            required=("source", "project_dir"),
        ),
        service.extract_keywords,
        output_schema=_KEYWORD_RESULT_SCHEMA,
        effects=frozenset(
            {
                OperationEffect.NETWORK,
                OperationEffect.CACHE_WRITE,
                OperationEffect.ARBITRARY_LOCAL_PATH,
                OperationEffect.RECURSIVE_LLM,
            }
        ),
    ),
    _spec(
        "cache-list",
        _CACHE_FILTER_INPUT,
        service.list_cache,
        output_schema=_object(
            {
                "as_of": _NONEMPTY_STRING,
                "since_seconds": {"type": ["integer", "null"], "minimum": 1},
                "threshold_at": {"type": ["string", "null"]},
                "entries": {"type": "array", "items": _CACHE_ENTRY_SCHEMA},
                "warnings": _STRING_ARRAY,
            },
            required=(
                "as_of",
                "since_seconds",
                "threshold_at",
                "entries",
                "warnings",
            ),
        ),
        effects=frozenset({OperationEffect.CACHE_ADMIN}),
        version=2,
    ),
    _spec(
        "cache-remove",
        _object(
            {
                "paper_ids": {"type": "array", "items": _NONEMPTY_STRING},
                "entry_ids": {"type": "array", "items": _NONEMPTY_STRING},
                "dry_run": {"type": "boolean", "default": True},
                "cache_root": {"type": ["string", "null"]},
            }
        ),
        service.remove_cache,
        output_schema=_object(
            {
                "dry_run": {"type": "boolean"},
                "selected": {"type": "array", "items": _CACHE_ENTRY_SCHEMA},
                "removed_entry_ids": {
                    "type": "array",
                    "items": _NONEMPTY_STRING,
                },
                "warnings": _STRING_ARRAY,
            },
            required=(
                "dry_run",
                "selected",
                "removed_entry_ids",
                "warnings",
            ),
        ),
        effects=frozenset(
            {OperationEffect.CACHE_ADMIN, OperationEffect.DESTRUCTIVE}
        ),
        version=2,
    ),
    _spec(
        "cache-update",
        _object(
            {
                "paper_ids": {"type": "array", "items": _NONEMPTY_STRING},
                "entry_ids": {"type": "array", "items": _NONEMPTY_STRING},
                "cache_root": {"type": ["string", "null"]},
            }
        ),
        service.update_cache,
        output_schema=_object(
            {
                "records": {
                    "type": "array",
                    "items": _object(
                        {
                            "entry_id": _NONEMPTY_STRING,
                            "component": _NONEMPTY_STRING,
                            "status": {"enum": ["updated", "failed", "skipped"]},
                            "message": _STRING,
                        },
                        required=(
                            "entry_id",
                            "component",
                            "status",
                            "message",
                        ),
                    ),
                },
                "warnings": _STRING_ARRAY,
            },
            required=("records", "warnings"),
        ),
        effects=frozenset(
            {
                OperationEffect.CACHE_ADMIN,
                OperationEffect.NETWORK,
                OperationEffect.CACHE_WRITE,
            }
        ),
        version=2,
    ),
    _spec(
        "cache-export",
        _object(
            {
                "output": _NONEMPTY_STRING,
                "entry_ids": {"type": "array", "items": _NONEMPTY_STRING},
                "all_entries": {"type": "boolean", "default": False},
                "cache_root": {"type": ["string", "null"]},
            },
            required=("output",),
        ),
        service.export_cache,
        output_schema=_object(
            {
                **_CACHE_ARCHIVE_RESULT_FIELDS,
                "file_count": {"type": "integer", "minimum": 0},
            },
            required=(*_CACHE_ARCHIVE_RESULT_FIELDS, "file_count"),
        ),
        effects=frozenset(
            {OperationEffect.CACHE_ADMIN, OperationEffect.ARBITRARY_LOCAL_PATH}
        ),
    ),
    _spec(
        "cache-import",
        _object(
            {
                "archive": _NONEMPTY_STRING,
                "replace_conflicts": {"type": "boolean", "default": False},
                "cache_root": {"type": ["string", "null"]},
            },
            required=("archive",),
        ),
        service.import_cache,
        output_schema=_object(
            {
                **_CACHE_ARCHIVE_RESULT_FIELDS,
                "added_count": {"type": "integer", "minimum": 0},
                "reused_count": {"type": "integer", "minimum": 0},
                "replaced_count": {"type": "integer", "minimum": 0},
            },
            required=(
                *_CACHE_ARCHIVE_RESULT_FIELDS,
                "added_count",
                "reused_count",
                "replaced_count",
            ),
        ),
        effects=frozenset(
            {
                OperationEffect.CACHE_ADMIN,
                OperationEffect.CACHE_WRITE,
                OperationEffect.DESTRUCTIVE,
                OperationEffect.ARBITRARY_LOCAL_PATH,
            }
        ),
    ),
)

OPERATION_REGISTRY: Mapping[str, OperationSpec[Any]] = registry_mapping(
    _OPERATIONS
)


def get_operation(operation: str) -> OperationSpec[Any] | None:
    return OPERATION_REGISTRY.get(operation)


def resolve_operations(
    *,
    excluded_effects: frozenset[OperationEffect] = DEFAULT_EXCLUDED_EFFECTS,
) -> tuple[OperationSpec[Any], ...]:
    """Return the default agent-safe projection of the one authoritative registry."""

    return resolve_operation_specs(
        _OPERATIONS, excluded_effects=excluded_effects
    )


def registry_document(
    *,
    excluded_effects: frozenset[OperationEffect] = DEFAULT_EXCLUDED_EFFECTS,
) -> dict[str, Any]:
    return operation_registry_document(
        _OPERATIONS,
        schema_version=REGISTRY_SCHEMA_VERSION,
        excluded_effects=excluded_effects,
    )


def dispatch_operation(operation: str, parameters: Mapping[str, Any]) -> Any:
    spec = get_operation(operation)
    if spec is None:
        raise OperationRequestError(
            "operation_not_found", f"unknown arc-paper operation: {operation}"
        )
    return spec.invoke(parameters)


__all__ = [
    "DEFAULT_EXCLUDED_EFFECTS",
    "JsonCodec",
    "JsonOutputCodec",
    "OPERATION_REGISTRY",
    "OperationEffect",
    "OperationRequestError",
    "OperationSpec",
    "REGISTRY_SCHEMA_VERSION",
    "dispatch_operation",
    "get_operation",
    "registry_document",
    "resolve_operations",
    "to_json_value",
]
