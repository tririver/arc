from __future__ import annotations

import argparse
import inspect

from jsonschema import Draft202012Validator
import pytest

from arc_paper import capabilities, cli, service


def test_catalog_publishes_strict_valid_schemas_and_effect_classification():
    document = capabilities.catalog_document()

    assert document["schema_version"] == "arc.paper.capability-catalog.v1"
    assert len(document["operations"]) == len(capabilities.OPERATION_CATALOG) == 42
    assert [item["name"] for item in document["operations"]] == sorted(
        capabilities.OPERATION_CATALOG
    )
    for operation in document["operations"]:
        Draft202012Validator.check_schema(operation["parameters"])
        Draft202012Validator.check_schema(operation["result"]["schema"])
        assert operation["parameters"]["additionalProperties"] is False
        assert set(operation["classification"]) == {
            "execution", "network", "cache", "llm", "job", "destructive", "admin"
        }
        assert operation["id"] == f"arc-paper.{operation['name']}.v1"
        assert operation["version"] == 1
        assert operation["result"]["serializer"] == "arc.result-envelope.v1"
        assert operation["result"]["delivery"] == {
            "owner": "controller",
            "pagination": {
                "mode": "byte-limited-artifact-handle",
                "max_inline_bytes": 64 * 1024,
            },
            "provenance": {
                "mode": "operation-receipt",
                "required_fields": [
                    "operation_id", "operation_version", "arguments_sha256"
                ],
            },
        }

    batch = capabilities.get_operation_spec("summary-batch.run")
    assert batch is not None
    assert batch.network_access == "may"
    assert batch.cache_access == "read-write"
    assert batch.uses_llm is True
    assert batch.is_job is True

    removal = capabilities.get_operation_spec("cache.remove")
    assert removal is not None
    assert removal.destructive is True
    assert removal.admin is True
    assert removal.as_dict()["authorization"]["supervision_required"] is True

    batch_create = capabilities.get_operation_spec("summary-batch.create")
    assert batch_create is not None
    assert batch_create.destructive is True
    assert batch_create.as_dict()["authorization"]["supervision_required"] is True


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    action = next(
        item for item in parser._actions if isinstance(item, argparse._SubParsersAction)
    )
    return action.choices


def test_catalog_covers_registered_cli_commands_and_aliases(monkeypatch):
    class ParserComplete(Exception):
        pass

    captured = {}

    def capture_parser(parser, _args=None, _namespace=None):
        captured["parser"] = parser
        raise ParserComplete

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_parser)
    with pytest.raises(ParserComplete):
        cli.main([])

    root_commands = _subcommands(captured["parser"])
    compound = {"cache", "doctor", "summary-batch"}
    for name, parser in root_commands.items():
        if name in compound:
            for subcommand in _subcommands(parser):
                assert capabilities.operation_name_from_argv([name, subcommand]) is not None
        else:
            assert capabilities.operation_name_from_argv([name]) is not None


def test_catalog_covers_every_public_service_operation():
    service_operations = {
        name
        for name, function in inspect.getmembers(service, inspect.isfunction)
        if not name.startswith("_") and function.__module__ == service.__name__
    }
    catalog_operations = {
        spec.executor.removeprefix("service.")
        for spec in capabilities.OPERATION_CATALOG.values()
        if spec.executor.startswith("service.")
    }

    assert catalog_operations == service_operations


def test_catalog_ids_and_aliases_are_globally_unique():
    operation_ids = [spec.operation_id for spec in capabilities.OPERATION_CATALOG.values()]
    aliases = [
        alias
        for spec in capabilities.OPERATION_CATALOG.values()
        for alias in spec.aliases
    ]

    assert len(operation_ids) == len(set(operation_ids))
    assert len(aliases) == len(set(aliases))
    assert not set(aliases) & set(capabilities.OPERATION_CATALOG)


def test_catalog_resolves_cli_aliases_and_compound_operations():
    assert capabilities.get_operation_spec("extract-ids").name == "extract-paper-ids"
    assert capabilities.get_operation_spec("llm-summary").name == "get-llm-summary"
    assert capabilities.operation_name_from_argv(["summary-batch", "run", "batch-a"]) == (
        "summary-batch.run"
    )
    assert capabilities.operation_name_from_argv(["doctor", "cache"]) == "doctor.cache"
    assert capabilities.operation_name_from_argv(["unknown", "--flag"]) is None


def test_cli_worker_capability_classification_comes_from_catalog():
    assert cli.command_capabilities(["llm-summary", "0911.3380"]) == frozenset(
        {capabilities.RECURSIVE_LLM_CAPABILITY}
    )
    assert cli.command_capabilities(["summary-batch", "run", "batch-a"]) == frozenset(
        {capabilities.RECURSIVE_LLM_CAPABILITY}
    )
    assert cli.command_capabilities(["get-title", "0911.3380"]) == frozenset()


def test_dispatcher_accepts_structured_parameters_and_resolves_service_lazily(monkeypatch):
    seen = {}

    def get_title(paper_ids, *, refresh=False):
        seen.update(paper_ids=paper_ids, refresh=refresh)
        return {"ok": True, "data": "A title", "errors": [], "meta": {}}

    monkeypatch.setattr(service, "get_title", get_title)

    result = capabilities.dispatch_operation(
        "get-title", {"paper_ids": ["0911.3380"], "refresh": True}
    )

    assert result["data"] == "A title"
    assert seen == {"paper_ids": ["0911.3380"], "refresh": True}


def test_dispatcher_rejects_unknown_or_unvalidated_parameters_before_execution(monkeypatch):
    monkeypatch.setattr(
        service,
        "get_title",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not execute")),
    )

    unknown = capabilities.dispatch_operation("shell", {"argv": ["rm", "anything"]})
    assert unknown["error"]["code"] == "paper_operation_unknown"

    invalid = capabilities.dispatch_operation(
        "get-title", {"paper_ids": "0911.3380", "argv": ["unexpected"]}
    )
    assert invalid["error"]["code"] == "paper_operation_parameters_invalid"
    assert "Additional properties" in invalid["validation_errors"][0]

    missing = capabilities.dispatch_operation("get-title", {})
    assert missing["error"]["code"] == "paper_operation_parameters_invalid"
    assert any("required property" in item for item in missing["validation_errors"])


def test_dispatcher_exposes_service_capabilities_not_previously_on_cli(monkeypatch):
    monkeypatch.setattr(
        service,
        "search_inspire",
        lambda query, *, limit=20: {
            "ok": True,
            "data": {"query": query, "limit": limit},
            "errors": [],
            "meta": {},
        },
    )

    result = capabilities.dispatch_operation("search-inspire", {"query": "gravity", "limit": 3})

    assert result["data"] == {"query": "gravity", "limit": 3}


def test_local_paths_require_controller_artifact_handles_and_resolver(monkeypatch):
    parse = capabilities.get_operation_spec("parse")
    export = capabilities.get_operation_spec("summary-batch.export")
    assert parse is not None
    assert export is not None
    assert dict(parse.artifact_parameters) == {
        "source_path": "read",
        "html_path": "read",
        "tex_path": "read",
        "markdown_path": "read",
        "pdf_path": "read",
    }
    assert dict(export.artifact_parameters) == {"output": "write"}
    assert parse.parameter_schema["properties"]["source_path"] != {"type": "string"}

    raw_path = capabilities.dispatch_operation("parse", {"source_path": "/etc/passwd"})
    assert raw_path["error"]["code"] == "paper_operation_parameters_invalid"
    raw_output = capabilities.dispatch_operation(
        "summary-batch.export", {"name": "batch-a", "output": "/tmp/export.jsonl"}
    )
    assert raw_output["error"]["code"] == "paper_operation_parameters_invalid"

    unresolved = capabilities.dispatch_operation(
        "parse", {"source_path": {"handle_id": "source-1"}}
    )
    assert unresolved["error"]["code"] == "paper_artifact_resolver_required"

    seen = {}

    def parse_source(source_path, **options):
        seen.update(source_path=source_path, options=options)
        return {"ok": True, "data": {}, "errors": [], "meta": {}}

    def resolve(handle_id, **context):
        seen.update(handle_id=handle_id, context=context)
        return "/controller/run/artifacts/source.pdf"

    monkeypatch.setattr(service, "parse_source", parse_source)
    resolved = capabilities.dispatch_operation(
        "parse",
        {"source_path": {"handle_id": "source-1"}},
        artifact_resolver=resolve,
    )

    assert resolved["ok"] is True
    assert seen == {
        "handle_id": "source-1",
        "context": {
            "access": "read",
            "operation": "parse",
            "parameter": "source_path",
        },
        "source_path": "/controller/run/artifacts/source.pdf",
        "options": {},
    }


def test_optional_positional_parameters_are_passed_as_none(monkeypatch):
    seen = {}

    def search_full_text(paper_ids, *, query, **options):
        seen.update(paper_ids=paper_ids, query=query, options=options)
        return {"ok": True, "data": [], "errors": [], "meta": {}}

    monkeypatch.setattr(service, "search_full_text", search_full_text)

    result = capabilities.dispatch_operation("search-full-text", {"query": "entropy"})

    assert result["ok"] is True
    assert seen == {"paper_ids": None, "query": "entropy", "options": {}}


@pytest.mark.parametrize(
    ("operation", "parameters", "missing"),
    [
        ("get-section", {"paper_ids": "0911.3380"}, "section"),
        ("get-equation-context", {"paper_ids": "0911.3380"}, "query"),
    ],
)
def test_required_service_parameters_fail_schema_validation(operation, parameters, missing):
    result = capabilities.dispatch_operation(operation, parameters)

    assert result["error"]["code"] == "paper_operation_parameters_invalid"
    assert any(missing in error for error in result["validation_errors"])
