from __future__ import annotations

import ast
from pathlib import Path

import pytest

import arc_companion.pipeline as pipeline
from arc_companion.callsite_inventory import (
    assert_paid_call_inventory_complete,
    inventory_paid_calls,
)
from arc_companion.recovery_units import (
    PIPELINE_LANE_REGISTRY,
    RECOVERY_UNIT_REGISTRY,
    pipeline_lane_binding,
)


PACKAGE_ROOT = Path(pipeline.__file__).parent


def test_production_paid_callsite_inventory_is_transitive_and_owned() -> None:
    sites = inventory_paid_calls(PACKAGE_ROOT)
    assert_paid_call_inventory_complete(sites)
    assert {site.disposition for site in sites} >= {
        "central_descriptor", "descriptor_wrapper", "exact_receipt_control",
        "caller_owned_callback_contract", "transparent_paid_wrapper",
    }
    assert any(site.callee == "run_json" for site in sites)


def test_inventory_ids_do_not_depend_on_line_numbers() -> None:
    baseline = inventory_paid_calls(PACKAGE_ROOT)
    source = (PACKAGE_ROOT / "pipeline.py").read_text(encoding="utf-8")
    shifted = inventory_paid_calls(
        PACKAGE_ROOT, source_overrides={"pipeline": "\n\n" + source},
    )
    assert {site.stable_id for site in shifted} == {
        site.stable_id for site in baseline
    }


@pytest.mark.parametrize("source", [
    "from arc_llm import run_json as paid\ndef broken():\n    return paid('x', schema={})\n",
    "import arc_llm\ndef broken():\n    return arc_llm.run_json('x', schema={})\n",
    (
        "from somewhere import _llm_call\npaid = _llm_call\n"
        "def broken():\n    return paid(None, 'x', {})\n"
    ),
    "def broken(call_model):\n    return call_model('x', {}, None, 'label')\n",
    "def wrapper(result_llm):\n    return result_llm('x', schema={})\n",
])
def test_added_name_attribute_alias_wrapper_or_callback_fails_closed(
    tmp_path: Path, source: str,
) -> None:
    (tmp_path / "fixture.py").write_text(source, encoding="utf-8")
    sites = inventory_paid_calls(tmp_path)
    assert sites
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


def test_higher_order_paid_callback_is_resolved_transitively(
    tmp_path: Path,
) -> None:
    (tmp_path / "fixture.py").write_text(
        "from arc_llm import run_json\n"
        "def leaf(fn):\n    return fn('x', schema={})\n"
        "def middle(callback):\n    return leaf(callback)\n"
        "def caller():\n    return middle(run_json)\n",
        encoding="utf-8",
    )
    sites = inventory_paid_calls(tmp_path)
    assert any(site.callee == "run_json" for site in sites)
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


@pytest.mark.parametrize("source", [
    (
        "from arc_llm import run_json\n"
        "def caller():\n"
        "    funcs = [run_json]\n"
        "    return funcs[0]('x', schema={})\n"
    ),
    (
        "from arc_llm import run_json\n"
        "def leaf(fn):\n    return fn\n"
        "def middle():\n    return leaf(run_json)\n"
        "def caller():\n    return middle()('x', schema={})\n"
    ),
    (
        "from arc_llm import run_json\n"
        "def caller():\n"
        "    return {'paid': run_json}['paid']('x', schema={})\n"
    ),
    (
        "from arc_llm import run_json\n"
        "def caller(registry):\n"
        "    registry['paid'] = run_json\n"
        "    return registry['paid']('x', schema={})\n"
    ),
    (
        "from arc_llm import run_json\n"
        "def caller(holder):\n"
        "    holder.paid = run_json\n"
        "    return holder.paid('x', schema={})\n"
    ),
    (
        "from arc_llm import run_json\n"
        "def exported():\n"
        "    return run_json\n"
    ),
    (
        "from arc_llm import run_json\n"
        "def caller():\n"
        "    paid: object = run_json\n"
        "    return paid('x', schema={})\n"
    ),
])
def test_paid_callable_escape_fails_closed(tmp_path: Path, source: str) -> None:
    (tmp_path / "fixture.py").write_text(source, encoding="utf-8")
    sites = inventory_paid_calls(tmp_path)
    assert sites and any(site.disposition is None for site in sites)
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


@pytest.mark.parametrize("source", [
    (
        "import arc_llm\n"
        "def caller():\n"
        "    return getattr(arc_llm, 'run_json')('x', schema={})\n"
    ),
    (
        "import arc_llm\n"
        "def caller():\n"
        "    paid = getattr(arc_llm, 'run_json')\n"
        "    return paid('x', schema={})\n"
    ),
    (
        "import arc_llm\n"
        "def exported():\n"
        "    return getattr(arc_llm, 'run_json')\n"
    ),
    (
        "from arc_llm import run_json\n"
        "def caller():\n"
        "    return globals()['run_json']('x', schema={})\n"
    ),
])
def test_literal_runtime_lookup_resolves_to_paid_leaf(
    tmp_path: Path, source: str,
) -> None:
    (tmp_path / "fixture.py").write_text(source, encoding="utf-8")
    sites = inventory_paid_calls(tmp_path)
    assert sites and all(site.callee == "run_json" for site in sites)
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


@pytest.mark.parametrize("source", [
    (
        "import arc_llm\n"
        "def caller(method):\n"
        "    return getattr(arc_llm, method)('x', schema={})\n"
    ),
    (
        "import arc_llm\n"
        "def caller(method):\n"
        "    paid = getattr(arc_llm, method)\n"
        "    alias = paid\n"
        "    return alias('x', schema={})\n"
    ),
    (
        "from arc_llm import run_json\n"
        "def caller(name):\n"
        "    return globals()[name]('x', schema={})\n"
    ),
    (
        "def caller(registry, name):\n"
        "    paid = registry[name]\n"
        "    return paid('x', schema={})\n"
    ),
    (
        "def caller(factory):\n"
        "    return factory()('x', schema={})\n"
    ),
])
def test_dynamic_runtime_callable_selection_fails_closed(
    tmp_path: Path, source: str,
) -> None:
    (tmp_path / "fixture.py").write_text(source, encoding="utf-8")
    sites = inventory_paid_calls(tmp_path)
    assert sites and any(
        site.callee == "<dynamic-dispatch>" and site.disposition is None
        for site in sites
    )
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


def test_unreachable_receipt_writer_does_not_control_paid_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "fixture.py").write_text(
        "from arc_llm import run_json\n"
        "def broken():\n"
        "    if False:\n"
        "        write_ledger_submission_receipt()\n"
        "    return run_json('x', schema={})\n",
        encoding="utf-8",
    )
    sites = inventory_paid_calls(tmp_path)
    assert len(sites) == 1 and sites[0].disposition is None
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


def test_mutually_exclusive_receipt_branch_does_not_control_paid_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "fixture.py").write_text(
        "from arc_llm import run_json\n"
        "def broken(flag):\n"
        "    if flag:\n"
        "        write_ledger_submission_receipt()\n"
        "    else:\n"
        "        return run_json('x', schema={})\n",
        encoding="utf-8",
    )
    sites = inventory_paid_calls(tmp_path)
    assert len(sites) == 1 and sites[0].disposition is None
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


@pytest.mark.parametrize("body", [
    "flag and write_ledger_submission_receipt()\n    return run_json('x', schema={})",
    (
        "write_ledger_submission_receipt() if flag else None\n"
        "    return run_json('x', schema={})"
    ),
    (
        "[write_ledger_submission_receipt() for _ in items]\n"
        "    return run_json('x', schema={})"
    ),
    (
        "tuple(write_ledger_submission_receipt() for _ in items)\n"
        "    return run_json('x', schema={})"
    ),
    (
        "{write_ledger_submission_receipt() for _ in items}\n"
        "    return run_json('x', schema={})"
    ),
    (
        "{item: write_ledger_submission_receipt() for item in items}\n"
        "    return run_json('x', schema={})"
    ),
    "assert write_ledger_submission_receipt()\n    return run_json('x', schema={})",
])
def test_expression_conditional_receipt_does_not_control_paid_call(
    tmp_path: Path, body: str,
) -> None:
    (tmp_path / "fixture.py").write_text(
        "from arc_llm import run_json\n"
        f"def broken(flag=None, items=()):\n    {body}\n",
        encoding="utf-8",
    )
    sites = inventory_paid_calls(tmp_path)
    assert len(sites) == 1 and sites[0].disposition is None
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


@pytest.mark.parametrize("construct", [
    (
        "from contextlib import suppress\n"
        "def broken():\n"
        "    with suppress(Exception):\n"
        "        write_ledger_submission_receipt()\n"
        "    return run_json('x', schema={})\n"
    ),
    (
        "def broken():\n"
        "    try:\n"
        "        write_ledger_submission_receipt()\n"
        "    except* Exception:\n"
        "        pass\n"
        "    return run_json('x', schema={})\n"
    ),
    (
        "class AsyncSuppress:\n"
        "    async def __aenter__(self): return self\n"
        "    async def __aexit__(self, *_): return True\n"
        "async def broken():\n"
        "    async with AsyncSuppress():\n"
        "        write_ledger_submission_receipt()\n"
        "    return run_json('x', schema={})\n"
    ),
])
def test_suppressed_receipt_failure_does_not_control_paid_call(
    tmp_path: Path, construct: str,
) -> None:
    (tmp_path / "fixture.py").write_text(
        "from arc_llm import run_json\n" + construct,
        encoding="utf-8",
    )
    sites = inventory_paid_calls(tmp_path)
    assert len(sites) == 1 and sites[0].disposition is None
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


def test_mutating_each_central_descriptor_to_none_is_detected() -> None:
    source = (PACKAGE_ROOT / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    targets = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_llm_call"
    ]
    assert targets
    for target in targets:
        changed = ast.parse(source)
        match = next(
            node for node in ast.walk(changed)
            if isinstance(node, ast.Call) and node.lineno == target.lineno
            and isinstance(node.func, ast.Name) and node.func.id == "_llm_call"
        )
        keyword = next(item for item in match.keywords if item.arg == "recovery_descriptor")
        keyword.value = ast.Constant(None)
        sites = inventory_paid_calls(
            PACKAGE_ROOT, source_overrides={"pipeline": ast.unparse(changed)},
        )
        with pytest.raises(AssertionError, match="unowned paid structured"):
            assert_paid_call_inventory_complete(sites)


def test_new_paid_call_cannot_borrow_a_sibling_receipt_control() -> None:
    source = (PACKAGE_ROOT / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wrapper = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_llm_call"
    )
    wrapper.body.append(ast.Expr(value=ast.Call(
        func=ast.Name(id="run_json", ctx=ast.Load()),
        args=[ast.Constant("undescribed")],
        keywords=[],
    )))
    sites = inventory_paid_calls(
        PACKAGE_ROOT, source_overrides={"pipeline": ast.unparse(tree)},
    )
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


def test_mutating_each_descriptor_wrapper_hook_is_detected() -> None:
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        module = path.stem
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        targets = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "call_model_with_recovery_descriptor"
        ]
        for target in targets:
            changed = ast.parse(source)
            match = next(
                node for node in ast.walk(changed)
                if isinstance(node, ast.Call) and node.lineno == target.lineno
                and isinstance(node.func, ast.Name)
                and node.func.id == "call_model_with_recovery_descriptor"
            )
            descriptor = next(
                (item for item in match.keywords if item.arg == "descriptor"),
                None,
            )
            if descriptor is not None:
                descriptor.value = ast.Constant(None)
            else:
                assert len(match.args) >= 6
                match.args[5] = ast.Constant(None)
            sites = inventory_paid_calls(
                PACKAGE_ROOT, source_overrides={module: ast.unparse(changed)},
            )
            with pytest.raises(AssertionError, match="unowned paid structured"):
                assert_paid_call_inventory_complete(sites)


def test_removing_each_exact_receipt_writer_exposes_its_paid_call() -> None:
    source = (PACKAGE_ROOT / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    writer_lines = {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "write_ledger_submission_receipt"
    }
    assert writer_lines
    for writer_line in writer_lines:
        changed = ast.parse(source)
        writer = next(
            node for node in ast.walk(changed)
            if isinstance(node, ast.Call) and node.lineno == writer_line
            and isinstance(node.func, ast.Name)
            and node.func.id == "write_ledger_submission_receipt"
        )
        writer.func.id = "removed_receipt_writer"
        sites = inventory_paid_calls(
            PACKAGE_ROOT, source_overrides={"pipeline": ast.unparse(changed)},
        )
        assert any(
            site.callee in {"result_llm", "run_json"}
            and site.disposition is None
            for site in sites
        )
        with pytest.raises(AssertionError, match="unowned paid structured"):
            assert_paid_call_inventory_complete(sites)


def test_removing_chapter_guide_callback_binding_is_detected() -> None:
    source = (PACKAGE_ROOT / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "generate_chapter_guide"
    ]
    assert calls
    for node in calls:
        binding = next(item for item in node.keywords if item.arg == "call_model")
        binding.value = ast.Name(id="uncontrolled_model", ctx=ast.Load())
    sites = inventory_paid_calls(
        PACKAGE_ROOT, source_overrides={"pipeline": ast.unparse(tree)},
    )
    with pytest.raises(AssertionError, match="unowned paid structured"):
        assert_paid_call_inventory_complete(sites)


def test_dynamic_lane_registry_is_closed_and_exact() -> None:
    assert set(PIPELINE_LANE_REGISTRY) == {"translation", "companion", "guide"}
    for lane, binding in PIPELINE_LANE_REGISTRY.items():
        spec = RECOVERY_UNIT_REGISTRY[binding.recovery_unit]
        assert binding.public_lane == lane
        assert binding.validator == spec.validator
        assert binding.application == spec.application
        assert pipeline_lane_binding(lane) is binding
    with pytest.raises(ValueError, match="Unknown dynamic pipeline lane"):
        pipeline_lane_binding("user-supplied-lane")
