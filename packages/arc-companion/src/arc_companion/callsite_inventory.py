from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Mapping


PAID_CALL_NAMES = frozenset({
    "run_text", "run_text_result", "run_json", "run_json_result",
    "_llm_call", "call_model", "result_llm",
    "call_model_with_recovery_descriptor",
})

# These callbacks are not exemptions: their production caller must supply an
# adapter whose transitive body owns either a descriptor or an exact receipt.
CALLBACK_RECOVERY_CONTRACTS = frozenset({
    ("chapter_guide", "generate_chapter_guide", "call_model"),
})

# Production currently has no genuinely nonrecoverable structured calls.
EXPLICIT_NONRECOVERABLE_EXEMPTIONS: Mapping[str, str] = {}

TRANSPARENT_WRAPPER_CONTRACTS = {
    ("pipeline", "limited"): "submission-limiter-only",
}


@dataclass(frozen=True)
class PaidCallSite:
    stable_id: str
    module: str
    function: str
    callee: str
    lineno: int
    disposition: str | None
    reason: str | None = None


def inventory_paid_calls(
    package_root: Path,
    *,
    source_overrides: Mapping[str, str] | None = None,
) -> tuple[PaidCallSite, ...]:
    """Resolve paid structured leaves, aliases and callback wrappers fail-closed."""

    overrides = dict(source_overrides or {})
    modules: dict[str, ast.Module] = {}
    paths = sorted(package_root.rglob("*.py"))
    for path in paths:
        relative = path.relative_to(package_root).with_suffix("")
        module = ".".join(relative.parts)
        modules[module] = ast.parse(
            overrides.get(module, path.read_text(encoding="utf-8")),
            filename=str(path),
        )
    sites: list[PaidCallSite] = []
    for module, tree in modules.items():
        aliases = _aliases(tree)
        parents = _parents(tree)
        functions = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        indirect_bindings = _indirect_paid_bindings(tree, aliases)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _resolved_callee(node.func, aliases)
            if callee not in PAID_CALL_NAMES:
                continue
            function_node = _enclosing_function(node, parents)
            function = _function_qualname(node, parents)
            disposition, reason = _disposition(
                module, function_node, node, callee, aliases, functions, tree,
            )
            normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
            sites.append(PaidCallSite(
                stable_id=f"{module}:{function}:{callee}:{digest}",
                module=module,
                function=function,
                callee=callee,
                lineno=int(node.lineno),
                disposition=disposition,
                reason=reason,
            ))
        for callee, outer_call, leaf_function, leaf_call in indirect_bindings:
            # A directly named/aliased leaf is already inventoried above. The
            # higher-order record is needed only when callback flow is the
            # sole way to discover that the leaf is paid.
            if _resolved_callee(leaf_call.func, aliases) == callee:
                continue
            disposition, reason = _disposition(
                module, leaf_function, leaf_call, callee, aliases, functions, tree,
            )
            normalized = "|".join((
                ast.dump(outer_call, annotate_fields=True, include_attributes=False),
                ast.dump(leaf_call, annotate_fields=True, include_attributes=False),
                callee,
            ))
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
            sites.append(PaidCallSite(
                stable_id=(
                    f"{module}:{leaf_function.name}:{callee}:indirect:{digest}"
                ),
                module=module,
                function=leaf_function.name,
                callee=callee,
                lineno=int(leaf_call.lineno),
                disposition=disposition,
                reason=reason,
            ))
        for callee, escaped in _unresolved_paid_callable_escapes(
            tree, aliases, indirect_bindings,
        ):
            function = _function_qualname(escaped, parents)
            normalized = ast.dump(
                escaped, annotate_fields=True, include_attributes=False,
            )
            digest = hashlib.sha256(
                f"{callee}|{normalized}".encode("utf-8")
            ).hexdigest()[:16]
            sites.append(PaidCallSite(
                stable_id=f"{module}:{function}:{callee}:escape:{digest}",
                module=module,
                function=function,
                callee=callee,
                lineno=int(escaped.lineno),
                disposition=None,
                reason="paid callable escapes analyzed invocation flow",
            ))
        for dynamic_index, escaped in enumerate(
            _unresolved_dynamic_invocations(tree, aliases), start=1,
        ):
            function = _function_qualname(escaped, parents)
            normalized = ast.dump(
                escaped, annotate_fields=True, include_attributes=False,
            )
            digest = hashlib.sha256(
                f"{dynamic_index}|{normalized}".encode("utf-8")
            ).hexdigest()[:16]
            sites.append(PaidCallSite(
                stable_id=f"{module}:{function}:dynamic-dispatch:{digest}",
                module=module,
                function=function,
                callee="<dynamic-dispatch>",
                lineno=int(escaped.lineno),
                disposition=None,
                reason="dynamic callable target cannot be proven non-paid",
            ))
    callback_ownership = _global_callback_ownership(modules)
    sites = [
        replace(
            site,
            disposition=(
                site.disposition if callback_ownership.get(
                    (site.module, site.function, site.callee), True
                ) else None
            ),
            reason=(
                site.reason if callback_ownership.get(
                    (site.module, site.function, site.callee), True
                ) else "production callback adapter has no transitive recovery owner"
            ),
        )
        for site in sites
    ]
    return tuple(sorted(sites, key=lambda item: (item.module, item.lineno, item.stable_id)))


_ESCAPE_SENSITIVE_PAID_CALLS = PAID_CALL_NAMES - {"call_model", "result_llm"}


def _unresolved_dynamic_invocations(
    tree: ast.Module, aliases: Mapping[str, str],
) -> tuple[ast.AST, ...]:
    """Find runtime callable selection that can hide a paid provider leaf.

    Literal ``getattr(obj, "run_json")`` and ``globals()["run_json"]``
    expressions are resolved by :func:`_raw_callee` and receive the normal
    paid-call ownership analysis.  A computed attribute/key or an arbitrary
    call/subscript used as a callable cannot be classified statically, so the
    inventory emits an unresolved record instead of silently omitting it.
    Simple aliases of such expressions are followed until invocation or
    escape.
    """

    parents = _parents(tree)
    dynamic_aliases: set[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, str]
    ] = set()
    dynamic_values: set[ast.AST] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            continue
        target = _simple_alias_target(node)
        if target is None or not _runtime_selected_callable(node.value, aliases):
            continue
        dynamic_aliases.add((_enclosing_function(node, parents), target))
        dynamic_values.add(node.value)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                continue
            target = _simple_alias_target(node)
            if (
                target is not None
                and isinstance(node.value, ast.Name)
                and (
                    _enclosing_function(node, parents), node.value.id
                ) in dynamic_aliases
                and (_enclosing_function(node, parents), target)
                not in dynamic_aliases
            ):
                dynamic_aliases.add((_enclosing_function(node, parents), target))
                changed = True

    unresolved: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                _runtime_selected_callable(node.func, aliases)
                or isinstance(node.func, ast.Call)
                and _resolved_callee(node.func, aliases) not in PAID_CALL_NAMES
            ):
                unresolved.append(node.func)
            elif (
                isinstance(node.func, ast.Name)
                and (_enclosing_function(node, parents), node.func.id)
                in dynamic_aliases
            ):
                unresolved.append(node.func)
        if node not in dynamic_values:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            # The target name is reported where it is actually invoked.  If
            # it never is, returning/storing/passing the value below still
            # counts as an unresolved escape.
            continue
        unresolved.append(node)

    return tuple(dict.fromkeys(unresolved))


def _runtime_selected_callable(
    value: ast.AST, aliases: Mapping[str, str],
) -> bool:
    """Return true only for an unresolved expression used as a callable value."""

    if _resolved_callee(value, aliases) in PAID_CALL_NAMES:
        return False
    if isinstance(value, ast.Call) and _raw_callee(value.func) == "getattr":
        return not (
            len(value.args) >= 2
            and isinstance(value.args[1], ast.Constant)
            and isinstance(value.args[1].value, str)
        )
    if isinstance(value, ast.Subscript):
        return _subscript_literal_key(value) is None
    return False


def _unresolved_paid_callable_escapes(
    tree: ast.Module,
    aliases: Mapping[str, str],
    indirect_bindings: tuple[
        tuple[
            str,
            ast.Call,
            ast.FunctionDef | ast.AsyncFunctionDef,
            ast.Call,
        ], ...
    ],
) -> tuple[tuple[str, ast.AST], ...]:
    """Reject paid callable values that escape the analyzed call graph.

    Direct invocations, simple aliases, and callback arguments proven by the
    higher-order fixpoint are handled elsewhere. Containers, returned
    callables, subscripts and unresolved call-result indirection fail closed.
    """

    parents = _parents(tree)
    parameter_names = {
        argument.arg
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    local_definition_names = {
        function.name for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    handled_indirect: set[ast.AST] = set()
    for callee, outer_call, _leaf_function, _leaf_call in indirect_bindings:
        for value in [*outer_call.args, *(item.value for item in outer_call.keywords)]:
            if _resolved_callee(value, aliases) == callee:
                handled_indirect.add(value)
    output: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
            continue
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            continue
        raw = _raw_callee(node)
        callee = (
            raw if raw in parameter_names or raw in local_definition_names
            else aliases.get(raw, raw)
        )
        if callee not in _ESCAPE_SENSITIVE_PAID_CALLS:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        if node in handled_indirect:
            continue
        if (
            isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
            and getattr(parent, "value", None) is node
            and _simple_alias_target(parent) is not None
        ):
            continue
        # A common optional-adapter binding keeps a caller-provided callable
        # or falls back to one concrete runner. The assigned name is analyzed
        # at every later invocation; containers remain rejected.
        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or):
            grandparent = parents.get(parent)
            if (
                isinstance(grandparent, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
                and getattr(grandparent, "value", None) is parent
            ):
                continue
        output.append((callee, node))
    return tuple(output)


def assert_paid_call_inventory_complete(sites: tuple[PaidCallSite, ...]) -> None:
    unresolved = [site for site in sites if site.disposition is None]
    if unresolved:
        detail = ", ".join(
            f"{item.module}:{item.function}:{item.callee}:{item.lineno}"
            for item in unresolved
        )
        raise AssertionError(f"unowned paid structured call site(s): {detail}")
    identifiers = [site.stable_id for site in sites]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("paid structured call stable IDs collide")


def _disposition(
    module: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef | None,
    call: ast.Call,
    callee: str,
    aliases: Mapping[str, str],
    functions: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    tree: ast.Module,
) -> tuple[str | None, str | None]:
    function_name = function.name if function is not None else "<module>"
    exemption_id = _literal_keyword(call, "recovery_exemption")
    if exemption_id is not None:
        if exemption_id in EXPLICIT_NONRECOVERABLE_EXEMPTIONS:
            return "explicit_nonrecoverable_exemption", exemption_id
        return None, "unknown recovery exemption"
    if callee == "_llm_call":
        descriptor = _keyword(call, "recovery_descriptor")
        if descriptor is not None and not _is_none(descriptor):
            return "central_descriptor", None
        return None, "central call has no live recovery descriptor"
    if callee == "call_model_with_recovery_descriptor":
        descriptor = _keyword(call, "descriptor")
        if descriptor is None and len(call.args) >= 6:
            descriptor = call.args[5]
        if descriptor is not None and not _is_none(descriptor):
            return "descriptor_wrapper", None
        return None, "descriptor wrapper has no live descriptor"
    if callee == "result_llm":
        if function is not None and _exact_receipt_owns(function, call, aliases):
            return "exact_receipt_control", None
        return None, "result call is not dominated by an exact receipt writer"
    if callee == "call_model":
        if module == "recovery_units" and function_name == "call_model_with_recovery_descriptor":
            return "descriptor_wrapper_dispatch", None
        contract = (module, function_name, callee)
        if contract in CALLBACK_RECOVERY_CONTRACTS and _callback_callers_owned(
            tree, function_name, functions, aliases,
        ):
            return "caller_owned_callback_contract", None
        return None, "paid callback has no transitive recovery owner"
    # Direct arc-llm calls may only appear behind a central wrapper that owns
    # the receipt/descriptor.  Adding one elsewhere is intentionally rejected.
    if (
        module == "pipeline"
        and function_name == "_llm_call"
        and function is not None
        and "recovery_descriptor" in {
            item.arg for item in [
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            ]
        }
        and any(
            isinstance(node, ast.Call)
            and _resolved_callee(node.func, aliases)
            == "write_ledger_submission_receipt"
            for node in _local_function_nodes(function)
        )
        and sum(
            1 for node in _local_function_nodes(function)
            if isinstance(node, ast.Call)
            and _resolved_callee(node.func, aliases) in {
                "run_text", "run_text_result", "run_json", "run_json_result",
            }
        ) == 1
    ):
        return "central_control_dispatch", None
    if function is not None and _exact_receipt_owns(function, call, aliases):
        return "transitive_control_wrapper", None
    if (module, function_name) in TRANSPARENT_WRAPPER_CONTRACTS:
        return "transparent_paid_wrapper", TRANSPARENT_WRAPPER_CONTRACTS[
            (module, function_name)
        ]
    return None, "direct provider runner has no recovery control"


def _callback_callers_owned(
    tree: ast.Module,
    target: str,
    functions: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    aliases: Mapping[str, str],
) -> bool:
    callers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _resolved_callee(node.func, aliases) == target
    ]
    # The production caller may live in another module; the closed contract is
    # checked there by the paid leaves in its adapter.  The callback function
    # itself must never be called internally without an explicit binding.
    return not callers


def _global_callback_ownership(
    modules: Mapping[str, ast.Module],
) -> dict[tuple[str, str, str], bool]:
    output: dict[tuple[str, str, str], bool] = {}
    for contract in CALLBACK_RECOVERY_CONTRACTS:
        owner_module, owner_function, parameter = contract
        callers: list[tuple[ast.Module, ast.Call]] = []
        for tree in modules.values():
            aliases = _aliases(tree)
            callers.extend(
                (tree, node) for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and _resolved_callee(node.func, aliases) == owner_function
            )
        owned = bool(callers)
        for tree, call in callers:
            value = _keyword(call, parameter)
            if value is None:
                owned = False
                continue
            functions = {
                node.name: node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            aliases = _aliases(tree)
            if isinstance(value, ast.Lambda):
                owned = owned and _expression_owns(value.body, aliases)
            elif isinstance(value, ast.Name) and value.id in functions:
                owned = owned and _callable_owns(
                    functions[value.id], aliases, functions, seen=set(),
                )
            else:
                owned = False
        output[(owner_module, owner_function, parameter)] = owned
    return output


def _callable_owns(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: Mapping[str, str],
    functions: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    *,
    seen: set[str],
) -> bool:
    if function.name in seen:
        return False
    seen = {*seen, function.name}
    if _function_has_live_descriptor(function, aliases) or any(
        _exact_receipt_owns(function, node, aliases)
        for node in _local_function_nodes(function)
        if isinstance(node, ast.Call)
    ):
        return True
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        target = aliases.get(node.func.id, node.func.id)
        if target in functions and _callable_owns(
            functions[target], aliases, functions, seen=seen,
        ):
            return True
    return False


def _expression_owns(value: ast.AST, aliases: Mapping[str, str]) -> bool:
    return bool(
        isinstance(value, ast.Call)
        and _resolved_callee(value.func, aliases) == "_llm_call"
        and (descriptor := _keyword(value, "recovery_descriptor")) is not None
        and not _is_none(descriptor)
    )


def _exact_receipt_owns(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
    aliases: Mapping[str, str],
) -> bool:
    nodes = tuple(_local_function_nodes(function))
    writers = [
        node for node in nodes
        if isinstance(node, ast.Call)
        and _resolved_callee(node.func, aliases) == "write_ledger_submission_receipt"
    ]
    controlled = [
        node for node in nodes
        if isinstance(node, ast.Call)
        and _resolved_callee(node.func, aliases) in {
            "run_text", "run_text_result", "run_json", "run_json_result",
            "result_llm",
        }
    ]
    parents = _parents(function)
    optional = (
        ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match,
        ast.BoolOp, ast.IfExp, ast.comprehension, ast.ListComp,
        ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Assert,
        ast.With, ast.AsyncWith,
        *((ast.TryStar,) if hasattr(ast, "TryStar") else ()),
    )
    writer_optional_ancestors: set[ast.AST] = set()
    current: ast.AST = writers[0] if writers else function
    while current in parents:
        current = parents[current]
        if isinstance(current, optional):
            writer_optional_ancestors.add(current)
    call_ancestors: set[ast.AST] = set()
    current = call
    while current in parents:
        current = parents[current]
        call_ancestors.add(current)
    same_optional_branches = all(
        _optional_branch_slot(ancestor, writers[0], parents)
        == _optional_branch_slot(ancestor, call, parents)
        for ancestor in writer_optional_ancestors
        if ancestor in call_ancestors
    )
    return (
        len(writers) == 1
        and len(controlled) == 1
        and controlled[0] is call
        and writers[0].lineno < call.lineno
        and writer_optional_ancestors <= call_ancestors
        and same_optional_branches
    )


def _optional_branch_slot(
    ancestor: ast.AST,
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
) -> tuple[str, int | None]:
    child = node
    while child in parents and parents[child] is not ancestor:
        child = parents[child]
    for field, value in ast.iter_fields(ancestor):
        if isinstance(value, list) and child in value:
            index = value.index(child) if field in {"handlers", "cases"} else None
            return field, index
        if value is child:
            return field, None
    return "<unknown>", None


def _indirect_paid_bindings(
    tree: ast.Module,
    aliases: Mapping[str, str],
) -> tuple[
    tuple[
        str,
        ast.Call,
        ast.FunctionDef | ast.AsyncFunctionDef,
        ast.Call,
    ], ...
]:
    """Resolve local higher-order callback flow to its paid leaf by fixpoint."""

    definitions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    duplicates: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in definitions:
            duplicates.add(node.name)
        else:
            definitions[node.name] = node
    for name in duplicates:
        definitions.pop(name, None)

    parameters = {
        name: [
            item.arg for item in (
                [*function.args.posonlyargs, *function.args.args]
            )
        ]
        for name, function in definitions.items()
    }
    keyword_parameters = {
        name: {
            item.arg for item in (
                [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
            )
        }
        for name, function in definitions.items()
    }
    # Each summary maps an invoked callback parameter to the final callable
    # body and call expression where provider execution actually occurs.
    summaries: dict[
        str,
        dict[
            str,
            set[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call]],
        ],
    ] = {name: {} for name in definitions}
    for name, function in definitions.items():
        allowed = keyword_parameters[name]
        for node in _local_function_nodes(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in allowed
            ):
                summaries[name].setdefault(node.func.id, set()).add((function, node))

    changed = True
    while changed:
        changed = False
        for name, function in definitions.items():
            own_parameters = keyword_parameters[name]
            for call in _local_function_nodes(function):
                if not isinstance(call, ast.Call):
                    continue
                target_name = _resolved_callee(call.func, aliases)
                if target_name not in definitions:
                    continue
                target_order = parameters[target_name]
                bound: dict[str, ast.AST] = {
                    target_order[index]: value
                    for index, value in enumerate(call.args)
                    if index < len(target_order)
                }
                bound.update({
                    item.arg: item.value for item in call.keywords
                    if item.arg is not None
                })
                for target_parameter, leaves in summaries[target_name].items():
                    value = bound.get(target_parameter)
                    if not isinstance(value, ast.Name) or value.id not in own_parameters:
                        continue
                    current = summaries[name].setdefault(value.id, set())
                    missing = leaves - current
                    if missing:
                        current.update(missing)
                        changed = True

    bindings: list[
        tuple[
            str,
            ast.Call,
            ast.FunctionDef | ast.AsyncFunctionDef,
            ast.Call,
        ]
    ] = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        target_name = _resolved_callee(call.func, aliases)
        if target_name not in definitions:
            continue
        target_order = parameters[target_name]
        bound = {
            target_order[index]: value
            for index, value in enumerate(call.args)
            if index < len(target_order)
        }
        bound.update({
            item.arg: item.value for item in call.keywords if item.arg is not None
        })
        for parameter, leaves in summaries[target_name].items():
            value = bound.get(parameter)
            if value is None:
                continue
            callee = _resolved_callee(value, aliases)
            if callee in PAID_CALL_NAMES:
                bindings.extend(
                    (callee, call, leaf_function, leaf_call)
                    for leaf_function, leaf_call in leaves
                )
    return tuple(bindings)


def _local_function_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    """Walk one callable body without borrowing controls from nested callables."""

    output: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        output.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            visit(child)

    for statement in function.body:
        visit(statement)
    return tuple(output)


def _function_has_live_descriptor(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: Mapping[str, str],
) -> bool:
    for node in _local_function_nodes(function):
        if not isinstance(node, ast.Call):
            continue
        callee = _resolved_callee(node.func, aliases)
        if callee == "_llm_call":
            descriptor = _keyword(node, "recovery_descriptor")
            if descriptor is not None and not _is_none(descriptor):
                return True
        if callee == "call_model_with_recovery_descriptor":
            descriptor = _keyword(node, "descriptor")
            if descriptor is None and len(node.args) >= 6:
                descriptor = node.args[5]
            if descriptor is not None and not _is_none(descriptor):
                return True
    return False


def _aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = item.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            target = _simple_alias_target(node)
            value = _traceable_alias_value(node.value, aliases)
            if target is not None and value:
                aliases[target] = aliases.get(value, value)
    changed = True
    while changed:
        changed = False
        for name, value in list(aliases.items()):
            resolved = aliases.get(value, value)
            if resolved != value:
                aliases[name] = resolved
                changed = True
    return aliases


def _simple_alias_target(
    node: ast.Assign | ast.AnnAssign | ast.NamedExpr,
) -> str | None:
    """Return a local name only when an assignment is a traceable alias.

    Attribute and subscript targets mutate an external/container address and
    therefore remain paid-callable escapes.  Treating every assignment value
    as an alias used to hide exactly those indirect invocation paths.
    """

    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        return node.targets[0].id
    return node.target.id if isinstance(node.target, ast.Name) else None


def _traceable_alias_value(value: ast.AST, aliases: Mapping[str, str]) -> str:
    direct = _raw_callee(value)
    if direct:
        return direct
    # Optional callback adapters such as ``supplied or run_json_result`` have
    # one concrete paid fallback.  Preserve that fallback in the call graph;
    # the caller-supplied branch is independently inventoried at its parameter
    # invocation site.
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
        paid = {
            aliases.get(raw, raw)
            for item in value.values
            if (raw := _raw_callee(item))
            and aliases.get(raw, raw) in PAID_CALL_NAMES
        }
        if len(paid) == 1:
            return next(iter(paid))
    return ""


def _resolved_callee(value: ast.AST, aliases: Mapping[str, str]) -> str:
    raw = _raw_callee(value)
    return aliases.get(raw, raw)


def _raw_callee(value: ast.AST) -> str:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    if (
        isinstance(value, ast.Call)
        and _raw_callee(value.func) == "getattr"
        and len(value.args) >= 2
        and isinstance(value.args[1], ast.Constant)
        and isinstance(value.args[1].value, str)
    ):
        return value.args[1].value
    if isinstance(value, ast.Subscript):
        return _subscript_literal_key(value) or ""
    return ""


def _subscript_literal_key(value: ast.Subscript) -> str | None:
    item = value.slice
    # ``ast.Index`` disappeared in Python 3.9, but accepting it keeps this
    # analyzer deterministic on older hosts supported by downstream users.
    if hasattr(ast, "Index") and isinstance(item, ast.Index):  # pragma: no cover
        item = item.value
    return (
        item.value
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
        else None
    )


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    result: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            result[child] = parent
    return result


def _enclosing_function(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return None


def _function_qualname(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST],
) -> str:
    names: list[str] = []
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(current.name)
    return ".<locals>.".join(reversed(names)) if names else "<module>"


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _literal_keyword(call: ast.Call, name: str) -> str | None:
    value = _keyword(call, name)
    return (
        value.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
        else None
    )


def _is_none(value: ast.AST) -> bool:
    return isinstance(value, ast.Constant) and value.value is None
