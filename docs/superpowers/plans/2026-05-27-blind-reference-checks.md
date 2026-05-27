# Blind Reference Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact blind-reference checking path where proposers derive from axioms/definitions without seeing target paper equations, while reviewers compare two independent derivations against a reviewer-only reference claim.

**Architecture:** Keep the change inside the existing research calculation and `arc-llm` proposers-reviewer consensus flow. Add two optional consensus-step fields: `proposer_runtime` for configurable proposer source access, and `reviewer_reference_claim` for reviewer-only target equations. Existing foundation checks remain available, but workflow docs should route paper-derived equations needing verification through blind reference checks instead of exposing them in foundation context.

**Tech Stack:** Python dataclasses and JSON config parsing in `packages/arc-llm`; pytest unit tests; ARC skill Markdown workflow docs and packaged skill copies.

---

## Instruction Review

The requested change is acceptable under `AGENTS.md`.

- It improves general theoretical-physics reliability: blind derivation reduces confirmation bias across domains.
- It is portable across agent hosts: it uses generic config fields and prompt/runtime flags, not Codex-only behavior.
- It preserves package boundaries: `arc-llm` owns consensus execution; `skills/arc` owns workflow instructions.
- It avoids hard-coded papers, authors, subfields, and keyword lists.
- It keeps the skill layer concise: docs describe when to use blind checks, while `arc-llm` owns mechanics.

## File Map

- Modify `packages/arc-llm/src/arc_llm/proposers_reviewer/consensus.py`
  - Parse optional `steps[].proposer_runtime`.
  - Parse optional `steps[].reviewer_reference_claim`.
  - Default blind reference checks to no internet and no MCP/paper tools.
  - Default post-check new calculations to internet and ARC MCP enabled.
  - Inject `reviewer_reference_claim` only into reviewer prompt/schema, never shared `caller_context`.
  - Accept `reference_disagrees` as a terminal checked outcome when two blind proposers agree with each other but disagree with the reviewer-only reference claim.

- Modify `packages/arc-llm/tests/test_proposers_reviewer_consensus.py`
  - Cover blind-check default no-source proposer runtime.
  - Cover post-check new-calculation default source-enabled proposer runtime.
  - Cover step-level source access override.
  - Cover reviewer-only reference claim isolation.
  - Cover accepted `reference_disagrees`.
  - Cover rejection when `reference_disagrees` lacks proposer agreement.

- Modify `skills/arc/references/research-workflows/research-foundation.md`
  - State initial foundation should contain only definitions, axioms, conventions, and truly foundational equations.
  - State paper equations needing verification should not be placed in foundation just to be checked.

- Modify `skills/arc/references/research-workflows/research-execute.md`
  - Add a compact blind reference check procedure.
  - State blind-check proposers default to no paper tools and no internet.
  - State post-check new calculations turn paper tools and internet on by default.
  - Show configurable `proposer_runtime`.
  - Explain reviewer-only C comparison outcomes.

- Modify `skills/arc/references/research-workflows/research-plan.md`
  - Warn not to include target formulas in plan steps or allowed inputs.
  - Route paper equations needing verification to blind reference checks.

- Modify `tests/test_arc_research_workflow_docs.py`
  - Add assertions for the new blind-check workflow text and default no-source proposer policy.
  - Update old assertions that expected proposer paper-tool access by default.

- Sync modified skill docs to:
  - `packaging/codex/arc/skills/arc/references/research-workflows/`
  - `packaging/claude/arc/skills/arc/references/research-workflows/`

---

### Task 1: Add Failing Consensus Tests

**Files:**
- Modify: `packages/arc-llm/tests/test_proposers_reviewer_consensus.py`

- [ ] **Step 1: Update the existing default-runtime assertions**

In `test_consensus_accepts_all_agree_on_first_attempt`, replace the current proposer runtime/source-access assertions:

```python
    assert proposer["runtime"]["allow_internet"] is True
    assert proposer["runtime"]["allow_mcp"] is True
    assert proposer["runtime"]["mcp_mode"] == "arc-only"
    assert "ARC paper MCP tools" in proposer_template
    assert "read the main reference" in proposer_template
    assert "internet search" in proposer_template.lower()
```

with:

```python
    assert proposer["runtime"]["allow_internet"] is True
    assert proposer["runtime"]["allow_mcp"] is True
    assert proposer["runtime"]["mcp_mode"] == "arc-only"
    assert "You may use ARC paper MCP tools" in proposer_template
    assert "Internet search is allowed" in proposer_template
```

- [ ] **Step 2: Add a blind-check no-source default test**

Add this test after `test_consensus_accepts_all_agree_on_first_attempt`:

```python
def test_blind_reference_check_disables_proposer_source_access_by_default(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    proposer = fake.calls[0]["loops"][0]["proposers"][0]
    proposer_template = proposer["prompt"]["template"]
    assert proposer["runtime"]["allow_internet"] is False
    assert proposer["runtime"]["allow_mcp"] is False
    assert "Do not use internet search" in proposer_template
    assert "Do not use ARC paper MCP tools" in proposer_template
```

- [ ] **Step 3: Add a reviewer-only isolation test**

Add:

```python
def test_reviewer_reference_claim_is_not_shared_with_proposers(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "all_agree",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
            )
        ]
    )
    reference_claim = {
        "id": "ref_eq_001",
        "label": "target reference equation",
        "latex": "x = y + z",
        "source": {"paper_id": "arXiv:1234.5678", "section": "S2"},
    }
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "kind": "new_calculation",
                "prompt": "Derive x in terms of y and z from the supplied definitions.",
                "allowed_context": {"definitions": ["x, y, z are scalar symbols"]},
                "reviewer_reference_claim": reference_claim,
            }
        ],
    )

    run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    loop = fake.calls[0]["loops"][0]
    caller_context_json = json.dumps(loop["caller_context"])
    proposer_json = json.dumps(loop["proposers"])
    reviewer_json = json.dumps(loop["reviewers"])
    assert "x = y + z" not in caller_context_json
    assert "x = y + z" not in proposer_json
    assert "x = y + z" in reviewer_json
    assert "reviewer_reference_claim" in reviewer_json
```

- [ ] **Step 4: Add an accepted reference-mismatch test**

Add:

```python
def test_reference_disagrees_accepts_when_two_blind_proposers_agree(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x = y - z", "reference_claim_status": "disagrees"},
                pairwise_check_overrides={
                    "A_minus_B_zero": True,
                    "B_minus_C_zero": False,
                    "A_minus_C_zero": False,
                    "true_count": 1,
                    "sympy_code": (
                        "simplify(expand(A-B)); "
                        "simplify(expand(B-C)); "
                        "simplify(expand(A-C))"
                    ),
                    "check_history": [
                        "A-B reduces to 0.",
                        "B-C reduces to 2*z.",
                        "A-C reduces to 2*z.",
                    ],
                },
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == "accepted"
    assert result["steps"][0]["reviewer_consensus"]["status"] == "reference_disagrees"
    assert result["steps"][0]["accepted_output"]["reference_claim_status"] == "disagrees"
```

- [ ] **Step 5: Add a rejection test for bad reference-mismatch evidence**

Add:

```python
def test_reference_disagrees_requires_blind_proposer_agreement(tmp_path):
    fake = FakeBatchRunner(
        [
            consensus_review(
                "reference_disagrees",
                agreed=["proposer_001", "proposer_002"],
                accepted={"result": "x"},
                pairwise_check_overrides={
                    "A_minus_B_zero": False,
                    "B_minus_C_zero": False,
                    "A_minus_C_zero": False,
                    "true_count": 0,
                    "sympy_code": (
                        "simplify(expand(A-B)); "
                        "simplify(expand(B-C)); "
                        "simplify(expand(A-C))"
                    ),
                },
            )
        ]
    )
    config = minimal_config(
        tmp_path,
        proposer_count=2,
        steps=[
            {
                "step_id": "blind_ref_eq_001",
                "prompt": "Derive x.",
                "reviewer_reference_claim": {"id": "ref_eq_001", "latex": "x = y + z"},
            }
        ],
    )

    result = run_proposers_reviewer_consensus(config, batch_runner=fake, base_env={})

    assert result["status"] == "failed"
    assert "reference_disagrees requires A-B=0" in result["steps"][0]["error"]
```

- [ ] **Step 6: Run tests and confirm failure**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests/test_proposers_reviewer_consensus.py -q
```

Expected: fail on missing fields/defaults/status support.

---

### Task 2: Implement Consensus Config Fields And Proposer Runtime Defaults

**Files:**
- Modify: `packages/arc-llm/src/arc_llm/proposers_reviewer/consensus.py`

- [ ] **Step 1: Extend `ConsensusStep`**

Change:

```python
@dataclass(frozen=True)
class ConsensusStep:
    step_id: str
    prompt: str
    kind: str
    allowed_context: dict[str, Any]
```

to:

```python
@dataclass(frozen=True)
class ConsensusStep:
    step_id: str
    prompt: str
    kind: str
    allowed_context: dict[str, Any]
    proposer_runtime: dict[str, Any]
    reviewer_reference_claim: dict[str, Any] | None
```

- [ ] **Step 2: Parse optional fields in `load_consensus_config`**

Inside the `steps.append(ConsensusStep(...))` call, add:

```python
                proposer_runtime=_dict(
                    step_data.get("proposer_runtime", {}),
                    f"{step_id}.proposer_runtime",
                ),
                reviewer_reference_claim=_optional_dict(
                    step_data.get("reviewer_reference_claim"),
                    f"{step_id}.reviewer_reference_claim",
                ),
```

Then add helper near `_dict`:

```python
def _optional_dict(value: Any, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = _dict(value, field_name)
    if not parsed:
        raise ConfigError(f"{field_name} must not be empty when provided")
    return parsed
```

- [ ] **Step 3: Merge proposer runtime from hard defaults, config defaults, and step override**

Add helper before `_proposer_config`:

```python
def _proposer_runtime(config: ConsensusConfig, step: ConsensusStep) -> dict[str, Any]:
    if step.reviewer_reference_claim:
        runtime = {
            "allow_internet": False,
            "allow_mcp": False,
            "codex_sandbox": "read-only",
        }
    elif step.kind == "new_calculation":
        runtime = {
            "allow_internet": True,
            "allow_mcp": True,
            "mcp_mode": "arc-only",
            "codex_sandbox": "read-only",
        }
    else:
        runtime = {
            "allow_internet": False,
            "allow_mcp": False,
            "codex_sandbox": "read-only",
        }
    runtime.update(_dict(config.defaults.get("proposer_runtime", {}), "defaults.proposer_runtime"))
    runtime.update(step.proposer_runtime)
    if runtime.get("allow_mcp") and "mcp_mode" not in runtime:
        runtime["mcp_mode"] = "arc-only"
    return runtime
```

- [ ] **Step 4: Pass runtime and reviewer reference claim into attempt config**

In `_attempt_batch_config`, add:

```python
    proposer_runtime = _proposer_runtime(config, step)
```

Change proposer/reviewer construction to:

```python
                "proposers": [
                    _proposer_config(proposer_id, runtime=proposer_runtime)
                    for proposer_id in active_proposer_ids
                ],
                "reviewers": [
                    _reviewer_config(
                        active_proposer_ids,
                        selectable_proposer_ids,
                        reviewer_reference_claim=step.reviewer_reference_claim,
                    )
                ],
```

- [ ] **Step 5: Make `_proposer_config` runtime-aware**

Change signature:

```python
def _proposer_config(proposer_id: str, *, runtime: Mapping[str, Any]) -> dict[str, Any]:
```

At top of function:

```python
    source_policy = _proposer_source_policy(runtime)
```

Replace the current paper/internet paragraph in the template with `{source_policy}` by using an f-string around the template segment:

```python
                f"{source_policy} "
```

Set runtime:

```python
        "runtime": dict(runtime),
```

Add helper:

```python
def _proposer_source_policy(runtime: Mapping[str, Any]) -> str:
    allow_mcp = bool(runtime.get("allow_mcp"))
    allow_internet = bool(runtime.get("allow_internet"))
    if not allow_mcp and not allow_internet:
        return (
            "Do not use internet search. Do not use ARC paper MCP tools. "
            "Do not read paper source sections, arXiv pages, INSPIRE pages, "
            "cached paper text, or any external source. Use only the supplied "
            "caller_context, accepted locked_outputs, and your own local algebra."
        )
    parts = []
    if allow_mcp:
        parts.append(
            "You may use ARC paper MCP tools only to read the main reference "
            "and cited sections explicitly named in caller_context."
        )
    else:
        parts.append("Do not use ARC paper MCP tools or cached paper text.")
    if allow_internet:
        parts.append(
            "Internet search is allowed only for source discovery or uncached paper access."
        )
    else:
        parts.append("Do not use internet search.")
    parts.append(
        "Cite any paper tool or internet source you use. Do not use validation-only final formulas as derivation inputs."
    )
    return " ".join(parts)
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests/test_proposers_reviewer_consensus.py::test_consensus_accepts_all_agree_on_first_attempt \
  packages/arc-llm/tests/test_proposers_reviewer_consensus.py::test_consensus_allows_opt_in_proposer_source_access \
  -q
```

Expected: these runtime tests pass; reference-claim tests still fail.

---

### Task 3: Add Reviewer-Only Reference Claim Support

**Files:**
- Modify: `packages/arc-llm/src/arc_llm/proposers_reviewer/consensus.py`

- [ ] **Step 1: Add reviewer-only prompt text**

Change `_reviewer_config` signature:

```python
def _reviewer_config(
    active_proposer_ids: list[str],
    selectable_proposer_ids: list[str],
    *,
    reviewer_reference_claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
```

Inside function, before `return`, add:

```python
    reference_instruction = _reviewer_reference_instruction(
        reviewer_reference_claim,
        active_proposer_ids=active_proposer_ids,
    )
```

Insert this into the reviewer template before `{current_proposer_outputs_json}`:

```python
                f"{reference_instruction}\n\n"
```

Add helper:

```python
def _reviewer_reference_instruction(
    reviewer_reference_claim: Mapping[str, Any] | None,
    *,
    active_proposer_ids: list[str],
) -> str:
    if not reviewer_reference_claim:
        return ""
    claim_json = json.dumps(reviewer_reference_claim, indent=2, ensure_ascii=False, sort_keys=True)
    first = active_proposer_ids[0] if len(active_proposer_ids) >= 1 else "proposer_001"
    second = active_proposer_ids[1] if len(active_proposer_ids) >= 2 else "proposer_002"
    return (
        "Reviewer-only blind reference check is active. Do not reveal the reference claim "
        "to proposers through proposer_messages. Treat A as the final result from "
        f"{first}, B as the final result from {second}, and C as reviewer_reference_claim. "
        "For A=B=C, set status=all_agree. For A=B but A-C and B-C are nonzero, "
        "set status=reference_disagrees, set agreed_proposer_ids to the agreeing proposer ids, "
        "and put the blind proposer result in accepted_result with reference_claim_status='disagrees'. "
        "If A and B disagree, do not accept the reference claim merely because one proposer matches it; "
        "set status=unresolved or all_disagree and request recalculation.\n\n"
        f"reviewer_reference_claim:\n{claim_json}"
    )
```

- [ ] **Step 2: Extend reviewer output schema status enum only when reference claim exists**

Change `_reviewer_config` output schema call:

```python
        "output_schema": _reviewer_output_schema(
            active_proposer_ids,
            selectable_proposer_ids,
            allow_reference_disagrees=bool(reviewer_reference_claim),
        ),
```

Change `_reviewer_output_schema` signature:

```python
def _reviewer_output_schema(
    active_proposer_ids: list[str],
    selectable_proposer_ids: list[str] | None = None,
    *,
    allow_reference_disagrees: bool = False,
) -> dict[str, Any]:
```

Before return, add:

```python
    status_values = ["all_agree", "two_agree", "all_disagree", "unresolved"]
    if allow_reference_disagrees:
        status_values.append("reference_disagrees")
```

Replace status enum:

```python
                            "status": {"enum": status_values},
```

- [ ] **Step 3: Pass reference claim to consensus validation**

In `_run_consensus_step`, change `_review_consensus(...)` call to include:

```python
                reviewer_reference_claim=step.reviewer_reference_claim,
```

Change `_review_consensus` signature:

```python
def _review_consensus(
    review: Mapping[str, Any],
    *,
    active_proposer_ids: list[str],
    selectable_proposer_ids: list[str] | None = None,
    proposer_outputs: Mapping[str, Any] | None = None,
    reviewer_reference_claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
```

Replace status validation with:

```python
    allowed_statuses = {"all_agree", "two_agree", "all_disagree", "unresolved"}
    if reviewer_reference_claim:
        allowed_statuses.add("reference_disagrees")
    if status not in allowed_statuses:
        raise ValueError(
            "consensus.status must be all_agree, two_agree, all_disagree, unresolved"
            + (", or reference_disagrees" if reviewer_reference_claim else "")
        )
```

- [ ] **Step 4: Validate `reference_disagrees`**

After existing `all_agree` validation block in `_review_consensus`, add:

```python
    if status == "reference_disagrees":
        _validate_best_written_selection(
            consensus,
            active_proposer_ids=active_proposer_ids,
            selectable_proposer_ids=selectable_proposer_ids,
        )
        _validate_reference_disagrees_pairwise_checks(consensus)
```

Add helper near `_validate_all_agree_pairwise_checks`:

```python
def _validate_reference_disagrees_pairwise_checks(consensus: Mapping[str, Any]) -> None:
    checks = consensus.get("pairwise_symbolic_checks")
    if not isinstance(checks, dict):
        raise ValueError("reference_disagrees requires pairwise_symbolic_checks")
    if checks.get("A_minus_B_zero") is not True:
        raise ValueError("reference_disagrees requires A-B=0 for the blind proposers")
    if checks.get("A_minus_C_zero") is not False or checks.get("B_minus_C_zero") is not False:
        raise ValueError("reference_disagrees requires A-C and B-C to be nonzero")
    method = str(checks.get("check_method", "")).strip().lower()
    if method not in {"analytic", "numerical", "mixed"}:
        raise ValueError("reference_disagrees requires check_method to be analytic, numerical, or mixed")
    if checks.get("used_sympy") is True:
        sympy_code = str(checks.get("sympy_code", "")).lower()
        if "expand" not in sympy_code or "simplify" not in sympy_code:
            raise ValueError("SymPy reference_disagrees requires expand and simplify checks")
    history = checks.get("check_history")
    if not isinstance(history, list) or not history:
        raise ValueError("reference_disagrees requires documented check history")
```

- [ ] **Step 5: Treat `reference_disagrees` as terminal accepted output**

In `_run_consensus_step`, change:

```python
        if status == "all_agree":
```

to:

```python
        if status in {"all_agree", "reference_disagrees"}:
```

- [ ] **Step 6: Run reference tests**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests/test_proposers_reviewer_consensus.py::test_reviewer_reference_claim_is_not_shared_with_proposers \
  packages/arc-llm/tests/test_proposers_reviewer_consensus.py::test_reference_disagrees_accepts_when_two_blind_proposers_agree \
  packages/arc-llm/tests/test_proposers_reviewer_consensus.py::test_reference_disagrees_requires_blind_proposer_agreement \
  -q
```

Expected: pass.

---

### Task 4: Update ARC Workflow Docs And Tests

**Files:**
- Modify: `skills/arc/references/research-workflows/research-plan.md`
- Modify: `skills/arc/references/research-workflows/research-foundation.md`
- Modify: `skills/arc/references/research-workflows/research-execute.md`
- Modify: `tests/test_arc_research_workflow_docs.py`

- [ ] **Step 1: Add workflow doc tests**

In `tests/test_arc_research_workflow_docs.py`, add:

```python
def test_research_plan_routes_reference_equations_to_blind_checks() -> None:
    text = (WF / "research-plan.md").read_text(encoding="utf-8").lower()

    assert "do not disclose the target reference equation" in text
    assert "blind reference check" in text
    assert "reviewer-only reference claim" in text


def test_research_foundation_keeps_initial_foundation_to_axioms_and_definitions() -> None:
    text = (WF / "research-foundation.md").read_text(encoding="utf-8").lower()

    assert "only definitions, axioms, conventions, and truly foundational equations" in text
    assert "do not add paper-derived equations merely so they can be checked" in text
    assert "blind reference check" in text


def test_research_execute_defaults_to_blind_no_source_reference_checks() -> None:
    text = (WF / "research-execute.md").read_text(encoding="utf-8").lower()

    assert "blind reference check" in text
    assert "reviewer_reference_claim" in text
    assert "proposer_runtime" in text
    assert '"allow_internet": false' in text
    assert '"allow_mcp": false' in text
    assert '"allow_internet": true' in text
    assert '"allow_mcp": true' in text
    assert "reference_disagrees" in text
```

Update `test_research_execute_requires_solid_symbolic_and_filtered_checks` by replacing:

```python
    assert "proposers may use arc paper mcp tools" in text
    assert "read the main reference" in text
    assert "internet" in text
```

with:

```python
    assert "proposers must not use paper tools or internet search by default" in text
    assert "explicitly configured" in text
    assert "internet" in text
```

- [ ] **Step 2: Update `research-plan.md`**

In Phase 3 after Step 4, add:

```markdown
For equations quoted from a reference or collaborator note that need checking,
do not disclose the target reference equation in `prompt`, `allowed_inputs`, or
`expected_output`. Make the step a blind reference check: proposers derive the
quantity from named dependencies, and the execute workflow supplies the target
only as a reviewer-only reference claim.
```

- [ ] **Step 3: Update `research-foundation.md`**

At start of Phase 3, add:

```markdown
The initial foundation should contain only definitions, axioms, conventions,
and truly foundational equations that are allowed as starting points. Do not add
paper-derived equations merely so they can be checked. If a reference equation
needs verification before use, keep it out of `equations[]` and route it through
a blind reference check in `research-execute.md`.
```

Keep the existing non-axiom/check-status wording for rare convention/foundation
items that still belong in foundation.

- [ ] **Step 4: Update `research-execute.md`**

In Phase 1 config example, add defaults:

```json
  "defaults": {
    "integrity_reference_path": "skills/arc/references/rules/integrity.md",
    "proposer_runtime": {
      "allow_internet": false,
      "allow_mcp": false
    }
  },
```

Add a new compact subsection after Phase 2:

```markdown
## Phase 2a: Add Blind Reference Checks

For paper or note equations that need checking, prefer blind reference checks
over `foundation_check`. Do not put the target equation in `foundation/latest.json`,
`prompt`, or `allowed_context`.

Add a `new_calculation` step with two proposers and a reviewer-only claim:

```json
{
  "step_id": "blind_ref_eq_001",
  "kind": "new_calculation",
  "prompt": "Derive the named target quantity from the supplied definitions and checked foundation items. Do not use papers, internet search, or target formulas.",
  "allowed_context": {
    "quantity_to_calculate": "target quantity name",
    "quantity_dependencies": ["dependency names"],
    "allowed_inputs": ["checked foundation ids only"]
  },
  "proposer_runtime": {
    "allow_internet": false,
    "allow_mcp": false
  },
  "reviewer_reference_claim": {
    "id": "ref_eq_001",
    "latex": "...",
    "source": {
      "paper_id": "arXiv:...",
      "section": "..."
    }
  }
}
```

For blind reference checks, proposers must not use paper tools or internet search
by default. If the user explicitly requests source access, set
`proposer_runtime.allow_mcp` or `proposer_runtime.allow_internet` to `true`.

The reviewer compares A, B, and C, where A and B are blind proposer results and
C is `reviewer_reference_claim`. Outcomes:

```text
A=B=C: reference verified.
A=B!=C: accept the blind derivation and mark `reference_disagrees`.
A!=B: proposer disagreement; recalculate or split the step.
```

For a post-check new calculation that is not checking a reference formula, turn
source access on by default unless the user requested otherwise:

```json
"proposer_runtime": {
  "allow_internet": true,
  "allow_mcp": true
}
```

- [ ] **Step 5: Run doc tests and confirm pass after docs update**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest tests/test_arc_research_workflow_docs.py -q
```

Expected: fail until packaged copies are synced, then pass after Task 5.

---

### Task 5: Sync Packaged Skill Copies

**Files:**
- Modify: `packaging/codex/arc/skills/arc/references/research-workflows/research-plan.md`
- Modify: `packaging/codex/arc/skills/arc/references/research-workflows/research-foundation.md`
- Modify: `packaging/codex/arc/skills/arc/references/research-workflows/research-execute.md`
- Modify: `packaging/claude/arc/skills/arc/references/research-workflows/research-plan.md`
- Modify: `packaging/claude/arc/skills/arc/references/research-workflows/research-foundation.md`
- Modify: `packaging/claude/arc/skills/arc/references/research-workflows/research-execute.md`

- [ ] **Step 1: Copy source workflow docs to packaged hosts**

Run:

```bash
cp skills/arc/references/research-workflows/research-plan.md \
  packaging/codex/arc/skills/arc/references/research-workflows/research-plan.md
cp skills/arc/references/research-workflows/research-foundation.md \
  packaging/codex/arc/skills/arc/references/research-workflows/research-foundation.md
cp skills/arc/references/research-workflows/research-execute.md \
  packaging/codex/arc/skills/arc/references/research-workflows/research-execute.md
cp skills/arc/references/research-workflows/research-plan.md \
  packaging/claude/arc/skills/arc/references/research-workflows/research-plan.md
cp skills/arc/references/research-workflows/research-foundation.md \
  packaging/claude/arc/skills/arc/references/research-workflows/research-foundation.md
cp skills/arc/references/research-workflows/research-execute.md \
  packaging/claude/arc/skills/arc/references/research-workflows/research-execute.md
```

- [ ] **Step 2: Verify packaged copies match**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest \
  tests/test_arc_research_workflow_docs.py::test_packaged_workflow_copies_match_source \
  tests/test_arc_research_workflow_docs.py::test_packaged_skill_references_stay_synced_with_source \
  -q
```

Expected: pass.

---

### Task 6: Full Focused Verification

**Files:**
- No edits.

- [ ] **Step 1: Run focused consensus tests**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests/test_proposers_reviewer_consensus.py -q
```

Expected: pass.

- [ ] **Step 2: Run workflow doc tests**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest tests/test_arc_research_workflow_docs.py -q
```

Expected: pass.

- [ ] **Step 3: Run combined local suite when practical**

Run:

```bash
packages/arc-paper/.venv/bin/python -m pytest \
  packages/arc-llm/tests \
  packages/arc-paper/tests \
  packages/arc-domain/tests \
  packages/arc-mcp/tests
```

Expected: pass. If runtime is too long, record which focused suites passed and why the full suite was skipped.

---

## Review Notes

This plan intentionally avoids a new workflow file or large schema migration. It adds one optional reviewer-only field and one optional proposer-runtime field to the existing consensus step shape.

Expected user-facing behavior:

- Blind reference checks default to no internet or ARC paper tools.
- Post-check new calculations default to internet and ARC MCP enabled for research context.
- A user can override either behavior through `defaults.proposer_runtime` or `steps[].proposer_runtime`.
- Paper-derived equations needing checking stay out of foundation and out of proposer prompts.
- Reviewer can compare two blind derivations against the hidden reference claim.
- `A=B!=C` becomes an accepted scientific result: the blind derivation agrees internally, while the reference claim fails or has a convention mismatch.
