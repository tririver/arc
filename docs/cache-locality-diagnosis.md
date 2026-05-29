# Prompt Cache Locality 诊断与修复方案

> **问题**: Python 确定性编排重构后，DeepSeek V4 Pro 的 prompt cache miss rate 急剧上升，LLM 推理费用显著增长。
> **时间窗口**: 2026-05-27 下半日至 2026-05-28（UTC），与 DeepSeek 计费后台观测一致。
> **根因**: 三次提交系统性地破坏了 prompt prefix 的跨请求共享。
> **分析基准**: 本诊断基于 repo 当前 HEAD — commit `9379ae2` (2026-05-29 16:47 HKT, "Enhance check and plan workflows with clearer task specifications and metadata requirements")，结合 `git log` 回溯至 5 月 25 日的完整历史。

---

## 1. 问题概述

### 1.1 背景

ARC 的 proposers-reviewer 共识系统通过 Python 代码确定性控制 subagent 分发：几个 proposer、收到什么任务、如何聚合结果。重构提高了可复现性，但 LLM 推理费用急剧上升。

### 1.2 缓存机制

DeepSeek V4 Pro 使用 **prefix caching**：如果两个连续请求的 prompt 共享一个长前缀（由 `messages` 数组的首个元素决定），后续请求只需为前缀之外的新 tokens 付费。对于 DeepSeek 的 `json_object` 模式，`messages[0]` 是 system message——它包含了完整的 JSON output schema。

### 1.3 损失量化

| 指标 | 重构前 | 重构后 | 变化 |
|---|---|---|---|
| System message 大小 (proposer) | 112 chars (~37 tokens) | 1785 chars (~595 tokens) | **16x** |
| Reviewer 跨重算共享前缀比例 | 100% (完全共享) | 18.5% (315/1702 chars) | **-81.5%** |
| Proposer template 文本 | 静态 (MCP/internet 始终开启) | 动态 (3 种 source policy) | 非确定性 |

---

## 2. 根因分析（按影响严重性排序）

### 🥇 根因 1: System Message 膨胀 16x

**提交**: `a60ad06` (2026-05-28 23:59 UTC) — "Add arc_llm_call_record for tracking LLM call details"

**变更** (`packages/arc-llm/src/arc_llm/proposers_reviewer/consensus.py`):

```diff
- "output_schema": {"type": "object"},
+ "output_schema": _proposer_output_schema(),
```

`_proposer_output_schema()` 生成一个 60+ 行的详细 JSON Schema，包含:
- 6 个 required 字段
- 嵌套 `plan_foundation_assessment` 对象 (5 required fields, 9 enum 值)
- `additionalProperties: true`

**缓存破坏机制:**

DeepSeek `json_object` 模式将 schema 嵌入 `messages[0]`（system message）:

```
# 重构前 (37 tokens):
[system] "Return exactly one JSON object... Schema: {type: object}"
[user]   ...

# 重构后 (595 tokens):
[system] "Return exactly one JSON object... Schema: {
           type: object,
           required: [result_summary, derivation, assumptions, ...],
           properties: {
             result_summary: {type: string},
             ...
             plan_foundation_assessment: {
               required: [needs_revision, issue_type, ...],
               properties: {
                 issue_type: {enum: [none, foundation_inadequate, ...]}
               }
             }
           }
         }"
[user]   ...
```

影响:
- **系统消息大小从 37 tokens 增至 595 tokens (16x)**
- 即使 prefix 被缓存，每次请求仍为更长的不可缓存后缀付费
- 长前缀可能超出 DeepSeek 的 prefix cache 容量限制，被更频繁地 evict
- 如果多个不同 schema 竞争缓存空间，eviction 更严重

### 🥈 根因 2: Reviewer Schema 随 active_proposer_ids 变化

**提交**: `d8a6082` (2026-05-27 00:26 UTC) — "Implement blind reference checks in consensus workflow"

**变更**: `_reviewer_output_schema` 新增 `selectable_proposer_ids` 参数，schema 中的 enum 约束依赖于动态的 proposer ID 列表:

```python
# consensus.py, _reviewer_output_schema:
"agreed_proposer_ids": {
    "type": "array",
    "items": {"enum": active_proposer_ids},  # ← 随重算变化!
},
"recalculate_proposer_ids": {
    "type": "array",
    "items": {"enum": active_proposer_ids},  # ← 随重算变化!
},
"best_written_proposer_id": {
    "anyOf": [
        {"enum": selectable_proposer_ids},   # ← 随重算变化!
        {"type": "null"},
    ]
},
```

**缓存破坏机制:**

当 `two_agree` 触发重算时，`active_proposer_ids` 从 3 个缩减为 1 个:

```
Round 1 reviewer schema:  active=["proposer_001", "proposer_002", "proposer_003"]  → 1702 chars
Round 2 reviewer schema:  active=["proposer_003"]                                  → 1390 chars
共享前缀:                 仅 315 chars (18.5%)
```

→ **Reviewer 跨重算尝试的 cache hit rate = 0%**

### 🥉 根因 3: Proposer Source Policy 变动态文本

**提交**: `2984cc7` (2026-05-27 01:58 UTC) — "Enhance ARC workflows with new check functionality"

**变更**: 引入 `_proposer_source_policy(runtime)`，根据 step 类型生成不同的 source 访问策略:

```python
# consensus.py, _proposer_source_policy:
if not allow_mcp and not allow_internet:
    return "Do not use internet search. Do not use ARC paper MCP tools. ..."
# vs.
if allow_mcp:
    parts.append("You may use ARC paper MCP tools only to read the main reference ...")
```

这个动态文本嵌入在 proposer 的 `prompt.template` 中，在 `{caller_context_json}` 之前。不同 step kind（`new_calculation` vs `foundation_check`）产生不同的 user message 前缀。

### 根因 4: Parallel Dispatch → Cache Stampede

**位置**: `runner.py`, `_run_proposers`:

```python
with ThreadPoolExecutor(max_workers=len(loop.proposers)) as executor:
    future_by_proposer = {
        executor.submit(_call_json_runner_with_error_artifact, ...)
        for proposer in loop.proposers
    }
```

所有 proposer 在同一 round 内同时发出 API 请求。虽然它们共享相同的 system message（proposer schema 相同），但是如果这些并行请求在 DeepSeek 的 prefix cache 尚未被第一个完成的请求 populate 之前到达，它们全部遇到 cold start → 全部 cache miss。

### 根因 5: Retry 修改 System Message

**位置**: `openai_compatible.py`, `_json_retry_request`:

```python
if request["messages"] and request["messages"][0]["role"] == "system":
    request["messages"][0]["content"] = f"{request['messages'][0]['content']}\n\n{retry_note}"
```

每次 JSON 解析失败重试时，system message 被修改 → 不同的前缀 → 必然 cache miss。

**位置**: `runner.py`, `_review_validation_retry_prompt`:

```python
return f"{prompt.rstrip()}\n\n## Reviewer Output Retry\n..."
```

Validation 重试修改 user message → 如果 provider 缓存粒度覆盖 user message 前缀，也会导致 cache miss。

### 根因 6: 缺少 Token/Cache 观测

`openai_compatible.py` 的 `_create` 方法:

```python
completion = client.chat.completions.create(**request)
return _first_message_content(completion)
# ↑ 丢弃了 completion.usage (prompt_tokens, completion_tokens,
#   prompt_cache_hit_tokens, prompt_cache_miss_tokens)
```

没有记录 cache 指标使得无法从生产数据直接验证以上假设。

---

## 3. 修复方案

### 3.1 P0: 恢复轻量 Output Schema

**问题**: System message 中的大型 JSON Schema（595 tokens）是前缀膨胀的首要原因。

**方案**: 将 `output_schema` 恢复为 `{"type": "object"}`，将结构化输出要求移到 prompt template（user message 末尾）:

```python
# _proposer_config 中:
"output_schema": {"type": "object"},  # 恢复轻量
"prompt": {
    "template": (
        "... existing template ...\n\n"
        "## Required Output Structure\n"
        "Return exactly one JSON object with these fields:\n"
        "- result_summary (string): ...\n"
        "- derivation (string): ...\n"
        "- assumptions (string, array, or object): ...\n"
        "- validity_scope (string): ...\n"
        "- final_result: ...\n"
        "- plan_foundation_assessment (object):\n"
        "  - needs_revision (boolean)\n"
        "  - issue_type: one of none, foundation_inadequate, ...\n"
        "  - ...\n"
        "{caller_context_json}"
    ),
}
```

**为什么有效**: System message 回到 37 tokens → 所有 proposers 和 reviewers 共享相同的短前缀 → cache hit rate 回升到重构前水平。

**风险**: 失去 JSON Schema 的 strict validation。但 `_validate_review_envelope` 和 `_review_consensus` 中已有应用层验证，且 proposer 输出本就有 `additionalProperties: true`——schema 的约束力有限。

### 3.2 P0: Reviewer Schema 固定化

**问题**: `_reviewer_output_schema` 中的 enum 值随 `active_proposer_ids` 变化。

**方案**: 将动态 enum 替换为通用类型:

```diff
# _reviewer_output_schema 中:
- "agreed_proposer_ids": {"type": "array", "items": {"enum": active_proposer_ids}},
+ "agreed_proposer_ids": {"type": "array", "items": {"type": "string"}},

- "recalculate_proposer_ids": {"type": "array", "items": {"enum": active_proposer_ids}},
+ "recalculate_proposer_ids": {"type": "array", "items": {"type": "string"}},

- "best_written_proposer_id": {"anyOf": [{"enum": selectable_proposer_ids}, {"type": "null"}]},
+ "best_written_proposer_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
```

`proposer_messages.required` 也因包含 proposer IDs 而变化，但这在 `properties` 内部——如果 `additionalProperties: true` 且 required 字段的顺序不影响缓存行为，可以保留。保险起见:

```diff
- "required": active_proposer_ids,
+ # 移除 required 约束（由 _validate_review_envelope 在应用层检查）
```

**为什么有效**: Reviewer schema 对所有 `active_proposer_ids` 组合完全一样 → reviewer 的 system message 在所有重算尝试间共享前缀 → cache hit。

**安全**: `_validate_review_envelope` (runner.py line 395-410) 已有应用层验证:
```python
missing = [proposer.id for proposer in loop.proposers if proposer.id not in proposer_messages]
if missing:
    raise ValueError(f"review.proposer_messages missing: {', '.join(missing)}")
```

### 3.3 P1: Retry 不修改前缀

```python
# openai_compatible.py, _json_retry_request:
def _json_retry_request(model, prompt, json_mode, *, schema, error, attempt):
    request = _json_request(model, prompt, json_mode, schema=schema)
    retry_note = (
        f"\n\n## Repair Attempt {attempt}\n"
        f"Previous response was invalid JSON. Error: {error}"
    )
    # 追加到 user message 末尾而非修改 system message
    request["messages"][-1]["content"] = request["messages"][-1]["content"] + retry_note
    return request
```

同样，`_review_validation_retry_prompt` 的修改也应该追加到末尾。

### 3.4 P1: Cache Warm-up (可选)

在 `_run_proposers` 中，在并行分发前先发一个同步请求:

```python
# runner.py, _run_proposers:
# 先用第一个 proposer 做 cache warm-up
first = loop.proposers[0]
_call_json_runner_with_error_artifact(
    json_runner, prompts[first.id], worker=first, ...
)
# 此时 DeepSeek prefix cache 已被 populate
# 然后再并行发其余请求
with ThreadPoolExecutor(max_workers=len(loop.proposers) - 1) as executor:
    # ... (其余 proposers)
```

或者：将并行 dispatch 改为顺序 dispatch 带 staggered delay（50ms）。

### 3.5 P1: Token/Cache 日志

在 `openai_compatible.py._create` 中记录 usage:

```python
def _create(self, request):
    # ...
    completion = client.chat.completions.create(**request)
    usage = _field(completion, "usage") or {}
    self._last_usage = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
        "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens", 0),
    }
    return _first_message_content(completion)
```

并在 `_call_record` 中包含这些指标。

### 3.6 P2: 确定性序列化加固

```python
# openai_compatible.py:264:
json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True)

# artifacts.py:126:
json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
```

---

## 4. 目标 Prompt 架构

```
[SYSTEM MESSAGE]                       ← ~37 tokens，所有 worker 完全相同
  "Return exactly one JSON object and no other text."

[USER MESSAGE]
  ## ARC Worker Instructions
  ### System                            ← worker.prompt.system (静态)
  ### Task                              ← worker.prompt.template (静态)
    ... {caller_context_json} ...       ← 动态 payload 在 template 末尾
  ## Output Structure                   ← 结构化要求 (静态，所有 proposers 共享)
    - result_summary (string): ...
    - derivation (string): ...
    ...
```

关键原则:
1. **System message 固定、短小**
2. **所有静态内容在动态内容之前**
3. **动态内容仅出现在 user message 末尾**
4. **所有同类型 worker (proposer/reviewer) 共享完全相同的 system message 和 task 前缀**

---

## 5. 验证计划

### 5.1 未来验证实验设计

| Test | System Message | Schema Location | Dispatch | 预期 Cache Hit |
|---|---|---|---|---|
| A (current production) | 595-token schema | system msg | parallel | 低 (根因1+2) |
| B (proposed fix) | 37-token fixed | user msg 末尾 | parallel | 高 |
| C (only fix schema) | 37-token fixed | user msg 末尾 | parallel | 高 |
| D (only fix dispatch) | 595-token schema | system msg | sequential | 中 |
| E (full fix) | 37-token fixed | user msg 末尾 | sequential | 最高 |

### 5.2 每个测试需记录

- `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` (每请求)
- 前缀哈希 (2k, 8k 字符): `hashlib.sha256(prompt[:N].encode()).hexdigest()[:16]`
- 总 cost
- 多个步骤（含 two_agree 重算场景）

---

## 6. 相关提交

| 提交 | 日期 (UTC) | 描述 | 缓存影响 |
|---|---|---|---|
| `d8a6082` | May 27 00:26 | Blind reference checks — reviewer schema 变动态 | Reviewer 跨重算 cache miss |
| `2984cc7` | May 27 01:58 | Source policy 变动态文本 | Proposer template 前缀非确定性 |
| `04ba614` | May 27 HKT | Accepted prior step outputs | caller_context 进一步差异化 |
| `a60ad06` | **May 28 23:59** | **_proposer_output_schema()** — **system message 16x 膨胀** | **主要根因** |

---

## 7. 关键代码位置

| 文件 | 行号 | 函数/内容 | 问题 |
|---|---|---|---|
| `consensus.py` | 548-599 | `_proposer_output_schema()` | 大型 schema → system msg 膨胀 |
| `consensus.py` | 765-896 | `_reviewer_output_schema()` | 动态 enum → 跨重算前缀不同 |
| `consensus.py` | 602-627 | `_proposer_source_policy()` | 动态文本 → 不同 step kind 前缀不同 |
| `openai_compatible.py` | 245-256 | `_json_messages()` | Schema 嵌入 system message |
| `openai_compatible.py` | 259-265 | `_json_only_system_message()` | 缺少 `sort_keys=True` |
| `openai_compatible.py` | 176-197 | `_json_retry_request()` | Retry 修改 system msg 前缀 |
| `openai_compatible.py` | 75-86 | `_create()` | 丢弃 usage 数据 |
| `runner.py` | 254 | `ThreadPoolExecutor` | 并行 dispatch → stampede |
| `runner.py` | 321-329 | `_review_validation_retry_prompt()` | 重试修改 prompt |
| `prompts.py` | 49-59 | `render_prompt()` | Context 在末尾 (✅ 正确模式) |
