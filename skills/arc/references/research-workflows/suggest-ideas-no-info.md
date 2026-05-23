# Suggest Ideas No-Info Workflow

Use this workflow to run a comparison idea-generation loop where proposers get
only the user's exact intent and internet access. This is an ablation of
`suggest-ideas.md`: do not feed proposers ARC domain Markdown files, ARC paper
tool guidance, or MCP access. Keep reviewer permissions the same as the normal
idea workflow so review quality remains comparable.

## Inputs

Read `<project-dir>/context.json`. Use the exact `user_intent`, `provider`,
and configured `model_tier` when present. Do not substitute a paraphrased
research goal into idea prompts.

### Phase 1: Prepare Comparison Run

Step 1: Create `<project-dir>/suggest-ideas/`.

Step 2: Choose a filesystem-safe `<run-id>` that makes the ablation clear, for
example `no_info_<timestamp>`. The run directory must be:

```text
<project-dir>/suggest-ideas/<run-id>/
```

Step 3: Do not require or attach `<project-dir>/domain/**/*.md` files for this
workflow. The point of this comparison is to withhold ARC-built domain context
from proposers.

### Phase 2: Prepare Idea Loop Config

Step 1: Write an `arc-llm` proposers-reviewer-loop config JSON to:

```text
<project-dir>/suggest-ideas/<run-id>.config.json
```

Use this package-level structure:

```json
{
  "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
  "run_id": "<run-id>",
  "run_dir": "<project-dir>/suggest-ideas",
  "max_concurrent_loops": 2,
  "artifact_options": {
    "save_prompts": true
  },
  "defaults": {
    "provider": "<provider>",
    "model_tier": "high"
  },
  "loops": []
}
```

Choose `model_tier: "high"` if available.

Step 2: Add the requested number of independent loops. For the comparison test,
use two loops and keep `max_concurrent_loops=2`. Future runs may increase the
loop count by adding loop objects and raising `max_concurrent_loops`.

Each loop must set:

```json
{
  "loop_id": "idea_001",
  "max_rounds": 5,
  "early_stop": {
    "enabled": false
  },
  "caller_context": {
    "user_intent": "<user_intent>",
    "comparison_condition": "no_info_proposer"
  },
  "proposers": [],
  "reviewers": []
}
```

Do not include `domain_markdown_files`, `arc_paper_tool_notes`, paper summaries,
paper ids beyond what appears in the user's intent, or any ARC-generated report
text in the proposer context.

Step 3: Configure one proposer per idea loop. The proposer must receive only
the user's exact intent, full prior correspondence, and internet permission:

```json
{
  "id": "proposer_001",
  "prompt": {
    "system": "You are a theoretical-physics researcher proposing one concrete, calculable research idea.",
    "template": "Use only the exact user_intent and full prior correspondence below. Do not assume access to ARC-built domain summaries, ARC paper tools, local paper caches, or MCP tools. You may use internet search. Propose or revise exactly one idea for the user's intent. Record literature checks and uncertainty honestly."
  },
  "output_schema": {
    "type": "object"
  },
  "runtime": {
    "allow_internet": true,
    "allow_mcp": false,
    "codex_sandbox": "read-only"
  }
}
```

Step 4: Configure exactly one reviewer per loop. Set `output_schema` to the
JSON object in
`references/research-workflows/suggest-ideas-reviewer-output.schema.json`. The
written config must contain the schema object, not the placeholder string below.
If the proposer id changes, update the schema's `proposer_messages` keys to
match. The reviewer has ARC-only MCP and internet permission:

```json
{
  "id": "reviewer_001",
  "prompt": {
    "system": "You are a skeptical but constructive theoretical-physics reviewer.",
    "template": "Review the current proposer output against the user's intent and all prior correspondence. Do not accept or reject. Search both local paper sources and the internet when judging evidence_of_novelty. ARC's local paper database can provide deep paper metadata, sections, cached full text, references, citers, equation context, and domain context; make targeted ARC checks before assigning low novelty evidence. Use ARC paper tools first, especially search_full_text for cached ar5iv full-text searches, then use internet search to catch uncached, very recent, or non-arXiv public evidence before final novelty judgment. Avoid open-ended exhaustive searching: use enough focused ARC and web queries to support the novelty mark, then summarize the evidence. Useful ARC paper tools include get_metadata, get_abstract, get_references, get_citers, get_citer_count, get_toc, get_section, search_full_text, get_equation_context, llm_get_summary, and llm_generate_summary; run --help for CLI usage or read MCP tool schemas for MCP usage. Example full-text query: search_full_text(paper_id=\"<paper-id>\", query=\"<phrase>\", context=1, limit=10). In review_payload.marks, give evidence_of_novelty on a 0-10 scale: 10 means you are completely confident from the checked evidence that the idea has not been done before, and 0 means you are completely confident it has already been done. This is evidence for the binary done/not-done question, not a score for how exotic or conceptually novel the idea feels. Give feasibility, scientific_value, user_intent_fit, and first_calculation_clarity on a 1-5 reviewer-relative scale against the best same-direction benchmark idea you can propose: 5 far better, 4 slightly better, 3 roughly equal, 2 slightly worse, 1 far worse. Decimal scores are allowed. Set total_score to the sum of all five marks. Return concrete improvement comments, evidence checked, tool queries used, and separate messages to the controller and proposer."
  },
  "output_schema": "<copy suggest-ideas-reviewer-output.schema.json here>",
  "runtime": {
    "allow_internet": true,
    "allow_mcp": true,
    "mcp_mode": "arc-only",
    "codex_sandbox": "read-only"
  }
}
```

The reviewer output must use the `arc.llm.review_envelope.v1` envelope. The
machine-readable marks belong in `review_payload.marks`. The reviewer may
request early stop in the envelope, but this workflow sets
`early_stop.enabled=false`, so all five rounds are recorded.

### Phase 3: Run Idea Loops

Step 1: Run:

```bash
arc-llm proposers-reviewer-loop \
  --config <project-dir>/suggest-ideas/<run-id>.config.json \
  --json
```

Step 2: Inspect the returned JSON. Continue only if every loop status is
`completed` or intentionally `stopped`. If any loop is `failed`, print
`WARNING:` with the failing loop id and stop before writing downstream idea
reports.

### Phase 4: Report Artifacts

Step 1: Report the idea loop artifact root:

```text
<project-dir>/suggest-ideas/<run-id>/
```

Step 2: Report each loop's `state.json`, `transcript.jsonl`, per-round
proposer outputs, per-round reviews, per-round prompt artifacts, and any
per-round worker error artifacts.

Step 3: Do not invent final idea rankings, novelty claims, or gap scores from
memory. Any later idea selection or report must read the recorded loop
artifacts and cite the relevant rounds and reviewer marks.

Step 4: Write `<project-dir>/suggest-ideas/<run-id>/report.md` for this phase.
In the main text, outline the generated ideas and their stated motivation,
calculation target, required checks, and main reviewer concerns. Do not hide
weak or worse later rounds.

The appendix must start with a rounds-and-marks summary table. Include one row
per loop and round, with columns for `loop_id`, `round`, idea title or short
label, `evidence_of_novelty`, `feasibility`, `scientific_value`,
`user_intent_fit`, `first_calculation_clarity`, and `total_score`. After the
table, append the detailed correspondence history grouped by loop and round:
proposer output, reviewer message to the controller, reviewer message to the
proposer, and full `review_payload`. Full rendered prompts are debug artifacts
under each round's `prompts/` directory; mention their paths when useful, but do
not paste them into the correspondence appendix by default.
