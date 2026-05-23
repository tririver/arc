# Suggest Ideas Workflow

Use this workflow to run ARC idea-generation loops from project-local domain
context. The loop execution is owned by `arc-llm`; this skill prepares
caller-specific prompts, permissions, and artifacts.

## Inputs

Read `<project-dir>/context.json`. Use the exact `user_intent`, `provider`,
configured `model_tier` when present, and existing domain artifact paths from
the project. Do not substitute a paraphrased research goal into idea prompts.

### Phase 1: Ensure Domain Context

Step 1: Complete Case 1 (building domain references with
`references/research-workflows/build-domain.md`) before Case 2 (suggesting
research ideas from a not-yet-explicit request).

Step 2: Verify that `<project-dir>/domain/` contains the domain summaries and
foundation summaries produced by the domain workflow. At least one
`<project-dir>/domain/**/*.md` file must be present. If a domain artifact is
missing, print `WARNING:` and stop before idea generation.

### Phase 2: Prepare Idea Loop Config

Step 1: Create `<project-dir>/suggest-ideas/`.

Step 2: Choose a filesystem-safe `<run-id>` for this idea run. The run
directory must be:

```text
<project-dir>/suggest-ideas/<run-id>/
```

Step 3: Write an `arc-llm` proposers-reviewer-loop config JSON to:

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

Step 4: Add the requested number of independent loops. For the initial test,
use two loops and keep `max_concurrent_loops=2`. Future runs may increase the
loop count by adding more loop objects and raising `max_concurrent_loops`.
Keep `artifact_options.save_prompts=true` for debugging unless the project
context explicitly disables prompt artifacts.

Attach all Markdown files under `<project-dir>/domain/` to `caller_context`,
including all reports and summaries from multiple built domains. Sort them by
relative path and include their full Markdown text. Do not attach HTML, JSON,
logs, lock files, cache files, transcripts, or unrelated project notes.

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
    "domain_markdown_files": [
      {
        "path": "domain/<relative-path>.md",
        "content": "<full markdown text>"
      }
    ],
    "arc_paper_tool_notes": "ARC paper MCP tools are available. Use them before internet search when checking papers, citations, references, sections, cached full-text terms, equation context, or nearby literature. Especially use search_full_text to search cached ar5iv full text for relevant words or phrases. Example MCP call: search_full_text(paper_id=\"<paper-id>\", query=\"<phrase>\", context=1, limit=10). CLI equivalent: arc-paper search-full-text <paper-id> --query \"<phrase>\" --context 1 --limit 10 --json. Useful paper tools include get_metadata, get_abstract, get_references, get_citers, get_citer_count, get_toc, get_section, search_full_text, get_equation_context, llm_get_summary, and llm_generate_summary. For CLI usage, run arc-paper <command> --help; for MCP usage, read the host-provided tool descriptions and input schema."
  },
  "proposers": [],
  "reviewers": []
}
```

Step 5: Configure one proposer per idea loop for the current idea workflow.
The proposer must receive the user's exact intent, attached domain Markdown
files, ARC paper-tool guidance, ARC-only MCP permission, and internet
permission:

```json
{
  "id": "proposer_001",
  "prompt": {
    "system": "You are a theoretical-physics researcher proposing one concrete, calculable research idea.",
    "template": "Use the exact user_intent, domain_markdown_files, arc_paper_tool_notes, and full prior correspondence below. Propose or revise exactly one idea for the user's intent. Use ARC paper tools before internet search when checking papers, citations, references, cached full-text terms, equation context, or nearby literature. Use search_full_text for cached ar5iv full-text checks. Record literature checks and uncertainty honestly."
  },
  "output_schema": {
    "type": "object"
  },
  "runtime": {
    "allow_internet": true,
    "allow_mcp": true,
    "mcp_mode": "arc-only",
    "codex_sandbox": "read-only"
  }
}
```

Step 6: Configure exactly one reviewer per loop. Set `output_schema` to the
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
    "template": "Review the current proposer output against the user's intent, arc_paper_tool_notes, and all prior correspondence. Do not accept or reject. Search both local paper sources and the internet when judging evidence_of_novelty. Use ARC paper tools first, especially search_full_text for cached ar5iv full-text searches, then use internet search when needed. In review_payload.marks, give evidence_of_novelty on a 0-10 scale: 10 means you are completely confident from the checked evidence that the idea has not been done before, and 0 means you are completely confident it has already been done. This is evidence for the binary done/not-done question, not a score for how exotic or conceptually novel the idea feels. Give feasibility, scientific_value, user_intent_fit, and first_calculation_clarity on a 1-5 reviewer-relative scale against the best same-direction benchmark idea you can propose: 5 far better, 4 slightly better, 3 roughly equal, 2 slightly worse, 1 far worse. Decimal scores are allowed. Set total_score to the sum of all five marks. Return concrete improvement comments, evidence checked, tool queries used, and separate messages to the controller and proposer."
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
Collect each loop's recorded proposer outputs and reviewer reviews from the
artifact root. In the main text, outline the generated ideas and their stated
motivation, calculation target, required checks, and main reviewer concerns. Do
not hide weak or worse later rounds.

The appendix must start with a rounds-and-marks summary table. Include one row
per loop and round, with columns for `loop_id`, `round`, idea title or short
label, `evidence_of_novelty`, `feasibility`, `scientific_value`,
`user_intent_fit`, `first_calculation_clarity`, and `total_score`. After the
table, append the detailed correspondence history grouped by loop and round:
proposer output, reviewer message to the controller, reviewer message to the
proposer, and full `review_payload`. Full rendered prompts are debug artifacts
under each round's `prompts/` directory; mention their paths when useful, but do
not paste them into the correspondence appendix by default.
