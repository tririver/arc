# Suggest Ideas Workflow

Use this workflow to run ARC idea-generation loops from project-local domain
context. The loop execution is owned by `arc-llm`; this skill prepares
caller-specific prompts, permissions, and artifacts.

## Inputs

Read `<project-dir>/context.json`. Use the exact `user_intent`, `provider`,
configured `model_tier` when present, and existing domain artifact paths from
the project. Do not substitute a paraphrased research goal into idea prompts.

### Phase 1: Ensure Domain Context

Step 1: Complete `references/research-workflows/build-domain.md` first. Case 2
depends on Case 1.

Step 2: Verify that `<project-dir>/domain/` contains the domain summaries and
foundation summaries produced by the domain workflow. If a domain artifact is
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
    "domain_artifacts": [],
    "arc_paper_tool_notes": ""
  },
  "proposers": [],
  "reviewers": []
}
```

Step 5: Configure one proposer per idea loop for the current idea workflow.
The proposer must receive the user's exact intent and internet permission:

```json
{
  "id": "proposer_001",
  "prompt": {
    "system": "You are a theoretical-physics researcher proposing one concrete, calculable research idea.",
    "template": "Use the caller context and the full prior correspondence below. Propose or revise exactly one idea for the user's intent. Record literature checks and uncertainty honestly."
  },
  "output_schema": {
    "type": "object"
  },
  "runtime": {
    "allow_internet": true
  }
}
```

Optionally give the proposer domain summaries and ARC paper-tool access notes
by adding them to `caller_context`. When ARC paper access is allowed, set
`runtime.allow_mcp=true` and instruct the proposer to use ARC paper tools
before internet search whenever it wants to check papers, citations, sections,
or equation context.

Step 6: Configure exactly one reviewer per loop. The reviewer has all
available permissions:

```json
{
  "id": "reviewer_001",
  "prompt": {
    "system": "You are a skeptical but constructive theoretical-physics reviewer.",
    "template": "Review the current proposer output against the user's intent and all prior correspondence. Do not accept or reject. Return improvement comments, marks, and separate messages to the controller and proposer."
  },
  "output_schema": {
    "type": "object"
  },
  "runtime": {
    "allow_internet": true,
    "allow_mcp": true
  }
}
```

The reviewer output must use the `arc.llm.review_envelope.v1` envelope. The
idea-specific marks and comments belong in `review_payload`. The reviewer may
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
proposer outputs, and per-round reviews.

Step 3: Do not invent final idea rankings, novelty claims, or gap scores from
memory. Any later idea selection or report must read the recorded loop
artifacts and cite the relevant rounds and reviewer marks.
