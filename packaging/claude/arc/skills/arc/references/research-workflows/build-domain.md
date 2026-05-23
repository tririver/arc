# Build Domain Workflow

Use this workflow to build project-local research-domain references from one or
more seed papers.

## Required References

Read these before executing:

- `references/rules/interaction.md`
- `references/rules/integrity.md`
- `references/package-manuals/arc-paper.md`
- `references/package-manuals/arc-domain.md`
- `references/package-manuals/arc-mcp.md`

## Inputs

Read `<project-dir>/context.json`. Use the exact values from that file for all
ARC calls, especially `user_intent`, `seed_paper_list`, `provider`, `model`,
`workers`, and `refresh`.

### Phase 1: Prepare Project Artifacts

Step 1: Create `<project-dir>/domain/`.

Step 2: Preserve `<project-dir>/context.json` as the workflow source of truth.
Do not substitute a paraphrased intent string into ARC calls.

### Phase 2: Build Domain Caches

Step 1: For each `<seed-paper>` in `seed_paper_list`, call the MCP tool
`llm_domain_build` with:

```text
seed_paper=<seed-paper>
intent=<user-intent>
provider=<provider>
model=<model>
refresh=<refresh>
workers=<workers>
background=true
```

Step 2: If the MCP response contains `status: "job_running"` and `job_id`,
immediately run:

```bash
arc-mcp jobs watch <job-id> --json
```

Step 3: Inspect the returned JSON body. Do not treat command exit code alone as
success. Continue only when the job result is successful. If the job failed,
was cancelled, or returned `needs_llm`, print `WARNING:` with the reason and
stop.

Step 4: If MCP is unavailable, use the blocking CLI fallback:

```bash
arc-domain llm-build <seed-paper> --intent "<user-intent>" --provider <provider> --model <model> --workers <workers> --json
```

Add `--refresh` only when `refresh` is true. Omit `--model` when `model` is not
set. Inspect the returned JSON before continuing.

### Phase 3: Copy Domain Artifacts

Step 1: Derive a safe file prefix:

```bash
arc-paper safe-dir-name <seed-paper> --json
```

Step 2: Read domain artifact paths from the successful build result or from:

```bash
arc-domain status <seed-paper> --intent "<user-intent>" --json
arc-domain get-summary <seed-paper> --intent "<user-intent>" --json
arc-domain get-graph <seed-paper> --intent "<user-intent>" --json
```

Step 3: Copy or write project-local files:

```text
<project-dir>/domain/<seed-safe>_domain.html
<project-dir>/domain/<seed-safe>_domain_summary.json
<project-dir>/domain/<seed-safe>_domain_summary.md
```

Use the graph HTML path for the HTML file. Use the domain summary JSON for the
JSON file. Render a concise Markdown summary from the JSON for the Markdown
file.

### Phase 4: Summarize Foundation Papers

Step 1: Read the selected foundation paper id from the domain build result or
domain summary JSON.

Step 2: For each foundation paper, call MCP `llm_get_summary` with
`background=true`.

Step 3: If the response contains `status: "job_running"` and `job_id`, run:

```bash
arc-mcp jobs watch <job-id> --json
```

Step 4: Inspect the JSON result. If successful, write a project-local Markdown
summary:

```text
<project-dir>/domain/foundation_<foundation-safe>.md
```

Derive `<foundation-safe>` with `arc-paper safe-dir-name <foundation-paper>
--json`.

Do not depend on copying a cache file. Cache-hit responses may not include a
stable file path, so write the returned summary content into the project.

### Phase 5: Interactive Review

Step 1: In `interactive` mode, show the domain artifact paths and ask with the
discrete selection protocol whether to continue, rebuild, or `Let's discuss`.

Step 2: In `auto` mode, continue without asking unless a warning or failure
occurred.
