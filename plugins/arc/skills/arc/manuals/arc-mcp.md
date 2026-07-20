# Optional ARC MCP Adapter

ARC workflows use Skills and CLI commands by default. Read this manual only
when the user explicitly installed the separate `arc-mcp` companion plugin or
explicitly requested MCP.

The base `arc` plugin does not contain an MCP manifest, install the `mcp`
Python dependency, or start a background server. The companion plugin is
self-contained and runs `arc-mcp` from its isolated `mcp` runtime profile.

## Installation And Diagnosis

Install `arc-mcp` from the same ARC marketplace only after installing the base
plugin when Skill workflows are also desired. The MCP plugin can start without
a plugin-to-plugin dependency.

Prewarm or diagnose its private runtime from the companion plugin root:

```bash
./bin/arc-runtime setup --profile mcp
./bin/arc-runtime doctor --profile mcp
```

If setup failed after its cause was fixed, retry explicitly:

```bash
./bin/arc-runtime setup --profile mcp --retry
```

## Behavior

- MCP tool names and result envelopes remain compatible with the ARC 0.9
  adapter.
- Long operations are persisted by the protocol-neutral `arc-jobs` package.
  Follow the returned `next.cli_command`; it uses the same ARC runtime instead
  of relying on the caller's `PATH`.
- File, project-directory, output, and resource-path arguments must be
  absolute. Relative paths are rejected because an MCP server's working
  directory may be its plugin cache rather than the user's project.
- `ARC_JOBS_DIR` controls shared job persistence under `ARC_HOME/jobs`;
  `ARC_JOBS_CACHE` remains a legacy layout override. MCP-only inline timeout
  settings remain under `ARC_MCP_INLINE_WAIT_SEC`,
  `ARC_MCP_TOOL_TIMEOUT_SEC`, and `ARC_MCP_BACKGROUND_MARGIN_SEC`.
- Installing MCP does not change internal ARC worker policy: proposer,
  reviewer, calculation, and companion workers still receive evidence through
  their controllers rather than direct MCP access.

Use `manuals/arc-jobs.md` for submission, status, result, cancellation, and
Markdown-to-PDF background jobs. Do not route ordinary Skill workflows through
MCP merely because the optional adapter happens to be installed.
