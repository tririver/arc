#!/usr/bin/env bash
set -euo pipefail

python -m build packages/arc-llm
python -m build packages/arc-paper
python -m build packages/arc-domain
python -m build packages/arc-typeset
python -m build packages/arc-mcp
