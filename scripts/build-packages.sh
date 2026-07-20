#!/usr/bin/env bash
set -euo pipefail

rm -rf dist
mkdir -p dist

python -m build --outdir dist packages/arc-llm
python -m build --outdir dist packages/arc-jobs
python -m build --outdir dist packages/arc-paper
python -m build --outdir dist packages/arc-domain
python -m build --outdir dist packages/arc-typeset
python -m build --outdir dist packages/arc-companion
python -m build --outdir dist packages/arc-mcp
