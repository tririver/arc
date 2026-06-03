#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: release-arc.sh VERSION\n' >&2
  printf 'Example: release-arc.sh 0.2.0\n' >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

pause() {
  printf '\n%s\n' "$1"
  printf 'Press Enter to continue, or Ctrl-C to abort... '
  read -r _
}

print_cmd() {
  prefix="$1"
  shift
  printf '%s:' "$prefix"
  for item in "$@"; do
    printf ' %s' "$item"
  done
  printf '\n'
}

run() {
  print_cmd RUN "$@"
  "$@"
}

run_dry() {
  print_cmd 'DRY RUN' "$@"
  "$@"
}

if [ "$#" -ne 1 ]; then
  usage
  exit 2
fi

version="$1"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  usage
  die "VERSION must be SemVer core format X.Y.Z without a leading v"
fi

tag="v${version}"
major="${version%%.*}"
rest="${version#*.}"
minor="${rest%%.*}"
next_minor=$((minor + 1))
internal_range=">=${major}.${minor},<${major}.${next_minor}"

root="$(git rev-parse --show-toplevel)"
cd "$root"

version_paths=(
  "plugins/arc/.codex-plugin/plugin.json"
  "plugins/arc/.claude-plugin/plugin.json"
  "packages/arc-llm/pyproject.toml"
  "packages/arc-paper/pyproject.toml"
  "packages/arc-domain/pyproject.toml"
  "packages/arc-typeset/pyproject.toml"
  "packages/arc-mcp/pyproject.toml"
  "packages/arc-paper/src/arc_paper/__init__.py"
  "packages/arc-mcp/src/arc_mcp/__init__.py"
  "packages/arc-paper/tests/test_import.py"
  "packages/arc-paper/tests/test_package_metadata.py"
)

existing_version_paths=()
for path in "${version_paths[@]}"; do
  if [ -e "$path" ]; then
    existing_version_paths+=("$path")
  fi
done

if [ "${#existing_version_paths[@]}" -eq 0 ]; then
  die "No ARC version files found under $root"
fi

pause "Step 1/8: preflight checks for clean worktree, upstream freshness, release commits, and tag availability."

dirty="$(git status --short --untracked-files=all)"
if [ -n "$dirty" ]; then
  printf '%s\n' "$dirty" >&2
  die "Worktree is dirty; commit or stash changes before release"
fi

branch="$(git branch --show-current)"
if [ -z "$branch" ]; then
  die "Detached HEAD; checkout a release branch before running this script"
fi

remote_name="$(git config "branch.${branch}.remote" || true)"
if [ -z "$remote_name" ]; then
  die "Branch $branch has no upstream remote"
fi

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
if [ -z "$upstream" ]; then
  die "Branch $branch has no upstream branch"
fi

run git fetch --tags "$remote_name"

local_rev="$(git rev-parse HEAD)"
upstream_rev="$(git rev-parse "$upstream")"
merge_base="$(git merge-base HEAD "$upstream")"
if [ "$local_rev" = "$upstream_rev" ]; then
  printf 'Branch %s is synchronized with %s.\n' "$branch" "$upstream"
elif [ "$local_rev" = "$merge_base" ]; then
  die "Branch is behind upstream $upstream; pull/rebase before release"
elif [ "$upstream_rev" = "$merge_base" ]; then
  ahead_count="$(git rev-list --count "${upstream}..HEAD")"
  printf 'Branch %s is ahead of %s by %s commit(s); release push will include them.\n' "$branch" "$upstream" "$ahead_count"
else
  die "Branch has diverged from upstream $upstream; reconcile before release"
fi

latest_release_tag="$(git tag --list 'v[0-9]*' --sort=-v:refname | sed -n '1p')"
if [ -n "$latest_release_tag" ]; then
  commit_count="$(git rev-list --count "${latest_release_tag}..HEAD")"
  if [ "$commit_count" = "0" ]; then
    die "No committed changes since $latest_release_tag; refusing empty release"
  fi
  printf 'Committed changes since %s: %s\n' "$latest_release_tag" "$commit_count"
else
  printf 'No existing v* release tag found; treating this as first release.\n'
fi

if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
  die "Tag already exists: $tag"
fi

pause "Step 2/8: bump plugin manifests, Python package versions, internal dependency ranges, and version tests to $version."

python3 - "$version" "$internal_range" "${existing_version_paths[@]}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

version = sys.argv[1]
internal_range = sys.argv[2]
paths = [Path(item) for item in sys.argv[3:]]

internal_dep_re = re.compile(r"(arc-(?:llm|paper|domain|typeset|mcp))>=\d+\.\d+,<\d+\.\d+")


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update version in {path}")
    path.write_text(new_text, encoding="utf-8")


def replace_all(path: Path, pattern: re.Pattern[str], replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    new_text = pattern.sub(replacement, text)
    path.write_text(new_text, encoding="utf-8")


for path in paths:
    if path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
        replace_once(path, r'^(\s*"version"\s*:\s*")[^"]+(")', rf"\g<1>{version}\2")
    elif path.name == "pyproject.toml":
        replace_once(path, r'^(version\s*=\s*")[^"]+(")', rf"\g<1>{version}\2")
        replace_all(path, internal_dep_re, rf"\1{internal_range}")
    elif path.name == "__init__.py":
        replace_once(path, r'^(__version__\s*=\s*")[^"]+(")', rf"\g<1>{version}\2")
    elif path.name == "test_import.py":
        replace_once(path, r'(__version__\s*==\s*")[^"]+(")', rf"\g<1>{version}\2")
    elif path.name == "test_package_metadata.py":
        replace_all(path, internal_dep_re, rf"\1{internal_range}")
PY

if git diff --quiet -- "${existing_version_paths[@]}"; then
  die "Version files already at $version; no release bump produced"
fi

printf '\nVersion diff:\n'
git diff -- "${existing_version_paths[@]}"

pause "Step 3/8: validate bumped metadata."

python3 - "$version" "$internal_range" "$root" <<'PY'
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

version = sys.argv[1]
internal_range = sys.argv[2]
root = Path(sys.argv[3])
packages = ["arc-llm", "arc-paper", "arc-domain", "arc-typeset", "arc-mcp"]

for manifest in [
    root / "plugins/arc/.codex-plugin/plugin.json",
    root / "plugins/arc/.claude-plugin/plugin.json",
]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("version") != version:
        raise SystemExit(f"{manifest} version mismatch")

for package in packages:
    pyproject = root / "packages" / package / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    if data["project"]["version"] != version:
        raise SystemExit(f"{pyproject} version mismatch")
    for dep in data["project"].get("dependencies", []):
        if dep.startswith("arc-") and internal_range not in dep:
            raise SystemExit(f"{pyproject} dependency mismatch: {dep}")
PY

run git diff --check -- "${existing_version_paths[@]}"
if command -v claude >/dev/null 2>&1; then
  run claude plugin validate plugins/arc
else
  printf 'SKIP: claude not found on PATH; using built-in manifest checks.\n'
fi

pause "Step 4/8: commit version bump."

run git add "${existing_version_paths[@]}"
run git commit -m "chore: release ${tag}"

pause "Step 5/8: create release tag $tag."

run git tag -a "$tag" -m "$tag"

pause "Step 6/8: dry-run push release branch and tag."

run_dry git push --dry-run "$remote_name" "HEAD:${branch}" "$tag"

pause "Step 7/8: push release branch and tag."

run git push "$remote_name" "HEAD:${branch}" "$tag"

pause "Step 8/8: dry-run then push stable branch to this release commit."

run_dry git push --dry-run "$remote_name" "HEAD:stable"
pause "Final remote mutation: push stable branch to $tag."
run git push "$remote_name" "HEAD:stable"

printf '\nRelease %s pushed. Create GitHub Release from tag %s when release notes are ready.\n' "$version" "$tag"
