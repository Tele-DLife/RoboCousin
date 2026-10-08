#!/usr/bin/env bash
# Quick commit from repo root (respects .gitignore). Optional push.
# Usage:
#   ./script/git_commit.sh
#   ./script/git_commit.sh -m "your message"
#   ./script/git_commit.sh -m "msg" -p          # also git push
#   ./script/git_commit.sh --push               # default timestamp message + push
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MSG=""
DO_PUSH=0

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    -m|--message)
      MSG="${2:-}"
      if [[ -z "$MSG" ]]; then echo "error: -m needs a message"; exit 1; fi
      shift 2
      ;;
    -p|--push) DO_PUSH=1; shift ;;
    *) break ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "error: unexpected arguments: $*"
  usage 1
fi

if [[ -z "$MSG" ]]; then
  MSG="chore: sync $(date -Iseconds)"
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "error: not a git repository"
  exit 1
fi

# Staged + unstaged + untracked (ignored paths stay ignored)
git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit (working tree clean or only ignored changes)."
  exit 0
fi

git commit -m "$MSG"

if [[ "$DO_PUSH" -eq 1 ]]; then
  branch="$(git branch --show-current 2>/dev/null || true)"
  if [[ -z "$branch" ]]; then
    echo "error: detached HEAD; push manually"
    exit 1
  fi
  git push -u origin "$branch"
fi

echo "Done."
