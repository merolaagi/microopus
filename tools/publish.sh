#!/usr/bin/env bash
# Commits this iteration, tags it with VERSION, pushes to GitHub, and attaches the uniquely named archive as a release.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="merolaagi/microopus"
VERSION="$(cat VERSION)"
ARCHIVE="$(cat ARCHIVE)"
ARCHIVE_PATH="$HOME/Downloads/$ARCHIVE"

[ -d .git ] || { git init -q; git branch -M main; }
if ! git remote get-url origin >/dev/null 2>&1; then
  if command -v gh >/dev/null 2>&1 && ! gh repo view "$REPO" >/dev/null 2>&1; then
    gh repo create "$REPO" --public --description "Micro Opus: a 7,440-weight transformer showing where every weight enters the forward pass"
  fi
  git remote add origin "https://github.com/$REPO.git"
fi

git add -A
git commit -q -m "Micro Opus $VERSION ($ARCHIVE)" || echo "No file changes to commit"
git tag -f "$VERSION" >/dev/null
git push -u origin main
git push -f origin "$VERSION"

if command -v gh >/dev/null 2>&1 && [ -f "$ARCHIVE_PATH" ]; then
  if gh release view "$VERSION" -R "$REPO" >/dev/null 2>&1; then
    gh release upload "$VERSION" "$ARCHIVE_PATH" -R "$REPO" --clobber
  else
    gh release create "$VERSION" "$ARCHIVE_PATH" -R "$REPO" -t "Micro Opus $VERSION" -n "Archive: $ARCHIVE"
  fi
fi
echo "Pushed $VERSION to https://github.com/$REPO"
