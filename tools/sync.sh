#!/usr/bin/env bash
# Apply the newest downloaded iteration zip to the project, test it, commit and push.
# Usage: tools/sync.sh [path/to/zip]    (default: newest ~/Downloads/malecns-lab.zip or "malecns-lab (N).zip")
set -euo pipefail

# This script lives in the project and gets replaced by the new iteration mid-run.
# Bash reads scripts incrementally, so run from a temporary copy.
if [ -z "${MALECNS_SYNC_COPY:-}" ]; then
  self=$(mktemp "${TMPDIR:-/tmp}/malecns-sync.XXXXXX"); cp "$0" "$self"
  MALECNS_SYNC_COPY="$self" exec bash "$self" "$@"
fi

PROJECT="${MALECNS_PROJECT:-$HOME/Sites/malecns-lab}"
DOWNLOADS="${MALECNS_DOWNLOADS:-$HOME/Downloads}"
APPLIED="$DOWNLOADS/malecns-lab-applied"

cd "$PROJECT"
[ -d .git ] || { echo "Not a git repo yet. Run tools/setup_repo.sh first." >&2; exit 1; }

ZIP="${1:-}"
if [ -z "$ZIP" ]; then
  # Browsers rename repeats to "malecns-lab (1).zip", so match the prefix and take the newest.
  ZIP=$(ls -t "$DOWNLOADS"/malecns-lab.zip "$DOWNLOADS"/malecns-lab\ \(*\).zip 2>/dev/null | head -n 1 || true)
fi
[ -n "$ZIP" ] && [ -f "$ZIP" ] || { echo "No malecns-lab*.zip found in $DOWNLOADS." >&2; exit 1; }
echo "Applying $ZIP"

# Never overwrite local edits that aren't committed yet.
if [ -n "$(git status --porcelain)" ]; then
  echo "You have uncommitted changes. Commit or stash them first, then re-run:" >&2
  git status --short >&2; exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then git pull -q --ff-only; fi

TMP=$(mktemp -d); trap 'rm -rf "$TMP" "$MALECNS_SYNC_COPY"' EXIT
unzip -q "$ZIP" -d "$TMP"
SRC="$TMP/malecns-lab"
[ -f "$SRC/server.py" ] || { echo "Zip doesn't contain malecns-lab/server.py; not a lab iteration." >&2; exit 1; }

# Only apply iterations produced by this workflow: they always carry CHANGES.md and tools/sync.sh.
# (A backup such as malecns-lab_v1.zip once replaced the project because it matched a loose pattern.)
if [ "${MALECNS_FORCE:-}" != 1 ] && { [ ! -f "$SRC/CHANGES.md" ] || [ ! -f "$SRC/tools/sync.sh" ]; }; then
  echo "Refusing $(basename "$ZIP"): it has no CHANGES.md or tools/sync.sh, so it isn't an iteration from this workflow." >&2
  echo "If you really mean to apply it, run: MALECNS_FORCE=1 tools/sync.sh \"$ZIP\"" >&2
  exit 4
fi

# Refuse to apply an older iteration over a newer one: if the zip's newest CHANGES entry already
# appears below the top of the committed CHANGES.md, this zip is superseded.
if [ -f "$SRC/CHANGES.md" ] && [ -f CHANGES.md ]; then
  ZTOP=$(grep -m1 '^## ' "$SRC/CHANGES.md" || true); RTOP=$(grep -m1 '^## ' CHANGES.md || true)
  if [ -n "$ZTOP" ] && [ "$ZTOP" != "$RTOP" ] && grep -qxF "$ZTOP" CHANGES.md; then
    echo "Skipping $(basename "$ZIP"): it is an older iteration (\"${ZTOP#\#\# }\") than what's committed." >&2
    mkdir -p "$APPLIED/superseded" && mv "$ZIP" "$APPLIED/superseded/"
    exit 3
  fi
fi
rm -rf "$SRC/.git" "$SRC/.venv" "$SRC/.runtime.json"
find "$SRC" -name __pycache__ -prune -exec rm -rf {} +

# Iterations add and change files; they rarely remove any. Stop if this one would delete many.
DROP=$(comm -23 <(git ls-files | sort) <(cd "$SRC" && find . -type f | sed 's|^\./||' | sort))
NDROP=$(printf '%s' "$DROP" | grep -c . || true)
if [ "$NDROP" -gt 3 ] && [ "${MALECNS_FORCE:-}" != 1 ]; then
  echo "Refusing $(basename "$ZIP"): it would delete $NDROP tracked files:" >&2
  printf '%s\n' "$DROP" | head -n 20 >&2
  echo "If that's intended, run: MALECNS_FORCE=1 tools/sync.sh \"$ZIP\"" >&2
  exit 4
fi

# Remove tracked files that the new iteration dropped. Untracked local files (raw data, .venv) are left alone.
comm -23 <(git ls-files | sort) <(cd "$SRC" && find . -type f | sed 's|^\./||' | sort) | while IFS= read -r f; do
  git rm -q -- "$f"
done
(cd "$SRC" && tar cf - .) | tar xf -

if [ -z "$(git status --porcelain)" ]; then
  echo "This zip matches what's already committed. Nothing to do."
  mkdir -p "$APPLIED" && mv "$ZIP" "$APPLIED/"; exit 0
fi

PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
[ "$PY" = python3 ] || .venv/bin/pip install -q -r requirements.txt
echo "Running tests…"
if ! "$PY" -m unittest discover -p 'test_*.py' >/tmp/malecns-tests.log 2>&1; then
  tail -n 30 /tmp/malecns-tests.log >&2
  echo "Tests failed, so nothing was committed. Undo the files with: git checkout -- . && git clean -fd" >&2
  exit 1
fi
echo "Tests passed."

# Commit message: the newest "## " entry in CHANGES.md, with its body; else a dated default.
SUBJECT="Iteration $(date +%Y-%m-%d) from $(basename "$ZIP")"; BODY=""
if [ -f CHANGES.md ]; then
  S=$(grep -m1 '^## ' CHANGES.md | sed 's/^## //' || true)
  [ -n "$S" ] && SUBJECT="$S"
  BODY=$(awk '/^## /{n++} n==1 && !/^## /' CHANGES.md)
fi
git add -A
git commit -q -m "$SUBJECT" ${BODY:+-m "$BODY"}
echo "Committed: $SUBJECT"

if git remote get-url origin >/dev/null 2>&1; then
  git push -q && echo "Pushed to $(git remote get-url origin)"
else
  echo "No GitHub remote yet; committed locally. Run tools/setup_repo.sh to publish."
fi
mkdir -p "$APPLIED" && mv "$ZIP" "$APPLIED/$(date +%Y%m%d-%H%M%S)-$(basename "$ZIP")"
# Restart the lab server so the new iteration is live (set MALECNS_NO_RESTART=1 to skip).
if [ "${MALECNS_NO_RESTART:-}" != 1 ] && [ -x tools/lab.sh ]; then
  MALECNS_PROJECT="$PROJECT" tools/lab.sh restart || echo "Couldn't restart the lab server; see tools/lab.sh logs" >&2
fi

# Any other lab zips still in Downloads predate this one; archive them so they're never applied later.
for old in "$DOWNLOADS"/malecns-lab.zip "$DOWNLOADS"/malecns-lab\ \(*\).zip; do
  [ -f "$old" ] || continue
  mkdir -p "$APPLIED/superseded" && mv "$old" "$APPLIED/superseded/" && echo "Archived older download: $(basename "$old")"
done
