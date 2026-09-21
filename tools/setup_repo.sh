#!/usr/bin/env bash
# One-time setup: turn the project folder into a git repo and publish it on GitHub.
# Safe to re-run; each step is skipped once done.
set -euo pipefail

PROJECT="${MALECNS_PROJECT:-$HOME/Sites/malecns-lab}"
REPO="${MALECNS_REPO:-merolaagi/malecns-lab}"
VISIBILITY="${MALECNS_VISIBILITY:-private}"   # set MALECNS_VISIBILITY=public to publish openly

cd "$PROJECT"
[ -f server.py ] || { echo "No server.py in $PROJECT. Set MALECNS_PROJECT to the lab folder." >&2; exit 1; }

# Keep local-only files out of the repo even if the current .gitignore is older.
for pattern in .venv/ .runtime.json __pycache__/ .DS_Store raw-data/ '*.feather'; do
  grep -qxF "$pattern" .gitignore 2>/dev/null || echo "$pattern" >> .gitignore
done

if [ ! -d .git ]; then
  git init -q -b main
  echo "Initialized git in $PROJECT"
fi

# Refuse to commit anything large (for example raw tables stored under another name).
big=$(git ls-files --others --exclude-standard -z | xargs -0 -I{} find {} -type f -size +50M 2>/dev/null || true)
if [ -n "$big" ]; then
  echo "These untracked files are over 50 MB; add them to .gitignore first:" >&2; echo "$big" >&2; exit 1
fi

if ! git rev-parse -q --verify HEAD >/dev/null; then
  git add -A
  git commit -q -m "Baseline: MaleCNS lab prototype before atlas" 
  echo "Committed the current folder as the baseline."
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv and installing requirements…"
  python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote already set: $(git remote get-url origin)"
  git push -u origin main
elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo create "$REPO" "--$VISIBILITY" --source . --remote origin --push \
    --description "Measured MaleCNS connectome subsets with assumed dynamics: locomotion LIF model, odor-learning module and circuit atlas"
else
  cat <<MSG
GitHub CLI isn't installed or signed in, so the repo wasn't created.
Either: brew install gh && gh auth login, then re-run this script,
or create an empty repo named ${REPO#*/} on github.com and run:
  git remote add origin git@github.com:$REPO.git && git push -u origin main
MSG
  exit 1
fi
echo "Done. Repo: https://github.com/$REPO"
