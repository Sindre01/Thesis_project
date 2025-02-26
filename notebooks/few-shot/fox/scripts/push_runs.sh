#!/bin/bash
PHASE=${1:-"validation"}
H=${2:-"0"}
M=${3:-"0"}
S=${4:-"0"}
EXAMPLES_TYPE=${5:-"N/A"}

# Get the directory where the script is located
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

# Define the target directory using an absolute path
TARGET_DIR="$(realpath "${SCRIPT_DIR}/../${PHASE}_runs/")" # ${EXAMPLES_TYPE}/")"

BRANCH="${PHASE}/${EXAMPLES_TYPE}"

echo "==== Pushing runs to github ====="
echo "Pushing changes from ${TARGET_DIR} to ${BRANCH} on GitHub..."
cd ~/Thesis_project
git fetch

# Stash only TARGET_DIR changes (leave other uncommitted changes untouched)
if ! git diff --quiet "$TARGET_DIR"; then
    git stash push -m "Saving changes" -- "$TARGET_DIR"
fi

# Checkout the branch (create if missing)
if git rev-parse --verify "${BRANCH}" >/dev/null 2>&1; then
    git checkout -f "${BRANCH}"
    git pull
    # Apply stash only if one exists
    if git stash list | grep -q "Saving changes"; then
        git stash pop
    fi
  
else
    echo "⚠️ Branch ${BRANCH} does not exist. Creating it..."
    git checkout -b "${BRANCH}"
fi

# Stage only TARGET_DIR changes (ignores any changes from other jobs)
git add --intent-to-add "$TARGET_DIR" && git reset  # Ensures only TARGET_DIR is tracked
git add "$TARGET_DIR"

# Check if there are actual changes before committing
if ! git diff --cached --exit-code >/dev/null; then
    git commit -m "🔄 New ${PHASE} runs from HPC Fox." -m "⏱️ Total execution time: ${H}h ${M}m ${S}s"
    git push origin "${BRANCH}"
    echo "✅ Push completed successfully!"
else
    echo "⚠️ No changes detected. Skipping commit."
fi
