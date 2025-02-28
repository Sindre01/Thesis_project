#!/bin/bash
PHASE=${1:-"validation"}
H=${2:-"0"}
M=${3:-"0"}
S=${4:-"0"}
EXAMPLES_TYPE=${5:-"N/A"}

# Get the directory where the script is located
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

# Define the target directory using an absolute path
TARGET_DIR="$(realpath "${SCRIPT_DIR}/../${PHASE}_runs/${EXAMPLES_TYPE}/")"

BRANCH="${PHASE}/${EXAMPLES_TYPE}"

echo "==== Pushing runs to github ====="
echo "Pushing changes from ${TARGET_DIR} to ${BRANCH} on GitHub..."
cd ~/Thesis_project
git fetch

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "🔍 Debug: Current branch is '${CURRENT_BRANCH}'"

# Commit non-TARGET_DIR changes if not on the target branch
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "💾 Committing all non-TARGET_DIR changes on ${CURRENT_BRANCH}..."
        git add .
        git reset "$TARGET_DIR"  # Unstage TARGET_DIR to avoid committing it
        git commit -m "🔄 Committing all non-TARGET_DIR changes before branch switch."
        git push origin "${CURRENT_BRANCH}"
    else
        echo "⚠️ No changes detected outside ${TARGET_DIR}. Skipping commit."
    fi

    echo "🔄 Switching to branch ${BRANCH}..."
    if git rev-parse --verify "${BRANCH}" >/dev/null 2>&1; then
        git checkout "${BRANCH}" || { echo "❌ Failed to checkout ${BRANCH}!"; exit 1; }
        git pull
    else
        echo "⚠️ Branch ${BRANCH} does not exist. Creating it..."
        git checkout -b "${BRANCH}"
    fi
else
    echo "✅ Already on ${CURRENT_BRANCH}, only committing ${TARGET_DIR} changes..."
fi

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