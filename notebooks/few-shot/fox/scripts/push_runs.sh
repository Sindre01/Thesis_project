#!/bin/bash
PHASE=${1:-"validation"}
H=${2:-"0"}
M=${3:-"0"}
S=${4:-"0"}

# Get the directory where the script is located
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

# Define the target directory using an absolute path
TARGET_DIR="$(realpath "${SCRIPT_DIR}/../${PHASE}_runs/")"

echo "==== Pushing runs to github ====="
echo "Pushing changes from ${TARGET_DIR} to GitHub..."

# Verify that TARGET_DIR exists
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "❌ Error: TARGET_DIR does not exist! Aborting."
    exit 1
fi

# Add changes
git add "$TARGET_DIR"

# Check if there are actual changes before committing
if ! git diff --cached --exit-code >/dev/null; then
    git commit -m "🔄 New ${PHASE} runs from HPC Fox." -m "⏱️ Total execution time: ${H}h ${M}m ${S}s"
    git push origin main
    echo "✅ Push completed successfully!"
else
    echo "⚠️ No changes detected. Skipping commit."
fi
