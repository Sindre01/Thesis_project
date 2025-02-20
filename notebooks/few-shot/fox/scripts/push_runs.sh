#!/bin/bash
PHASE=${1:-"validation"}
H=${2:-"0"}
M=${3:-"0"}
S=${4:-"0"}

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the target directory using an absolute path
TARGET_DIR="${SCRIPT_DIR}/../${PHASE}_runs/"
echo "Pushing changes from ${TARGET_DIR} to GitHub..."

git add "$TARGET_DIR"
git commit -m "🔄 New ${PHASE} runs from HPC Fox." -m "⏱️ Total execution time: ${H}h ${M}m ${S}s"
git push origin main
