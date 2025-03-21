#!/bin/bash
EXPERIMENT=${1:-"few-shot"}
PHASE=${2:-"validation"}
EXPERIMENT_TYPE=${3:-""}
PROMPT_TYPE=${4:-"regular"}
H=${5:-"0"}
M=${6:-"0"}
S=${7:-"0"}
FOLD=${8:-"-1"}

EXPERIMENT_DIR="${EXPERIMENT_TYPE}/${PROMPT_TYPE}"
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath "${SCRIPT_DIR}/../../")"

TARGET_DIR="${ROOT_DIR}/experiments/${EXPERIMENT}/fox/${PHASE}_runs/${EXPERIMENT_DIR}"

mkdir -p "$TARGET_DIR"

TARGET_DIR="$(realpath "$TARGET_DIR")"
echo "TARGET_DIR is now: $TARGET_DIR"

if [[ "$FOLD" == "-1" ]]; then
    REMOTE_DIR="/fp/homes01/u01/ec-sindrre/slurm_jobs/${EXPERIMENT}/${PHASE}/${EXPERIMENT_DIR}/runs/hold_out/"
    BRANCH="${PHASE}/${EXPERIMENT}-${EXPERIMENT_TYPE}-${PROMPT_TYPE}"
else
    REMOTE_DIR="/fp/homes01/u01/ec-sindrre/slurm_jobs/${EXPERIMENT}/${PHASE}/${EXPERIMENT_DIR}/runs/3_fold/"
    BRANCH="${PHASE}/${EXPERIMENT}-${EXPERIMENT_TYPE}-${PROMPT_TYPE}-fold_${FOLD}"
fi
echo "==== Pushing runs to GitHub ====="
echo "Pushing changes from ${TARGET_DIR} to ${BRANCH} on GitHub..."
git pull
echo "🔍 Debug: Current Git repository is:"
git rev-parse --show-toplevel

#  Get the current branch
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "🔍 Debug: Current branch is '${CURRENT_BRANCH}'"

#  Ensure we are on the correct branch
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    echo "❌ Not on branch ${BRANCH}. Switching..."
    if git rev-parse --verify "${BRANCH}" >/dev/null 2>&1; then
        git checkout "${BRANCH}" || { echo "❌ Failed to checkout ${BRANCH}!"; exit 1; }
        git pull
    else
        echo "⚠️ Branch ${BRANCH} does not exist. Creating it..."
        git checkout -b "${BRANCH}"
        git push --set-upstream origin "${BRANCH}"
    fi
else
    echo "✅ Already on branch ${CURRENT_BRANCH}."
fi

# Ensure TARGET_DIR exists
if [ ! -d "${TARGET_DIR}" ]; then
    echo "📂 TARGET_DIR does not exist. Creating: ${TARGET_DIR}"
    mkdir -p "${TARGET_DIR}"
fi

# Load files from REMOTE_DIR into TARGET_DIR
if [[ "$FOLD" == "-1" ]]; then
    echo "📥 Loading files from ${REMOTE_DIR} into '${TARGET_DIR}'..."
    rsync -av "${REMOTE_DIR}/" "${TARGET_DIR}/"
else
    echo "📥 Loading 'fold_${FOLD}.json' files from ${REMOTE_DIR} into '${TARGET_DIR}'..."
    rsync -av --include='*/' --include="*fold_${FOLD}.json" --exclude='*' "${REMOTE_DIR}/" "${TARGET_DIR}/"
fi

# Stage changes only in TARGET_DIR
git add --intent-to-add "$TARGET_DIR" && git reset  # Ensures only TARGET_DIR is tracked
git add "$TARGET_DIR"

# Check if there are actual changes before committing
if ! git diff --cached --exit-code >/dev/null; then
    git commit -m "🔄 New ${PHASE} runs from HPC Fox." -m "⏱️ Total execution time: ${H}h ${M}m ${S}s"
    git push origin "${BRANCH}"
    echo "✅ Push completed successfully!"
else
    echo "⚠️ No new changes in ${TARGET_DIR}. Skipping commit."
fi
