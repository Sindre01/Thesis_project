#!/bin/bash
PHASE=${1:-"validation"}
H=${2:-"0"}
M=${3:-"0"}
S=${4:-"0"}
TARGET_DIR="notebooks/few-shot/fox/${PHASE}_runs/"

git add "$TARGET_DIR"
git commit -m "🔄 Updated ${PHASE} runs from HPC FOX at $(date). \n⏱️ Total execution time: ${H}h ${M}m ${S}s"
git push origin main
