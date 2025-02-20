#!/bin/bash
PHASE=${1:-"validation"}
HOUR=${2:-"0"}
MINUTES=${3:-"0"}
TARGET_DIR="notebooks/few-shot/fox/${PHASE}_runs/"

git add "$TARGET_DIR"
git commit -m "🔄 Updated ${PHASE} runs from HPC FOX at $(date). \n⏱️ Total execution time: ${HOURS}h ${MINUTES}"
git push origin main
