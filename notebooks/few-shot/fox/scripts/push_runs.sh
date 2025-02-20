#!/bin/bash
PHASE=${1:-"validation"}

TARGET_DIR="notebooks/few-shot/fox/${PHASE}_runs/"

git add "$TARGET_DIR"
git commit -m "🔄 Updated ${PHASE} runs from HPC at $(date)"
git push origin main
