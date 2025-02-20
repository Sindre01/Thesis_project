#!/bin/bash
PHASE=${1:-"validation"}
git add ./$PHASE_runs/
git commit -m "🔄 Updated ${PHASE} runs from HPC at $(date)"
git push origin main
