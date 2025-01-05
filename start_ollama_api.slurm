#!/bin/bash

###############################################################################
# Slurm Batch Script to Run Ollama Serve for Hosting an API
###############################################################################

# Job Configuration
#SBATCH --job-name=ollama_api                     # Job name
#SBATCH --account=ec12                      # Project account
#SBATCH --partition=accel                  # Partition ('accel' or 'accel_long')
#SBATCH --gpus=2                             # Number of GPUs
#SBATCH --time=01:00:00                             # Walltime (D-HH:MM:SS)
#SBATCH --mem-per-cpu=16G              # Memory per CPU
#SBATCH --output=ollama_api_%j.out                 # Standard output and error log

###############################################################################
# Environment Setup
###############################################################################

# Fail on errors and treat unset variables as errors
set -o errexit
set -o nounset

# Reset modules to system default
# module --quiet purge

export OLLAMA_MODELS=/fp/projects01/ec12/ec-sindrre/cache/ollama    # Path to where the Ollama models are stored and loaded
export OLLAMA_HOST=0.0.0.0:11434      # Host and port where Ollama listens
# export OLLAMA_DEBUG=1


###############################################################################
# Start Ollama Server
###############################################################################

echo "Starting Ollama server on $(hostname)..."

ollama serve

###############################################################################
# End of Script
###############################################################################
