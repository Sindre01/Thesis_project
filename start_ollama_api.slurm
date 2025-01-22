#!/bin/bash


###############################################################################
# Slurm Batch Script to Run Ollama Serve for Hosting an API
###############################################################################

# Job Configuration
#SBATCH --job-name=ollama_api                     # Job name
#SBATCH --account=ec12                      # Project account
#SBATCH --partition=accel                  # Partition ('accel' or 'accel_long')
#SBATCH --nodes=1                  # Amount of nodes. Ollama one support single node inference
#SBATCH --gpus=a100:1                             # Number of GPUs
#SBATCH --time=06:00:00                             # Walltime (D-HH:MM:SS)
#SBATCH --mem-per-gpu=80GB              # Memory per CPU
#SBATCH --output=ollama_api_%j.out                 # Standard output and error log


###############################################################################
# Environment Setup
###############################################################################

source /etc/profile.d/z00_lmod.sh

# Fail on errors and treat unset variables as errors
set -o errexit
set -o nounset

# Reset modules to system default
module purge
# module load CUDA/12.4.0
# module list

export OLLAMA_MODELS=/cluster/work/projects/ec12/ec-sindrre/ollama-models    # Path to where the Ollama models are stored and loaded
export OLLAMA_HOST=0.0.0.0:11434      # Host and port where Ollama listens
export OLLAMA_ORIGINS=”*”
# export OLLAMA_DEBUG=1
export OLLAMA_LLM_LIBRARY="cuda_v12_avx" 
export OLLAMA_FLASH_ATTENTION=1
# export OLLAMA_KV_CACHE_TYPE="f16" # f16 (default), q8_0 (half of the memory of f16, try this), q4_0 different quantization types to find the best balance between memory usage and quality.
# export OLLAMA_NUM_PARALLEL=2 # Number of parallel models to run. 
# export CUDA_ERROR_LEVEL=50
# export AMD_LOG_LEVEL=3

# export CUDA_VISIBLE_DEVICES=0,1


# Setup monitoring
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory 	--format=csv --loop=1 > "gpu_util-.csv" &
NVIDIA_MONITOR_PID=  # Capture PID of monitoring process


###############################################################################
# Start Ollama Server
###############################################################################

echo "Starting Ollama server..."

ollama serve

# After computation stop monitoring
kill -SIGINT ""

###############################################################################
# End of Script
###############################################################################
