#!/bin/bash


###############################################################################
# Slurm Batch Script to Run Ollama Serve for Hosting an API
###############################################################################

# Job Configuration
#SBATCH --job-name=testing_context_RAG_regular_2        # Job name
#SBATCH --account=ec30                      # Project account
#SBATCH --partition=ifi_accel                  # Partition ('accel' or 'accel_long')
#SBATCH --nodes=1                           # Amount of nodes. Ollama one support single node inference
#SBATCH --array=0-2  
#SBATCH --ntasks=1
#SBATCH --nodelist=                   # List of nodes that the job can run on
#SBATCH --gpus=2                             # Number of GPUs
#SBATCH --time=2-0:00:00                             # Walltime (D-HH:MM:SS)
#SBATCH --mem-per-gpu=20G              # Memory per CPU
#SBATCH --output=Job_testing_%j.out                 # Standard output and error log



###############################################################################
# Environment Setup
###############################################################################

source /etc/profile.d/z00_lmod.sh

# Fail on errors and treat unset variables as errors
set -o errexit
set -o nounset

# Reset modules to system default
module purge
module load Python/3.11.5-GCCcore-13.2.0
# module load CUDA/12.4.0

source ~/.bashrc # may ovewrite previous modules
OLLAMA_PORT_K_FOLD=$((11390 + $SLURM_ARRAY_TASK_ID))

export OLLAMA_MODELS=/cluster/work/projects/ec12/ec-sindrre/ollama-models    # Path to where the Ollama models are stored and loaded
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT_K_FOLD    # Host and port where Ollama listens
export OLLAMA_ORIGINS="*"
export OLLAMA_LLM_LIBRARY="cuda_v12_avx" 
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE="f16" # f16 (default), q8_0 (half of the memory of f16, try this), q4_0 different quantization types to find the best balance between memory usage and quality.

# export OLLAMA_DEBUG=1
# export OLLAMA_NUM_PARALLEL=2 # Number of parallel models to run. 
# export OLLAMA_MAX_LOADED_MODELS=2
# export OLLAMA_MAX_QUEUE

# export CUDA_ERROR_LEVEL=50
# export CUDA_VISIBLE_DEVICES=0,1
# export AMD_LOG_LEVEL=3

#############CLEANUP OLD JOBS################
# Set the target directory (default: current directory)


# Set the age threshold 4 days
AGE_THRESHOLD=345600

echo "🧹 Cleaning up files older than 4 days in: /fp/homes01/u01/ec-sindrre/slurm_jobs/context/testing/RAG/regular"

# Find and delete .out, .slurm, and .csv files older than 1 hours
find "/fp/homes01/u01/ec-sindrre/slurm_jobs/context/testing/RAG/regular" -type f \( -name "*.out" -o -name "*.slurm" -o -name "*.csv" \) -mmin +1800 -exec rm -v {} \;

echo "✅ Cleanup completed!"

####################### Setup monitoring ######################################
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory 	--format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=  # Capture PID of monitoring process


###############################################################################
# Start Ollama Server in Background with Log Redirection
###############################################################################
ollama serve > ollama_API_${SLURM_JOB_ID}_fold_$SLURM_ARRAY_TASK_ID.out 2>&1 &  

sleep 5

###############################################################################
# Run Python Script
###############################################################################

echo "============= Pulling latest changes from Git... ============="

# Check if the repository already exists
if [ -d "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID/.git" ]; then
    echo "✅ Repository already exists: /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID"
    cd "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID" || { echo "❌ Failed to enter /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID"; exit 1; }
    
    # Pull latest changes
    echo "🔄 Pulling latest changes..."

else
    echo "🚀 Cloning repository..."
    git clone https://github.com/Sindre01/Thesis_project.git "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID" || { echo "❌ Clone failed!"; exit 1; }
    
    echo "✅ Repository cloned to /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID"
    cd "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID" || { echo "❌ Failed to enter /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID"; exit 1; }
fi

echo "🔍 Debug: Current Git repository is:"
git rev-parse --show-toplevel

export GIT_DIR="/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID/.git"
export GIT_WORK_TREE="/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID"

git checkout main

git reset --hard HEAD  # Ensure a clean state
git pull --rebase --autostash || { echo "❌ Git pull failed!"; exit 1; }

source ~/Thesis_project/thesis_venv/bin/activate  # Activate it to ensure the correct Python environment

# Define a cleanup function
cleanup() {
    echo "⚠️ Job failed or completed — cleaning up /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID"
    rm -rf "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID"
    echo "✅ Repository removed: /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID"
}

# Ensure cleanup is called on exit (both success or error)
trap cleanup EXIT

echo "============= Running testing context Python script for Fold ... ============="
export PYTHONPATH="/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID:~/Thesis_project:~/Thesis_project:"
python -u /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_RAG_$SLURM_JOB_ID/experiments/context/fox/run_testing.py     --model_provider 'ollama'     --models '[
    "phi4:14b-fp16"
]'     --experiments '[
        {
            "name": "regular_RAG",
            "prompt_prefix": "Create a function",
            "num_shots": [5, 10],
            "prompt_type": "regular",
            "semantic_selector": true
        }
]'     --ollama_port $OLLAMA_PORT_K_FOLD     --fold $SLURM_ARRAY_TASK_ID     > /fp/homes01/u01/ec-sindrre/slurm_jobs/context/testing/RAG/regular/AI_${SLURM_JOB_ID}_fold_$SLURM_ARRAY_TASK_ID.out 2>&1


###############################################################################
# End of Script
###############################################################################
