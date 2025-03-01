#!/bin/bash


###############################################################################
# Slurm Batch Script to Run Ollama Serve for Hosting an API
###############################################################################

# Job Configuration
#SBATCH --job-name=validation_few-shot_similarity         # Job name
#SBATCH --account=ec12                      # Project account
#SBATCH --partition=accel                  # Partition ('accel' or 'accel_long')
#SBATCH --nodes=1                           # Amount of nodes. Ollama one support single node inference
#SBATCH --nodelist=gpu-9,gpu-7,gpu-8                   # List of nodes that the job can run on
#SBATCH --gpus=a100:1                             # Number of GPUs
#SBATCH --time=0-24:00:00                             # Walltime (D-HH:MM:SS)
#SBATCH --mem-per-gpu=80G              # Memory per CPU
#SBATCH --output=Job_validation_%j.out                 # Standard output and error log


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

export OLLAMA_MODELS=/cluster/work/projects/ec12/ec-sindrre/ollama-models    # Path to where the Ollama models are stored and loaded
export OLLAMA_HOST=0.0.0.0:11434      # Host and port where Ollama listens
export OLLAMA_ORIGINS=”*”
export OLLAMA_LLM_LIBRARY="cuda_v12_avx" 
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE="f16" # f16 (default), q8_0 (half of the memory of f16, try this), q4_0 different quantization types to find the best balance between memory usage and quality.

# export OLLAMA_DEBUG=1
# export OLLAMA_NUM_PARALLEL=2 # Number of parallel models to run. 
# export OLLAMA_MAX_LOADED_MODELS
# export OLLAMA_MAX_QUEUE

# export CUDA_ERROR_LEVEL=50
# export CUDA_VISIBLE_DEVICES=0,1
# export AMD_LOG_LEVEL=3
#############CLEANUP OLD JOBS################
# Set the target directory (default: current directory)


# Set the age threshold (1 hours)
AGE_THRESHOLD=3600 # 1 hours in seconds

echo "🧹 Cleaning up files older than 1 hours in: /fp/homes01/u01/ec-sindrre/slurm_jobs/few-shot/validation/similarity/regular"

# Find and delete .out, .slurm, and .csv files older than 1 hours
find "/fp/homes01/u01/ec-sindrre/slurm_jobs/few-shot/validation/similarity/regular" -type f \( -name "*.out" -o -name "*.slurm" -o -name "*.csv" \) -mmin +1800 -exec rm -v {} \;

echo "✅ Cleanup completed!"

####################### Setup monitoring ######################################
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory 	--format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=  # Capture PID of monitoring process


###############################################################################
# Start Ollama Server in Background with Log Redirection
###############################################################################
ollama serve > ollama_API_$SLURM_JOB_ID.out 2>&1 &  

sleep 5

###############################################################################
# Run Python Script
###############################################################################

echo "============= Pulling latest changes from Git... ============="

# Check if the repository already exists
if [ -d "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID/.git" ]; then
    echo "✅ Repository already exists: /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID"
    cd "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID" || { echo "❌ Failed to enter /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID"; exit 1; }
    
    # Pull latest changes
    echo "🔄 Pulling latest changes..."

else
    echo "🚀 Cloning repository..."
    git clone https://github.com/Sindre01/Thesis_project.git "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID" || { echo "❌ Clone failed!"; exit 1; }
    
    echo "✅ Repository cloned to /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID"
    cd "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID" || { echo "❌ Failed to enter /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID"; exit 1; }
fi

echo "🔍 Debug: Current Git repository is:"
git rev-parse --show-toplevel

export GIT_DIR="/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID/.git"
export GIT_WORK_TREE="/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID"

branch_name="validation/similarity"
if [ -n "regular" ]; then
    branch_name="${branch_name}-regular"
fi

if git rev-parse --verify --quiet "refs/heads/${branch_name}"; then
    git checkout "${branch_name}"
else
    git checkout -b "${branch_name}"
    git push --set-upstream origin "${branch_name}"  # Set remote tracking
fi

git reset --hard HEAD  # Ensure a clean state
git pull --rebase --autostash || { echo "❌ Git pull failed!"; exit 1; }

#PULL changes to Main git repo as well
# git --git-dir=~/Thesis_project/.git --work-tree=~/Thesis_project/ pull origin main
source ~/Thesis_project/thesis_venv/bin/activate  # Activate it to ensure the correct Python environment

echo "============= Running validation few-shot Python script... ============="
export PYTHONPATH="/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID:~/Thesis_project:~/Thesis_project:"
python -u /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID/notebooks/few-shot/fox/run_validation.py     --model_provider 'ollama'     --models '[
    "llama3.3:70b-instruct-fp16",
    "qwen2.5:72b-instruct-fp16",
]'     --experiments '[
        {
            "name": "regular_similarity",
            "prompt_prefix": "Create a function",
            "num_shots": [1, 5, 10],
            "prompt_type": "regular",
            "semantic_selector": true,
        },
        
]'     > /fp/homes01/u01/ec-sindrre/slurm_jobs/few-shot/validation/similarity/regular/AI_$SLURM_JOB_ID.out 2>&1

# Cleanup after job completion
echo "🚀 Cleaning up cloned repository..."
rm -rf "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID"
echo "✅ Repository removed: /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_similarity_$SLURM_JOB_ID"

###############################################################################
# End of Script
###############################################################################
