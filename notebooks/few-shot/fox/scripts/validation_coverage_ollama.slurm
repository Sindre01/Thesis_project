#!/bin/bash


###############################################################################
# Slurm Batch Script to Run Ollama Serve for Hosting an API
###############################################################################

# Job Configuration
#SBATCH --job-name=validation_few-shot_coverage         # Job name
#SBATCH --account=ec12                      # Project account
#SBATCH --partition=accel                  # Partition ('accel' or 'accel_long')
#SBATCH --nodes=1                           # Amount of nodes. Ollama one support single node inference
#SBATCH --gpus=2                             # Number of GPUs
#SBATCH --time=00-02:00:00                             # Walltime (D-HH:MM:SS)
#SBATCH --mem-per-gpu=20G              # Memory per CPU
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


####################### Setup monitoring ######################################
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory 	--format=csv --loop=1 > "gpu_util-.csv" &
NVIDIA_MONITOR_PID=  # Capture PID of monitoring process


###############################################################################
# Start Ollama Server in Background with Log Redirection
###############################################################################
ollama serve > ollama_API.out 2>&1 &  

sleep 5

###############################################################################
# Run Python Script
###############################################################################
echo "============= Pulling latest changes from git... ============="

#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=job_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=standard

echo "============= Pulling latest changes from Git... ============="

# Check if the repository already exists
if [ -d "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID/.git" ]; then
    echo "✅ Repository already exists: /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID"
    cd "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID" || { echo "❌ Failed to enter /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID"; exit 1; }
    
    # Pull latest changes
    echo "🔄 Pulling latest changes..."

else
    echo "🚀 Cloning repository..."
    git clone https://github.com/Sindre01/Thesis_project.git "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID" || { echo "❌ Clone failed!"; exit 1; }
    
    echo "✅ Repository cloned to /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID"
    cd "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID" || { echo "❌ Failed to enter /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID"; exit 1; }
fi

git checkout "validation/coverage"
git reset --hard HEAD  # Ensure a clean state
git pull --rebase --autostash || { echo "❌ Git pull failed!"; exit 1; }

source ~/Thesis_project/thesis_venv/bin/activate  # Activate it to ensure the correct Python environment

echo "============= Running validation few-shot Python script... ============="
python -u /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID/notebooks/few-shot/fox/run_validation.py > /fp/homes01/u01/ec-sindrre/slurm_jobs/few-shot/validation/coverage/validation.out 2>&1

# Cleanup after job completion
echo "🚀 Cleaning up cloned repository..."
rm -rf "/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID"
echo "✅ Repository removed: /fp/homes01/u01/ec-sindrre/tmp/Thesis_project_coverage_$SLURM_JOB_ID"

###############################################################################
# End of Script
###############################################################################
