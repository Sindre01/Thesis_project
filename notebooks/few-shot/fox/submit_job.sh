#!/usr/bin/env bash

###############################################################################
# Script to Automate Launching Ollama API on Fox HPC Cluster Using sbatch
###############################################################################

# Configuration
EXPERIMENT="few-shot"                    # Experiment ('few-shot' or 'COT')
PHASE="validation"                       # Phase ('testing' or 'validation')
EXAMPLES_TYPE="similarity"                 #'coverage' or 'similarity'
PROMPT_TYPE="signature"                 # 'regular' or 'cot' or 'signature'   
# SEMANTIC_SELECTOR=true                   # Use semantic selector
USER="ec-sindrre"                        # Your Educloud username
HOST="fox.educloud.no"                   # Fox login address (matches SSH config)
SSH_CONFIG_NAME="fox"                    # Name of the SSH config entry
ACCOUNT="ec12"                           # Fox project account
PARTITION="accel"                        # 'accel' or 'accel_long' (or 'ifi_accel' if access to ec11,ec29,ec30,ec34,ec35 or ec232)
GPUS=a100:2                            # a100 have 40GB or 80GB VRAM, while rtx30 have 24GB VRAM.
NODES=1                                  # Number of nodes. OLLAMA does currently only support single node inference
NODE_LIST=gpu-9,gpu-7,gpu-8          # List of nodes that the job can run on gpu-9,gpu-7,gpu-8
TIME="0-24:00:00"                       # Slurm walltime (D-HH:MM:SS)
MEM_PER_GPU="80G"                       # Memory per GPU. 
OLLAMA_MODELS_DIR="/cluster/work/projects/ec12/ec-sindrre/ollama-models"  # Path to where the Ollama models are stored and loaded                      
LOCAL_PORT="11434"                        # Local port for forwarding
OLLAMA_PORT="11434"                       # Remote port where Ollama listens. If different parallell runs, change ollama_port to avoid conflicts if same node is allocated.
SBATCH_SCRIPT="${PHASE}_${EXAMPLES_TYPE}_ollama.slurm"           # Slurm batch script name
# Directory on Fox to store scripts and output
if [ -n "$PROMPT_TYPE" ]; then
    REMOTE_DIR="/fp/homes01/u01/ec-sindrre/slurm_jobs/${EXPERIMENT}/${PHASE}/${EXAMPLES_TYPE}/${PROMPT_TYPE}"
else
    REMOTE_DIR="/fp/homes01/u01/ec-sindrre/slurm_jobs/${EXPERIMENT}/${PHASE}/${EXAMPLES_TYPE}"
fi
#--exclusive #Job will not share nodes with other jobs. 
# Define unique folder name
CLONE_DIR="/fp/homes01/u01/ec-sindrre/tmp/Thesis_project_${EXAMPLES_TYPE}_\$SLURM_JOB_ID"
##############Experiment config################
model_provider='ollama'
# experiments='[
#         {
#             "name": "regular_coverage",
#             "prompt_prefix": "Create a function",
#             "num_shots": [1, 5, 10],
#             "prompt_type": "regular",
#             "semantic_selector": false
#         }
# ]'
experiments='[
        {
            "name": "signature_similarity",
            "prompt_prefix": "Create a function",
            "num_shots": [10],
            "prompt_type": "signature",
            "semantic_selector": true
        }
]'
# models='[
#     "phi4:14b-fp16",
#     "qwen2.5:14b-instruct-fp16",
#     "qwq:32b-preview-fp16",
#     "qwen2.5-coder:32b-instruct-fp16"
# ]'
models='[
    "qwen2.5:72b-instruct-fp16"
]'

# normal* c1-[5-28]
# accel gpu-[1-2,4-5,7-9,11-13]
# accel_long gpu-[1-2,7-8]
# mig gpu-10
# bigmem bigmem-[1-2],c1-29
# ifi_accel gpu-[4-6,11]
# ifi_dj_accel gpu-12
# ifi_bigmem c1-29
# pods c1-[6-7]
# fhi_bigmem bigmem-[1-2]
# hf_accel gpu-9

#80GB a100: gpu-9, gpu-7, gpu-8?

###############################################################################
# Step 1: Create the Slurm Batch Script Locally
###############################################################################

echo $'\n==== Creating Slurm batch script locally ===='

cat <<EOT > "./scripts/${SBATCH_SCRIPT}"
#!/bin/bash


###############################################################################
# Slurm Batch Script to Run Ollama Serve for Hosting an API
###############################################################################

# Job Configuration
#SBATCH --job-name=${PHASE}_${EXPERIMENT}_${EXAMPLES_TYPE}         # Job name
#SBATCH --account=${ACCOUNT}                      # Project account
#SBATCH --partition=${PARTITION}                  # Partition ('accel' or 'accel_long')
#SBATCH --nodes=${NODES}                           # Amount of nodes. Ollama one support single node inference
#SBATCH --nodelist=${NODE_LIST}                   # List of nodes that the job can run on
#SBATCH --gpus=${GPUS}                             # Number of GPUs
#SBATCH --time=${TIME}                             # Walltime (D-HH:MM:SS)
#SBATCH --mem-per-gpu=${MEM_PER_GPU}              # Memory per CPU
#SBATCH --output=Job_${PHASE}_%j.out                 # Standard output and error log


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

export OLLAMA_MODELS=${OLLAMA_MODELS_DIR}    # Path to where the Ollama models are stored and loaded
export OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}      # Host and port where Ollama listens
export OLLAMA_ORIGINS=”*”
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


# Set the age threshold (1 hours)
AGE_THRESHOLD=$((1 * 3600)) # 1 hours in seconds

echo "🧹 Cleaning up files older than 1 hours in: $REMOTE_DIR"

# Find and delete .out, .slurm, and .csv files older than 1 hours
find "$REMOTE_DIR" -type f \( -name "*.out" -o -name "*.slurm" -o -name "*.csv" \) -mmin +1800 -exec rm -v {} \;

echo "✅ Cleanup completed!"

####################### Setup monitoring ######################################
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
	--format=csv --loop=1 > "gpu_util-\$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process


###############################################################################
# Start Ollama Server in Background with Log Redirection
###############################################################################
ollama serve > ollama_API_\$SLURM_JOB_ID.out 2>&1 &  

sleep 5

###############################################################################
# Run Python Script
###############################################################################

echo "============= Pulling latest changes from Git... ============="

# Check if the repository already exists
if [ -d "$CLONE_DIR/.git" ]; then
    echo "✅ Repository already exists: $CLONE_DIR"
    cd "$CLONE_DIR" || { echo "❌ Failed to enter $CLONE_DIR"; exit 1; }
    
    # Pull latest changes
    echo "🔄 Pulling latest changes..."

else
    echo "🚀 Cloning repository..."
    git clone https://github.com/Sindre01/Thesis_project.git "$CLONE_DIR" || { echo "❌ Clone failed!"; exit 1; }
    
    echo "✅ Repository cloned to $CLONE_DIR"
    cd "$CLONE_DIR" || { echo "❌ Failed to enter $CLONE_DIR"; exit 1; }
fi

echo "🔍 Debug: Current Git repository is:"
git rev-parse --show-toplevel

export GIT_DIR="$CLONE_DIR/.git"
export GIT_WORK_TREE="$CLONE_DIR"

# branch_name="$PHASE/$EXAMPLES_TYPE"
# if [ -n "$PROMPT_TYPE" ]; then
#     branch_name="\${branch_name}-$PROMPT_TYPE"
# fi

# if git rev-parse --verify --quiet "refs/heads/\${branch_name}"; then
#     git checkout "\${branch_name}"
# else
#     git checkout -b "\${branch_name}"
#     git push --set-upstream origin "\${branch_name}"  # Set remote tracking
# fi

git checkout main

git reset --hard HEAD  # Ensure a clean state
git pull --rebase --autostash || { echo "❌ Git pull failed!"; exit 1; }

#PULL changes to Main git repo as well
# git --git-dir=~/Thesis_project/.git --work-tree=~/Thesis_project/ pull origin main
source ~/Thesis_project/thesis_venv/bin/activate  # Activate it to ensure the correct Python environment

echo "============= Running ${PHASE} ${EXPERIMENT} Python script... ============="
export PYTHONPATH="${CLONE_DIR}:$PYTHONPATH"
python -u ${CLONE_DIR}/notebooks/${EXPERIMENT}/fox/run_${PHASE}.py \
    --model_provider '${model_provider}' \
    --models '${models}' \
    --experiments '${experiments}' \
    > ${REMOTE_DIR}/AI_\$SLURM_JOB_ID.out 2>&1

# Cleanup after job completion
echo "🚀 Cleaning up cloned repository..."
rm -rf "$CLONE_DIR"
echo "✅ Repository removed: $CLONE_DIR"

###############################################################################
# End of Script
###############################################################################
EOT

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to create Slurm batch script locally."
    exit 1
fi

echo "Slurm batch script '${SBATCH_SCRIPT}' created locally."

###############################################################################
# Step 2: Transfer the Batch Script to Fox
###############################################################################

echo $'\n==== Transferring Slurm batch script to Fox ===='

# Ensure the remote directory exists
ssh "${SSH_CONFIG_NAME}" "mkdir -p '${REMOTE_DIR}'"

# Transfer the batch script to the remote directory
scp "./scripts/${SBATCH_SCRIPT}" "${SSH_CONFIG_NAME}:'${REMOTE_DIR}/'"

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to transfer './scripts/${SBATCH_SCRIPT}' to Fox."
    exit 1
fi

echo "Slurm batch script transferred to '${REMOTE_DIR}/' on Fox."

###############################################################################
# Step 2: Submit the Slurm Batch Job via SSH
###############################################################################

echo $'\n==== Submitting Slurm batch job for Ollama API ===='

JOB_SUBMISSION_OUTPUT=$(ssh "${SSH_CONFIG_NAME}" "cd '${REMOTE_DIR}' && sbatch '${SBATCH_SCRIPT}'")

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to submit Slurm job."
    echo "SSH Output:"
    echo "$JOB_SUBMISSION_OUTPUT"
    exit 1
fi

# Extract Job ID
JOB_ID=$(echo "$JOB_SUBMISSION_OUTPUT" | awk '{print $4}')

if [[ -z "$JOB_ID" ]]; then
    echo "Error: Could not parse Slurm job ID from submission output."
    echo "Submission Output:"
    echo "$JOB_SUBMISSION_OUTPUT"
    exit 1
fi

echo "Job submitted with ID: $JOB_ID"

###############################################################################
# Step 3: Wait for the Job to Start Running
###############################################################################

echo $'\n==== Waiting for job '"$JOB_ID"' to start running ===='
echo $"Run command 'ssh $SSH_CONFIG_NAME scancel $JOB_ID' to cancel slurm job request."

while true; do
    JOB_STATE=$(ssh "${SSH_CONFIG_NAME}" "squeue -j $JOB_ID -h -o '%T'")
    if [[ "$JOB_STATE" == "RUNNING" ]]; then
        echo "Job $JOB_ID is now running."
        break
    elif [[ "$JOB_STATE" == "PENDING" ]]; then
        echo "Job $JOB_ID is pending. Waiting..."
    else
        echo "Job $JOB_ID entered unexpected state: $JOB_STATE"
        echo "Check job status with 'squeue -j $JOB_ID'"
        exit 1
    fi
    sleep 10
done

###############################################################################
# Step 4: Retrieve the Allocated Node Name (First Node in List)
###############################################################################

echo $'\n==== Retrieving allocated node name for job '"$JOB_ID"' ===='

NODE_NAME=$(
  ssh "${SSH_CONFIG_NAME}" "
    scontrol show job $JOB_ID \
      | grep 'NodeList=' \
      | grep -v '(null)' \
      | head -n1 \
      | sed 's/.*NodeList=\\([^ ]*\\).*/\\1/' \
      | sed 's/\\[.*//'   # Remove brackets and ranges
  "
)

if [[ -z $NODE_NAME ]]; then
    echo \"Error: Could not retrieve the allocated node name.\"
    exit 1
fi

echo "Allocated node: ${NODE_NAME}"

###############################################################################
# Step 5: Retrieve Ollama API url
###############################################################################
echo $'\n==== Retrieve Ollama API url on Fox ===='
echo "Ollama now exists at fox on url: ${NODE_NAME}:${OLLAMA_PORT} ${SSH_CONFIG_NAME}"

#Write to file, to later setup ssh connection dynamically if lost.
cat <<EOT > "./scripts/Ollama_url.sh"
#!/bin/bash
${NODE_NAME}:${OLLAMA_PORT} ${SSH_CONFIG_NAME}
EOT

# Make the script executable
chmod +x ./scripts/Ollama_url.sh

###############################################################################
# LAST: Handle Script Termination
###############################################################################
# Keep the script running to maintain port forwarding
echo $'\n Experiment '$PHASE'_'$EXPERIMENT' is now running on Fox 🎉'
echo $"Run command 'ssh $SSH_CONFIG_NAME scancel $JOB_ID' to cancel slurm job."


