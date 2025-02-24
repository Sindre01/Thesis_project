#!/usr/bin/env bash

###############################################################################
# Script to Automate Launching Ollama API on Fox HPC Cluster Using sbatch
###############################################################################

# Configuration
EXPERIMENT="few_shot"                    # Experiment ('few_shot' or 'COT')
PHASE="validation"                       # Phase ('testing' or 'validation')
EXAMPLES_TYPE="similarity"                 #'coverage' or 'similarity'
USER="ec-sindrre"                        # Your Educloud username
HOST="fox.educloud.no"                   # Fox login address (matches SSH config)
SSH_CONFIG_NAME="fox"                    # Name of the SSH config entry
ACCOUNT="ec12"                           # Fox project account
PARTITION="ifi_accel"                        # 'accel' or 'accel_long' (or 'ifi_accel' if access to ec11,ec29,ec30,ec34,ec35 or ec232)
GPUS=rtx30:3                              # a100 have 40GB or 80GB VRAM, while rtx30 have 24GB VRAM.
NODES=1                                  # Number of nodes. OLLAMA does currently only support single node inference
TIME="05-00:00:00"                       # Slurm walltime (D-HH:MM:SS)
MEM_PER_GPU="30GB"                       # Memory per GPU. 
OLLAMA_MODELS_DIR="/cluster/work/projects/ec12/ec-sindrre/ollama-models"  # Path to where the Ollama models are stored and loaded                      
LOCAL_PORT="11434"                        # Local port for forwarding
OLLAMA_PORT="11434"                       # Remote port where Ollama listens. If different parallell runs, change ollama_port to avoid conflicts if same node is allocated.
SBATCH_SCRIPT="${PHASE}_${EXAMPLES_TYPE}_ollama.slurm"           # Slurm batch script name
REMOTE_DIR="/fp/homes01/u01/ec-sindrre/slurm_jobs/${EXPERIMENT}/${PHASE}/${EXAMPLES_TYPE}" # Directory on Fox to store scripts and output

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
#SBATCH --job-name=${PHASE}_${EXPERIMENT}         # Job name
#SBATCH --account=${ACCOUNT}                      # Project account
#SBATCH --partition=${PARTITION}                  # Partition ('accel' or 'accel_long')
#SBATCH --nodes=${NODES}                           # Amount of nodes. Ollama one support single node inference
#SBATCH --gpus=${GPUS}                             # Number of GPUs
#SBATCH --time=${TIME}                             # Walltime (D-HH:MM:SS)
#SBATCH --mem-per-gpu=${MEM_PER_GPU}              # Memory per CPU
#SBATCH --output=Job_${PHASE}.out                 # Standard output and error log


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
# export OLLAMA_MAX_LOADED_MODELS
# export OLLAMA_MAX_QUEUE

# export CUDA_ERROR_LEVEL=50
# export CUDA_VISIBLE_DEVICES=0,1
# export AMD_LOG_LEVEL=3


####################### Setup monitoring ######################################
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
	--format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process


###############################################################################
# Start Ollama Server in Background with Log Redirection
###############################################################################
ollama serve > ollama_API.out 2>&1 &  

sleep 5

###############################################################################
# Run Python Script
###############################################################################
echo "============= Pulling latest changes from git... ============="
cd ~/Thesis_project
git fetch
git checkout ${PHASE}/${EXAMPLES_TYPE}
git fetch
git pull
source thesis_venv/bin/activate  # Activate it to ensure the correct Python environment

cd ~/Thesis_project/notebooks/few-shot/fox

echo "============= Running ${PHASE} ${EXPERIMENT} Python script... ============="
python -u run_${PHASE}.py > ${REMOTE_DIR}/${PHASE}.out 2>&1

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


