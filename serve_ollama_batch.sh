#!/usr/bin/env bash

###############################################################################
# Script to Automate Launching Ollama API on Fox HPC Cluster Using sbatch
###############################################################################

# Configuration
USER="ec-sindrre"                        # Your Educloud username
HOST="fox.educloud.no"                   # Fox login address (matches SSH config)
SSH_CONFIG_NAME="fox"                # Name of the SSH config entry
ACCOUNT="ec12"                           # Fox project account
PARTITION="accel"                   # 'accel' or 'accel_long' (or 'ifi_accel' if access to ec11,ec29,ec30,ec34,ec35 or ec232)
GPUS=a100:1                                # a100 have 40GB or 80GB VRAM, while rtx30 have 24GB VRAM.
NODES=1                                 # Number of nodes. OLLAMA does currently only support single node inference
TIME="06:00:00"                         # Slurm walltime (D-HH:MM:SS)
MEM_PER_GPU="80GB"                       # Memory per GPU. 
OLLAMA_MODELS_DIR="/cluster/work/projects/ec12/ec-sindrre/ollama-models"  # Path to where the Ollama models are stored and loaded                      
LOCAL_PORT="11434"                        # Local port for forwarding
OLLAMA_PORT="11434"                       # Remote port where Ollama listens
SBATCH_SCRIPT="start_ollama_api.slurm"           # Slurm batch script name
REMOTE_DIR="/fp/homes01/u01/ec-sindrre/slurm_jobs" # Directory on Fox to store scripts and output

###############################################################################
# Step 1: Create the Slurm Batch Script Locally
###############################################################################

echo $'\n==== Creating Slurm batch script locally ===='

# mkdir -p "slurm_scripts"
# SBATCH_SCRIPT="slurm_scripts/${SBATCH_SCRIPT}"ss

cat <<EOT > "${SBATCH_SCRIPT}"
#!/bin/bash


###############################################################################
# Slurm Batch Script to Run Ollama Serve for Hosting an API
###############################################################################

# Job Configuration
#SBATCH --job-name=ollama_api                     # Job name
#SBATCH --account=${ACCOUNT}                      # Project account
#SBATCH --partition=${PARTITION}                  # Partition ('accel' or 'accel_long')
#SBATCH --nodes=${NODES}                  # Amount of nodes. Ollama one support single node inference
#SBATCH --gpus=${GPUS}                             # Number of GPUs
#SBATCH --time=${TIME}                             # Walltime (D-HH:MM:SS)
#SBATCH --mem-per-gpu=${MEM_PER_GPU}              # Memory per CPU
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

export OLLAMA_MODELS=${OLLAMA_MODELS_DIR}    # Path to where the Ollama models are stored and loaded
export OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}      # Host and port where Ollama listens
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
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
	--format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process


###############################################################################
# Start Ollama Server
###############################################################################

echo "Starting Ollama server..."

ollama serve

# After computation stop monitoring
kill -SIGINT "$NVIDIA_MONITOR_PID"

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
scp "${SBATCH_SCRIPT}" "${SSH_CONFIG_NAME}:'${REMOTE_DIR}/'"

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to transfer '${SBATCH_SCRIPT}' to Fox."
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
# Step 5: Set Up SSH Port Forwarding
###############################################################################
echo $'\n==== Setting up SSH port forwarding ===='
echo "Command about to run:"
echo "ssh -O forward -L ${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT} ${SSH_CONFIG_NAME}"
# echo "autossh -N -M 0 -L ${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT} ${SSH_CONFIG_NAME}"

# Establish SSH port forwarding in the background
ssh -O forward -L "${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT}" "${SSH_CONFIG_NAME}" &
# ssh -O -N -L "${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT}" "${SSH_CONFIG_NAME}" &
# autossh -N -M 0 -L "${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT}" "${SSH_CONFIG_NAME}" &
# PARENT_SSH_PID=$!  # Capture the PID of the initial SSH process

# Wait briefly to ensure the port forwarding process is established
sleep 2

# Retrieve the PID of the actual SSH port forwarding process
PORT_FORWARD_PID=$(lsof -i TCP:"${LOCAL_PORT}" -s TCP:LISTEN -t)
# PORT_FORWARD_PID=$(pgrep -f "autossh -N -L ${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT}")

# Step 4: Check if port forwarding was established
if [[ -z "${PORT_FORWARD_PID}" ]]; then
    echo "Error: Failed to establish SSH port forwarding."
    exit 1
fi
echo "SSH port forwarding established (PID: ${PORT_FORWARD_PID})."
echo "You can now access the API at http://localhost:${LOCAL_PORT}"

#Write to file, to later setup ssh connection dynamically if lost.
cat <<EOT > "SSH_FORWARDING.sh"
#!/bin/bash
ssh -O forward -L ${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT} ${SSH_CONFIG_NAME}
EOT

# open-webui serve
# gnome-terminal -- bash -c "open-webui serve; exec bash"
# echo "You can now access open webui http://localhost:8080"

###############################################################################
# Step 6: Handle Script Termination and Cleanup
###############################################################################

cleanup() {
    echo $'\n==== Terminating SSH port forwarding (PID: '"$PORT_FORWARD_PID"') ===='
    ssh -O cancel -L "${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT}" "${SSH_CONFIG_NAME}" &
    # kill "${PORT_FORWARD_PID}" 2>/dev/null

    if [[ $? -eq 0 ]]; then
        echo "Port forwarding terminated."
    else
        echo "Warning: Failed to terminate port forwarding. Process may already be stopped."
    fi
    # ssh -O exit fox # (optional) Close the master SSH control socket

    echo $'\n==== Cancelling Slurm job '"$JOB_ID"'===='
    ssh "${SSH_CONFIG_NAME}" "scancel $JOB_ID"
    echo "Slurm job $JOB_ID cancelled."

    echo "Cleanup complete."
    exit
}

# Trap SIGINT and SIGTERM to perform cleanup
trap cleanup SIGINT SIGTERM

# Keep the script running to maintain port forwarding
echo $'\nPress Ctrl+C to terminate port forwarding, cancel slurm job and exit.'
while true; do
    sleep 60
done
