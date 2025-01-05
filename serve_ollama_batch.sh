#!/usr/bin/env bash

###############################################################################
# Script to Automate Launching Ollama API on Fox HPC Cluster Using sbatch
###############################################################################

# Configuration
USER="ec-sindrre"                        # Your Educloud username
HOST="fox.educloud.no"                   # Fox login address (matches SSH config)
SSH_CONFIG_NAME="fox"                # Name of the SSH config entry
ACCOUNT="ec12"                           # Fox project account
PARTITION="accel"                   # 'accel' or 'accel_long'
GPUS=2                                   # Number of GPUs
TIME="01:00:00"                         # Slurm walltime (D-HH:MM:SS)
MEM_PER_CPU="16G"                        # Memory per CPU
OLLAMA_MODELS_DIR="/fp/projects01/ec12/ec-sindrre/cache/ollama"  # Path to where the Ollama models are stored and loaded                      
LOCAL_PORT="11434"                        # Local port for forwarding
OLLAMA_PORT="11434"                       # Remote port where Ollama listens
SBATCH_SCRIPT="start_ollama_api.slurm"           # Slurm batch script name
REMOTE_DIR="/fp/homes01/u01/ec-sindrre/slurm_jobs" # Directory on Fox to store scripts and output

###############################################################################
# Step 1: Create the Slurm Batch Script Locally
###############################################################################

echo $'\n==== Creating Slurm batch script locally ===='

# mkdir -p "slurm_scripts"
# SBATCH_SCRIPT="slurm_scripts/${SBATCH_SCRIPT}"

cat <<EOT > "${SBATCH_SCRIPT}"
#!/bin/bash

###############################################################################
# Slurm Batch Script to Run Ollama Serve for Hosting an API
###############################################################################

# Job Configuration
#SBATCH --job-name=ollama_api                     # Job name
#SBATCH --account=${ACCOUNT}                      # Project account
#SBATCH --partition=${PARTITION}                  # Partition ('accel' or 'accel_long')
#SBATCH --gpus=${GPUS}                             # Number of GPUs
#SBATCH --time=${TIME}                             # Walltime (D-HH:MM:SS)
#SBATCH --mem-per-cpu=${MEM_PER_CPU}              # Memory per CPU
#SBATCH --output=ollama_api_%j.out                 # Standard output and error log

###############################################################################
# Environment Setup
###############################################################################

# Fail on errors and treat unset variables as errors
set -o errexit
set -o nounset

# Reset modules to system default
# module --quiet purge

export OLLAMA_MODELS=${OLLAMA_MODELS_DIR}    # Path to where the Ollama models are stored and loaded
export OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}      # Host and port where Ollama listens
# export OLLAMA_DEBUG=1


###############################################################################
# Start Ollama Server
###############################################################################

echo "Starting Ollama server on \$(hostname)..."

ollama serve

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
# Step 4: Retrieve the Allocated Node Name
###############################################################################

echo $'\n==== Retrieving allocated node name for job '"$JOB_ID"' ===='

NODE_NAME=$(
  ssh "${SSH_CONFIG_NAME}" "
    scontrol show job $JOB_ID \
      | grep 'NodeList=' \
      | grep -v '(null)' \
      | head -n1 \
      | sed 's/.*NodeList=\\([^ ]*\\).*/\\1/'
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
echo "ssh -f -N -L ${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT} ${SSH_CONFIG_NAME}"

# Establish SSH port forwarding in the background
ssh -f -N -L "${LOCAL_PORT}:${NODE_NAME}:${OLLAMA_PORT}" "${SSH_CONFIG_NAME}" &
PORT_FORWARD_PID=$!

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to establish SSH port forwarding."
    exit 1
fi

echo "SSH port forwarding established (PID: ${PORT_FORWARD_PID})."
echo "You can now access the Ollama API running at http://localhost:${LOCAL_PORT}"

###############################################################################
# Step 6: Handle Script Termination and Cleanup
###############################################################################

# Function to clean up SSH port forwarding and optionally cancel the job
cleanup() {
  echo $'\n==== Terminating SSH port forwarding (PID: '"$PORT_FORWARD_PID"') ===='
    kill "${PORT_FORWARD_PID}"
    echo "Port forwarding terminated."
    
    # ssh -O exit fox # (optional) Close the master SSH control socket

    echo $'\n==== Cancelling Slurm job '"$PORT_FORWARD_PID"'===='
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
