#!/bin/bash
# Function to cleanup child processes
cleanup() {
    echo "[$(date)] Received interrupt signal on GPU $GPU_ID. Cleaning up..."
    pkill -P $$
    exit 1
}

# Set up trap for SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

# Change to the project root directory
cd /millcreek/home/sunil/r1_vlm
attempt=0
max_attempts=100
while true; do
    attempt=$((attempt+1))
    echo "[$(date)] Starting job on GPU $GPU_ID, Attempt $attempt"
    
    # Launch the Python script and capture its exit status
    CUDA_VISIBLE_DEVICES=0 uv run python src/r1_vlm/environments/real_iad_env/completion_generation.py
    exit_status=$?
    
    # Check if the script exited normally (exit status 0)
    if [ $exit_status -eq 0 ]; then
        echo "[$(date)] Job completed successfully on GPU $GPU_ID"
        break
    else
        if [ $attempt -ge $max_attempts ]; then
            echo "[$(date)] Job failed with exit status $exit_status after $attempt attempts. Exiting."
            exit 1
        fi
        echo "[$(date)] Job failed with exit status $exit_status. Restarting in 10 seconds..."
        sleep 10
    fi
done
