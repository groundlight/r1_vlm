#!/bin/bash

# Exit on error
set -e

echo "Building R1 VLM Docker image..."

# Get the absolute path of the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Build the Docker image
docker build -t r1_vlm -f Dockerfile "$PROJECT_ROOT"

echo "✅ Docker image built successfully!"
echo "Image name: r1_vlm"
echo "To run the container: docker run -it r1_vlm" 