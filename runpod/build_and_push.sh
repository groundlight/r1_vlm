#!/bin/bash

# Exit on error
set -e

# Check if username is provided
if [ -z "$1" ]; then
    echo "❌ Error: Docker Hub username is required"
    echo "Usage: ./build_and_push.sh <dockerhub-username>"
    exit 1
fi

DOCKER_USERNAME="$1"
IMAGE_NAME="r1_vlm"

# Function to get next version number
get_next_version() {
    # Get the latest version tag (v1, v2, etc.), default to v0 if none exists
    local latest_tag=$(git tag | grep "^v[0-9]\+$" | sort -V | tail -n 1 || echo "v0")
    # Extract number and increment
    local current_num=${latest_tag#v}
    local next_num=$((current_num + 1))
    echo "v$next_num"
}

# Get the new version
NEW_VERSION=$(get_next_version)
echo "ℹ️  Creating new version: ${NEW_VERSION}"

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}"
VERSIONED_IMAGE_NAME="${FULL_IMAGE_NAME}:${NEW_VERSION}"
LATEST_IMAGE_NAME="${FULL_IMAGE_NAME}:latest"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🔨 Building Docker image using build_docker.sh..."
if "${SCRIPT_DIR}/build_docker.sh"; then
    echo "✅ Build completed successfully!"
else
    echo "❌ Build failed!"
    exit 1
fi

echo "🔄 Checking Docker Hub login status..."
if ! docker info 2>/dev/null | grep "Username:" > /dev/null; then
    echo "❌ Not logged into Docker Hub!"
    echo "Please run 'docker login' first"
    exit 1
fi

echo "🔄 Tagging images..."
docker tag "${IMAGE_NAME}" "${VERSIONED_IMAGE_NAME}"
docker tag "${IMAGE_NAME}" "${LATEST_IMAGE_NAME}"

echo "🔄 Pushing images to Docker Hub..."
if docker push "${VERSIONED_IMAGE_NAME}" && docker push "${LATEST_IMAGE_NAME}"; then
    echo "✅ Successfully pushed images to Docker Hub!"
    
    # Create and push new git tag
    echo "🏷️  Creating git tag ${NEW_VERSION}..."
    git tag "${NEW_VERSION}"
    git push origin "${NEW_VERSION}"
    
    echo "✅ Versioned image: ${VERSIONED_IMAGE_NAME}"
    echo "✅ Latest image: ${LATEST_IMAGE_NAME}"
else
    echo "❌ Failed to push images to Docker Hub"
    echo "Please check your permissions and connection"
    exit 1
fi

echo "🎉 Build and push completed successfully!"
echo "Your images are now available at:"
echo "- ${VERSIONED_IMAGE_NAME}"
echo "- ${LATEST_IMAGE_NAME}"

# play terminal bell sound so we know it's done
echo -e "\a" 