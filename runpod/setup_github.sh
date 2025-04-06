#!/bin/bash

echo "Setting up GitHub configuration..."

# Create persistent directories in /workspace
mkdir -p /workspace/.config/gh
mkdir -p /workspace/.ssh
mkdir -p /workspace/.gitconfig
mkdir -p /workspace/.config/wandb
mkdir -p /workspace/cache/huggingface/datasets
mkdir -p /workspace/cache/uv

# Set environment variables for cache directories
export HF_HOME=/workspace/cache/huggingface
export HF_DATASETS_CACHE="/workspace/cache/huggingface/datasets"
export UV_CACHE_DIR="/workspace/cache/uv"   
export XDG_CONFIG_HOME="/workspace/.config"

# GitHub CLI authentication with custom config home
echo "Authenticating with GitHub..."
gh auth login -p https -w

# Configure git to use GitHub CLI as credential helper
git config --global --file /workspace/.gitconfig/config credential.helper "!/usr/bin/gh auth git-credential"

# Configure git globals if not set, storing in /workspace
if [ -z "$(git config --global user.name)" ]; then
    echo "Setting up git globals..."
    read -p "Enter your Git username: " git_username
    read -p "Enter your Git email: " git_email
    git config --global --file /workspace/.gitconfig/config user.name "$git_username"
    git config --global --file /workspace/.gitconfig/config user.email "$git_email"
fi

# Set up SSH key in /workspace/.ssh if not exists
if [ ! -f /workspace/.ssh/id_ed25519 ]; then
    echo "Generating SSH key..."
    ssh-keygen -t ed25519 -C "$(git config --global user.email)" -f /workspace/.ssh/id_ed25519 -N ""
    echo "Your public SSH key:"
    cat /workspace/.ssh/id_ed25519.pub
    echo "Please add this key to your GitHub account at https://github.com/settings/keys"
fi

# Create symbolic links for Git and SSH to use the workspace locations
ln -sf /workspace/.gitconfig/config ~/.gitconfig
ln -sf /workspace/.ssh/id_ed25519 ~/.ssh/id_ed25519
ln -sf /workspace/.ssh/id_ed25519.pub ~/.ssh/id_ed25519.pub

# Set proper permissions
chmod 700 /workspace/.ssh
chmod 600 /workspace/.ssh/id_ed25519
chmod 644 /workspace/.ssh/id_ed25519.pub

# Setup Weights & Biases
echo -e "\nSetting up Weights & Biases..."
if [ ! -f /workspace/.config/wandb/settings ]; then
    read -p "Enter your Weights & Biases API key: " wandb_key
    # Login to wandb with the API key
    wandb login "$wandb_key"
    # Move the wandb config to persistent storage and create symlink
    mv ~/.config/wandb/settings /workspace/.config/wandb/ 2>/dev/null || true
    ln -sf /workspace/.config/wandb/settings ~/.config/wandb/settings
fi

echo "GitHub and Weights & Biases setup complete!"