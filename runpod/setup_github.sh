#!/bin/bash

echo "Setting up GitHub configuration..."

# GitHub CLI authentication
echo "Authenticating with GitHub..."
gh auth login -p https -w

# Configure git globals if not set
if [ -z "$(git config --global user.name)" ]; then
    echo "Setting up git globals..."
    read -p "Enter your Git username: " git_username
    read -p "Enter your Git email: " git_email
    git config --global user.name "$git_username"
    git config --global user.email "$git_email"
fi

# Set up SSH key if not exists
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "Generating SSH key..."
    ssh-keygen -t ed25519 -C "$(git config --global user.email)" -f ~/.ssh/id_ed25519 -N ""
    echo "Your public SSH key:"
    cat ~/.ssh/id_ed25519.pub
    echo "Please add this key to your GitHub account at https://github.com/settings/keys"
fi

echo "GitHub setup complete!" 