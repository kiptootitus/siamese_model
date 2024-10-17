#!/bin/bash
set -e

# Function to display error messages
error() {
    echo "Error: $1" >&2
    exit 1
}

# Ensure the .kaggle directory exists
mkdir -p /root/.kaggle

# Check if the kaggle.json file is provided via the mounted volume
if [ -f "/app/kaggle/kaggle.json" ]; then
    echo "kaggle.json found, setting it up."

    # Copy kaggle.json and set permissions
    cp /app/kaggle/kaggle.json /root/.kaggle/ || error "Failed to copy kaggle.json"
    chmod 600 /root/.kaggle/kaggle.json || error "Failed to set permissions on kaggle.json"
    echo "kaggle.json successfully copied to /root/.kaggle/"
else
    error "kaggle.json not found! Please mount it to /app/kaggle/kaggle.json"
fi

# Run the main Python script
echo "Running siamese_model.py"
exec python siamese_model.py "$@"
