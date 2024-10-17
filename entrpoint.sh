#!/bin/bash

# Check if the kaggle.json file is provided via the mounted volume
if [ -f "/app/kaggle/kaggle.json" ]; then
    echo "kaggle.json found, setting it up."
    # Ensure .kaggle directory exists
    mkdir -p /root/.kaggle/
    
    # Copy kaggle.json and set permissions
    cp /app/kaggle/kaggle.json /root/.kaggle/
    chmod 600 /root/.kaggle/kaggle.json
    echo "kaggle.json successfully copied to /root/.kaggle/"
else
    echo "kaggle.json not found! Please mount it to /app/kaggle/kaggle.json"
    exit 1
fi

# Run the main Python script
echo "Running siamese_model.py"
python siamese_model.py
