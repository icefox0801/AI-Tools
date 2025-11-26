#!/bin/bash
# Download Vosk model if not present in volume

MODEL_DIR="/app/model"
MODEL_NAME="vosk-model-en-us-0.22"

# Check if model exists (look for a key file)
if [ ! -f "$MODEL_DIR/am/final.mdl" ]; then
    echo "Model not found in volume, downloading $MODEL_NAME (1.8GB)..."
    echo "This only happens once - model will persist in Docker volume."
    
    mkdir -p "$MODEL_DIR"
    cd /tmp
    
    curl -L --progress-bar -o model.zip "https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"
    
    echo "Extracting model..."
    unzip -q model.zip
    mv ${MODEL_NAME}/* "$MODEL_DIR/"
    rm -rf model.zip ${MODEL_NAME}
    
    echo "Model downloaded and ready!"
else
    echo "Vosk model found in volume, starting service..."
fi

# Start the service
exec python vosk_service.py
