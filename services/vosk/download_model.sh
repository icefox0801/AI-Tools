#!/bin/bash
# Download Vosk model if not present in volume

MODEL_DIR="/app/model"
MODEL_NAME="${VOSK_MODEL_NAME:-vosk-model-en-us-0.22}"

# Check if model exists (look for a key file)
if [ ! -f "$MODEL_DIR/am/final.mdl" ]; then
    echo "Model not found in volume, downloading $MODEL_NAME..."
    echo "This only happens once - model will persist in Docker volume."
    
    mkdir -p "$MODEL_DIR"
    cd /tmp
    
    curl -L --progress-bar -o model.zip "https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"
    
    echo "Extracting model..."
    unzip -q model.zip
    mv ${MODEL_NAME}/* "$MODEL_DIR/"
    rm -rf model.zip ${MODEL_NAME}
    
    # Save model name for runtime identification
    echo "$MODEL_NAME" > "$MODEL_DIR/.model_name"
    
    echo "Model downloaded and ready!"
else
    echo "Vosk model found in volume, starting service..."
fi

# Start the service (endpoint config is applied in Python before model load)
exec python vosk_service.py
