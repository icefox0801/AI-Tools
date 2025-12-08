#!/bin/bash
# Pre-download Vosk model to volume
# This script checks for cached models and prompts before downloading

set -e

MODEL_DIR="${VOSK_MODEL_DIR:-/app/model}"
MODEL_NAME="${VOSK_MODEL_NAME:-vosk-model-en-us-0.22}"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"

# Determine model size based on model name
case "$MODEL_NAME" in
    *small*)
        MODEL_SIZE="~50MB"
        ;;
    *0.22*)
        MODEL_SIZE="~1.8GB"
        ;;
    *0.22-lgraph*)
        MODEL_SIZE="~128MB"
        ;;
    *)
        MODEL_SIZE="~1-2GB"
        ;;
esac

mkdir -p "$MODEL_DIR"

echo "================================"
echo "Vosk Model Setup"
echo "Model directory: $MODEL_DIR"
echo "================================"

# Function to check if model is cached
check_model_cached() {
    if [ -f "$MODEL_DIR/am/final.mdl" ]; then
        echo "✓ Model found in cache"
        return 0
    else
        echo "✗ Model not in cache"
        return 1
    fi
}

# Function to download the model
download_model() {
    echo ""
    echo "Model: $MODEL_NAME"
    echo "Size: $MODEL_SIZE"
    read -p "Download this model? [y/N]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipped $MODEL_NAME"
        return 1
    fi
    
    echo "Downloading $MODEL_NAME..."
    
    cd /tmp
    curl -L --progress-bar -o model.zip "$MODEL_URL"
    
    echo "Extracting model..."
    unzip -q model.zip
    
    # Move contents to model directory
    if [ -d "${MODEL_NAME}" ]; then
        mv ${MODEL_NAME}/* "$MODEL_DIR/"
        rm -rf ${MODEL_NAME}
    else
        echo "✗ Expected directory ${MODEL_NAME} not found after extraction"
        rm -f model.zip
        return 1
    fi
    
    rm -f model.zip
    
    # Save model name for runtime identification
    echo "$MODEL_NAME" > "$MODEL_DIR/.model_name"
    
    echo "✓ $MODEL_NAME downloaded successfully"
    return 0
}

# Check and download model
echo ""
echo "Checking Vosk Model ($MODEL_NAME)..."
if check_model_cached; then
    echo "Model already cached, skipping download"
else
    download_model
fi

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "Model directory size:"
du -sh "$MODEL_DIR" 2>/dev/null || echo "Unable to calculate size"
echo "================================"
