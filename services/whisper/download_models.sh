#!/bin/bash
# Pre-download Whisper model to cache volume using HuggingFace CLI
# This script automatically downloads the model without confirmation
# The service uses local_files_only=True to avoid any network requests.

set -e

# Transformers expects models in /root/.cache/huggingface/hub/
CACHE_DIR="/root/.cache/huggingface/hub"

mkdir -p "$CACHE_DIR"

echo "================================"
echo "Whisper Model Setup"
echo "Cache directory: $CACHE_DIR"
echo "================================"

# Model information
MODEL="openai/whisper-large-v3-turbo"

# Function to check if model is fully cached
check_model_cached() {
    local model_id=$1
    echo "Checking cache for $model_id..."
    
    # Check if model directory exists and has no incomplete files
    local model_dir="$CACHE_DIR/models--${model_id//\//-}"
    if [ -d "$model_dir" ] && [ $(find "$model_dir/blobs" -name "*.incomplete" 2>/dev/null | wc -l) -eq 0 ]; then
        echo "✓ Model found in cache"
        return 0
    else
        echo "✗ Model not in cache or incomplete"
        return 1
    fi
}

# Function to download a model
download_model() {
    local model_id=$1
    
    echo ""
    echo "Downloading $model_id using HuggingFace CLI..."
    if hf download "$model_id" --cache-dir "$CACHE_DIR"; then
        echo "✓ $model_id downloaded successfully"
        return 0
    else
        echo "✗ Failed to download $model_id"
        return 1
    fi
}

# Check and download model
echo ""
echo "Checking Whisper Turbo Model ($MODEL)..."
if check_model_cached "$MODEL"; then
    echo "Model already cached, skipping download"
else
    download_model "$MODEL"
fi

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "Cache size:"
du -sh "$CACHE_DIR" 2>/dev/null || echo "Unable to calculate size"
echo "================================"
