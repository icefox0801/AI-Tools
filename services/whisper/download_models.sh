#!/bin/bash
# Pre-download Whisper models to cache volume using HuggingFace CLI
# This script checks for cached models and prompts before downloading
# The service uses local_files_only=True to avoid any network requests.

set -e

CACHE_DIR="${HF_HOME:-/root/.cache/huggingface}"
MODELS_DIR="$CACHE_DIR/hub"

mkdir -p "$MODELS_DIR"

echo "================================"
echo "Whisper Models Setup"
echo "Cache directory: $CACHE_DIR"
echo "================================"

# Model information
STREAMING_MODEL="openai/whisper-large-v3-turbo"
OFFLINE_MODEL="openai/whisper-large-v3"
TURBO_SIZE="~1.6GB"
LARGE_SIZE="~3.1GB"

# Function to check if model is fully cached
check_model_cached() {
    local model_id=$1
    echo "Checking cache for $model_id..."
    
    # Use hf scan-cache to check if model exists and is complete
    if hf scan-cache | grep -q "$model_id"; then
        echo "✓ Model found in cache"
        return 0
    else
        echo "✗ Model not in cache"
        return 1
    fi
}

# Function to download a model
download_model() {
    local model_id=$1
    local model_size=$2
    
    echo ""
    echo "Model: $model_id"
    echo "Size: $model_size"
    read -p "Download this model? [y/N]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipped $model_id"
        return 1
    fi
    
    echo "Downloading $model_id using HuggingFace CLI..."
    if hf download "$model_id" --cache-dir "$CACHE_DIR"; then
        echo "✓ $model_id downloaded successfully"
        return 0
    else
        echo "✗ Failed to download $model_id"
        return 1
    fi
}

# Check and download streaming model
echo ""
echo "1. Checking Streaming Model ($STREAMING_MODEL)..."
if check_model_cached "$STREAMING_MODEL"; then
    echo "Streaming model already cached, skipping download"
else
    download_model "$STREAMING_MODEL" "$TURBO_SIZE"
fi

# Check and download offline model
echo ""
echo "2. Checking Offline Model ($OFFLINE_MODEL)..."
if check_model_cached "$OFFLINE_MODEL"; then
    echo "Offline model already cached, skipping download"
else
    download_model "$OFFLINE_MODEL" "$LARGE_SIZE"
fi

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "Cache size:"
du -sh "$CACHE_DIR" 2>/dev/null || echo "Unable to calculate size"
echo "================================"
