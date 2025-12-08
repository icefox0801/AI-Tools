#!/bin/bash
# Pre-download Parakeet NeMo models to cache volume using HuggingFace CLI
# This script automatically downloads models without confirmation
# The service uses local_files_only=True to avoid any network requests.

set -e

# HuggingFace cache directory
CACHE_DIR="/root/.cache/huggingface/hub"

mkdir -p "$CACHE_DIR"

echo "================================"
echo "Parakeet Models Setup"
echo "Cache directory: $CACHE_DIR"
echo "================================"

# Model information
STREAMING_MODEL="${PARAKEET_STREAMING_MODEL:-nvidia/parakeet-tdt-1.1b}"
OFFLINE_MODEL="${PARAKEET_OFFLINE_MODEL:-nvidia/parakeet-rnnt-1.1b}"

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

# Check and download streaming model
echo ""
echo "1. Checking Streaming Model ($STREAMING_MODEL)..."
if check_model_cached "$STREAMING_MODEL"; then
    echo "Streaming model already cached, skipping download"
else
    download_model "$STREAMING_MODEL"
fi

# Check and download offline model
echo ""
echo "2. Checking Offline Model ($OFFLINE_MODEL)..."
if check_model_cached "$OFFLINE_MODEL"; then
    echo "Offline model already cached, skipping download"
else
    download_model "$OFFLINE_MODEL"
fi

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "Cache size:"
du -sh "$CACHE_DIR" 2>/dev/null || echo "Unable to calculate size"
echo "================================"
