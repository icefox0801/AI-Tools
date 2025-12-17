#!/bin/bash
# Pre-download FastConformer NeMo model to cache volume using HuggingFace CLI
# This script automatically downloads models without confirmation
# The service uses local_files_only=True to avoid any network requests.

set -e

# HuggingFace cache directory
CACHE_DIR="/root/.cache/huggingface/hub"
NEMO_CACHE_DIR="/root/.cache/nemo"

mkdir -p "$CACHE_DIR"
mkdir -p "$NEMO_CACHE_DIR"

echo "================================"
echo "FastConformer Model Setup"
echo "HF Cache directory: $CACHE_DIR"
echo "NeMo Cache directory: $NEMO_CACHE_DIR"
echo "================================"

# Model information
MODEL="${FASTCONFORMER_MODEL:-nvidia/stt_en_fastconformer_hybrid_large_streaming_multi}"

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

# Main workflow
echo ""
echo "Step 1: Check/Download FastConformer model"
echo "-------------------------------------------"

if check_model_cached "$MODEL"; then
    echo "Model already cached, skipping download"
else
    echo "Model not found, downloading..."
    if download_model "$MODEL"; then
        echo "Model downloaded successfully"
    else
        echo "ERROR: Failed to download model"
        exit 1
    fi
fi

echo ""
echo "================================"
echo "Model setup complete!"
echo "Model: $MODEL"
echo "Cache location: $CACHE_DIR"
echo "================================"
