#!/bin/bash
# Pre-download Text Refiner models to cache volume using HuggingFace CLI
# This script automatically downloads models without confirmation
# The service uses these pre-cached models with local_files_only=True to avoid network requests

set -e

# Transformers expect models in /root/.cache/huggingface/hub/
CACHE_DIR="/root/.cache/huggingface/hub"

mkdir -p "$CACHE_DIR"

echo "================================"
echo "Text Refiner Models Setup"
echo "Cache directory: $CACHE_DIR"
echo "================================"

# Get model names from environment or use defaults (must match text_refiner_service.py)
PUNCTUATION_MODEL=${PUNCTUATION_MODEL:-pcs_en}
CORRECTION_MODEL=${CORRECTION_MODEL:-oliverguhr/spelling-correction-english-base}

echo "Punctuation model: $PUNCTUATION_MODEL"
echo "Correction model: $CORRECTION_MODEL"
echo ""

# Function to check if model is cached using hf CLI
check_model_cached() {
    local model_name=$1
    echo "Checking cache for $model_name..."
    
    # Check if model directory exists and has no incomplete files
    local model_dir="$CACHE_DIR/models--${model_name//\//-}"
    if [ -d "$model_dir" ] && [ $(find "$model_dir/blobs" -name "*.incomplete" 2>/dev/null | wc -l) -eq 0 ]; then
        echo "✓ Model found in cache"
        return 0
    else
        echo "✗ Model not in cache or incomplete"
        return 1
    fi
}

# Function to download a HuggingFace model using CLI
download_model() {
    local model_name=$1
    
    echo ""
    echo "Downloading $model_name using HuggingFace CLI..."
    if hf download "$model_name" --cache-dir "$CACHE_DIR"; then
        echo "✓ $model_name downloaded successfully"
        return 0
    else
        echo "✗ Failed to download $model_name"
        return 1
    fi
}

# Check and download punctuation model
echo "1. Checking Punctuation Model ($PUNCTUATION_MODEL)..."
if check_model_cached "$PUNCTUATION_MODEL"; then
    echo "Punctuation model already cached, skipping download"
else
    download_model "$PUNCTUATION_MODEL"
fi

# Check and download correction model
echo ""
echo "2. Checking Correction Model ($CORRECTION_MODEL)..."
if check_model_cached "$CORRECTION_MODEL"; then
    echo "Correction model already cached, skipping download"
else
    download_model "$CORRECTION_MODEL"
fi

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "Cache size:"
du -sh "$CACHE_DIR" 2>/dev/null || echo "Unable to calculate size"
echo "================================"
