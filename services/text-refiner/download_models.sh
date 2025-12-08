#!/bin/bash
# Pre-download Text Refiner models to cache volume using HuggingFace CLI
# This script checks for cached models and prompts before downloading
# The service uses these pre-cached models with local_files_only=True to avoid network requests

set -e

CACHE_DIR="${HF_HOME:-/root/.cache/huggingface}"

echo "================================"
echo "Text Refiner Models Setup"
echo "Cache directory: $CACHE_DIR"
echo "================================"

# Get model names from environment or use defaults (must match text_refiner_service.py)
PUNCTUATION_MODEL=${PUNCTUATION_MODEL:-pcs_en}
CORRECTION_MODEL=${CORRECTION_MODEL:-oliverguhr/spelling-correction-english-base}
PUNCTUATION_SIZE="~900MB"
CORRECTION_SIZE="~500MB"

echo "Punctuation model: $PUNCTUATION_MODEL"
echo "Correction model: $CORRECTION_MODEL"
echo ""

# Function to check if model is cached using hf CLI
check_model_cached() {
    local model_name=$1
    echo "Checking cache for $model_name..."
    
    # Use hf scan-cache to check if model exists
    if hf scan-cache | grep -q "$model_name"; then
        echo "✓ Model found in cache"
        return 0
    else
        echo "✗ Model not in cache"
        return 1
    fi
}

# Function to download a HuggingFace model using CLI
download_model() {
    local model_name=$1
    local model_size=$2
    
    echo ""
    echo "Model: $model_name"
    echo "Size: $model_size"
    read -p "Download this model? [y/N]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipped $model_name"
        return 1
    fi
    
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
    download_model "$PUNCTUATION_MODEL" "$PUNCTUATION_SIZE"
fi

# Check and download correction model
echo ""
echo "2. Checking Correction Model ($CORRECTION_MODEL)..."
if check_model_cached "$CORRECTION_MODEL"; then
    echo "Correction model already cached, skipping download"
else
    download_model "$CORRECTION_MODEL" "$CORRECTION_SIZE"
fi

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "Cache size:"
du -sh "$CACHE_DIR" 2>/dev/null || echo "Unable to calculate size"
echo "================================"
