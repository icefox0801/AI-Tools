#!/bin/bash
# Pre-download Whisper models to cache volume using Python
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

# Model information
STREAMING_MODEL="openai/whisper-large-v3-turbo"
OFFLINE_MODEL="openai/whisper-large-v3"
TURBO_SIZE="~1.6GB"
LARGE_SIZE="~3.1GB"

# Function to check if model is cached
check_model_cached() {
    local model_id=$1
    echo "Checking cache for $model_id..."
    
    python3 -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import sys
try:
    AutoModelForSpeechSeq2Seq.from_pretrained('${model_id}', local_files_only=True)
    AutoProcessor.from_pretrained('${model_id}', local_files_only=True)
    print('✓ Model found in cache')
    sys.exit(0)
except Exception:
    print('✗ Model not in cache')
    sys.exit(1)
" 2>/dev/null
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
    
    echo "Downloading $model_id..."
    python3 -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
try:
    print('Downloading model...')
    model = AutoModelForSpeechSeq2Seq.from_pretrained('${model_id}')
    print('Downloading processor...')
    processor = AutoProcessor.from_pretrained('${model_id}')
    print('✓ ${model_id} downloaded successfully')
except Exception as e:
    print(f'✗ Failed to download ${model_id}: {e}')
    exit(1)
"
    return $?
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
