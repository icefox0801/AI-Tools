#!/bin/bash
# Pre-download Parakeet NeMo models to cache volume
# This script checks for cached models and prompts before downloading
# The service uses these pre-cached models to avoid network requests

set -e

echo "================================"
echo "Parakeet Models Setup"
echo "================================"

# Get model names from environment or use defaults
STREAMING_MODEL=${PARAKEET_STREAMING_MODEL:-nvidia/parakeet-tdt-1.1b}
OFFLINE_MODEL=${PARAKEET_OFFLINE_MODEL:-nvidia/parakeet-rnnt-1.1b}
STREAMING_SIZE="~4.7GB"
OFFLINE_SIZE="~4.7GB"

echo "Streaming model: $STREAMING_MODEL"
echo "Offline model: $OFFLINE_MODEL"
echo ""

# Function to check if model is cached
check_model_cached() {
    local model_name=$1
    echo "Checking cache for $model_name..."
    
    python3 -c "
import nemo.collections.asr as nemo_asr
import sys
try:
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained('${model_name}', map_location='cpu')
    print('✓ Model found in cache')
    del model
    sys.exit(0)
except Exception:
    print('✗ Model not in cache')
    sys.exit(1)
" 2>/dev/null
}

# Function to download a model using Python
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
    
    echo "Downloading $model_name..."
    python3 -c "
import nemo.collections.asr as nemo_asr
try:
    print(f'Downloading ${model_name}...')
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained('${model_name}')
    print(f'✓ ${model_name} downloaded successfully')
    del model
except Exception as e:
    print(f'✗ Failed to download ${model_name}: {e}')
    exit(1)
"
    return $?
}

# Check and download streaming model
echo "1. Checking Streaming Model ($STREAMING_MODEL)..."
if check_model_cached "$STREAMING_MODEL"; then
    echo "Streaming model already cached, skipping download"
else
    download_model "$STREAMING_MODEL" "$STREAMING_SIZE"
fi

# Check and download offline model
echo ""
echo "2. Checking Offline Model ($OFFLINE_MODEL)..."
if check_model_cached "$OFFLINE_MODEL"; then
    echo "Offline model already cached, skipping download"
else
    download_model "$OFFLINE_MODEL" "$OFFLINE_SIZE"
fi

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "================================"
