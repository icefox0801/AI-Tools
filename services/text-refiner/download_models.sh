#!/bin/bash
# Pre-download Text Refiner models to cache volume
# This script checks for cached models and prompts before downloading
# The service uses these pre-cached models with local_files_only=True to avoid network requests

set -e

echo "================================"
echo "Text Refiner Models Setup"
echo "================================"

# Get model names from environment or use defaults (must match text_refiner_service.py)
PUNCTUATION_MODEL=${PUNCTUATION_MODEL:-pcs_en}
CORRECTION_MODEL=${CORRECTION_MODEL:-oliverguhr/spelling-correction-english-base}
PUNCTUATION_SIZE="~900MB"
CORRECTION_SIZE="~500MB"

echo "Punctuation model: $PUNCTUATION_MODEL"
echo "Correction model: $CORRECTION_MODEL"
echo ""

# Function to check if model is cached
check_model_cached() {
    local model_name=$1
    local model_type=$2
    echo "Checking cache for $model_name..."
    
    python3 -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from punctuators.models import PunctCapSegModelONNX
import sys
try:
    if '$model_type' == 'punctuation':
        # PunctCapSegModelONNX doesn't support local_files_only parameter
        model = PunctCapSegModelONNX.from_pretrained('${model_name}')
        print('✓ Model found in cache')
        del model
    else:
        tokenizer = AutoTokenizer.from_pretrained('${model_name}', local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained('${model_name}', local_files_only=True)
        print('✓ Model found in cache')
        del tokenizer, model
    sys.exit(0)
except Exception:
    print('✗ Model not in cache')
    sys.exit(1)
" 2>/dev/null
}

# Function to download a HuggingFace model using Python
download_model() {
    local model_name=$1
    local model_type=$2
    local model_size=$3
    
    echo ""
    echo "Model: $model_name"
    echo "Type: $model_type"
    echo "Size: $model_size"
    read -p "Download this model? [y/N]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipped $model_name"
        return 1
    fi
    
    echo "Downloading $model_type: $model_name..."
    python3 -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from punctuators.models import PunctCapSegModelONNX
try:
    if '$model_type' == 'punctuation':
        print(f'Downloading ${model_name}...')
        model = PunctCapSegModelONNX.from_pretrained('${model_name}')
        print(f'✓ ${model_name} downloaded successfully')
        del model
    else:
        print(f'Downloading tokenizer and model for ${model_name}...')
        tokenizer = AutoTokenizer.from_pretrained('${model_name}')
        model = AutoModelForSeq2SeqLM.from_pretrained('${model_name}')
        print(f'✓ ${model_name} downloaded successfully')
        del tokenizer, model
except Exception as e:
    print(f'✗ Failed to download ${model_name}: {e}')
    exit(1)
"
    return $?
}

# Check and download punctuation model
echo "1. Checking Punctuation Model ($PUNCTUATION_MODEL)..."
if check_model_cached "$PUNCTUATION_MODEL" "punctuation"; then
    echo "Punctuation model already cached, skipping download"
else
    download_model "$PUNCTUATION_MODEL" "punctuation" "$PUNCTUATION_SIZE"
fi

# Check and download correction model
echo ""
echo "2. Checking Correction Model ($CORRECTION_MODEL)..."
if check_model_cached "$CORRECTION_MODEL" "correction"; then
    echo "Correction model already cached, skipping download"
else
    download_model "$CORRECTION_MODEL" "correction" "$CORRECTION_SIZE"
fi

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "================================"
