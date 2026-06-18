#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-/app/weights}"
HF_HOME="${HF_HOME:-${WEIGHTS_DIR}/huggingface}"
export HF_HOME
mkdir -p "${WEIGHTS_DIR}"
cd "${WEIGHTS_DIR}"

fetch_if_missing() {
  local url="$1"
  local out="$2"
  if [ -f "$out" ]; then
    echo "[skip] $out already exists"
    return
  fi
  echo "[download] $out"
  curl -L --retry 5 --retry-delay 2 "$url" -o "$out"
}

# Check if a HuggingFace model is already cached (snapshot dir exists and has symlinks)
hf_model_cached() {
  local model_id="$1"
  local cache_dir="${HF_HOME}/hub/models--$(echo "$model_id" | tr '/' '--')"
  if [ -d "$cache_dir/snapshots" ] && [ "$(find "$cache_dir/snapshots" -type l | wc -l)" -gt 0 ]; then
    return 0
  fi
  return 1
}

fetch_hf_model() {
  local model_id="$1"
  local variant="${2:-}"
  if hf_model_cached "$model_id"; then
    echo "[skip] $model_id already cached"
    return
  fi
  echo "[download] $model_id → $HF_HOME"
  if [ -n "$variant" ]; then
    # Use diffusers pipeline for fp16-only download (avoids ONNX/OpenVINO/fp32 bloat)
    python3 -c "
import os
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HOME'] = '$HF_HOME'
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    '$model_id',
    torch_dtype=torch.float16,
    variant='$variant',
    cache_dir='$HF_HOME',
)
print('$model_id download complete.')
"
  else
    # Raw checkpoint — use snapshot_download
    python3 -c "
import os
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HOME'] = '$HF_HOME'
from huggingface_hub import snapshot_download
snapshot_download('$model_id')
"
  fi
}

fetch_hf_model_optional() {
  local model_id="$1"
  local variant="${2:-}"
  if ! fetch_hf_model "$model_id" "$variant"; then
    echo "[warn] Could not download $model_id. Using local SUPIR files if available."
  fi
}

BASE="https://github.com/xinntao/Real-ESRGAN/releases/download"
fetch_if_missing "$BASE/v0.1.0/RealESRGAN_x4plus.pth" "RealESRGAN_x4plus.pth"
fetch_if_missing "$BASE/v0.2.5.0/realesr-general-x4v3.pth" "realesr-general-x4v3.pth"
fetch_if_missing "$BASE/v0.2.5.0/realesr-general-wdn-x4v3.pth" "realesr-general-wdn-x4v3.pth"

if [ "${DOWNLOAD_SUPIR_FROM_HF:-false}" = "true" ]; then
  fetch_hf_model "stabilityai/stable-diffusion-xl-base-1.0" "fp16"
  fetch_hf_model_optional "Kijai/SUPIR_pruned"
else
  echo "[skip] SUPIR HF download disabled (set DOWNLOAD_SUPIR_FROM_HF=true to enable)."
fi

echo "Image SR models are ready in $WEIGHTS_DIR"
