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

# Check if a HuggingFace model is already cached (snapshot dir exists and has files)
hf_model_cached() {
  local model_id="$1"
  local cache_dir="${HF_HOME}/hub/models--$(echo "$model_id" | tr '/' '--')"
  if [ -d "$cache_dir/snapshots" ] && [ "$(find "$cache_dir/snapshots" -type f | wc -l)" -gt 0 ]; then
    return 0
  fi
  return 1
}

fetch_hf_model() {
  local model_id="$1"
  if hf_model_cached "$model_id"; then
    echo "[skip] $model_id already cached"
    return
  fi
  echo "[download] $model_id → $HF_HOME"
  huggingface-cli download "$model_id" --resume-download
}

BASE="https://github.com/xinntao/Real-ESRGAN/releases/download"
fetch_if_missing "$BASE/v0.1.0/RealESRGAN_x4plus.pth" "RealESRGAN_x4plus.pth"
fetch_if_missing "$BASE/v0.2.5.0/realesr-general-x4v3.pth" "realesr-general-x4v3.pth"
fetch_if_missing "$BASE/v0.2.5.0/realesr-general-wdn-x4v3.pth" "realesr-general-wdn-x4v3.pth"

fetch_hf_model "stabilityai/sd-x2-latent-upscaler"
fetch_hf_model "stabilityai/stable-diffusion-x4-upscaler"

echo "Image SR models are ready in $WEIGHTS_DIR"
