#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-/app/weights}"
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

BASE="https://github.com/xinntao/Real-ESRGAN/releases/download"
fetch_if_missing "$BASE/v0.1.0/RealESRGAN_x4plus.pth" "RealESRGAN_x4plus.pth"
fetch_if_missing "$BASE/v0.2.5.0/realesr-general-x4v3.pth" "realesr-general-x4v3.pth"
fetch_if_missing "$BASE/v0.2.5.0/realesr-general-wdn-x4v3.pth" "realesr-general-wdn-x4v3.pth"

echo "Image SR models are ready in $WEIGHTS_DIR"
