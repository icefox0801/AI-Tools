#!/bin/bash
# Pre-download model weights into the weights volume.
# Idempotent: skips files/directories that already exist.
#
# Models downloaded:
#   Real-ESRGAN
#     RealESRGAN_x4plus.pth                        (~65 MB)
#   BasicVSR++ (temporal mode)
#     spynet + basicvsr_plusplus weights            (~80 MB)
#
# Total first-run download: ~145 MB
# Run inside the container: docker exec video-upscaler bash /app/download_models.sh

set -e

WEIGHTS_DIR="${WEIGHTS_DIR:-/app/weights}"
mkdir -p "$WEIGHTS_DIR"

echo "========================================"
echo "Video Upscaler — Weights Setup"
echo "Weights directory: $WEIGHTS_DIR"
echo "========================================"

# ── helper: wget with retry ──────────────────────────────────────────────────
download_file() {
    local name=$1 url=$2 dest="$WEIGHTS_DIR/$1"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "✓ $name already present"
        return 0
    fi
    echo "↓ Downloading $name …"
    wget --tries=3 --timeout=60 -q -O "$dest.tmp" "$url" && mv "$dest.tmp" "$dest"
    echo "✓ $name done"
}

# ── 1. Real-ESRGAN (single representative model) ─────────────────────────────
RELEASE="https://github.com/xinntao/Real-ESRGAN/releases/download"
download_file "RealESRGAN_x4plus.pth" "${RELEASE}/v0.1.0/RealESRGAN_x4plus.pth"

# BasicVSR++ temporal model + SPyNet optical-flow backbone (kept for BasicVSR mode)
download_file "spynet_20210409-c6c1bd09.pth" \
    "https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth"
download_file "basicvsr_plusplus_c64n7_8x1_600k_reds4.pth" \
    "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"

echo "========================================"
echo "All weights ready in $WEIGHTS_DIR"
echo "========================================"
