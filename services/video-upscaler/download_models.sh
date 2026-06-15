#!/bin/bash
# Pre-download Real-ESRGAN model weights into the weights volume.
# Idempotent: skips files that already exist.

set -e

WEIGHTS_DIR="${WEIGHTS_DIR:-/app/weights}"
mkdir -p "$WEIGHTS_DIR"

echo "================================"
echo "Real-ESRGAN Weights Setup"
echo "Weights directory: $WEIGHTS_DIR"
echo "================================"

RELEASE="https://github.com/xinntao/Real-ESRGAN/releases/download"

# name|url
WEIGHTS=(
    "realesr-general-x4v3.pth|${RELEASE}/v0.2.5.0/realesr-general-x4v3.pth"
    "realesr-general-wdn-x4v3.pth|${RELEASE}/v0.2.5.0/realesr-general-wdn-x4v3.pth"
    "RealESRGAN_x4plus.pth|${RELEASE}/v0.1.0/RealESRGAN_x4plus.pth"
    "RealESRGAN_x4plus_anime_6B.pth|${RELEASE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    "RealESRGAN_x2plus.pth|${RELEASE}/v0.2.1/RealESRGAN_x2plus.pth"
)

download() {
    local name=$1 url=$2 dest="$WEIGHTS_DIR/$1"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "✓ $name already present"
        return 0
    fi
    echo "↓ Downloading $name ..."
    wget --tries=3 --timeout=60 -q -O "$dest.tmp" "$url" && mv "$dest.tmp" "$dest"
    echo "✓ $name done"
}

for entry in "${WEIGHTS[@]}"; do
    name="${entry%%|*}"
    url="${entry##*|}"
    download "$name" "$url"
done

echo "================================"
echo "All weights ready in $WEIGHTS_DIR"
echo "================================"
