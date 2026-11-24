#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/gdown" ]]; then
  echo "ERROR: .venv/bin/gdown was not found."
  echo "Run: bash scripts/setup_ubuntu.sh"
  echo "Then: source .venv/bin/activate"
  exit 1
fi

DOWNLOAD_DIR="data/raw/ednet/downloads"
KT3_DIR="data/raw/ednet/KT3"
CONTENTS_DIR="data/raw/ednet/contents"
mkdir -p "$DOWNLOAD_DIR" "$KT3_DIR" "$CONTENTS_DIR"

KT3_ZIP="$DOWNLOAD_DIR/ednet_kt3.zip"
CONTENTS_ZIP="$DOWNLOAD_DIR/ednet_contents.zip"

# Official EdNet GitHub short links currently resolve to these public Google Drive files.
KT3_URL="https://drive.google.com/uc?id=1TVyGIWU1Mn3UCjjeD6bcZ57YspByUV7-"
CONTENTS_URL="https://drive.google.com/uc?id=117aYJAWG3GU48suS66NPaB82HwFj6xWS"

echo "Downloading EdNet-KT3 (~0.8 GB compressed)..."
.venv/bin/gdown "$KT3_URL" -O "$KT3_ZIP"

echo "Downloading EdNet content metadata..."
.venv/bin/gdown "$CONTENTS_URL" -O "$CONTENTS_ZIP"

echo "Extracting EdNet-KT3..."
unzip -q -o "$KT3_ZIP" -d "$KT3_DIR"

echo "Extracting EdNet Contents..."
unzip -q -o "$CONTENTS_ZIP" -d "$CONTENTS_DIR"

echo
echo "Primary dataset download/extraction complete."
echo "Next command: python scripts/verify_raw_data.py --config configs/ednet_kt3.yaml"
