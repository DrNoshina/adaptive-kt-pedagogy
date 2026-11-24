#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Installing Ubuntu prerequisites..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip curl

echo "[2/4] Creating Python virtual environment..."
python3 -m venv .venv

echo "[3/4] Installing the research package and reproducibility tools..."
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install -e ".[dev]"

echo "[4/4] Creating data directories..."
mkdir -p \
  data/raw/ednet/KT3 \
  data/raw/ednet/contents \
  data/raw/ednet/downloads \
  data/raw/assistments \
  data/raw/kddcup \
  data/processed \
  results

echo
echo "Setup complete."
echo "Next command: source .venv/bin/activate"
echo "Then run:      python scripts/check_environment.py"
