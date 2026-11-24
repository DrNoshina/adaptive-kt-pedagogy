#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest
rm -rf data/processed/synthetic results/smoke
python scripts/make_synthetic_data.py

if command -v adaptive-kt >/dev/null 2>&1; then
  adaptive-kt run --config configs/smoke.yaml
else
  python -m adaptive_kt.cli run --config configs/smoke.yaml
fi

echo "Smoke test completed successfully."
