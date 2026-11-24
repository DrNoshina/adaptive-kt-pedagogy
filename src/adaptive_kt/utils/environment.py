from __future__ import annotations

import platform
import sys
from importlib import metadata

import torch


def environment_snapshot() -> dict[str, object]:
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "torch", "shap", "matplotlib", "PyYAML"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }
