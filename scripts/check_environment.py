#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"XGBoost: {xgboost.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    else:
        print("WARNING: CUDA is not available to PyTorch. Deep KT will run on CPU.")


if __name__ == "__main__":
    main()
