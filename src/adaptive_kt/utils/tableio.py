from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def parquet_available() -> bool:
    return importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None


def write_table(frame: pd.DataFrame, path_without_suffix: Path) -> Path:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    if parquet_available():
        target = path_without_suffix.with_suffix(".parquet")
        frame.to_parquet(target, index=False)
    else:
        target = path_without_suffix.with_suffix(".pkl")
        frame.to_pickle(target)
    return target


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported prepared table format: {path}")


def read_collection(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(f"Prepared data not found: {directory}")
    if directory.is_file():
        return read_table(directory)
    files = sorted(directory.glob("*.parquet")) + sorted(directory.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No prepared data shards found in {directory}")
    return pd.concat((read_table(path) for path in files), ignore_index=True)
