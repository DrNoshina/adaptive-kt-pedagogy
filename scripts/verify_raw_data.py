#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adaptive_kt.config import load_config
from adaptive_kt.data.ednet import find_kt3_user_files, load_questions


def verify_ednet(config: dict) -> None:
    root = Path(config["dataset"]["raw_root"])
    if not root.exists():
        raise FileNotFoundError(f"EdNet raw directory does not exist: {root}")
    questions = load_questions(root)
    user_files = find_kt3_user_files(root)
    sample = pd.read_csv(user_files[0], nrows=5)
    required = {"timestamp", "action_type", "item_id"}
    missing = required.difference(sample.columns)
    if missing:
        raise ValueError(f"KT3 sample file {user_files[0]} is missing {sorted(missing)}")
    print(f"EdNet root: {root.resolve()}")
    print(f"KT3 user files found: {len(user_files):,}")
    print(f"questions.csv rows: {len(questions):,}")
    print(f"sample user file: {user_files[0]}")
    print("EdNet raw-data check: OK")


def verify_generic(config: dict) -> None:
    dataset = config["dataset"]
    path = Path(dataset["input_file"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    sep = dataset.get("separator", ",")
    sample = pd.read_csv(path, sep=sep, nrows=5, low_memory=False, encoding=dataset.get("encoding"))
    columns = dataset["columns"]
    required_aliases = ["user_id", "item_id", "skill_id", "correct", "timestamp"]
    missing = {alias: columns.get(alias) for alias in required_aliases if columns.get(alias) not in sample.columns}
    if missing:
        raise ValueError(f"Configured columns are absent from {path}: {missing}")
    print(f"Dataset file: {path.resolve()}")
    print(f"Columns detected: {len(sample.columns)}")
    print("Raw-data check: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["dataset"]["name"] == "ednet_kt3":
        verify_ednet(config)
    else:
        verify_generic(config)


if __name__ == "__main__":
    main()
