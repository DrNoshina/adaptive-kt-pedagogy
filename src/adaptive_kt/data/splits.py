from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_user(
    frame: pd.DataFrame,
    train_size: float,
    validation_size: float,
    test_size: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    total = train_size + validation_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, received {total:.6f}.")
    if frame["user_id"].nunique() < 3:
        raise ValueError("At least three unique users are required for train/validation/test splitting.")

    first = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, remainder_idx = next(first.split(frame, groups=frame["user_id"]))
    train = frame.iloc[train_idx].copy()
    remainder = frame.iloc[remainder_idx].copy()

    relative_validation = validation_size / (validation_size + test_size)
    second = GroupShuffleSplit(n_splits=1, train_size=relative_validation, random_state=seed + 1)
    val_idx, test_idx = next(second.split(remainder, groups=remainder["user_id"]))
    validation = remainder.iloc[val_idx].copy()
    test = remainder.iloc[test_idx].copy()

    split = {"train": train, "validation": validation, "test": test}
    assert_user_disjoint(split)
    return split


def assert_user_disjoint(split: dict[str, pd.DataFrame]) -> None:
    names = ["train", "validation", "test"]
    users = {name: set(split[name]["user_id"].astype(str).unique()) for name in names}
    overlaps = {
        "train_validation": users["train"] & users["validation"],
        "train_test": users["train"] & users["test"],
        "validation_test": users["validation"] & users["test"],
    }
    bad = {name: values for name, values in overlaps.items() if values}
    if bad:
        counts = {name: len(values) for name, values in bad.items()}
        raise AssertionError(f"Learner leakage detected across splits: {counts}")


def split_assignment_frame(split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name in ("train", "validation", "test"):
        user_ids = sorted(split[name]["user_id"].astype(str).unique())
        rows.append(pd.DataFrame({"user_id": user_ids, "split": name}))
    result = pd.concat(rows, ignore_index=True)
    if result["user_id"].duplicated().any():
        raise AssertionError("A learner was assigned to more than one split.")
    return result
