from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_kt.data.features import build_causal_features


def _column(frame: pd.DataFrame, name: str | None, default: object = None) -> pd.Series:
    if name and name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _parse_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return numeric.fillna(0).astype("int64")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    values = parsed.astype("int64") // 10**6
    values = values.where(parsed.notna(), 0)
    return values.astype("int64")


def prepare_generic_dataset(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_cfg = config["dataset"]
    path = Path(dataset_cfg["input_file"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    separator = dataset_cfg.get("separator", ",")
    encoding = dataset_cfg.get("encoding")
    frame = pd.read_csv(path, sep=separator, low_memory=False, encoding=encoding)
    columns = dataset_cfg["columns"]

    required_aliases = ["user_id", "item_id", "skill_id", "correct", "timestamp"]
    unresolved = [alias for alias in required_aliases if columns.get(alias) not in frame.columns]
    if unresolved:
        details = {alias: columns.get(alias) for alias in unresolved}
        raise ValueError(f"Configured columns not present in {path}: {details}")

    response_time_ms = pd.Series(np.zeros(len(frame)), index=frame.index, dtype="float64")
    if columns.get("response_time_ms") in frame.columns:
        response_time_ms = pd.to_numeric(frame[columns["response_time_ms"]], errors="coerce").fillna(0.0)
    elif columns.get("response_time_seconds") in frame.columns:
        response_time_ms = pd.to_numeric(frame[columns["response_time_seconds"]], errors="coerce").fillna(0.0) * 1000.0

    attempt_count = pd.Series(np.ones(len(frame)), index=frame.index, dtype="float64")
    if columns.get("attempt_count") in frame.columns:
        attempt_count = pd.to_numeric(frame[columns["attempt_count"]], errors="coerce").fillna(1.0).clip(lower=1.0)
    elif columns.get("incorrects") in frame.columns:
        incorrects = pd.to_numeric(frame[columns["incorrects"]], errors="coerce").fillna(0.0).clip(lower=0.0)
        attempt_count = 1.0 + incorrects

    correct_numeric = pd.to_numeric(frame[columns["correct"]], errors="coerce")
    valid = (
        frame[columns["user_id"]].notna()
        & frame[columns["item_id"]].notna()
        & correct_numeric.notna()
    )
    frame = frame.loc[valid].copy()
    correct_numeric = correct_numeric.loc[valid]
    response_time_ms = response_time_ms.loc[valid]
    attempt_count = attempt_count.loc[valid]

    skill = frame[columns["skill_id"]].fillna("unknown").astype(str)
    skill = skill.str.split("~~", regex=False).str[0].replace({"": "unknown", "nan": "unknown"})

    item_id = frame[columns["item_id"]].astype(str)
    extra_item = columns.get("item_id_extra")
    if extra_item and extra_item in frame.columns:
        item_id = item_id + "::" + frame[extra_item].fillna("unknown").astype(str)

    interactions = pd.DataFrame(
        {
            "user_id": frame[columns["user_id"]].astype(str),
            "timestamp": _parse_timestamp(frame[columns["timestamp"]]),
            "item_id": item_id,
            "skill_id": skill,
            "correct": correct_numeric.clip(0, 1).astype("int8"),
            "response_time_ms": response_time_ms,
            "attempt_count": attempt_count,
            "explanation_count_prior": 0,
            "explanation_time_ms_prior": 0,
            "lecture_count_prior": 0,
            "lecture_time_ms_prior": 0,
            "review_count_prior": 0,
            "adaptive_offer_count_prior": 0,
        }
    )
    interactions = interactions.dropna(subset=["user_id", "item_id", "skill_id"]).reset_index(drop=True)
    features = build_causal_features(
        interactions,
        recent_window=int(dataset_cfg.get("recent_window", 10)),
        min_history=int(dataset_cfg.get("min_history", 5)),
        max_response_time_seconds=float(dataset_cfg.get("max_response_time_seconds", 1200)),
    )
    return interactions, features
