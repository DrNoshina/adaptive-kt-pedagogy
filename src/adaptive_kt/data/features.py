from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adaptive_kt.data.schema import ALL_FEATURE_COLUMNS


@dataclass
class LearnerState:
    recent_window: int
    correct_total: int = 0
    interaction_total: int = 0
    response_time_total_s: float = 0.0
    attempt_total: float = 0.0
    error_streak: int = 0
    recent_correct: deque = field(default_factory=deque)
    recent_response_s: deque = field(default_factory=deque)
    skill_correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    skill_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self) -> None:
        self.recent_correct = deque(maxlen=self.recent_window)
        self.recent_response_s = deque(maxlen=self.recent_window)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def build_causal_features(
    interactions: pd.DataFrame,
    recent_window: int = 10,
    min_history: int = 5,
    max_response_time_seconds: float = 1200.0,
) -> pd.DataFrame:
    """Build learner-state features using only information available before each target response.

    Each learner is processed independently in chronological order. The current row's correctness,
    response time, and attempt count update the state only *after* the feature vector for that row
    has been emitted, which prevents target leakage.
    """
    if interactions.empty:
        return pd.DataFrame(columns=list(interactions.columns) + ALL_FEATURE_COLUMNS)

    required = {"user_id", "timestamp", "skill_id", "correct", "response_time_ms", "attempt_count"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"Missing required interaction columns: {sorted(missing)}")

    frame = interactions.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0).astype("int64")
    frame["correct"] = pd.to_numeric(frame["correct"], errors="coerce").fillna(0).astype("int8")
    frame["response_time_ms"] = pd.to_numeric(frame["response_time_ms"], errors="coerce").fillna(0.0)
    frame["attempt_count"] = pd.to_numeric(frame["attempt_count"], errors="coerce").fillna(1.0).clip(lower=1.0)
    frame = frame.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")

    prior_columns = {
        "explanation_count_prior": 0.0,
        "explanation_time_ms_prior": 0.0,
        "lecture_count_prior": 0.0,
        "lecture_time_ms_prior": 0.0,
        "review_count_prior": 0.0,
        "adaptive_offer_count_prior": 0.0,
    }
    for column, default in prior_columns.items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)

    outputs: list[dict] = []

    for _, user_rows in frame.groupby("user_id", sort=False):
        state = LearnerState(recent_window=recent_window)
        for row in user_rows.itertuples(index=False):
            row_dict = row._asdict()
            skill_id = str(row_dict["skill_id"])
            history_count = state.interaction_total

            historical_accuracy = _safe_rate(state.correct_total, history_count) if history_count else 0.5
            recent_accuracy = float(np.mean(state.recent_correct)) if state.recent_correct else 0.5
            skill_n = state.skill_total[skill_id]
            # Beta(1,1) prior. Only previous attempts on the same skill contribute.
            topic_mastery = (state.skill_correct[skill_id] + 1.0) / (skill_n + 2.0)
            mean_response_time = _safe_rate(state.response_time_total_s, history_count) if history_count else 30.0
            mean_attempt_count = _safe_rate(state.attempt_total, history_count) if history_count else 1.0

            explanation_count = float(row_dict.get("explanation_count_prior", 0.0) or 0.0)
            explanation_time_ms = float(row_dict.get("explanation_time_ms_prior", 0.0) or 0.0)
            lecture_count = float(row_dict.get("lecture_count_prior", 0.0) or 0.0)
            lecture_time_ms = float(row_dict.get("lecture_time_ms_prior", 0.0) or 0.0)
            review_count = float(row_dict.get("review_count_prior", 0.0) or 0.0)
            adaptive_offer_count = float(row_dict.get("adaptive_offer_count_prior", 0.0) or 0.0)
            history_denom = max(history_count, 1)

            feature_row = {
                **row_dict,
                "historical_accuracy": historical_accuracy,
                "recent_accuracy": recent_accuracy,
                "topic_mastery": topic_mastery,
                "mean_response_time": mean_response_time,
                "mean_attempt_count": mean_attempt_count,
                "error_rate": 1.0 - historical_accuracy,
                "explanation_rate": explanation_count / history_denom,
                "mean_explanation_time_s": (explanation_time_ms / 1000.0) / max(explanation_count, 1.0),
                "lecture_rate": lecture_count / history_denom,
                "mean_lecture_time_s": (lecture_time_ms / 1000.0) / max(lecture_count, 1.0),
                "review_rate": review_count / history_denom,
                "adaptive_offer_rate": adaptive_offer_count / history_denom,
            }

            if history_count >= min_history:
                outputs.append(feature_row)

            # Update learner state only after the current target's feature vector is created.
            correct = int(row_dict["correct"])
            response_s = float(row_dict["response_time_ms"]) / 1000.0
            response_s = float(np.clip(response_s, 0.0, max_response_time_seconds))
            attempts = float(row_dict["attempt_count"])

            state.interaction_total += 1
            state.correct_total += correct
            state.response_time_total_s += response_s
            state.attempt_total += attempts
            state.recent_correct.append(correct)
            state.recent_response_s.append(response_s)
            state.skill_total[skill_id] += 1
            state.skill_correct[skill_id] += correct
            state.error_streak = 0 if correct else state.error_streak + 1

    result = pd.DataFrame(outputs)
    if result.empty:
        return pd.DataFrame(columns=list(frame.columns) + ALL_FEATURE_COLUMNS)
    for column in ALL_FEATURE_COLUMNS:
        result[column] = (
            pd.to_numeric(result[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
        )
    return result.reset_index(drop=True)
