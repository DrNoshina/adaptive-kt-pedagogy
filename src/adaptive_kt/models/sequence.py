from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceVocabulary:
    mapping: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.mapping) + 2  # PAD=0, UNK=1

    def encode(self, value: object) -> int:
        return self.mapping.get(str(value), 1)

    @classmethod
    def fit(cls, values: pd.Series) -> "SequenceVocabulary":
        values = values.astype(str)
        unique = sorted(values.unique().tolist())
        return cls({value: idx + 2 for idx, value in enumerate(unique)})


SkillVocabulary = SequenceVocabulary

CAUSAL_BEHAVIOR_DIM = 12
BEHAVIOR_DIM = 14

BEHAVIOR_FEATURE_INDEX = {
    "historical_accuracy": 0,
    "recent_accuracy": 1,
    "topic_mastery": 2,
    "mean_response_time": 3,
    "mean_attempt_count": 4,
    "error_rate": 5,
    "explanation_rate": 6,
    "mean_explanation_time_s": 7,
    "lecture_rate": 8,
    "mean_lecture_time_s": 9,
    "review_rate": 10,
    "adaptive_offer_rate": 11,
    "current_elapsed": 12,
    "current_attempts": 13,
}


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _log_scale(value: float, cap: float) -> float:
    value = float(np.clip(value, 0.0, cap))
    return float(np.log1p(value) / np.log1p(cap)) if cap > 0 else 0.0


def _column_array(group: pd.DataFrame, name: str, default: float = 0.0, dtype=np.float64) -> np.ndarray:
    if name not in group.columns:
        return np.full(len(group), default, dtype=dtype)
    values = pd.to_numeric(group[name], errors="coerce").fillna(default)
    return values.to_numpy(dtype=dtype, copy=False)


def _causal_behavior_matrix(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strictly causal learner-state features for one chronological learner sequence."""
    n = len(group)
    behavior = np.zeros((n, BEHAVIOR_DIM), dtype=np.float32)
    elapsed = np.zeros(n, dtype=np.float32)
    lag = np.zeros(n, dtype=np.float32)

    skills = group["skill_id"].astype(str).to_numpy(copy=False)
    corrects = _column_array(group, "correct", 0.0, np.int8)
    response_ms = _column_array(group, "response_time_ms", 0.0)
    attempts_arr = _column_array(group, "attempt_count", 1.0)
    timestamps = _column_array(group, "timestamp", 0.0, np.int64)
    explanation_count_arr = _column_array(group, "explanation_count_prior", 0.0)
    explanation_time_arr = _column_array(group, "explanation_time_ms_prior", 0.0)
    lecture_count_arr = _column_array(group, "lecture_count_prior", 0.0)
    lecture_time_arr = _column_array(group, "lecture_time_ms_prior", 0.0)
    review_count_arr = _column_array(group, "review_count_prior", 0.0)
    adaptive_count_arr = _column_array(group, "adaptive_offer_count_prior", 0.0)

    correct_total = 0
    interaction_total = 0
    response_total_s = 0.0
    attempt_total = 0.0
    recent_correct: deque[int] = deque(maxlen=10)
    skill_correct: dict[str, int] = defaultdict(int)
    skill_total: dict[str, int] = defaultdict(int)
    previous_timestamp: int | None = None

    for idx in range(n):
        skill = skills[idx]
        history = interaction_total
        historical_accuracy = _rate(correct_total, history) if history else 0.5
        recent_accuracy = float(np.mean(recent_correct)) if recent_correct else 0.5
        skill_n = skill_total[skill]
        topic_mastery = (skill_correct[skill] + 1.0) / (skill_n + 2.0)
        mean_response_s = _rate(response_total_s, history) if history else 30.0
        mean_attempts = _rate(attempt_total, history) if history else 1.0

        explanation_count = float(explanation_count_arr[idx])
        explanation_time_s = float(explanation_time_arr[idx]) / 1000.0
        lecture_count = float(lecture_count_arr[idx])
        lecture_time_s = float(lecture_time_arr[idx]) / 1000.0
        review_count = float(review_count_arr[idx])
        adaptive_count = float(adaptive_count_arr[idx])
        denom = max(history, 1)

        response_s = float(response_ms[idx]) / 1000.0
        attempts = float(attempts_arr[idx])
        timestamp = int(timestamps[idx])
        lag_s = 0.0 if previous_timestamp is None else max(0.0, (timestamp - previous_timestamp) / 1000.0)

        elapsed[idx] = _log_scale(response_s, 1200.0)
        lag[idx] = _log_scale(lag_s, 30.0 * 24.0 * 3600.0)
        behavior[idx] = np.asarray(
            [
                historical_accuracy,
                recent_accuracy,
                topic_mastery,
                _log_scale(mean_response_s, 1200.0),
                _log_scale(mean_attempts, 10.0),
                1.0 - historical_accuracy,
                explanation_count / denom,
                _log_scale(explanation_time_s / max(explanation_count, 1.0), 3600.0),
                lecture_count / denom,
                _log_scale(lecture_time_s / max(lecture_count, 1.0), 7200.0),
                review_count / denom,
                adaptive_count / denom,
                elapsed[idx],
                _log_scale(attempts, 10.0),
            ],
            dtype=np.float32,
        )

        correct = int(corrects[idx])
        interaction_total += 1
        correct_total += correct
        response_total_s += float(np.clip(response_s, 0.0, 1200.0))
        attempt_total += max(attempts, 1.0)
        recent_correct.append(correct)
        skill_total[skill] += 1
        skill_correct[skill] += correct
        previous_timestamp = timestamp

    return behavior, elapsed, lag


class KTSequenceDataset(Dataset):
    """Memory-efficient KT sequence dataset.

    The previous implementation copied every overlapping window into ``self.samples``. On
    EdNet-KT3 that duplicated tens of millions of values and consumed >20 GB RAM before the
    first epoch. This implementation stores each interaction exactly once and stores only
    compact (start, end) window indices.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        skill_vocab: SequenceVocabulary,
        item_vocab: SequenceVocabulary,
        max_len: int,
        min_history: int = 1,
        min_sequence_len: int = 3,
        split_name: str = "split",
    ):
        self.max_len = int(max_len)
        self.min_history = int(min_history)
        self.split_name = str(split_name)

        required = {"user_id", "timestamp", "item_id", "skill_id", "correct"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Sequence data missing columns: {sorted(missing)}")

        started = time.time()
        print(
            f"[sequence:{self.split_name}] compact build starting: {len(frame):,} interactions",
            flush=True,
        )

        ordered = frame.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
        n_rows = len(ordered)

        # Encode once. int32 is sufficient for the vocabulary sizes and halves storage vs int64.
        skill_map = ordered["skill_id"].astype(str).map(skill_vocab.mapping)
        item_map = ordered["item_id"].astype(str).map(item_vocab.mapping)
        self.skills = skill_map.fillna(1).to_numpy(dtype=np.int32, copy=True)
        self.items = item_map.fillna(1).to_numpy(dtype=np.int32, copy=True)
        self.correct = pd.to_numeric(ordered["correct"], errors="coerce").fillna(0).to_numpy(dtype=np.uint8, copy=True)
        self.behavior = np.zeros((n_rows, BEHAVIOR_DIM), dtype=np.float32)
        self.lag = np.zeros(n_rows, dtype=np.float32)
        self.eligible = np.zeros(n_rows, dtype=np.bool_)

        users = ordered["user_id"].astype(str).to_numpy(copy=False)
        if n_rows:
            boundaries = np.concatenate(
                ([0], np.flatnonzero(users[1:] != users[:-1]) + 1, [n_rows])
            ).astype(np.int64, copy=False)
        else:
            boundaries = np.asarray([0], dtype=np.int64)

        windows: list[tuple[int, int]] = []
        user_count = max(len(boundaries) - 1, 0)
        progress_every = max(10000, user_count // 20 if user_count else 10000)

        for user_idx in range(user_count):
            start = int(boundaries[user_idx])
            end = int(boundaries[user_idx + 1])
            length = end - start
            if length <= 0:
                continue

            group = ordered.iloc[start:end]
            behavior, _elapsed, lag = _causal_behavior_matrix(group)
            self.behavior[start:end] = behavior
            self.lag[start:end] = lag
            self.eligible[start:end] = np.arange(length, dtype=np.int64) >= self.min_history

            if length >= 2:
                window = self.max_len + 1
                for local_start in range(0, length - 1, self.max_len):
                    local_end = min(local_start + window, length)
                    if local_end - local_start >= 2:
                        windows.append((start + local_start, start + local_end))

            done = user_idx + 1
            if done % progress_every == 0 or done == user_count:
                elapsed_s = time.time() - started
                print(
                    f"[sequence:{self.split_name}] users {done:,}/{user_count:,} "
                    f"({100.0 * done / max(user_count, 1):.1f}%) | windows={len(windows):,} | {elapsed_s/60:.1f} min",
                    flush=True,
                )

        self.windows = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
        # elapsed is already behavior feature 12, so do not store a duplicate full-size array.
        del ordered
        print(
            f"[sequence:{self.split_name}] compact build complete: {len(self.windows):,} windows | "
            f"{(time.time() - started)/60:.1f} min",
            flush=True,
        )

    def __len__(self):
        return int(self.windows.shape[0])

    def __getitem__(self, index):
        start, end = self.windows[index]
        start = int(start)
        end = int(end)
        behavior = self.behavior[start:end]
        return (
            torch.from_numpy(self.skills[start:end]),
            torch.from_numpy(self.items[start:end]),
            torch.from_numpy(self.correct[start:end]),
            torch.from_numpy(behavior),
            torch.from_numpy(behavior[:, BEHAVIOR_FEATURE_INDEX["current_elapsed"]]),
            torch.from_numpy(self.lag[start:end]),
            torch.from_numpy(self.eligible[start:end]),
        )


def collate_sequences(batch):
    max_len = max(items.numel() for _, items, _, _, _, _, _ in batch)
    batch_size = len(batch)

    skills_out = torch.zeros((batch_size, max_len), dtype=torch.long)
    items_out = torch.zeros((batch_size, max_len), dtype=torch.long)
    correct_out = torch.zeros((batch_size, max_len), dtype=torch.long)
    behavior_out = torch.zeros((batch_size, max_len, BEHAVIOR_DIM), dtype=torch.float32)
    elapsed_out = torch.zeros((batch_size, max_len), dtype=torch.float32)
    lag_out = torch.zeros((batch_size, max_len), dtype=torch.float32)
    eligible_out = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for idx, (skills, items, correct, behavior, elapsed, lag, eligible) in enumerate(batch):
        length = items.numel()
        skills_out[idx, :length] = skills.long()
        items_out[idx, :length] = items.long()
        correct_out[idx, :length] = correct.long()
        behavior_out[idx, :length] = behavior.float()
        elapsed_out[idx, :length] = elapsed.float()
        lag_out[idx, :length] = lag.float()
        eligible_out[idx, :length] = eligible.bool()

    return {
        "skills": skills_out,
        "items": items_out,
        "correct": correct_out,
        "behavior": behavior_out,
        "elapsed": elapsed_out,
        "lag": lag_out,
        "eligible": eligible_out,
    }
