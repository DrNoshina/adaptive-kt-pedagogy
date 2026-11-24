"""Proposed-V2 sequence data.

Adds a pre-response context vector for every interaction. Every value is known BEFORE the learner
answers that question (bundle entry time, previous answers, earlier explanation/lecture activity,
earlier attempts on the same question/skill, source and platform of the bundle). The target row's
own correctness, response time and answer changes are never used for its own context.

Rows, windows and eligible targets are identical to KTSequenceDataset (V1), so V1 and V2 are
scored on exactly the same targets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adaptive_kt.models.sequence import (
    BEHAVIOR_FEATURE_INDEX,
    KTSequenceDataset,
    SequenceVocabulary,
    _column_array,
    collate_sequences,
)

CTX_FEATURES = [
    "first_interaction",
    "lag_before_start",
    "same_bundle",
    "bundle_position",
    "session_position",
    "history_length",
    "item_seen_before",
    "item_prior_attempts",
    "item_last_correct",
    "item_time_since",
    "skill_seen_before",
    "skill_prior_attempts",
    "skill_last_correct",
    "skill_time_since",
    "explanation_since_prev",
    "explanation_count_delta",
    "explanation_time_delta",
    "lecture_since_prev",
    "lecture_count_delta",
    "lecture_time_delta",
]
CTX_DIM = len(CTX_FEATURES)

CTX_GROUPS = {
    "experience": [0, 5],
    "timing": [1, 2, 3, 4],
    "recency": [6, 7, 8, 9, 10, 11, 12, 13],
    "engagement": [14, 15, 16, 17, 18, 19],
}

_MONTH_S = 30.0 * 24.0 * 3600.0
_SESSION_BREAK_S = 3600.0
_MAX_CATEGORIES = 500


def _log_scale(values: np.ndarray, cap: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, cap)
    return (np.log1p(clipped) / np.log1p(cap)).astype(np.float32)


def _position_since(flags: np.ndarray) -> np.ndarray:
    """Rows since the latest True flag (0 on the flag row). flags[0] must be True."""
    starts = np.flatnonzero(flags)
    group = np.cumsum(flags) - 1
    return np.arange(flags.shape[0], dtype=np.int64) - starts[group]


def _previous_same_key(users: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Previous row index of the same learner with the same key (-1 if none) and the number of
    earlier rows with that key. Rows must already be chronological within each learner."""
    n = keys.shape[0]
    order = np.lexsort((np.arange(n, dtype=np.int64), keys, users))
    sorted_users = users[order]
    sorted_keys = keys[order]
    same = np.zeros(n, dtype=bool)
    same[1:] = (sorted_users[1:] == sorted_users[:-1]) & (sorted_keys[1:] == sorted_keys[:-1])
    previous_sorted = np.full(n, -1, dtype=np.int64)
    previous_sorted[1:] = np.where(same[1:], order[:-1], -1)
    count_sorted = _position_since(~same)
    previous = np.empty(n, dtype=np.int64)
    count = np.empty(n, dtype=np.int64)
    previous[order] = previous_sorted
    count[order] = count_sorted
    return previous, count


def build_context(ordered: pd.DataFrame) -> np.ndarray:
    """Pre-response context (float16, values in [0, 1]) for rows sorted by (user_id, timestamp, item_id)."""
    n = len(ordered)
    ctx = np.zeros((n, CTX_DIM), dtype=np.float16)
    if n == 0:
        return ctx

    users = pd.factorize(ordered["user_id"].astype(str))[0].astype(np.int64)
    ts = _column_array(ordered, "timestamp", 0.0, np.float64)
    response_ms = np.maximum(_column_array(ordered, "response_time_ms", 0.0, np.float64), 0.0)
    correct = _column_array(ordered, "correct", 0.0, np.float64)
    # KT3 parser: timestamp = first answer; response_time = time since bundle entry (first
    # question of a bundle) or since the previous answer in the same bundle. So start is the
    # moment the question became available, i.e. known before the learner answers.
    start = ts - response_ms

    first = np.ones(n, dtype=bool)
    first[1:] = users[1:] != users[:-1]
    previous_ts = np.concatenate(([0.0], ts[:-1]))
    raw_gap = start - previous_ts
    gap_s = np.where(first, 0.0, np.maximum(raw_gap, 0.0) / 1000.0)
    same_bundle = (~first) & (raw_gap <= 0.0)

    ctx[:, 0] = first
    ctx[:, 1] = _log_scale(gap_s, _MONTH_S)
    ctx[:, 2] = same_bundle
    ctx[:, 3] = _log_scale(_position_since(~same_bundle), 10.0)
    ctx[:, 4] = _log_scale(_position_since(first | (gap_s > _SESSION_BREAK_S)), 500.0)
    ctx[:, 5] = _log_scale(_position_since(first), 10000.0)
    del previous_ts, raw_gap, gap_s, same_bundle

    for column, base, count_cap in (("item_id", 6, 10.0), ("skill_id", 10, 1000.0)):
        keys = pd.factorize(ordered[column].astype(str))[0].astype(np.int64)
        previous, count = _previous_same_key(users, keys)
        seen = previous >= 0
        safe = np.maximum(previous, 0)
        ctx[:, base] = seen
        ctx[:, base + 1] = _log_scale(count, count_cap)
        ctx[:, base + 2] = np.where(seen, correct[safe], 0.0)
        ctx[:, base + 3] = _log_scale(np.where(seen, np.maximum(start - ts[safe], 0.0) / 1000.0, 0.0), _MONTH_S)
        del keys, previous, count, seen, safe

    def delta(column: str) -> np.ndarray:
        values = _column_array(ordered, column, 0.0, np.float64)
        previous_values = np.concatenate(([0.0], values[:-1]))
        previous_values[first] = 0.0
        return np.maximum(values - previous_values, 0.0)

    explanation_n = delta("explanation_count_prior")
    lecture_n = delta("lecture_count_prior")
    ctx[:, 14] = explanation_n > 0
    ctx[:, 15] = _log_scale(explanation_n, 10.0)
    ctx[:, 16] = _log_scale(delta("explanation_time_ms_prior") / 1000.0, 3600.0)
    ctx[:, 17] = lecture_n > 0
    ctx[:, 18] = _log_scale(lecture_n, 10.0)
    ctx[:, 19] = _log_scale(delta("lecture_time_ms_prior") / 1000.0, 7200.0)
    return ctx


def fit_category_vocab(frame: pd.DataFrame, column: str) -> SequenceVocabulary:
    """Fit on TRAINING learners only. Returns an empty vocabulary if the column is absent."""
    if column not in frame.columns:
        return SequenceVocabulary({})
    vocab = SequenceVocabulary.fit(frame[column].fillna("").astype(str))
    if len(vocab.mapping) > _MAX_CATEGORIES:
        print(f"[sequence_v2] {column}: {len(vocab.mapping)} categories; disabled", flush=True)
        return SequenceVocabulary({})
    return vocab


def _encode_category(frame: pd.DataFrame, column: str, vocab: SequenceVocabulary | None) -> np.ndarray:
    if vocab is None or not vocab.mapping or column not in frame.columns:
        return np.zeros(len(frame), dtype=np.int16)
    codes = frame[column].fillna("").astype(str).map(vocab.mapping)
    return codes.fillna(1).to_numpy(dtype=np.int16)


class KTSequenceDatasetV2(KTSequenceDataset):
    _ARRAYS = ("skills", "items", "correct", "behavior", "lag", "eligible", "windows", "ctx", "source", "platform")

    def __init__(
        self,
        frame: pd.DataFrame,
        skill_vocab: SequenceVocabulary,
        item_vocab: SequenceVocabulary,
        source_vocab: SequenceVocabulary | None,
        platform_vocab: SequenceVocabulary | None,
        max_len: int,
        min_history: int = 1,
        split_name: str = "split",
        cache_path: str | Path | None = None,
    ):
        self.max_len = int(max_len)
        self.min_history = int(min_history)
        self.split_name = str(split_name)
        signature = np.asarray(
            [
                len(frame),
                frame["user_id"].nunique(),
                self.max_len,
                self.min_history,
                skill_vocab.size,
                item_vocab.size,
                source_vocab.size if source_vocab is not None else 0,
                platform_vocab.size if platform_vocab is not None else 0,
                CTX_DIM,
            ],
            dtype=np.int64,
        )
        cache_path = Path(cache_path) if cache_path else None
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    if np.array_equal(data["signature"], signature):
                        for name in self._ARRAYS:
                            setattr(self, name, data[name])
                        print(f"[sequence_v2:{self.split_name}] cache loaded: {cache_path}", flush=True)
                        return
                print(f"[sequence_v2:{self.split_name}] cache signature changed; rebuilding", flush=True)
            except Exception as exc:  # corrupted/partial cache -> rebuild
                print(f"[sequence_v2:{self.split_name}] cache unreadable ({exc}); rebuilding", flush=True)

        super().__init__(
            frame,
            skill_vocab=skill_vocab,
            item_vocab=item_vocab,
            max_len=max_len,
            min_history=min_history,
            split_name=split_name,
        )
        ordered = frame.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
        check = ordered["item_id"].astype(str).map(item_vocab.mapping).fillna(1).to_numpy(dtype=np.int32)
        if not np.array_equal(check, self.items):
            raise AssertionError("V2 context rows are not aligned with the V1 sequence rows")
        del check
        self.ctx = build_context(ordered)
        self.source = _encode_category(ordered, "source", source_vocab)
        self.platform = _encode_category(ordered, "platform", platform_vocab)
        del ordered

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_name(cache_path.stem + ".tmp.npz")
                np.savez(temporary, signature=signature, **{name: getattr(self, name) for name in self._ARRAYS})
                temporary.replace(cache_path)
                print(f"[sequence_v2:{self.split_name}] cache saved: {cache_path}", flush=True)
            except Exception as exc:
                print(f"[sequence_v2:{self.split_name}] cache not saved ({exc})", flush=True)

    def __getitem__(self, index):
        start, end = (int(value) for value in self.windows[index])
        behavior = self.behavior[start:end]
        return (
            torch.from_numpy(self.skills[start:end]),
            torch.from_numpy(self.items[start:end]),
            torch.from_numpy(self.correct[start:end]),
            torch.from_numpy(behavior),
            torch.from_numpy(behavior[:, BEHAVIOR_FEATURE_INDEX["current_elapsed"]]),
            torch.from_numpy(self.lag[start:end]),
            torch.from_numpy(self.eligible[start:end]),
            torch.from_numpy(self.ctx[start:end].astype(np.float32)),
            torch.from_numpy(self.source[start:end].astype(np.int64)),
            torch.from_numpy(self.platform[start:end].astype(np.int64)),
        )


def collate_sequences_v2(batch):
    out = collate_sequences([sample[:7] for sample in batch])
    batch_size, max_len = out["items"].shape
    ctx = torch.zeros((batch_size, max_len, CTX_DIM), dtype=torch.float32)
    source = torch.zeros((batch_size, max_len), dtype=torch.long)
    platform = torch.zeros((batch_size, max_len), dtype=torch.long)
    for row, sample in enumerate(batch):
        length = int(sample[7].shape[0])
        ctx[row, :length] = sample[7]
        source[row, :length] = sample[8]
        platform[row, :length] = sample[9]
    out["ctx"] = ctx
    out["source"] = source
    out["platform"] = platform
    return out
