"""Proposed-V3 data: difficulty-adjusted mastery on top of the V2 inputs.

The V1/V2 learner profile uses raw percentages (historical accuracy, topic mastery). A raw
percentage ignores how hard the answered questions were, and on an adaptive platform stronger
learners get harder questions, so raw percentages separate learners poorly. V3 adds an online
Elo-style estimate: learner ability overall, per knowledge tag and per TOEIC part, measured
against question easiness.

Leakage rules:
  * Every value of a row is stored BEFORE that row's answer updates the learner state.
  * Question easiness comes from TRAINING learners only. Inside training it is out-of-fold by
    learner, so a learner's own answers never enter the easiness used for that learner.
Rows, windows and targets are identical to V1/V2.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adaptive_kt.models.sequence import SequenceVocabulary, collate_sequences
from adaptive_kt.models.sequence_v2 import CTX_DIM, CTX_FEATURES, CTX_GROUPS, KTSequenceDatasetV2

MASTERY_FEATURES = [
    "item_easiness",
    "ability_long",
    "ability_recent",
    "tag_ability_mean",
    "tag_ability_min",
    "part_ability",
    "expected_correct",
    "tag_practice",
    "part_practice",
]
MASTERY_DIM = len(MASTERY_FEATURES)
CTX_FEATURES_V3 = CTX_FEATURES + MASTERY_FEATURES
CTX_DIM_V3 = CTX_DIM + MASTERY_DIM
CTX_GROUPS_V3 = dict(CTX_GROUPS, elo=list(range(CTX_DIM, CTX_DIM_V3)))
MAX_TAGS = 7
N_PARTS = 8  # 0 = unknown, 1..7 = TOEIC part

# Fixed before any run (standard Elo-for-education form K = a / (1 + b * n)); not tuned on test.
K_GLOBAL, D_GLOBAL, FLOOR_GLOBAL = 0.4, 0.05, 0.01
K_TAG, D_TAG, FLOOR_TAG = 0.4, 0.10, 0.02
K_PART, D_PART, FLOOR_PART = 0.3, 0.05, 0.01
K_RECENT = 0.10
EASINESS_FOLDS = 5
EASINESS_STRENGTH = 20.0


def load_question_meta(raw_root: str | Path) -> tuple[dict[str, int], dict[str, tuple[int, ...]], int]:
    """question_id -> part (1..7, 0 unknown) and question_id -> tag indices (1..n_tags)."""
    candidates = [p for p in sorted(Path(raw_root).rglob("questions.csv")) if "__MACOSX" not in p.parts]
    if not candidates:
        raise FileNotFoundError(f"questions.csv not found below {raw_root}")
    questions = pd.read_csv(candidates[0], dtype=str)
    parsed: dict[str, list[str]] = {}
    tokens: set[str] = set()
    for question_id, raw in zip(questions["question_id"], questions["tags"].fillna("")):
        tags = [t.strip() for t in str(raw).split(";")]
        tags = [t for t in dict.fromkeys(tags) if t and t not in ("-1", "nan")]
        parsed[str(question_id)] = tags
        tokens.update(tags)
    order = sorted(tokens, key=lambda t: (len(t), t))
    tag_index = {token: index + 1 for index, token in enumerate(order)}
    tags_of = {qid: tuple(tag_index[t] for t in tags)[:MAX_TAGS] for qid, tags in parsed.items()}
    part_of: dict[str, int] = {}
    for question_id, part in zip(questions["question_id"], questions["part"]):
        try:
            value = int(float(part))
        except (TypeError, ValueError):
            value = 0
        part_of[str(question_id)] = value if 1 <= value <= 7 else 0
    return part_of, tags_of, len(order)


def item_lookup(item_vocab: SequenceVocabulary, part_of: dict, tags_of: dict) -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables indexed by the item vocabulary code (0 = PAD, 1 = unseen item)."""
    item_part = np.zeros(item_vocab.size, dtype=np.int64)
    item_tags = np.zeros((item_vocab.size, MAX_TAGS), dtype=np.int64)
    for item, code in item_vocab.mapping.items():
        key = str(item)
        item_part[code] = part_of.get(key, 0)
        tags = tags_of.get(key, ())
        item_tags[code, :len(tags)] = tags
    return item_part, item_tags


def fit_item_easiness(items: np.ndarray, correct: np.ndarray, first: np.ndarray, n_items: int):
    """Smoothed logit accuracy per question from TRAINING rows.

    Returns (full, out_of_fold, fold_of_row). Training rows must use out_of_fold[fold_of_row]
    so that a learner's own answers are excluded; validation/test rows use full.
    """
    user_index = np.cumsum(first.astype(np.int64)) - 1
    fold = user_index % EASINESS_FOLDS
    y = correct.astype(np.float64)
    prior = float(y.mean())
    key = fold * n_items + items.astype(np.int64)
    size = EASINESS_FOLDS * n_items
    counts = np.bincount(key, minlength=size).reshape(EASINESS_FOLDS, n_items).astype(np.float64)
    hits = np.bincount(key, weights=y, minlength=size).reshape(EASINESS_FOLDS, n_items)

    def logit(h: np.ndarray, c: np.ndarray) -> np.ndarray:
        p = (h + EASINESS_STRENGTH * prior) / (c + EASINESS_STRENGTH)
        return np.log(p / (1.0 - p))

    full = logit(hits.sum(0), counts.sum(0)).astype(np.float32)
    out_of_fold = logit(hits.sum(0)[None, :] - hits, counts.sum(0)[None, :] - counts).astype(np.float32)
    return full, out_of_fold, fold.astype(np.int64)


def mastery_features(
    items: np.ndarray,
    correct: np.ndarray,
    first: np.ndarray,
    easiness_row: np.ndarray,
    item_part: np.ndarray,
    item_tags: np.ndarray,
) -> np.ndarray:
    """Online Elo-style mastery, one row per interaction (float16). State resets at each learner."""
    n = int(len(items))
    out = np.zeros((n, MASTERY_DIM), dtype=np.float32)
    item_list = items.tolist()
    answers = correct.tolist()
    starts = first.tolist()
    easiness = easiness_row.tolist()
    parts = item_part.tolist()
    tag_lists = [tuple(int(t) for t in row if t) for row in item_tags.tolist()]
    exp, tanh, log1p = math.exp, math.tanh, math.log1p
    log_tag, log_part = log1p(1000.0), log1p(5000.0)

    ability = recent = 0.0
    seen = 0
    tag_ability: dict[int, float] = {}
    tag_seen: dict[int, int] = {}
    part_ability = [0.0] * N_PARTS
    part_seen = [0] * N_PARTS
    for i in range(n):
        if starts[i]:
            ability = recent = 0.0
            seen = 0
            tag_ability = {}
            tag_seen = {}
            part_ability = [0.0] * N_PARTS
            part_seen = [0] * N_PARTS
        question = item_list[i]
        ease = easiness[i]
        part = parts[question]
        tags = tag_lists[question]
        if tags:
            total = 0.0
            lowest = 1e9
            practice = 0
            for tag in tags:
                value = tag_ability.get(tag, 0.0)
                total += value
                if value < lowest:
                    lowest = value
                practice += tag_seen.get(tag, 0)
            tag_mean = total / len(tags)
            tag_practice = practice / len(tags)
        else:
            tag_mean = lowest = 0.0
            tag_practice = 0.0
        part_value = part_ability[part]
        z = ability + tag_mean + part_value + ease
        expected = 1.0 / (1.0 + exp(-z)) if z > -30.0 else 0.0

        row = out[i]
        row[0] = tanh(0.5 * ease)
        row[1] = tanh(0.5 * ability)
        row[2] = tanh(0.5 * recent)
        row[3] = tanh(0.5 * tag_mean)
        row[4] = tanh(0.5 * lowest)
        row[5] = tanh(0.5 * part_value)
        row[6] = expected
        row[7] = log1p(min(tag_practice, 1000.0)) / log_tag
        row[8] = log1p(min(part_seen[part], 5000)) / log_part

        # The row's own answer is used only now, after its features were stored.
        answer = answers[i]
        error = answer - expected
        ability += max(K_GLOBAL / (1.0 + D_GLOBAL * seen), FLOOR_GLOBAL) * error
        seen += 1
        z_recent = recent + ease
        recent += K_RECENT * (answer - (1.0 / (1.0 + exp(-z_recent)) if z_recent > -30.0 else 0.0))
        for tag in tags:
            count = tag_seen.get(tag, 0)
            tag_ability[tag] = tag_ability.get(tag, 0.0) + max(K_TAG / (1.0 + D_TAG * count), FLOOR_TAG) * error
            tag_seen[tag] = count + 1
        part_ability[part] = part_value + max(K_PART / (1.0 + D_PART * part_seen[part]), FLOOR_PART) * error
        part_seen[part] += 1
    return out.astype(np.float16)


class KTSequenceDatasetV3(KTSequenceDatasetV2):
    """V2 dataset plus the mastery block appended to the context vector."""

    mastery: np.ndarray | None = None

    def first_rows(self) -> np.ndarray:
        return np.asarray(self.ctx[:, 0] > 0.5)

    def set_mastery(self, mastery: np.ndarray) -> None:
        if mastery.shape != (len(self.items), MASTERY_DIM):
            raise ValueError(f"mastery shape {mastery.shape} does not match {len(self.items)} rows")
        self.mastery = mastery

    def __getitem__(self, index):
        if self.mastery is None:
            raise RuntimeError("set_mastery() must be called before the dataset is used")
        sample = super().__getitem__(index)
        start, end = (int(value) for value in self.windows[index])
        extra = torch.from_numpy(self.mastery[start:end].astype(np.float32))
        return sample[:7] + (torch.cat([sample[7], extra], dim=1),) + sample[8:]


def collate_sequences_v3(batch):
    out = collate_sequences([sample[:7] for sample in batch])
    batch_size, max_len = out["items"].shape
    ctx = torch.zeros((batch_size, max_len, CTX_DIM_V3), dtype=torch.float32)
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
