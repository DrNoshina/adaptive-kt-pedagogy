from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def select_f1_threshold(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    """Select a binary decision threshold using validation data only."""
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    if y_true_arr.size == 0:
        raise ValueError("Cannot select a threshold from an empty target array.")
    candidates = np.unique(np.concatenate(([0.0], y_prob_arr, [1.0])))
    if len(candidates) > 1001:
        candidates = np.linspace(0.0, 1.0, 1001)
    scores = np.asarray(
        [f1_score(y_true_arr, (y_prob_arr >= threshold).astype(int), zero_division=0) for threshold in candidates]
    )
    best = int(np.argmax(scores))
    return float(candidates[best])


def binary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    if y_true_arr.size == 0:
        raise ValueError("Cannot evaluate an empty target array.")
    y_pred = (y_prob_arr >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "rmse": float(np.sqrt(np.mean((y_true_arr - y_prob_arr) ** 2))),
        "threshold": float(threshold),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def recommendation_metrics(y_true: np.ndarray, probabilities: np.ndarray, k: int = 3) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or len(y_true) != probabilities.shape[0]:
        raise ValueError("Recommendation probabilities must have shape [n_samples, n_classes].")
    k = max(1, min(int(k), probabilities.shape[1]))
    top_k = np.argsort(-probabilities, axis=1)[:, :k]
    hits = top_k == y_true[:, None]
    hit_any = hits.any(axis=1)
    precision = hit_any.astype(float) / k
    recall = hit_any.astype(float)
    ndcg = np.zeros(len(y_true), dtype=float)
    for i, row in enumerate(hits):
        ranks = np.flatnonzero(row)
        if ranks.size:
            ndcg[i] = 1.0 / math.log2(int(ranks[0]) + 2.0)
    return {
        f"precision@{k}": float(np.mean(precision)),
        f"recall@{k}": float(np.mean(recall)),
        f"ndcg@{k}": float(np.mean(ndcg)),
    }
