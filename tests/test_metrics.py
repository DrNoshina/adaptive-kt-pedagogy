import numpy as np

from adaptive_kt.evaluation.metrics import binary_metrics, recommendation_metrics


def test_binary_metrics_are_bounded():
    result = binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert result["accuracy"] == 1.0
    assert result["auc"] == 1.0
    assert 0.0 <= result["rmse"] <= 1.0


def test_recommendation_metrics_perfect_top1():
    y = np.array([0, 1, 2])
    p = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
    result = recommendation_metrics(y, p, k=1)
    assert result["precision@1"] == 1.0
    assert result["recall@1"] == 1.0
    assert result["ndcg@1"] == 1.0
