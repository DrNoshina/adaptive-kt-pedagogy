import pandas as pd

from adaptive_kt.models.pedagogy import assign_pedagogy_strategy


def test_policy_emits_expected_extremes():
    frame = pd.DataFrame(
        {
            "topic_mastery": [0.2, 0.9],
            "recent_accuracy": [0.2, 0.95],
            "mean_attempt_count": [1.0, 1.0],
            "error_rate": [0.8, 0.05],
        }
    )
    labels = assign_pedagogy_strategy(frame)
    assert labels.iloc[0] == "remedial_learning"
    assert labels.iloc[1] == "challenge_based_learning"
