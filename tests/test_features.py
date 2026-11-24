import pandas as pd

from adaptive_kt.data.features import build_causal_features


def test_causal_features_do_not_use_current_outcome():
    frame = pd.DataFrame(
        {
            "user_id": ["u1"] * 6,
            "timestamp": [1, 2, 3, 4, 5, 6],
            "item_id": [f"q{i}" for i in range(6)],
            "skill_id": ["s1"] * 6,
            "correct": [1, 0, 1, 1, 0, 1],
            "response_time_ms": [10000] * 6,
            "attempt_count": [1] * 6,
            "explanation_count_prior": [0] * 6,
            "lecture_count_prior": [0] * 6,
            "review_count_prior": [0] * 6,
            "adaptive_offer_count_prior": [0] * 6,
        }
    )
    features = build_causal_features(frame, recent_window=3, min_history=2)
    first = features.iloc[0]
    assert first["timestamp"] == 3
    assert abs(first["historical_accuracy"] - 0.5) < 1e-6
    assert abs(first["topic_mastery"] - 0.5) < 1e-6


def test_feature_builder_returns_expected_rows():
    frame = pd.DataFrame(
        {
            "user_id": ["u1"] * 5,
            "timestamp": range(5),
            "item_id": [f"q{i}" for i in range(5)],
            "skill_id": ["s"] * 5,
            "correct": [1, 1, 0, 1, 0],
            "response_time_ms": [1000] * 5,
            "attempt_count": [1] * 5,
        }
    )
    features = build_causal_features(frame, min_history=2)
    assert len(features) == 3


def test_behavior_features_are_prior_only():
    frame = pd.DataFrame(
        {
            "user_id": ["u1"] * 4,
            "timestamp": [1, 2, 3, 4],
            "item_id": ["q1", "q2", "q3", "q4"],
            "skill_id": ["s1"] * 4,
            "correct": [0, 1, 1, 0],
            "response_time_ms": [1000] * 4,
            "attempt_count": [1] * 4,
            "explanation_count_prior": [0, 1, 1, 2],
            "explanation_time_ms_prior": [0, 10000, 10000, 30000],
            "lecture_count_prior": [0, 0, 1, 1],
            "lecture_time_ms_prior": [0, 0, 60000, 60000],
            "review_count_prior": [0, 0, 0, 1],
            "adaptive_offer_count_prior": [0, 0, 1, 1],
        }
    )
    features = build_causal_features(frame, min_history=1)
    second_target = features.iloc[0]  # original timestamp=2; one previous interaction
    assert second_target["timestamp"] == 2
    assert second_target["explanation_rate"] == 1.0
    assert second_target["mean_explanation_time_s"] == 10.0
    assert second_target["lecture_rate"] == 0.0
