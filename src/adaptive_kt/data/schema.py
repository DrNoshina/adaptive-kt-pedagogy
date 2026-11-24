from __future__ import annotations

INTERACTION_COLUMNS = [
    "user_id",
    "timestamp",
    "item_id",
    "skill_id",
    "correct",
    "response_time_ms",
    "attempt_count",
    "explanation_count_prior",
    "explanation_time_ms_prior",
    "lecture_count_prior",
    "lecture_time_ms_prior",
    "review_count_prior",
    "adaptive_offer_count_prior",
]

CORE_FEATURE_COLUMNS = [
    "historical_accuracy",
    "recent_accuracy",
    "topic_mastery",
    "mean_response_time",
    "mean_attempt_count",
    "error_rate",
]

EDNET_BEHAVIOR_FEATURE_COLUMNS = [
    "explanation_rate",
    "mean_explanation_time_s",
    "lecture_rate",
    "mean_lecture_time_s",
    "review_rate",
    "adaptive_offer_rate",
]

ALL_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + EDNET_BEHAVIOR_FEATURE_COLUMNS

# Backward-compatible alias used by a few modules/tests.
FEATURE_COLUMNS = CORE_FEATURE_COLUMNS

PEDAGOGY_STRATEGIES = [
    "remedial_learning",
    "example_based_learning",
    "practice_heavy_learning",
    "step_by_step_guidance",
    "challenge_based_learning",
]


def feature_columns(feature_set: str) -> list[str]:
    name = str(feature_set).strip().lower()
    if name == "core":
        return list(CORE_FEATURE_COLUMNS)
    if name == "ednet_enriched":
        return list(ALL_FEATURE_COLUMNS)
    raise ValueError(f"Unknown feature_set={feature_set!r}; expected 'core' or 'ednet_enriched'.")
