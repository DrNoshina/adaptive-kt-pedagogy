from pathlib import Path

import pandas as pd

from adaptive_kt.data.ednet import parse_user_actions


def test_ednet_parser_uses_last_response(tmp_path: Path):
    user_file = tmp_path / "u1.csv"
    pd.DataFrame(
        [
            {"timestamp": 1000, "action_type": "enter", "item_id": "b1", "source": "sprint", "user_answer": "", "platform": "web"},
            {"timestamp": 2000, "action_type": "respond", "item_id": "q1", "source": "sprint", "user_answer": "a", "platform": "web"},
            {"timestamp": 2500, "action_type": "respond", "item_id": "q1", "source": "sprint", "user_answer": "b", "platform": "web"},
            {"timestamp": 3000, "action_type": "submit", "item_id": "b1", "source": "sprint", "user_answer": "", "platform": "web"},
        ]
    ).to_csv(user_file, index=False)
    questions = pd.DataFrame(
        [{"question_id": "q1", "correct_answer": "b", "skill_id": "10"}]
    ).set_index("question_id", drop=False)
    parsed = parse_user_actions(user_file, questions)
    assert len(parsed) == 1
    assert parsed.iloc[0]["correct"] == 1
    assert parsed.iloc[0]["attempt_count"] == 2
    assert parsed.iloc[0]["response_time_ms"] == 1000


def test_ednet_engagement_is_recorded_only_before_later_questions(tmp_path: Path):
    user_file = tmp_path / "u2.csv"
    pd.DataFrame(
        [
            {"timestamp": 1000, "action_type": "enter", "item_id": "b1", "source": "sprint", "user_answer": "", "platform": "web"},
            {"timestamp": 2000, "action_type": "respond", "item_id": "q1", "source": "sprint", "user_answer": "a", "platform": "web"},
            {"timestamp": 3000, "action_type": "submit", "item_id": "b1", "source": "sprint", "user_answer": "", "platform": "web"},
            {"timestamp": 4000, "action_type": "enter", "item_id": "e1", "source": "after_sprint", "user_answer": "", "platform": "web"},
            {"timestamp": 9000, "action_type": "quit", "item_id": "e1", "source": "after_sprint", "user_answer": "", "platform": "web"},
            {"timestamp": 10000, "action_type": "enter", "item_id": "b2", "source": "in_review", "user_answer": "", "platform": "web"},
            {"timestamp": 11000, "action_type": "respond", "item_id": "q2", "source": "in_review", "user_answer": "b", "platform": "web"},
            {"timestamp": 12000, "action_type": "submit", "item_id": "b2", "source": "in_review", "user_answer": "", "platform": "web"},
        ]
    ).to_csv(user_file, index=False)
    questions = pd.DataFrame(
        [
            {"question_id": "q1", "correct_answer": "a", "skill_id": "10"},
            {"question_id": "q2", "correct_answer": "b", "skill_id": "10"},
        ]
    ).set_index("question_id", drop=False)
    parsed = parse_user_actions(user_file, questions)
    first, second = parsed.iloc[0], parsed.iloc[1]
    assert first["explanation_count_prior"] == 0
    assert second["explanation_count_prior"] == 1
    assert second["explanation_time_ms_prior"] == 5000
    # Current review bundle is not counted as prior review history for its own target.
    assert second["review_count_prior"] == 0
