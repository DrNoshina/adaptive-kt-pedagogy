#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_kt.data.features import build_causal_features
from adaptive_kt.utils.tableio import write_table


def main() -> None:
    rng = np.random.default_rng(7)
    rows = []
    n_users = 80
    n_skills = 12
    for user in range(n_users):
        ability = rng.normal(0.0, 0.9)
        timestamp = 1_700_000_000_000 + user * 10_000_000
        explanation_count = 0
        lecture_count = 0
        review_count = 0
        adaptive_count = 0
        for step in range(45):
            skill = int(rng.integers(0, n_skills))
            difficulty = (skill - n_skills / 2) / n_skills
            p = 1.0 / (1.0 + np.exp(-(ability - difficulty + 0.025 * step)))
            correct = int(rng.random() < p)
            attempts = 1 + int((not correct) and rng.random() < 0.35)
            response_s = float(np.clip(rng.normal(28 - 5 * ability + 9 * (not correct), 7), 4, 120))
            # Snapshot prior engagement before the current target outcome can affect it.
            prior_explanation_count = explanation_count
            prior_lecture_count = lecture_count
            prior_review_count = review_count
            prior_adaptive_count = adaptive_count
            timestamp += int(response_s * 1000 + rng.integers(1000, 5000))
            rows.append(
                {
                    "user_id": f"u{user:04d}",
                    "timestamp": timestamp,
                    "item_id": f"q{skill:02d}_{step:03d}",
                    "skill_id": f"s{skill:02d}",
                    "correct": correct,
                    "response_time_ms": response_s * 1000,
                    "attempt_count": attempts,
                    "explanation_count_prior": prior_explanation_count,
                    "explanation_time_ms_prior": prior_explanation_count * 25_000,
                    "lecture_count_prior": prior_lecture_count,
                    "lecture_time_ms_prior": prior_lecture_count * 180_000,
                    "review_count_prior": prior_review_count,
                    "adaptive_offer_count_prior": prior_adaptive_count,
                }
            )
            if not correct and rng.random() < 0.15:
                explanation_count += 1
            if not correct and rng.random() < 0.08:
                lecture_count += 1
            if step > 8 and rng.random() < 0.05:
                review_count += 1
            if step > 8 and rng.random() < 0.04:
                adaptive_count += 1
            ability += 0.015 * correct - 0.003 * (not correct)

    interactions = pd.DataFrame(rows)
    features = build_causal_features(interactions, recent_window=5, min_history=3)
    root = Path("data/processed/synthetic")
    for name, frame in [("interactions", interactions), ("features", features)]:
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        write_table(frame, directory / "part-00000")
    print(f"Wrote {len(interactions)} interactions and {len(features)} feature rows to {root}")


if __name__ == "__main__":
    main()
