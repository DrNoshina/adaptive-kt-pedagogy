from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from adaptive_kt.data.features import build_causal_features
from adaptive_kt.data.schema import INTERACTION_COLUMNS


@dataclass
class BundleState:
    bundle_id: str
    enter_timestamp: int
    source: str | None
    platform: str | None
    explanation_count_prior: int = 0
    explanation_time_ms_prior: int = 0
    lecture_count_prior: int = 0
    lecture_time_ms_prior: int = 0
    review_count_prior: int = 0
    adaptive_offer_count_prior: int = 0
    responses: dict[str, list[tuple[int, str]]] = field(default_factory=dict)


def find_first(root: Path, names: Iterable[str]) -> Path:
    lowered = {name.lower() for name in names}
    for path in root.rglob("*.csv"):
        if path.name.lower() in lowered:
            return path
    raise FileNotFoundError(f"Could not find any of {sorted(lowered)} below {root}")


def find_kt3_user_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*.csv"):
        name = path.name.lower()
        if name in {"questions.csv", "lectures.csv", "payments.csv", "coupons.csv"}:
            continue
        if path.stem.startswith("u") or path.stem.isdigit():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No EdNet-KT3 user CSV files found below {root}. Extract the official KT3 archive first."
        )
    return sorted(candidates)


def load_questions(root: Path) -> pd.DataFrame:
    questions_path = find_first(root, ["questions.csv"])
    questions = pd.read_csv(questions_path, dtype=str)
    required = {"question_id", "correct_answer", "tags"}
    missing = required.difference(questions.columns)
    if missing:
        raise ValueError(f"{questions_path} is missing columns: {sorted(missing)}")
    questions = questions.copy()
    questions["question_id"] = questions["question_id"].astype(str)
    questions["correct_answer"] = questions["correct_answer"].astype(str).str.lower()
    # Preserve the complete EdNet knowledge-component/tag signature.
    # The previous implementation kept only the first tag via .str[0],
    # discarding additional KCs for multi-tag questions.
    def normalize_tag_signature(value: object) -> str:
        raw = "" if pd.isna(value) else str(value)
        parts = []
        seen = set()
        for token in raw.split(";"):
            token = token.strip()
            if not token or token.lower() == "nan":
                continue
            if token not in seen:
                seen.add(token)
                parts.append(token)
        return ";".join(parts) if parts else "unknown"

    questions["skill_id"] = questions["tags"].map(normalize_tag_signature)
    return questions.set_index("question_id", drop=False)


def _safe_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _item_prefix(item_id: str) -> str:
    return item_id[:1].lower() if item_id else ""


def parse_user_actions(user_file: Path, questions: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(user_file, dtype=str)
    required = {"timestamp", "action_type", "item_id"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{user_file} is missing KT3 columns: {sorted(missing)}")

    raw = raw.copy()
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "action_type", "item_id"])
    raw["timestamp"] = raw["timestamp"].astype("int64")
    raw = raw.sort_values("timestamp", kind="mergesort")

    user_id = user_file.stem
    current_bundle: BundleState | None = None
    explanation_enter: dict[str, int] = {}
    lecture_enter: dict[str, int] = {}
    explanation_count = 0
    lecture_count = 0
    explanation_time_ms = 0
    lecture_time_ms = 0
    review_count = 0
    adaptive_offer_count = 0
    emitted: list[dict] = []

    for record in raw.to_dict(orient="records"):
        timestamp = _safe_int(record.get("timestamp"))
        action = str(record.get("action_type") or "").lower()
        item_id = str(record.get("item_id") or "")
        prefix = _item_prefix(item_id)
        source = str(record.get("source") or "") or None
        platform = str(record.get("platform") or "") or None

        if action == "enter" and prefix == "e":
            explanation_enter[item_id] = timestamp
            continue
        if action == "quit" and prefix == "e":
            start = explanation_enter.pop(item_id, None)
            if start is not None and timestamp >= start:
                explanation_time_ms += timestamp - start
                explanation_count += 1
            continue
        if action == "enter" and prefix == "l":
            lecture_enter[item_id] = timestamp
            continue
        if action == "quit" and prefix == "l":
            start = lecture_enter.pop(item_id, None)
            if start is not None and timestamp >= start:
                lecture_time_ms += timestamp - start
                lecture_count += 1
            continue

        if action == "enter" and prefix == "b":
            # Freeze engagement state at bundle entry.  Explanation/lecture actions that
            # occur later inside this bundle must not become "prior" features for the
            # questions whose outcomes are being predicted in the same bundle.
            current_bundle = BundleState(
                item_id,
                timestamp,
                source,
                platform,
                explanation_count_prior=explanation_count,
                explanation_time_ms_prior=explanation_time_ms,
                lecture_count_prior=lecture_count,
                lecture_time_ms_prior=lecture_time_ms,
                review_count_prior=review_count,
                adaptive_offer_count_prior=adaptive_offer_count,
            )
            continue

        if action == "respond" and prefix == "q" and current_bundle is not None:
            answer = str(record.get("user_answer") or "").lower()
            if answer:
                current_bundle.responses.setdefault(item_id, []).append((timestamp, answer))
            continue

        if action == "submit" and prefix == "b" and current_bundle is not None:
            if item_id != current_bundle.bundle_id:
                current_bundle = None
                continue

            ordered = sorted(
                current_bundle.responses.items(),
                key=lambda pair: pair[1][0][0] if pair[1] else timestamp,
            )
            previous_timestamp = current_bundle.enter_timestamp
            for question_id, responses in ordered:
                if question_id not in questions.index or not responses:
                    continue
                first_ts = responses[0][0]
                last_answer = responses[-1][1]
                metadata = questions.loc[question_id]
                correct_answer = str(metadata["correct_answer"]).lower()
                response_time_ms = max(0, first_ts - previous_timestamp)
                previous_timestamp = first_ts
                emitted.append(
                    {
                        "user_id": user_id,
                        "timestamp": first_ts,
                        "item_id": question_id,
                        "skill_id": str(metadata["skill_id"]),
                        "correct": int(last_answer == correct_answer),
                        "response_time_ms": response_time_ms,
                        "attempt_count": len(responses),
                        "explanation_count_prior": current_bundle.explanation_count_prior,
                        "explanation_time_ms_prior": current_bundle.explanation_time_ms_prior,
                        "lecture_count_prior": current_bundle.lecture_count_prior,
                        "lecture_time_ms_prior": current_bundle.lecture_time_ms_prior,
                        "review_count_prior": current_bundle.review_count_prior,
                        "adaptive_offer_count_prior": current_bundle.adaptive_offer_count_prior,
                        "source": current_bundle.source or "",
                        "platform": current_bundle.platform or "",
                    }
                )
            completed_source = current_bundle.source
            if completed_source == "in_review":
                review_count += 1
            if completed_source == "adaptive_offer":
                adaptive_offer_count += 1
            current_bundle = None

    if not emitted:
        return pd.DataFrame(columns=INTERACTION_COLUMNS + ["source", "platform"])
    return pd.DataFrame(emitted)


def process_user_file(
    user_file: Path,
    questions: pd.DataFrame,
    recent_window: int,
    min_history: int,
    max_response_time_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = parse_user_actions(user_file, questions)
    features = build_causal_features(
        interactions,
        recent_window=recent_window,
        min_history=min_history,
        max_response_time_seconds=max_response_time_seconds,
    )
    return interactions, features
