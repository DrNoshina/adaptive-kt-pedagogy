from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier

from adaptive_kt.data.schema import PEDAGOGY_STRATEGIES
from adaptive_kt.utils.reproducibility import torch_device


def assign_pedagogy_strategy(frame: pd.DataFrame) -> pd.Series:
    """Transparent weak-supervision policy using proposal-defined learner-state variables."""
    required = {
        "topic_mastery",
        "recent_accuracy",
        "mean_attempt_count",
        "error_rate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot assign pedagogical strategies; missing {sorted(missing)}")

    mastery = frame["topic_mastery"].to_numpy(dtype=float)
    recent = frame["recent_accuracy"].to_numpy(dtype=float)
    attempts = frame["mean_attempt_count"].to_numpy(dtype=float)
    error_rate = frame["error_rate"].to_numpy(dtype=float)
    if "kt_probability" in frame.columns:
        kt_probability = frame["kt_probability"].to_numpy(dtype=float)
    else:
        kt_probability = 0.5 * mastery + 0.5 * recent

    # Transparent weak-supervision policy.  The KT prediction is explicitly part of the
    # learner-state signal so the recommendation stage is genuinely KT-conditioned.
    labels = np.full(len(frame), "practice_heavy_learning", dtype=object)
    high_state = (mastery >= 0.80) & (recent >= 0.80) & (kt_probability >= 0.75)
    low_state = (mastery < 0.35) & (recent < 0.50) & (kt_probability < 0.50)
    labels[high_state] = "challenge_based_learning"
    labels[((mastery < 0.55) | (kt_probability < 0.55)) & ~low_state] = "example_based_learning"
    labels[(error_rate >= 0.45) | (attempts >= 1.50)] = "step_by_step_guidance"
    labels[low_state] = "remedial_learning"
    return pd.Series(labels, index=frame.index, name="pedagogy_strategy")


class PedagogyRecommender:
    def __init__(self, config: dict, seed: int) -> None:
        self.encoder = LabelEncoder()
        self.encoder.fit(PEDAGOGY_STRATEGIES)
        params = {
            "n_estimators": int(config.get("n_estimators", 500)),
            "max_depth": int(config.get("max_depth", 6)),
            "learning_rate": float(config.get("learning_rate", 0.05)),
            "subsample": float(config.get("subsample", 0.9)),
            "colsample_bytree": float(config.get("colsample_bytree", 0.9)),
            "objective": "multi:softprob",
            "num_class": len(PEDAGOGY_STRATEGIES),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
            "early_stopping_rounds": int(config.get("early_stopping_rounds", 40)),
        }
        if torch_device().type == "cuda":
            params["device"] = "cuda"
        self.model = XGBClassifier(**params)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        x_validation: pd.DataFrame,
        y_validation: pd.Series,
        *,
        checkpoint_dir: str | Path | None = None,
        checkpoint_name: str = "pedagogy",
    ) -> "PedagogyRecommender":
        encoded = self.encoder.transform(y.astype(str))
        encoded_validation = self.encoder.transform(y_validation.astype(str))
        present = np.unique(encoded)
        if len(present) < 2:
            raise ValueError("Pedagogy training data must contain at least two strategies.")

        resume_model = None
        target_rounds = int(self.model.get_params().get("n_estimators", 500))
        completed_rounds = 0
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            candidates = list(checkpoint_dir.glob(f"{checkpoint_name}_*.ubj"))
            if candidates:
                def count_rounds(path: Path) -> int:
                    booster = xgb.Booster()
                    booster.load_model(path)
                    return int(booster.num_boosted_rounds())

                latest = max(candidates, key=count_rounds)
                completed_rounds = count_rounds(latest)
                resume_model = str(latest)
                print(
                    f"[{checkpoint_name}] resume checkpoint found: "
                    f"{completed_rounds}/{target_rounds} rounds -> {latest}",
                    flush=True,
                )

            if completed_rounds >= target_rounds and resume_model is not None:
                self.model.load_model(resume_model)
                print(f"[{checkpoint_name}] target rounds already present; no further fitting needed", flush=True)
                return self

            remaining = target_rounds - completed_rounds
            self.model.set_params(
                n_estimators=remaining,
                callbacks=[
                    xgb.callback.TrainingCheckPoint(
                        directory=checkpoint_dir,
                        name=checkpoint_name,
                        as_pickle=False,
                        interval=25,
                    )
                ],
            )
        else:
            remaining = target_rounds

        print(
            f"[{checkpoint_name}] pedagogy XGBoost training: target={target_rounds}, "
            f"completed={completed_rounds}, remaining={remaining}; log every 25 rounds",
            flush=True,
        )
        self.model.fit(
            x,
            encoded,
            eval_set=[(x_validation, encoded_validation)],
            verbose=25,
            xgb_model=resume_model,
        )
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(x)
        if raw.shape[1] == len(self.encoder.classes_):
            return raw
        full = np.zeros((len(x), len(self.encoder.classes_)), dtype=float)
        for local_index, class_index in enumerate(self.model.classes_):
            full[:, int(class_index)] = raw[:, local_index]
        return full

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.encoder.inverse_transform(np.argmax(self.predict_proba(x), axis=1))
