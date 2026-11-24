from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from adaptive_kt.utils.reproducibility import torch_device


@dataclass
class BaselineBundle:
    name: str
    model: object


def make_baselines(config: dict, seed: int) -> list[BaselineBundle]:
    lr_cfg = config["logistic_regression"]
    rf_cfg = config["random_forest"]
    xgb_cfg = config["xgboost"]

    lr = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=int(lr_cfg.get("max_iter", 1000)),
                    C=float(lr_cfg.get("C", 1.0)),
                    random_state=seed,
                    verbose=1,
                ),
            ),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 400)),
        max_depth=rf_cfg.get("max_depth"),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        random_state=seed,
        class_weight="balanced_subsample",
    )
    xgb_params = {
        "n_estimators": int(xgb_cfg.get("n_estimators", 800)),
        "max_depth": int(xgb_cfg.get("max_depth", 8)),
        "learning_rate": float(xgb_cfg.get("learning_rate", 0.05)),
        "subsample": float(xgb_cfg.get("subsample", 0.85)),
        "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.85)),
        "reg_lambda": float(xgb_cfg.get("reg_lambda", 1.0)),
        "min_child_weight": float(xgb_cfg.get("min_child_weight", 1.0)),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": int(xgb_cfg.get("early_stopping_rounds", 50)),
    }
    if torch_device().type == "cuda":
        xgb_params["device"] = "cuda"
    xgb = XGBClassifier(**xgb_params)

    return [
        BaselineBundle("logistic_regression", lr),
        BaselineBundle("random_forest", rf),
        BaselineBundle("xgboost", xgb),
    ]


def deterministic_row_sample(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not max_rows or len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed).sort_index()


def predict_probability(model: object, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model {type(model).__name__} does not expose predict_proba().")
    probabilities = model.predict_proba(x)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected binary-class probabilities with two columns.")
    return probabilities[:, 1]
