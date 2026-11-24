from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _as_classwise_array(values: object, n_classes: int | None = None) -> np.ndarray:
    if isinstance(values, list):
        return np.stack([np.asarray(v) for v in values], axis=-1)
    array = np.asarray(values)
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim == 3:
        if n_classes is not None and array.shape[-1] != n_classes and array.shape[0] == n_classes:
            return np.moveaxis(array, 0, -1)
        return array
    raise ValueError(f"Unsupported SHAP value shape: {array.shape}")


def tree_shap_importance(model: object, x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    values = _as_classwise_array(explainer.shap_values(x))
    mean_abs = np.mean(np.abs(values), axis=(0, 2))
    ranking = pd.DataFrame({"feature": x.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    return ranking.reset_index(drop=True), values


def explanation_quality(
    model: object,
    x: pd.DataFrame,
    top_features: int,
    seed: int,
) -> dict[str, float]:
    if len(x) == 0:
        return {"fidelity_probability_drop": float("nan"), "consistency_jaccard": float("nan")}
    explainer = shap.TreeExplainer(model)
    probabilities = model.predict_proba(x)
    predicted = np.argmax(probabilities, axis=1)
    values = _as_classwise_array(explainer.shap_values(x), n_classes=probabilities.shape[1])
    if values.shape[-1] == 1:
        selected = np.abs(values[:, :, 0])
    else:
        selected = np.abs(values[np.arange(len(x)), :, predicted])
    top_n = max(1, min(int(top_features), x.shape[1]))
    top_idx = np.argsort(-selected, axis=1)[:, :top_n]

    medians = x.median(numeric_only=True)
    perturbed = x.copy()
    for row_idx in range(len(perturbed)):
        for feature_idx in top_idx[row_idx]:
            column = x.columns[int(feature_idx)]
            perturbed.iat[row_idx, int(feature_idx)] = medians[column]
    perturbed_prob = model.predict_proba(perturbed)
    original_selected = probabilities[np.arange(len(x)), predicted]
    perturbed_selected = perturbed_prob[np.arange(len(x)), predicted]
    fidelity = float(np.mean(np.clip(original_selected - perturbed_selected, 0.0, None)))

    rng = np.random.default_rng(seed)
    std = x.std(numeric_only=True).replace(0, 1.0).fillna(1.0)
    noise = rng.normal(0.0, 0.01, size=x.shape) * std.to_numpy(dtype=float)[None, :]
    jitter_array = x.to_numpy(dtype=float) + noise
    lower = x.min(numeric_only=True).to_numpy(dtype=float)
    upper = x.max(numeric_only=True).to_numpy(dtype=float)
    jitter_array = np.clip(jitter_array, lower[None, :], upper[None, :])
    jittered = pd.DataFrame(jitter_array, columns=x.columns, index=x.index)
    jitter_values = _as_classwise_array(explainer.shap_values(jittered), n_classes=probabilities.shape[1])
    jitter_pred = np.argmax(model.predict_proba(jittered), axis=1)
    if jitter_values.shape[-1] == 1:
        jitter_selected = np.abs(jitter_values[:, :, 0])
    else:
        jitter_selected = np.abs(jitter_values[np.arange(len(x)), :, jitter_pred])
    jitter_top = np.argsort(-jitter_selected, axis=1)[:, :top_n]
    jaccards = []
    for first, second in zip(top_idx, jitter_top):
        a, b = set(first.tolist()), set(second.tolist())
        jaccards.append(len(a & b) / len(a | b))
    return {
        "fidelity_probability_drop": fidelity,
        "consistency_jaccard": float(np.mean(jaccards)),
    }


def local_shap_reasons(
    model: object,
    x: pd.DataFrame,
    class_names: list[str] | tuple[str, ...],
    top_features: int = 3,
) -> pd.DataFrame:
    """Return per-sample SHAP reasons for the predicted class.

    This is intentionally compact and reviewer-friendly: each row contains the predicted class,
    probability, and the top contributing features with their observed values and SHAP impacts.
    """
    if len(x) == 0:
        return pd.DataFrame(columns=["row_index", "predicted_class", "predicted_probability", "reason"])
    explainer = shap.TreeExplainer(model)
    probabilities = model.predict_proba(x)
    predicted = np.argmax(probabilities, axis=1)
    values = _as_classwise_array(explainer.shap_values(x), n_classes=probabilities.shape[1])
    if values.shape[-1] == 1:
        selected = values[:, :, 0]
    else:
        selected = values[np.arange(len(x)), :, predicted]
    top_n = max(1, min(int(top_features), x.shape[1]))
    top_idx = np.argsort(-np.abs(selected), axis=1)[:, :top_n]
    rows = []
    class_names = list(class_names)
    for local_row, source_index in enumerate(x.index):
        details = []
        for feature_idx in top_idx[local_row]:
            feature_idx = int(feature_idx)
            feature = str(x.columns[feature_idx])
            value = float(x.iloc[local_row, feature_idx])
            impact = float(selected[local_row, feature_idx])
            details.append(f"{feature}={value:.4g} (SHAP {impact:+.4g})")
        class_idx = int(predicted[local_row])
        rows.append(
            {
                "row_index": source_index,
                "predicted_class": class_names[class_idx] if class_idx < len(class_names) else str(class_idx),
                "predicted_probability": float(probabilities[local_row, class_idx]),
                "reason": "; ".join(details),
            }
        )
    return pd.DataFrame(rows)
