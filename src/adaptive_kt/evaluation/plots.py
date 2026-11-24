from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_model_comparison(metrics: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    order = metrics.sort_values("auc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(order["model"], order["auc"])
    ax.set_ylabel("AUC")
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def save_strategy_distribution(distribution: pd.Series, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    distribution.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Pedagogical strategy")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def save_shap_importance(importance: pd.DataFrame, output: Path, top_n: int = 12) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    view = importance.head(top_n).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(view["feature"], view["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
