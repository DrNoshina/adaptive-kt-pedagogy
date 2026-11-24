from __future__ import annotations

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch.utils.data import DataLoader

from adaptive_kt.data.schema import (
    CORE_FEATURE_COLUMNS,
    feature_columns,
)
from adaptive_kt.data.splits import assert_user_disjoint, split_assignment_frame, split_by_user
from adaptive_kt.evaluation.metrics import binary_metrics, recommendation_metrics, select_f1_threshold
from adaptive_kt.evaluation.plots import save_model_comparison, save_shap_importance, save_strategy_distribution
from adaptive_kt.explain.shap_tools import explanation_quality, local_shap_reasons, tree_shap_importance
from adaptive_kt.models.baselines import deterministic_row_sample, make_baselines, predict_probability
from adaptive_kt.models.dkt import DKT
from adaptive_kt.models.sakt import SAKT
from adaptive_kt.models.saint_plus import SAINTPlus
from adaptive_kt.models.akt import AKT
from adaptive_kt.models.behavior_aware_kt import BehaviorAwareKT
from adaptive_kt.models.pedagogy import PedagogyRecommender, assign_pedagogy_strategy
from adaptive_kt.models.sequence import (
    BEHAVIOR_DIM,
    BEHAVIOR_FEATURE_INDEX,
    KTSequenceDataset,
    SequenceVocabulary,
    collate_sequences,
)
from adaptive_kt.models.transformer_kt import TransformerKT
from adaptive_kt.training.kt_trainer import predict_kt, train_kt_model
from adaptive_kt.utils.environment import environment_snapshot
from adaptive_kt.utils.io import ensure_dir, write_json
from adaptive_kt.utils.reproducibility import seed_everything, torch_device
from adaptive_kt.utils.tableio import read_collection


def _read_collection(path: Path) -> pd.DataFrame:
    return read_collection(path)


def _split_with_user_sets(
    frame: pd.DataFrame, split_cfg: dict, seed: int
) -> tuple[dict[str, pd.DataFrame], dict[str, set[str]]]:
    split = split_by_user(
        frame,
        train_size=float(split_cfg["train"]),
        validation_size=float(split_cfg["validation"]),
        test_size=float(split_cfg["test"]),
        seed=seed,
    )
    assert_user_disjoint(split)
    user_sets = {name: set(part["user_id"].astype(str).unique()) for name, part in split.items()}
    return split, user_sets


def _filter_users(frame: pd.DataFrame, users: set[str]) -> pd.DataFrame:
    return frame[frame["user_id"].astype(str).isin(users)].copy()


def _split_from_saved_assignments(
    frame: pd.DataFrame, assignment_path: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, set[str]]]:
    assignments = pd.read_csv(assignment_path, dtype={"user_id": str, "split": str})
    required = {"user_id", "split"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Saved split assignment is missing columns: {sorted(missing)}")
    if assignments["user_id"].duplicated().any():
        raise AssertionError("Saved split assignment contains duplicate learners.")
    valid_names = {"train", "validation", "test"}
    if set(assignments["split"].unique()) != valid_names:
        raise ValueError("Saved split assignment must contain train, validation, and test.")

    frame_users = set(frame["user_id"].astype(str).unique())
    assigned_users = set(assignments["user_id"].astype(str))
    if frame_users != assigned_users:
        missing_users = len(frame_users - assigned_users)
        extra_users = len(assigned_users - frame_users)
        raise AssertionError(
            f"Saved split does not match prepared data: missing={missing_users}, extra={extra_users}."
        )

    user_sets = {
        name: set(assignments.loc[assignments["split"].eq(name), "user_id"].astype(str))
        for name in ("train", "validation", "test")
    }
    split = {name: _filter_users(frame, users) for name, users in user_sets.items()}
    assert_user_disjoint(split)
    return split, user_sets


def _stage(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def _latest_xgb_checkpoint(directory: Path, name: str) -> Path | None:
    if not directory.exists():
        return None
    candidates = list(directory.glob(f"{name}_*.ubj"))
    if not candidates:
        return None

    def rounds(path: Path) -> int:
        booster = xgb.Booster()
        booster.load_model(path)
        return int(booster.num_boosted_rounds())

    return max(candidates, key=rounds)


def _fit_xgboost_with_resume(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    checkpoint_dir: Path,
    checkpoint_name: str,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_rounds = int(model.get_params().get("n_estimators", 100))
    latest = _latest_xgb_checkpoint(checkpoint_dir, checkpoint_name)
    resume_model = None
    completed_rounds = 0

    if latest is not None:
        booster = xgb.Booster()
        booster.load_model(latest)
        completed_rounds = int(booster.num_boosted_rounds())
        resume_model = str(latest)
        print(
            f"[{checkpoint_name}] resuming XGBoost from {completed_rounds}/{target_rounds} boosted rounds: {latest}",
            flush=True,
        )

    if completed_rounds >= target_rounds and latest is not None:
        model.load_model(latest)
        print(f"[{checkpoint_name}] target rounds already present in checkpoint; no further fitting needed", flush=True)
        return model

    remaining = target_rounds - completed_rounds
    model.set_params(
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
    print(
        f"[{checkpoint_name}] XGBoost live training: target={target_rounds}, "
        f"completed={completed_rounds}, remaining={remaining}; validation log every 25 rounds",
        flush=True,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=25,
        xgb_model=resume_model,
    )
    return model


def _fit_random_forest_with_resume(model, x_train, y_train, checkpoint_path: Path, chunk_size: int = 25):
    target_trees = int(model.get_params()["n_estimators"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        model = joblib.load(checkpoint_path)
        current_trees = len(model.estimators_)
        print(
            f"[random_forest] resume checkpoint found: {current_trees}/{target_trees} trees",
            flush=True,
        )
    else:
        current_trees = 0
        model.set_params(warm_start=True, n_estimators=0)

    while current_trees < target_trees:
        next_trees = min(target_trees, current_trees + chunk_size)
        model.set_params(warm_start=True, n_estimators=next_trees)
        print(f"[random_forest] training trees {current_trees + 1}-{next_trees}/{target_trees}", flush=True)
        model.fit(x_train, y_train)
        current_trees = len(model.estimators_)
        joblib.dump(model, checkpoint_path)
        print(f"[random_forest] checkpoint saved: {current_trees}/{target_trees} trees -> {checkpoint_path}", flush=True)

    return model


def _make_loaders(interactions: dict[str, pd.DataFrame], config: dict, seed: int):
    seq_cfg = config["sequence"]

    # Fit vocabularies ONLY on training learners.
    skill_vocab = SequenceVocabulary.fit(
        interactions["train"]["skill_id"]
    )

    item_vocab = SequenceVocabulary.fit(
        interactions["train"]["item_id"]
    )

    datasets = {
        name: KTSequenceDataset(
            part,
            skill_vocab=skill_vocab,
            item_vocab=item_vocab,
            max_len=int(seq_cfg["max_len"]),
            min_history=int(config["dataset"].get("min_history", 1)),
            split_name=name,
        )
        for name, part in interactions.items()
    }
    generator = torch.Generator().manual_seed(seed)
    workers = int(seq_cfg.get("num_workers", 0))
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(seq_cfg["batch_size"]),
            shuffle=True,
            collate_fn=collate_sequences,
            generator=generator,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "train_eval": DataLoader(
            datasets["train"],
            batch_size=int(seq_cfg["batch_size"]),
            shuffle=False,
            collate_fn=collate_sequences,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=int(seq_cfg["batch_size"]),
            shuffle=False,
            collate_fn=collate_sequences,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(seq_cfg["batch_size"]),
            shuffle=False,
            collate_fn=collate_sequences,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    for name, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(f"No valid KT sequences were created for the {name} split.")
    return skill_vocab, item_vocab, loaders


def _resized_sequence_loaders(
    loaders: dict[str, DataLoader],
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader]:
    """Reuse identical sequence datasets with a smaller per-model batch size."""
    workers = int(loaders["train"].num_workers)
    pin_memory = bool(loaders["train"].pin_memory)
    generator = torch.Generator().manual_seed(seed)

    return {
        "train": DataLoader(
            loaders["train"].dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_sequences,
            generator=generator,
            num_workers=workers,
            pin_memory=pin_memory,
        ),
        "validation": DataLoader(
            loaders["validation"].dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sequences,
            num_workers=workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            loaders["test"].dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sequences,
            num_workers=workers,
            pin_memory=pin_memory,
        ),
    }


def _train_sequence_models(
    interactions: dict[str, pd.DataFrame],
    config: dict,
    output: Path,
    seed: int,
) -> tuple[
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    torch.nn.Module,
    SequenceVocabulary,
    SequenceVocabulary,
    dict[str, DataLoader],
]:
    seq_cfg = config["sequence"]
    device = torch_device()
    skill_vocab, item_vocab, loaders = _make_loaders(interactions, config, seed)
    test_results: list[dict[str, float | str]] = []
    validation_results: list[dict[str, float | str]] = []
    proposed_model: torch.nn.Module | None = None

    model_specs = [
        (
            "dkt_lstm",
            DKT(
                num_skills=skill_vocab.size,
                num_items=item_vocab.size,
                embedding_dim=int(seq_cfg["embedding_dim"]),
                hidden_dim=int(seq_cfg["hidden_dim"]),
                dropout=float(seq_cfg["dropout"]),
            ),
        ),
        (
            "transformer_kt",
            TransformerKT(
                num_skills=skill_vocab.size,
                num_items=item_vocab.size,
                max_len=int(seq_cfg["max_len"]) + 1,
                embedding_dim=int(seq_cfg["embedding_dim"]),
                num_heads=int(seq_cfg["transformer_heads"]),
                num_layers=int(seq_cfg["transformer_layers"]),
                dropout=float(seq_cfg["dropout"]),
            ),
        ),
        (
            "sakt",
            SAKT(
                num_skills=skill_vocab.size,
                num_items=item_vocab.size,
                max_len=int(seq_cfg["max_len"]) + 1,
                embedding_dim=int(seq_cfg["embedding_dim"]),
                num_heads=int(seq_cfg["transformer_heads"]),
                dropout=float(seq_cfg["dropout"]),
            ),
        ),
        (
            "saint_plus",
            SAINTPlus(
                num_skills=skill_vocab.size,
                num_items=item_vocab.size,
                max_len=int(seq_cfg["max_len"]) + 1,
                embedding_dim=int(seq_cfg["embedding_dim"]),
                num_heads=int(seq_cfg["transformer_heads"]),
                num_layers=int(seq_cfg["transformer_layers"]),
                dropout=float(seq_cfg["dropout"]),
            ),
        ),
        (
            "akt",
            AKT(
                num_skills=skill_vocab.size,
                num_items=item_vocab.size,
                embedding_dim=int(seq_cfg["embedding_dim"]),
                num_heads=int(seq_cfg["transformer_heads"]),
                num_layers=int(seq_cfg["transformer_layers"]),
                dropout=float(seq_cfg["dropout"]),
            ),
        ),
        (
            "proposed_behavior_kt",
            BehaviorAwareKT(
                num_skills=skill_vocab.size,
                num_items=item_vocab.size,
                max_len=int(seq_cfg["max_len"]) + 1,
                behavior_dim=BEHAVIOR_DIM,
                embedding_dim=int(seq_cfg["embedding_dim"]),
                num_heads=int(seq_cfg["transformer_heads"]),
                num_layers=int(seq_cfg["transformer_layers"]),
                dropout=float(seq_cfg["dropout"]),
            ),
        ),
    ]

    checkpoint_dir = ensure_dir(output / "checkpoints")
    history_dir = ensure_dir(output / "training_history")
    for name, model in model_specs:
        _stage(f"Sequence model: {name}")

        active_loaders = loaders
        if name == "akt":
            akt_batch_size = int(
                seq_cfg.get("akt_batch_size", seq_cfg["batch_size"])
            )
            if akt_batch_size != int(seq_cfg["batch_size"]):
                print(
                    f"[akt] using memory-safe batch_size={akt_batch_size} "
                    f"(default={seq_cfg['batch_size']})",
                    flush=True,
                )
                active_loaders = _resized_sequence_loaders(
                    loaders,
                    akt_batch_size,
                    seed,
                )

        restart_checkpoint = checkpoint_dir / f"{name}.resume.pt"
        history_path = history_dir / f"{name}.csv"

        trained = train_kt_model(
            model=model,
            train_loader=active_loaders["train"],
            validation_loader=active_loaders["validation"],
            device=device,
            epochs=int(seq_cfg["epochs"]),
            learning_rate=float(seq_cfg["learning_rate"]),
            weight_decay=float(seq_cfg["weight_decay"]),
            patience=int(seq_cfg["patience"]),
            checkpoint_path=restart_checkpoint,
            history_path=history_path,
            model_name=name,
            seed=seed,
        )
        val_true, val_prob = predict_kt(trained.model, active_loaders["validation"], device)
        threshold = select_f1_threshold(val_true, val_prob)
        val_metrics = binary_metrics(val_true, val_prob, threshold=threshold)
        test_true, test_prob = predict_kt(trained.model, active_loaders["test"], device)
        test_metrics = binary_metrics(test_true, test_prob, threshold=threshold)
        validation_results.append({"model": name, **val_metrics, "best_epoch": trained.best_epoch})
        test_results.append({"model": name, **test_metrics, "best_epoch": trained.best_epoch})
        torch.save(
            {
                "state_dict": trained.model.state_dict(),
                "skill_vocabulary": skill_vocab.mapping,
                "item_vocabulary": item_vocab.mapping,
                "model_name": name,
                "config": seq_cfg,
                "decision_threshold": threshold,
            },
            checkpoint_dir / f"{name}.pt",
        )
        pd.DataFrame(trained.history).to_csv(history_dir / f"{name}.csv", index=False)
        if name == "proposed_behavior_kt":
            proposed_model = trained.model
    if proposed_model is None:
        raise RuntimeError("Proposed BehaviorAwareKT model was not trained.")
    return test_results, validation_results, proposed_model, skill_vocab, item_vocab, loaders


def _attach_kt_probability(
    frame: pd.DataFrame,
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    """Align proposed-KT probabilities with causal feature rows.

    The sequence loader and feature builder both use the same stable user/timestamp/item ordering.
    A hard target-label assertion prevents silent misalignment.
    """
    y_true, y_prob = predict_kt(model, loader, device)
    ordered = frame.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
    if len(ordered) != len(y_prob):
        raise AssertionError(
            f"KT/feature alignment failed: feature_rows={len(ordered):,}, kt_targets={len(y_prob):,}."
        )
    expected = ordered["correct"].to_numpy(dtype=int)
    if not np.array_equal(expected, y_true.astype(int)):
        mismatch = int(np.sum(expected != y_true.astype(int)))
        raise AssertionError(f"KT/feature target alignment mismatch in {mismatch:,} rows.")
    ordered["kt_probability"] = y_prob.astype("float32")
    return ordered


def _fit_binary_model(
    bundle,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    checkpoint_dir: Path | None = None,
    checkpoint_name: str | None = None,
):
    checkpoint_dir = checkpoint_dir or Path("results/restart_checkpoints")
    checkpoint_name = checkpoint_name or bundle.name

    if bundle.name == "xgboost":
        return _fit_xgboost_with_resume(
            bundle.model,
            x_train,
            y_train,
            x_val,
            y_val,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
        )
    if bundle.name == "random_forest":
        return _fit_random_forest_with_resume(
            bundle.model,
            x_train,
            y_train,
            checkpoint_path=checkpoint_dir / f"{checkpoint_name}.partial.joblib",
        )

    print(f"[{bundle.name}] fitting model...", flush=True)
    bundle.model.fit(x_train, y_train)
    print(f"[{bundle.name}] fitting complete", flush=True)
    return bundle.model



def _run_proposed_ablations(
    config: dict,
    output: Path,
    seed: int,
    skill_vocab: SequenceVocabulary,
    item_vocab: SequenceVocabulary,
    loaders: dict[str, DataLoader],
    sequence_test_results: list[dict[str, float | str]],
) -> pd.DataFrame:
    """Required proposal ablations on the proposed KT model itself.

    The former pipeline ablated an XGBoost baseline, which did not test the proposed model.
    This routine keeps the architecture/split fixed and masks exactly one learner-state input.
    """
    seq_cfg = config["sequence"]
    required = {
        "full_model": (),
        "without_mastery_score": (BEHAVIOR_FEATURE_INDEX["topic_mastery"],),
        "without_response_time": (BEHAVIOR_FEATURE_INDEX["mean_response_time"],),
        "without_recent_accuracy": (BEHAVIOR_FEATURE_INDEX["recent_accuracy"],),
    }
    result_path = output / "ablation_results.csv"
    rows: list[dict] = []
    completed: set[str] = set()
    if result_path.exists():
        previous = pd.read_csv(result_path)
        if "model_family" in previous.columns and set(previous["model_family"].astype(str)) == {"proposed_behavior_kt"}:
            rows = previous.to_dict("records")
            completed = set(previous["ablation"].astype(str))

    if "full_model" not in completed:
        full = next(row for row in sequence_test_results if row["model"] == "proposed_behavior_kt")
        rows.append(
            {
                "ablation": "full_model",
                "model_family": "proposed_behavior_kt",
                "removed_features": "",
                **{k: v for k, v in full.items() if k != "model"},
            }
        )
        pd.DataFrame(rows).to_csv(result_path, index=False)
        completed.add("full_model")

    device = torch_device()
    checkpoint_dir = ensure_dir(output / "checkpoints" / "ablations")
    history_dir = ensure_dir(output / "training_history" / "ablations")

    for name, mask_indices in required.items():
        if name == "full_model" or name in completed:
            continue
        _stage(f"Proposed-model ablation: {name}")
        model = BehaviorAwareKT(
            num_skills=skill_vocab.size,
            num_items=item_vocab.size,
            max_len=int(seq_cfg["max_len"]) + 1,
            behavior_dim=BEHAVIOR_DIM,
            embedding_dim=int(seq_cfg["embedding_dim"]),
            num_heads=int(seq_cfg["transformer_heads"]),
            num_layers=int(seq_cfg["transformer_layers"]),
            dropout=float(seq_cfg["dropout"]),
            ablate_behavior_indices=tuple(mask_indices),
        )
        trained = train_kt_model(
            model=model,
            train_loader=loaders["train"],
            validation_loader=loaders["validation"],
            device=device,
            epochs=int(seq_cfg["epochs"]),
            learning_rate=float(seq_cfg["learning_rate"]),
            weight_decay=float(seq_cfg["weight_decay"]),
            patience=int(seq_cfg["patience"]),
            checkpoint_path=checkpoint_dir / f"{name}.resume.pt",
            history_path=history_dir / f"{name}.csv",
            model_name=f"proposed_{name}",
            seed=seed,
        )
        val_true, val_prob = predict_kt(trained.model, loaders["validation"], device)
        threshold = select_f1_threshold(val_true, val_prob)
        test_true, test_prob = predict_kt(trained.model, loaders["test"], device)
        metrics = binary_metrics(test_true, test_prob, threshold=threshold)
        rows.append(
            {
                "ablation": name,
                "model_family": "proposed_behavior_kt",
                "removed_features": ";".join(
                    key for key, idx in BEHAVIOR_FEATURE_INDEX.items() if idx in mask_indices
                ),
                "best_epoch": trained.best_epoch,
                **metrics,
            }
        )
        pd.DataFrame(rows).to_csv(result_path, index=False)
        del trained, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def run_experiment(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    dataset_cfg = config["dataset"]
    selected_features = feature_columns(dataset_cfg.get("feature_set", "core"))
    processed_root = Path(dataset_cfg["processed_root"])
    output = ensure_dir(config["output"]["root"])
    write_json(output / "environment.json", environment_snapshot())
    write_json(output / "resolved_config.json", config)
    write_json(output / "feature_columns.json", {"feature_set": dataset_cfg.get("feature_set", "core"), "columns": selected_features})

    _stage("1/8 Loading prepared interaction and feature tables")
    interactions = _read_collection(processed_root / "interactions")
    features = _read_collection(processed_root / "features")
    print(f"Loaded interactions={len(interactions):,}, feature_rows={len(features):,}", flush=True)
    if interactions.empty or features.empty:
        raise ValueError("Prepared interactions/features are empty.")

    _stage("2/8 Creating leakage-safe learner-disjoint train/validation/test split")
    assignment_path = output / "split_assignments.csv"
    if assignment_path.exists():
        interaction_split, user_sets = _split_from_saved_assignments(interactions, assignment_path)
        print(f"Reusing exact saved learner split -> {assignment_path}", flush=True)
    else:
        interaction_split, user_sets = _split_with_user_sets(interactions, config["split"], seed)
        split_assignment_frame(interaction_split).to_csv(assignment_path, index=False)
        print(f"Saved learner split -> {assignment_path}", flush=True)
    feature_split = {name: _filter_users(features, users) for name, users in user_sets.items()}
    if any(part.empty for part in feature_split.values()):
        raise ValueError("At least one feature split is empty after applying the learner-level split.")

    # Hard leakage guard: learner sets must be mutually exclusive for every model family.
    assert_user_disjoint(interaction_split)
    split_summary = {
        name: {
            "users": int(interaction_split[name]["user_id"].nunique()),
            "interactions": int(len(interaction_split[name])),
            "feature_rows": int(len(feature_split[name])),
            "positive_rate": float(feature_split[name]["correct"].mean()),
        }
        for name in ("train", "validation", "test")
    }
    write_json(output / "split_summary.json", split_summary)
    print(pd.DataFrame(split_summary).T.to_string(), flush=True)

    _stage("3/8 Training baseline models: Logistic Regression, Random Forest, XGBoost")
    baseline_cfg = config["baseline"]
    train_features = deterministic_row_sample(feature_split["train"], baseline_cfg.get("max_train_rows"), seed)
    val_features = deterministic_row_sample(feature_split["validation"], baseline_cfg.get("max_validation_rows"), seed + 1)
    x_train = train_features[selected_features]
    y_train = train_features["correct"].astype(int)
    x_val = val_features[selected_features]
    y_val = val_features["correct"].astype(int)
    x_test = feature_split["test"][selected_features]
    y_test = feature_split["test"]["correct"].astype(int)

    baseline_test_results: list[dict[str, float | str]] = []
    baseline_validation_results: list[dict[str, float | str]] = []
    model_dir = ensure_dir(output / "models")
    trained_baselines: dict[str, object] = {}
    baseline_bundles = make_baselines(baseline_cfg, seed)
    for index, bundle in enumerate(baseline_bundles, start=1):
        _stage(f"Baseline {index}/{len(baseline_bundles)}: {bundle.name}")
        final_model_path = model_dir / f"{bundle.name}.joblib"
        if final_model_path.exists():
            payload = joblib.load(final_model_path)
            model = payload["model"]
            threshold = float(payload["decision_threshold"])
            print(f"[{bundle.name}] completed model found; skipping fit -> {final_model_path}", flush=True)
        else:
            model = _fit_binary_model(
                bundle,
                x_train,
                y_train,
                x_val,
                y_val,
                checkpoint_dir=output / "restart_checkpoints" / bundle.name,
                checkpoint_name=bundle.name,
            )
            val_prob_for_threshold = predict_probability(model, x_val)
            threshold = select_f1_threshold(y_val, val_prob_for_threshold)
            joblib.dump(
                {"model": model, "decision_threshold": threshold, "features": selected_features},
                final_model_path,
            )
            print(f"[{bundle.name}] final model saved -> {final_model_path}", flush=True)

        val_prob = predict_probability(model, x_val)
        val_metrics = binary_metrics(y_val, val_prob, threshold=threshold)
        test_prob = predict_probability(model, x_test)
        test_metrics = binary_metrics(y_test, test_prob, threshold=threshold)
        baseline_validation_results.append({"model": bundle.name, **val_metrics})
        baseline_test_results.append({"model": bundle.name, **test_metrics})
        trained_baselines[bundle.name] = model
        print(
            f"[{bundle.name}] validation AUC={val_metrics['auc']:.6f}, test AUC={test_metrics['auc']:.6f}",
            flush=True,
        )
        pd.DataFrame(baseline_test_results).to_csv(output / "baseline_results_partial.csv", index=False)

    _stage("4/8 Training knowledge-tracing models: DKT, Transformer-KT, SAKT, SAINT+, AKT, Proposed Behavior-KT")
    (
        sequence_test_results,
        sequence_validation_results,
        proposed_model,
        skill_vocab,
        item_vocab,
        sequence_loaders,
    ) = _train_sequence_models(interaction_split, config, output, seed)
    model_comparison = pd.DataFrame(baseline_test_results + sequence_test_results)
    validation_comparison = pd.DataFrame(baseline_validation_results + sequence_validation_results)
    model_comparison.to_csv(output / "model_comparison.csv", index=False)
    validation_comparison.to_csv(output / "model_comparison_validation.csv", index=False)
    save_model_comparison(model_comparison, output / "figures" / "model_comparison_auc.png")

    _stage("5/8 Training pedagogical recommender")
    # Pedagogy labels are weak-supervision labels from the documented policy; no test labels are
    # observed by the model during fitting. Validation is used for early stopping only.
    device = torch_device()
    pedagogy_train = _attach_kt_probability(
        feature_split["train"], sequence_loaders["train_eval"], proposed_model, device
    )
    pedagogy_validation = _attach_kt_probability(
        feature_split["validation"], sequence_loaders["validation"], proposed_model, device
    )
    pedagogy_test = _attach_kt_probability(
        feature_split["test"], sequence_loaders["test"], proposed_model, device
    )
    proposed_model = proposed_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pedagogy_features = list(selected_features) + ["kt_probability"]
    for part in (pedagogy_train, pedagogy_validation, pedagogy_test):
        part["pedagogy_strategy"] = assign_pedagogy_strategy(part)

    pedagogy_cfg = config["pedagogy"]
    pedagogy_train_fit = deterministic_row_sample(
        pedagogy_train, pedagogy_cfg.get("max_train_rows"), seed + 11
    )
    pedagogy_validation_fit = deterministic_row_sample(
        pedagogy_validation, pedagogy_cfg.get("max_validation_rows"), seed + 12
    )

    pedagogy_model_path = model_dir / "pedagogy_recommender.joblib"
    if pedagogy_model_path.exists():
        payload = joblib.load(pedagogy_model_path)
        recommender = payload["model"]
        print(f"[pedagogy] completed model found; skipping fit -> {pedagogy_model_path}", flush=True)
    else:
        recommender = PedagogyRecommender(config["pedagogy"], seed)
        recommender.fit(
            pedagogy_train_fit[pedagogy_features],
            pedagogy_train_fit["pedagogy_strategy"],
            pedagogy_validation_fit[pedagogy_features],
            pedagogy_validation_fit["pedagogy_strategy"],
            checkpoint_dir=output / "restart_checkpoints" / "pedagogy",
            checkpoint_name="pedagogy",
        )
        joblib.dump(
            {"model": recommender, "features": pedagogy_features},
            pedagogy_model_path,
        )
        print(f"[pedagogy] final model saved -> {pedagogy_model_path}", flush=True)
    probabilities = recommender.predict_proba(pedagogy_test[pedagogy_features])
    encoded_truth = recommender.encoder.transform(pedagogy_test["pedagogy_strategy"])
    rec_metrics = recommendation_metrics(
        encoded_truth,
        probabilities,
        k=int(config["pedagogy"].get("top_k", 3)),
    )
    predicted_strategy = recommender.predict(pedagogy_test[pedagogy_features])
    rec_metrics["top1_accuracy"] = float(
        np.mean(predicted_strategy == pedagogy_test["pedagogy_strategy"].to_numpy())
    )
    distribution = pd.Series(predicted_strategy).value_counts()
    rec_metrics["strategy_distribution"] = {str(k): int(v) for k, v in distribution.items()}
    rec_metrics["interpretation"] = "Agreement with documented weak-supervision policy; not causal treatment-effect evidence."
    write_json(output / "pedagogy_metrics.json", rec_metrics)
    save_strategy_distribution(distribution, output / "figures" / "strategy_distribution.png")
    _stage("6/8 Computing SHAP explainability")
    explain_cfg = config["explainability"]
    explain_n = min(int(explain_cfg.get("sample_size", 1000)), len(pedagogy_test))
    explain_sample = pedagogy_test[pedagogy_features].sample(n=explain_n, random_state=seed)
    shap_importance, _ = tree_shap_importance(recommender.model, explain_sample)
    shap_importance.to_csv(output / "pedagogy_shap_importance.csv", index=False)
    save_shap_importance(shap_importance, output / "figures" / "pedagogy_shap_importance.png")
    explain_metrics = explanation_quality(
        recommender.model,
        explain_sample,
        top_features=int(explain_cfg.get("top_features", 3)),
        seed=seed,
    )
    write_json(output / "explainability_metrics.json", explain_metrics)
    local_reasons = local_shap_reasons(
        recommender.model,
        explain_sample,
        class_names=recommender.encoder.classes_.tolist(),
        top_features=int(explain_cfg.get("top_features", 3)),
    )
    local_reasons.to_csv(output / "pedagogy_local_explanations.csv", index=False)

    xgb_model = trained_baselines["xgboost"]
    predictor_sample = x_test.sample(n=min(explain_n, len(x_test)), random_state=seed)
    predictor_importance, _ = tree_shap_importance(xgb_model, predictor_sample)
    predictor_importance.to_csv(output / "prediction_shap_importance.csv", index=False)
    save_shap_importance(predictor_importance, output / "figures" / "prediction_shap_importance.png")

    _stage("7/8 Running ablation study")
    ablation = _run_proposed_ablations(
        config=config,
        output=output,
        seed=seed,
        skill_vocab=skill_vocab,
        item_vocab=item_vocab,
        loaders=sequence_loaders,
        sequence_test_results=sequence_test_results,
    )
    ablation.to_csv(output / "ablation_results.csv", index=False)

    _stage("8/8 Writing final summary and result tables")
    summary = {
        "dataset": dataset_cfg["name"],
        "feature_set": dataset_cfg.get("feature_set", "core"),
        "device": str(torch_device()),
        "seed": seed,
        "split": config["split"],
        "models_evaluated": model_comparison["model"].astype(str).tolist(),
        "pedagogy_metrics": rec_metrics,
        "explainability_metrics": explain_metrics,
    }
    write_json(output / "summary.json", summary)
    return summary
