#!/usr/bin/env python3
"""All baselines on the SAME frozen Track-B 70/10/20 split used by Proposed V1/V2 and HGAKT.

Classical: logistic_regression, random_forest, xgboost (causal feature table, same settings as Track-A).
Sequence : dkt_lstm, transformer_kt, sakt, saint_plus, akt (same windows/targets as V1/V2).
Threshold from validation only. A model whose metrics file exists is skipped (no reruns).
Every model saves validation/test predictions for later paired significance tests.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

CLASSICAL = ["logistic_regression", "random_forest", "xgboost"]
SEQUENCE = ["dkt_lstm", "transformer_kt", "sakt", "saint_plus", "akt"]
METRICS = ["auc", "accuracy", "precision", "recall", "f1", "rmse"]
OTHER_RESULTS = {
    "proposed_v1": "results/ednet_kt3_trackB/proposed_seed42_metrics.json",
    "hgakt_base_paper": "results/hgakt_kt3_trackB/hgakt_seed42_metrics.json",
    "proposed_v2": "results/ednet_kt3_trackB_v2/v2_seed42_metrics.json",
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_result(out: Path, name: str, seed: int, val_metrics, test_metrics, extra, val_true, val_prob, test_true, test_prob):
    pred = out / "predictions"
    pred.mkdir(parents=True, exist_ok=True)
    np.savez(pred / f"{name}_seed{seed}_validation.npz", y_true=np.asarray(val_true).astype(np.uint8), y_prob=np.asarray(val_prob).astype(np.float32))
    np.savez(pred / f"{name}_seed{seed}_test.npz", y_true=np.asarray(test_true).astype(np.uint8), y_prob=np.asarray(test_prob).astype(np.float32))
    write_json(out / f"{name}_seed{seed}_metrics.json", {
        "model": name, "seed": seed,
        "split": "frozen learner-disjoint 70/10/20 (results/ednet_kt3_trackB/split_assignments.csv)",
        **extra, "validation": val_metrics, "test": test_metrics,
    })
    print(f"[{name}] validation AUC={val_metrics['auc']:.6f} | test AUC={test_metrics['auc']:.6f}", flush=True)


def summary(out: Path, seed: int) -> None:
    rows = []
    for name in CLASSICAL + SEQUENCE:
        path = out / f"{name}_seed{seed}_metrics.json"
        if path.exists():
            rows.append((name, json.loads(path.read_text())))
    for name, path in OTHER_RESULTS.items():
        if Path(path).exists():
            rows.append((name, json.loads(Path(path).read_text())))
    table = []
    for name, res in rows:
        if "test" not in res:
            continue
        row = {"model": name, "val_auc": res["validation"]["auc"]}
        row.update({f"test_{m}": res["test"][m] for m in METRICS})
        table.append(row)
    if not table:
        print("[summary] no finished results yet")
        return
    frame = pd.DataFrame(table).sort_values("test_auc").reset_index(drop=True)
    frame.to_csv(out / f"trackB_comparison_seed{seed}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n===== TRACK-B COMPARISON (same split, seed %d) =====" % seed)
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"saved -> {out / f'trackB_comparison_seed{seed}.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ednet_kt3_trackB.yaml")
    parser.add_argument("--split", default="results/ednet_kt3_trackB/split_assignments.csv")
    parser.add_argument("--out", default="results/ednet_kt3_trackB_baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default=",".join(CLASSICAL + SEQUENCE))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summary(out, args.seed)
        return

    import joblib
    import torch

    from adaptive_kt.data.schema import feature_columns
    from adaptive_kt.evaluation.metrics import binary_metrics, select_f1_threshold
    from adaptive_kt.experiments.pipeline import (
        _filter_users,
        _fit_binary_model,
        _make_loaders,
        _read_collection,
        _resized_sequence_loaders,
        _split_from_saved_assignments,
    )
    from adaptive_kt.models.akt import AKT
    from adaptive_kt.models.baselines import deterministic_row_sample, make_baselines, predict_probability
    from adaptive_kt.models.dkt import DKT
    from adaptive_kt.models.saint_plus import SAINTPlus
    from adaptive_kt.models.sakt import SAKT
    from adaptive_kt.models.transformer_kt import TransformerKT
    from adaptive_kt.training.kt_trainer import predict_kt, train_kt_model
    from adaptive_kt.utils.reproducibility import seed_everything, torch_device

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    seed = int(args.seed)
    cfg["seed"] = seed
    seed_everything(seed)
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    todo = [m for m in wanted if not (out / f"{m}_seed{seed}_metrics.json").exists()]
    print(f"[baselines] already done: {sorted(set(wanted) - set(todo)) or 'none'} | to run: {todo or 'none'}", flush=True)
    if not todo:
        summary(out, seed)
        return

    processed_root = Path(cfg["dataset"]["processed_root"])
    print("[baselines] loading interactions...", flush=True)
    interactions = _read_collection(processed_root / "interactions")
    split, user_sets = _split_from_saved_assignments(interactions, Path(args.split))
    del interactions
    for name, part in split.items():
        print(f"[baselines] {name}: users={part['user_id'].nunique():,} interactions={len(part):,}", flush=True)

    classical_todo = [m for m in todo if m in CLASSICAL]
    if classical_todo:
        print("[baselines] loading causal feature table...", flush=True)
        features = _read_collection(processed_root / "features")
        fsplit = {name: _filter_users(features, users) for name, users in user_sets.items()}
        del features
        columns = feature_columns(cfg["dataset"].get("feature_set", "core"))
        bcfg = cfg["baseline"]
        train_f = deterministic_row_sample(fsplit["train"], bcfg.get("max_train_rows"), seed)
        val_f = deterministic_row_sample(fsplit["validation"], bcfg.get("max_validation_rows"), seed + 1)
        x_tr, y_tr = train_f[columns], train_f["correct"].astype(int)
        x_va, y_va = val_f[columns], val_f["correct"].astype(int)
        x_va_all, y_va_all = fsplit["validation"][columns], fsplit["validation"]["correct"].astype(int)
        x_te, y_te = fsplit["test"][columns], fsplit["test"]["correct"].astype(int)
        print(f"[baselines] features={columns} | train rows used={len(x_tr):,} (cap={bcfg.get('max_train_rows')})", flush=True)
        for bundle in make_baselines(bcfg, seed):
            if bundle.name not in classical_todo:
                continue
            print(f"\n=== {bundle.name} ===", flush=True)
            model = _fit_binary_model(bundle, x_tr, y_tr, x_va, y_va,
                                      checkpoint_dir=out / "restart_checkpoints" / bundle.name, checkpoint_name=bundle.name)
            val_prob = predict_probability(model, x_va_all)
            threshold = select_f1_threshold(y_va_all, val_prob)
            test_prob = predict_probability(model, x_te)
            joblib.dump({"model": model, "decision_threshold": threshold, "features": columns}, out / f"{bundle.name}_seed{seed}.joblib")
            save_result(out, bundle.name, seed,
                        binary_metrics(y_va_all, val_prob, threshold), binary_metrics(y_te, test_prob, threshold),
                        {"decision_threshold": threshold, "train_rows_used": len(x_tr)},
                        y_va_all, val_prob, y_te, test_prob)
        del fsplit, x_tr, x_va, x_va_all, x_te

    seq_todo = [m for m in todo if m in SEQUENCE]
    if seq_todo:
        seq = cfg["sequence"]
        device = torch_device()
        skill_vocab, item_vocab, loaders = _make_loaders(split, cfg, seed)
        del split
        e, h, layers, drop = int(seq["embedding_dim"]), int(seq["transformer_heads"]), int(seq["transformer_layers"]), float(seq["dropout"])
        L = int(seq["max_len"]) + 1
        builders = {
            "dkt_lstm": lambda: DKT(num_skills=skill_vocab.size, num_items=item_vocab.size, embedding_dim=e, hidden_dim=int(seq["hidden_dim"]), dropout=drop),
            "transformer_kt": lambda: TransformerKT(num_skills=skill_vocab.size, num_items=item_vocab.size, max_len=L, embedding_dim=e, num_heads=h, num_layers=layers, dropout=drop),
            "sakt": lambda: SAKT(num_skills=skill_vocab.size, num_items=item_vocab.size, max_len=L, embedding_dim=e, num_heads=h, dropout=drop),
            "saint_plus": lambda: SAINTPlus(num_skills=skill_vocab.size, num_items=item_vocab.size, max_len=L, embedding_dim=e, num_heads=h, num_layers=layers, dropout=drop),
            "akt": lambda: AKT(num_skills=skill_vocab.size, num_items=item_vocab.size, embedding_dim=e, num_heads=h, num_layers=layers, dropout=drop),
        }
        for name in SEQUENCE:
            if name not in seq_todo:
                continue
            print(f"\n=== {name} ===", flush=True)
            seed_everything(seed)
            active = loaders
            if name == "akt" and int(seq.get("akt_batch_size", seq["batch_size"])) != int(seq["batch_size"]):
                active = _resized_sequence_loaders(loaders, int(seq["akt_batch_size"]), seed)
            trained = train_kt_model(
                model=builders[name](), train_loader=active["train"], validation_loader=active["validation"], device=device,
                epochs=int(seq["epochs"]), learning_rate=float(seq["learning_rate"]), weight_decay=float(seq["weight_decay"]),
                patience=int(seq["patience"]), checkpoint_path=out / "checkpoints" / f"{name}_seed{seed}.resume.pt",
                history_path=out / "training_history" / f"{name}_seed{seed}.csv", model_name=f"{name}_trackB_seed{seed}", seed=seed,
            )
            val_true, val_prob = predict_kt(trained.model, active["validation"], device)
            threshold = select_f1_threshold(val_true, val_prob)
            test_true, test_prob = predict_kt(trained.model, active["test"], device)
            save_result(out, name, seed,
                        binary_metrics(val_true, val_prob, threshold), binary_metrics(test_true, test_prob, threshold),
                        {"decision_threshold": threshold, "best_epoch": int(trained.best_epoch)},
                        val_true, val_prob, test_true, test_prob)
            del trained
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary(out, seed)


if __name__ == "__main__":
    main()
