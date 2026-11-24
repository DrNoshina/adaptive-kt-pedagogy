#!/usr/bin/env python3
"""Track-B Proposed-V3 = V2 + difficulty-adjusted mastery (Elo) + part/tag embeddings.

Same frozen 70/10/20 split, same windows/targets and same hyper-parameters as V1/V2.
Decision rule (fixed before the run): V3 replaces V2 only if seed-42 VALIDATION AUC is at least
V2 validation AUC + 0.002. Test is evaluated only with --test.

  smoke : --max-shards 20 --epochs 1 --batch-size 64 --screen --out results/ednet_kt3_trackB_v3_smoke
  full  : (no flags)            -> results/ednet_kt3_trackB_v3
  ablate: --drop mastery        (see DROP_CHOICES)
  test  : --test                (reuses the finished checkpoint, no retraining)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import adaptive_kt.training.kt_trainer as kt_trainer
from adaptive_kt.evaluation.metrics import binary_metrics, select_f1_threshold
from adaptive_kt.models.behavior_aware_kt_v2 import PROFILE_KEEP
from adaptive_kt.models.behavior_aware_kt_v3 import BehaviorAwareKTV3
from adaptive_kt.models.sequence import BEHAVIOR_FEATURE_INDEX, SequenceVocabulary
from adaptive_kt.models.sequence_v2 import fit_category_vocab
from adaptive_kt.models.sequence_v3 import (
    CTX_FEATURES_V3,
    CTX_GROUPS_V3,
    MASTERY_DIM,
    KTSequenceDatasetV3,
    collate_sequences_v3,
    fit_item_easiness,
    item_lookup,
    load_question_meta,
    mastery_features,
)
from adaptive_kt.utils.reproducibility import seed_everything, torch_device
from adaptive_kt.utils.tableio import read_table

warnings.filterwarnings("ignore")

V2_TRACKB_VALIDATION_AUC = 0.7659315577236128
ADOPT_MARGIN = 0.002
MODEL_KEYS = ("skills", "items", "correct", "behavior", "elapsed", "lag", "ctx", "source", "platform")
BEHAVIOR_DROP = {
    "recent_accuracy": [BEHAVIOR_FEATURE_INDEX["recent_accuracy"]],
    "mastery": [BEHAVIOR_FEATURE_INDEX["topic_mastery"]],
    "response_time": [BEHAVIOR_FEATURE_INDEX["mean_response_time"], BEHAVIOR_FEATURE_INDEX["current_elapsed"]],
}
DROP_CHOICES = sorted(set(CTX_GROUPS_V3) | set(BEHAVIOR_DROP) | {"source", "profile", "meta"})
ELO = CTX_GROUPS_V3["elo"]
# Ablations that will be reported for V3 (checked in the smoke test, run later by the queue).
ABLATION_PLAN = [
    "profile,experience,timing,recency,engagement,source,elo,meta",
    "experience,timing,recency,engagement,source,elo,meta",
    "experience,timing,engagement,source,elo,meta",
    "experience,timing,engagement,elo,meta",
    "experience,timing,engagement",
    "mastery",
    "response_time",
    "recent_accuracy",
]


def parse_drops(text: str) -> list[str]:
    drops = sorted({item.strip() for item in text.split(",") if item.strip()})
    unknown = set(drops) - set(DROP_CHOICES)
    if unknown:
        raise ValueError(f"unknown --drop {sorted(unknown)}; choose from {DROP_CHOICES}")
    return drops


def drop_indices(drops: list[str]) -> tuple[list[int], list[int]]:
    ctx = {i for group in drops if group in CTX_GROUPS_V3 for i in CTX_GROUPS_V3[group]}
    if "mastery" in drops:  # the proposal's "mastery score" = raw topic mastery + difficulty-adjusted mastery
        ctx |= set(ELO)
    behavior = {i for group in drops if group in BEHAVIOR_DROP for i in BEHAVIOR_DROP[group]}
    return sorted(ctx), sorted(behavior)


def batch_predictions_v3(model, batch, device):
    inputs = {key: batch[key].to(device, non_blocking=True) for key in MODEL_KEYS if key in batch}
    logits = model(**inputs)
    targets = inputs["correct"][:, 1:].float()
    valid = inputs["items"][:, 1:].ne(0)
    if "eligible" in batch:
        valid = valid & batch["eligible"].to(device, non_blocking=True)[:, 1:]
    return logits[valid], targets[valid]


# V3 batches carry extra tensors; route them through the shared trainer without editing it.
kt_trainer._batch_predictions = batch_predictions_v3


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(processed_root: Path, assignment_path: Path, max_shards: int) -> dict[str, pd.DataFrame]:
    directory = processed_root / "interactions"
    files = sorted(directory.glob("*.parquet")) + sorted(directory.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No prepared interaction shards in {directory}")
    if max_shards > 0:
        files = files[:max_shards]
    frame = pd.concat((read_table(path) for path in files), ignore_index=True)
    frame["user_id"] = frame["user_id"].astype(str)
    print(f"[v3] loaded {len(frame):,} interactions from {len(files)} shards", flush=True)

    assignments = pd.read_csv(assignment_path, dtype={"user_id": str, "split": str})
    if assignments["user_id"].duplicated().any():
        raise AssertionError("frozen split has duplicate learners")
    split_of = frame["user_id"].map(dict(zip(assignments["user_id"], assignments["split"])))
    if split_of.isna().any():
        raise AssertionError(f"{int(split_of.isna().sum())} rows have learners missing from {assignment_path}")
    if max_shards <= 0 and set(frame["user_id"].unique()) != set(assignments["user_id"]):
        raise AssertionError("frozen split does not match the prepared data")
    parts = {name: frame[split_of.eq(name).to_numpy()].reset_index(drop=True) for name in ("train", "validation", "test")}
    del frame
    users = {name: set(part["user_id"].unique()) for name, part in parts.items()}
    if users["train"] & users["validation"] or users["train"] & users["test"] or users["validation"] & users["test"]:
        raise AssertionError("learner leakage across splits")
    for name, part in parts.items():
        print(f"[v3] {name}: users={len(users[name]):,} interactions={len(part):,}", flush=True)
    return parts


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sequences_v3,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def link_v2_cache(cache_dir: Path, v2_cache_dir: Path) -> None:
    """Reuse the V2 sequence cache (read-only) through symlinks; V2 files are never rewritten."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        source, target = v2_cache_dir / f"{name}.npz", cache_dir / f"{name}.npz"
        if source.exists() and not target.exists() and not target.is_symlink():
            target.symlink_to(source.resolve())


def attach_mastery(dataset, easiness_row: np.ndarray, item_part, item_tags, cache_file: Path | None) -> None:
    if cache_file is not None and cache_file.exists():
        cached = np.load(cache_file)
        if cached.shape == (len(dataset.items), MASTERY_DIM):
            dataset.set_mastery(cached)
            print(f"[v3:{dataset.split_name}] mastery cache loaded", flush=True)
            return
    started = time.time()
    mastery = mastery_features(dataset.items, dataset.correct, dataset.first_rows(), easiness_row, item_part, item_tags)
    dataset.set_mastery(mastery)
    print(f"[v3:{dataset.split_name}] mastery built for {len(mastery):,} rows in {time.time() - started:.0f}s", flush=True)
    if cache_file is not None:
        temporary = cache_file.with_name(cache_file.stem + ".tmp.npy")
        np.save(temporary, mastery)
        temporary.replace(cache_file)


def mastery_leakage_check(dataset, easiness_full, item_part, item_tags, seed: int, n_rows: int = 200) -> None:
    """Flip a row's OWN answer, then every LATER answer: the row's mastery inputs must not move."""
    first = dataset.first_rows()
    bounds = np.flatnonzero(np.r_[first, True])
    rng = np.random.default_rng(seed)
    worst_own = worst_future = 0.0
    effective = 0

    def build(lo, hi, answers):
        items = dataset.items[lo:hi]
        return mastery_features(items, answers, first[lo:hi], easiness_full[items], item_part, item_tags).astype(np.float32)

    picks = rng.choice(len(dataset.items), size=min(n_rows, len(dataset.items)), replace=False)
    for row in picks:
        position = np.searchsorted(bounds, row, side="right")
        lo, hi = int(bounds[position - 1]), int(bounds[position])
        local = int(row - lo)
        answers = dataset.correct[lo:hi]
        base = build(lo, hi, answers)
        own = answers.copy()
        own[local] = 1 - own[local]
        changed = build(lo, hi, own)
        worst_own = max(worst_own, float(np.abs(changed[local] - base[local]).max()))
        if local + 1 < hi - lo:
            effective += int(np.abs(changed[local + 1:] - base[local + 1:]).max() > 0)
            future = answers.copy()
            future[local + 1:] = 1 - future[local + 1:]
            worst_future = max(worst_future, float(np.abs(build(lo, hi, future)[local] - base[local]).max()))
    status = "PASS" if worst_own == 0.0 and worst_future == 0.0 else "FAIL"
    print(f"[leakage-mastery] {status}: {len(picks)} rows | input change from own answer={worst_own:.1e}, "
          f"from later answers={worst_future:.1e} | flip reached later rows in {effective} cases", flush=True)
    if status != "PASS":
        raise AssertionError("target leakage in V3 mastery inputs")


def screen(datasets, source_vocab, platform_vocab, seed: int) -> None:
    """XGBoost on validation: does the Elo block add signal beyond everything V2 already uses?"""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    train, val = datasets["train"], datasets["validation"]
    n_items = int(max(train.items.max(), val.items.max())) + 1
    counts = np.bincount(train.items, minlength=n_items).astype(np.float64)
    hits = np.bincount(train.items, weights=train.correct.astype(np.float64), minlength=n_items)
    prior = float(train.correct.mean())
    difficulty = (hits + 20.0 * prior) / (counts + 20.0)

    def one_hot(codes, size):
        out = np.zeros((len(codes), max(size, 1)), dtype=np.float32)
        out[np.arange(len(codes)), np.clip(codes, 0, size - 1)] = 1.0
        return out

    def block(dataset, limit):
        rows = np.flatnonzero(dataset.eligible)
        if len(rows) > limit:
            rows = np.sort(rng.choice(rows, size=limit, replace=False))
        v2 = np.column_stack([
            dataset.behavior[rows][:, PROFILE_KEEP],
            difficulty[dataset.items[rows]],
            dataset.ctx[rows].astype(np.float32),
            one_hot(dataset.source[rows].astype(np.int64), source_vocab.size),
            one_hot(dataset.platform[rows].astype(np.int64), platform_vocab.size),
        ]).astype(np.float32)
        return v2, dataset.mastery[rows].astype(np.float32), dataset.correct[rows].astype(np.int32)

    tr_v2, tr_elo, tr_y = block(train, 800_000)
    va_v2, va_elo, va_y = block(val, 400_000)
    names = CTX_FEATURES_V3[-MASTERY_DIM:]
    print("[elo] alone AUC on validation: " + ", ".join(
        f"{name}={roc_auc_score(va_y, va_elo[:, k]):.4f}" for k, name in enumerate(names) if name in
        ("item_easiness", "ability_long", "tag_ability_mean", "expected_correct")), flush=True)
    base_auc = None
    print(f"[screen] XGBoost, train rows={len(tr_y):,}, validation rows={len(va_y):,}", flush=True)
    for name, use_elo in (("everything V2 uses", False), ("  + difficulty-adjusted mastery", True)):
        x_tr = np.hstack([tr_v2, tr_elo]) if use_elo else tr_v2
        x_va = np.hstack([va_v2, va_elo]) if use_elo else va_v2
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, random_state=seed,
        )
        clf.fit(x_tr, tr_y)
        auc = roc_auc_score(va_y, clf.predict_proba(x_va)[:, 1])
        base_auc = auc if base_auc is None else base_auc
        print(f"[screen] {name:34s} val AUC={auc:.4f}  delta={auc - base_auc:+.4f}", flush=True)


def smoke_checks(model, batch, device) -> None:
    print("[smoke] batch shapes: " + ", ".join(f"{k}={tuple(v.shape)}" for k, v in batch.items()), flush=True)
    model.train()
    logits, targets = batch_predictions_v3(model, batch, device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    print(f"[smoke] forward/backward loss={loss.item():.4f} finite_grads={finite}", flush=True)
    if not (finite and torch.isfinite(loss)):
        raise AssertionError("non-finite loss/gradients")

    model.eval()
    inputs = {key: batch[key].to(device) for key in MODEL_KEYS if key in batch}
    j = min(20, inputs["items"].shape[1] - 2)
    with torch.no_grad():
        base = model(**inputs)
        changed = {key: value.clone() for key, value in inputs.items()}
        changed["correct"][:, j] = 1 - changed["correct"][:, j]
        changed["elapsed"][:, j] = torch.rand_like(changed["elapsed"][:, j])
        changed["lag"][:, j] = torch.rand_like(changed["lag"][:, j])
        changed["behavior"][:, j, 12:] = torch.rand_like(changed["behavior"][:, j, 12:])
        after = model(**changed)
    own = float((after[:, j - 1] - base[:, j - 1]).abs().max())
    later = float((after[:, j] - base[:, j]).abs().max())
    status = "PASS" if own < 1e-4 and later > 0 else "FAIL"
    print(f"[leakage-model] {status}: prediction of row {j} moved {own:.2e} when its own answer/time changed "
          f"(next prediction moved {later:.2e})", flush=True)
    if status != "PASS":
        raise AssertionError("model uses post-response information of its own target")


def ablation_selftest(build_model, batch, device) -> None:
    """For every planned ablation: scrambling the removed inputs must not change the output."""
    inputs = {key: batch[key].to(device) for key in MODEL_KEYS if key in batch}
    for text in ABLATION_PLAN:
        drops = parse_drops(text)
        ctx_idx, behavior_idx = drop_indices(drops)
        model = build_model(drops).to(device).eval()
        changed = {key: value.clone() for key, value in inputs.items()}
        if ctx_idx:
            changed["ctx"][:, :, ctx_idx] = torch.rand_like(changed["ctx"][:, :, ctx_idx])
        if behavior_idx:
            changed["behavior"][:, :, behavior_idx] = torch.rand_like(changed["behavior"][:, :, behavior_idx])
        if "profile" in drops:
            changed["behavior"] = torch.rand_like(changed["behavior"])
        if "response_time" in drops:
            changed["elapsed"] = torch.rand_like(changed["elapsed"])
        if "source" in drops:
            changed["source"] = torch.ones_like(changed["source"])
            changed["platform"] = torch.ones_like(changed["platform"])
        with torch.no_grad():
            base = model(**inputs)
            if "meta" in drops:
                model.item_part.fill_(1)
                model.item_tags.fill_(1)
            after = model(**changed)
        effect = float((after - base).abs().max())
        print(f"[ablation-check] {'OK  ' if effect < 1e-5 else 'FAIL'} effect of removed inputs={effect:.1e} | drop {text}", flush=True)
        if effect >= 1e-5:
            raise AssertionError(f"ablation '{text}' does not fully remove its inputs")
        del model


def evaluate(model, loader, device, threshold=None):
    y_true, y_prob = kt_trainer.predict_kt(model, loader, device)
    if threshold is None:
        threshold = select_f1_threshold(y_true, y_prob)
    return binary_metrics(y_true, y_prob, threshold=threshold), float(threshold), y_true, y_prob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ednet_kt3_trackB.yaml")
    parser.add_argument("--split", default="results/ednet_kt3_trackB/split_assignments.csv")
    parser.add_argument("--out", default="results/ednet_kt3_trackB_v3")
    parser.add_argument("--v2-cache", default="results/ednet_kt3_trackB_v2/cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop", default="", help="comma list of: " + ",".join(DROP_CHOICES))
    parser.add_argument("--max-shards", type=int, default=0, help="smoke only: first N prepared shards")
    parser.add_argument("--epochs", type=int, default=0, help="override epochs (smoke only)")
    parser.add_argument("--batch-size", type=int, default=0, help="override batch size (smoke only)")
    parser.add_argument("--screen", action="store_true", help="CPU XGBoost signal check before training")
    parser.add_argument("--test", action="store_true", help="evaluate the finished checkpoint on TEST")
    args = parser.parse_args()

    drops = parse_drops(args.drop)
    tag = "v3" if not drops else "v3_no_" + "_".join(drops)
    smoke = args.max_shards > 0

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    seq_cfg = cfg["sequence"]
    seed_everything(args.seed)
    device = torch_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = None if smoke else out / "cache"
    if cache_dir is not None:
        link_v2_cache(cache_dir, Path(args.v2_cache))
    started = time.time()

    split = load_split(Path(cfg["dataset"]["processed_root"]), Path(args.split), args.max_shards)
    skill_vocab = SequenceVocabulary.fit(split["train"]["skill_id"])
    item_vocab = SequenceVocabulary.fit(split["train"]["item_id"])
    source_vocab = fit_category_vocab(split["train"], "source")
    platform_vocab = fit_category_vocab(split["train"], "platform")
    part_of, tags_of, num_tags = load_question_meta(cfg["dataset"]["raw_root"])
    item_part, item_tags = item_lookup(item_vocab, part_of, tags_of)
    print(f"[v3] items={item_vocab.size} with part={int((item_part > 0).sum())} with tags={int((item_tags[:, 0] > 0).sum())} "
          f"single tags={num_tags}", flush=True)

    max_len = int(seq_cfg["max_len"])
    min_history = int(cfg["dataset"].get("min_history", 1))
    batch_size = args.batch_size if args.batch_size > 0 else int(seq_cfg["batch_size"])
    workers = int(seq_cfg.get("num_workers", 0))

    def dataset(name):
        return KTSequenceDatasetV3(
            split[name], skill_vocab, item_vocab, source_vocab, platform_vocab,
            max_len=max_len, min_history=min_history, split_name=name,
            cache_path=None if cache_dir is None else cache_dir / f"{name}.npz",
        )

    def mastery_file(name):
        return None if cache_dir is None else cache_dir / f"{name}_mastery_v1.npy"

    def build_model(model_drops):
        ctx_idx, behavior_idx = drop_indices(model_drops)
        return BehaviorAwareKTV3(
            num_skills=skill_vocab.size,
            num_items=item_vocab.size,
            num_sources=source_vocab.size,
            num_platforms=platform_vocab.size,
            num_tags=num_tags,
            item_part=torch.from_numpy(item_part),
            item_tags=torch.from_numpy(item_tags),
            max_len=max_len + 1,
            embedding_dim=int(seq_cfg["embedding_dim"]),
            num_heads=int(seq_cfg["transformer_heads"]),
            num_layers=int(seq_cfg["transformer_layers"]),
            dropout=float(seq_cfg["dropout"]),
            ablate_ctx_indices=tuple(ctx_idx),
            ablate_behavior_indices=tuple(behavior_idx),
            use_source="source" not in model_drops,
            use_profile="profile" not in model_drops,
            use_meta="meta" not in model_drops,
            use_elapsed="response_time" not in model_drops,
        )

    model = build_model(drops)
    print(f"[v3] tag={tag} dropped={drops or 'none'} params={sum(p.numel() for p in model.parameters()):,} device={device}", flush=True)

    checkpoint = out / "checkpoints" / f"{tag}_seed{args.seed}.resume.pt"
    finished = False
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        finished = state.get("status") == "complete"
    need_train = (not finished) or args.screen

    datasets = {"train": dataset("train"), "validation": dataset("validation")}
    split.pop("train", None)
    train = datasets["train"]
    easiness_full, easiness_oof, fold = fit_item_easiness(train.items, train.correct, train.first_rows(), item_vocab.size)
    validation = datasets["validation"]
    attach_mastery(validation, easiness_full[validation.items], item_part, item_tags, mastery_file("validation"))
    if need_train:
        attach_mastery(train, easiness_oof[fold, train.items], item_part, item_tags, mastery_file("train"))
    else:
        datasets.pop("train")
    del train
    if smoke:
        mastery_leakage_check(validation, easiness_full, item_part, item_tags, args.seed)
    if args.screen:
        screen(datasets, source_vocab, platform_vocab, args.seed)

    loaders = {"validation": make_loader(validation, batch_size, False, args.seed, workers)}
    if finished:
        model.load_state_dict(state["best_state_dict"])
        model.to(device)
        best_epoch = int(state.get("best_epoch", 0))
        print(f"[v3] finished checkpoint found (best epoch {best_epoch}); no retraining", flush=True)
    else:
        loaders["train"] = make_loader(datasets["train"], batch_size, True, args.seed, workers)
        model.to(device)
        if smoke:
            batch = next(iter(loaders["train"]))
            smoke_checks(model, batch, device)
            ablation_selftest(build_model, batch, device)
        trained = kt_trainer.train_kt_model(
            model=model,
            train_loader=loaders["train"],
            validation_loader=loaders["validation"],
            device=device,
            epochs=args.epochs if args.epochs > 0 else int(seq_cfg["epochs"]),
            learning_rate=float(seq_cfg["learning_rate"]),
            weight_decay=float(seq_cfg["weight_decay"]),
            patience=int(seq_cfg["patience"]),
            checkpoint_path=checkpoint,
            history_path=out / "training_history" / f"{tag}_seed{args.seed}.csv",
            model_name=f"{tag}_trackB_seed{args.seed}",
            seed=args.seed,
        )
        model = trained.model
        best_epoch = int(trained.best_epoch)
    datasets.pop("train", None)
    loaders.pop("train", None)

    val_metrics, threshold, val_true, val_prob = evaluate(model, loaders["validation"], device)
    predictions = out / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    np.savez(predictions / f"{tag}_seed{args.seed}_validation.npz", y_true=val_true.astype(np.uint8), y_prob=val_prob.astype(np.float32))

    delta = float(val_metrics["auc"] - V2_TRACKB_VALIDATION_AUC)
    result_path = out / f"{tag}_seed{args.seed}_metrics.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    result.update({
        "model": f"proposed_behavior_kt_{tag}",
        "seed": int(args.seed),
        "split": "frozen learner-disjoint 70/10/20 (results/ednet_kt3_trackB/split_assignments.csv)",
        "dropped_groups": drops,
        "smoke_subset_shards": int(args.max_shards),
        "best_epoch": best_epoch,
        "decision_threshold": threshold,
        "validation": val_metrics,
        "v2_trackB_validation_auc": V2_TRACKB_VALIDATION_AUC,
        "validation_auc_delta_vs_v2": delta,
        "adopt_rule": f"validation AUC >= V2 + {ADOPT_MARGIN}",
        "adopt": bool(not drops and not smoke and delta >= ADOPT_MARGIN),
    })
    if args.test:
        test_data = dataset("test")
        attach_mastery(test_data, easiness_full[test_data.items], item_part, item_tags, mastery_file("test"))
        test_metrics, _, test_true, test_prob = evaluate(model, make_loader(test_data, batch_size, False, args.seed, workers), device, threshold)
        np.savez(predictions / f"{tag}_seed{args.seed}_test.npz", y_true=test_true.astype(np.uint8), y_prob=test_prob.astype(np.float32))
        result["test"] = test_metrics
    write_json(result_path, result)

    print(f"\n===== TRACK-B PROPOSED {tag.upper()} seed {args.seed} ({(time.time() - started) / 60:.1f} min) =====")
    print(f"best epoch {best_epoch} | validation AUC {val_metrics['auc']:.6f} "
          f"(V2 {V2_TRACKB_VALIDATION_AUC:.6f}, delta {delta:+.6f})")
    if smoke:
        print("(smoke subset: AUC here is NOT comparable to V2)")
    if "test" in result and args.test:
        print(f"TEST AUC {result['test']['auc']:.6f} (threshold from validation {threshold:.3f})")
    print(f"saved -> {result_path}")


if __name__ == "__main__":
    main()
