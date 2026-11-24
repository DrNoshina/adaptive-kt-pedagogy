#!/usr/bin/env python3
"""Same-information baselines for Proposed V3 (answers the question "is the gain only the features?").

  xgb_f : XGBoost on exactly the per-target inputs V3 gets (learner profile, pre-response context,
          difficulty-adjusted mastery incl. item easiness, source/platform, part). No sequence model.
  dkt_f : V3 with its attention backbone (history Transformer + cross-attention) replaced by one LSTM
          (DKT-style). Same inputs, same prediction head, same training settings.
Same split, same targets, same threshold rule (validation F1) as every other model. Test is scored once.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import yaml
from torch import nn

import run_trackB_proposed_v3 as v3r          # importing only defines functions (and routes V3 batches through the trainer)
import adaptive_kt.training.kt_trainer as kt_trainer
from adaptive_kt.evaluation.metrics import binary_metrics, select_f1_threshold
from adaptive_kt.models.behavior_aware_kt_v2 import PROFILE_KEEP
from adaptive_kt.models.behavior_aware_kt_v3 import BehaviorAwareKTV3
from adaptive_kt.models.sequence import SequenceVocabulary
from adaptive_kt.models.sequence_v2 import fit_category_vocab
from adaptive_kt.models.sequence_v3 import KTSequenceDatasetV3, fit_item_easiness, item_lookup, load_question_meta
from adaptive_kt.utils.reproducibility import seed_everything, torch_device


class LSTMHistory(nn.Module):
    """Drop-in for V3's history Transformer: causal by construction (output t sees history 0..t)."""
    def __init__(self, dim: int):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, batch_first=True)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        return self.lstm(x)[0]


class LastState(nn.Module):
    """Drop-in for V3's cross-attention: target t+1 reads the LSTM state after interaction t."""
    def forward(self, query, key, value, **_):
        return value, None


def one_hot(codes, size):
    out = np.zeros((len(codes), max(int(size), 1)), dtype=np.float32)
    out[np.arange(len(codes)), np.clip(codes.astype(np.int64), 0, max(int(size), 1) - 1)] = 1.0
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--split", required=True)
    p.add_argument("--out", required=True); p.add_argument("--cache", required=True, help="the V3 cache folder of this dataset (reused, never rewritten differently)")
    p.add_argument("--model", choices=["xgb_f", "dkt_f"], required=True); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop", default="", help="input groups removed (dkt_f only), same names as the V3 runner")
    args = p.parse_args()
    drops = v3r.parse_drops(args.drop)
    if drops and args.model != "dkt_f": raise SystemExit("--drop is only for dkt_f")
    name = f"{args.model}{'_no_' + '_'.join(drops) if drops else ''}_seed{args.seed}"
    cfg = yaml.safe_load(open(args.config)); seq = cfg["sequence"]
    seed_everything(args.seed); device = torch_device(); started = time.time()
    out = Path(args.out); (out / "predictions").mkdir(parents=True, exist_ok=True); cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)

    split = v3r.load_split(Path(cfg["dataset"]["processed_root"]), Path(args.split), 0)
    skill_vocab, item_vocab = SequenceVocabulary.fit(split["train"]["skill_id"]), SequenceVocabulary.fit(split["train"]["item_id"])
    source_vocab, platform_vocab = fit_category_vocab(split["train"], "source"), fit_category_vocab(split["train"], "platform")
    part_of, tags_of, num_tags = load_question_meta(cfg["dataset"]["raw_root"]); item_part, item_tags = item_lookup(item_vocab, part_of, tags_of)
    max_len, min_history = int(seq["max_len"]), int(cfg["dataset"].get("min_history", 1))

    def dataset(n):
        return KTSequenceDatasetV3(split[n], skill_vocab, item_vocab, source_vocab, platform_vocab, max_len=max_len, min_history=min_history,
                                   split_name=n, cache_path=cache / f"{n}.npz")
    data = {n: dataset(n) for n in ("train", "validation", "test")}
    split.clear()
    tr = data["train"]
    full, oof, fold = fit_item_easiness(tr.items, tr.correct, tr.first_rows(), item_vocab.size)
    v3r.attach_mastery(tr, oof[fold, tr.items], item_part, item_tags, cache / "train_mastery_v1.npy")
    for n in ("validation", "test"):
        v3r.attach_mastery(data[n], full[data[n].items], item_part, item_tags, cache / f"{n}_mastery_v1.npy")

    if args.model == "xgb_f":
        import xgboost as xgb
        bcfg = cfg["baseline"]; xcfg = bcfg["xgboost"]; rng = np.random.default_rng(args.seed)
        def block(d, limit=None):
            rows = np.flatnonzero(d.eligible)                       # ascending = the order every other model writes its predictions
            if limit and len(rows) > limit: rows = np.sort(rng.choice(rows, size=int(limit), replace=False))
            x = np.hstack([d.behavior[rows][:, PROFILE_KEEP].astype(np.float32), d.ctx[rows].astype(np.float32), d.mastery[rows].astype(np.float32),
                           one_hot(d.source[rows], source_vocab.size), one_hot(d.platform[rows], platform_vocab.size), one_hot(item_part[d.items[rows]], 8)])
            return x, d.correct[rows].astype(np.int32)
        x_tr, y_tr = block(tr, bcfg.get("max_train_rows")); x_es, y_es = block(data["validation"], bcfg.get("max_validation_rows"))
        print(f"[{name}] inputs per target={x_tr.shape[1]} | train rows={len(y_tr):,}", flush=True)
        params = dict(n_estimators=int(xcfg["n_estimators"]), max_depth=int(xcfg["max_depth"]), learning_rate=float(xcfg["learning_rate"]),
                      subsample=float(xcfg["subsample"]), colsample_bytree=float(xcfg["colsample_bytree"]), reg_lambda=float(xcfg["reg_lambda"]),
                      min_child_weight=float(xcfg["min_child_weight"]), early_stopping_rounds=int(xcfg["early_stopping_rounds"]),
                      objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=args.seed, n_jobs=-1)
        if device.type == "cuda": params["device"] = "cuda"
        model = xgb.XGBClassifier(**params); model.fit(x_tr, y_tr, eval_set=[(x_es, y_es)], verbose=100)
        x_va, y_va = block(data["validation"]); x_te, y_te = block(data["test"])
        p_va, p_te = model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]
        best_epoch = int(getattr(model, "best_iteration", -1))
    else:
        dim = int(seq["embedding_dim"]); ctx_idx, behavior_idx = v3r.drop_indices(drops)
        model = BehaviorAwareKTV3(num_skills=skill_vocab.size, num_items=item_vocab.size, num_sources=source_vocab.size, num_platforms=platform_vocab.size,
                                  num_tags=num_tags, item_part=torch.from_numpy(item_part), item_tags=torch.from_numpy(item_tags), max_len=max_len + 1,
                                  embedding_dim=dim, num_heads=int(seq["transformer_heads"]), num_layers=int(seq["transformer_layers"]), dropout=float(seq["dropout"]),
                                  ablate_ctx_indices=tuple(ctx_idx), ablate_behavior_indices=tuple(behavior_idx), use_source="source" not in drops,
                                  use_profile="profile" not in drops, use_meta="meta" not in drops, use_elapsed="response_time" not in drops)
        model.history_encoder = LSTMHistory(dim); model.cross_attention = LastState()
        print(f"[{name}] params={sum(p.numel() for p in model.parameters()):,} device={device}", flush=True)
        bs, workers = int(seq["batch_size"]), int(seq.get("num_workers", 0))
        train_loader = v3r.make_loader(tr, bs, True, args.seed, workers); val_loader = v3r.make_loader(data["validation"], bs, False, args.seed, workers)
        model.to(device)
        v3r.smoke_checks(model, next(iter(train_loader)), device)   # stops the run if a prediction could see its own answer
        trained = kt_trainer.train_kt_model(model=model, train_loader=train_loader, validation_loader=val_loader, device=device,
                                            epochs=int(seq["epochs"]), learning_rate=float(seq["learning_rate"]), weight_decay=float(seq["weight_decay"]),
                                            patience=int(seq["patience"]), checkpoint_path=out / "checkpoints" / f"{name}.resume.pt",
                                            history_path=out / "training_history" / f"{name}.csv", model_name=name, seed=args.seed)
        model, best_epoch = trained.model, int(trained.best_epoch)
        y_va, p_va = kt_trainer.predict_kt(model, val_loader, device)
        y_te, p_te = kt_trainer.predict_kt(model, v3r.make_loader(data["test"], bs, False, args.seed, workers), device)

    threshold = select_f1_threshold(y_va, p_va)
    result = {"model": args.model, "dropped_groups": drops, "seed": args.seed, "best_epoch": best_epoch, "decision_threshold": float(threshold),
              "validation": binary_metrics(y_va, p_va, threshold), "test": binary_metrics(y_te, p_te, threshold)}
    for part, (y, prob) in (("validation", (y_va, p_va)), ("test", (y_te, p_te))):
        np.savez(out / "predictions" / f"{name}_{part}.npz", y_true=np.asarray(y).astype(np.uint8), y_prob=np.asarray(prob).astype(np.float32))
    (out / f"{name}_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[{name}] done in {(time.time() - started) / 60:.1f} min | validation AUC {result['validation']['auc']:.4f} | test AUC {result['test']['auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
