"""Pedagogy + explainability with the trained Proposed-V3 model, one stage per process.
Usage: python -u kt_ped_stages.py data|features|kt|final      (every stage skips work that is already done)"""
import os, sys, json, gc, time, types, shutil, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
STAGE = sys.argv[1]; K = os.environ["KT_ROOT"]; OUT = os.environ["KT_OUTPUT"]; T0 = float(os.environ.get("KT_T0", time.time()))
V3 = f"{K}/" + os.environ.get("KT_PED_DIR", "results/ednet_kt3_trackB_v3"); P = f"{V3}/pedagogy"; SEED = 42
STEM = os.environ.get("KT_PED_STEM", f"v3_seed{SEED}"); PFX = os.environ.get("KT_PED_PREFIX", "pedagogy_v3_"); DSN = os.environ.get("KT_PED_NAME", "EdNet-KT3")
CKPT = f"{V3}/checkpoints/{STEM}.resume.pt"; METRICS = f"{V3}/{STEM}_metrics.json"
# With the trained model file: KT probabilities are computed for train / validation / test learners (recommender fit on train learners).
# Without it: the predictions saved by the V3 seed-42 run (validation and test learners) are used (recommender fit on validation learners).
HAVE_MODEL = os.environ.get("KT_PED_USE_PREDICTIONS") != "1" and (os.path.exists(CKPT) or os.path.exists(f"{P}/kt_prob_train.npz"))
NAMES = ("train", "validation", "test") if HAVE_MODEL else ("validation", "test")
def ram_gb():
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo")); kb = lambda key: float(info[key].split()[0])
        return (kb("MemTotal") - kb("MemAvailable")) / 1e6
    except Exception: return float("nan")
def say(text): print(f"[{(time.time() - T0) / 60:6.1f} min | RAM {ram_gb():4.1f} GB] {text}", flush=True)
os.chdir(K); sys.path[:0] = [f"{K}/scripts", f"{K}/src"]; os.makedirs(P, exist_ok=True)
import yaml
try: import kaggle_run as kr          # needed only by the EdNet 'data'/'features' stages
except Exception: kr = None
from adaptive_kt.utils.tableio import read_table, write_table
cfg = yaml.safe_load(open(f"{K}/" + os.environ.get("KT_PED_CFG", "configs/ednet_kt3_trackB.yaml"))); seq = cfg["sequence"]; ds_cfg = cfg["dataset"]; ped_cfg = dict(cfg["pedagogy"]); ex_cfg = cfg["explainability"]
SPLIT_CSV = f"{K}/" + os.environ.get("KT_PED_SPLIT", "results/ednet_kt3_trackB/split_assignments.csv")
FEAT = Path(f"{K}/" + os.environ.get("KT_PED_FEATURES", "data/processed/ednet_kt3/features_small"))
def shard_files(): return sorted(kr.INTERACTIONS.glob("part-*.parquet")) + sorted(kr.INTERACTIONS.glob("part-*.pkl"))

def build_shard(args):
    """Learner-state features of one shard, built with the project's own causal feature code."""
    from adaptive_kt.data.features import build_causal_features
    from adaptive_kt.data.schema import ALL_FEATURE_COLUMNS
    source, target, window, min_history, max_seconds = args
    frame = build_causal_features(read_table(Path(source)), recent_window=window, min_history=min_history, max_response_time_seconds=max_seconds)
    if len(frame): write_table(frame[["user_id", "timestamp", "item_id", "correct"] + list(ALL_FEATURE_COLUMNS)], Path(target))
    Path(target + ".done").write_text(str(len(frame)))
    return len(frame)

# ------------------------------------------------------------------ stage: data
if STAGE == "data":
    if not (kr.INTERACTIONS / "_DONE").exists(): kr.prepare(kr.find_raw_kt3())
    kr.make_split()
    say("data and split ready")

# ------------------------------------------------------------------ stage: features
elif STAGE == "features":
    FEAT.mkdir(parents=True, exist_ok=True); shards = shard_files()
    jobs = [(str(s), str(FEAT / s.stem), int(ds_cfg.get("recent_window", 10)), int(ds_cfg.get("min_history", 1)), float(ds_cfg.get("max_response_time_seconds", 1200)))
            for s in shards if not (FEAT / (s.stem + ".done")).exists()]
    if jobs:
        from multiprocessing import Pool
        say(f"building learner-state features for {len(jobs)} of {len(shards)} shards with {os.cpu_count()} processes...")
        with Pool(os.cpu_count() or 2) as pool:
            for done, rows in enumerate(pool.imap_unordered(build_shard, jobs, chunksize=1), start=1):
                if done % 25 == 0 or done == len(jobs): say(f"  features: {done}/{len(jobs)} shards")
    say(f"feature table ready: {sum(int((FEAT / (s.stem + '.done')).read_text()) for s in shards):,} rows (original machine: 16,488,101)")

# ------------------------------------------------------------------ stage: kt
elif STAGE == "kt" and not HAVE_MODEL:
    from sklearn.metrics import roc_auc_score
    saved = json.load(open(METRICS))
    for name in NAMES:
        source = f"{V3}/predictions/{STEM}_{name}.npz"
        if not os.path.exists(source): raise SystemExit(f"saved V3 predictions not found: {source}")
        z = np.load(source); y_true, y_prob = z["y_true"], z["y_prob"]; auc = float(roc_auc_score(y_true, y_prob))
        if abs(auc - float(saved[name]["auc"])) >= 5e-4: raise SystemExit(f"{name}: saved predictions give AUC {auc:.4f} but the V3 result file says {saved[name]['auc']:.4f}. Stopped.")
        np.savez(f"{P}/kt_prob_{name}.npz", y_true=y_true.astype(np.uint8), y_prob=y_prob.astype(np.float32))
        say(f"KT probabilities, {name}: {len(y_prob):,} rows taken from the saved run {STEM}, AUC {auc:.4f} (same as the reported result: OK)")
elif STAGE == "kt":
    # >>> KT
    todo = [name for name in NAMES if not os.path.exists(f"{P}/kt_prob_{name}.npz")]
    if not todo: say("KT probabilities already saved for train / validation / test")
    else:
        import torch
        from sklearn.metrics import roc_auc_score
        spec = importlib.util.spec_from_file_location("rv3", f"{K}/scripts/run_trackB_proposed_v3.py"); rv3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv3)
        rv3.seed_everything(SEED); device = rv3.torch_device()
        if device.type != "cuda": raise SystemExit("No GPU: Settings -> Accelerator -> GPU T4 x2, then run again")
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        if state.get("status") != "complete": raise SystemExit("V3 seed 42 is not a finished run: finish the seeds cell first")
        say("loading interactions ...")
        split = rv3.load_split(Path(ds_cfg["processed_root"]), Path(SPLIT_CSV), 0)
        skill_vocab = rv3.SequenceVocabulary.fit(split["train"]["skill_id"]); item_vocab = rv3.SequenceVocabulary.fit(split["train"]["item_id"])
        source_vocab = rv3.fit_category_vocab(split["train"], "source"); platform_vocab = rv3.fit_category_vocab(split["train"], "platform")
        part_of, tags_of, num_tags = rv3.load_question_meta(ds_cfg["raw_root"]); item_part, item_tags = rv3.item_lookup(item_vocab, part_of, tags_of)
        cache = Path(V3) / "cache"; cache.mkdir(parents=True, exist_ok=True)
        def dataset(name):
            return rv3.KTSequenceDatasetV3(split[name], skill_vocab, item_vocab, source_vocab, platform_vocab, max_len=int(seq["max_len"]),
                                           min_history=int(ds_cfg.get("min_history", 1)), split_name=name, cache_path=cache / f"{name}.npz")
        model = rv3.BehaviorAwareKTV3(
            num_skills=skill_vocab.size, num_items=item_vocab.size, num_sources=source_vocab.size, num_platforms=platform_vocab.size, num_tags=num_tags,
            item_part=torch.from_numpy(item_part), item_tags=torch.from_numpy(item_tags), max_len=int(seq["max_len"]) + 1,
            embedding_dim=int(seq["embedding_dim"]), num_heads=int(seq["transformer_heads"]), num_layers=int(seq["transformer_layers"]), dropout=float(seq["dropout"]))
        model.load_state_dict(state["best_state_dict"]); model.to(device).eval(); saved = json.load(open(METRICS)); del state
        say(f"loaded V3 seed {SEED}; building the train sequence cache (about 20-30 min, no output) ...")
        data = dataset("train"); split.pop("train"); gc.collect()
        easiness_full, easiness_oof, fold = rv3.fit_item_easiness(data.items, data.correct, data.first_rows(), item_vocab.size)
        for name in NAMES:                                           # train first, then it is freed before validation and test are built
            if name != "train": data = dataset(name); split.pop(name); gc.collect()
            if name in todo:
                ease = easiness_oof[fold, data.items] if name == "train" else easiness_full[data.items]   # same rule as in training
                rv3.attach_mastery(data, ease, item_part, item_tags, cache / f"{name}_mastery_v1.npy")
                y_true, y_prob = rv3.kt_trainer.predict_kt(model, rv3.make_loader(data, int(seq["batch_size"]), False, SEED, 0), device)
                auc = float(roc_auc_score(y_true, y_prob)); note = ""
                if name in saved:
                    gap = abs(auc - float(saved[name]["auc"])); note = f" | saved run: {saved[name]['auc']:.4f} ({'same model: OK' if gap < 5e-4 else 'DIFFERENT'})"
                    if gap >= 5e-4: raise SystemExit(f"{name} AUC {auc:.4f} does not match the saved V3 run {saved[name]['auc']:.4f}: wrong checkpoint or data. Stopped.")
                say(f"KT probabilities, {name}: {len(y_prob):,} rows, AUC {auc:.4f}{note}")
                np.savez(f"{P}/kt_prob_{name}.tmp.npz", y_true=y_true.astype(np.uint8), y_prob=y_prob.astype(np.float32)); os.replace(f"{P}/kt_prob_{name}.tmp.npz", f"{P}/kt_prob_{name}.npz")
            del data; gc.collect()
    # <<< KT

# ------------------------------------------------------------------ stage: final
elif STAGE == "final":
    from adaptive_kt.data.schema import feature_columns
    from adaptive_kt.evaluation.metrics import recommendation_metrics
    from adaptive_kt.models.pedagogy import PedagogyRecommender, assign_pedagogy_strategy
    # join features with KT probabilities: row counts and every target must match
    assign = pd.read_csv(SPLIT_CSV, dtype={"user_id": str, "split": str}); split_of = dict(zip(assign["user_id"], assign["split"])); del assign
    buckets = {name: [] for name in NAMES}
    for path in sorted(FEAT.glob("part-*.parquet")) + sorted(FEAT.glob("part-*.pkl")):
        table = read_table(path); table["user_id"] = table["user_id"].astype(str); where = table["user_id"].map(split_of).to_numpy()
        for name in NAMES:
            piece = table[where == name]
            if len(piece): buckets[name].append(piece)
    ped_features = list(feature_columns(ds_cfg.get("feature_set", "core"))) + ["kt_probability"]
    from adaptive_kt.data.schema import PEDAGOGY_STRATEGIES
    classes = sorted(PEDAGOGY_STRATEGIES); rel_cols = ["rel_" + c for c in classes]; k = int(ped_cfg.get("top_k", 3))
    TRAJ = ["topic_mastery", "recent_accuracy", "historical_accuracy", "kt_probability"]
    rich_features = ped_features + [f"{p}_{c}" for c in TRAJ for p in ("prev", "delta")] + [f"policy_{c}" for c in classes]
    def sample(frame, limit, seed): return frame if not limit or len(frame) <= limit else frame.sample(n=int(limit), random_state=seed).sort_index()
    def ranked_multilabel_metrics(relevance, probabilities, k=3):
        relevance = np.asarray(relevance, dtype=bool); probabilities = np.asarray(probabilities, dtype=float)
        order = np.argsort(-probabilities, axis=1, kind="stable")[:, :k]; rel = np.take_along_axis(relevance, order, axis=1).astype(float)
        n_rel = relevance.sum(axis=1).clip(min=1); discounts = 1.0 / np.log2(np.arange(2, k + 2)); best = np.cumsum(discounts)[np.minimum(n_rel, k) - 1]
        return {f"precision@{k}": float((rel.sum(axis=1) / k).mean()), f"recall@{k}": float((rel.sum(axis=1) / n_rel).mean()),
                f"ndcg@{k}": float(((rel * discounts).sum(axis=1) / best).mean()), f"hit@{k}": float((rel.sum(axis=1) > 0).mean()), "top1_relevance": float(rel[:, 0].mean())}
    def future_relevance(frame, classes):
        """Which strategies fit the learner's state at the NEXT step (same thresholds as the policy; several can be relevant)."""
        cols = ["topic_mastery", "recent_accuracy", "mean_attempt_count", "error_rate", "kt_probability"]
        nxt = frame.groupby("user_id", sort=False)[cols].shift(-1); valid = nxt.notna().all(axis=1).to_numpy(); f = nxt.loc[valid]
        mastery, recent, attempts, error, kt = (f[c].to_numpy(float) for c in cols)
        rules = {"challenge_based_learning": (mastery >= 0.80) & (recent >= 0.80) & (kt >= 0.75), "remedial_learning": (mastery < 0.35) & (recent < 0.50) & (kt < 0.50),
                 "example_based_learning": (mastery < 0.55) | (kt < 0.55), "step_by_step_guidance": (error >= 0.45) | (attempts >= 1.50),
                 "practice_heavy_learning": ((recent >= 0.50) & (recent < 0.80)) | ((kt >= 0.50) & (kt < 0.75))}
        rel = np.column_stack([rules[c] for c in classes]); rel[~rel.any(axis=1), classes.index("practice_heavy_learning")] = True
        return valid, rel
    parts = {}; counts = {}
    for name in NAMES:
        part = pd.concat(buckets.pop(name), ignore_index=True).sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
        kt = np.load(f"{P}/kt_prob_{name}.npz")
        if len(part) != len(kt["y_prob"]): raise SystemExit(f"{name}: {len(part):,} feature rows but {len(kt['y_prob']):,} KT rows. Stopped.")
        wrong = int(np.sum(part["correct"].to_numpy(dtype=int) != kt["y_true"].astype(int)))
        if wrong: raise SystemExit(f"{name}: {wrong:,} rows do not line up between features and KT targets. Stopped.")
        part["kt_probability"] = kt["y_prob"].astype("float32"); part["pedagogy_strategy"] = assign_pedagogy_strategy(part); counts[name] = int(len(part))
        has_next, rel = future_relevance(part, classes); full = np.zeros((len(part), len(classes)), dtype=bool); full[has_next] = rel   # on complete learner sequences, before any sampling
        part["has_next"] = has_next
        for i, col in enumerate(rel_cols): part[col] = full[:, i]
        # learner trajectory: the previous state of the same learner (known before this step) and the change since then
        prev = part.groupby("user_id", sort=False)[TRAJ].shift(1)
        for c in TRAJ:
            before = prev[c].fillna(part[c]); part[f"prev_{c}"] = before.astype("float32"); part[f"delta_{c}"] = (part[c] - before).astype("float32")
        for c in classes: part[f"policy_{c}"] = (part["pedagogy_strategy"].to_numpy() == c).astype("float32")   # rule-based policy as prior knowledge
        say(f"{name}: {len(part):,} rows aligned with KT probabilities")
        if HAVE_MODEL and name == "train": part = sample(part, ped_cfg.get("max_train_rows"), SEED + 11)
        if HAVE_MODEL and name == "validation": part = sample(part, ped_cfg.get("max_validation_rows"), SEED + 12)
        parts[name] = part; gc.collect()
    test = parts["test"]
    if HAVE_MODEL:
        train_fit, val_fit = parts["train"], parts["validation"]
        protocol = "recommenders fit on train learners, early stopping on validation learners, evaluated on test learners"
    else:                                                            # learner-disjoint 80/20 split of the validation learners
        users = np.array(sorted(parts["validation"]["user_id"].unique())); order = np.random.RandomState(SEED).permutation(len(users))
        fit_users = set(users[order[: int(round(0.8 * len(users)))]]); in_fit = parts["validation"]["user_id"].isin(fit_users).to_numpy()
        train_fit = sample(parts["validation"][in_fit], ped_cfg.get("max_train_rows"), SEED + 11)
        val_fit = sample(parts["validation"][~in_fit], ped_cfg.get("max_validation_rows"), SEED + 12)
        protocol = "recommenders fit on 80% of the validation learners, early stopping on the other 20%, evaluated on test learners (the KT model was never trained on any of these learners; validation learners were used only for its early stopping)"
    say("protocol: " + protocol)

    # Recommender designs (same XGBoost model and settings). Every design is scored on held-out VALIDATION learners; test is used once.
    #   proposed                     : trained on the strategies the learner needs at the NEXT step; inputs = behaviour-aware state + KT probability
    #                                  + learner trajectory (previous state and change) + the rule-based policy's current strategy as prior knowledge
    #   ablation_combined_target     : same inputs, trained on current-state policy label + next-step strategies (earlier design)
    #   ablation_current_state_only  : same inputs, trained on the policy's current-state label only (imitates the rule)
    #   ablation_basic_inputs        : trained on next-step strategies, but only the basic state + KT probability (no trajectory, no policy prior)
    def pairs(frame, now, nxt, cols):
        xs, ys = [], []
        if now: xs.append(frame[cols]); ys.append(pd.Series(frame["pedagogy_strategy"].to_numpy(dtype=object)))
        if nxt:
            usable = frame[frame["has_next"].to_numpy()]; rows, picked = np.nonzero(usable[rel_cols].to_numpy())
            xs.append(usable[cols].iloc[rows]); ys.append(pd.Series(np.asarray(classes, dtype=object)[picked]))
        return pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True)
    DESIGNS = {"proposed": (False, True, rich_features), "ablation_combined_target": (True, True, rich_features),
               "ablation_current_state_only": (True, False, rich_features), "ablation_basic_inputs": (False, True, ped_features)}
    FEATS = {name: cols for name, (_, _, cols) in DESIGNS.items()}
    models, prob, prob_hold, priors = {}, {}, {}, {}
    for name, (now, nxt, cols) in DESIGNS.items():
        x_fit, y_fit = pairs(train_fit, now, nxt, cols); x_stop, y_stop = pairs(val_fit, now, nxt, cols)
        priors[name] = y_fit.astype(str).value_counts(normalize=True).reindex(classes).fillna(0.0).clip(lower=1e-6).to_numpy()   # how often each strategy is a training target
        model = PedagogyRecommender(ped_cfg, SEED)
        if model.encoder.classes_.tolist() != classes: raise SystemExit("strategy order changed. Stopped.")
        say(f"training '{name}' on {len(x_fit):,} (learner state, target strategy) examples, {len(cols)} inputs ...")
        model.fit(x_fit, y_fit, x_stop, y_stop); del x_fit, y_fit, x_stop, y_stop; gc.collect()
        models[name] = model; prob[name] = model.predict_proba(test[cols]); prob_hold[name] = model.predict_proba(val_fit[cols])

    # Class balance (logit adjustment): rare strategies such as remedial are almost never ranked first by a model trained on
    # imbalanced targets. Scores are divided by prior**tau. tau is chosen on held-out VALIDATION learners only: the largest
    # per-strategy (macro) recall averaged over K=1..k, among the tau values that keep validation NDCG@k within 0.01 of tau=0.
    # The same tau is applied to every learned design, so the comparison between designs stays fair.
    def adjust(p, prior, tau):
        q = np.asarray(p, dtype=float) / np.power(prior, tau); return q / q.sum(axis=1, keepdims=True)
    hold_ok = val_fit["has_next"].to_numpy(); hold_rel = val_fit[rel_cols].to_numpy()[hold_ok]
    def macro_recall(p, rel):
        out = []
        for kk in range(1, k + 1):
            order = np.argsort(-p, axis=1, kind="stable")[:, :kk]; shown = np.zeros_like(rel); np.put_along_axis(shown, order, True, axis=1)
            out.append(np.mean([(shown[:, i] & rel[:, i]).sum() / rel[:, i].sum() for i in range(rel.shape[1]) if rel[:, i].any()]))
        return float(np.mean(out))
    TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]; tau_table = []
    for tau in TAUS:
        p = adjust(prob_hold["proposed"], priors["proposed"], tau)[hold_ok]
        tau_table.append({"tau": tau, f"validation_ndcg@{k}": ranked_multilabel_metrics(hold_rel, p, k)[f"ndcg@{k}"], "validation_macro_recall": macro_recall(p, hold_rel)})
    base_ndcg = tau_table[0][f"validation_ndcg@{k}"]
    allowed = [r for r in tau_table if r[f"validation_ndcg@{k}"] >= base_ndcg - 0.01]
    TAU = max(allowed, key=lambda r: r["validation_macro_recall"])["tau"]
    for r in tau_table: say(f"class balance tau={r['tau']:.2f}: validation NDCG@{k}={r[f'validation_ndcg@{k}']:.4f}, macro recall={r['validation_macro_recall']:.4f}" + ("  <- chosen" if r["tau"] == TAU else ""))
    for name in DESIGNS:
        prob[name] = adjust(prob[name], priors[name], TAU); prob_hold[name] = adjust(prob_hold[name], priors[name], TAU)

    # two baselines that give every learner the same list
    fit_next = train_fit[train_fit["has_next"].to_numpy()]
    prior_relevant = fit_next[rel_cols].to_numpy().mean(axis=0)                                                   # most often relevant at the next step
    prior_labels = pd.Series(train_fit["pedagogy_strategy"]).value_counts(normalize=True).reindex(classes).fillna(0.0).to_numpy()  # most common policy labels
    def constant(frame, scores): return np.tile(scores, (len(frame), 1))
    for name, scores in (("baseline_most_often_relevant", prior_relevant), ("baseline_most_common_labels", prior_labels)):
        prob[name] = constant(test, scores); prob_hold[name] = constant(val_fit, scores)
    METHODS = list(prob); BASELINES = METHODS[len(DESIGNS):]
    def evaluate(frame, probabilities, top=None):
        ok = frame["has_next"].to_numpy(); return ranked_multilabel_metrics(frame[rel_cols].to_numpy()[ok], probabilities[ok], top or k)
    on_test = {m: evaluate(test, prob[m]) for m in METHODS}; on_holdout = {m: evaluate(val_fit, prob_hold[m]) for m in METHODS}
    selected = max(DESIGNS, key=lambda m: on_holdout[m][f"ndcg@{k}"])                                             # design choice on validation only
    say(f"best design on held-out validation learners (NDCG@{k}): {selected}" + ("" if selected == "proposed" else "  <- NOT the proposed design; report this"))
    shorter = {kk: {m: evaluate(test, prob[m], kk) for m in METHODS} for kk in range(1, k)}                       # the same table for shorter lists
    names = list(on_test["proposed"])
    wins = [m for m in names if all(on_test["proposed"][m] > on_test[b][m] for b in BASELINES)]
    valid = test["has_next"].to_numpy(); relevance = test[rel_cols].to_numpy()[valid]; policy = test["pedagogy_strategy"].to_numpy()
    top1 = {m: np.asarray(classes, dtype=object)[np.argmax(prob[m], axis=1)] for m in METHODS}
    top1_share = {m: {c: float(np.mean(top1[m] == c)) for c in classes} for m in METHODS}
    policy_share = {c: float(np.mean(policy == c)) for c in classes}
    agreement = {m: float(np.mean(top1[m] == policy)) for m in METHODS}                                           # top-1 equals the policy's strategy for the current state
    def recall_by_strategy(probabilities):                                                                        # when a strategy is relevant at the next step, is it in the top-k list?
        order = np.argsort(-probabilities[valid], axis=1, kind="stable")[:, :k]; shown = np.zeros_like(relevance); np.put_along_axis(shown, order, True, axis=1)
        return {c: (float((shown[:, i] & relevance[:, i]).sum() / relevance[:, i].sum()) if relevance[:, i].any() else float("nan")) for i, c in enumerate(classes)}
    by_strategy = {m: recall_by_strategy(prob[m]) for m in METHODS}

    # explainability on a fixed test sample: the shap library first; if it fails with this XGBoost version, XGBoost's own TreeSHAP values
    try: import shap
    except Exception: sys.modules["shap"] = types.ModuleType("shap"); sys.modules["shap"].TreeExplainer = None
    from adaptive_kt.explain import shap_tools
    class NativeTreeShap:
        def __init__(self, model): self.model = model
        def shap_values(self, x):
            import xgboost as xgb
            best = getattr(self.model, "best_iteration", None); extra = {"iteration_range": (0, int(best) + 1)} if best is not None else {}
            values = np.asarray(self.model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True, **extra))   # rows x classes x (features + 1)
            return np.moveaxis(values[:, :, :-1], 1, 2)                                                            # rows x features x classes
    sample_index = test.sample(n=min(int(ex_cfg.get("sample_size", 1000)), len(test)), random_state=SEED).index; top = int(ex_cfg.get("top_features", 3))
    backend = ["shap.TreeExplainer"]
    def explain(model, cols):
        rows = test.loc[sample_index, cols]
        try: model.get_booster().set_param({"device": "cpu"})
        except Exception: pass
        def run():
            importance, _ = shap_tools.tree_shap_importance(model, rows)
            return importance, shap_tools.explanation_quality(model, rows, top_features=top, seed=SEED), shap_tools.local_shap_reasons(model, rows, class_names=classes, top_features=top)
        try: return run()
        except Exception as exc:
            say(f"shap library failed ({type(exc).__name__}: {str(exc)[:120]}); using XGBoost's own TreeSHAP values instead")
            shap_tools.shap.TreeExplainer = NativeTreeShap; backend[0] = "XGBoost pred_contribs (TreeSHAP)"; return run()
    explained = {m: explain(models[m].model, FEATS[m]) for m in DESIGNS}
    importance, quality, reasons = explained["proposed"]

    # save + print
    LABEL = {"proposed": "PROPOSED recommender", "ablation_combined_target": "ablation: combined target", "ablation_current_state_only": "ablation: current state only",
             "ablation_basic_inputs": "ablation: basic inputs only",
             "baseline_most_often_relevant": "baseline: most often relevant", "baseline_most_common_labels": "baseline: most common labels"}
    result = {"kt_model": f"{DSN}: {STEM}, frozen 70/10/20 split", "protocol": protocol, "rows": counts, "recommender_fit_rows": {"train": int(len(train_fit)), "validation": int(len(val_fit))},
              "methods": {"proposed": "XGBoost trained on the strategies relevant at the next step; inputs: behaviour-aware learner state, KT probability, learner trajectory (previous state and change), rule-based policy strategy as prior",
                          "ablation_combined_target": "same inputs, trained on the current-state policy label plus next-step strategies",
                          "ablation_current_state_only": "same inputs, trained on the policy's current-state strategy only",
                          "ablation_basic_inputs": "trained on next-step strategies with the basic learner state and KT probability only",
                          "baseline_most_often_relevant": "same list for every learner: strategies most often relevant at the next step", "baseline_most_common_labels": "same list for every learner: most common policy labels"},
              "next_step_relevance_test": dict(on_test, rows_evaluated=int(valid.sum()), rows_without_next_step=int((~valid).sum()), mean_relevant_strategies_per_row=float(relevance.sum(axis=1).mean()),
                                               note="Relevance comes from the learner state at the next step (rule-based proxy). Not causal evidence."),
              "next_step_relevance_test_shorter_lists": {f"K={kk}": shorter[kk] for kk in shorter},
              "next_step_relevance_heldout_validation_learners": on_holdout,
              "proposed_beats_both_baselines_on_test": wins, "design_selected_on_validation": selected,
              "proposed_beats_every_ablation_on_test": [m for m in names if all(on_test["proposed"][m] > on_test[a][m] for a in DESIGNS if a != "proposed")],
              "top1_share_by_strategy_test": dict(top1_share, policy_for_current_state=policy_share),
              "share_of_test_rows_where_strategy_is_relevant_next_step": {c: float(relevance[:, i].mean()) for i, c in enumerate(classes)},
              f"recall_at_{k}_by_strategy_test": by_strategy,
              "top1_agreement_with_policy_test": dict(agreement, note="Agreement with the rule-based policy shows consistency with it, not educational effectiveness."),
              "explainability": dict({m: dict(explained[m][1], top_shap_features=explained[m][0].head(6)["feature"].tolist()) for m in DESIGNS}, backend=backend[0], sample_rows=int(len(sample_index)), top_features=top),
              "xgboost_best_iteration": {m: int(getattr(models[m].model, "best_iteration", -1) or -1) for m in DESIGNS},
              "class_balance": {"tau_chosen_on_validation": TAU, "candidates": tau_table, "rule": f"largest macro recall (K=1..{k}) on held-out validation learners with NDCG@{k} within 0.01 of tau=0; same tau for every learned design"}}
    json.dump(result, open(f"{P}/{PFX}results.json", "w"), indent=2)
    importance.to_csv(f"{P}/{PFX}shap_importance.csv", index=False); reasons.to_csv(f"{P}/{PFX}local_explanations.csv", index=False)
    plain = lambda d: {key.split("@")[0]: value for key, value in d.items()}
    table = pd.DataFrame([dict(evaluated_on="test learners", K=k, method=m, **plain(on_test[m])) for m in METHODS]
                         + [dict(evaluated_on="test learners", K=kk, method=m, **plain(shorter[kk][m])) for kk in shorter for m in METHODS]
                         + [dict(evaluated_on="held-out validation learners", K=k, method=m, **plain(on_holdout[m])) for m in METHODS])
    table.to_csv(f"{P}/{PFX}next_step_table.csv", index=False)
    pd.DataFrame([dict(method=m, measure="top-1 share", **top1_share[m]) for m in METHODS] + [dict(method="policy for the current state", measure="share of rows", **policy_share)]
                 + [dict(method=m, measure=f"recall@{k} when relevant at next step", **by_strategy[m]) for m in METHODS]).to_csv(f"{P}/{PFX}by_strategy.csv", index=False)
    try: models["proposed"].model.save_model(f"{P}/{PFX}recommender.json")
    except Exception: pass
    for name in ("results.json", "next_step_table.csv", "by_strategy.csv", "shap_importance.csv", "local_explanations.csv"): shutil.copy(f"{P}/{PFX}{name}", f"{OUT}/{PFX}{name}")
    short = lambda key: key.replace("precision", "P").replace("recall", "R").replace("top1_relevance", "top-1")
    def block(title, scores, keys):
        print("\n" + f"{title:30s}" + "".join(f"{short(key):>9s}" for key in keys))
        for m in METHODS: print(f"{LABEL[m]:30s}" + "".join(f"{scores[m][key]:9.4f}" for key in keys))
    print(f"\n===== PEDAGOGY + EXPLAINABILITY with the final model ({DSN}: {STEM}) =====")
    print("protocol:", protocol)
    print(f"class balance chosen on validation: tau={TAU} (0 = no balancing)")
    print(f"test learners: {int(valid.sum()):,} rows with a next step, {relevance.sum(axis=1).mean():.2f} relevant strategies per row")
    block(f"TEST, top-{k} list", on_test, names)
    print("PROPOSED beats BOTH baselines on:", ", ".join(wins) or "nothing", "| not on:", ", ".join(m for m in names if m not in wins) or "nothing")
    for kk in shorter:
        keys = [f"precision@{kk}", f"recall@{kk}", f"ndcg@{kk}"]; block(f"TEST, top-{kk} list", shorter[kk], keys)
        print("PROPOSED beats BOTH baselines on:", ", ".join(key for key in keys if all(shorter[kk]["proposed"][key] > shorter[kk][b][key] for b in BASELINES)) or "nothing")
    block(f"HELD-OUT VALIDATION, top-{k}", on_holdout, names)
    print(f"design chosen on validation: {selected} | PROPOSED beats every ablation on test on: " + (", ".join(result["proposed_beats_every_ablation_on_test"]) or "nothing"))
    tag = [c.replace("_learning", "").replace("_guidance", "").replace("_based", "")[:12] for c in classes]
    print("\n" + f"{'share of top-1 recommendations':30s}" + "".join(f"{t:>14s}" for t in tag))
    for m in DESIGNS: print(f"{LABEL[m]:30s}" + "".join(f"{top1_share[m][c]:14.4f}" for c in classes))
    print(f"{'policy for the current state':30s}" + "".join(f"{policy_share[c]:14.4f}" for c in classes))
    print("\n" + f"{'in top-' + str(k) + ' when relevant next':30s}" + "".join(f"{t:>14s}" for t in tag))
    for m in METHODS: print(f"{LABEL[m]:30s}" + "".join(f"{by_strategy[m][c]:14.4f}" for c in classes))
    print("\ntop-1 equals the policy strategy :", "  ".join(f"{LABEL[m]}={agreement[m]:.4f}" for m in DESIGNS), " <- consistency with the policy, not effectiveness")
    for m in DESIGNS:
        print(f"explainability, {LABEL[m]:28s}: fidelity={explained[m][1]['fidelity_probability_drop']:.4f}  consistency={explained[m][1]['consistency_jaccard']:.4f}  top features: "
              + ", ".join(f"{r.feature} ({r.mean_abs_shap:.3f})" for r in explained[m][0].head(6).itertuples()))
    print(f"[explanations computed with {backend[0]}]", flush=True)
else:
    raise SystemExit(f"unknown stage {STAGE}")
