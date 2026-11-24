#!/usr/bin/env python3
"""Paper figures from the finished results (read-only). One chart per file, no titles (only axis labels / legend),
same style everywhere, 600 dpi PNG + vector PDF. Output: ~/Downloads/kt_figures/"""
import glob, io, json, os, re, zipfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HOME = os.path.expanduser("~"); ROOT = os.path.expanduser(os.environ.get("KT_PROJECT", "~/Downloads/adaptive-kt-pedagogy-final"))
OUT = os.path.expanduser(os.environ.get("KT_FIGS", "~/Downloads/kt_figures")); os.makedirs(OUT, exist_ok=True)
for old in glob.glob(f"{OUT}/fig*.png") + glob.glob(f"{OUT}/fig*.pdf"): os.remove(old)   # old versions removed: no duplicates
DPI = int(os.environ.get("KT_DPI", "600"))
BLUE, ORANGE, GREEN, YELLOW, PURPLE, GREY, DARK = "#1f5fa8", "#d9622b", "#2e9e6b", "#d9a300", "#7b5ea7", "#a3a29c", "#222222"
MODEL_COLOR = {"Proposed": BLUE, "HGAKT (base paper)": ORANGE, "AKT": GREEN, "DKT": YELLOW, "Same inputs + XGBoost": PURPLE}
SIZE, SIZE_TALL = (4.2, 3.0), (4.2, 3.6)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "axes.edgecolor": DARK, "axes.labelcolor": DARK,
                     "xtick.color": DARK, "ytick.color": DARK, "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
KT3_FOLDERS = ("ednet_kt3_trackB_v3", "ednet_kt3_trackB_baselines", "ednet_kt3_kaggle_baselines", "hgakt_kt3_trackB", "ednet_kt3_trackB_samefeat", "kt3_final")

# ---------------------------------------------------------------- files: project results + ~/Downloads (also inside the result zips)
FILES = {}
for base in (f"{ROOT}/results", f"{HOME}/Downloads"):
    for cur, dirs, files in os.walk(base):
        if os.path.abspath(cur) == os.path.abspath(f"{HOME}/Downloads"): dirs[:] = []           # only the top of ~/Downloads
        dirs[:] = [d for d in dirs if d not in ("cache", "checkpoints", "restart_checkpoints", "kt_figures")]
        for name in files:
            path = os.path.join(cur, name)
            if name.endswith(".zip") and name.startswith(("kt_assist17", "kt3_final", "kt_numbers", "kt_resume")):
                try:
                    z = zipfile.ZipFile(path)
                    for n in z.namelist(): FILES.setdefault(n, (z, n))
                except Exception: pass
            elif name.endswith((".csv", ".json", ".npz")): FILES.setdefault(path, (None, path))
def open_file(key):
    z, n = FILES[key]; return io.BytesIO(z.read(n)) if z else open(n, "rb")
def is_ds(key, ds):
    return "assist2017" in key if ds == "assist" else ("assist2017" not in key and any(f in key for f in KT3_FOLDERS))
def find(suffix, ds=None):
    hits = sorted((k for k in FILES if k.endswith(suffix) and (ds is None or is_ds(k, ds))),        # newest file on disk wins over zip copies
                  key=lambda k: (FILES[k][0] is None, os.path.getmtime(k) if FILES[k][0] is None else 0, k))
    return open_file(hits[-1]) if hits else None
def csv(suffix, ds=None):
    f = find(suffix, ds); return pd.read_csv(f) if f is not None else None
made, missing = [], []
SKIP = ("fig03_rmse", "fig06_prediction", "fig11_seed", "fig13_pedagogy_precision", "fig13_pedagogy_recall")
NUMBER = {"fig01": "fig01", "fig02": "fig02", "fig04": "fig03", "fig05": "fig04", "fig07": "fig05", "fig08": "fig06", "fig09": "fig07", "fig10": "fig08",
          "fig12": "fig09", "fig13": "fig10", "fig14": "fig11", "fig15": "fig12"}
INTERNAL = [(r"\s*\((?:V3|v3|dkt_f|DKT-F|samefeat)\)", ""), (r"\b(?:V3|v3)\b\s*", ""), (r"\bdkt_f\b|\bDKT-F\b", "LSTM"),
            (r"_seed\d+", ""), (r"\s{2,}", " ")]
def clean(s):
    """Remove internal experiment names (V3, dkt_f, ...) from every label a reader sees."""
    s = str(s)
    for pat, rep in INTERNAL: s = re.sub(pat, rep, s)
    return s.strip()
def save(fig, name):
    if name.startswith(SKIP): plt.close(fig); return
    name = NUMBER[name[:5]] + name[5:]
    for a in fig.axes:
        a.set_title(""); a.set_xlabel(clean(a.get_xlabel())); a.set_ylabel(clean(a.get_ylabel()))
        fig.canvas.draw()
        for axis in (a.xaxis, a.yaxis):
            old = [t.get_text() for t in axis.get_majorticklabels()]; cleaned = [clean(t) for t in old]
            if cleaned != old: axis.set_ticks(axis.get_majorticklocs()); axis.set_ticklabels(cleaned)
        leg = a.get_legend()
        for t in (leg.get_texts() if leg else []) + list(a.texts): t.set_text(clean(t.get_text()))
    fig.suptitle("")
    for ext in ("png", "pdf"): fig.savefig(f"{OUT}/{name}.{ext}", dpi=DPI)
    plt.close(fig); made.append(name)
def legend_below(ax, ncol=2):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, handlelength=1.8, columnspacing=1.2)
def short(m):
    return {"Proposed (final)": "Proposed", "HGAKT (base paper, our adaptation)": "HGAKT (base paper)", "Same inputs as Proposed + XGBoost": "Same inputs + XGBoost",
            "Logistic Regression": "Logistic regression", "Random Forest": "Random forest"}.get(m, m)

DS = {"kt3": dict(name="EdNet-KT3", main="kt3_main_table.csv", abl="kt3_ablation_final.csv", sel="kt3_selection.csv", seljson="kt3_final/selection.json",
                  data="data/processed/ednet_kt3/interactions", split="results/ednet_kt3_trackB/split_assignments.csv", ped="pedagogy_v3_"),
      "assist": dict(name="ASSISTments 2017", main="assist2017_main_table.csv", abl="assist2017_ablation_final.csv", sel="assist2017_selection.csv", seljson="assist2017/selection.json",
                     data="data/processed/assist2017/interactions", split="results/assist2017/split_assignments.csv", ped="assist2017_pedagogy_")}
STEMS = {"HGAKT (base paper)": "hgakt", "AKT": "akt", "DKT": "dkt_lstm", "Same inputs + XGBoost": "xgb_f"}
for ds, d in DS.items():
    d["final_stem"] = "v3" if ds == "kt3" else "dkt_f_no_experience_timing"
    f = find(d["seljson"])
    if f is not None:
        s = json.load(f); drops = sorted(x for x in s["final_drops"].split(",") if x)
        d["final_stem"] = s["final_prefix"] + ("_no_" + "_".join(drops) if drops else "")
    d["main_t"] = csv(d["main"]); d["abl_t"] = csv(d["abl"]); d["sel_t"] = csv(d["sel"])
    if d["main_t"] is None: missing.append(f"{d['name']}: {d['main']}")
def stem_of(ds, label): return DS[ds]["final_stem"] if label == "Proposed" else STEMS[label]
def predictions(ds, label, seed=42, part="test"):
    f = find(f"predictions/{stem_of(ds, label)}_seed{seed}_{part}.npz", ds)
    if f is None: return None
    p = np.load(f); return p["y_true"].astype(np.int8), p["y_prob"].astype(np.float64)
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def calibrated(ds, label):
    """Platt scaling fitted on the model's VALIDATION predictions, applied to its test predictions (same for every model;
    monotonic, so AUC, ranking and accuracy at the validation-chosen threshold do not change)."""
    from sklearn.linear_model import LogisticRegression
    test, val = predictions(ds, label), predictions(ds, label, part="validation")
    if test is None or val is None: return None
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(val[1]).reshape(-1, 1), val[0])
    return test[0], lr.predict_proba(logit(test[1]).reshape(-1, 1))[:, 1], test[1]
def ece(y, p, bins=10):
    b = np.clip((p * bins).astype(int), 0, bins - 1); return float(sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))
ACTIVE = [ds for ds in DS if DS[ds]["main_t"] is not None]

# ---------------------------------------------------------------- 1) test AUC of every model (mean of seeds, error bar = std)
for ds in ACTIVE:
    t = DS[ds]["main_t"].sort_values("test_auc"); labels = [short(m) for m in t["model"]]
    err = t["test_auc_std"].fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=SIZE_TALL)
    ax.barh(labels, t["test_auc"], color=[MODEL_COLOR.get(l, GREY) for l in labels], height=0.62, xerr=err, error_kw=dict(ecolor=DARK, lw=0.7, capsize=1.5))
    for y, (v, e) in enumerate(zip(t["test_auc"], err)): ax.text(v + e + 0.003, y, f"{v:.4f}", va="center", fontsize=7)
    ax.set_xlim(max(0.5, t["test_auc"].min() - 0.03), t["test_auc"].max() + 0.035); ax.set_xlabel("Test AUC"); ax.grid(axis="y", visible=False)
    save(fig, f"fig01_auc_all_models_{ds}")

# ---------------------------------------------------------------- 2) all metrics: proposed vs base paper vs strongest baselines
METRICS = [("test_auc", "AUC"), ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
for ds in ACTIVE:
    t = DS[ds]["main_t"].copy(); t["label"] = [short(m) for m in t["model"]]
    order = [l for l in ("Proposed", "HGAKT (base paper)", "AKT", "DKT") if l in set(t["label"])]; t = t.set_index("label").loc[order]
    fig, ax = plt.subplots(figsize=SIZE); w = 0.8 / len(order); x = np.arange(len(METRICS))
    for i, l in enumerate(order): ax.bar(x - 0.4 + w / 2 + i * w, [t.loc[l, k] for k, _ in METRICS], w * 0.92, color=MODEL_COLOR[l], label=l)
    vals = t[[k for k, _ in METRICS]].to_numpy(); ax.set_ylim(max(0, vals.min() - 0.08), min(1.0, vals.max() + 0.06))
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in METRICS]); ax.set_xlabel("Metric"); ax.set_ylabel("Score"); ax.grid(axis="x", visible=False)
    legend_below(ax, 4); save(fig, f"fig02_metrics_comparison_{ds}")

# ---------------------------------------------------------------- 3) RMSE of every model (lower is better)
for ds in ACTIVE:
    t = DS[ds]["main_t"].sort_values("rmse", ascending=False); labels = [short(m) for m in t["model"]]
    fig, ax = plt.subplots(figsize=SIZE_TALL)
    ax.barh(labels, t["rmse"], color=[MODEL_COLOR.get(l, GREY) for l in labels], height=0.62)
    for y, v in enumerate(t["rmse"]): ax.text(v + 0.001, y, f"{v:.4f}", va="center", fontsize=7)
    ax.set_xlim(t["rmse"].min() - 0.02, t["rmse"].max() + 0.012); ax.set_xlabel("Test RMSE (lower is better)"); ax.grid(axis="y", visible=False)
    save(fig, f"fig03_rmse_all_models_{ds}")

# ---------------------------------------------------------------- 4/5) ROC curves and calibration (reliability) curves
rng = np.random.default_rng(0)
for ds in ACTIVE:
    curves = [(l, p) for l in MODEL_COLOR if (p := predictions(ds, l)) is not None]
    if not curves: missing.append(f"{DS[ds]['name']}: test predictions (ROC, calibration)"); continue
    fig, ax = plt.subplots(figsize=SIZE)
    for l, (y, prob) in curves:
        if len(y) > 800_000: idx = rng.choice(len(y), 800_000, replace=False); y, prob = y[idx], prob[idx]
        fpr, tpr, _ = roc_curve(y, prob); ax.plot(fpr, tpr, color=MODEL_COLOR[l], label=f"{l} ({roc_auc_score(y, prob):.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=0.8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right"); save(fig, f"fig04_roc_{ds}")
    cal = [(l, c) for l in MODEL_COLOR if (c := calibrated(ds, l)) is not None]
    pd.DataFrame([dict(model=l, ece_before=ece(y, raw), ece_after=ece(y, p), rmse_before=float(np.sqrt(np.mean((y - raw) ** 2))), rmse_after=float(np.sqrt(np.mean((y - p) ** 2))),
                       auc=roc_auc_score(y, p)) for l, (y, p, raw) in cal]).to_csv(f"{OUT}/calibration_{ds}.csv", index=False)
    fig, ax = plt.subplots(figsize=SIZE)
    for l, (y, prob, _) in cal:
        bins = np.clip((prob * 10).astype(int), 0, 9); keep = [b for b in range(10) if (bins == b).sum() >= 200]
        ax.plot([prob[bins == b].mean() for b in keep], [y[bins == b].mean() for b in keep], color=MODEL_COLOR[l], marker="o", label=l)
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=0.8, label="Perfect calibration"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability of a correct answer"); ax.set_ylabel("Observed share of correct answers"); ax.legend(loc="upper left"); save(fig, f"fig05_calibration_{ds}")
    y, prob = curves[0][1] if curves[0][0] == "Proposed" else (None, None)
    if y is not None:
        fig, ax = plt.subplots(figsize=SIZE); edges = np.linspace(0, 1, 41)
        ax.hist(prob[y == 1], bins=edges, density=True, color=BLUE, alpha=0.65, label="Answer was correct")
        ax.hist(prob[y == 0], bins=edges, density=True, color=ORANGE, alpha=0.65, label="Answer was wrong")
        ax.set_xlabel("Predicted probability of a correct answer (Proposed)"); ax.set_ylabel("Density"); ax.legend(loc="upper left"); save(fig, f"fig06_prediction_distribution_{ds}")

# ---------------------------------------------------------------- 7) AUC by learner history length (how the model predicts for new vs experienced learners)
def read_any(path, columns):
    return pd.read_parquet(path, columns=columns) if path.endswith(".parquet") else pd.read_pickle(path)[columns]
BUCKETS = [(5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 10**9)]
for ds in ACTIVE:
    d = DS[ds]; shards = sorted(glob.glob(f"{ROOT}/{d['data']}/*.parquet")) + sorted(glob.glob(f"{ROOT}/{d['data']}/*.pkl"))
    if not shards or not os.path.exists(f"{ROOT}/{d['split']}"): missing.append(f"{d['name']}: prepared interactions (AUC by history length)"); continue
    split = pd.read_csv(f"{ROOT}/{d['split']}", dtype=str); users = set(split.loc[split["split"].eq("test"), "user_id"])
    parts = []
    for s in shards:
        part = read_any(s, ["user_id", "timestamp", "item_id", "correct"]); part["user_id"] = part["user_id"].astype(str); parts.append(part[part["user_id"].isin(users)])
    test = pd.concat(parts, ignore_index=True).sort_values(["user_id", "timestamp", "item_id"], kind="mergesort"); del parts
    position = test.groupby("user_id", sort=False).cumcount().to_numpy(); keep = position >= 5
    y_ref, pos = test["correct"].to_numpy(np.int8)[keep], position[keep]; del test
    lines = []
    for l in MODEL_COLOR:
        p = predictions(ds, l)
        if p is None or len(p[0]) != len(y_ref) or not np.array_equal(p[0], y_ref): continue
        aucs = []
        for lo, hi in BUCKETS:
            m = (pos >= lo) & (pos < hi); aucs.append(roc_auc_score(y_ref[m], p[1][m]) if m.sum() > 500 and 0 < y_ref[m].mean() < 1 else np.nan)
        lines.append((l, aucs))
    if not lines: missing.append(f"{d['name']}: predictions not aligned with the interactions (AUC by history length)"); continue
    fig, ax = plt.subplots(figsize=SIZE); x = np.arange(len(BUCKETS))
    for l, aucs in lines: ax.plot(x, aucs, color=MODEL_COLOR[l], marker="o", label=l)
    ax.set_xticks(x); ax.set_xticklabels(["5-9", "10-19", "20-49", "50-99", "100-199", "200-499", "500+"])
    ax.set_xlabel("Number of previous interactions of the learner"); ax.set_ylabel("Test AUC"); legend_below(ax, 3); save(fig, f"fig07_auc_by_history_length_{ds}")

# ---------------------------------------------------------------- 8) learning curves (validation AUC per epoch)
for ds in ACTIVE:
    lines = []
    for l in MODEL_COLOR:
        if l == "Same inputs + XGBoost": continue
        stem = f"{stem_of(ds, l)}_seed42"; f = find(f"training_history/{stem}.csv", ds) or find(f"{stem}_history.csv", ds)
        if f is None: continue
        h = pd.read_csv(f)
        if "auc" in h: lines.append((l, h["epoch"].to_numpy() if "epoch" in h else np.arange(1, len(h) + 1), h["auc"].to_numpy()))
    if not lines: missing.append(f"{DS[ds]['name']}: training histories"); continue
    fig, ax = plt.subplots(figsize=SIZE)
    for l, ep, auc in lines:
        ax.plot(ep, auc, color=MODEL_COLOR[l], label=l); b = int(np.nanargmax(auc)); ax.scatter([ep[b]], [auc[b]], color=MODEL_COLOR[l], s=22, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUC"); ax.legend(loc="lower right"); save(fig, f"fig08_learning_curves_{ds}")

# ---------------------------------------------------------------- 9) ablation: AUC lost when one part is removed (95% CI)
for ds in ACTIVE:
    a = DS[ds]["abl_t"]
    if a is None: missing.append(f"{DS[ds]['name']}: ablation table"); continue
    a = a.sort_values("auc_drop_when_removed"); y = np.arange(len(a)); sig = a["verdict"].astype(str).str.startswith("helps").to_numpy()
    fig, ax = plt.subplots(figsize=SIZE_TALL)
    ax.barh(y, a["auc_drop_when_removed"], color=[BLUE if s else GREY for s in sig], height=0.6,
            xerr=[(a["auc_drop_when_removed"] - a["ci95_low"]).clip(lower=0), (a["ci95_high"] - a["auc_drop_when_removed"]).clip(lower=0)], error_kw=dict(ecolor=DARK, lw=0.7, capsize=1.5))
    ax.set_yticks(y); ax.set_yticklabels([str(r)[0].upper() + str(r)[1:] for r in a["removed"]]); ax.axvline(0, color=DARK, lw=0.6)
    ax.set_xlabel("Drop in test AUC when the component is removed"); ax.grid(axis="y", visible=False); save(fig, f"fig09_ablation_{ds}")

# ---------------------------------------------------------------- 10) model selection on the validation set
for ds in ACTIVE:
    s = DS[ds]["sel_t"]
    if s is None: continue
    s = s.iloc[::-1].reset_index(drop=True); chosen = s["result"].astype(str).eq("chosen").to_numpy()
    labels = [("Backbone: " if a == "backbone" else "") + str(b)[0].upper() + str(b)[1:] for a, b in zip(s["step"], s["option"])]
    fig, ax = plt.subplots(figsize=SIZE_TALL); ax.barh(range(len(s)), s["validation_auc"], color=[BLUE if c else GREY for c in chosen], height=0.6)
    lo, hi = s["validation_auc"].min(), s["validation_auc"].max(); ax.set_xlim(lo - (hi - lo) * 0.5 - 1e-4, hi + (hi - lo) * 0.25 + 1e-4)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(labels, fontsize=7); ax.set_xlabel("Validation AUC"); ax.grid(axis="y", visible=False)
    save(fig, f"fig10_model_selection_{ds}")

# ---------------------------------------------------------------- 11) seed stability
for ds in ACTIVE:
    rows = []
    for l in MODEL_COLOR:
        for seed in (42, 52, 62):
            f = find(f"{stem_of(ds, l)}_seed{seed}_metrics.json", ds)
            if f is not None:
                m = json.load(f)
                if isinstance(m, dict) and "test" in m: rows.append((l, seed, m["test"]["auc"]))
    if not rows: continue
    r = pd.DataFrame(rows, columns=["model", "seed", "auc"]); order = [l for l in MODEL_COLOR if l in set(r["model"])]
    fig, ax = plt.subplots(figsize=SIZE)
    for i, l in enumerate(order):
        v = r[r["model"] == l]["auc"].to_numpy(); ax.scatter(np.full(len(v), i) + np.linspace(-0.1, 0.1, len(v)), v, color=MODEL_COLOR[l], s=26, edgecolor="white", linewidth=0.8, zorder=3)
        ax.hlines(v.mean(), i - 0.28, i + 0.28, color=DARK, lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace(" (base paper)", "\n(base paper)").replace("Same inputs + ", "Same inputs\n+ ") for o in order], fontsize=7)
    ax.set_ylabel("Test AUC (seeds 42, 52, 62)"); ax.grid(axis="x", visible=False); save(fig, f"fig11_seed_stability_{ds}")

# ---------------------------------------------------------------- 12-15) pedagogy recommendation and explainability
NAMES = {"proposed": "Proposed recommender", "ablation_combined_target": "Combined target", "ablation_current_state_only": "Current state only",
         "ablation_basic_inputs": "Basic inputs only", "ablation_next_step_only": "Next step only",
         "baseline_most_often_relevant": "Most often relevant", "baseline_most_common_labels": "Most common labels"}
PCOL = {"proposed": BLUE, "ablation_combined_target": "#0d3a6e", "ablation_basic_inputs": "#4f86c6", "ablation_current_state_only": "#7fa8d8", "ablation_next_step_only": "#0d3a6e", "baseline_most_often_relevant": GREY, "baseline_most_common_labels": "#6e6d68"}
PSTYLE = {"proposed": "-", "ablation_combined_target": "-.", "ablation_basic_inputs": (0, (5, 1)), "ablation_current_state_only": "--", "ablation_next_step_only": "-.", "baseline_most_often_relevant": ":", "baseline_most_common_labels": ":"}
for ds in DS:
    pfx = DS[ds]["ped"]; t = csv(f"{pfx}next_step_table.csv")
    if t is None: missing.append(f"{DS[ds]['name']}: {pfx}next_step_table.csv (pedagogy)"); continue
    t = t[t["evaluated_on"] == "test learners"]; methods = [m for m in NAMES if m in set(t["method"])]
    k3 = t[t["K"] == 3].set_index("method"); mets = [m for m in ("precision", "recall", "ndcg", "top1_relevance") if m in k3]
    fig, ax = plt.subplots(figsize=SIZE); w = 0.8 / len(methods); x = np.arange(len(mets))
    for i, m in enumerate(methods): ax.bar(x - 0.4 + w / 2 + i * w, [k3.loc[m, c] for c in mets], w * 0.92, color=PCOL[m], label=NAMES[m])
    ax.set_xticks(x); ax.set_xticklabels({"precision": "Precision@3", "recall": "Recall@3", "ndcg": "NDCG@3", "top1_relevance": "Top-1 relevance"}[c] for c in mets)
    ax.set_ylim(0, 1.05); ax.set_xlabel("Metric"); ax.set_ylabel("Score"); ax.grid(axis="x", visible=False); legend_below(ax, 3)
    save(fig, f"fig12_pedagogy_metrics_k3_{ds}")
    for col, ylabel, tag in (("ndcg", "NDCG@K", "ndcg"), ("precision", "Precision@K", "precision"), ("recall", "Recall@K", "recall")):
        if col not in t or t["K"].nunique() < 2: continue
        fig, ax = plt.subplots(figsize=SIZE)
        for m in methods:
            s = t[t["method"] == m].sort_values("K"); ax.plot(s["K"], s[col], color=PCOL[m], ls=PSTYLE[m], marker="o", label=NAMES[m])
        ax.set_xticks(sorted(t["K"].unique())); ax.set_xlabel("Number of recommended strategies (K)"); ax.set_ylabel(ylabel); legend_below(ax, 3)
        save(fig, f"fig13_pedagogy_{tag}_at_k_{ds}")
    s = csv(f"{pfx}by_strategy.csv")
    if s is not None:                                            # coverage: when a strategy is needed next, is it in the top-3 list?
        classes = [c for c in s.columns if c.endswith(("learning", "guidance"))]
        rows = s[s["measure"].astype(str).str.contains("when relevant") & s["method"].isin([m for m in methods if m in NAMES])].set_index("method")
        shown = [m for m in methods if m in rows.index]
        fig, ax = plt.subplots(figsize=SIZE); x = np.arange(len(classes)); w = 0.8 / max(len(shown), 1)
        for i, m in enumerate(shown): ax.bar(x - 0.4 + w / 2 + i * w, [rows.loc[m, c] for c in classes], w * 0.92, color=PCOL[m], label=NAMES[m])
        ax.set_xticks(x); ax.set_xticklabels([c.replace("_learning", "").replace("_guidance", "").replace("_", "-").capitalize() for c in classes], rotation=20, ha="right")
        ax.set_ylim(0, 1.05); ax.set_xlabel("Pedagogical strategy needed at the next step"); ax.set_ylabel("Share recommended in the top-3 list"); ax.grid(axis="x", visible=False); legend_below(ax, 3)
        save(fig, f"fig14_strategy_coverage_{ds}")
    sh = csv(f"{pfx}shap_importance.csv")
    if sh is not None and "mean_abs_shap" in sh:
        sh = sh.sort_values("mean_abs_shap", ascending=False).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=SIZE_TALL); ax.barh(sh["feature"].astype(str).str.replace("_", " "), sh["mean_abs_shap"], color=BLUE, height=0.6)
        ax.set_xlabel("Mean |SHAP value| (importance for the recommendation)"); ax.grid(axis="y", visible=False); save(fig, f"fig15_explainability_shap_{ds}")

print(f"{len(made)} figures saved in {OUT} (PNG {DPI} dpi + PDF):"); [print("  " + m) for m in made]
if missing: print("not made (input not found): " + "; ".join(missing))
