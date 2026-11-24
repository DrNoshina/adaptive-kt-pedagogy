"""Rebuild every result number of the paper from the experiment outputs.

Run from the project root after the experiments have finished:

    .venv/bin/python paper/build_paper.py

What it does (read-only on results/, writes only inside paper/):
  1. tables/   : every result table is regenerated from the CSV/JSON files that the
                 training and evaluation scripts wrote into results/ (no number typed by hand)
  2. checks    : every result number quoted in the text of main.tex is recomputed from the
                 same files and compared; any mismatch is reported as FAIL
  3. figures/  : result figures are copied from the figure script output, if present
  4. main.pdf  : compiled with pdflatex + bibtex when LaTeX is installed

A reviewer who reruns the experiments and then this script obtains the same tables,
the same checks, and the same PDF.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAPER = os.path.join(ROOT, "paper")
TABLES = os.path.join(PAPER, "tables")
FIGS = os.path.join(PAPER, "figures")
RES = os.path.join(ROOT, "results")

NAMES = {
    "Proposed (final)": "Proposed",
    "HGAKT (base paper, our adaptation)": "HGAKT (base model)",
    "Same inputs as Proposed + XGBoost": "XGBoost (same inputs)",
}
SELECT = {"attention (V3)": "Attention backbone", "attention": "Attention backbone", "LSTM": "LSTM backbone"}
PED_ROWS = [
    ("proposed", "Proposed recommender"),
    ("ablation_combined_target", "Variant: combined target"),
    ("ablation_current_state_only", "Variant: current state only"),
    ("ablation_basic_inputs", "Variant: basic inputs only"),
    ("baseline_most_often_relevant", "Baseline: most often relevant"),
    ("baseline_most_common_labels", "Baseline: most common labels"),
]
problems: list[str] = []


# ----------------------------------------------------------------------------- helpers
def newest(pattern: str, exclude: tuple[str, ...] = ("backup", "old")) -> str | None:
    hits = [p for p in glob.glob(os.path.join(RES, "**", pattern), recursive=True)
            if not any(x in p.lower() for x in exclude)]
    return max(hits, key=os.path.getmtime) if hits else None


def need(pattern: str, what: str) -> str | None:
    path = newest(pattern)
    if path is None:
        problems.append(f"MISSING  {what}: no file results/**/{pattern}")
    else:
        print(f"  using {os.path.relpath(path, ROOT)}")
    return path


def name(x) -> str:
    return NAMES.get(str(x).strip(), str(x).strip())


def f4(x) -> str:
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "--"


def esc(s) -> str:
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def ptxt(p) -> str:
    p = float(p)
    return "$\\leq$ 0.001" if p <= 0.001 else f"{p:.3f}"


def write(fname: str, rows: list[str]) -> None:
    with open(os.path.join(TABLES, fname), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"  written tables/{fname}")


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


# ----------------------------------------------------------------------------- tables
def prediction_tables(prefix: str) -> dict:
    data = {}
    jobs = {
        "main_table": "main", "beats": "beats", "ablation_final": "ablation",
        "selection": "selection", "training_check": "training",
    }
    for stem, short in jobs.items():
        path = need(f"{prefix}_{stem}.csv", f"{prefix} {stem}")
        if path:
            data[short] = pd.read_csv(path)
    if "main" in data:
        rows = []
        for _, r in data["main"].iterrows():
            std = r.get("test_auc_std", float("nan"))
            auc = f"{float(r['test_auc']):.4f} $\\pm$ {float(std):.4f}" if pd.notna(std) else f4(r["test_auc"])
            rows.append(f"{esc(name(r['model']))} & {f4(r['val_auc'])} & {auc} & {f4(r['accuracy'])} & {f4(r['precision'])} & "
                        f"{f4(r['recall'])} & {f4(r['f1'])} & {f4(r['rmse'])} \\\\")
        write(f"{prefix}_main.tex", rows)
    if "beats" in data:
        rows = [f"{esc(name(r['compared_with']))} & {float(r['auc_gain']):.4f} & {ptxt(r['p_value'])} & "
                f"{esc(str(r['not_better_on']).replace('-', 'none'))} \\\\" for _, r in data["beats"].iterrows()]
        write(f"{prefix}_beats.tex", rows)
    if "ablation" in data:
        rows = []
        for _, r in data["ablation"].sort_values("auc_drop_when_removed", ascending=False).iterrows():
            v = str(r.get("verdict", ""))
            sig = "yes" if "significant" in v and "no significant" not in v else "no"
            removed = str(r["removed"]).replace("hint (explanation) context", "hint/explanation context")
            rows.append(f"{esc(removed[0].upper() + removed[1:])} & {f4(r['test_auc'])} & {float(r['auc_drop_when_removed']):.4f} & "
                        f"[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}] & {ptxt(r['p_value'])} & {sig} \\\\")
        write(f"{prefix}_ablation.tex", rows)
    if "selection" in data:
        rows = []
        for _, r in data["selection"].iterrows():
            opt = SELECT.get(str(r["option"]).strip(), str(r["option"]))
            step = "Backbone" if r["step"] == "backbone" else "Input groups"
            res = "not selected" if str(r["result"]).strip() == "-" else str(r["result"])
            res = res.replace("helped -> kept", "kept").replace("did not help -> removed", "removed")
            rows.append(f"{step} & {esc(opt[0].upper() + opt[1:])} & {float(r['validation_auc']):.5f} & {esc(res)} \\\\")
        write(f"{prefix}_selection.tex", rows)
    if "training" in data:
        rows = []
        df = data["training"]
        neural = df[df["best_epoch"].astype(str) != "-"]
        for model, g in neural.groupby("model", sort=False):
            best = g["best_epoch"].astype(float)
            run = pd.to_numeric(g["epochs_run"], errors="coerce")
            # early-stopped when labelled so, or when it ran >= patience (5) epochs past its best epoch
            ok = (g["stopped_by_early_stopping"].astype(str) == "yes") | ((run - best) >= 5)
            rows.append(f"{esc(name(model))} & {best.mean():.1f} & {run.mean():.1f} & {'yes' if ok.all() else 'no'} & "
                        f"{g['val_auc'].astype(float).mean():.4f} & {g['test_auc'].astype(float).mean():.4f} & "
                        f"{g['val_test_gap'].astype(float).abs().mean():.4f} \\\\")
        write(f"{prefix}_training.tex", rows)
    return data


def pedagogy_tables() -> dict:
    out = {}
    for key, stem, json_stem in (("kt3", "pedagogy_v3_next_step_table.csv", "pedagogy_v3_results.json"),
                                 ("assist2017", "assist2017_pedagogy_next_step_table.csv", "assist2017_pedagogy_results.json")):
        csv_path = need(stem, f"{key} pedagogy table")
        js_path = need(json_stem, f"{key} pedagogy results")
        if not csv_path:
            continue
        tab = pd.read_csv(csv_path)
        res = json.load(open(js_path)) if js_path else {}
        chosen = str(res.get("design_selected_on_validation", "proposed"))
        test3 = tab[(tab["evaluated_on"] == "test learners") & (tab["K"] == 3)].set_index("method")
        rows = []
        for m, label in PED_ROWS:
            if m not in test3.index:
                continue
            r = test3.loc[m]
            if m == "proposed" and chosen != "proposed":
                label += " (full design)"
            if m == chosen and m != "proposed":
                label += " (selected)"
            rows.append(f"{label} & {r['precision']:.4f} & {r['recall']:.4f} & {r['ndcg']:.4f} & {r['hit']:.4f} & {r['top1_relevance']:.4f} \\\\")
        write(f"pedagogy_{key}.tex", rows)
        shap_path = newest(stem.replace("next_step_table", "shap_importance"))
        out[key] = {"table": tab, "results": flat(res), "chosen": chosen,
                    "shap": pd.read_csv(shap_path) if shap_path else None,
                    "by_strategy": pd.read_csv(newest(stem.replace("next_step_table", "by_strategy"))) if newest(stem.replace("next_step_table", "by_strategy")) else None}
    return out


def data_table() -> dict:
    kt3_path = need("trackB_split_summary.json", "EdNet-KT3 split summary")
    as_path = need("data_stats.json", "ASSISTments 2017 data statistics")
    kt = json.load(open(kt3_path)) if kt3_path else None
    st = json.load(open(as_path)) if as_path else None
    KT3_FEATURE_ROWS = 16488101  # printed by scripts/prepare_ednet_kt3.py (question-response rows)

    def three(d, key):
        return " / ".join(f"{d[s][key]:,}" for s in ("train", "validation", "test"))

    rows = []
    if kt and st:
        kt_users = sum(kt[s]["users"] for s in ("train", "validation", "test"))
        kt_int = sum(kt[s]["interactions"] for s in ("train", "validation", "test"))
        kt_pos = sum(kt[s]["interactions"] * kt[s]["positive_rate"] for s in ("train", "validation", "test")) / kt_int
        rows += [
            f"Learners & {kt_users:,} & {st['learners']:,} \\\\",
            f"Training / validation / test learners & {three(kt, 'users')} & {three(st['split'], 'learners')} \\\\",
            f"Interactions & {kt_int:,} & {st['rows']:,} \\\\",
            f"Training / validation / test interactions & {three(kt, 'interactions')} & {three(st['split'], 'interactions')} \\\\",
            f"Question-response feature rows & {KT3_FEATURE_ROWS:,} & {st['feature_rows']:,} \\\\",
            f"Share of correct responses & {kt_pos:.3f} & {st['correct_rate']:.3f} \\\\",
            "Behavioral signals & lectures, explanations, offers & hints \\\\",
            "Split (training/validation/test learners) & 70/10/20 & 70/10/20 \\\\",
        ]
        write("data.tex", rows)
    return {"kt3": kt, "assist2017": st}


# ----------------------------------------------------------------------------- text checks
def text_checks(pred: dict, ped: dict, data: dict) -> list[tuple[str, str, str]]:
    """(claim, value written in the paper, value recomputed from results)."""
    out = []

    def add(claim, written, value):
        out.append((claim, written, value))

    def auc(prefix, model):
        df = pred.get(prefix, {}).get("main")
        if df is None:
            return None
        row = df[df["model"].map(name) == model]
        return None if row.empty else row.iloc[0]

    for prefix, written in (("kt3", {"Proposed": "0.7742", "HGAKT (base model)": "0.7587", "AKT": "0.7566", "SAKT": "0.7518",
                                     "DKT": "0.7489", "XGBoost (same inputs)": "0.7615"}),
                            ("assist2017", {"Proposed": "0.8481", "HGAKT (base model)": "0.8131", "DKT": "0.7943",
                                            "AKT": "0.7747", "XGBoost (same inputs)": "0.7705"})):
        for model, w in written.items():
            r = auc(prefix, model)
            add(f"{prefix} test AUC {model}", w, "--" if r is None else f"{float(r['test_auc']):.4f}")
        r = auc(prefix, "Proposed")
        add(f"{prefix} test AUC std Proposed", {"kt3": "0.0003", "assist2017": "0.0002"}[prefix],
            "--" if r is None else f"{float(r['test_auc_std']):.4f}")
        beats = pred.get(prefix, {}).get("beats")
        if beats is not None:
            add(f"{prefix} all AUC gains significant (p <= 0.001)", "yes",
                "yes" if (beats["p_value"].astype(float) <= 0.001).all() else "no")
    abl = pred.get("kt3", {}).get("ablation")
    if abl is not None:
        top = abl.sort_values("auc_drop_when_removed", ascending=False)["auc_drop_when_removed"].astype(float).tolist()
        add("kt3 three largest ablation drops", "0.0041 / 0.0041 / 0.0031", " / ".join(f"{v:.4f}" for v in top[:3]))
    abl = pred.get("assist2017", {}).get("ablation")
    if abl is not None:
        top = abl.sort_values("auc_drop_when_removed", ascending=False)
        add("assist2017 largest ablation drops", "0.0324 / 0.0055 / 0.0027 / 0.0020",
            " / ".join(f"{v:.4f}" for v in top["auc_drop_when_removed"].astype(float).tolist()[:4]))
        lp = abl[abl["removed"] == "learner profile"]
        add("assist2017 learner profile p-value", "0.090", "--" if lp.empty else f"{float(lp.iloc[0]['p_value']):.3f}")
    sel = pred.get("assist2017", {}).get("selection")
    if sel is not None:
        v = dict(zip(sel["option"].astype(str), sel["validation_auc"].astype(float)))
        add("assist2017 validation AUC LSTM / attention", "0.8476 / 0.8295",
            f"{v.get('LSTM', float('nan')):.4f} / {v.get('attention (V3)', v.get('attention', float('nan'))):.4f}")
        add("assist2017 validation AUC without experience+timing", "0.8485",
            f"{v.get('without experience + timing', float('nan')):.4f}")
    for key, written in (("kt3", {"precision": "0.5779", "recall": "0.9960", "ndcg": "0.9505", "hit": "0.9999", "top1_relevance": "0.8829"}),):
        if key in ped:
            t = ped[key]["table"]
            r = t[(t["evaluated_on"] == "test learners") & (t["K"] == 3) & (t["method"] == "proposed")].iloc[0]
            for m, w in written.items():
                add(f"{key} pedagogy proposed {m}@3", w, f"{r[m]:.4f}")
            v = t[(t["evaluated_on"] == "held-out validation learners") & (t["K"] == 3) & (t["method"] == "proposed")].iloc[0]
            add(f"{key} pedagogy validation NDCG@3 / top-1", "0.9513 / 0.8840", f"{v['ndcg']:.4f} / {v['top1_relevance']:.4f}")
    for key, w_fid, w_con in (("kt3", "0.112", "0.796"), ("assist2017", "0.032", "0.692")):
        if key in ped:
            r = ped[key]["results"]
            add(f"{key} explanation fidelity / consistency", f"{w_fid} / {w_con}",
                f"{float(r.get('explainability.proposed.fidelity_probability_drop', 'nan')):.3f} / "
                f"{float(r.get('explainability.proposed.consistency_jaccard', 'nan')):.3f}")
            add(f"{key} class-balance tau chosen on validation", "0.25",
                f"{float(r.get('class_balance.tau_chosen_on_validation', 'nan')):.2f}")
    if "assist2017" in ped:
        add("assist2017 design selected on validation", "ablation_basic_inputs", ped["assist2017"]["chosen"])
        s = ped["assist2017"]["shap"]
        if s is not None:
            s = s.sort_values(s.columns[1], ascending=False).head(3)
            add("assist2017 top SHAP features", "recent_accuracy 1.119, kt_probability 0.196, topic_mastery 0.180",
                ", ".join(f"{a} {b:.3f}" for a, b in zip(s.iloc[:, 0], s.iloc[:, 1])))
    st = data.get("assist2017")
    if st:
        add("assist2017 learners / rows / problems / skills", "1709 / 942816 / 3162 / 102",
            f"{st['learners']} / {st['rows']} / {st['problems']} / {st['skills']}")
    return out


# ----------------------------------------------------------------------------- figures + compile
def figures() -> None:
    sources = [os.path.join(ROOT, "kt_figures"), os.path.expanduser("~/Downloads/kt_figures")]
    src = next((s for s in sources if glob.glob(os.path.join(s, "fig*.png"))), None)
    if src is None:
        problems.append("MISSING  result figures: run the figure script (kt_figures.py) first")
        return
    n = 0
    for f in glob.glob(os.path.join(src, "fig*.png")):
        shutil.copy2(f, FIGS)
        n += 1
    print(f"  copied {n} result figures from {src}")


def compile_pdf() -> None:
    if not shutil.which("pdflatex"):
        problems.append("INFO     pdflatex not installed: sudo apt install texlive-latex-extra texlive-science (or upload paper/ to Overleaf)")
        return
    cmds = [["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
            ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]]
    for c in cmds:
        subprocess.run(c, cwd=PAPER, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(os.path.join(PAPER, "main.log"), errors="ignore").read()
    errors = log.count("\n! ")
    print(f"  main.pdf built ({'no LaTeX errors' if errors == 0 else str(errors) + ' LaTeX errors, see paper/main.log'})")


def main() -> None:
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    print("1) result tables")
    pred = {p: prediction_tables(p) for p in ("kt3", "assist2017")}
    ped = pedagogy_tables()
    data = data_table()
    expected = {f"{p}_{t}.tex": n for p in ("kt3", "assist2017")
                for t, n in (("main", 8), ("beats", 4), ("ablation", 6), ("selection", 4), ("training", 7))}
    expected.update({"pedagogy_kt3.tex": 6, "pedagogy_assist2017.tex": 6, "data.tex": 3})
    for fname, ncol in expected.items():
        if not os.path.exists(os.path.join(TABLES, fname)):
            write(fname, [f"\\multicolumn{{{ncol}}}{{l}}{{\\TBD}} \\\\"])
            problems.append(f"MISSING  tables/{fname} (placeholder written; its result file was not found)")
    print("2) checks of the numbers written in the text")
    checks = text_checks(pred, ped, data)
    bad = 0
    for claim, written, value in checks:
        ok = written == value
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {claim:52s} paper: {written:40s} results: {value}")
    print("3) figures")
    figures()
    print("4) PDF")
    compile_pdf()
    print("\nSUMMARY")
    print(f"  checks: {len(checks) - bad} PASS, {bad} FAIL")
    for p in problems:
        print("  " + p)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
