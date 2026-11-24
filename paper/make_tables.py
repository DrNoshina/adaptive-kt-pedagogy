"""Build the LaTeX result tables of the paper from the final result CSV files.

Usage (from the project root):
    .venv/bin/python make_tables.py results/kt3_final results/assist2017 paper/tables

Reads, for each dataset prefix (kt3, assist2017):
    <prefix>_main_table.csv, <prefix>_beats.csv, <prefix>_ablation_final.csv,
    <prefix>_selection.csv, <prefix>_training_check.csv
Writes tables/<prefix>_main.tex, _beats.tex, _ablation.tex, _selection.tex, _training.tex.
Read-only on results; no number is typed by hand.
"""
import os
import sys

import pandas as pd

NAMES = {
    "Proposed (final)": "Proposed",
    "HGAKT (base paper, our adaptation)": "HGAKT (base model)",
    "Same inputs as Proposed + XGBoost": "XGBoost (same inputs)",
}
SELECT = {"attention (V3)": "Attention backbone", "attention": "Attention backbone", "LSTM": "LSTM backbone"}


def name(x):
    x = str(x).strip()
    return NAMES.get(x, x)


def f4(x):
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "--"


def tex_escape(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("+", "+")


def write(path, body):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print("written:", path)


def main_table(df, prefix, out):
    rows = []
    for _, r in df.iterrows():
        std = r.get("test_auc_std", float("nan"))
        auc = f"{float(r['test_auc']):.4f} $\\pm$ {float(std):.4f}" if pd.notna(std) else f4(r["test_auc"])
        rows.append(f"{tex_escape(name(r['model']))} & {f4(r['val_auc'])} & {auc} & {f4(r['accuracy'])} & "
                    f"{f4(r['precision'])} & {f4(r['recall'])} & {f4(r['f1'])} & {f4(r['rmse'])} \\\\")
    write(f"{out}/{prefix}_main.tex", "\n".join(rows) + "\n")


def beats_table(df, prefix, out):
    rows = []
    for _, r in df.iterrows():
        p = float(r["p_value"])
        ptxt = "$\\leq$ 0.001" if p <= 0.001 else f"{p:.3f}"
        notb = str(r["not_better_on"]).replace("-", "none")
        rows.append(f"{tex_escape(name(r['compared_with']))} & {float(r['auc_gain']):.4f} & {ptxt} & {tex_escape(notb)} \\\\")
    write(f"{out}/{prefix}_beats.tex", "\n".join(rows) + "\n")


def ablation_table(df, prefix, out):
    rows = []
    df = df.sort_values("auc_drop_when_removed", ascending=False)
    for _, r in df.iterrows():
        p = float(r["p_value"])
        ptxt = "$\\leq$ 0.001" if p <= 0.001 else f"{p:.3f}"
        verdict = "yes" if "significant" in str(r.get("verdict", "")) and "no significant" not in str(r.get("verdict", "")) else "no"
        removed = str(r["removed"]).replace("hint (explanation) context", "hint/explanation context")
        rows.append(f"{tex_escape(removed[0].upper() + removed[1:])} & {f4(r['test_auc'])} & {float(r['auc_drop_when_removed']):.4f} & "
                    f"[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}] & {ptxt} & {verdict} \\\\")
    write(f"{out}/{prefix}_ablation.tex", "\n".join(rows) + "\n")


def selection_table(df, prefix, out):
    rows = []
    for _, r in df.iterrows():
        opt = SELECT.get(str(r["option"]).strip(), str(r["option"]))
        step = "Backbone" if r["step"] == "backbone" else "Input groups"
        res = str(r["result"]).replace("-", "not selected") if str(r["result"]).strip() == "-" else str(r["result"])
        res = res.replace("helped -> kept", "kept").replace("did not help -> removed", "removed")
        rows.append(f"{step} & {tex_escape(opt[0].upper() + opt[1:])} & {float(r['validation_auc']):.5f} & {tex_escape(res)} \\\\")
    write(f"{out}/{prefix}_selection.tex", "\n".join(rows) + "\n")


def training_table(df, prefix, out):
    rows = []
    neural = df[df["best_epoch"].astype(str) != "-"].copy()
    for model, g in neural.groupby("model", sort=False):
        best = g["best_epoch"].astype(float)
        run = pd.to_numeric(g["epochs_run"], errors="coerce")
        # a run counts as early-stopped when its label says so or when it ran at least
        # `patience` (5) epochs past its best epoch, i.e. it stopped before its epoch limit
        ok = (g["stopped_by_early_stopping"].astype(str) == "yes") | ((run - best) >= 5)
        stopped = "yes" if ok.all() else "no"
        rows.append(f"{tex_escape(name(model))} & {best.mean():.1f} & {run.mean():.1f} & {stopped} & "
                    f"{g['val_auc'].astype(float).mean():.4f} & {g['test_auc'].astype(float).mean():.4f} & "
                    f"{g['val_test_gap'].astype(float).abs().mean():.4f} \\\\")
    write(f"{out}/{prefix}_training.tex", "\n".join(rows) + "\n")


def run(folder, prefix, out):
    jobs = [("main_table", main_table), ("beats", beats_table), ("ablation_final", ablation_table),
            ("selection", selection_table), ("training_check", training_table)]
    for stem, fn in jobs:
        path = os.path.join(folder, f"{prefix}_{stem}.csv")
        if os.path.exists(path):
            fn(pd.read_csv(path), prefix, out)
        else:
            print("missing (table not rebuilt):", path)


if __name__ == "__main__":
    defaults = ["results/kt3_final", "results/assist2017", "paper/tables"]
    args = sys.argv[1:4]
    kt3_dir, assist_dir, out_dir = args + defaults[len(args):]
    os.makedirs(out_dir, exist_ok=True)
    run(kt3_dir, "kt3", out_dir)
    run(assist_dir, "assist2017", out_dir)
