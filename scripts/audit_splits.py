#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved learner-disjoint split assignments.")
    parser.add_argument("--results", required=True, help="Result directory, e.g. results/ednet_kt3")
    args = parser.parse_args()

    root = Path(args.results)
    assignments_path = root / "split_assignments.csv"
    summary_path = root / "split_summary.json"

    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing: {assignments_path}")

    df = pd.read_csv(assignments_path)
    required = {"user_id", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"split_assignments.csv missing columns: {sorted(missing)}")

    allowed = {"train", "validation", "test"}
    invalid = sorted(set(df["split"].dropna().astype(str)) - allowed)
    if invalid:
        raise ValueError(f"Unexpected split labels: {invalid}")

    duplicate_users = df.loc[df["user_id"].duplicated(keep=False), "user_id"].astype(str).unique()
    if len(duplicate_users):
        preview = ", ".join(duplicate_users[:10])
        raise RuntimeError(
            f"LEAKAGE CHECK FAILED: {len(duplicate_users)} learner IDs occur more than once. "
            f"Examples: {preview}"
        )

    counts = df["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0)
    total = int(counts.sum())
    ratios = (counts / total).to_dict() if total else {}

    print("Learner-disjoint split audit: PASS")
    print(f"Unique learners: {total:,}")
    for split in ["train", "validation", "test"]:
        print(f"  {split:10s}: {int(counts[split]):,} learners ({ratios.get(split, 0.0):.4f})")
    print("No learner ID appears in more than one split.")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(f"Loaded split summary: {summary_path}")
        for split in ["train", "validation", "test"]:
            if split in summary and "users" in summary[split]:
                expected = int(summary[split]["users"])
                actual = int(counts[split])
                if expected != actual:
                    raise RuntimeError(
                        f"Split summary mismatch for {split}: assignments={actual}, summary={expected}"
                    )
        print("split_summary.json agrees with split_assignments.csv.")


if __name__ == "__main__":
    main()
