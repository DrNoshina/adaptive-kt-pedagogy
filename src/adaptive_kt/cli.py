from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_kt.config import load_config
from adaptive_kt.data.generic import prepare_generic_dataset
from adaptive_kt.experiments.pipeline import run_experiment
from adaptive_kt.utils.tableio import write_table


def _write_prepared(interactions, features, root: Path) -> None:
    interactions_dir = root / "interactions"
    features_dir = root / "features"
    interactions_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    write_table(interactions, interactions_dir / "part-00000")
    write_table(features, features_dir / "part-00000")


def main() -> None:
    parser = argparse.ArgumentParser(prog="adaptive-kt")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-generic", help="Normalize ASSISTments/KDD-style data.")
    prepare.add_argument("--config", required=True)

    run = sub.add_parser("run", help="Run the complete experiment on prepared data.")
    run.add_argument("--config", required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "prepare-generic":
        interactions, features = prepare_generic_dataset(config)
        target = Path(config["dataset"]["processed_root"])
        _write_prepared(interactions, features, target)
        print(json.dumps({"interactions": len(interactions), "feature_rows": len(features)}, indent=2))
        return

    if args.command == "run":
        summary = run_experiment(config)
        print(json.dumps(summary, indent=2))
        return


if __name__ == "__main__":
    main()
