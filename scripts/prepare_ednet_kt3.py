#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from adaptive_kt.config import load_config
from adaptive_kt.data.ednet import find_kt3_user_files, load_questions, process_user_file
from adaptive_kt.utils.tableio import write_table

_WORKER_QUESTIONS: pd.DataFrame | None = None
_WORKER_RECENT_WINDOW = 10
_WORKER_MIN_HISTORY = 1
_WORKER_MAX_RESPONSE_SECONDS = 1200.0


def _init_worker(
    questions: pd.DataFrame,
    recent_window: int,
    min_history: int,
    max_response_time_seconds: float,
) -> None:
    global _WORKER_QUESTIONS
    global _WORKER_RECENT_WINDOW
    global _WORKER_MIN_HISTORY
    global _WORKER_MAX_RESPONSE_SECONDS
    _WORKER_QUESTIONS = questions
    _WORKER_RECENT_WINDOW = recent_window
    _WORKER_MIN_HISTORY = min_history
    _WORKER_MAX_RESPONSE_SECONDS = max_response_time_seconds


def _process_path(path_text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if _WORKER_QUESTIONS is None:
        raise RuntimeError("EdNet worker was not initialized.")
    return process_user_file(
        Path(path_text),
        _WORKER_QUESTIONS,
        recent_window=_WORKER_RECENT_WINDOW,
        min_history=_WORKER_MIN_HISTORY,
        max_response_time_seconds=_WORKER_MAX_RESPONSE_SECONDS,
    )


def flush_shard(frames: list[pd.DataFrame], directory: Path, shard_index: int) -> int:
    if not frames:
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames, ignore_index=True)
    write_table(frame, directory / f"part-{shard_index:05d}")
    return len(frame)


def _result_iterator(
    user_files: list[Path],
    questions: pd.DataFrame,
    workers: int,
    recent_window: int,
    min_history: int,
    max_response_time_seconds: float,
):
    if workers <= 1:
        for user_file in user_files:
            yield process_user_file(
                user_file,
                questions,
                recent_window=recent_window,
                min_history=min_history,
                max_response_time_seconds=max_response_time_seconds,
            )
        return

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(questions, recent_window, min_history, max_response_time_seconds),
    ) as executor:
        yield from executor.map(_process_path, map(str, user_files), chunksize=16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EdNet-KT3 for the experimental pipeline.")
    parser.add_argument("--config", default="configs/ednet_kt3.yaml")
    parser.add_argument("--max-users", type=int, default=None, help="Override dataset.max_users for a development run.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = config["dataset"]
    raw_root = Path(dataset["raw_root"])
    processed_root = Path(dataset["processed_root"])
    interactions_dir = processed_root / "interactions"
    features_dir = processed_root / "features"

    if processed_root.exists() and args.overwrite:
        shutil.rmtree(processed_root)
    if interactions_dir.exists() and (
        any(interactions_dir.glob("*.parquet")) or any(interactions_dir.glob("*.pkl"))
    ):
        raise FileExistsError(
            f"Prepared EdNet data already exists at {processed_root}; pass --overwrite to replace it."
        )

    questions = load_questions(raw_root)
    user_files = find_kt3_user_files(raw_root)
    max_users = args.max_users if args.max_users is not None else dataset.get("max_users")
    if max_users:
        user_files = user_files[: int(max_users)]

    workers = max(1, int(dataset.get("workers", 1)))
    users_per_shard = int(dataset.get("users_per_shard", 1000))
    recent_window = int(dataset.get("recent_window", 10))
    min_history = int(dataset.get("min_history", 1))
    max_response_time_seconds = float(dataset.get("max_response_time_seconds", 1200))

    interaction_buffer: list[pd.DataFrame] = []
    feature_buffer: list[pd.DataFrame] = []
    interaction_rows = 0
    feature_rows = 0
    shard_index = 0

    results = _result_iterator(
        user_files,
        questions,
        workers,
        recent_window,
        min_history,
        max_response_time_seconds,
    )
    for idx, (interactions, features) in enumerate(
        tqdm(results, total=len(user_files), desc="EdNet-KT3 users"), start=1
    ):
        if not interactions.empty:
            interaction_buffer.append(interactions)
        if not features.empty:
            feature_buffer.append(features)

        if idx % users_per_shard == 0 or idx == len(user_files):
            interaction_rows += flush_shard(interaction_buffer, interactions_dir, shard_index)
            feature_rows += flush_shard(feature_buffer, features_dir, shard_index)
            interaction_buffer.clear()
            feature_buffer.clear()
            shard_index += 1

    manifest = pd.DataFrame(
        [
            {
                "dataset": "ednet_kt3",
                "preprocessing_version": 3,
                "users_processed": len(user_files),
                "interaction_rows": interaction_rows,
                "feature_rows": feature_rows,
                "shards": shard_index,
                "questions": len(questions),
                "workers": workers,
            }
        ]
    )
    processed_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(processed_root / "manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
