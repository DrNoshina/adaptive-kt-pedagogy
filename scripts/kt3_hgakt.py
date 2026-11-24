from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from adaptive_kt.evaluation.metrics import (
    binary_metrics,
    select_f1_threshold,
)
from adaptive_kt.experiments.pipeline import _make_loaders
from adaptive_kt.models.hgakt_kt3 import HGAKTKT3
from adaptive_kt.models.sequence import BEHAVIOR_DIM
from adaptive_kt.training.kt_trainer import (
    _batch_predictions,
    predict_kt,
)
from adaptive_kt.utils.tableio import read_collection
from adaptive_kt.utils.reproducibility import torch_device


def _atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def load_saved_split(
    interactions: pd.DataFrame,
    assignment_path: Path,
):
    assignment = pd.read_csv(
        assignment_path,
        dtype={"user_id": str},
    )

    user_col = interactions["user_id"].astype(str)

    result = {}

    for split in ["train", "validation", "test"]:
        users = set(
            assignment.loc[
                assignment["split"] == split,
                "user_id",
            ].astype(str)
        )

        result[split] = interactions[
            user_col.isin(users)
        ].copy()

        print(
            split,
            "users=",
            result[split]["user_id"].nunique(),
            "rows=",
            len(result[split]),
            flush=True,
        )

    return result


def build_hierarchy(
    item_vocab,
    metadata_path: Path,
    max_tags: int = 8,
):
    questions = pd.read_csv(
        metadata_path,
        dtype={
            "question_id": str,
            "tags": str,
        },
    )

    # Part IDs already use a small integer range.
    part_numeric = pd.to_numeric(
        questions["part"],
        errors="coerce",
    ).fillna(0).astype(int)

    num_parts = int(part_numeric.max()) + 2

    all_tags = set()

    for raw in questions["tags"].fillna(""):
        for tag in str(raw).split(";"):
            tag = tag.strip()

            if tag and tag.lower() != "nan":
                all_tags.add(tag)

    tag_mapping = {
        tag: idx + 1
        for idx, tag in enumerate(sorted(all_tags))
    }

    num_tags = len(tag_mapping) + 1

    item_to_part = np.zeros(
        item_vocab.size,
        dtype=np.int64,
    )

    item_to_tags = np.zeros(
        (item_vocab.size, max_tags),
        dtype=np.int64,
    )

    mapped_questions = 0

    for row in questions.itertuples(index=False):

        qid = str(row.question_id)

        item_idx = item_vocab.mapping.get(qid)

        if item_idx is None:
            continue

        part = pd.to_numeric(
            pd.Series([row.part]),
            errors="coerce",
        ).fillna(0).iloc[0]

        part = int(part)

        if part >= 0:
            item_to_part[item_idx] = part + 1

        tags = []

        raw_tags = str(row.tags)

        if raw_tags.lower() != "nan":
            for tag in raw_tags.split(";"):
                tag = tag.strip()

                if tag in tag_mapping:
                    tags.append(
                        tag_mapping[tag]
                    )

        tags = tags[:max_tags]

        if tags:
            item_to_tags[
                item_idx,
                :len(tags),
            ] = tags

        mapped_questions += 1

    print(
        f"hierarchy: mapped_questions={mapped_questions:,} "
        f"parts={num_parts - 1:,} "
        f"tags={num_tags - 1:,} "
        f"max_tags={max_tags}",
        flush=True,
    )

    return (
        torch.from_numpy(item_to_part),
        torch.from_numpy(item_to_tags),
        num_parts,
        num_tags,
    )


def build_training_item_difficulty(
    train: pd.DataFrame,
    item_vocab,
):
    mapped = (
        train["item_id"]
        .astype(str)
        .map(item_vocab.mapping)
    )

    correct = pd.to_numeric(
        train["correct"],
        errors="coerce",
    ).fillna(0).to_numpy(dtype=np.float64)

    valid = mapped.notna().to_numpy()

    indices = (
        mapped[valid]
        .astype(int)
        .to_numpy()
    )

    correct = correct[valid]

    count = np.bincount(
        indices,
        minlength=item_vocab.size,
    ).astype(np.float64)

    correct_sum = np.bincount(
        indices,
        weights=correct,
        minlength=item_vocab.size,
    )

    # Beta(1,1) smoothing.
    probability_correct = (
        correct_sum + 1.0
    ) / (
        count + 2.0
    )

    difficulty = (
        1.0 - probability_correct
    ).astype(np.float32)

    difficulty[count == 0] = 0.5
    difficulty[0] = 0.5
    difficulty[1] = 0.5

    print(
        "training-only item difficulty:",
        f"mean={difficulty[count > 0].mean():.4f}",
        f"items_seen={(count > 0).sum():,}",
        flush=True,
    )

    return torch.from_numpy(difficulty)


def evaluate(model, loader, device):
    y_true, y_prob = predict_kt(
        model,
        loader,
        device,
    )

    return y_true, y_prob


def train(
    model,
    train_loader,
    validation_loader,
    device,
    epochs,
    patience,
    learning_rate,
    weight_decay,
    seed,
    checkpoint_path,
    history_path,
):
    model = model.to(device)

    # Base paper specifies Adam, not AdamW.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # RTX 3080 memory-safe training:
    # physical batch 32 x accumulation 8 = effective batch 256.
    accumulation_steps = 8
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = -1.0
    best_epoch = 0
    best_state = cpu_state(model)
    stale = 0
    history = []
    start_epoch = 1

    if checkpoint_path.exists():

        ckpt = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            ckpt["model_state_dict"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )

        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

        best_auc = float(
            ckpt.get("best_auc", -1.0)
        )

        best_epoch = int(
            ckpt.get("best_epoch", 0)
        )

        best_state = ckpt.get(
            "best_state_dict",
            best_state,
        )

        stale = int(
            ckpt.get("stale", 0)
        )

        history = list(
            ckpt.get("history", [])
        )

        start_epoch = (
            int(ckpt.get("epoch", 0)) + 1
        )

        if (
            ckpt.get("status") == "complete"
            or stale >= patience
            or start_epoch > epochs
        ):
            model.load_state_dict(best_state)

            print(
                "HGAKT checkpoint already complete",
                flush=True,
            )

            return model, history, best_epoch

        print(
            f"resuming HGAKT from epoch {start_epoch}",
            flush=True,
        )

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        epoch_seed = seed + epoch

        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                epoch_seed
            )

        generator = getattr(
            train_loader,
            "generator",
            None,
        )

        if generator is not None:
            generator.manual_seed(epoch_seed)

        model.train()

        running = 0.0
        batches = 0

        progress = tqdm(
            train_loader,
            desc=f"hgakt_kt3 epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress, start=1):

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits, targets = _batch_predictions(
                    model,
                    batch,
                    device,
                )

                if targets.numel() == 0:
                    continue

                raw_loss = loss_fn(logits, targets)
                loss = raw_loss / accumulation_steps

            scaler.scale(loss).backward()

            do_step = (
                batch_idx % accumulation_steps == 0
                or batch_idx == len(train_loader)
            )

            if do_step:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            running += float(
                raw_loss.detach().cpu()
            )

            batches += 1

            progress.set_postfix(
                loss=f"{running / batches:.5f}"
            )

        train_loss = (
            running / max(batches, 1)
        )

        val_true, val_prob = evaluate(
            model,
            validation_loader,
            device,
        )

        val_metrics = binary_metrics(
            val_true,
            val_prob,
            threshold=0.5,
        )

        auc = float(
            val_metrics["auc"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }

        history.append(row)

        if auc > best_auc + 1e-6:
            best_auc = auc
            best_epoch = epoch
            best_state = cpu_state(model)
            stale = 0
        else:
            stale += 1

        print(
            f"[hgakt_kt3] epoch {epoch}/{epochs} "
            f"loss={train_loss:.6f} "
            f"val_auc={auc:.6f} "
            f"val_f1={val_metrics['f1']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"best_epoch={best_epoch} "
            f"patience={stale}/{patience}",
            flush=True,
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        _atomic_save(
            {
                "status": "in_progress",
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_state_dict":
                    best_state,
                "best_auc":
                    best_auc,
                "best_epoch":
                    best_epoch,
                "stale":
                    stale,
                "history":
                    history,
                "seed":
                    seed,
            },
            checkpoint_path,
        )

        if stale >= patience:

            print(
                f"[hgakt_kt3] early stopping "
                f"after epoch {epoch}; "
                f"best epoch={best_epoch}",
                flush=True,
            )

            break

    model.load_state_dict(
        best_state
    )

    _atomic_save(
        {
            "status": "complete",
            "epoch": history[-1]["epoch"],
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "best_state_dict":
                best_state,
            "best_auc":
                best_auc,
            "best_epoch":
                best_epoch,
            "stale":
                stale,
            "history":
                history,
            "seed":
                seed,
        },
        checkpoint_path,
    )

    return model, history, best_epoch


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/ednet_kt3.yaml",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_all(args.seed)

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    # Published HGAKT settings.
    config = deepcopy(config)

    config["sequence"]["epochs"] = 200
    config["sequence"]["learning_rate"] = 1e-3
    config["sequence"]["weight_decay"] = 1e-4

    # Paper does not state these clearly:
    # preserve current KT3 values.
    config["sequence"]["batch_size"] = 32
    config["sequence"]["max_len"] = 200
    config["sequence"]["embedding_dim"] = 128
    config["sequence"]["hidden_dim"] = 256
    config["sequence"]["dropout"] = 0.1
    config["sequence"]["patience"] = 5

    root = Path(
        config["dataset"]["processed_root"]
    )

    print(
        "Loading KT3 interactions...",
        flush=True,
    )

    all_interactions = read_collection(
        root / "interactions"
    )

    split_path = Path(
        config["output"]["root"]
    ) / "split_assignments.csv"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing existing split: {split_path}"
        )

    interactions = load_saved_split(
        all_interactions,
        split_path,
    )

    del all_interactions

    print(
        "Building the SAME KT loaders used by "
        "the current experiment...",
        flush=True,
    )

    skill_vocab, item_vocab, loaders = (
        _make_loaders(
            interactions,
            config,
            args.seed,
        )
    )

    metadata_path = Path(
        config["dataset"]["raw_root"]
    ) / "contents/contents/questions.csv"

    (
        item_to_part,
        item_to_tags,
        num_parts,
        num_tags,
    ) = build_hierarchy(
        item_vocab,
        metadata_path,
        max_tags=8,
    )

    item_difficulty = (
        build_training_item_difficulty(
            interactions["train"],
            item_vocab,
        )
    )

    model = HGAKTKT3(
        num_skills=skill_vocab.size,
        num_items=item_vocab.size,
        num_parts=num_parts,
        num_tags=num_tags,
        item_to_part=item_to_part,
        item_to_tags=item_to_tags,
        item_difficulty=item_difficulty,
        behavior_dim=BEHAVIOR_DIM,
        embedding_dim=128,
        hidden_dim=256,
        graph_heads=6,
        graph_layers=2,
        dropout=0.1,
    )

    device = torch_device()

    output = Path(
        "results/hgakt_kt3_trackB_preds"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = (
        output
        / f"hgakt_seed{args.seed}.resume.pt"
    )

    history_path = (
        output
        / f"hgakt_seed{args.seed}_history.csv"
    )

    model, history, best_epoch = train(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        device=device,
        epochs=200,
        patience=5,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=args.seed,
        checkpoint_path=checkpoint,
        history_path=history_path,
    )

    val_true, val_prob = evaluate(
        model,
        loaders["validation"],
        device,
    )

    threshold = select_f1_threshold(
        val_true,
        val_prob,
    )

    val_metrics = binary_metrics(
        val_true,
        val_prob,
        threshold=threshold,
    )

    test_true, test_prob = evaluate(
        model,
        loaders["test"],
        device,
    )

    test_metrics = binary_metrics(
        test_true,
        test_prob,
        threshold=threshold,
    )

    (output / "predictions").mkdir(parents=True, exist_ok=True)
    np.savez(output / "predictions" / f"hgakt_seed{args.seed}_validation.npz", y_true=np.asarray(val_true).astype(np.uint8), y_prob=np.asarray(val_prob).astype(np.float32))
    np.savez(output / "predictions" / f"hgakt_seed{args.seed}_test.npz", y_true=np.asarray(test_true).astype(np.uint8), y_prob=np.asarray(test_prob).astype(np.float32))
    result = {
        "model": "HGAKT_KT3_trackB_adaptation",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "decision_threshold":
            float(threshold),
        "validation": val_metrics,
        "test": test_metrics,
    }

    with open(
        output
        / f"hgakt_seed{args.seed}_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("\n===== FINAL HGAKT-KT3 TRACK-B =====")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
