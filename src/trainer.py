from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .autograd import Tensor
from .data import BatchIterator, Sample, compute_channel_stats, discover_dataset, stratified_split
from .losses import cross_entropy_loss
from .metrics import accuracy_score
from .model import MLPClassifier
from .optim import SGD, StepLRScheduler
from .utils import ensure_dir, plot_training_curves, save_json, set_seed


@dataclass
class PreparedData:
    class_names: list[str]
    train_samples: list[Sample]
    val_samples: list[Sample]
    test_samples: list[Sample]
    mean: np.ndarray
    std: np.ndarray
    image_size: int


def prepare_data(
    data_root: str,
    image_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stats_sample_size: int | None = None,
) -> PreparedData:
    class_names, samples = discover_dataset(data_root)
    train_samples, val_samples, test_samples = stratified_split(
        samples=samples,
        num_classes=len(class_names),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    mean, std = compute_channel_stats(train_samples, image_size=image_size, max_samples=stats_sample_size)
    return PreparedData(
        class_names=class_names,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        mean=mean,
        std=std,
        image_size=image_size,
    )


def evaluate_split(
    model: MLPClassifier,
    samples: list[Sample],
    batch_size: int,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    losses = []
    all_preds = []
    all_targets = []
    iterator = BatchIterator(
        samples=samples,
        batch_size=batch_size,
        image_size=image_size,
        mean=mean,
        std=std,
        shuffle=False,
        seed=seed,
    )
    for batch_x, batch_y in iterator:
        x = Tensor(batch_x, requires_grad=False)
        logits = model(x)
        loss = cross_entropy_loss(logits, batch_y)
        preds = logits.data.argmax(axis=1)
        losses.append(loss.item())
        all_preds.append(preds)
        all_targets.append(batch_y)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_epoch(
    model: MLPClassifier,
    optimizer: SGD,
    samples: list[Sample],
    batch_size: int,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    epoch_seed: int,
) -> dict[str, float]:
    losses = []
    preds_all = []
    targets_all = []

    iterator = BatchIterator(
        samples=samples,
        batch_size=batch_size,
        image_size=image_size,
        mean=mean,
        std=std,
        shuffle=True,
        seed=epoch_seed,
    )

    for batch_x, batch_y in iterator:
        x = Tensor(batch_x, requires_grad=False)
        logits = model(x)
        loss = cross_entropy_loss(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.data.argmax(axis=1)
        losses.append(loss.item())
        preds_all.append(preds)
        targets_all.append(batch_y)

    y_true = np.concatenate(targets_all)
    y_pred = np.concatenate(preds_all)
    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(y_true, y_pred),
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(config["output_dir"])
    set_seed(int(config["seed"]))

    prepared = prepare_data(
        data_root=config["data_root"],
        image_size=int(config["image_size"]),
        train_ratio=float(config["train_ratio"]),
        val_ratio=float(config["val_ratio"]),
        test_ratio=float(config["test_ratio"]),
        seed=int(config["seed"]),
        stats_sample_size=config.get("stats_sample_size"),
    )

    input_dim = int(config["image_size"]) * int(config["image_size"]) * 3
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=int(config["hidden_dim"]),
        hidden_dim2=(int(config["hidden_dim"]) if config.get("hidden_dim2") is None else int(config["hidden_dim2"])),
        num_classes=len(prepared.class_names),
        activation=str(config["activation"]),
    )

    optimizer = SGD(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = StepLRScheduler(
        optimizer=optimizer,
        step_size=int(config["lr_step"]),
        gamma=float(config["lr_gamma"]),
    )

    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    best_model_path = output_dir / "best_model.npz"

    for epoch in range(1, int(config["epochs"]) + 1):
        optimizer.lr = scheduler.get_lr(epoch, float(config["lr"]))

        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            samples=prepared.train_samples,
            batch_size=int(config["batch_size"]),
            image_size=prepared.image_size,
            mean=prepared.mean,
            std=prepared.std,
            epoch_seed=int(config["seed"]) + epoch,
        )
        val_metrics = evaluate_split(
            model=model,
            samples=prepared.val_samples,
            batch_size=int(config["batch_size"]),
            image_size=prepared.image_size,
            mean=prepared.mean,
            std=prepared.std,
            seed=int(config["seed"]),
        )

        history["epoch"].append(epoch)
        history["lr"].append(float(optimizer.lr))
        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_acc"].append(float(train_metrics["acc"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["acc"]))

        print(
            f"[Epoch {epoch:03d}] "
            f"lr={optimizer.lr:.6f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = float(val_metrics["acc"])
            best_epoch = epoch
            model.save(str(best_model_path))

    config_to_save = dict(config)
    config_to_save["input_dim"] = input_dim
    config_to_save["num_classes"] = len(prepared.class_names)

    save_json(output_dir / "config.json", config_to_save)
    save_json(
        output_dir / "stats.json",
        {
            "mean": prepared.mean.tolist(),
            "std": prepared.std.tolist(),
            "class_names": prepared.class_names,
            "image_size": int(prepared.image_size),
        },
    )
    save_json(
        output_dir / "splits.json",
        {
            "train": [{"path": s.path, "label": int(s.label)} for s in prepared.train_samples],
            "val": [{"path": s.path, "label": int(s.label)} for s in prepared.val_samples],
            "test": [{"path": s.path, "label": int(s.label)} for s in prepared.test_samples],
        },
    )
    save_json(output_dir / "history.json", history)
    plot_training_curves(history, output_dir / "training_curves.png")

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_model_path": str(best_model_path),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "train_summary.json", summary)
    return summary
