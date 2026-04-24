from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data import Sample
from src.metrics import confusion_matrix, format_confusion_matrix, per_class_accuracy
from src.model import MLPClassifier
from src.trainer import evaluate_split
from src.utils import load_json, save_json
from src.visualization import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained EuroSAT MLP model on the test split.")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)

    config = load_json(experiment_dir / "config.json")
    stats = load_json(experiment_dir / "stats.json")
    splits = load_json(experiment_dir / "splits.json")
    weights_path = Path(args.weights_path) if args.weights_path else experiment_dir / "best_model.npz"

    class_names = list(stats["class_names"])
    test_samples = [Sample(path=item["path"], label=int(item["label"])) for item in splits["test"]]

    model = MLPClassifier.load_from_checkpoint(str(weights_path), config)
    metrics = evaluate_split(
        model=model,
        samples=test_samples,
        batch_size=int(config["batch_size"]),
        image_size=int(stats["image_size"]),
        mean=np.array(stats["mean"], dtype=np.float32),
        std=np.array(stats["std"], dtype=np.float32),
        seed=int(config["seed"]),
    )

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    cm = confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    cls_acc = per_class_accuracy(cm)

    print(f"Test Accuracy: {metrics['acc']:.4f}")
    print("\nConfusion Matrix:")
    cm_text = format_confusion_matrix(cm, class_names)
    print(cm_text)
    print("\nPer-class Accuracy:")
    for name, acc in zip(class_names, cls_acc):
        print(f"{name}: {acc:.4f}")

    (experiment_dir / "confusion_matrix.txt").write_text(cm_text, encoding="utf-8")
    plot_confusion_matrix(cm, class_names, experiment_dir / "confusion_matrix.png")

    save_json(
        experiment_dir / "test_metrics.json",
        {
            "test_accuracy": float(metrics["acc"]),
            "class_names": class_names,
            "per_class_accuracy": {name: float(acc) for name, acc in zip(class_names, cls_acc)},
            "confusion_matrix": cm.tolist(),
        },
    )


if __name__ == "__main__":
    main()
