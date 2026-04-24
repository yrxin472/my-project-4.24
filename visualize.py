from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data import Sample
from src.model import MLPClassifier
from src.trainer import evaluate_split
from src.utils import load_json
from src.visualization import plot_misclassified_samples, visualize_first_layer_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualization outputs for a trained EuroSAT MLP.")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--max_units", type=int, default=64)
    parser.add_argument("--max_errors", type=int, default=12)
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

    visualize_first_layer_weights(
        first_layer_weight=model.fc1.weight.data,
        image_size=int(stats["image_size"]),
        out_path=experiment_dir / "first_layer_weights.png",
        max_units=args.max_units,
    )

    metrics = evaluate_split(
        model=model,
        samples=test_samples,
        batch_size=int(config["batch_size"]),
        image_size=int(stats["image_size"]),
        mean=np.array(stats["mean"], dtype=np.float32),
        std=np.array(stats["std"], dtype=np.float32),
        seed=int(config["seed"]),
    )

    plot_misclassified_samples(
        samples=test_samples,
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        class_names=class_names,
        out_path=experiment_dir / "misclassified_samples.png",
        image_size=int(stats["image_size"]),
        max_samples=args.max_errors,
    )
    print("Saved first-layer weight visualization and error analysis figures.")


if __name__ == "__main__":
    main()
