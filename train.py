from __future__ import annotations

import argparse

from src.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a three-layer MLP from scratch on EuroSAT_RGB.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to EuroSAT_RGB directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save logs, weights, and figures.")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_dim2", type=int, default=None)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats_sample_size", type=int, default=None, help="Optional cap for mean/std estimation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = vars(args)
    summary = run_training(config)
    print("\nTraining finished.")
    print(f"Best validation accuracy: {summary['best_val_acc']:.4f}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best model saved to: {summary['best_model_path']}")


if __name__ == "__main__":
    main()
