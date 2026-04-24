from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

from src.trainer import run_training
from src.utils import ensure_dir, save_csv, save_json


def parse_csv_str(value: str, cast):
    return [cast(v) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyper-parameter search for EuroSAT MLP.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--search_type", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--hidden_dims", type=str, default="128,256,512")
    parser.add_argument("--activations", type=str, default="relu,tanh")
    parser.add_argument("--lrs", type=str, default="0.1,0.05,0.01")
    parser.add_argument("--weight_decays", type=str, default="0.0,1e-4,5e-4")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats_sample_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    hidden_dims = parse_csv_str(args.hidden_dims, int)
    activations = parse_csv_str(args.activations, str)
    lrs = parse_csv_str(args.lrs, float)
    weight_decays = parse_csv_str(args.weight_decays, float)

    all_trials = [
        {"hidden_dim": h, "activation": a, "lr": lr, "weight_decay": wd}
        for h, a, lr, wd in itertools.product(hidden_dims, activations, lrs, weight_decays)
    ]

    if args.search_type == "random":
        rng = random.Random(args.seed)
        if len(all_trials) > args.num_trials:
            all_trials = rng.sample(all_trials, args.num_trials)
    else:
        all_trials = all_trials[:]

    results = []
    best_result = None

    for trial_idx, trial in enumerate(all_trials, start=1):
        trial_name = f"trial_{trial_idx:03d}_h{trial['hidden_dim']}_{trial['activation']}_lr{trial['lr']}_wd{trial['weight_decay']}"
        trial_dir = output_dir / trial_name

        config = {
            "data_root": args.data_root,
            "output_dir": str(trial_dir),
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": trial["hidden_dim"],
            "hidden_dim2": None,
            "activation": trial["activation"],
            "lr": trial["lr"],
            "weight_decay": trial["weight_decay"],
            "lr_step": args.lr_step,
            "lr_gamma": args.lr_gamma,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "stats_sample_size": args.stats_sample_size,
        }

        print(f"\n=== Trial {trial_idx}/{len(all_trials)}: {trial_name} ===")
        summary = run_training(config)
        row = {
            "trial": trial_name,
            "hidden_dim": trial["hidden_dim"],
            "activation": trial["activation"],
            "lr": trial["lr"],
            "weight_decay": trial["weight_decay"],
            "best_val_acc": summary["best_val_acc"],
            "best_epoch": summary["best_epoch"],
            "model_path": summary["best_model_path"],
        }
        results.append(row)

        if best_result is None or row["best_val_acc"] > best_result["best_val_acc"]:
            best_result = row

    save_csv(output_dir / "search_results.csv", results)
    save_json(output_dir / "search_results.json", {"results": results, "best_result": best_result})
    if best_result is not None:
        save_json(output_dir / "best_config.json", best_result)
        print("\nBest hyper-parameter setting:")
        for k, v in best_result.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
