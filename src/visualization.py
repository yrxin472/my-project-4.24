from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .data import Sample, load_image
from .metrics import confusion_matrix


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize_first_layer_weights(
    first_layer_weight: np.ndarray,
    image_size: int,
    out_path: str | Path,
    max_units: int = 64,
) -> None:
    if first_layer_weight.ndim != 2:
        raise ValueError("first_layer_weight must have shape [input_dim, hidden_dim]")

    input_dim, hidden_dim = first_layer_weight.shape
    expected_dim = image_size * image_size * 3
    if input_dim != expected_dim:
        raise ValueError(f"Input dim mismatch: expected {expected_dim}, got {input_dim}")

    num_show = min(max_units, hidden_dim)
    cols = int(math.ceil(math.sqrt(num_show)))
    rows = int(math.ceil(num_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx >= num_show:
            continue
        filt = first_layer_weight[:, idx].reshape(image_size, image_size, 3)
        filt_min, filt_max = filt.min(), filt.max()
        if filt_max - filt_min < 1e-8:
            vis = np.zeros_like(filt)
        else:
            vis = (filt - filt_min) / (filt_max - filt_min)
        ax.imshow(vis)
        ax.set_title(f"Unit {idx}", fontsize=8)

    fig.suptitle("First-layer Weight Visualization", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_samples(
    samples: Sequence[Sample],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    out_path: str | Path,
    image_size: int = 64,
    max_samples: int = 12,
) -> None:
    wrong_indices = np.where(y_true != y_pred)[0]
    if wrong_indices.size == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No misclassified samples in the evaluated split.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    wrong_indices = wrong_indices[:max_samples]
    cols = 4
    rows = int(math.ceil(len(wrong_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx >= len(wrong_indices):
            continue
        sample_idx = int(wrong_indices[idx])
        sample = samples[sample_idx]
        img = load_image(sample.path, image_size=image_size)
        ax.imshow(img)
        ax.set_title(f"T:{class_names[int(y_true[sample_idx])]}\\nP:{class_names[int(y_pred[sample_idx])]}", fontsize=9)

    fig.suptitle("Misclassified Test Samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
