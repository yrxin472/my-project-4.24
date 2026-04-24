from __future__ import annotations

from typing import Dict

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    denom = np.clip(cm.sum(axis=1), 1, None)
    return np.diag(cm) / denom


def format_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> str:
    width = max(8, max(len(name) for name in class_names) + 2)
    header = " " * width + "".join(name[:width-1].ljust(width) for name in class_names)
    rows = [header]
    for i, name in enumerate(class_names):
        row = name[:width-1].ljust(width)
        row += "".join(str(int(v)).ljust(width) for v in cm[i])
        rows.append(row)
    return "\n".join(rows)
