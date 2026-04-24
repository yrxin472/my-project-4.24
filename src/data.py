from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class Sample:
    path: str
    label: int


def discover_dataset(root: str | Path) -> tuple[list[str], list[Sample]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())
    if not class_dirs:
        raise ValueError(f"No class folders found under: {root}")

    class_names = [p.name for p in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    valid_suffix = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    samples: list[Sample] = []
    for class_dir in class_dirs:
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in valid_suffix:
                samples.append(Sample(str(file_path), class_to_idx[class_dir.name]))

    if not samples:
        raise ValueError(f"No image files found under: {root}")
    return class_names, samples


def stratified_split(
    samples: Sequence[Sample],
    num_classes: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    grouped: dict[int, list[Sample]] = {i: [] for i in range(num_classes)}
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    test_samples: list[Sample] = []

    for label in range(num_classes):
        group = grouped[label][:]
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)

        train_samples.extend(group[:n_train])
        val_samples.extend(group[n_train : n_train + n_val])
        test_samples.extend(group[n_train + n_val :])

    return train_samples, val_samples, test_samples


def load_image(path: str | Path, image_size: int = 64) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5   # <--- 缩放到 [-1,1]
    return arr


def compute_channel_stats(samples: Sequence[Sample], image_size: int = 64, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is not None and max_samples > 0 and len(samples) > max_samples:
        rng = random.Random(1234)
        chosen = rng.sample(list(samples), max_samples)
    else:
        chosen = list(samples)

    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for sample in chosen:
        arr = load_image(sample.path, image_size=image_size)
        pixels = arr.reshape(-1, 3)
        ch_sum += pixels.sum(axis=0)
        ch_sq_sum += (pixels ** 2).sum(axis=0)
        pixel_count += pixels.shape[0]

    mean = ch_sum / pixel_count
    std = np.sqrt(np.clip(ch_sq_sum / pixel_count - mean ** 2, 1e-12, None))
    std = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


def preprocess_array(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, flatten: bool = True) -> np.ndarray:
    arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    if flatten:
        return arr.reshape(-1).astype(np.float32)
    return arr.astype(np.float32)


def sample_to_model_input(path: str, image_size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = load_image(path, image_size=image_size)
    return preprocess_array(arr, mean=mean, std=std, flatten=True)


class BatchIterator:
    def __init__(
        self,
        samples: Sequence[Sample],
        batch_size: int,
        image_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        shuffle: bool = True,
        seed: int = 42,
        return_paths: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.return_paths = return_paths

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]]:
        indices = list(range(len(self.samples)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_x = []
            batch_y = []
            batch_paths = []
            for idx in batch_idx:
                sample = self.samples[idx]
                arr = sample_to_model_input(sample.path, self.image_size, self.mean, self.std)
                batch_x.append(arr)
                batch_y.append(sample.label)
                batch_paths.append(sample.path)
            x = np.stack(batch_x).astype(np.float32)
            y = np.array(batch_y, dtype=np.int64)
            if self.return_paths:
                yield x, y, batch_paths
            else:
                yield x, y
