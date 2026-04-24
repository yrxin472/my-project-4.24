from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .model import Parameter


class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float = 0.01, weight_decay: float = 0.0) -> None:
        self.params: List[Parameter] = list(params)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0.0 and p.data.ndim >= 2:
                grad = grad + self.weight_decay * p.data
            p.data = p.data - self.lr * grad


class StepLRScheduler:
    def __init__(self, optimizer: SGD, step_size: int = 10, gamma: float = 0.5) -> None:
        self.optimizer = optimizer
        self.step_size = max(1, int(step_size))
        self.gamma = float(gamma)

    def step(self, epoch: int) -> None:
        factor = self.gamma ** (epoch // self.step_size)
        return factor

    def get_lr(self, epoch: int, base_lr: float) -> float:
        return float(base_lr) * (self.gamma ** ((epoch - 1) // self.step_size))
