from __future__ import annotations

import numpy as np

from .autograd import Tensor


# def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
#     if logits.ndim != 2:
#         raise ValueError(f"logits must be 2D, got shape {logits.shape}")
#     batch_size, num_classes = logits.shape
#     if targets.shape[0] != batch_size:
#         raise ValueError("targets length must equal batch size")
#
#     shift = logits.data.max(axis=1, keepdims=True)
#     shifted = logits - shift
#     exp_scores = shifted.exp()
#     logsumexp = exp_scores.sum(axis=1, keepdims=True).log()
#     log_probs = shifted - logsumexp
#
#     one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
#     one_hot[np.arange(batch_size), targets.astype(int)] = 1.0
#     target_tensor = Tensor(one_hot, requires_grad=False)
#
#     loss = -((log_probs * target_tensor).sum()) / float(batch_size)
#     return loss


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    batch_size, num_classes = logits.shape
    # 稳定版 logsumexp
    max_logits = logits.data.max(axis=1, keepdims=True)
    shifted = logits - max_logits
    exp_shifted = shifted.exp()
    logsumexp = (exp_shifted.sum(axis=1, keepdims=True)).log()
    log_probs = shifted - logsumexp
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), targets.astype(int)] = 1.0
    target_tensor = Tensor(one_hot, requires_grad=False)
    loss = -((log_probs * target_tensor).sum()) / float(batch_size)
    return loss

def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shift = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shift)
    return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)
