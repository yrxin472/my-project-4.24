from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[float, int, np.ndarray, Sequence[float], Sequence[int]]


def _to_array(data: ArrayLike) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float32, copy=False)
    return np.array(data, dtype=np.float32)


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data = _to_array(data)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data) if requires_grad else None
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        return float(self.data.item())

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require gradients.")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensors.")
            grad = np.ones_like(self.data, dtype=np.float32)
        topo = []
        visited = set()

        def build(v: "Tensor") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        for node in topo:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)
        self.grad = _to_array(grad)

        for node in reversed(topo):
            node._backward()

    @staticmethod
    def ensure(other: Any) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward() -> None:
            if self.requires_grad and self.grad is not None and out.grad is not None:
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __add__(self, other: Any) -> "Tensor":
        other = Tensor.ensure(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, requires_grad=requires_grad, _children=(self, other), _op="add")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad and other.grad is not None:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-Tensor.ensure(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return Tensor.ensure(other) - self

    def __mul__(self, other: Any) -> "Tensor":
        other = Tensor.ensure(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, requires_grad=requires_grad, _children=(self, other), _op="mul")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad and other.grad is not None:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __truediv__(self, other: Any) -> "Tensor":
        other = Tensor.ensure(other)
        return self * other.pow(-1.0)

    def __rtruediv__(self, other: Any) -> "Tensor":
        return Tensor.ensure(other) / self

    def pow(self, power: float) -> "Tensor":
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, _children=(self,), _op="pow")

        def _backward() -> None:
            if self.requires_grad and self.grad is not None and out.grad is not None:
                self.grad += out.grad * (power * (self.data ** (power - 1)))

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Tensor":
        return self.pow(power)

    def __matmul__(self, other: Any) -> "Tensor":
        other = Tensor.ensure(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, requires_grad=requires_grad, _children=(self, other), _op="matmul")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad @ other.data.T
            if other.requires_grad and other.grad is not None:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward() -> None:
            if not self.requires_grad or self.grad is None or out.grad is None:
                return
            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = int(np.prod([self.data.shape[ax] for ax in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / float(denom)

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), _op="reshape")

        def _backward() -> None:
            if self.requires_grad and self.grad is not None and out.grad is not None:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad, _children=(self,), _op="transpose")

        def _backward() -> None:
            if self.requires_grad and self.grad is not None and out.grad is not None:
                inverse = np.argsort(axes)
                self.grad += out.grad.transpose(*inverse)

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        if self.ndim != 2:
            raise ValueError("T is only defined for 2D tensors in this implementation.")
        return self.transpose(1, 0)

    def exp(self) -> "Tensor":
        out_data = np.exp(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="exp")

        def _backward() -> None:
            if self.requires_grad and self.grad is not None and out.grad is not None:
                self.grad += out.grad * out_data

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op="log")

        def _backward() -> None:
            if self.requires_grad and self.grad is not None and out.grad is not None:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out_data = np.maximum(0.0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward() -> None:
            if self.requires_grad and self.grad is not None and out.grad is not None:
                self.grad += out.grad * (self.data > 0).astype(np.float32)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out_data = np.tanh(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward() -> None:
            if self.requires_grad and self.grad is not None and out.grad is not None:
                self.grad += out.grad * (1.0 - out_data ** 2)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        out_data = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if self.requires_grad and self.grad is not None and out.grad is not None:
                self.grad += out.grad * out_data * (1.0 - out_data)

        out._backward = _backward
        return out
