from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .autograd import Tensor


class Parameter(Tensor):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data.astype(np.float32), requires_grad=True)


class Module:
    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def state_dict(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        # 修改点：He 初始化的方差缩放应除以输入维度 in_features
        limit = np.sqrt(2.0 / in_features)
        weight = np.random.randn(in_features, out_features).astype(np.float32) * limit
        bias = np.zeros((1, out_features), dtype=np.float32)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"weight": self.weight.data.copy(), "bias": self.bias.data.copy()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.weight.data = state["weight"].astype(np.float32)
        self.bias.data = state["bias"].astype(np.float32)


class MLPClassifier(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        activation: str = "relu",
        hidden_dim2: int | None = None,
    ) -> None:
        hidden_dim2 = hidden_dim if hidden_dim2 is None else hidden_dim2
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_dim2 = int(hidden_dim2)
        self.num_classes = int(num_classes)
        self.activation_name = activation.lower()

        if self.activation_name not in {"relu", "tanh", "sigmoid"}:
            raise ValueError("activation must be one of: relu, tanh, sigmoid")

        self.fc1 = Linear(self.input_dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.hidden_dim2)
        self.fc3 = Linear(self.hidden_dim2, self.num_classes)

    def _activate(self, x: Tensor) -> Tensor:
        if self.activation_name == "relu":
            return x.relu()
        if self.activation_name == "tanh":
            return x.tanh()
        if self.activation_name == "sigmoid":
            return x.sigmoid()
        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def __call__(self, x: Tensor) -> Tensor:
        h1 = self._activate(self.fc1(x))
        h2 = self._activate(self.fc2(h1))
        logits = self.fc3(h2)
        return logits

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "fc1_weight": self.fc1.weight.data.copy(),
            "fc1_bias": self.fc1.bias.data.copy(),
            "fc2_weight": self.fc2.weight.data.copy(),
            "fc2_bias": self.fc2.bias.data.copy(),
            "fc3_weight": self.fc3.weight.data.copy(),
            "fc3_bias": self.fc3.bias.data.copy(),
        }

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.fc1.weight.data = state["fc1_weight"].astype(np.float32)
        self.fc1.bias.data = state["fc1_bias"].astype(np.float32)
        self.fc2.weight.data = state["fc2_weight"].astype(np.float32)
        self.fc2.bias.data = state["fc2_bias"].astype(np.float32)
        self.fc3.weight.data = state["fc3_weight"].astype(np.float32)
        self.fc3.bias.data = state["fc3_bias"].astype(np.float32)

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.state_dict())

    @classmethod
    def load_from_checkpoint(cls, path: str, config: Dict[str, int | str]) -> "MLPClassifier":
        model = cls(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            hidden_dim2=(int(config["hidden_dim"]) if config.get("hidden_dim2") is None else int(config["hidden_dim2"])),
            num_classes=int(config["num_classes"]),
            activation=str(config["activation"]),
        )
        ckpt = np.load(path)
        state = {key: ckpt[key] for key in ckpt.files}
        model.load_state_dict(state)
        return model