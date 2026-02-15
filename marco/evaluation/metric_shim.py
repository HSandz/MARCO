from __future__ import annotations
from typing import Any, Dict, List

class Metric:
    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}

    def add_state(self, name: str, default: Any = 0) -> None:
        self._state[name] = default

    def __getattr__(self, item: str) -> Any:
        if item in self._state:
            return self._state[item]
        raise AttributeError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in ("_state",):
            super().__setattr__(key, value)
        elif hasattr(self, "_state") and key in self._state:
            self._state[key] = value
        else:
            super().__setattr__(key, value)

    def update(self, output: dict) -> None:
        raise NotImplementedError()

    def compute(self) -> dict:
        raise NotImplementedError()

class MeanSquaredError(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.preds: List[float] = []
        self.targets: List[float] = []

    def update(self, preds: List[float] | float | None = None, target: List[float] | float | None = None, output: dict | None = None) -> None:
        if output is not None:
            preds = output.get('answer')
            target = output.get('label')

        if preds is None or target is None:
            return

        if isinstance(preds, (int, float)):
            self.preds.append(float(preds))
            self.targets.append(float(target))
        else:
            for p, t in zip(preds, target):
                self.preds.append(float(p))
                self.targets.append(float(t))

    def compute(self) -> dict:
        if not self.preds:
            return {'mse': 0.0}
        s = 0.0
        for p, t in zip(self.preds, self.targets):
            d = p - t
            s += d * d
        mse = s / len(self.preds)
        return {'mse': mse}

class MeanAbsoluteError(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.preds: List[float] = []
        self.targets: List[float] = []

    def update(self, preds: List[float] | float | None = None, target: List[float] | float | None = None, output: dict | None = None) -> None:
        if output is not None:
            preds = output.get('answer')
            target = output.get('label')

        if preds is None or target is None:
            return

        if isinstance(preds, (int, float)):
            self.preds.append(float(preds))
            self.targets.append(float(target))
        else:
            for p, t in zip(preds, target):
                self.preds.append(float(p))
                self.targets.append(float(t))

    def compute(self) -> dict:
        if not self.preds:
            return {'mae': 0.0}
        s = 0.0
        for p, t in zip(self.preds, self.targets):
            s += abs(p - t)
        mae = s / len(self.preds)
        return {'mae': mae}

class Accuracy(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.correct = 0
        self.total = 0

    def update(self, output: dict) -> None:
        answer = output.get('answer')
        label = output.get('label')
        if answer == label:
            self.correct += 1
        self.total += 1

    def compute(self) -> dict:
        if self.total == 0:
            return {'accuracy': 0.0}
        return {'accuracy': self.correct / self.total}
